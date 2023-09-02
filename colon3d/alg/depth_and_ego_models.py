from pathlib import Path

import numpy as np
import torch
import yaml

import monodepth2.layers as monodepth2_layers
import monodepth2.networks as monodepth2_networks
import monodepth2.networks.depth_decoder as monodepth2_depth_decoder
import monodepth2.networks.pose_decoder as monodepth2_pose_decoder
import monodepth2.networks.resnet_encoder as monodepth2_resnet_encoder
import monodepth2.utils as monodepth2_utils
from colon3d.net_train.common_transforms import img_to_net_in_format
from colon3d.util.rotations_util import axis_angle_to_quaternion
from colon3d.util.torch_util import get_device
from endo_sfm.models_def.DispResNet import DispResNet as endo_sfm_DispResNet
from endo_sfm.models_def.PoseResNet import PoseResNet as endo_sfm_PoseResNet

# --------------------------------------------------------------------------------------------------------------------


class DepthModel:
    """
    The depth estimation network.
    Note that we use a network that estimates the disparity and then convert it to depth by taking 1/disparity.
    """

    def __init__(self, depth_lower_bound: float, depth_upper_bound: float, method: str, model_path: Path) -> None:
        self.method = method
        assert model_path is not None, "model_path is None"
        print(f"Loading depth model from {model_path}")
        self.depth_lower_bound = 0 if depth_lower_bound is None else depth_lower_bound
        self.depth_upper_bound = depth_upper_bound

        self.model_info = get_model_info(model_path)

        # the dimensions of the output depth maps
        self.depth_map_width = self.model_info["frame_width"]
        self.depth_map_height = self.model_info["frame_height"]

        # the output of the network (translation part) needs to be multiplied by this number to get the depth\ego-translations in mm (based on the analysis of sample data in examine_depths.py):
        self.depth_calib_a = self.model_info["depth_calib_a"]
        self.depth_calib_b = self.model_info["depth_calib_b"]
        print(f"depth_calibrations: a={self.depth_calib_a}, b={self.depth_calib_b}")

        # load the camera matrix corresponding to the depth maps:
        self.depth_map_K = get_camera_matrix(self.model_info)

        self.device = get_device()

        if method in {"EndoSFM", "EndoSFM_GTD"}:
            # create the disparity estimation network
            self.disp_net = endo_sfm_DispResNet(num_layers=self.model_info["ResNet_layers"], pretrained=True)
            # load the Disparity network
            self.disp_net_path = model_path / "DispNet_best.pt"
            weights = torch.load(self.disp_net_path)
            self.disp_net.load_state_dict(weights["state_dict"], strict=False)
            self.disp_net.to(self.device)
            self.disp_net.eval()  # switch to evaluate mode
            self.dtype = self.disp_net.get_weight_dtype()

        elif method == "MonoDepth2":  # source: https://github.com/nianticlabs/monodepth2
            monodepth2_utils.download_model_if_doesnt_exist(model_path.name, models_dir=model_path.parent)
            encoder_path = model_path / "encoder.pth"
            depth_decoder_path = model_path / "depth.pth"
            self.encoder = monodepth2_networks.resnet_encoder.ResnetEncoder(18, False)
            self.depth_decoder = monodepth2_depth_decoder.DepthDecoder(
                num_ch_enc=self.encoder.num_ch_enc,
                scales=range(4),
            )
            loaded_dict_enc = torch.load(encoder_path)
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
            self.encoder.load_state_dict(filtered_dict_enc)

            loaded_dict = torch.load(depth_decoder_path)
            self.depth_decoder.load_state_dict(loaded_dict)
            self.encoder.to(self.device)
            self.depth_decoder.to(self.device)
            self.encoder.eval()
            self.depth_decoder.eval()
            self.dtype = self.encoder.get_weight_dtype()

        else:
            raise ValueError(f"Unknown depth estimation method: {method}")

    # --------------------------------------------------------------------------------------------------------------------

    def get_depth_info(self) -> dict:
        #  metadata for the depth maps
        depth_info = {
            "K_of_depth_map": self.depth_map_K,
            "depth_map_size": {"width": self.depth_map_width, "height": self.depth_map_height},
        }
        return depth_info

    # --------------------------------------------------------------------------------------------------------------------

    def estimate_depth_map(self, img: np.ndarray) -> torch.Tensor:
        """Estimate the depth map from the image.
        Args:
            imgs: the input images [H x W x 3]
        Returns:
            depth_map (torch.Tensor): the estimated depth maps [H x W] (units: mm)

        Note:
            * The estimated depth map might not be the same size as the input image.
            This is accounted for when estimating the depth at a certain pixel of the original image since we use the camera matrix associated with  depth map.
        """
        assert img.ndim == 3  # [H x W x 3]
        assert img.shape[2] == 3  # RGB image

        # resize and change dimension order of the images to fit the network input format  # [3 x H x W]
        img = img_to_net_in_format(
            img=img,
            device=self.device,
            dtype=self.dtype,
            add_batch_dim=True,
        )

        if self.method in {"EndoSFM", "EndoSFM_GTD"}:
            with torch.no_grad():
                disparity_map = self.disp_net(img)  # [N x 1 x H x W]
                # remove the n_sample and n_channels dimension
                disparity_map = disparity_map.squeeze(0).squeeze(0)  # [H x W]
                # convert the disparity to depth
                depth_map = 1 / disparity_map

        elif self.method == "MonoDepth2":
            # based on monodepth2/evaluate_depth.py
            with torch.no_grad():
                encoded = self.encoder(img)
                output = self.depth_decoder(encoded)
                disparity_map = output[("disp", 0)]  # [N x 1 x H x W]
                # remove the n_sample and n_channels dimension
                disparity_map = disparity_map.squeeze(0).squeeze(0)  # [H x W]
                # convert the disparity to depth
                _, depth_map = monodepth2_layers.disp_to_depth(
                    disparity_map,
                    min_depth=self.depth_lower_bound,
                    max_depth=self.depth_upper_bound,
                )
        else:
            raise ValueError(f"Unknown depth estimation method: {self.method}")
        # use the depth calibration to get the depth in mm
        depth_map = self.depth_calib_a * depth_map + self.depth_calib_b
            
        # clip the depth if needed
        if self.depth_lower_bound is not None or self.depth_upper_bound is not None:
            depth_map = torch.clamp(depth_map, self.depth_lower_bound, self.depth_upper_bound)

        return depth_map


# --------------------------------------------------------------------------------------------------------------------


class EgomotionModel:
    def __init__(self, method: str, model_path: Path) -> None:
        print(f"Loading egomotion model from {model_path}")
        assert model_path is not None, "model_path is None"
        self.method = method
        self.model_info = get_model_info(model_path)
        self.device = get_device()
        # the output of the network (translation part) needs to be multiplied by this number to get the depth\ego-translations in mm (based on the analysis of sample data in examine_depths.py):
        self.depth_calib_a = self.model_info["depth_calib_a"]
        # the camera matrix corresponding to the depth maps.
        self.depth_map_K = get_camera_matrix(self.model_info)
        # create the egomotion estimation network

        if method in {"EndoSFM", "EndoSFM_GTD"}:
            pose_net_path = model_path / "PoseNet_best.pt"
            self.pose_net = endo_sfm_PoseResNet(num_layers=self.model_info["ResNet_layers"], pretrained=True)
            weights = torch.load(pose_net_path)
            self.pose_net.load_state_dict(weights["state_dict"], strict=False)
            self.pose_net.to(self.device)
            self.pose_net.eval()  # switch to evaluate mode
            self.dtype = self.pose_net.get_weight_dtype()

        elif method == "MonoDepth2":
            # based on monodepth2/evaluate_pose.py
            pose_encoder_path = model_path / "pose_encoder.pth"
            pose_decoder_path = model_path / "pose.pth"
            self.pose_encoder = monodepth2_resnet_encoder.ResnetEncoder(18, False, 2)
            self.pose_encoder.load_state_dict(torch.load(pose_encoder_path))
            self.pose_decoder = monodepth2_pose_decoder.PoseDecoder(self.pose_encoder.num_ch_enc, 1, 2)
            self.pose_decoder.load_state_dict(torch.load(pose_decoder_path))
            self.pose_encoder.to(self.device)
            self.pose_decoder.to(self.device)
            self.pose_encoder.eval()
            self.pose_decoder.eval()
            self.dtype = self.pose_encoder.get_weight_dtype()

        else:
            raise ValueError(f"Unknown egomotion estimation method: {method}")

    # --------------------------------------------------------------------------------------------------------------------

    def estimate_egomotion(self, from_img: np.ndarray, to_img: np.ndarray) -> torch.Tensor:
        """Estimate the 6DOF egomotion from the from image to to image.
            The egomotion is the pose change from the previous frame to the current frame, defined in the previous frame system.
        Args:
            from_imgs: the 'from' images (target) [N x 3 x H x W] where N is the number of image pairs
            to_imgs: the corresponding 'to' images  (reference) [N X 3 x H x W]
        Returns:
            egomotions: the estimated egomotions [N x 7] 6DoF pose parameters from from_imgs to to_imgs, in the format:
                (x,y,z,qw,qx,qy,qz) where (x, y, z) is the translation [mm] and (qw, qx, qy , qz) is the unit-quaternion of the rotation.
        """
        assert from_img.shape == to_img.shape  # same shape
        assert from_img.ndim == 3  # [3 x H x W]

        from_img = img_to_net_in_format(
            img=from_img,
            device=self.device,
            dtype=self.dtype,
            add_batch_dim=True,
        )
        to_img = img_to_net_in_format(
            img=to_img,
            device=self.device,
            dtype=self.dtype,
            add_batch_dim=True,
        )

        if self.method in {"EndoSFM", "EndoSFM_GTD"}:
            # note: if you want to use the ground truth depth maps, you need to change depth_maps_source to "ground_trutg"
            # "EndoSFM_GTD" is still and estimate, but it was trained with GT depth maps
            with torch.no_grad():
                pose_out = self.pose_net(from_img, to_img)
            # this returns the estimated egomotion [N x 6] 6DoF pose parameters from target to reference  in the order of tx, ty, tz, rx, ry, rz
            pose_out = pose_out.squeeze(0)  # remove the n_sample dimension
            translation = pose_out[:3]
            rotation_axis_angle = pose_out[3:]

        elif self.method == "MonoDepth2":
            # based on monodepth2/evaluate_pose.py
            with torch.no_grad():
                # concat the input images in axis 1 (channel dimension)
                net_input = torch.cat((from_img, to_img), dim=1)
                features = [self.pose_encoder(net_input)]
                rotation_axis_angle, translation = self.pose_decoder(features)
            # # remove the n_sample dimension and take the first item - motion from the first image to the second image
            rotation_axis_angle = rotation_axis_angle.squeeze(0)[0].squeeze(0)  # [3]
            translation = translation.squeeze(0)[0].squeeze(0)  # [3]

        else:
            raise ValueError(f"Unknown egomotion estimation method: {self.method}")
        # convert the axis-angle to quaternion
        rotation_quaternion = axis_angle_to_quaternion(rotation_axis_angle)

        # multiply the translation by the conversion factor to get mm units
        translation = self.depth_calib_a * translation
        egomotion = torch.cat((translation, rotation_quaternion), dim=0)
        assert egomotion.ndim == 1
        return egomotion

    # --------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------


def get_model_info(model_dir_path: Path):
    model_info_path = model_dir_path / "model_info.yaml"
    assert model_info_path.is_file(), f"Model info file not found at {model_info_path}"
    with model_info_path.open("r") as f:
        model_info = yaml.load(f, Loader=yaml.FullLoader)
    if "depth_calib_type" not in model_info:
        print("depth_calib not found in model info, using default 'none'")
        model_info["depth_calib_type"] = "none"
        model_info["depth_calib_a"] = 1
        model_info["depth_calib_b"] = 0

    return model_info


# --------------------------------------------------------------------------------------------------------------------


def get_camera_matrix(model_info: dict) -> np.ndarray:
    fx = model_info["fx"]
    fy = model_info["fy"]
    cx = model_info["cx"]
    cy = model_info["cy"]
    distort_pram = model_info["distort_pram"]
    assert distort_pram is None, "we assume no distortion"
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


# --------------------------------------------------------------------------------------------------------------------

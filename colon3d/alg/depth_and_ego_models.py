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
from colon3d.util.general_util import resize_color_images
from colon3d.util.rotations_util import axis_angle_to_quaternion
from colon3d.util.torch_util import get_device, to_torch
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
        self.net_out_to_mm = self.model_info["net_out_to_mm"]
        print(f"net_out_to_mm: {self.net_out_to_mm}")

        # the camera matrix corresponding to the depth maps:
        self.depth_map_K = get_camera_matrix(self.model_info)
        self.device = get_device()

        if method == "EndoSFM":
            # create the disparity estimation network
            self.disp_net = endo_sfm_DispResNet(num_layers=self.model_info["DispResNet_layers"], pretrained=True)
            # load the Disparity network
            self.disp_net_path = model_path / "DispNet_best.pt"
            weights = torch.load(self.disp_net_path)
            self.disp_net.load_state_dict(weights["state_dict"], strict=False)
            self.disp_net.to(self.device)
            self.disp_net.eval()  # switch to evaluate mode
            self.dtype = torch.float32  # the network weights type

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
            self.feed_height = loaded_dict_enc["height"]
            self.feed_width = loaded_dict_enc["width"]
            loaded_dict = torch.load(depth_decoder_path)
            self.depth_decoder.load_state_dict(loaded_dict)
            self.encoder.to(self.device)
            self.depth_decoder.to(self.device)
            self.encoder.eval()
            self.depth_decoder.eval()
            self.dtype = torch.float32  # the network weights type

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

    def estimate_depth_maps(self, imgs: torch.Tensor, is_singleton) -> torch.Tensor:
        """Estimate the depth map from the image.

        Args:
            img (torch.Tensor): the input images [N x H x W x 3]
        Returns:
            depth_map (torch.Tensor): the estimated depth maps [N X H x W] (units: mm)
        """
        if is_singleton:
            # add a batch dimension
            imgs = torch.unsqueeze(imgs, dim=0)

        # resize and change dimension order of the images to fit the network input format  # [N x 3 x H x W]
        imgs = imgs_to_net_in(imgs, self.device, self.dtype, self.depth_map_height, self.depth_map_width)

        if self.method == "EndoSFM":
            with torch.no_grad():
                disparity_maps = self.disp_net(imgs)
                # remove the n_channels dimension
                disparity_maps.squeeze_(dim=1)  # [N x H x W]
                # convert the disparity to depth
                depth_maps = 1 / disparity_maps

        elif self.method == "MonoDepth2":
            # based on monodepth2/evaluate_depth.py
            with torch.no_grad():
                encoded = self.encoder(imgs)
                output = self.depth_decoder(encoded)
                disparity_maps = output[("disp", 0)]  # [N x C X H x W]
                disparity_maps = disparity_maps.squeeze(1)  # [N x H x W]
                depth_maps = monodepth2_layers.disp_to_depth(
                    disparity_maps,
                    min_depth=self.depth_lower_bound,
                    max_depth=self.depth_upper_bound,
                )

        else:
            raise ValueError(f"Unknown depth estimation method: {self.method}")
        # multiply by the scale factor to get the depth in mm
        depth_maps *= self.net_out_to_mm
        # clip the depth if needed
        if self.depth_lower_bound is not None or self.depth_upper_bound is not None:
            depth_maps = torch.clamp(depth_maps, self.depth_lower_bound, self.depth_upper_bound)
            
        if is_singleton:
            # remove the batch dimension
            depth_maps = depth_maps[0]
        return depth_maps
    
# --------------------------------------------------------------------------------------------------------------------


class EgomotionModel:
    def __init__(self, method: str, model_path: Path) -> None:
        print(f"Loading egomotion model from {model_path}")
        assert model_path is not None, "model_path is None"
        self.method = method
        self.model_info = get_model_info(model_path)
        self.device = get_device()
        self.model_im_height = self.model_info["frame_height"]
        self.model_im_width = self.model_info["frame_width"]
        # the output of the network (translation part) needs to be multiplied by this number to get the depth\ego-translations in mm (based on the analysis of sample data in examine_depths.py):
        self.net_out_to_mm = self.model_info["net_out_to_mm"]
        # the camera matrix corresponding to the depth maps.
        self.depth_map_K = get_camera_matrix(self.model_info)
        # create the egomotion estimation network

        if method == "EndoSFM":
            pose_net_path = model_path / "PoseNet_best.pt"
            self.pose_net = endo_sfm_PoseResNet(num_layers=self.model_info["PoseResNet_layers"], pretrained=True)
            weights = torch.load(pose_net_path)
            self.pose_net.load_state_dict(weights["state_dict"], strict=False)
            self.pose_net.to(self.device)
            self.pose_net.eval()  # switch to evaluate mode
            self.dtype = torch.float32  # the network weights type

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
            self.dtype = torch.float32  # the network weights type

        else:
            raise ValueError(f"Unknown egomotion estimation method: {method}")

    # --------------------------------------------------------------------------------------------------------------------

    def estimate_egomotions(self, from_imgs: np.ndarray, to_imgs: np.ndarray, is_singleton=False) -> torch.Tensor:
        """Estimate the 6DOF egomotion from the from image to to image.
            The egomotion is the pose change from the previous frame to the current frame, defined in the previous frame system.
        Args:
            from_imgs: the 'from' images (target) [N x 3 x H x W] where N is the number of image pairs
            to_imgs: the corresponding 'to' images  (reference) [N X 3 x H x W]
        Returns:
            egomotions: the estimated egomotions [N x 7] 6DoF pose parameters from from_imgs to to_imgs, in the format:
                (x,y,z,qw,qx,qy,qz) where (x, y, z) is the translation [mm] and (qw, qx, qy , qz) is the unit-quaternion of the rotation.
        """
        if is_singleton:
            # add a batch dimension
            from_imgs = np.expand_dims(from_imgs, axis=0)
            to_imgs = np.expand_dims(to_imgs, axis=0)
        n_imgs = len(from_imgs)
        assert len(to_imgs) == n_imgs
        assert from_imgs.shape[1:] == to_imgs.shape[1:]  # same shape
        assert from_imgs.ndim == 4  # [N x 3 x H x W]
        from_imgs = imgs_to_net_in(from_imgs, self.device, self.dtype, self.model_im_height, self.model_im_width)
        to_imgs = imgs_to_net_in(to_imgs, self.device, self.dtype, self.model_im_height, self.model_im_width)

        if self.method == "EndoSFM":
            with torch.no_grad():
                pose_out = self.pose_net(from_imgs, to_imgs)
            # this returns the estimated egomotion [N x 6] 6DoF pose parameters from target to reference  in the order of tx, ty, tz, rx, ry, rz
            translation = pose_out[:, :3]
            rotation_axis_angle = pose_out[:, 3:]

        elif self.method == "MonoDepth2":
            # based on monodepth2/evaluate_pose.py
            # concat the input images in axis 1 (channel dimension)
            with torch.no_grad():
                all_color_aug = torch.cat((from_imgs, to_imgs), dim=1)
                features = [self.pose_encoder(all_color_aug)]
                rotation_axis_angle, translation = self.pose_decoder(features)
            # take the first item - motion from the first image to the second image
            rotation_axis_angle = rotation_axis_angle.squeeze(2)[:, 0, :]  # [N, 3]
            translation = translation.squeeze(2)[:, 0, :]  # [N, 3]

        else:
            raise ValueError(f"Unknown egomotion estimation method: {self.method}")
        # convert the axis-angle to quaternion
        rotation_quaternion = axis_angle_to_quaternion(rotation_axis_angle)

        # multiply the translation by the conversion factor to get mm units
        translation *= self.net_out_to_mm
        egomotions = torch.cat((translation, rotation_quaternion), dim=1)
        if is_singleton:
            # remove the batch dimension
            egomotions = egomotions[0]
        return egomotions

    # --------------------------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------------------------------------------


def imgs_to_net_in(
    imgs: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    depth_map_height: int,
    depth_map_width: int,
) -> torch.Tensor:
    """Transform the input images to the network input format.
    Args:
        imgs: the input images [n_imgs x height x width x n_channels]
    Returns:
        imgs: the input images in the network format [n_imgs x n_channels x depth_map_height x depth_map_width]
    """
    n_imgs, height, width, n_channels = imgs.shape
    assert n_channels in [1, 3]
    if (height, width) != (depth_map_height, depth_map_width):
        # resize the images
        imgs = resize_color_images(imgs, new_height=depth_map_height, new_width=depth_map_width)

    # transform to channels first
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    imgs = to_torch(imgs, device=device).to(dtype)
    # normalize the images to fit the pre-trained weights (based on https://github.com/CapsuleEndoscope/EndoSLAM/blob/master/EndoSfMLearner/run_inference.py)
    imgs = (imgs / 255 - 0.45) / 0.225
    return imgs


# --------------------------------------------------------------------------------------------------------------------


def get_model_info(model_dir_path: Path):
    model_info_path = model_dir_path / "model_info.yaml"
    assert model_info_path.is_file(), f"Model info file not found at {model_info_path}"
    with model_info_path.open("r") as f:
        model_info = yaml.load(f, Loader=yaml.FullLoader)
    if "net_out_to_mm" not in model_info:
        print("net_out_to_mm not found in model info, using default value of 1.0")
        model_info["net_out_to_mm"] = 1.0

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

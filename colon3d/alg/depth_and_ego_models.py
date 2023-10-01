from pathlib import Path

import numpy as np
import torch

import monodepth2.layers as monodepth2_layers
import monodepth2.networks as monodepth2_networks
import monodepth2.networks.depth_decoder as monodepth2_depth_decoder
import monodepth2.networks.pose_decoder as monodepth2_pose_decoder
import monodepth2.networks.resnet_encoder as monodepth2_resnet_encoder
import monodepth2.utils as monodepth2_utils
from colon3d.net_train.train_utils import load_model_model_info
from colon3d.util.rotations_util import axis_angle_to_quaternion
from colon3d.util.torch_util import get_device, resize_images_batch, resize_single_image, to_torch
from endo_sfm.models_def.DispResNet import DispResNet as endo_sfm_DispResNet
from endo_sfm.models_def.PoseResNet import PoseResNet as endo_sfm_PoseResNet

# --------------------------------------------------------------------------------------------------------------------


def normalize_image_channels(img: torch.Tensor, img_normalize_mean: float = 0.45, img_normalize_std: float = 0.225):
    # normalize the values from [0,255] to [0, 1]
    img = img / 255
    # normalize the values to the mean and std of the ImageNet dataset
    img = (img - img_normalize_mean) / img_normalize_std
    return img


# --------------------------------------------------------------------------------------------------------------------


def img_to_net_in_format(
    img: np.ndarray | torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    normalize_values: bool = True,
    img_normalize_mean: float = 0.45,
    img_normalize_std: float = 0.225,
    add_batch_dim: bool = False,
    feed_height: int | None = None,
    feed_width: int | None = None,
) -> torch.Tensor:
    """Transform an single input image to the network input format.
    Args:
        imgs: the input images [height x width x n_channels] or [height x width]
    Returns:
        imgs: the input images in the network format [1 x n_channels x new_width x new_width] or [1 x new_width x new_width]
    """

    # transform to torch tensor
    img = to_torch(img, device=device).to(dtype)

    # transform to channels first (HWC to CHW format)
    if img.ndim == 3:  # color
        img = torch.permute(img, (2, 0, 1))
    elif img.ndim == 2:  # depth
        img = torch.unsqueeze(img, 0)  # add channel dimension
    else:
        raise ValueError("Invalid image dimension.")

    img = normalize_image_channels(img, img_normalize_mean, img_normalize_std) if normalize_values else img

    if add_batch_dim:
        img = img.unsqueeze(0)

    img = resize_images_batch(
        imgs=img,
        new_height=feed_height,
        new_width=feed_width,
    )

    return img


# --------------------------------------------------------------------------------------------------------------------


class DepthModel:
    """
    The depth estimation network.
    Note that we use a network that estimates the disparity and then convert it to depth by taking 1/disparity.
    """

    def __init__(self, depth_lower_bound: float, depth_upper_bound: float, model_name: str, model_path: Path) -> None:
        self.model_name = model_name
        assert model_path is not None, "model_path is None"
        print(f"Loading depth model from {model_path}")
        self.depth_lower_bound = 0 if depth_lower_bound is None else depth_lower_bound
        self.depth_upper_bound = depth_upper_bound

        self.model_info = load_model_model_info(model_path)

        # the dimensions of the input images to the network
        self.feed_width = self.model_info.feed_width
        self.feed_height = self.model_info.feed_height

        # the dimensions of the output depth maps are the same as the input images
        self.depth_map_width = self.model_info.feed_width
        self.depth_map_height = self.model_info.feed_height
        # the output of the network (translation part) needs to be multiplied by this number to get the depth\ego-translations in mm (based on the analysis of sample data in examine_depths.py):
        self.depth_calib_a = self.model_info.depth_calib_a
        self.depth_calib_b = self.model_info.depth_calib_b
        self.depth_calib_type = self.model_info.depth_calib_type
        print(f"depth_calibrations: a={self.depth_calib_a}, b={self.depth_calib_b}")
        self.device = get_device()

        # create the disparity\depth estimation network
        if model_name == "EndoSFM":
            self.disp_net = endo_sfm_DispResNet(num_layers=self.model_info.num_layers, pretrained=True)
            # load the Disparity network
            self.disp_net_path = model_path / "DispNet_best.pt"
            weights = torch.load(self.disp_net_path)
            self.disp_net.load_state_dict(weights["state_dict"], strict=False)
            self.disp_net.to(self.device)
            self.disp_net.eval()  # switch to evaluate mode
            self.dtype = self.disp_net.get_weight_dtype()

        elif model_name == "MonoDepth2":
            monodepth2_utils.download_model_if_doesnt_exist(model_path.name, models_dir=model_path.parent)
            encoder_path = model_path / "encoder.pth"
            depth_decoder_path = model_path / "depth.pth"
            # Get the ResNet-18 model for the depth encoder (ImageNet pre-trained), note the DeptNet gets 1 image as input
            self.encoder = monodepth2_networks.resnet_encoder.ResnetEncoder(
                num_layers=self.model_info.num_layers,
                pretrained=True,
                num_input_images=1,
            )
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
            raise ValueError(f"Unknown depth estimation method: {model_name}")

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
        frame_height = img.shape[0]
        frame_width = img.shape[1]

        # resize and change dimension order of the images to fit the network input format  # [3 x H x W]
        img = img_to_net_in_format(
            img=img,
            device=self.device,
            dtype=self.dtype,
            add_batch_dim=True,
            feed_height=self.feed_height,
            feed_width=self.feed_width,
        )

        # estimate the disparity map
        if self.model_name == "EndoSFM":
            with torch.no_grad():
                disparity_map = self.disp_net(img)  # [N x 1 x H x W]
                # remove the n_sample and n_channels dimension
                disparity_map = disparity_map.squeeze(0).squeeze(0)  # [H x W]
                # convert the disparity to depth
                depth_map = 1 / disparity_map

        elif self.model_name == "MonoDepth2":
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
            raise ValueError(f"Unknown depth estimation method: {self.model_name}")

        # resize to the original image size
        depth_map = resize_single_image(
            img=depth_map,
            new_height=frame_height,
            new_width=frame_width,
        )

        # use the depth calibration to get the depth in mm
        depth_map = self.depth_calib_a * depth_map + self.depth_calib_b

        # clip the depth if needed
        if self.depth_lower_bound is not None or self.depth_upper_bound is not None:
            depth_map = torch.clamp(depth_map, self.depth_lower_bound, self.depth_upper_bound)

        return depth_map


# --------------------------------------------------------------------------------------------------------------------


class EgomotionModel:
    def __init__(self, model_name: str, model_path: Path) -> None:
        print(f"Loading egomotion model from {model_path}")
        assert model_path is not None, "model_path is None"
        self.model_name = model_name
        self.model_info = load_model_model_info(model_path)
        self.n_ref_imgs = self.model_info.n_ref_imgs
        self.device = get_device()
        # the output of the network (translation part) needs to be multiplied by this number to get the depth\ego-translations in mm (based on the analysis of sample data in examine_depths.py):
        self.feed_width = self.model_info.feed_width
        self.feed_height = self.model_info.feed_height

        # create the egomotion estimation network
        if model_name == "EndoSFM":
            pose_net_path = model_path / "PoseNet_best.pt"
            # Get the ResNet-18 model for the pose encoder (ImageNet pre-trained)
            self.pose_net = endo_sfm_PoseResNet(
                num_input_images=1 + self.n_ref_imgs,
                num_frames_to_predict_for=self.n_ref_imgs,
                num_layers=self.model_info.num_layers,
                pretrained=True,
            )
            weights = torch.load(pose_net_path)
            self.pose_net.load_state_dict(weights["state_dict"], strict=False)
            self.pose_net.to(self.device)
            self.pose_net.eval()  # switch to evaluate mode
            self.dtype = self.pose_net.get_weight_dtype()

        elif model_name == "MonoDepth2":
            # based on monodepth2/evaluate_pose.py
            pose_encoder_path = model_path / "pose_encoder.pth"
            pose_decoder_path = model_path / "pose.pth"
            # Get the ResNet-18 model for the pose encoder (ImageNet pre-trained)
            # the pose network gets the target image and the reference images as input and outputs the 6DoF pose parameters from target to reference images
            self.pose_encoder = monodepth2_resnet_encoder.ResnetEncoder(
                num_layers=self.model_info.num_layers,
                pretrained=True,
                num_input_images=self.n_ref_imgs + 1,
            )
            self.pose_decoder = monodepth2_pose_decoder.PoseDecoder(
                num_ch_enc=self.pose_encoder.num_ch_enc,
                num_frames_to_predict_for=self.n_ref_imgs,
            )
            self.pose_encoder.load_state_dict(torch.load(pose_encoder_path))
            # Get the pose decoder: 6DoF pose parameters from target to reference images
            self.pose_decoder = monodepth2_pose_decoder.PoseDecoder(
                num_ch_enc=self.pose_encoder.num_ch_enc,
                num_frames_to_predict_for=self.n_ref_imgs,
            )
            self.pose_decoder.load_state_dict(torch.load(pose_decoder_path))
            self.pose_encoder.to(self.device)
            self.pose_decoder.to(self.device)
            self.pose_encoder.eval()
            self.pose_decoder.eval()
            self.dtype = self.pose_encoder.get_weight_dtype()

        else:
            raise ValueError(f"Unknown egomotion estimation method: {model_name}")

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
            feed_height=self.feed_height,
            feed_width=self.feed_width,
        )
        to_img = img_to_net_in_format(
            img=to_img,
            device=self.device,
            dtype=self.dtype,
            add_batch_dim=True,
            feed_height=self.feed_height,
            feed_width=self.feed_width,
        )

        if self.model_name == "EndoSFM":
            # note: if you want to use the ground truth depth maps, you need to change depth_maps_source to "ground_trutg"
            # "EndoSFM_GTD" is still and estimate, but it was trained with GT depth maps
            with torch.no_grad():
                pose_out = self.pose_net(from_img, to_img)
            # this returns the estimated egomotion [N x 6] 6DoF pose parameters from target to reference  in the order of tx, ty, tz, rx, ry, rz
            pose_out = pose_out.squeeze(0)  # remove the n_sample dimension
            translation = pose_out[:3]
            rotation_axis_angle = pose_out[3:]

        elif self.model_name == "MonoDepth2":
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
            raise ValueError(f"Unknown egomotion estimation method: {self.model_name}")
        # convert the axis-angle to quaternion
        rotation_quaternion = axis_angle_to_quaternion(rotation_axis_angle)

        # multiply the translation by the conversion factor to get mm units
        translation = self.depth_calib_a * translation
        egomotion = torch.cat((translation, rotation_quaternion), dim=0)
        assert egomotion.ndim == 1
        return egomotion

    # --------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------

import queue
from pathlib import Path

import numpy as np
import torch

from colon_nav.net_train.train_utils import load_model_model_info
from colon_nav.util.pose_transforms import invert_pose_motion
from colon_nav.util.rotations_util import axis_angle_to_quaternion
from colon_nav.util.torch_util import get_device

# --------------------------------------------------------------------------------------------------------------------


class DepthModel:
    """
    The depth estimation network.
    Note that we use a network that estimates the disparity and then convert it to depth by taking 1/disparity.
    """

    def __init__(self, depth_lower_bound: float, depth_upper_bound: float, model_path: Path) -> None:
        assert model_path is not None, "model_path is None"
        print(f"Loading depth model from {model_path}")
        self.depth_lower_bound = 0 if depth_lower_bound is None else depth_lower_bound
        self.depth_upper_bound = depth_upper_bound

        self.model_info = load_model_model_info(model_path)
        self.model_name = self.model_info.model_name

        # the dimensions of the input images to the network
        self.feed_width = self.model_info.feed_width
        self.feed_height = self.model_info.depth_model_feed_height

        # the dimensions of the output depth maps are the same as the input images
        self.depth_map_width = self.model_info.feed_width
        self.depth_map_height = self.model_info.depth_model_feed_height
        # the output of the network (translation part) needs to be multiplied by this number to get the depth\ego-translations in mm (based on the analysis of sample data in examine_depths.py):
        self.depth_calib_a = self.model_info.depth_calib_a
        self.depth_calib_b = self.model_info.depth_calib_b
        self.depth_calib_type = self.model_info.depth_calib_type
        print(f"depth_calibrations: a={self.depth_calib_a}, b={self.depth_calib_b}")
        self.device = get_device()

        # TODO: use @torch.compile on the models

        # # create the disparity\depth estimation network
        # if self.model_name == "EndoSFM":
        #     self.disp_net = endo_sfm_DispResNet(num_layers=self.model_info.num_layers, pretrained=True)
        #     # load the Disparity network
        #     self.disp_net_path = model_path / "DispNet_best.pt"
        #     weights = torch.load(self.disp_net_path)
        #     self.disp_net.load_state_dict(weights["state_dict"], strict=False)
        #     self.disp_net.to(self.device)
        #     self.disp_net.eval()  # switch to evaluate mode
        #     self.dtype = self.disp_net.get_weight_dtype()

        # elif self.model_name == "MonoDepth2":
        #     monodepth2_utils.download_model_if_doesnt_exist(model_path.name, models_dir=model_path.parent)
        #     encoder_path = model_path / "encoder.pth"
        #     depth_decoder_path = model_path / "depth.pth"
        #     # Get the ResNet-18 model for the depth encoder (ImageNet pre-trained), note the DeptNet gets 1 image as input
        #     self.encoder = monodepth2_networks.resnet_encoder.ResnetEncoder(
        #         num_layers=self.model_info.num_layers,
        #         pretrained=True,
        #         num_input_images=1,
        #     )
        #     self.depth_decoder = monodepth2_depth_decoder.DepthDecoder(
        #         num_ch_enc=self.encoder.num_ch_enc,
        #         scales=range(4),
        #     )
        #     loaded_dict_enc = torch.load(encoder_path)
        #     filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        #     self.encoder.load_state_dict(filtered_dict_enc)
        #     loaded_dict = torch.load(depth_decoder_path)
        #     self.depth_decoder.load_state_dict(loaded_dict)
        #     self.encoder.to(self.device)
        #     self.depth_decoder.to(self.device)
        #     self.encoder.eval()
        #     self.depth_decoder.eval()
        #     self.dtype = self.encoder.get_weight_dtype()

        # else:
        #     raise ValueError(f"Unknown depth estimation method: {self.model_name}")

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

        # # resize and change dimension order of the images to fit the network input format  # [3 x H x W]
        # img = img_to_net_in_format(
        #     img=img,
        #     device=self.device,
        #     dtype=self.dtype,
        #     add_batch_dim=True,
        #     feed_height=self.feed_height,
        #     feed_width=self.feed_width,
        # )

        # # estimate the disparity map
        # if self.model_name == "EndoSFM":
        #     with torch.no_grad():
        #         disparity_map = self.disp_net(img)  # [N x 1 x H x W]
        #         # remove the n_sample and n_channels dimension
        #         disparity_map = disparity_map.squeeze(0).squeeze(0)  # [H x W]
        #         # convert the disparity to depth
        #         depth_map = 1 / disparity_map

        # elif self.model_name == "MonoDepth2":
        #     # based on monodepth2/evaluate_depth.py
        #     with torch.no_grad():
        #         encoded = self.encoder(img)
        #         output = self.depth_decoder(encoded)
        #         disparity_map = output[("disp", 0)]  # [N x 1 x H x W]
        #         # remove the n_sample and n_channels dimension
        #         disparity_map = disparity_map.squeeze(0).squeeze(0)  # [H x W]
        #         # convert the disparity to depth
        #         _, depth_map = monodepth2_layers.disp_to_depth(
        #             disparity_map,
        #             min_depth=self.depth_lower_bound,
        #             max_depth=self.depth_upper_bound,
        #         )
        # else:
        #     raise ValueError(f"Unknown depth estimation method: {self.model_name}")

        # # resize to the original image size
        # depth_map = resize_single_image(
        #     img=depth_map,
        #     new_height=frame_height,
        #     new_width=frame_width,
        # )

        # # use the depth calibration to get the depth in mm
        # depth_map = self.depth_calib_a * depth_map + self.depth_calib_b

        depth_map = 0

        # clip the depth if needed
        if self.depth_lower_bound is not None or self.depth_upper_bound is not None:
            depth_map = torch.clamp(depth_map, self.depth_lower_bound, self.depth_upper_bound)

        return depth_map


# --------------------------------------------------------------------------------------------------------------------


class EgomotionModel:
    def __init__(self, model_path: Path) -> None:
        print(f"Loading egomotion model from {model_path}")
        assert model_path is not None, "model_path is None"
        self.model_info = load_model_model_info(model_path)
        self.model_name = self.model_info.model_name
        self.n_ref_imgs = self.model_info.n_ref_imgs
        self.device = get_device()
        # the output of the network (translation part) needs to be multiplied by this number to get the depth\ego-translations in mm (based on the analysis of sample data in examine_depths.py):
        self.depth_model_feed_width = self.model_info.depth_model_feed_width
        self.depth_model_feed_height = self.model_info.depth_model_feed_height

        # TODO: use transform

        # # create the egomotion estimation network
        # if self.model_name == "EndoSFM":
        #     pose_net_path = model_path / "PoseNet_best.pt"
        #     # Get the ResNet-18 model for the pose encoder (ImageNet pre-trained)
        #     self.pose_net = endo_sfm_PoseResNet(
        #         num_input_images=1 + self.n_ref_imgs,
        #         num_frames_to_predict_for=self.n_ref_imgs,
        #         num_layers=self.model_info.num_layers,
        #         pretrained=True,
        #     )
        #     weights = torch.load(pose_net_path)
        #     self.pose_net.load_state_dict(weights["state_dict"], strict=False)
        #     self.pose_net.to(self.device)
        #     self.pose_net.eval()  # switch to evaluate mode
        #     self.dtype = self.pose_net.get_weight_dtype()

        # # elif self.model_name == "MonoDepth2":
        #     # based on monodepth2/evaluate_pose.py
        #     # Get the ResNet-18 model for the pose encoder (ImageNet pre-trained)
        #     # the pose network gets the target image and the reference images as input and outputs the 6DoF pose parameters from target to reference images
        #     self.pose_net = monodepth2_pose_net.PoseNet(
        #         n_ref_imgs=self.n_ref_imgs,
        #         num_layers=self.model_info.num_layers,
        #     )
        #     self.pose_encoder.load_state_dict(torch.load(model_path / "pose.pth"))
        #     self.pose_net.to(self.device)
        #     self.pose_net.eval()
        # #     self.dtype = self.pose_net.get_weight_dtype()

        # else:
        #     raise ValueError(f"Unknown egomotion estimation method: {self.model_name}")

    # --------------------------------------------------------------------------------------------------------------------

    def estimate_egomotion(self, cur_rgb_frame: np.ndarray, prev_rgb_frames: queue.Queue[np.ndarray]) -> torch.Tensor:
        """Estimate the 6DOF egomotion from the from image to to image.
            The egomotion is the pose change from the previous frame to the current frame, defined in the previous frame system.
        Args:
            cur_rgb_frame: the current RGB image [3 x H x W]
            prev_rgb_frames: the previous RGB images: queue of length N with elements [3 x H x W] (N is the number of reference images the PoseNet gets as input)
        Returns:
            egomotions: the estimated egomotions [N x 7]  6DoF pose parameters from each reference image to the current image in the order of (tx, ty, tz, qw, qx, qy, qz)
            where (x, y, z) is the translation [mm] and (qw, qx, qy , qz) is the unit-quaternion of the rotation.
        """
        assert cur_rgb_frame.shape == cur_rgb_frame.shape  # same shape
        assert cur_rgb_frame.ndim == 3  # [3 x H x W]
        self.depth_calib_a = self.model_info.depth_calib_a

        # resize and change dimension order of the images to fit the network input format  # [3 x H x W]
        # cur_rgb_frame = img_to_net_in_format(
        #     img=cur_rgb_frame,
        #     device=self.device,
        #     dtype=self.dtype,
        #     add_batch_dim=True,
        #     feed_height=self.feed_height,
        #     feed_width=self.feed_width,
        # )
        # Create a list of the reference images (frames at t-n_ref_imgs, t-n_ref_imgs+1, ..., t-1)
        ref_imgs = []
        # for prev_img in prev_rgb_frames.queue:
        #     img = img_to_net_in_format(
        #         img=prev_img,
        #         device=self.device,
        #         dtype=self.dtype,
        #         add_batch_dim=True,
        #         feed_height=self.feed_height,
        #         feed_width=self.feed_width,
        #     )
        #     ref_imgs.append(img)

        if self.model_name == "EndoSFM":
            # Get the estimated egomotion from the target (current frame) image to the reference images (previous frames)
            with torch.no_grad():
                est_motion_cur_to_ref_imgs = self.pose_net(target_img=cur_rgb_frame, ref_imgs=ref_imgs)
            est_motion_cur_to_ref_imgs = est_motion_cur_to_ref_imgs.squeeze(0)  # remove the batch dimension
            # Get the egomotion from the previous frame (last ref frame) to the current frame (target image)
            est_motion_cur_to_prev = est_motion_cur_to_ref_imgs[-1]  # [6] 6DoF pose parameters
            est_trans_cur_to_prev = est_motion_cur_to_prev[:3]  # [3] translation
            est_rot_cur_to_prev = est_motion_cur_to_prev[3:]  # [3] rotation (axis-angle)

        elif self.model_name == "MonoDepth2":
            # based on monodepth2/evaluate_pose.py
            with torch.no_grad():
                # concat the input images in axis 1 (channel dimension)
                net_input = torch.cat((cur_rgb_frame, *ref_imgs), dim=1)
                features = [self.pose_encoder(net_input)]
                est_rot_cur_to_ref_imgs, est_trans_cur_to_ref_imgs = self.pose_decoder(features)
            # Take the last reference image as the previous image, and remove the batch dimension
            est_trans_cur_to_prev = est_trans_cur_to_ref_imgs.squeeze(0)[-1]  # [3] translation
            est_rot_cur_to_prev = est_rot_cur_to_ref_imgs.squeeze(0)[-1]
        else:
            raise ValueError(f"Unknown egomotion estimation method: {self.model_name}")

        # convert the axis-angle to quaternion
        est_rot_cur_to_prev = axis_angle_to_quaternion(est_rot_cur_to_prev)
        # multiply the translation by the conversion factor to get mm units
        est_trans_cur_to_prev = self.depth_calib_a * est_trans_cur_to_prev
        motion_cur_to_prev = torch.cat((est_trans_cur_to_prev, est_rot_cur_to_prev), dim=0)
        # Invert the motion to het the motion from the previous frame to the current frame:
        egomotion = invert_pose_motion(motion_cur_to_prev)
        return egomotion

    # --------------------------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------------------------

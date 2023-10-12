import random

import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose

from colon_nav.net_train.train_utils import ModelInfo
from colon_nav.util.pose_transforms import compose_poses, get_pose, get_pose_delta
from colon_nav.util.rotations_util import axis_angle_to_quaternion
from colon_nav.util.torch_util import to_torch

# ---------------------------------------------------------------------------------------------------------------------


def get_train_transform(model_info: ModelInfo):
    # set data transforms
    transform_list = [
        ToTensors(dtype=torch.float32, model_info=model_info),
        NormalizeImageChannels(model_info=model_info),
        RandomHorizontalFlip(flip_prob=0.5, model_info=model_info),
        AddRelativePose(model_info=model_info),
        AddInvIntrinsics(),
    ]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------


def get_val_transform(model_info: ModelInfo):
    # set data transforms
    transform_list = [
        ToTensors(dtype=torch.float32, model_info=model_info),
        NormalizeImageChannels(model_info=model_info),
        AddRelativePose(model_info=model_info),
        AddInvIntrinsics(),
    ]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------
class ToTensors:
    def __init__(self, dtype: torch.dtype, model_info: ModelInfo):
        self.device = torch.device("cpu")  # we use the CPU to allow for multi-processing
        self.dtype = dtype
        self.depth_map_size = model_info.depth_map_size
        self.depth_map_resizer = torchvision.transforms.Resize(
            size=self.depth_map_size,
            antialias=True,
        )

        # All the frame shifts that we need to load from the dataset (relative to the target frame)
        self.all_frame_shifts = [*model_info.ref_frame_shifts, 0]

    def __call__(self, sample: dict) -> dict:
        # convert the images to torch tensors
        for shift in self.all_frame_shifts:
            k = ("color", shift)
            sample[k] = to_torch(sample[k], dtype=self.dtype, device=self.device)
            # transform RGB images to channels first (HWC to CHW format)
            sample[k] = torch.permute(sample[k], (2, 0, 1))

        # convert the depth maps to torch tensors
        for shift in self.all_frame_shifts:
            k = ("depth_gt", shift)
            if k in sample:
                # In case we have a depth map.
                sample[k] = to_torch(sample[k], dtype=self.dtype, device=self.device)
                # transform to channels first (HW to CHW format, with C=1)
                sample[k] = torch.unsqueeze(sample[k], dim=0)
                # Resize the depth map to the desired size
                sample[k] = self.depth_map_resizer(sample[k])
            else:
                # create a dummy depth map (with all NaN) for the target frame (CHW format)
                sample[k] = torch.full(
                    (1, *self.depth_map_size),
                    fill_value=np.nan,
                    dtype=self.dtype,
                    device=self.device,
                )

        # convert the camera intrinsics to torch tensors
        sample["K"] = to_torch(sample["K"], dtype=self.dtype, device=self.device)

        # convert the absolute camera poses to torch tensors
        for shift in self.all_frame_shifts:
            k = ("abs_pose", shift)
            if k in sample:
                sample[k] = to_torch(sample[k], dtype=self.dtype, device=self.device)
            else:
                # create a dummy pose for the target frame with NaN values
                sample[k] = torch.full((7,), fill_value=np.nan, dtype=self.dtype, device=self.device)

        # String that lists the augmentations done on the sample:
        sample["augments"] = "Augmentations: "

        return sample


# --------------------------------------------------------------------------------------------------------------------


def get_tgt_im_size(sample: dict):
    # Note: we assume the images were converted by the ImgsToNetInput transform to be of size [..., H, W]
    im_height, im_width = sample[("color", 0)].shape[-2:]
    return im_height, im_width


# --------------------------------------------------------------------------------------------------------------------


class RandomHorizontalFlip:
    """Randomly flips the images horizontally.
    Note that we only use horizontal flipping, since it allows to adjust the pose accordingly by easy transformation.
    """

    def __init__(self, flip_prob: float, model_info: ModelInfo):
        self.flip_prob = flip_prob
        self.device = torch.device("cpu")  # we use the CPU to allow for multi-processing
        self.all_frames_shifts = [*model_info.ref_frame_shifts, 0]

    def __call__(self, sample: dict):
        flip_x = random.random() < self.flip_prob
        if not flip_x:
            return sample
        im_height, im_width = get_tgt_im_size(sample)
        # Note: we assume the images were converted by the ImgsToNetInput transform to be of size [..., H, W]

        # flip the images
        for shift in self.all_frames_shifts:
            sample[("color", shift)] = torch.flip(sample[("color", shift)], dims=[-1])
            sample[("depth_gt", shift)] = torch.flip(sample[("depth_gt", shift)], dims=[-1])

        # flip the x-coordinate of the camera center
        sample["K"][0, 2] = im_width - sample["K"][0, 2]

        # in case we have the ground-truth camera pose, we need to rotate the pose to fit the x-flipped image.
        # i.e., we need to rotate the pose around the y-axis by 180 degrees
        if ("abs_pose", 0) in sample:
            rot_axis = torch.tensor([0, 1, 0], device=self.device)
            rot_angle = np.pi
            rot_quat = axis_angle_to_quaternion(axis_angle=rot_axis * rot_angle)
            aug_pose = get_pose(rot_quat=rot_quat, device=self.device)
            for shift in self.all_frames_shifts:
                # apply the pose-augmentation to the cam poses
                # the pose format: (x,y,z,qw,qx,qy,qz)
                new_pose = compose_poses(pose1=sample[("abs_pose", shift)], pose2=aug_pose)
                sample[("abs_pose", shift)] = new_pose.reshape(7)
        sample["augments"] += " flip_x, "
        return sample


# --------------------------------------------------------------------------------------------------------------------


class AddInvIntrinsics:
    """ " Extends the sample with the inverse camera intrinsics matrix (use this after scale in the transform chain)"""

    def __call__(self, sample):
        if "K" in sample:
            sample["inv_K"] = torch.linalg.inv(sample["K"])
        return sample


# --------------------------------------------------------------------------------------------------------------------


def normalize_image_channels(img: torch.Tensor, img_normalize_mean: float = 0.45, img_normalize_std: float = 0.225):
    # normalize the values from [0,255] to [0, 1]
    img = img / 255
    # normalize the values to the mean and std of the ImageNet dataset
    img = (img - img_normalize_mean) / img_normalize_std
    return img


# --------------------------------------------------------------------------------------------------------------------


class NormalizeImageChannels:
    def __init__(self, model_info: ModelInfo):
        # Normalize the image channels to the mean and std of the ImageNet dataset
        self.mean = model_info.img_normalize_mean
        self.std = model_info.img_normalize_std
        self.all_frames_shifts = [*model_info.ref_frame_shifts, 0]

    def __call__(self, sample: dict):
        for shift in self.all_frames_shifts:
            sample[("color", shift)] = normalize_image_channels(sample[("color", shift)], self.mean, self.std)
        return sample


# --------------------------------------------------------------------------------------------------------------------


class AddRelativePose:
    """Extends the sample with the relative pose between the reference and target frames
    (in case the ground-truth poses are available)
    pose format: (x,y,z,qw,qx,qy,qz)
    """

    def __init__(self, model_info: ModelInfo):
        self.ref_frame_shifts = model_info.ref_frame_shifts

    def __call__(self, sample):
        if ("abs_pose", 0) in sample:
            # get the relative pose between the target and reference frames
            for shift in self.ref_frame_shifts:
                sample[("tgt_to_ref_pose", shift)] = get_pose_delta(
                    pose1=sample[("abs_pose", 0)],  # target frame
                    pose2=sample[("abs_pose", shift)],  # reference frame
                ).reshape(7)
        return sample


# ---------------------------------------------------------------------------------------------------------------------

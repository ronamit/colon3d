import random

import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose
from torchvision.transforms.functional import crop

from colon3d.net_train.common_transforms import normalize_image_channels
from colon3d.util.pose_transforms import compose_poses, get_pose, get_pose_delta
from colon3d.util.rotations_util import axis_angle_to_quaternion, quaternion_to_axis_angle
from colon3d.util.torch_util import get_device, to_torch

# ---------------------------------------------------------------------------------------------------------------------


def get_transforms(feed_height: int, feed_width: int) -> (Compose, Compose):
    train_trans = get_train_transform(feed_height=feed_height, feed_width=feed_width)
    val_trans = get_validation_transform(feed_height=feed_height, feed_width=feed_width)
    return train_trans, val_trans


# ---------------------------------------------------------------------------------------------------------------------


def get_train_transform(feed_height: int, feed_width: int) -> Compose:
    """Training transform for EndoSFM"""
    # set data transforms
    transform_list = [
        AllToTorch(dtype=torch.float32, device=get_device(), feed_height=feed_height, feed_width=feed_width),
        RandomHorizontalFlip(),
        RandomScaleCrop(max_scale=1.15),
        NormalizeImageChannels(),
        AddInvIntrinsics(),
        AddRelativePose(),
    ]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------


def get_validation_transform(feed_height: int, feed_width: int) -> Compose:
    """Validation transform for EndoSFM"""
    transform_list = [
        AllToTorch(dtype=torch.float32, device=get_device(), feed_height=feed_height, feed_width=feed_width),
        NormalizeImageChannels(),
        AddInvIntrinsics(),
        AddRelativePose(),
    ]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------


def get_sample_image_keys(sample: dict, img_type: str = "all") -> list:
    """Get the possible names of the images in the sample dict"""
    rgb_image_keys = ["target_img", "ref_img"]
    depth_image_keys = ["target_depth", "ref_depth"]
    if img_type == "RGB":
        img_names = rgb_image_keys
    elif img_type == "depth":
        img_names = depth_image_keys
    elif img_type == "all":
        img_names = rgb_image_keys + depth_image_keys
    else:
        raise ValueError(f"Invalid image type: {img_type}")
    img_keys = [k for k in sample if k in img_names or isinstance(k, tuple) and k[0] in img_names]
    return img_keys


# ---------------------------------------------------------------------------------------------------------------------


def get_orig_im_size(sample: dict):
    # Note: we assume the images were converted to be of size [..., H, W]
    im_height, im_width = sample["target_img"].shape[-2:]
    return im_height, im_width


# ---------------------------------------------------------------------------------------------------------------------


class AllToTorch:
    def __init__(self, device: torch.device, dtype: torch.dtype, feed_height: int, feed_width: int):
        self.device = device
        self.dtype = dtype
        self.resizer = torchvision.transforms.Resize(size=(feed_height, feed_width), antialias=True)

    def __call__(self, sample: dict) -> dict:
        rgb_img_keys = get_sample_image_keys(sample, img_type="RGB")
        depth_img_keys = get_sample_image_keys(sample, img_type="depth")
        for k, v in sample.items():
            sample[k] = to_torch(v, dtype=self.dtype, device=self.device)
            if k in rgb_img_keys:
                # transform to channels first (HWC to CHW format)
                sample[k] = torch.permute(sample[k], (2, 0, 1))
                # resize the image to the feed size (if needed)
                sample[k] = self.resizer(sample[k])

            elif k in depth_img_keys:
                # transform to channels first (HW to CHW format)
                sample[k] = torch.unsqueeze(sample[k], dim=0)
                # resize the image to the feed size (if needed)
                sample[k] = self.resizer(sample[k])
        return sample


# --------------------------------------------------------------------------------------------------------------------


class RandomHorizontalFlip:
    """Randomly flips the images horizontally.
    Note that we only use horizontal flipping, since it allows to adjust the pose accordingly by easy transformation.
    """

    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob
        self.device = get_device()

    def __call__(self, sample: dict):
        flip_x = random.random() < self.flip_prob
        if not flip_x:
            return sample

        img_keys = get_sample_image_keys(sample)
        im_height, im_width = get_orig_im_size(sample)
        # Note: we assume the images were converted by the ImgsToNetInput transform to be of size [..., H, W]

        # flip the images
        for k in img_keys:
            img = sample[k]
            sample[k] = torch.flip(img, dims=[-1])

        # flip the x-coordinate of the camera center
        sample["intrinsics_K"][0, 2] = im_width - sample["intrinsics_K"][0, 2]

        # in case we have the ground-truth camera pose, we need to rotate the pose to fit the x-flipped image.
        # i.e., we need to rotate the pose around the y-axis by 180 degrees
        if "ref_abs_pose" in sample:
            rot_axis = torch.tensor([0, 1, 0], device=self.device)
            rot_angle = np.pi
            rot_quat = axis_angle_to_quaternion(rot_axis_angle=rot_axis * rot_angle)
            aug_pose = get_pose(rot_quat=rot_quat)
            # apply the pose-augmentation to the reference pose and the target pose
            # the pose format: (x,y,z,qw,qx,qy,qz)
            ref_abs_pose = compose_poses(pose1=sample["ref_abs_pose"], pose2=aug_pose).reshape(7)
            tgt_abs_pose = compose_poses(pose1=sample["tgt_abs_pose"], pose2=aug_pose).reshape(7)
            sample["ref_abs_pose"] = ref_abs_pose
            sample["tgt_abs_pose"] = tgt_abs_pose

        return sample


# --------------------------------------------------------------------------------------------------------------------


class RandomScaleCrop:
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __init__(self, max_scale=1.15):
        self.max_scale = max_scale

    def __call__(self, sample: dict) -> dict:
        images_keys = get_sample_image_keys(sample)

        # draw the scaling factor
        zoom_factor = 1 + np.random.rand() * (self.max_scale - 1)

        # draw the offset ratio
        x_offset_ratio, y_offset_ratio = np.random.uniform(0, 1, 2)

        for k in images_keys:
            img = sample[k]
            in_h, in_w = img.shape[-2:]

            scaled_h, scaled_w = int(in_h * zoom_factor), int(in_w * zoom_factor)
            offset_y = np.round(y_offset_ratio * (scaled_h - in_h)).astype(int)
            offset_x = np.round(x_offset_ratio * (scaled_w - in_w)).astype(int)

            # increase the size of the image to be able to crop the same size as before
            scaled_image = torchvision.transforms.Resize((scaled_h, scaled_w), antialias=True)(img)
            cropped_image = crop(img=scaled_image, top=offset_y, left=offset_x, height=in_h, width=in_w)
            sample[k] = cropped_image

        sample["intrinsics_K"][0, :] *= zoom_factor
        sample["intrinsics_K"][1, :] *= zoom_factor
        sample["intrinsics_K"][0, 2] -= offset_x
        sample["intrinsics_K"][1, 2] -= offset_y

        return sample


# --------------------------------------------------------------------------------------------------------------------


class AddInvIntrinsics:
    """Extends the sample with the inverse camera intrinsics matrix"""

    def __call__(self, sample):
        sample["intrinsics_inv_K"] = torch.linalg.inv(sample["intrinsics_K"])
        return sample


# --------------------------------------------------------------------------------------------------------------------


class NormalizeImageChannels:
    def __init__(self, mean: float = 0.45, std: float = 0.225):
        # Normalize the image channels to the mean and std of the ImageNet dataset
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict):
        img_keys = get_sample_image_keys(sample, img_type="RGB")
        for k in img_keys:
            sample[k] = normalize_image_channels(sample[k], self.mean, self.std)
        return sample


# --------------------------------------------------------------------------------------------------------------------


class AddRelativePose:
    """Extends the sample with the relative pose between the reference and target frames
    (in case the ground-truth poses are available)
    pose format: (x,y,z,qw,qx,qy,qz)
    """

    def __call__(self, sample):
        if "ref_abs_pose" in sample and "tgt_abs_pose" in sample:
            # get the relative pose
            sample["tgt_to_ref_pose"] = get_pose_delta(pose1=sample["tgt_abs_pose"], pose2=sample["ref_abs_pose"]).reshape(7)
            sample["ref_to_tgt_pose"] =  get_pose_delta(pose1=sample["ref_abs_pose"], pose2=sample["tgt_abs_pose"]).reshape(7)
        return sample


# --------------------------------------------------------------------------------------------------------------------


def poses_to_enfosfm_format(poses: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """Convert the poses from our format to the format used by EndoSFM"
    The EndoSFM code works with  tx, ty, tz, rx, ry, rz  (translation and rotation in axis-angle format)
    Our data in in the format:  x, y, z, qw, qx, qy, qz  (translation and rotation in quaternion format)
    Args:
        poses: the poses in our format [N x 7] (tx, ty, tz, qw, qx, qy, qz)
    Returns:
        poses: the poses in the EndoSFM format [N x 6] (tx, ty, tz, rx, ry, rz)
    """
    n = poses.shape[0]
    poses_new = torch.zeros((n, 6), dtype=dtype, device=poses.device)
    poses_new[:, :3] = poses[:, :3]
    poses_new[:, 3:] = quaternion_to_axis_angle(poses[:, 3:])
    return poses_new



# --------------------------------------------------------------------------------------------------------------------


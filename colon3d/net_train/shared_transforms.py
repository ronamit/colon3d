import random

import numpy as np
import torch
import torchvision

from colon3d.alg.depth_and_ego_models import normalize_image_channels
from colon3d.net_train.train_utils import DatasetMeta
from colon3d.util.pose_transforms import compose_poses, get_pose, get_pose_delta
from colon3d.util.rotations_util import axis_angle_to_quaternion
from colon3d.util.torch_util import get_device, to_device, to_torch

# ---------------------------------------------------------------------------------------------------------------------


class AllToTorch:
    def __init__(self, dtype: torch.dtype, dataset_meta: DatasetMeta):
        self.device = torch.device("cpu")  # we use the CPU to allow for multi-processing
        self.dtype = dtype
        self.feed_height = dataset_meta.feed_height
        self.feed_width = dataset_meta.feed_width
        self.num_input_images = dataset_meta.num_input_images
        self.resizer = torchvision.transforms.Resize(
            size=(self.feed_height, self.feed_width),
            antialias=True,
        )

    def __call__(self, sample: dict) -> dict:
        # convert the images to torch tensors
        for i in range(self.num_input_images):
            k = ("color", i)
            sample[k] = to_torch(sample[k], dtype=self.dtype, device=self.device)
            # transform RGB images to channels first (HWC to CHW format)
            sample[k] = torch.permute(sample[k], (2, 0, 1))
            # resize the RGB image to the feed size (if needed)
            sample[k] = self.resizer(sample[k])

        # convert the depth maps to torch tensors
        for i in range(self.num_input_images):
            k = ("depth_gt", i)
            if k in sample:
                sample[k] = to_torch(sample[k], dtype=self.dtype, device=self.device)
                # transform to channels first (HW to CHW format, with C=1)
                sample[k] = torch.unsqueeze(sample[k], dim=0)
                # resize the image to the feed size (if needed)
                sample[k] = self.resizer(sample[k])

        # convert the camera intrinsics to torch tensors
        sample["K"] = to_torch(sample["K"], dtype=self.dtype, device=self.device)

        # convert the absolute camera poses to torch tensors
        for i in range(self.num_input_images):
            k = ("abs_pose", i)
            if k in sample:
                sample[k] = to_torch(sample[k], dtype=self.dtype, device=self.device)
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

    def __init__(self, flip_prob: float, dataset_meta: DatasetMeta):
        self.flip_prob = flip_prob
        self.device = torch.device("cpu")  # we use the CPU to allow for multi-processing
        self.num_input_images = dataset_meta.num_input_images
        self.load_gt_pose = dataset_meta.load_gt_pose

    def __call__(self, sample: dict):
        flip_x = random.random() < self.flip_prob
        if not flip_x:
            return sample
        im_height, im_width = get_tgt_im_size(sample)
        # Note: we assume the images were converted by the ImgsToNetInput transform to be of size [..., H, W]

        # flip the images
        for i in range(self.num_input_images):
            sample[("color", i)] = torch.flip(sample[("color", i)], dims=[-1])
            sample[("depth_gt", i)] = torch.flip(sample[("depth_gt", i)], dims=[-1])

        # flip the x-coordinate of the camera center
        sample["K"][0, 2] = im_width - sample["K"][0, 2]

        # in case we have the ground-truth camera pose, we need to rotate the pose to fit the x-flipped image.
        # i.e., we need to rotate the pose around the y-axis by 180 degrees
        if self.load_gt_pose:
            rot_axis = torch.tensor([0, 1, 0], device=self.device)
            rot_angle = np.pi
            rot_quat = axis_angle_to_quaternion(axis_angle=rot_axis * rot_angle)
            aug_pose = get_pose(rot_quat=rot_quat, device=self.device)
            for i in range(self.num_input_images):
                # apply the pose-augmentation to the cam poses
                # the pose format: (x,y,z,qw,qx,qy,qz)
                sample[("abs_pose", i)] = compose_poses(pose1=sample[("abs_pose", i)], pose2=aug_pose).reshape(7)
        return sample


# --------------------------------------------------------------------------------------------------------------------


class AddInvIntrinsics:
    """ " Extends the sample with the inverse camera intrinsics matrix (use this after scale in the transform chain)"""

    def __init__(self, n_scales: int = 0):
        self.n_scales = n_scales

    def __call__(self, sample):
        sample["inv_K"] = torch.linalg.inv(sample["K"])
        if self.n_scales is not None:
            for i_scale in range(self.n_scales):
                sample[("inv_K", i_scale)] = torch.linalg.pinv(sample["K", i_scale])
        return sample


# --------------------------------------------------------------------------------------------------------------------


class NormalizeImageChannels:
    def __init__(self, dataset_meta: DatasetMeta, mean: float = 0.45, std: float = 0.225):
        # Normalize the image channels to the mean and std of the ImageNet dataset
        self.mean = mean
        self.std = std
        self.num_input_images = dataset_meta.num_input_images

    def __call__(self, sample: dict):
        for i in range(self.num_input_images):
            sample[("color", i)] = normalize_image_channels(sample[("color", i)], self.mean, self.std)
        return sample


# --------------------------------------------------------------------------------------------------------------------


class AddRelativePose:
    """Extends the sample with the relative pose between the reference and target frames
    (in case the ground-truth poses are available)
    pose format: (x,y,z,qw,qx,qy,qz)
    """

    def __init__(self, dataset_meta: DatasetMeta):
        self.load_gt_pose = dataset_meta.load_gt_pose
        self.num_input_images = dataset_meta.num_input_images

    def __call__(self, sample):
        if self.load_gt_pose:
            # get the relative pose between the target and reference frames
            for i in range(self.num_input_images):  # note: we start from 1 since the target frame has id 0
                sample[("tgt_to_ref_pose", i)] = get_pose_delta(
                    pose1=sample[("abs_pose", 0)],
                    pose2=sample[("abs_pose", i)],
                ).reshape(7)
        # note: sample[("tgt_to_ref_pose", 0)] should be the identity pose
        return sample


# ---------------------------------------------------------------------------------------------------------------------

def sample_to_gpu(sample: dict, device: torch.device | None = None) -> dict:
    """
    Note: this must be applied only after a sampled batch is created by the data loader.
    See: https://github.com/pytorch/pytorch/issues/98002#issuecomment-1511972876
    """
    if device is None:
        device = get_device()
    for k, v in sample.items():
        sample[k] = to_device(v, device=device)
    return sample


# ---------------------------------------------------------------------------------------------------------------------

import random

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from torchvision.transforms import Compose

from colon3d.net_train.common_transforms import (
    normalize_image_channels,
    resize_tensor_image,
)
from colon3d.util.general_util import replace_keys
from colon3d.util.pose_transforms import compose_poses, get_pose, get_pose_delta
from colon3d.util.rotations_util import axis_angle_to_quaternion, quaternion_to_axis_angle
from colon3d.util.torch_util import get_device, to_torch

# ---------------------------------------------------------------------------------------------------------------------

"""
The format of each data item that MonoDepth2 expects is a dictionary with the following keys:
    Values correspond to torch tensors.
    Keys in the dictionary are either strings or tuples:

        ("color", <frame_id>, <scale>)          for raw colour images,
        ("color_aug", <frame_id>, <scale>)      for augmented colour images,
        ("K", scale) or ("inv_K", scale)        for camera intrinsics,
        "stereo_T"                              for camera extrinsics, and [for stereo pairs only - we don't use this]
        "depth_gt"                              ground truth depth maps of the target frame.

    <frame_id> is either:
        an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index', [we use 0=target, 1=ref]
    or
        "s" for the opposite image in the stereo pair. [we don't use this]

    <scale> is an integer representing the scale of the image relative to the fullsize image:
        0       images resized to (self.width,      self.height     )
        1       images resized to (self.width // 2, self.height // 2)
        2       images resized to (self.width // 4, self.height // 4)
        3       images resized to (self.width // 8, self.height // 8)


"""
# ---------------------------------------------------------------------------------------------------------------------


def get_transforms(n_scales: int, feed_height: int, feed_width: int) -> (Compose, Compose):
    train_trans = get_train_transform(n_scales=n_scales, feed_height=feed_height, feed_width=feed_width)
    val_trans = get_validation_transform(n_scales=n_scales, feed_height=feed_height, feed_width=feed_width)
    return train_trans, val_trans


# ---------------------------------------------------------------------------------------------------------------------


def get_train_transform(n_scales: int, feed_height: int, feed_width: int):
    """Training transform for MonoDepth2.
    Args:
        n_scales: number of scales the network expects
        im_height: height of the input image at scale 0
        im_width: width of the input image at scale 0
    """

    # set data transforms
    transform_list = [
        MonoDepth2Format(),
        AllToTorch(dtype=torch.float32, device=get_device(), feed_height=feed_height, feed_width=feed_width),
        SwitchTargetAndRef(prob=0.5),
        ColorJitter(
            field_postfix="_aug",
            p=0.5,
        ),
        RandomHorizontalFlip(flip_prob=0.5),
        RandomScaleCrop(max_scale=1.15),
        CreateScalesArray(n_scales=n_scales),
        AddInvIntrinsics(n_scales=n_scales),
        NormalizeImageChannels(),
        AddRelativePose(),
    ]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------


def get_validation_transform(n_scales: int, feed_height: int, feed_width: int):
    """Validation transform for MonoDepth2
    Args:
    n_scales: number of scales the network expects
    im_height: height of the input image at scale 0
    im_width: width of the input image at scale 0
    """
    transform_list = [
        MonoDepth2Format(),
        AllToTorch(dtype=torch.float32, device=get_device(), feed_height=feed_height, feed_width=feed_width),
        ColorJitter(
            field_postfix="_aug",
            p=0.0,
        ),  # no jitter - use this just for consistency with MonoDepth2 format
        CreateScalesArray(n_scales=n_scales),
        NormalizeImageChannels(),
        AddInvIntrinsics(n_scales=n_scales),
        AddRelativePose(),
    ]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------


class MonoDepth2Format:
    def __call__(self, sample: dict) -> dict:
        replace_keys(sample, old_key="target_img", new_key=("color", 0))
        replace_keys(sample, old_key="ref_img", new_key=("color", 1))
        replace_keys(sample, old_key="target_depth", new_key=("depth_gt"))
        replace_keys(sample, old_key="ref_depth", new_key=("ref_depth_gt"))
        replace_keys(sample, old_key="intrinsics_K", new_key="K")

        # Normalize the intrinsic matrix (as suggested in monodepth2/datasets/kitti_dataset.py)
        target_img = sample[("color", 0)]
        sample["K"][0, :] /= target_img.width
        sample["K"][1, :] /= target_img.height
        sample["K"] = intrinsic_mat_to_4x4(sample["K"])
        return sample


# ---------------------------------------------------------------------------------------------------------------------
def intrinsic_mat_to_4x4(K: torch.Tensor) -> torch.Tensor:
    K_new = torch.eye(4)
    K_new[:3, :3] = K
    return K_new


# ---------------------------------------------------------------------------------------------------------------------


def get_sample_image_keys(sample: dict, img_type="all") -> list:
    """Get the possible names of the images in the sample dict"""
    rgb_image_keys = ["color", "color_aug"]
    depth_image_keys = ["depth_gt", "ref_depth_gt"]
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
    # Note: we assume the images were converted by the ImgsToNetInput transform to be of size [..., H, W]
    im_height, im_width = sample[("color", 0)].shape[-2:]
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


# ---------------------------------------------------------------------------------------------------------------------


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
        sample["K"][0, 2] = im_width - sample["K"][0, 2]

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


# ---------------------------------------------------------------------------------------------------------------------


class RandomScaleCrop:
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __init__(self, max_scale=1.15):
        self.max_scale = max_scale

    def __call__(self, sample: dict) -> dict:
        images_keys = get_sample_image_keys(sample)
        # Note: we assume the images were converted by the ImgsToNetInput transform to be of size [..., H, W]

        # draw the scaling factor
        assert self.max_scale >= 1
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
            cropped_image = TF.crop(img=scaled_image, top=offset_y, left=offset_x, height=in_h, width=in_w)
            sample[k] = cropped_image

        sample["K"][0, :] *= zoom_factor
        sample["K"][1, :] *= zoom_factor
        sample["K"][0, 2] -= offset_x
        sample["K"][1, 2] -= offset_y
        return sample


# ---------------------------------------------------------------------------------------------------------------------

class SwitchTargetAndRef:
    """Randomly switches the target and reference images.
    I.e. we reverse the time axis.
    Note that we also need to switch the ground-truth depth maps and the camera poses accordingly.
    """
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(self, sample: dict) -> dict:
        if random.random() < self.prob:
            sample["target_frame_idx"], sample["ref_frame_idx"] = sample["ref_frame_idx"], sample["target_frame_idx"]
            sample["color", 0], sample["color", 1] = sample["color", 1], sample["color", 0]
            sample["depth_gt"], sample["ref_depth_gt"] = sample["ref_depth_gt"], sample["depth_gt"]
            if "ref_abs_pose" in sample:
                sample["ref_abs_pose"], sample["tgt_abs_pose"] = sample["tgt_abs_pose"], sample["ref_abs_pose"]
        return sample


# ---------------------------------------------------------------------------------------------------------------------

# Color jittering transform
class ColorJitter:
    def __init__(
        self,
        field_postfix: str,
        p=0.5,
        brightness=(0.8, 1.2),
        contrast=(0.8, 1.2),
        saturation=(0.8, 1.2),
        hue=(-0.1, 0.1),
    ):
        # based on monodepth2/datasets/mono_dataset.py
        # note that they apply this after the scaling transform
        self.field_postfix = field_postfix
        self.p = p
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, sample: dict) -> dict:
        rgb_im_names = get_sample_image_keys(sample, img_type="RGB")
        # We create the color_mapping mapping in advance and apply the same augmentation to all
        # images in this item. This ensures that all images input to the pose network receive the
        # same augmentation.
        (
            _,
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
        ) = torchvision.transforms.ColorJitter.get_params(
            brightness=self.color_jitter.brightness,
            contrast=self.color_jitter.contrast,
            saturation=self.color_jitter.saturation,
            hue=self.color_jitter.hue,
        )
        do_augment = random.random() < self.p
        for im_name in rgb_im_names:
            img = sample[im_name]
            if do_augment:
                img = TF.adjust_brightness(img, brightness_factor)
                img = TF.adjust_contrast(img, contrast_factor)
                img = TF.adjust_saturation(img, saturation_factor)
                img = TF.adjust_hue(img, hue_factor)
            new_name = (im_name[0] + self.field_postfix, *im_name[1:])
            sample[new_name] = img
        return sample


# ---------------------------------------------------------------------------------------------------------------------


class AddInvIntrinsics:
    """ " Extends the sample with the inverse camera intrinsics matrix (use this after scale in the transform chain)"""

    def __init__(self, n_scales: int):
        self.n_scales = n_scales

    def __call__(self, sample):
        sample["inv_K"] = torch.linalg.inv(sample["K"])
        if self.n_scales is not None:
            for i_scale in range(self.n_scales):
                sample[("inv_K", i_scale)] = torch.linalg.pinv(sample["K", i_scale])
        return sample


# --------------------------------------------------------------------------------------------------------------------


# create scales array transform
class CreateScalesArray:
    """ " Extends the sample with scaled variants of the images
    <scale> is an integer representing the scale of the image relative to the fullsize image:
        0       images resized to (self.width,      self.height     )
        1       images resized to (self.width // 2, self.height // 2)
        2       images resized to (self.width // 4, self.height // 4)
        3       images resized to (self.width // 8, self.height // 8)
    based on monodepth2/datasets/mono_dataset.py
    """

    def __init__(self, n_scales: int):
        self.n_scales = n_scales
        self.resize = {}
        self.scale_factors = [2**i for i in range(self.n_scales)]

    def __call__(self, sample: dict) -> dict:
        # Note that only the RGB images are scaled
        image_names = get_sample_image_keys(sample, img_type="RGB")
        for i_scale in range(self.n_scales):
            for im_name in image_names:
                img = sample[im_name]
                img_height, img_width = img.shape[-2:]
                scaled_img = resize_tensor_image(
                    img=img,
                    new_height=np.round(img_height / self.scale_factors[i_scale]).astype(int),
                    new_width=np.round(img_width / self.scale_factors[i_scale]).astype(int),
                )
                sample[(*im_name, i_scale)] = scaled_img
                sample[("K", i_scale)] = sample["K"].clone()
                s = self.scale_factors[i_scale]
                sample[("K", i_scale)][0, :] /= s
                sample[("K", i_scale)][1, :] /= s
        return sample


# ---------------------------------------------------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------------------------------------------------


class AddRelativePose:
    """Extends the sample with the relative pose between the reference and target frames
    (in case the ground-truth poses are available)
    pose format: (x,y,z,qw,qx,qy,qz)
    """

    def __call__(self, sample):
        if "ref_abs_pose" in sample and "tgt_abs_pose" in sample:
            # get the relative pose
            sample["tgt_to_ref_pose"] = get_pose_delta(
                pose1=sample["tgt_abs_pose"],
                pose2=sample["ref_abs_pose"],
            ).reshape(7)
            sample["ref_to_tgt_pose"] = get_pose_delta(
                pose1=sample["ref_abs_pose"],
                pose2=sample["tgt_abs_pose"],
            ).reshape(7)
        return sample


# --------------------------------------------------------------------------------------------------------------------


def poses_to_md2_format(poses: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """Convert the poses from our format to the format used by EndoSFM"
    The EndoSFM code works with  tx, ty, tz, rx, ry, rz  (translation and rotation in axis-angle format)
    Our data in in the format:  x, y, z, qw, qx, qy, qz  (translation and rotation in quaternion format)
    Args:
        poses: the poses in our format [N x 7]
    Returns:
        translation [N x 3] and axis-angle rotation [N x 3]
    """
    # check that all poses are valid (finite)
    assert torch.isfinite(poses).all(), "Poses should be finite."
    translation = poses[:, :3].to(dtype=dtype)
    axisangle = quaternion_to_axis_angle(poses[:, 3:]).to(dtype=dtype)
    assert torch.isfinite(translation).all(), "Translation should be finite."
    assert torch.isfinite(axisangle).all(), "Axis-angle should be finite."
    return translation, axisangle


# --------------------------------------------------------------------------------------------------------------------

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
from colon3d.util.torch_util import get_device, to_torch

"""
The format of each data item that MonoDepth2 expects is a dictionary with the following keys:
    Values correspond to torch tensors.
    Keys in the dictionary are either strings or tuples:

        ("color", <frame_id>, <scale>)          for raw colour images,
        ("color_aug", <frame_id>, <scale>)      for augmented colour images,
        ("K", scale) or ("inv_K", scale)        for camera intrinsics,
        "stereo_T"                              for camera extrinsics, and [for stereo pairs only - we don't use this]
        "depth_gt"                              for ground truth depth maps.

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


def get_transforms(n_scales: int):
    train_trans = get_train_transform(n_scales=n_scales)
    val_trans = get_validation_transform(n_scales=n_scales)
    return train_trans, val_trans


# ---------------------------------------------------------------------------------------------------------------------


def get_train_transform(n_scales: int):
    """Training transform for MonoDepth2.
    Args:
        n_scales: number of scales the network expects
        im_height: height of the input image at scale 0
        im_width: width of the input image at scale 0
    """

    # set data transforms
    transform_list = [
        MonoDepth2Format(),
        AllToTorch(dtype=torch.float32, device=get_device()),
        ColorJitter(
            field_postfix="_aug",
            p=0.5,
        ),
        RandomFlip(flip_x_p=0.5, flip_y_p=0.5),
        RandomScaleCrop(max_scale=1.15),
        CreateScalesArray(n_scales=n_scales),
        AddInvIntrinsics(n_scales=n_scales),
        NormalizeImageChannels(),
    ]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------


def get_validation_transform(n_scales: int):
    """Validation transform for MonoDepth2
    Args:
    n_scales: number of scales the network expects
    im_height: height of the input image at scale 0
    im_width: width of the input image at scale 0
    """
    transform_list = [
        MonoDepth2Format(),
        AllToTorch(dtype=torch.float32, device=get_device()),
        ColorJitter(
            field_postfix="_aug",
            p=0.0,
        ),  # no jitter - use this just for consistency with MonoDepth2 format
        CreateScalesArray(n_scales=n_scales),
        NormalizeImageChannels(),
        AddInvIntrinsics(n_scales=n_scales),
    ]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------


class MonoDepth2Format:
    def __call__(self, sample: dict) -> dict:
        replace_keys(sample, old_key="target_img", new_key=("color", 0))
        replace_keys(sample, old_key="ref_img", new_key=("color", 1))
        replace_keys(sample, old_key="target_depth", new_key="depth_gt")
        replace_keys(sample, old_key="ref_depth", new_key="depth_gt_ref")
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
    depth_image_keys = ["depth_gt", "depth_gt_ref"]
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
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype

    def __call__(self, sample: dict) -> dict:
        rgb_img_keys = get_sample_image_keys(sample, img_type="RGB")
        depth_img_keys = get_sample_image_keys(sample, img_type="depth")
        for k, v in sample.items():
            sample[k] = to_torch(v, dtype=self.dtype, device=self.device)
            if k in rgb_img_keys:
                # transform to channels first (HWC to CHW format)
                sample[k] = torch.permute(sample[k], (2, 0, 1))
            elif k in depth_img_keys:
                # transform to channels first (HW to CHW format)
                sample[k] = torch.unsqueeze(sample[k], dim=0)
        return sample





# ---------------------------------------------------------------------------------------------------------------------


# Random horizontal flip transform
class RandomFlip:
    def __init__(self, flip_x_p=0.5, flip_y_p=0.5):
        self.flip_x_p = flip_x_p
        self.flip_y_p = flip_y_p

    def __call__(self, sample: dict):
        im_height, im_width = get_orig_im_size(sample)
        img_keys = get_sample_image_keys(sample)
        # Note: we assume the images were converted by the ImgsToNetInput transform to be of size [..., H, W]
        flip_x = random.random() < self.flip_x_p
        flip_y = random.random() < self.flip_y_p
        if flip_x:
            for k in img_keys:
                img = sample[k]
                sample[k] = torch.flip(img, dims=[-1])
            sample["K"][0, 2] = im_width - sample["K"][0, 2]

            if flip_y:
                for k in img_keys:
                    img = sample[k]
                    sample[k] = torch.flip(img, dims=[-2])
                sample["K"][1, 2] = im_height  - sample["K"][1, 2]
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
        image_names = get_sample_image_keys(sample)
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

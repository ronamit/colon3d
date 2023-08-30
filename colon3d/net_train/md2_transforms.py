import random

import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose

from colon3d.net_train.common_transforms import (
    AllToNumpy,
    img_to_net_in_format,
    resize_image,
)
from colon3d.util.general_util import replace_keys
from colon3d.util.torch_util import get_device

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


def get_train_transform(n_scales: int = 4):
    """Training transform for MonoDepth2"""

    # set data transforms
    transform_list = [
        MonoDepth2Format(),
        AllToNumpy(),
        NormalizeCamIntrinsicMat(),
        ColorJitter(
            field_postfix="_aug",
            p=0.5,
            brightness=(0.8, 1.2),
            contrast=(0.8, 1.2),
            saturation=(0.8, 1.2),
            hue=(-0.1, 0.1),
        ),
        RandomFlip(flip_x_p=0.5, flip_y_p=0.5),
        RandomScaleCrop(max_scale=1.15),
        CreateScalesArray(n_scales=n_scales),
        ImgsToNetInput(),
        AddInvIntrinsics(n_scales=n_scales),
    ]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------


def get_validation_transform(n_scales: int = 4):
    """Validation transform for MonoDepth2"""
    transform_list = [
        MonoDepth2Format(),
        AllToNumpy(),
        NormalizeCamIntrinsicMat(),
        CreateScalesArray(n_scales=n_scales),
        ImgsToNetInput(),
        AddInvIntrinsics(n_scales=n_scales),
    ]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------


class MonoDepth2Format:
    def __call__(self, sample: dict) -> dict:
        replace_keys(sample, old_key="target_img", new_key=("color", 0))
        replace_keys(sample, old_key="ref_img", new_key=("color", 1))
        replace_keys(sample, old_key="target_depth", new_key="depth_gt")
        replace_keys(sample, old_key="intrinsics_K", new_key="K")
        sample["K"] = intrinsic_mat_to_4x4(sample["K"])
        return sample


# ---------------------------------------------------------------------------------------------------------------------
def intrinsic_mat_to_4x4(K: torch.Tensor) -> torch.Tensor:
    K_new = torch.eye(4)
    K_new[:3, :3] = K
    return K_new


# ---------------------------------------------------------------------------------------------------------------------


def get_sample_image_keys(sample: dict, im_type="all") -> list:
    """Get the names of the images in the sample dict"""
    if im_type == "all":
        img_names = ["color", "color_aug", "depth_gt"]
    elif im_type == "RGB":
        img_names = ["color", "color_aug"]
    else:
        raise ValueError(f"Invalid image type: {im_type}")
    img_keys = [k for k in sample if k in img_names or isinstance(k, tuple) and k[0] in img_names]
    return img_keys


# ---------------------------------------------------------------------------------------------------------------------
class NormalizeCamIntrinsicMat:
    def __call__(self, sample: dict) -> dict:
        img = sample[("color", 0)]
        im_height, im_width = img.size
        sample["K"][0, :] /= im_width
        sample["K"][1, :] /= im_height
        return sample


# ---------------------------------------------------------------------------------------------------------------------


def get_orig_im_size(sample: dict):
    return sample[("color", 0)].shape


# ---------------------------------------------------------------------------------------------------------------------
# Random horizontal flip transform
class RandomFlip:
    def __init__(self, flip_x_p=0.5, flip_y_p=0.5):
        self.flip_x_p = flip_x_p
        self.flip_y_p = flip_y_p

    def __call__(self, sample: dict):
        im_size = get_orig_im_size(sample)
        img_keys = get_sample_image_keys(sample)
        flip_x = random.random() < self.flip_x_p
        flip_y = random.random() < self.flip_y_p
        if flip_x:
            for k in img_keys:
                img = sample[k]
                sample[k] = np.flip(img, axis=1)
            sample["K"][0, 2] = im_size[1] - sample["K"][0, 2]

            if flip_y:
                for k in img_keys:
                    img = sample[k]
                    sample[k] = np.flip(img, axis=0)
                sample["K"][1, 2] = im_size[0] - sample["K"][1, 2]
        return sample


# ---------------------------------------------------------------------------------------------------------------------


class RandomScaleCrop:
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __init__(self, max_scale=1.15):
        self.max_scale = max_scale

    def __call__(self, sample: dict) -> dict:
        images_keys = get_sample_image_keys(sample)

        # draw the scaling factor
        x_scaling, y_scaling = np.random.uniform(1, self.max_scale, 2)

        # draw the offset ratio
        x_offset_ratio, y_offset_ratio = np.random.uniform(0, 1, 2)

        for k in images_keys:
            img = sample[k]
            in_h, in_w = img.shape[:2]

            scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)
            offset_y = np.round(y_offset_ratio * (scaled_h - in_h)).astype(int)
            offset_x = np.round(x_offset_ratio * (scaled_w - in_w)).astype(int)

            scaled_image = resize_image(img=img, new_height=scaled_h, new_width=scaled_w)
            cropped_image = scaled_image[offset_y : offset_y + in_h, offset_x : offset_x + in_w]
            sample[k] = cropped_image

        sample["K"][0, :] *= x_scaling
        sample["K"][1, :] *= y_scaling
        sample["K"][0, 2] -= offset_x
        sample["K"][1, 2] -= offset_y

        return sample


# ---------------------------------------------------------------------------------------------------------------------


class ImgsToNetInput:
    def __init__(self, img_keys: list, img_normalize_mean=None, img_normalize_std=None, dtype=torch.float32):
        self.device = get_device()
        self.dtype = dtype
        self.img_normalize_mean = img_normalize_mean
        self.img_normalize_std = img_normalize_std
        self.img_keys = img_keys

    def __call__(self, sample: dict) -> dict:
        images_keys = get_sample_image_keys(sample)
        for k in images_keys:
            sample[k] = img_to_net_in_format(
                img=sample[k],
                device=self.device,
                dtype=self.dtype,
                normalize_values=True,
                img_normalize_mean=self.img_normalize_mean,
                img_normalize_std=self.img_normalize_std,
                new_height=None,  # no reshape
                new_width=None,
            )
        return sample


# ---------------------------------------------------------------------------------------------------------------------


# Color jittering transform
class ColorJitter:
    def __init__(
        self,
        field_postfix: str = "",
        p=0.5,
        brightness=(0.8, 1.2),
        contrast=(0.8, 1.2),
        saturation=(0.8, 1.2),
        hue=(-0.1, 0.1),
    ):
        # based on monodepth2/datasets/mono_dataset.py
        self.field_postfix = field_postfix
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
        )

    def __call__(self, sample: dict) -> dict:
        rgb_im_names = get_sample_image_keys(sample, im_type="RGB")
        if random.random() < self.p:
            for im_name in rgb_im_names:
                sample[im_name + self.field_postfix] = self.color_jitter(sample[im_name])
        return sample


# ---------------------------------------------------------------------------------------------------------------------


class AddInvIntrinsics:
    """ " Extends the sample with the inverse camera intrinsics matrix (use this after scale in the transform chain)"""

    def __init__(self, n_scales: int | None = None):
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
    """

    def __init__(self, n_scales: int):
        self.n_scales = n_scales

    def __call__(self, sample: dict) -> dict:
        image_names = get_sample_image_keys(sample)
        for scale in range(self.n_scales):
            for im_name in image_names:
                sample[(im_name, scale)] = torch.nn.functional.interpolate(
                    sample[im_name].unsqueeze(0),
                    scale_factor=2**scale,
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(0)
                sample[("K", scale)] = sample["K"].clone()
                sample[("K", scale)][0, :] /= 2**scale
                sample[("K", scale)][1, :] /= 2**scale
        return sample


# ---------------------------------------------------------------------------------------------------------------------

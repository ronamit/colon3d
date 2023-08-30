import random

import numpy as np
import torch
from torchvision.transforms import Compose

from colon3d.net_train.common_transforms import (
    AllToNumpy,
    img_to_net_in_format,
    resize_image,
)
from colon3d.util.torch_util import get_device

# ---------------------------------------------------------------------------------------------------------------------


def get_train_transform() -> Compose:
    """Training transform for EndoSFM"""
    # set data transforms
    transform_list = [
        AllToNumpy(),
        RandomFlip(),
        RandomScaleCrop(max_scale=1.15),
        ImgsToNetInput(),
        AddInvIntrinsics(),
    ]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------


def get_validation_transform():
    """Validation transform for EndoSFM"""
    transform_list = [AllToNumpy(), ImgsToNetInput()]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------


def get_sample_image_keys(sample: dict) -> list:
    """Get the names of the images in the sample dict"""
    img_names = ["target_img", "ref_img", "target_depth"]
    img_keys = [k for k in sample if k in img_names or isinstance(k, tuple) and k[0] in img_names]
    return img_keys


# ---------------------------------------------------------------------------------------------------------------------


# Random horizontal flip transform
class RandomFlip:
    def __init__(self, flip_x_p=0.5, flip_y_p=0.5):
        self.flip_x_p = flip_x_p
        self.flip_y_p = flip_y_p

    def __call__(self, sample: dict):
        flip_x = random.random() < self.flip_x_p
        flip_y = random.random() < self.flip_y_p
        img_keys = get_sample_image_keys(sample)
        if flip_x:
            for k in img_keys:
                img = sample[k]
                sample[k] = np.flip(img, axis=1)
            sample["intrinsics_K"][0, 2] = sample["target_img"].shape[2] - sample["intrinsics_K"][0, 2]

            if flip_y:
                for k in img_keys:
                    img = sample[k]
                    sample[k] = np.flip(img, axis=0)
                sample["intrinsics_K"][1, 2] = sample["target_img"].shape[1] - sample["intrinsics_K"][1, 2]
        return sample


# --------------------------------------------------------------------------------------------------------------------


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

        sample["intrinsics_K"][0, :] *= x_scaling
        sample["intrinsics_K"][1, :] *= y_scaling
        sample["intrinsics_K"][0, 2] -= offset_x
        sample["intrinsics_K"][1, 2] -= offset_y

        return sample


# --------------------------------------------------------------------------------------------------------------------
class ImgsToNetInput:
    def __init__(self, img_normalize_mean: float = 0.45, img_normalize_std: float = 0.225, dtype=torch.float32):
        self.device = get_device()
        self.dtype = dtype
        self.img_normalize_mean = img_normalize_mean
        self.img_normalize_std = img_normalize_std

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


# --------------------------------------------------------------------------------------------------------------------


class AddInvIntrinsics:
    """Extends the sample with the inverse camera intrinsics matrix"""

    def __call__(self, sample):
        sample["intrinsics_inv_K"] = np.linalg.inv(sample["intrinsics_K"])
        return sample


# --------------------------------------------------------------------------------------------------------------------

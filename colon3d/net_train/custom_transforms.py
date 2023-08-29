import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

from colon3d.util.general_util import path_to_str
from colon3d.util.torch_util import get_device, to_torch

# --------------------------------------------------------------------------------------------------------------------


def get_image_names():
    # the names of the images in the sample of the dataset defined in colon3d/net_train/scenes_dataset.py
    return ["target_img", "ref_img", "target_depth"]


# --------------------------------------------------------------------------------------------------------------------


def get_sample_image_names(sample: dict):
    # the names of the images in the sample (the key is part of the defined names, or is a tuple which the first element is part of the defined names)
    return [k for k in sample if k in get_image_names() or (k is tuple and k[0] in get_image_names())]


# --------------------------------------------------------------------------------------------------------------------
def resize_image(img: np.ndarray, new_height: int, new_width: int) -> np.ndarray:
    """Resize an image of shape [height x width] or [height x width x n_channels]"""
    if img.ndim == 1:  # depth
        return cv2.resize(img, dsize=(new_width, new_height), fx=1.0, fy=1.0, interpolation=cv2.INTER_NEAREST)
    if img.ndim == 3:  # color
        return cv2.resize(img, dsize=(new_width, new_height), interpolation=cv2.INTER_LINEAR)
    raise ValueError("Invalid image dimension.")


# --------------------------------------------------------------------------------------------------------------------


def img_to_net_in_format(
    img: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    normalize_values: bool = True,
    img_normalize_mean: float = 0.45,
    img_normalize_std: float = 0.225,
    new_height: int | None = None,
    new_width: int | None = None,
    add_batch_dim: bool = False,
) -> torch.Tensor:
    """Transform an single input image to the network input format.
    Args:
        imgs: the input images [height x width x n_channels] or [height x width]
    Returns:
        imgs: the input images in the network format [1 x n_channels x new_width x new_width] or [1 x new_width x new_width]
    """
    height, width = img.shape[:2]

    if new_height and new_width and (height, width) != (new_height, new_width):
        # resize the images
        img = resize_image(img, new_height=new_height, new_width=new_width)

    # transform to channels first (HWC to CHW format)
    if img.ndim == 3:  # color
        img = np.transpose(img, (2, 0, 1))
    elif img.ndim == 2:  # depth
        img = np.expand_dims(img, axis=0)
    else:
        raise ValueError("Invalid image dimension.")

    # transform to torch tensor and normalize the values to [0, 1]
    img = to_torch(img, device=device).to(dtype)

    if normalize_values:
        # normalize the values from [0,255] to [0, 1]
        img = img / 255
        # normalize the values to the mean and std of the ImageNet dataset
        img = (img - img_normalize_mean) / img_normalize_std
    if add_batch_dim:
        img = img.unsqueeze(0)
    return img


# --------------------------------------------------------------------------------------------------------------------


# Random horizontal flip transform
class RandomFlip:
    def __init__(self, flip_x_p=0.5, flip_y_p=0.5):
        self.flip_x_p = flip_x_p
        self.flip_y_p = flip_y_p

    def __call__(self, sample: dict):
        flip_x = random.random() < self.flip_x_p
        flip_y = random.random() < self.flip_y_p
        image_names = get_sample_image_names(sample)

        if flip_x:
            for img_name in image_names:
                img = sample[img_name]
                sample[img_name] = np.flip(img, axis=1)
            sample["intrinsics_K"][0, 2] = sample["target_img"].shape[2] - sample["intrinsics_K"][0, 2]

            if flip_y:
                for img_name in image_names:
                    img = sample[img_name]
                    sample[img_name] = np.flip(img, axis=0)
                sample["intrinsics_K"][1, 2] = sample["target_img"].shape[1] - sample["intrinsics_K"][1, 2]
        return sample


# ---------------------------------------------------------------------------------------------------------------------


class ImagesToNumpy:
    def __call__(self, sample: dict) -> dict:
        image_names = get_sample_image_names(sample)
        for k in image_names:
            sample[k] = np.array(sample[k])
        return sample


# ---------------------------------------------------------------------------------------------------------------------


class RandomScaleCrop:
    """Randomly zooms images up to 15% and crop them to keep same size as before."""

    def __init__(self, max_scale=1.15):
        self.max_scale = max_scale

    def __call__(self, sample: dict) -> dict:
        image_names = get_sample_image_names(sample)

        # draw the scaling factor
        x_scaling, y_scaling = np.random.uniform(1, self.max_scale, 2)

        # draw the offset ratio
        x_offset_ratio, y_offset_ratio = np.random.uniform(0, 1, 2)

        for im_name in image_names:
            img = sample[im_name]
            in_h, in_w = img.shape[:2]

            scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)
            offset_y = np.round(y_offset_ratio * (scaled_h - in_h)).astype(int)
            offset_x = np.round(x_offset_ratio * (scaled_w - in_w)).astype(int)

            scaled_image = resize_image(img=img, new_height=scaled_h, new_width=scaled_w)
            cropped_image = scaled_image[offset_y : offset_y + in_h, offset_x : offset_x + in_w]
            sample[im_name] = cropped_image

        sample["intrinsics_K"][0, :] *= x_scaling
        sample["intrinsics_K"][1, :] *= y_scaling
        sample["intrinsics_K"][0, 2] -= offset_x
        sample["intrinsics_K"][1, 2] -= offset_y

        return sample


# --------------------------------------------------------------------------------------------------------------------


# Color jittering transform
class ColorJitter:
    def __init__(self, p=0.5, brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1)):
        # based on monodepth2/datasets/mono_dataset.py
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
        if random.random() < self.p:
            sample["target_img"] = self.color_jitter(sample["target_img"])
            sample["ref_img"] = self.color_jitter(sample["ref_img"])
        return sample


# ---------------------------------------------------------------------------------------------------------------------


# create scales array transform
class CreateScalesArray:
    """ " Extends the sample with scaled variants of the images
    <scale> is an integer representing the scale of the image relative to the fullsize image:
        -1      images at native resolution as loaded from disk
        0       images resized to (self.width,      self.height     )
        1       images resized to (self.width // 2, self.height // 2)
        2       images resized to (self.width // 4, self.height // 4)
        3       images resized to (self.width // 8, self.height // 8)
    """

    def __init__(self, n_scales: int):
        self.n_scales = n_scales

    def __call__(self, sample: dict) -> dict:
        sample["scales"] = self.n_scales
        for scale in range(self.n_scales):
            if scale == -1:
                # scale == -1 means the original image
                for k, v in sample.items():
                    sample[(k, scale)] = v
            else:
                for im_name in ["target_img", "ref_img"]:
                    sample[(im_name, scale)] = torch.nn.functional.interpolate(
                        sample[im_name].unsqueeze(0),
                        scale_factor=2**scale,
                        mode="bilinear",
                        align_corners=True,
                    ).squeeze(0)
                    sample[("intrinsics_K", scale)] = sample["intrinsics_K"].clone()
                    sample[("intrinsics_K", scale)][0, :] /= 2**scale
                    sample[("intrinsics_K", scale)][1, :] /= 2**scale
        return sample


# --------------------------------------------------------------------------------------------------------------------


class AddInvIntrinsics:
    """ " Extends the sample with the inverse camera intrinsics matrix (use this after scale in the transform chain)"""
    def __init__(self, n_scales: int | None = None):
        self.n_scales = n_scales

    def __call__(self, sample):
        sample["intrinsics_inv_K"] = torch.linalg.inv(sample["intrinsics_K"])
        if self.n_scales is not None:
            for i_scale in range(self.n_scales):
                sample[("intrinsics_inv_K", i_scale)] = torch.linalg.pinv(sample["intrinsics_K", i_scale])
        return sample


# --------------------------------------------------------------------------------------------------------------------


class ToTensors:
    # Use to make sure all values are tensors
    def __init__(self, img_normalize_mean=None, img_normalize_std=None, dtype=torch.float32):
        self.device = get_device()
        self.dtype = dtype
        self.img_normalize_mean = img_normalize_mean
        self.img_normalize_std = img_normalize_std

    def __call__(self, sample: dict) -> dict:
        image_names = get_sample_image_names(sample)

        for k, v in sample.items():
            if k in image_names:
                sample[k] = img_to_net_in_format(
                    img=v,
                    device=self.device,
                    dtype=self.dtype,
                    normalize_values=True,
                    img_normalize_mean=self.img_normalize_mean,
                    img_normalize_std=self.img_normalize_std,
                    new_height=None,  # no reshape
                    new_width=None,
                )
            elif isinstance(v, Path):
                sample[k] = path_to_str(v)
            else:
                sample[k] = to_torch(v, dtype=self.dtype, device=self.device)

        return sample


# --------------------------------------------------------------------------------------------------------------------


class NormalizeCamIntrinsicMat:
    def __call__(self, sample: dict) -> dict:
        im_height, im_width = sample["target_img"].size
        sample["intrinsics_K"][0, :] /= im_width
        sample["intrinsics_K"][1, :] /= im_height
        return sample


# --------------------------------------------------------------------------------------------------------------------

import random

import numpy as np
import torch
import torchvision
from torchvision.transforms import Compose
from torchvision.transforms.functional import crop

from colon3d.net_train.common_transforms import normalize_image_channels
from colon3d.util.torch_util import get_device, to_torch

# ---------------------------------------------------------------------------------------------------------------------


def get_transforms():
    train_trans = get_train_transform()
    val_trans = get_validation_transform()
    return train_trans, val_trans


# ---------------------------------------------------------------------------------------------------------------------


def get_train_transform() -> Compose:
    """Training transform for EndoSFM"""
    # set data transforms
    transform_list = [
        AllToTorch(dtype=torch.float32, device=get_device()),
        RandomFlip(),
        RandomScaleCrop(max_scale=1.15),
        NormalizeImageChannels(),
        AddInvIntrinsics(),
    ]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------


def get_validation_transform() -> Compose:
    """Validation transform for EndoSFM"""
    transform_list = [
        AllToTorch(dtype=torch.float32, device=get_device()),
        NormalizeImageChannels(),
        AddInvIntrinsics(),
    ]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------


def get_sample_image_keys(sample: dict, img_type: str = "all") -> list:
    """Get the names of the images in the sample dict"""
    if img_type == "RGB":
        img_names = ["target_img", "ref_img"]
    elif img_type == "all":
        img_names = ["target_img", "ref_img", "target_depth"]
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
    def __init__(self, device: torch.device, dtype: torch.dtype):
        self.device = device
        self.dtype = dtype

    def __call__(self, sample: dict) -> dict:
        rgb_img_keys = get_sample_image_keys(sample, img_type="RGB")
        for k, v in sample.items():
            sample[k] = to_torch(v, dtype=self.dtype, device=self.device)
            if k in rgb_img_keys:
                # transform to channels first (HWC to CHW format)
                sample[k] = torch.permute(sample[k], (2, 0, 1))
        return sample


# --------------------------------------------------------------------------------------------------------------------
# Random horizontal flip transform
class RandomFlip:
    def __init__(self, flip_x_p=0.5, flip_y_p=0.5):
        self.flip_x_p = flip_x_p
        self.flip_y_p = flip_y_p

    def __call__(self, sample: dict):
        flip_x = random.random() < self.flip_x_p
        flip_y = random.random() < self.flip_y_p
        img_keys = get_sample_image_keys(sample)
        im_height, im_width = get_orig_im_size(sample)
        # Note: we assume the images were converted by the ImgsToNetInput transform to be of size [..., H, W]
        if flip_x:
            for k in img_keys:
                img = sample[k]
                sample[k] = torch.flip(img, dims=[-1])
            sample["intrinsics_K"][0, 2] = im_width - sample["intrinsics_K"][0, 2]

            if flip_y:
                for k in img_keys:
                    img = sample[k]
                    sample[k] = torch.flip(img, dims=[-2])
                sample["intrinsics_K"][1, 2] = im_height - sample["intrinsics_K"][1, 2]
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
            in_h, in_w = img.shape[-2:]

            scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)
            offset_y = np.round(y_offset_ratio * (scaled_h - in_h)).astype(int)
            offset_x = np.round(x_offset_ratio * (scaled_w - in_w)).astype(int)

            # increase the size of the image to be able to crop the same size as before
            scaled_image = torchvision.transforms.Resize((scaled_h, scaled_w), antialias=True)(img)
            cropped_image = crop(img=scaled_image, top=offset_y, left=offset_x, height=in_h, width=in_w)
            sample[k] = cropped_image

        sample["intrinsics_K"][0, :] *= x_scaling
        sample["intrinsics_K"][1, :] *= y_scaling
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

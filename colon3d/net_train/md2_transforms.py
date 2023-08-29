import torch
from torchvision.transforms import Compose

from colon3d.net_train.custom_transforms import (
    AddInvIntrinsics,
    ColorJitter,
    CreateScalesArray,
    ImagesToNumpy,
    NormalizeCamIntrinsicMat,
    RandomFlip,
    RandomScaleCrop,
    ToTensors,
)
from colon3d.util.general_util import replace_keys

# ---------------------------------------------------------------------------------------------------------------------


class FormatToMonoDepth2:
    """
    The format of each data item that MonoDepth2 expects is a dictionary with the following keys:
        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)


    """

    def __init__(self, n_scales: int = 4):
        self.n_scales = n_scales

    def __call__(self, sample: dict) -> dict:
        for i_scale in range(self.n_scales):
            sample = replace_keys(sample, old_key=("target_img", i_scale), new_key=("color_aug", 0, i_scale))
            sample = replace_keys(sample, old_key=("ref_img", i_scale), new_key=("color_aug", 1, i_scale))
            sample = replace_keys(sample, old_key=("intrinsics_K", i_scale), new_key=("K", i_scale))
            sample[("K", i_scale)] = intrinsic_mat_to_4x4(sample[("K", i_scale)])
            sample = replace_keys(sample, old_key=("intrinsics_inv_K", i_scale), new_key=("inv_K", i_scale))
            sample[("inv_K", i_scale)] = intrinsic_mat_to_4x4(sample[("inv_K", i_scale)])

        sample = replace_keys(sample, old_key="intrinsics_K", new_key="K")
        sample["K"] = intrinsic_mat_to_4x4(sample["K"])
        sample = replace_keys(sample, old_key="intrinsics_inv_K", new_key="inv_K")
        sample["inv_K"] = intrinsic_mat_to_4x4(sample["inv_K"])
        sample = replace_keys(sample, old_key="target_depth", new_key="depth_gt")
        return sample


# ---------------------------------------------------------------------------------------------------------------------
def intrinsic_mat_to_4x4(K: torch.Tensor) -> torch.Tensor:
    K_new = torch.eye(4)
    K_new[:3, :3] = K
    return K_new


# ---------------------------------------------------------------------------------------------------------------------


def get_train_transform(n_scales: int = 4):
    """Training transform for MonoDepth2"""

    # set data transforms
    transform_list = [
        NormalizeCamIntrinsicMat(),
        ColorJitter(p=0.5, brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.1, 0.1)),
        ImagesToNumpy(),
        RandomFlip(flip_x_p=0.5, flip_y_p=0.5),
        RandomScaleCrop(max_scale=1.15),
        ToTensors(img_normalize_mean=0.45, img_normalize_std=0.225),
        CreateScalesArray(n_scales=n_scales),
        AddInvIntrinsics(n_scales=n_scales),
        FormatToMonoDepth2(n_scales=n_scales),
    ]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------


def get_validation_transform(n_scales: int = 4):
    """Validation transform for MonoDepth2"""
    transform_list = [
        ImagesToNumpy(),
        ToTensors(img_normalize_mean=0.45, img_normalize_std=0.225),
        CreateScalesArray(n_scales=n_scales),
        NormalizeCamIntrinsicMat(),
        FormatToMonoDepth2(n_scales=4),
    ]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------

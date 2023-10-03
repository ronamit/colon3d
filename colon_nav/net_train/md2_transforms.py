import numpy as np
import torch
from torchvision.transforms import Compose

from colon_nav.net_train.shared_transforms import (
    AddInvIntrinsics,
    AddRelativePose,
    AllToTorch,
    NormalizeImageChannels,
    RandomHorizontalFlip,
)
from colon_nav.net_train.train_utils import DatasetMeta
from colon_nav.util.rotations_util import quaternion_to_axis_angle
from colon_nav.util.torch_util import resize_tensor_image

# ---------------------------------------------------------------------------------------------------------------------
"""
Note: make sure the tensors are on the CPU before applying the transforms.

Note: we use the transforms suggested in SimCol3D challenge (https://simcol3d.github.io/) - by the EndoAI team.
"""

"""
The format of each data item that MonoDepth2 expects is a dictionary with the following keys:
    Values correspond to torch tensors.
    Keys in the dictionary are either strings or tuples:

        ("color", <i>, <scale>)          for raw colour images,.
        ("K", scale) or ("inv_K", scale)        for camera intrinsics.
        ("depth_gt",  <i>)               ground truth depth map.

    <i> is the frame index within the sample. The target frame has id 0, the reference frames have ids 1, 2, 3, ...
    or
        "s" for the opposite image in the stereo pair. [we don't use this]

    <scale> is an integer representing the scale of the image relative to the fullsize image:
        0       images resized to (self.width,      self.height     )
        1       images resized to (self.width // 2, self.height // 2)
        2       images resized to (self.width // 4, self.height // 4)
        3       images resized to (self.width // 8, self.height // 8)

"""

# ---------------------------------------------------------------------------------------------------------------------


def get_train_transform(n_scales: int, dataset_meta: DatasetMeta):
    """Training transform for MonoDepth2.
    Args:
        n_scales: number of scales the network expects
        im_height: height of the input image at scale 0
        im_width: width of the input image at scale 0
    """

    # set data transforms
    transform_list = [
        MonoDepth2Format(),
        AllToTorch(dtype=torch.float32, dataset_meta=dataset_meta),
        RandomHorizontalFlip(flip_prob=0.5, dataset_meta=dataset_meta),
        CreateScalesArray(n_scales=n_scales, dataset_meta=dataset_meta),
        AddInvIntrinsics(n_scales=n_scales),
        NormalizeImageChannels(dataset_meta=dataset_meta),
        AddRelativePose(dataset_meta=dataset_meta),
    ]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------


def get_val_transform(dataset_meta: DatasetMeta, n_scales: int):
    """Validation transform for MonoDepth2
    Args:
    n_scales: number of scales the network expects
    im_height: height of the input image at scale 0
    im_width: width of the input image at scale 0
    """
    transform_list = [
        MonoDepth2Format(),
        AllToTorch(dtype=torch.float32, dataset_meta=dataset_meta),
        CreateScalesArray(n_scales=n_scales, dataset_meta=dataset_meta),
        NormalizeImageChannels(dataset_meta=dataset_meta),
        AddInvIntrinsics(n_scales=n_scales),
        AddRelativePose(dataset_meta=dataset_meta),
    ]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------


class MonoDepth2Format:
    def __call__(self, sample: dict) -> dict:
        # Normalize the intrinsic matrix (as suggested in monodepth2/datasets/kitti_dataset.py)
        target_img = sample[("color", 0)]
        sample["K"][0, :] /= target_img.width
        sample["K"][1, :] /= target_img.height
        sample["K"] = intrinsic_mat_to_4x4(sample["K"])
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

    def __init__(self, n_scales: int, dataset_meta: DatasetMeta):
        self.n_scales = n_scales
        self.resize = {}
        self.scale_factors = [2**i for i in range(self.n_scales)]
        self.n_ref_imgs = dataset_meta.n_ref_imgs
        self.ref_frame_shifts = dataset_meta.ref_frame_shifts

    def __call__(self, sample: dict) -> dict:
        # Note that only the RGB images are scaled

        for i_scale in range(self.n_scales):
            for shift in [*self.ref_frame_shifts, 0]:
                im_name = ("color", shift)
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
def intrinsic_mat_to_4x4(K: torch.Tensor) -> torch.Tensor:
    K_new = torch.eye(4)
    K_new[:3, :3] = K
    return K_new


# ---------------------------------------------------------------------------------------------------------------------

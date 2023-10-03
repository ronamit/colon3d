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

# ---------------------------------------------------------------------------------------------------------------------
"""
Note: make sure the tensors are on the CPU before applying the transforms.

Note: we use the transforms suggested in SimCol3D challenge (https://arxiv.org/abs/2307.11261) - by the EndoAI team.
"""
# ---------------------------------------------------------------------------------------------------------------------


def get_train_transform(dataset_meta: DatasetMeta) -> Compose:
    """Training transform for EndoSFM"""
    # set data transforms
    transform_list = [
        AllToTorch(dtype=torch.float32, dataset_meta=dataset_meta),
        RandomHorizontalFlip(flip_prob=0.5, dataset_meta=dataset_meta),
        NormalizeImageChannels(dataset_meta=dataset_meta),
        AddInvIntrinsics(),
        AddRelativePose(dataset_meta=dataset_meta),
    ]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------


def get_val_transform(dataset_meta: DatasetMeta) -> Compose:
    """Validation transform for EndoSFM"""
    transform_list = [
        AllToTorch(dtype=torch.float32, dataset_meta=dataset_meta),
        NormalizeImageChannels(dataset_meta=dataset_meta),
        AddInvIntrinsics(),
        AddRelativePose(dataset_meta=dataset_meta),
    ]
    return Compose(transform_list)


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
    assert poses.shape[1] == 7, "Invalid poses"
    assert torch.isfinite(poses).all(), "Invalid poses"
    poses_new = torch.zeros((n, 6), dtype=dtype, device=poses.device)
    poses_new[:, :3] = poses[:, :3]
    poses_new[:, 3:] = quaternion_to_axis_angle(poses[:, 3:])
    assert torch.isfinite(poses_new).all(), "Invalid poses"
    return poses_new


# --------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn.functional as nnF

from colon_nav.util.rotations_util import axis_angle_to_rot_mat

# --------------------------------------------------------------------------------------------------------------------




def compute_pose_loss(
    trans_pred: torch.Tensor,
    trans_gt: torch.Tensor,
    rot_pred: torch.Tensor,
    rot_gt: torch.Tensor,
) -> torch.Tensor:
    """Compute the pose loss between two poses, defined as the smooth L1 distance between the poses in the a 4x4 matrix format.
    Args:
        trans_pred: the predicted translation [N x 3]
        trans_gt: the ground-truth translation [N x 3]
        rot_pred: the predicted rotation [N x 3] in axis-angle format (rx, ry, rz)
        rot_gt: the ground-truth rotation [N x 3] in axis-angle format (rx, ry, rz)
        rot_loss_scale: the scale of the rotation loss relative to the translation loss
    Returns:
        pose_loss: the pose loss [N x 1]
    """
    # Transform the rotation from axis-angle to rotation matrix format and flatten it to a 9-vector
    rot_mat_pred = axis_angle_to_rot_mat(rot_pred).reshape(-1, 9)
    rot_mat_gt = axis_angle_to_rot_mat(rot_gt).reshape(-1, 9)
    # Compute the translation and rotation losses
    trans_loss = nnF.l1_loss(trans_pred, trans_gt, reduction="none").sum(dim=-1)
    rot_loss = nnF.l1_loss(rot_mat_pred, rot_mat_gt, reduction="none").sum(dim=-1)
    # Average over batch:
    trans_loss = trans_loss.mean()
    rot_loss = rot_loss.mean()
    return trans_loss, rot_loss


# --------------------------------------------------------------------------------------------------------------------

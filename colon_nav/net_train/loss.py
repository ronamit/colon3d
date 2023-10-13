import torch
import torch.nn.functional as nnF

from colon_nav.util.rotations_util import quaternions_to_rot_matrices

# --------------------------------------------------------------------------------------------------------------------


def loss_function(
    loss_terms_lambdas: dict,
    tgt_depth_est: torch.Tensor,
    list_tgt_to_refs_motion_est: list[torch.Tensor],
    depth_gt: torch.Tensor,
    list_tgt_to_refs_motion_gt: list[torch.Tensor],
) -> torch.Tensor:
    """Compute the loss function.
    Args:
        loss_terms_lambdas: a dictionary of the loss terms and their weights
        tgt_depth_est: the estimated depth map of the target frame [B x H x W]
        list_tgt_to_refs_motion_est: list of the estimated egomotion from the target to each reference frame [B x 7]
        depth_gt: the ground truth depth map of the target frame [B x H x W]
        list_tgt_to_refs_motion_gt: list of the ground truth egomotion from the target to each reference frame [B x 7]
    Returns:
        total_loss: the total loss
        losses: a dictionary of the losses
    """
    losses = {}
    for loss_name, loss_lambda in loss_terms_lambdas:
        if loss_lambda > 0:
            losses[loss_name] = 0

    # Binary vector indicating which samples in the batch have a ground truth depth map:
    is_depth_gt_available = ~torch.isnan(depth_gt[:, 0, 0])  # (B)

    if loss_terms_lambdas["depth_sup_L1"] > 0:
        # Compute the depth L1 loss for all samples in the batch
        depth_loss_all = nnF.l1_loss(tgt_depth_est, depth_gt, reduction="none")
        # sum over the pixels of each sample in the batch
        losses["depth_sup_L1"] = torch.sum(depth_loss_all[is_depth_gt_available])

    if loss_terms_lambdas["depth_sup_SSIM"] > 0:
        # Compute the depth SSIM loss for all samples in the batch
        depth_loss_all = 1 - nnF.ssim(tgt_depth_est, depth_gt, reduction="none")
        # sum over the pixels of each sample in the batch
        losses["depth_sup_SSIM"] = torch.sum(depth_loss_all[is_depth_gt_available])

    # Compute the pose loss (sum over the reference frames)
    n_ref = len(list_tgt_to_refs_motion_est)
    for i_ref in range(n_ref):
        # Get the estimated egomotion from the target to the current reference frame
        tgt_to_ref_motion_est = list_tgt_to_refs_motion_est[i_ref]  # (B, 7)
        # Estimated translation vector [mm]
        trans_est = tgt_to_ref_motion_est[:, :3]  # (B, 3) (x,y,z) in [mm]
        # Estimated unit quaternion
        rot_quat_est = tgt_to_ref_motion_est[:, 3:]  # (B, 4) (qw,qx,qy,qz) unit quaternion
        # Get the ground truth egomotion from the target to the current reference frame
        tgt_to_ref_motion_gt = list_tgt_to_refs_motion_gt[i_ref]
        # Ground truth translation vector [mm]
        trans_gt = tgt_to_ref_motion_gt[:, :3]  # (B, 3) (x,y,z) in [mm]
        # Ground truth unit quaternion
        rot_quat_gt = tgt_to_ref_motion_gt[:, 3:]  # (B, 4) (qw,qx,qy,qz) unit quaternion

        if "trans_sup_L1_quat" in losses:
            # Add the L1 loss between the GT and estimated translation vectors of this ref frame (sum over the samples in the batch)
            losses["pose_sup_L1_quat"] += nnF.l1_loss(trans_est, trans_gt, reduction="sum")

        if "rot_sup_L1_quat" in losses:
            # Add the L1 loss between the GT and estimated unit quaternion of this ref frame (sum over the samples in the batch)
            losses["rot_sup_L1_quat"] += nnF.l1_loss(rot_quat_est, rot_quat_gt, reduction="sum")

        if "rot_sup_L1_mat" in losses:
            # Compute the pose loss between two poses, defined as the L1 distance between the poses in the a flattened 4x4 matrix format.
            # The pose format: (x,y,z,qw,qx,qy,qz) where (x,y,z) is the translation [mm] and (qw,qx,qy,qz) is the rotation unit quaternion.
            rot_mat_est = quaternions_to_rot_matrices(rot_quat_est).view(-1, 16)  # (B, 16)
            rot_mat_gt = quaternions_to_rot_matrices(rot_quat_gt).view(-1, 16)  # (B, 16)
            # Add the L1 loss between the GT and estimated poses for this ref frame (sum over the samples in the batch)
            losses["rot_sup_L1_mat"] += nnF.l1_loss(rot_mat_est, rot_mat_gt, reduction="sum")

    # Scale the losses by their weights
    losses_scaled = {loss_name: val * loss_terms_lambdas[loss_name] for loss_name, val in losses.items()}
    # sum all the scaled losses
    tot_loss = sum(losses_scaled.values())
    assert tot_loss.isfinite(), f"total_loss is not finite: {tot_loss}"
    return tot_loss, losses_scaled

# --------------------------------------------------------------------------------------------------------------------

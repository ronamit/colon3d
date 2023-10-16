import torch
import torch.nn.functional as nnF
from torch import nn

from colon_nav.util.rotations_util import quaternions_to_rot_matrices
from colon_nav.util.torch_util import to_numpy

# --------------------------------------------------------------------------------------------------------------------

class LossFunc(nn.Module):
    # --------------------------------------------------------------------------------------------------------------------
    def __init__(self, loss_terms_lambdas: dict, ref_frame_shifts: list[int]):
        super().__init__()
        self.loss_terms_lambdas = loss_terms_lambdas
        self.ref_frame_shifts = ref_frame_shifts
        self.ssim = SSIM() # the SSIM loss

    # --------------------------------------------------------------------------------------------------------------------
    def forward(self,
        batch: dict[str, torch.Tensor],
        outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the loss function.
        Args:
            loss_terms_lambdas: a dictionary of the loss terms and their weights
            depth_gt: the ground truth depth map of the target frame [B x 1 x H x W]
            list_tgt_to_refs_motion_gt: list of the ground truth egomotion from the target to each reference frame [B x 7]
        Returns:
            total_loss: the total loss
            losses: a dictionary of the losses
        """

        # Get the ground truth depth map of the target frame
        depth_gt = batch[("depth_gt", 0)]  # (B, H, W)

        # Get the ground truth egomotion from the target to each reference frame
        list_tgt_to_refs_motion_gt = [batch[("tgt_to_ref_motion", shift)] for shift in self.ref_frame_shifts]

        #  tgt_depth_est: the estimated depth map of the target frame [B x 1 x H x W]
        tgt_depth_est = outputs["tgt_depth_est"]
        #  list_tgt_to_refs_motion_est: list of the estimated egomotion from the target to each reference frame [B x 7]
        list_tgt_to_refs_motion_est = outputs["list_tgt_to_refs_motion_est"]
        losses = {}
        for loss_name, loss_lambda in self.loss_terms_lambdas.items():
            if loss_lambda > 0:
                losses[loss_name] = 0

        # Binary vector indicating which samples in the batch have a ground truth depth map:
        is_depth_gt = ~torch.isnan(depth_gt[:, 0, 0, 0])  # (B)
        n_pix_depth = depth_gt[0, 0].numel()  # number of pixels in the depth map

        if self.loss_terms_lambdas["depth_sup_L1"] > 0:
            # Compute the depth L1 loss (avg over all pixels and sum all samples in the batch)
            losses["depth_sup_L1"] = nnF.l1_loss(tgt_depth_est[is_depth_gt], depth_gt[is_depth_gt], reduction="sum") / n_pix_depth

        if self.loss_terms_lambdas["depth_sup_SSIM"] > 0:
            # Compute the depth SSIM loss avg over all pixels and sum all samples in the batch)
            losses["depth_sup_SSIM"] = self.ssim(tgt_depth_est[is_depth_gt], depth_gt[is_depth_gt]).sum() / n_pix_depth

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
                losses["trans_sup_L1_quat"] += nnF.l1_loss(trans_est, trans_gt, reduction="sum")

            if "rot_sup_L1_quat" in losses:
                # Add the L1 loss between the GT and estimated unit quaternion of this ref frame (sum over the samples in the batch)
                losses["rot_sup_L1_quat"] += nnF.l1_loss(rot_quat_est, rot_quat_gt, reduction="sum")

            if "rot_sup_L1_mat" in losses:
                # Compute the pose loss between two poses, defined as the L1 distance between the GT and estimated flattened 3x3 rotation matrix.
                rot_mat_est = quaternions_to_rot_matrices(rot_quat_est).view(-1, 9)  # (B, 9)
                rot_mat_gt = quaternions_to_rot_matrices(rot_quat_gt).view(-1, 9)  # (B, 9)
                # (sum the L1 loss over the samples in the batch)
                losses["rot_sup_L1_mat"] += nnF.l1_loss(rot_mat_est, rot_mat_gt, reduction="sum")

        # Scale the losses by their weights
        losses_scaled = {loss_name: val * self.loss_terms_lambdas[loss_name] for loss_name, val in losses.items()}
        # sum all the scaled losses
        tot_loss = sum(losses_scaled.values())

        loss_terms = {loss_name: to_numpy(val) for loss_name, val in losses.items()}
        assert tot_loss.isfinite(), f"total_loss is not finite: {tot_loss}"
        return tot_loss, loss_terms, outputs


# --------------------------------------------------------------------------------------------------------------------


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images.
    The SSIM loss is defined as 1 - SSIM, where SSIM is the structural similarity index between two images.
    Source: https://github.com/CapsuleEndoscope/EndoSLAM/blob/master/EndoSfMLearner/loss_functions.py
    """

    def __init__(self):
        super().__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


# --------------------------------------------------------------------------------------------------------------------

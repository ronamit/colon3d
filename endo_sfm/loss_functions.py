import torch
import torch.nn.functional as F
from torch import nn

from colon3d.util.torch_util import get_device
from endo_sfm.inverse_warp import inverse_warp2

device = get_device()

# --------------------------------------------------------------------------------------------------------------------

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""

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


compute_ssim_loss = SSIM().to(device)


def brightness_equator(source, target):
    def image_stats(image):
        # compute the mean and standard deviation of each channel

        L = image[:, 0, :, :]
        a = image[:, 1, :, :]
        b = image[:, 2, :, :]

        (lMean, lStd) = (torch.mean(torch.squeeze(L)), torch.std(torch.squeeze(L)))

        (aMean, aStd) = (torch.mean(torch.squeeze(a)), torch.std(torch.squeeze(a)))

        (bMean, bStd) = (torch.mean(torch.squeeze(b)), torch.std(torch.squeeze(b)))

        # return the color statistics
        return (lMean, lStd, aMean, aStd, bMean, bStd)

    def color_transfer(source, target):
        # convert the images from the RGB to L*ab* color space, being
        # sure to utilizing the floating point data type (note: OpenCV
        # expects floats to be 32-bit, so use that instead of 64-bit)

        # compute color statistics for the source and target images
        (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = image_stats(source)
        (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

        # subtract the means from the target image
        L = target[:, 0, :, :]
        a = target[:, 1, :, :]
        b = target[:, 2, :, :]

        L = L - lMeanTar
        # print("after l",torch.isnan(l))
        a = a - aMeanTar
        b = b - bMeanTar
        # scale by the standard deviations
        L = (lStdTar / lStdSrc) * L
        a = (aStdTar / aStdSrc) * a
        b = (bStdTar / bStdSrc) * b
        # add in the source mean
        L = L + lMeanSrc
        a = a + aMeanSrc
        b = b + bMeanSrc
        transfer = torch.cat((L.unsqueeze(1), a.unsqueeze(1), b.unsqueeze(1)), 1)
        # print(torch.isnan(transfer))
        return transfer

    # return the color transferred image
    transfered_image = color_transfer(target, source)
    return transfered_image


def compute_photo_and_geometry_loss(
    tgt_img,
    ref_imgs,
    intrinsics,
    tgt_depth,
    ref_depths,
    pred_poses,
    pred_poses_inv,
    max_scales,
    with_ssim,
    with_mask,
    with_auto_mask,
    padding_mode,
):
    """Compute photo and geometry loss between a pair of (consecutive) images.
    Args:
        tgt_img: target image
        ref_imgs: reference images
        intrinsics: camera intrinsics
        tgt_depth: target depth frames
        ref_depths: reference depth frames
        poses: 6DoF pose parameters from target to source -- [B, 6]

    """
    photo_loss = 0
    geometry_loss = 0

    num_scales = min(len(tgt_depth), max_scales)
    for ref_img, ref_depth, pose, pose_inv in zip(ref_imgs, ref_depths, pred_poses, pred_poses_inv, strict=True):
        for s in range(num_scales):
            # # downsample img
            # b, _, h, w = tgt_depth[s].size()
            # downscale = tgt_img.size(2)/h
            # if s == 0:
            #     tgt_img_scaled = tgt_img
            #     ref_img_scaled = ref_img
            # else:
            #     tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area')
            #     ref_img_scaled = F.interpolate(ref_img, (h, w), mode='area')
            # intrinsic_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)
            # tgt_depth_scaled = tgt_depth[s]
            # ref_depth_scaled = ref_depth[s]
            
            # TODO: Ron - why they didn't use the code above?

            # upsample depth
            b, _, h, w = tgt_img.size()
            tgt_img_scaled = tgt_img
            ref_img_scaled = ref_img
            intrinsic_scaled = intrinsics
            if s == 0:
                tgt_depth_scaled = tgt_depth[s]
                ref_depth_scaled = ref_depth[s]
            else:
                tgt_depth_scaled = F.interpolate(tgt_depth[s], (h, w), mode="nearest")
                ref_depth_scaled = F.interpolate(ref_depth[s], (h, w), mode="nearest")

            photo_loss1, geometry_loss1 = compute_pairwise_loss(
                tgt_img_scaled,
                ref_img_scaled,
                tgt_depth_scaled,
                ref_depth_scaled,
                pose,
                intrinsic_scaled,
                with_ssim,
                with_mask,
                with_auto_mask,
                padding_mode,
            )
            photo_loss2, geometry_loss2 = compute_pairwise_loss(
                ref_img_scaled,
                tgt_img_scaled,
                ref_depth_scaled,
                tgt_depth_scaled,
                pose_inv,
                intrinsic_scaled,
                with_ssim,
                with_mask,
                with_auto_mask,
                padding_mode,
            )

            photo_loss += photo_loss1 + photo_loss2
            geometry_loss += geometry_loss1 + geometry_loss2

    return photo_loss, geometry_loss


def compute_pairwise_loss(
    tgt_img,
    ref_img,
    tgt_depth,
    ref_depth,
    pose,
    intrinsic,
    with_ssim,
    with_mask,
    with_auto_mask,
    padding_mode,
):
    ref_img_warped, valid_mask, projected_depth, computed_depth = inverse_warp2(
        ref_img,
        tgt_depth,
        ref_depth,
        pose,
        intrinsic,
        padding_mode,
    )

    # torch.save(ref_img_warped, "ref_im_warped.pt")

    # print("ref_image_warped",ref_img_warped.shape)

    ref_img_warped2 = brightness_equator(ref_img_warped, tgt_img)

    # torch.save(ref_img_warped2, "ref_im_warped2.pt")

    # torch.save(tgt_img, "target_image.pt")

    diff_img = (tgt_img - ref_img_warped2).abs().clamp(0, 1)

    diff_depth = ((computed_depth - projected_depth).abs() / (computed_depth + projected_depth)).clamp(0, 1)

    if with_auto_mask is True:
        auto_mask = (
            diff_img.mean(dim=1, keepdim=True) < (tgt_img - ref_img).abs().mean(dim=1, keepdim=True)
        ).float() * valid_mask
        valid_mask = auto_mask

    if with_ssim is True:
        ssim_map = compute_ssim_loss(tgt_img, ref_img_warped2)
        diff_img = 0.15 * diff_img + 0.85 * ssim_map

    if with_mask is True:
        weight_mask = 1 - diff_depth
        diff_img = diff_img * weight_mask

    # compute all loss
    reconstruction_loss = mean_on_mask(diff_img, valid_mask)
    geometry_consistency_loss = mean_on_mask(diff_depth, valid_mask)

    return reconstruction_loss, geometry_consistency_loss


# compute mean value given a binary mask
def mean_on_mask(diff, valid_mask):
    mask = valid_mask.expand_as(diff)
    mean_value = (diff * mask).sum() / mask.sum() if mask.sum() > 10000 else torch.tensor(0).float().to(device)
    return mean_value


def compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs):
    def get_smooth_loss(disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness
        """

        # normalize
        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        disp = norm_disp

        grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
        grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    loss = get_smooth_loss(tgt_depth[0], tgt_img)

    for ref_depth, ref_img in zip(ref_depths, ref_imgs, strict=True):
        loss += get_smooth_loss(ref_depth[0], ref_img)

    return loss


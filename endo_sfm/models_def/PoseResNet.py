# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


import numpy as np
import torch
from torch import nn

from endo_sfm.models_def.resnet_encoder import ResnetEncoder

# ---------------------------------------------------------------------------------------------------------------------


class PoseDecoder(nn.Module):
    r"""
    Predict pose (6-DoF) change from the reference frame (i) to the target \ source frame (i+1)
    """

    def __init__(self, num_ch_enc: np.ndarray, num_frames_to_predict_for: int, stride=1):
        super().__init__()

        self.num_ch_enc = num_ch_enc
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = nn.ModuleDict()
        self.convs["squeeze"] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[f"pose_{0}"] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[f"pose_{1}"] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[f"pose_{2}"] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU(inplace=False)

    # ---------------------------------------------------------------------------------------------------------------------

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[f"pose_{i}"](out)
            if i != 2:
                out = self.relu(out)
        # Average the output features spatially
        out = out.mean(dim=-1).mean(dim=-1) * 0.01
        # Reshape from [B, 6 * num_frames_to_predict_for] to [B, num_frames_to_predict_for, 6]
        pose = out.view(-1, self.num_frames_to_predict_for, 6)

        return pose

    # ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------


class PoseResNet(nn.Module):

    """
    PoseNet model:
    Args:
        num_input_images: number of input images.
        num_frames_to_predict_for: number of frames to predict pose for
        num_layers: number of resnet layers (default: 18)
        pretrained: initialize with ImageNet pre-trained weights
    """

    def __init__(
        self,
        num_input_images: int,
        num_frames_to_predict_for: int,
        num_layers: int = 18,
        pretrained: bool = True,
    ):
        super().__init__()
        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained, num_input_images=num_input_images)
        self.decoder = PoseDecoder(self.encoder.num_ch_enc, num_frames_to_predict_for=num_frames_to_predict_for)

    # ---------------------------------------------------------------------------------------------------------------------

    def init_weights(self):
        pass

    # ---------------------------------------------------------------------------------------------------------------------

    def forward(self, target_img: torch.Tensor, ref_imgs: list[torch.Tensor]):
        """
        gets as input the target image (at index 0) and reference images (indices 1,2,..) and computes the pose change from the target to each reference frame.
        Args:
            target_img: the target image.(B,3,H,W)
            ref_imgs: the reference images (shifts w.r.t. target frame: -n_ref_imgs, .., -1) List of (B,3,H,W) tensors
        Returns:
            pose: the pose change from the target to each reference frame (B,6)
        """
        # concatenate the target and reference images along the channel dimension
        x = torch.cat([*ref_imgs, target_img], dim=1)  # (B, n_imgs*3, H, W)
        features = self.encoder(x)
        pose = self.decoder([features])
        return pose

    # ---------------------------------------------------------------------------------------------------------------------
    def get_weight_dtype(self):
        return self.encoder.get_weight_dtype()


# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    model = PoseResNet().cuda()
    model.train()

    tgt_img = torch.randn(4, 3, 320, 320).cuda()
    ref_imgs = [torch.randn(4, 3, 320, 320).cuda() for i in range(2)]

    pose = model(tgt_img, ref_imgs[0])

    print(pose.size())

# ---------------------------------------------------------------------------------------------------------------------

from collections import OrderedDict

import torch
from torch import nn

from monodepth2.networks.resnet_encoder import ResnetEncoder

# ----------------------------------------------------------------------------------------------------------------------

class PoseNet(nn.Module):
    def __init__(self, n_ref_imgs: int, num_layers: int=18):
        """
        PoseNet model: Gets as input list of reference images and target image, and outputs the relative pose from
            the target image and each reference image.
        Args:
            n_ref_imgs: number of reference images
            num_layers: number of resnet layers (default: 18)
        """
        super().__init__()
        self.n_ref_imgs = n_ref_imgs
        self.num_layers = num_layers
        self.encoder = ResnetEncoder(
            num_layers=self.num_layers,
            pretrained=True, # initialize with ImageNet pre-trained weights
            num_input_images=self.n_ref_imgs + 1,
        )
        self.decoder = PoseDecoder(num_ch_enc= self.encoder.num_ch_enc,
            num_frames_to_predict_for=self.n_ref_imgs,
        )

    # ----------------------------------------------------------------------------------------------------------------------

    def forward(self, ref_imgs: list[torch.Tensor], tgt_img: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            ref_imgs: list of reference images
            tgt_img: target image
        Returns:
            list of relative poses from the target image to each reference image, in axis-angle format (B,6)
        """
        input_imgs = [*ref_imgs, tgt_img]
        # concatenate the reference images and the target image along the channel dimension:
        net_input = torch.cat(input_imgs, dim=1)
        input_features = self.encoder(net_input)
        axisangle, translation = self.decoder(input_features)
        return axisangle, translation

# ----------------------------------------------------------------------------------------------------------------------
class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_frames_to_predict_for, stride=1):
        super().__init__()

        self.num_ch_enc = num_ch_enc
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        """
        Args:
            input_features: list of feature maps from the encoder
        Returns:
            Pose estimations: list of relative poses from the target image to each reference image, in the format (B,7)
                where the first 3 elements are x,y,z and the last 4 elements are the quaternion elements (qw, qx, qy, qz)
        """
        n_scales = len(input_features)
    


        # last_features = [f[-1] for f in input_features]

        # cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        # cat_features = torch.cat(cat_features, 1)

        # out = cat_features
        # for i in range(3):
        #     out = self.convs[("pose", i)](out)
        #     if i != 2:
        #         out = self.relu(out)

        # out = out.mean(3).mean(2)

        # out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        # axisangle = out[..., :3]
        # translation = out[..., 3:]

        # return axisangle, translation

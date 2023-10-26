from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

# --------------------------------------------------------------------------------------------------------------------


class DispResNet(nn.Module):

    """
    The disparity model of EndoSFMLearner
    https://github.com/CapsuleEndoscope/EndoSLAM/blob/master/EndoSfMLearner/models/DispResNet.py
    """

    def __init__(self, pretrained=True):
        super().__init__()
        num_layers = 18
        self.encoder = ResnetEncoder(num_layers=num_layers, pretrained=pretrained)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc)

    # --------------------------------------------------------------------------------------------------------------------

    def init_weights(self):
        pass

    # --------------------------------------------------------------------------------------------------------------------

    def forward(self, x):
        features = self.encoder(x)
        outputs = self.decoder(features)

        if self.training:
            # return outputs for all scales
            return outputs

        return outputs[0]  # in inference mode, we only predict the disparity of the full resolution output

    # --------------------------------------------------------------------------------------------------------------------
    def get_weight_dtype(self):
        return self.encoder.get_weight_dtype()


# ----------------------------------------------------------------------------------------


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder.
        Source: https://github.com/CapsuleEndoscope/EndoSLAM/blob/master/EndoSfMLearner/models/resnet_encoder2.py
    """

    def __init__(self, num_layers, pretrained):
        super().__init__()
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }
        if num_layers not in resnets:
            raise ValueError(f"{num_layers} is not a valid number of resnet layers")

        self.encoder = resnets[num_layers](weights="IMAGENET1K_V1" if pretrained else None)
        # note: this line was changed because of the warning:  UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=None`.

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        # x = (input_image - 0.45) / 0.225 # this was already done in the dataloader
        x = input_image
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))
        return self.features

    def get_weight_dtype(self):
        return self.encoder.conv1.weight.dtype


# ----------------------------------------------------------------------------------------


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=(0, 1, 2, 3), num_output_channels=1, use_skips=True):
        super().__init__()

        self.alpha = 10
        self.beta = 0.01

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = "nearest"
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = []

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs.append(self.alpha * self.sigmoid(self.convs[("dispconv", i)](x)) + self.beta)

        self.outputs = self.outputs[::-1]
        return self.outputs


# --------------------------------------------------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=False)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


# ----------------------------------------------------------------------------------------


class Conv3x3(nn.Module):
    """Layer to pad and convolve input"""

    def __init__(self, in_channels, out_channels, use_refl=True):
        super().__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


# ----------------------------------------------------------------------------------------


def upsample(x):
    """Upsample input tensor by a factor of 2"""
    return F.interpolate(x, scale_factor=2, mode="nearest")


# ----------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    model = DispResNet().cuda()
    model.train()

    B = 12

    tgt_img = torch.randn(B, 3, 256, 832).cuda()
    ref_imgs = [torch.randn(B, 3, 256, 832).cuda() for i in range(2)]

    tgt_depth = model(tgt_img)

    print(tgt_depth[0].size())
# --------------------------------------------------------------------------------------------------------------------

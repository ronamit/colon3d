# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 license
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


import numpy as np
import torch
from torch import nn
from torchvision import models

# --------------------------------------------------------------------------------------------------------------------

class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    # --------------------------------------------------------------------------------------------------------------------

    def __init__(self, block, layers, num_input_images=1):
        super().__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

# --------------------------------------------------------------------------------------------------------------------

def resnet_multi_image_input(num_layers: int, pretrained: bool = False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"

    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)
    if pretrained:
        # Load pre-trained weights for encoder, and modify for different number of input channels
        model_name = f"resnet{num_layers}"
        dummy_model = models.get_model(model_name, weights="DEFAULT")
        loaded = dummy_model.state_dict()
        loaded["conv1.weight"] = torch.cat([loaded["conv1.weight"]] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model

# --------------------------------------------------------------------------------------------------------------------

class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder"""
    # --------------------------------------------------------------------------------------------------------------------

    def __init__(self, num_layers, pretrained, num_input_images=1):
        super().__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        model_name = f"resnet{num_layers}"
        # see https://pytorch.org/vision/main/models/resnet.html

        if num_input_images > 1:
            # modify first layer to accept multiple images (by adding channels to the convolutional layer)
            self.encoder = resnet_multi_image_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = models.get_model(model_name, weights="DEFAULT" if pretrained else None)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4
    # --------------------------------------------------------------------------------------------------------------------

    def forward(self, input_image):
        self.features = []
        x = input_image

        x = self.encoder.conv1(x)

        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))

        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
# --------------------------------------------------------------------------------------------------------------------

    def get_weight_dtype(self):
        return self.encoder.conv1.weight.dtype
    
# --------------------------------------------------------------------------------------------------------------------

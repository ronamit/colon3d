"""
Based on:  https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet18_Weights, ResNet50_Weights, conv1x1
from torchvision.utils import _log_api_usage_once

from colon_nav.nets.training_utils import ModelInfo

# -------------------------------------------------------------------------------------------------------------------


class EgomotionResNet(nn.Module):
    def __init__(
        self,
        n_input_channels: int,
        block: type[BasicBlock | Bottleneck],
        layers: list[int],
        output_dim: int,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: list[bool] | None = None,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.n_input_channels = n_input_channels
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}",
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(self.n_input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d | nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: type[BasicBlock | Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            ),
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                ),
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # we got a 7-dimensional vector: 3 for translation, 4 for rotation
        # We normalize the rotation vector to be a unit quaternion (l2-norm = 1)
        eps = 1e-12
        x[:, 3:] = x[:, 3:] / (torch.norm(x[:, 3:], dim=1, keepdim=True) + eps)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# -------------------------------------------------------------------------------------------------------------------


def get_resnet_egomotion_model(model_info: ModelInfo, pretrained=True) -> EgomotionResNet:
    """ Get the ResNet egomotion model.
    Args:
        model_info: The model info object.
        pretrained: If True, load the ImageNet pretrained weights.
    Returns:
        The ResNet egomotion model.
    """

    model_name = model_info.egomotion_model_name
    ref_frame_shifts = model_info.ref_frame_shifts
    n_ref_imgs = len(ref_frame_shifts)
    n_input_imgs = n_ref_imgs + 1  #  reference frames & target frame
    n_input_channels = 3 * n_input_imgs  # The RGB images are concatenated along the channel dimension

    # The output of the network is a 7-dimensional vector: 3 for translation, 4 for rotation
    # The rotation is represented by a unit quaternion (l2-norm = 1)
    output_dim = 7

    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT
        model = EgomotionResNet(n_input_channels=n_input_channels, output_dim=output_dim, block=BasicBlock, layers=[2, 2, 2, 2])
    elif model_name == "resnet50":
        weights = ResNet50_Weights.DEFAULT
        model = EgomotionResNet(n_input_channels=n_input_channels, output_dim=output_dim, block=Bottleneck, layers=[3, 4, 6, 3])
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if pretrained:
    # Load ImageNet pretrained weights
        weights_dict = weights.get_state_dict(progress=True)
        conv1_weights = weights_dict["conv1.weight"]  # [out_channels, in_channels, kernel_size, kernel_size]

        # We concatenate the RGB images along the channel dimension, so we need to duplicate the weights
        #  along the input_channel dimension for n_input_imgs times:
        conv1_weights = conv1_weights.repeat(1, n_input_imgs, 1, 1)

        # Since we increased input_channel times n_input_imgs, we need to adjust the scale of the weights
        weights_dict["conv1.weight"] = conv1_weights / n_input_imgs

        # Load the weights to the model (note that the output layer is not loaded)
        # exclude the output later, since we have a different output dimension
        weights_dict = {k: v for k, v in weights_dict.items() if k not in {"fc.weight", "fc.bias"}}
        model.load_state_dict(weights_dict, strict=False)
    return model


# -------------------------------------------------------------------------------------------------------------------

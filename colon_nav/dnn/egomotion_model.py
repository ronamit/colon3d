from pathlib import Path

import torch
from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet18_Weights, ResNet50_Weights

from colon_nav.dnn.models_def.resnet import ResNet
from colon_nav.dnn.train_utils import ModelInfo

# -------------------------------------------------------------------------------------------------------------------


class EgomotionModel(nn.Module):
    """ResNet egomotion model.
    Args:
        model_info: The model info object.
        pretrained: If True, load the ImageNet pretrained weights.
        load_egomotion_model_path: If not None, load the weights from this path.
    """

    def __init__(self, model_info: ModelInfo, imagent_pretrained=True, load_egomotion_model_path: Path | None = None):
        super().__init__()
        self.model_info = model_info
        self.imagent_pretrained = imagent_pretrained
        model_name = model_info.egomotion_model_name
        ref_frame_shifts = model_info.ref_frame_shifts
        self.n_ref_imgs = len(ref_frame_shifts)
        self.n_input_imgs = self.n_ref_imgs + 1  #  reference frames & target frame
        n_input_channels = 3 * self.n_input_imgs  # The RGB images are concatenated along the channel dimension

        # The output of the network is a vector of length 7*n_ref_imgs : n_ref_imgs blocks of length 7.
        # each block represents the ego-motion from the target to the reference image :
        # 3 translation parameters (x,y,z) [mm]
        # 4 rotation parameters (qw, qx, qy, qz) [unit quaternion]
        output_dim = 7 * self.n_ref_imgs

        if model_name == "resnet18":
            self.model = ResNet(
                n_input_channels=n_input_channels,
                output_dim=output_dim,
                block=BasicBlock,
                layers=[2, 2, 2, 2],
            )
            weights = ResNet18_Weights.DEFAULT
        elif model_name == "resnet50":
            self.model = ResNet(
                n_input_channels=n_input_channels,
                output_dim=output_dim,
                block=Bottleneck,
                layers=[3, 4, 6, 3],
            )
            weights = ResNet50_Weights.DEFAULT
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Load ImageNet pretrained weights
        if imagent_pretrained and load_egomotion_model_path is None:
            weights_dict = weights.get_state_dict(progress=True)
            conv1_weights = weights_dict["conv1.weight"]  # [out_channels, in_channels, kernel_size, kernel_size]

            # We concatenate the RGB images along the channel dimension, so we need to duplicate the weights
            #  along the input_channel dimension for n_input_imgs times:
            conv1_weights = conv1_weights.repeat(1, self.n_input_imgs, 1, 1)

            # Since we increased input_channel times n_input_imgs, we need to adjust the scale of the weights
            weights_dict["conv1.weight"] = conv1_weights / self.n_input_imgs

            # Load the weights to the model (note that the output layer is not loaded)
            # exclude the output later, since we have a different output dimension
            weights_dict = {k: v for k, v in weights_dict.items() if k not in {"fc.weight", "fc.bias"}}
            self.model.load_state_dict(weights_dict, strict=False)

        # Load pretrained egomotion model from a checkpoint
        if load_egomotion_model_path is not None:
            self.model.load_state_dict(torch.load(load_egomotion_model_path))

    # -------------------------------------------------------------------------------------------------------------------

    def forward(self, frames: list[Tensor]) -> Tensor:
        """Forward pass of the network.
        Args:
            frames: list of [B, 3, H, W] tensors, where B is the batch size, H and W are the image height and width.
                The first elements in the list are the RGB images of the reference frames, and the last element is the RGB image of the target frame.
        Returns:
            List of the estimated ego-motion from the target to each reference frame. (list of [B, 7] tensors)
            The ego-motion format: 3 translation parameters (x,y,z) [mm], 4 rotation parameters (qw, qx, qy, qz) [unit quaternion]
        """
        # concatenate the RGB images along the channel dimension
        x = torch.cat(frames, dim=1)  # [B, 3*(1+n_ref_imgs), H, W]
        # forward pass
        net_out = self.model(x)  # [B, 7*n_ref_imgs]
        # The output of the network is a vector of length 7*n_ref_imgs : n_ref_imgs blocks of length 7.
        # each block represents the ego-motion from the target to the reference image :
        tgt_to_refs_motion_est = []
        # Go over all the reference frames
        for i_ref in range(self.n_ref_imgs):
            est_motion = net_out[:, 7 * i_ref : 7 * (i_ref + 1)]  # [B, 7]
            # the first 3  parameters are translation (x,y,z) [mm]
            # the last are 4 rotation parameters (qw, qx, qy, qz) [unit quaternion] are normalized to be l2-norm = 1
            eps = 1e-12
            est_motion = est_motion / (torch.norm(est_motion[:, 3:], dim=-1, keepdim=True) + eps)
            tgt_to_refs_motion_est.append(est_motion)
        return tgt_to_refs_motion_est


# -------------------------------------------------------------------------------------------------------------------

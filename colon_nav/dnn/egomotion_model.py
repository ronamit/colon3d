from collections.abc import Sequence
from pathlib import Path

import torch
import torchvision
from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet18_Weights, ResNet50_Weights

from colon_nav.dnn.model_info import ModelInfo
from colon_nav.dnn.models_def.resnet import ResNet
from colon_nav.util.torch_util import get_device

# -------------------------------------------------------------------------------------------------------------------


class EgomotionModel(nn.Module):
    """Wrapper for the egomotion model.
    Args:
        - model_info: ModelInfo object
        - load_model_path: Path to a pretrained model to load. If None, then the model is initialized using ImageNet pre-trained weights.
        - device: The device to use for the model
        - mode: "train" or "eval"
    Notes:
        - The input to the forward function is a list of RGB images (B, C, H, W) as tensors, the list corresponds to the reference frames and the target frame (last element).
        - The input images are resized to fit as network input.
    """

    def __init__(self, model_info: ModelInfo, load_model_path: Path | None, device: torch.device, is_train: bool):
        super().__init__()
        self.is_train = is_train
        self.device = device or get_device()
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
            # see https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18
            self.egomotion_model = ResNet(
                n_input_channels=n_input_channels,
                output_dim=output_dim,
                block=BasicBlock,
                layers=[2, 2, 2, 2],
            )
            weights = ResNet18_Weights.DEFAULT
            self.in_resolution = 320  # 224
            # RGB values are first rescaled to [0.0, 1.0] and then normalized using:
            chan_nrm_mean = [0.485, 0.456, 0.406]
            chan_nrm_std = [0.229, 0.224, 0.225]

        elif model_name == "resnet50":
            # see https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
            self.egomotion_model = ResNet(
                n_input_channels=n_input_channels,
                output_dim=output_dim,
                block=Bottleneck,
                layers=[3, 4, 6, 3],
            )
            weights = ResNet50_Weights.DEFAULT
            self.in_resolution = 320  # 224
            # RGB values are first rescaled to [0.0, 1.0] and then normalized using:
            chan_nrm_mean = [0.485, 0.456, 0.406]
            chan_nrm_std = [0.229, 0.224, 0.225]
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # Load ImageNet pretrained weights, if no model_path is given
        if load_model_path is None:
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
            self.egomotion_model.load_state_dict(weights_dict, strict=False)

        else:
            # Load pretrained egomotion model from a checkpoint
            self.egomotion_model.load_state_dict(torch.load(load_model_path / "egomotion_model.pth"))

        # We resize the input images to fit as network input
        self.input_resizer = torchvision.transforms.Resize(
            size=(self.in_resolution, self.in_resolution),
            antialias=True,
        )

        self.channel_normalizer = torchvision.transforms.Normalize(
            mean=chan_nrm_mean,
            std=chan_nrm_std,
            inplace=False,
        )

        # Move to device
        self.to(self.device)

        print("Egomotion model weights type:", self.egomotion_model.named_parameters().__next__()[1].dtype)

        # Set the model to train or eval mode
        if self.is_train:
            self.train()
        else:
            self.eval()

        # The depth_calib_a parameter is used to scale the translation parameters (x,y,z) to be mm distance units
        self.depth_calib_type = model_info.depth_calib_type
        self.depth_calib_a = model_info.depth_calib_a

    # -------------------------------------------------------------------------------------------------------------------

    def forward(self, frames: Sequence[Tensor]) -> Tensor:
        """Forward pass of the network.
        Args:
            frames: Sequence of [B, 3, H, W] tensors, where B is the batch size, H and W are the image height and width. The image values are in the range [0,255].
                The first elements in the list are the RGB images of the reference frames, and the last element is the RGB image of the target frame.
        Returns:
            List of the estimated ego-motion from the target to each reference frame. (list of [B, 7] tensors)
            The ego-motion format: 3 translation parameters (x,y,z) [mm], 4 rotation parameters (qw, qx, qy, qz) [unit-quaternion]
        """
        # Resize each image in the frames list to fit as network input
        frames = [self.input_resizer(frame) for frame in frames]

        # Normalize the image values from [0,255] to [0,1]
        frames = [frame / 255.0 for frame in frames]

        # Apply channels normalization
        frames = [self.channel_normalizer(frame) for frame in frames]

        # concatenate the RGB images along the channel dimension
        x = torch.cat(frames, dim=1)  # [B, 3*(1+n_ref_imgs), H, W]

        # Apply the egomotion model:
        if self.is_train:
            net_out = self.egomotion_model(x)
        else:
            with torch.no_grad():
                net_out = self.egomotion_model(x)

        net_out = self.egomotion_model(x)  # [B, 7*n_ref_imgs]

        # The output of the network is a vector of length 7*n_ref_imgs : n_ref_imgs blocks of length 7.
        # each block represents the ego-motion from the target to the reference image :
        tgt_to_refs_motion_est = []
        # Go over all the reference frames
        for i_ref in range(self.n_ref_imgs):
            cur_block_start = 7 * i_ref
            est_trans = net_out[:, cur_block_start : (cur_block_start + 3)]  # [B, 3] translation (x,y,z) [mm]
            est_rot = net_out[
                :,
                (cur_block_start + 3) : (cur_block_start + 7),
            ]  # [B, 4] rotation (qw,qx,qy,qz) unit quaternion
            # Normalize the rotation quaternion:
            eps = 1e-12
            est_rot = est_rot / (torch.norm(est_rot, dim=-1, keepdim=True) + eps)

            # Apply distance units calibration for the translation estimate
            if self.depth_calib_type == "linear":
                est_trans = est_trans * self.depth_calib_a
            elif self.depth_calib_type == "none":
                pass
            else:
                raise ValueError(f"Unknown depth calibration type: {self.depth_calib_type}")

            est_motion = torch.cat((est_trans, est_rot), dim=-1)  # [B, 7]
            tgt_to_refs_motion_est.append(est_motion)

        return tgt_to_refs_motion_est

    # -------------------------------------------------------------------------------------------------------------------

    def save_model(self, save_model_path: Path):
        """Save the model to a checkpoint.
        Args:
            model_path: The path to save the model to.
        """
        torch.save(self.egomotion_model.state_dict(), save_model_path / "egomotion_model.pth")


# -------------------------------------------------------------------------------------------------------------------

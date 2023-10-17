from pathlib import Path

import torch
import torchvision
from torch import nn

from colon_nav.dnn.model_info import ModelInfo
from colon_nav.dnn.models_def.dense_depth import DenseDepth
from colon_nav.dnn.models_def.fcb_former import FCBFormer
from colon_nav.util.torch_util import get_device

# ---------------------------------------------------------------------------------------------------------------------


class DepthModel(nn.Module):
    """Wrapper for the depth model.
    Args:
        - load_model_path: Path to a pretrained model to load. If None, then the model is initialized using ImageNet pre-trained weights.
        - device: The device to use for the model
        - mode: "train" or "eval"
        - depth_map_size: The size of the output depth map (H, W)
    Notes:
        - The input to the forward function is a batch of RGB images (B, C, H, W)
        - The input images are resized to fit as network input.
        - The network output depth maps are resized to the desired size as specified in the model_info.
        - The output depth maps are calibrated and clipped according to the model_info.
    """

    def __init__(self, model_info: ModelInfo, load_model_path: Path | None, device: torch.device, is_train: bool, depth_map_size: tuple[int, int]):
        super().__init__()
        self.is_train = is_train
        self.device = device or get_device()
        self.depth_map_size = depth_map_size
        self.model_name = model_info.depth_model_name
        # Create the depth model
        if self.model_name == "DenseDepth":
            # The DenseDepth uses DenseNet-169 as encoder (pre-trained on ImageNet)
            # See https://pytorch.org/vision/main/models/generated/torchvision.models.densenet169.html
            # RGB values are first rescaled to [0.0, 1.0] and then normalized using:
            chan_nrm_mean = [0.485, 0.456, 0.406]
            chan_nrm_std = [0.229, 0.224, 0.225]
            self.in_resolution = 320  # 475
            self.depth_model = DenseDepth()

        elif self.model_name == "FCBFormer":
            self.in_resolution = 320  # 352
            self.depth_model = FCBFormer(in_resolution=self.in_resolution)
            # RGB values are first scaled to [0,1] and then to [-1, 1] with (x - 0.5) / 0.5
            chan_nrm_mean = [0.5, 0.5, 0.5]
            chan_nrm_std = [0.5, 0.5, 0.5]

        else:
            raise ValueError(f"Unknown depth model name: {self.model_name}")

        ### Load pretrained weights
        if load_model_path is not None:
            loaded_state = torch.load(load_model_path / "depth_model.pth")
            self.depth_model.load_state_dict(loaded_state)

        # We resize the input images to fit as network input
        self.input_resizer = torchvision.transforms.Resize(
            size=(self.in_resolution, self.in_resolution),
            antialias=True,
        )
        # We resize the output depth map to the desired size
        self.output_resizer = torchvision.transforms.Resize(
            size=self.depth_map_size,
            antialias=True,
        )
        self.channel_normalizer = torchvision.transforms.Normalize(
            mean=chan_nrm_mean,
            std=chan_nrm_std,
            inplace=False,
        )

        # Move to device
        self.to(self.device)

        # Set the model to train or eval mode
        if self.is_train:
            self.train()
        else:
            self.eval()

        print("Depth model weights type:", self.depth_model.named_parameters().__next__()[1].dtype)

        # The depth calibration parameters
        self.depth_calib_type = model_info.depth_calib_type
        self.depth_calib_a = model_info.depth_calib_a
        self.depth_calib_b = model_info.depth_calib_b
        self.depth_lower_bound = model_info.depth_lower_bound
        self.depth_upper_bound = model_info.depth_upper_bound

    # ---------------------------------------------------------------------------------------------------------------------

    def forward(self, x):
        """Forward pass of the depth model.
        Arg:
            x: a batch of RGB images (B, C, H, W) (values in [0,255])
        Returns:
            out: the output depth map (B, 1, H, W) (units: mm)
        """
            
        # Apply input image resizing
        x = self.input_resizer(x)

        # Normalize the image values from [0,255] to [0,1]
        x = x / 255.0

        # Apply channels normalization
        x = self.channel_normalizer(x)

        # Apply the depth model
        if self.is_train:
            out = self.depth_model(x)
        else:
            with torch.no_grad():
                out = self.depth_model(x)

        # Resize the output depth map to the desired size
        out = self.output_resizer(out)

        # Apply depth calibration
        if self.depth_calib_type == "affine":
            out = self.depth_calib_a * out + self.depth_calib_b
        elif self.depth_calib_type == "none":
            pass
        else:
            raise ValueError(f"Unknown depth calibration type: {self.depth_calib_type}")

        # Apply depth clippingx
        if self.depth_lower_bound is not None:
            out = torch.clamp(out, min=self.depth_lower_bound)
        if self.depth_upper_bound is not None:
            out = torch.clamp(out, max=self.depth_upper_bound)
        return out

    # ---------------------------------------------------------------------------------------------------------------------

    def save_model(self, save_model_path: Path):
        save_path = save_model_path / "depth_model.pth"
        torch.save(self.depth_model.state_dict(), save_path)
        print(f"Saved depth model to {save_path}")


# ---------------------------------------------------------------------------------------------------------------------

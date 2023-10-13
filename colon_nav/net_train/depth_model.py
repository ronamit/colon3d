from pathlib import Path

import torch
import torchvision
from torch import nn

from colon_nav.net_def.dense_depth import DenseDepth
from colon_nav.net_def.fcb_former import FCBFormer
from colon_nav.net_train.train_utils import ModelInfo

# ---------------------------------------------------------------------------------------------------------------------


class DepthModel(nn.Module):
    """The depth model: wraps the FCBFormer model to resize the input and output to the desired size.
    Args:
        out_size: The desired output size of the depth map (height, width)
    """

    def __init__(self, model_info: ModelInfo, load_depth_model_path: Path | None = None):
        super().__init__()
        self.out_size = model_info.depth_map_size
        self.model_name = model_info.depth_model_name
        # Create the depth model
        if self.model_name == "DenseDepth":
            self.in_resolution = 475 # as in the DenseDepth's MIVA team adaptation in the SimCol3D challenge.
            self.model = DenseDepth()

        elif self.model_name == "FCBFormer":
            # the FCBFormer was pre-trained on 352x352 images
            self.in_resolution = 352
            self.model = FCBFormer(in_resolution=self.in_resolution)

        else:
            raise ValueError(f"Unknown depth model name: {self.model_name}")

        ### Load pretrained weights
        if load_depth_model_path is not None:
            self.depth_model.load_state_dict(torch.load(load_depth_model_path))

        # We resize the input to the desired size
        self.input_resizer = torchvision.transforms.Resize(
            size=(self.in_resolution, self.in_resolution),
            antialias=True,
        )
        # We resize the output to the desired size
        self.output_resizer = torchvision.transforms.Resize(
            size=self.out_size,
            antialias=True,
        )

    # ---------------------------------------------------------------------------------------------------------------------

    def forward(self, x):
        x = self.input_resizer(x)
        out = self.model(x)
        out = self.output_resizer(out)
        return out


# ---------------------------------------------------------------------------------------------------------------------

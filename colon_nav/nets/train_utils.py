from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from colon_nav.util.general_util import save_dict_to_yaml

# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class DatasetMeta:
    feed_height: int  # The height of the input images to the network
    feed_width: int  # The width of the input images to the network
    ref_frame_shifts: list[int]  # The time shifts of the reference frames w.r.t. the target frame
    load_gt_depth: bool  # Whether to add the depth map of the target frame to each sample (default: False)
    load_gt_pose: bool  # Whether to add the ground-truth pose change between from target to the reference frames,  each sample (default: False)


# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class ModelInfo:
    depth_model_name: str
    egomotion_model_name: str
    ref_frame_shifts: list[int]  # The time shifts of the reference frames w.r.t. the target frame
    feed_height: int
    feed_width: int
    depth_calib_type: str = "none"
    depth_calib_a: float = 1.0
    depth_calib_b: float = 0.0
    model_description: str = ""


# ---------------------------------------------------------------------------------------------------------------------


def save_model_info(
    save_dir_path: Path,
    model_info: ModelInfo,
    overwrite: bool = True,
):
    model_info_path = save_dir_path / "model_info.yaml"
    if model_info_path.exists() and not overwrite:
        print(f"Model info file {model_info_path} already exists, overwriting")
        model_info_path.unlink()

    model_info_dict = model_info.__dict__

    save_dict_to_yaml(save_path=model_info_path, dict_to_save=model_info_dict)


# ---------------------------------------------------------------------------------------------------------------------


def load_model_model_info(path: Path) -> ModelInfo:
    model_info_path = path / "model_info.yaml"
    if not model_info_path.exists():
        raise FileNotFoundError(f"Model info file {model_info_path} does not exist")
    with model_info_path.open("r") as f:
        model_info = yaml.load(f, Loader=yaml.FullLoader)
    return ModelInfo(**model_info)


# ---------------------------------------------------------------------------------------------------------------------


def get_depth_model_input(batch: dict) -> dict:
    # Get the target image:
    tgt_img = batch[("color", 0)]
    return tgt_img


# ---------------------------------------------------------------------------------------------------------------------


def get_egomotion_model_input(batch: dict, ref_frame_shifts: list[int]) -> dict:
    # Get the target image:
    tgt_img = batch[("color", 0)]

    # Get the reference images:
    ref_imgs = [batch[("color", i)] for i in ref_frame_shifts]

    # Concatenate the reference and target images along the channel dimension:
    imgs = [*ref_imgs, tgt_img]
    imgs = torch.cat(imgs, dim=1)
    return imgs


# ---------------------------------------------------------------------------------------------------------------------


class TensorBoardLogger:
    def __init__(
        self,
        log_dir: Path,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_info: ModelInfo,
        depth_model: torch.nn.Module,
        egomotion_model: torch.nn.Module,
    ) -> None:
        self.writer = SummaryWriter(log_dir=log_dir)
        self.train_loader = train_loader
        self.val_loader = val_loader
        assert len(self.train_loader) > 0, "Training data loader is empty"
        assert len(self.val_loader) > 0, "Validation data loader is empty"
        self.model_info = model_info
        self.depth_model = depth_model
        self.egomotion_model = egomotion_model
        self.visualize_graph()

    def log_train_loss(self, epoch: int, loss: float, lr: float) -> None:
        self.writer.add_scalar("train/loss", loss, epoch)
        self.writer.add_scalar("train/lr", lr, epoch)

    def visualize_graph(self) -> None:
        sample = next(iter(self.val_loader))
        depth_model_input = get_depth_model_input(sample)
        ego_model_input = get_egomotion_model_input(sample, self.model_info.ref_frame_shifts)
        self.writer.add_graph(self.depth_model, depth_model_input)
        self.writer.add_graph(self.egomotion_model, ego_model_input)
        self.writer.close()

    def visualize_output(self, epoch: int) -> None:
        # Sample a batch of data from the validation set
        sample = next(iter(self.train_loader))
        images = sample["color"]
        img_grid = torchvision.utils.make_grid(images)
        self.writer.add_image("val/input", img_grid, global_step=epoch)


# ---------------------------------------------------------------------------------------------------------------------

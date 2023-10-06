from pathlib import Path

import attrs
import torch
import torchvision
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from colon_nav.util.general_util import save_dict_to_yaml, to_str

# ---------------------------------------------------------------------------------------------------------------------


@attrs.define
class ModelInfo:
    depth_model_name: str
    egomotion_model_name: str
    ref_frame_shifts: list[int]  # The time shifts of the reference frames w.r.t. the target frame
    img_normalize_mean: float = 0.45  # used in "normalize_image_channels"
    img_normalize_std: float = 0.225  #  used in "normalize_image_channels"
    depth_calib_type: str = "none"
    depth_calib_a: float = 1.0
    depth_calib_b: float = 0.0
    model_description: str = ""
    # Fields that will be initialized in post-init:
    depth_model_feed_height: int = attrs.field(init=False)
    depth_model_feed_width: int = attrs.field(init=False)
    ego_model_feed_height: int = attrs.field(init=False)
    ego_model_feed_width: int = attrs.field(init=False)

    def __attrs_post_init__(self):
        if self.depth_model_name == "fcb_former":
            self.depth_model_feed_height = 352
            self.depth_model_feed_width = 352
        else:
            raise ValueError(f"Unknown depth model name: {self.depth_model_name}")
        if self.egomotion_model_name in ["resnet18", "resnet50"]:
            self.ego_model_feed_height = 224
            self.ego_model_feed_width = 224
        else:
            raise ValueError(f"Unknown egomotion model name: {self.egomotion_model_name}")


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

    model_info_dict = attrs.asdict(model_info)

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


class TensorBoardWriter:
    def __init__(
        self,
        log_dir: Path,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_info: ModelInfo,
        depth_model: torch.nn.Module,
        egomotion_model: torch.nn.Module,
        show_graph: bool = False,
    ) -> None:
        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer.add_text("Model info", str(model_info), global_step=0)
        self.train_loader = train_loader
        self.val_loader = val_loader
        assert len(self.train_loader) > 0, "Training data loader is empty"
        assert len(self.val_loader) > 0, "Validation data loader is empty"
        print(f"Saving tensorboard logs to {log_dir}")
        self.model_info = model_info
        self.depth_model = depth_model
        self.egomotion_model = egomotion_model
        self.show_data()
        if show_graph:
            self.visualize_graph()

    def log_train_loss(self, epoch: int, loss: float, lr: float) -> None:
        self.writer.add_scalar("train/loss", loss, epoch)
        self.writer.add_scalar("train/lr", lr, epoch)

    def visualize_graph(self) -> None:
        sample = next(iter(self.val_loader))
        depth_model_input = sample["depth_model_in"]
        ego_model_input = sample["ego_model_in"]
        self.writer.add_graph(self.depth_model, depth_model_input)
        self.writer.add_graph(self.egomotion_model, ego_model_input)
        self.writer.close()

    def show_data(self) -> None:
        sample = next(iter(self.train_loader))
        ind = 0  # index of the sample in the batch
        img_normalize_mean = self.model_info.img_normalize_mean
        img_normalize_std = self.model_info.img_normalize_std
        # Take the RGB images from the first example in the batch
        all_shifts = [*self.model_info.ref_frame_shifts, 0]
        n_ref_frames = len(self.model_info.ref_frame_shifts)
        input_images = [sample[("color", shift)][ind] for shift in all_shifts]
        # Un-normalize the images:
        input_images = [img * img_normalize_std + img_normalize_mean for img in input_images]
        img_grid = torchvision.utils.make_grid(input_images)
        self.writer.add_image(f"RGB images: {n_ref_frames} reference + target", img_grid, global_step=0)

        # show the GT depth, if available
        if ("depth_gt", 0) in sample:
            depth_gt_images = [sample[("depth_gt", shift)][ind] for shift in all_shifts]
            # create a heatmap subplot with scale, for each depth image
            fig, axs = plt.subplots(1, len(depth_gt_images))
            for i, img in enumerate(depth_gt_images):
                axs[i].imshow(img[0, :, :], cmap="hot", interpolation="nearest")
                axs[i].set_title(f"Depth image {i}")
                axs[i].axis("off")
                fig.colorbar(
                    axs[i].imshow(img[0, :, :], cmap="hot", interpolation="nearest"),
                    ax=axs[i],
                    location="bottom",
                )

            fig.tight_layout()
            self.writer.add_figure("GT depth images", fig)
        # show the GT poses, if available,
        if ("abs_pose", 0) in sample:
            text_string = ""
            for shift in all_shifts:
                pose = sample[("abs_pose", shift)][ind]
                text_string += f"Shift {shift}: translation: {to_str(pose[:3], precision=4)} [mm], rotation: {to_str(pose[3:], precision=4)} [unit-quaternion]<br>"
            self.writer.add_text(f"GT poses (for frame shifts: {all_shifts})", text_string=text_string, global_step=0)

        # print the list of augmentation transforms done on the sample:
        self.writer.add_text("Augmentation transforms", str(sample["augments"][ind]), global_step=0)

        # save updates to disk
        self.writer.flush()


# ---------------------------------------------------------------------------------------------------------------------

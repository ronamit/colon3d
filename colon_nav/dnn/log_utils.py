from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from colon_nav.dnn.model_info import ModelInfo
from colon_nav.util.general_util import to_str
from colon_nav.util.torch_util import to_numpy

# ---------------------------------------------------------------------------------------------------------------------


class TensorBoardWriter:
    def __init__(
        self,
        log_dir: Path,
        log_freq: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_info: ModelInfo,
        depth_model: torch.nn.Module,
        egomotion_model: torch.nn.Module,
        show_graph: bool = False,
    ) -> None:
        self.writer = SummaryWriter(log_dir=log_dir)
        self.writer.add_text("Model info", str(model_info), global_step=0)
        self.log_freq = log_freq
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_set = train_loader.dataset
        self.val_set = val_loader.dataset
        assert len(self.train_loader) > 0, "Training data loader is empty"
        assert len(self.val_loader) > 0, "Validation data loader is empty"
        print(f"Saving tensorboard logs to {log_dir}")
        self.model_info = model_info
        self.depth_model = depth_model
        self.egomotion_model = egomotion_model

        self.running_loss = 0.0
        self.running_loss_terms = {}

        if show_graph:
            self.visualize_graph()
        # Show some sample data before training:
        sample = next(iter(self.train_loader))
        self.plot_sample(sample, is_train=True, global_step=0)

    # -----------------------------------------------------------------------------------------------------------------
    def visualize_graph(self):
        sample = next(iter(self.val_loader))
        depth_model_input = sample["depth_model_in"]
        ego_model_input = sample["ego_model_in"]
        self.writer.add_graph(self.depth_model, depth_model_input)
        self.writer.add_graph(self.egomotion_model, ego_model_input)
        self.writer.close()

    # -----------------------------------------------------------------------------------------------------------------

    def update_running_train_loss(
        self,
        tot_loss: torch.Tensor,
        loss_terms: dict[str, float],
        global_step: int,
    ):
        self.running_loss += to_numpy(tot_loss)
        self.running_loss_terms = add_to_dict_vals(self.running_loss_terms, loss_terms)
        if global_step % self.log_freq == 0:
            # log the averaged running loss
            running_loss = self.running_loss / self.log_freq
            self.writer.add_scalar("train/loss", running_loss, global_step=global_step)
            for loss_name, loss_val in self.running_loss_terms.items():
                self.writer.add_scalar(f"train/{loss_name}", loss_val / self.log_freq, global_step=global_step)
            # reset the running loss
            self.running_loss = 0.0
            self.running_loss_terms = {}

    # -----------------------------------------------------------------------------------------------------------------

    def plot_sample(self, sample: dict, is_train: bool, global_step: int, outputs: dict | None = None):
        ind = 0  # index of the sample in the batch
        # Take the RGB images from the first example in the batch
        all_shifts = [*self.model_info.ref_frame_shifts, 0]
        n_images = len(all_shifts)
        rgb_images = [sample[("color", shift)][ind] for shift in all_shifts]
        depth_gt_images = [sample[("depth_gt", shift)][ind] for shift in all_shifts]
        n_rows = 2 if outputs is None else 3
        fig, axs = plt.subplots(n_rows, n_images)
        for i in range(n_images):
            img = rgb_images[i].permute(1, 2, 0).cpu().numpy().astype("uint8")
            axs[0, i].imshow(img, interpolation="nearest")
            axs[0, i].set_title(f"RGB image: {all_shifts[i]}")
            axs[0, i].axis("off")
            img = depth_gt_images[i].cpu().numpy()
            axs[1, i].imshow(img[0, :, :], cmap="hot", interpolation="nearest")
            axs[1, i].set_title(f"Depth image: {all_shifts[i]}")
            axs[1, i].axis("off")
            # add colorbar at the bottom of the image
            fig.colorbar(
                axs[1, i].imshow(img[0, :, :], cmap="hot", interpolation="nearest"),
                ax=axs[1, i],
                shrink=0.6,
                location="bottom",
            )
        # If outputs are given, show the estimated depth image of the target image (last image in the list)
        if outputs is not None:
            tgt_depth_est = outputs["tgt_depth_est"][0].detach().cpu().numpy()
            axs[2, -1].imshow(tgt_depth_est[0, :, :], cmap="hot", interpolation="nearest")
            # add colorbar at the bottom of the image
            fig.colorbar(
                axs[2, -1].imshow(tgt_depth_est[0, :, :], cmap="hot", interpolation="nearest"),
                ax=axs[2, -1],
                shrink=0.6,
                location="bottom",
            )

        fig.tight_layout()
        self.writer.add_figure("Sample images", fig, global_step=global_step)

        # Print the sample source info:
        dataset = self.train_set if is_train else self.val_set
        scenes_paths = dataset.scenes_paths
        scene_index = sample["scene_index"][ind]
        target_frame_index = sample["target_frame_idx"][ind]
        fps = dataset.get_scene_metadata()["fps"]
        self.writer.add_text(
            "Sample source",
            f"Scene index: {scene_index}, target frame index: {target_frame_index}, Scene path: {scenes_paths[scene_index]}, target frame time: {to_str(target_frame_index/fps, precision=4)} [s]",
            global_step=global_step,
        )
        # show the GT poses, if available,
        if ("abs_pose", 0) in sample:
            text_string = ""
            for shift in all_shifts:
                pose = sample[("abs_pose", shift)][ind]
                text_string += f"Shift {shift}: translation: {to_str(pose[:3], precision=4)} [mm], rotation: {to_str(pose[3:], precision=4)} [unit-quaternion]<br>"
            self.writer.add_text("GT poses", text_string=text_string, global_step=global_step)

        # print the list of augmentation transforms done on the sample:
        self.writer.add_text("Augmentation transforms", str(sample["augments"][ind]), global_step=global_step)

        # save updates to disk
        self.writer.flush()


# ---------------------------------------------------------------------------------------------------------------------


def add_to_dict_vals(dict_orig: dict, dict_to_add: dict):
    for name, val in dict_to_add.items():
        dict_orig[name] = dict_to_add.get(name, 0.0) + val
    return dict_orig


# ---------------------------------------------------------------------------------------------------------------------

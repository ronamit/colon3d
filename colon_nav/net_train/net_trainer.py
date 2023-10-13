from pathlib import Path

import torch
from torch.utils.data import DataLoader

from colon_nav.net_train.depth_model import DepthModel
from colon_nav.net_train.egomotion_model import EgomotionModel
from colon_nav.net_train.loss_func import LossFunc
from colon_nav.net_train.train_utils import ModelInfo, TensorBoardWriter
from colon_nav.util.general_util import get_time_now_str
from colon_nav.util.torch_util import get_device, sample_to_gpu

# ---------------------------------------------------------------------------------------------------------------------


class NetTrainer:
    def __init__(
        self,
        save_model_path: Path,  # path to save the trained model
        train_loader: DataLoader,  # training data loader
        val_loader: DataLoader,  # validation data loader
        model_info: ModelInfo,  # model info
        load_depth_model_path: Path | None = None,  # path to load a pretrained depth model
        load_egomotion_model_path: Path | None = None,  # path to load a pretrained egomotion model
        n_epochs: int = 300,  # number of epochs to train
        run_name: str = "",  # name of the run
    ):
        self.save_model_path = save_model_path
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_info = model_info
        self.depth_map_size = model_info.depth_map_size
        self.n_epochs = n_epochs
        self.run_name = run_name or get_time_now_str()
        self.ref_frame_shifts = model_info.ref_frame_shifts
        self.device = get_device()
        # The loss terms to use in the loss function and their weights
        self.loss_terms_lambdas = {"depth_sup_L1": 1, "depth_sup_SSIM": 0.1, "trans_sup_L1_quat": 1, "rot_sup_L1_quat": 1, "rot_sup_L1_mat": 1}
        self.loss_func = LossFunc(loss_terms_lambdas=self.loss_terms_lambdas)

        ### Initialize the depth model
        self.depth_model = DepthModel(model_info=model_info, load_depth_model_path=load_depth_model_path)

        ### Initial the egomotion model
        self.egomotion_model = EgomotionModel(
            model_info=model_info,
            load_egomotion_model_path=load_egomotion_model_path,
        )

        # Move the models to the GPU
        self.depth_model = self.depth_model.to(self.device)
        self.egomotion_model = self.egomotion_model.to(self.device)

        ### Initialize the optimizer
        depth_model_params = list(self.depth_model.parameters())
        egomotion_model_params = list(self.egomotion_model.parameters())
        self.optimizer = torch.optim.AdamW(lr=1e-4, params=depth_model_params + egomotion_model_params)

        ### Initialize the learning-rate scheduler
        # LR will be multiplied by 'factor' when the validation loss plateaus for 'patience' epochs
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            verbose=True,
        )

        ### Initialize the tensorboard writer
        self.logger = TensorBoardWriter(
            log_dir=save_model_path / "runs" / self.run_name,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            model_info=self.model_info,
            depth_model=self.depth_model,
            egomotion_model=self.egomotion_model,
        )

        ### Initialize the checkpoint manager
        # TODO: initialize the checkpoint manager

    # ---------------------------------------------------------------------------------------------------------------------

    def train(self):
        """Train the network."""
        self.global_step = 0
        for epoch in range(self.n_epochs):
            self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)
            self.lr_scheduler.step(metrics=val_loss)
            # self.checkpoint_manager.step()

            # TODO: plot example image grid + depth output + info to tensorboard - see https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html

    # ---------------------------------------------------------------------------------------------------------------------

    def train_one_epoch(self, epoch):
        """Train the network for one epoch."""
        print(f"Training epoch {epoch}:")
        for batch_idx, batch_cpu in enumerate(self.train_loader):
            # move the batch to the GPU
            batch = sample_to_gpu(batch_cpu, self.device)
            # train the networks for one batch
            self.train_one_batch(batch_idx, batch)

    # ---------------------------------------------------------------------------------------------------------------------

    def train_one_batch(self, batch_idx: int, batch):
        """Train the network for one batch."""
        self.optimizer.zero_grad()
        tot_loss, losses_scaled = self.compute_loss_func(batch)
        tot_loss.backward()
        self.optimizer.step()
        print(f"  Batch {batch_idx}:")
        print({loss_name: loss_val.item for loss_name, loss_val in losses_scaled.items()}, prefix="    ")
        self.logger.writer.add_scalar("train/loss", tot_loss.item(), self.global_step)
        self.global_step += 1

    # ---------------------------------------------------------------------------------------------------------------------

    def compute_outputs(self, batch):
        """Compute the outputs of the networks."""
        # The RGB images of the reference frames
        tgt_rgb = batch[("color", 0)]  # (B, 3, H, W)
        ref_rgb_frames = [batch[("color", shift)] for shift in self.ref_frame_shifts]  # list of (B, 3, H, W)

        # Depth estimate:
        # run the depth model on the target frame only.
        # (since the depth model is designed to work on a single image)
        tgt_depth_est = self.depth_model(x=tgt_rgb)  # (B, H, W)

        # Egomotion estimation:
        # The ego-motion model gets the RGB images of the target and reference frames
        # We get a list of the estimated egomotion from the target to each reference frame, each is a tensor of shape [B, 7].
        # The egomotion format: (x,y,z,qw,qx,qy,qz) where (x,y,z) is the translation [mm] and (qw,qx,qy,qz) is the rotation unit quaternion.
        tgt_to_refs_motion_est = self.egomotion_model(frames=[*ref_rgb_frames, tgt_rgb])

        return tgt_depth_est, tgt_to_refs_motion_est

    # ---------------------------------------------------------------------------------------------------------------------

    def compute_loss_func(self, batch):
        """Compute the loss function."""

        # Get the outputs of the networks
        tgt_depth_est, tgt_to_refs_motion_est = self.compute_outputs(batch)

        # Get the ground truth depth map of the target frame
        depth_gt = batch[("depth_gt", 0)]  # (B, H, W)

        # Get the ground truth egomotion from the target to each reference frame
        list_tgt_to_refs_motion_gt = [batch[("tgt_to_ref_motion", shift)] for shift in self.ref_frame_shifts]

        # Compute the loss function
        tot_loss, losses_scaled = self.loss_func(
            tgt_depth_est=tgt_depth_est,
            list_tgt_to_refs_motion_est=tgt_to_refs_motion_est,
            depth_gt=depth_gt,
            list_tgt_to_refs_motion_gt = list_tgt_to_refs_motion_gt,
        )

        return tot_loss, losses_scaled

    # ---------------------------------------------------------------------------------------------------------------------

    def validate(self, epoch):
        """Validate the network."""
        self.model.eval()
        print(f"Validation epoch {epoch}:")
        with torch.no_grad():
            for batch_idx, batch_cpu in enumerate(self.val_loader):
                batch = sample_to_gpu(batch_cpu, self.device)
                self.validate_one_batch(batch_idx, batch)

    # ---------------------------------------------------------------------------------------------------------------------

    def validate_one_batch(self, batch_idx: int, batch):
        """Validate the network for one batch."""
        tot_loss, losses_scaled = self.compute_loss_func(batch)
        print(f"  Batch {batch_idx}:")
        print({loss_name: loss_val.item for loss_name, loss_val in losses_scaled.items()}, prefix="    ")
        self.logger.writer("val/loss", tot_loss.item(), self.global_step)

    # ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------

from pathlib import Path

import torch
import torch.nn.functional as nnF
from torch.utils.data import DataLoader

from colon_nav.net_train.depth_model import DepthModel
from colon_nav.net_train.egomotion_model import EgomotionModel
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
        losses = self.compute_loss_func(batch)
        losses["loss"].backward()
        self.optimizer.step()
        print(f"  Batch {batch_idx}:")
        print(losses, prefix="    ")
        self.logger.writer.add_scalar("train/loss", losses["loss"].item(), self.global_step)
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
        # and outputs the relative pose change (egomotion) from the target to the reference frames
        # The egomotion format: (x,y,z,qw,qx,qy,qz) where (x,y,z) is the translation [mm] and (qw,qx,qy,qz) is the rotation unit quaternion.
        tgt_to_refs_motion_est = self.egomotion_model(frames=[*ref_rgb_frames, tgt_rgb])  # [B, n_ref_imgs, 7]

        return tgt_depth_est, tgt_to_refs_motion_est

    # ---------------------------------------------------------------------------------------------------------------------

    def compute_loss_func(self, batch):
        """Compute the loss function."""
        # Get the outputs of the networks
        loss = 0
        tgt_depth_est, tgt_to_refs_motion_est = self.compute_outputs(batch)

        # Compute the depth loss, sum over sample with available GT depth (non-NaN)
        depth_gt = batch[("depth_gt", 0)]  # (B, H, W)
        is_depth_gt_available = torch.isnan(depth_gt[:, 0, 0])  # (B)
        depth_loss_all = nnF.mse_loss(tgt_depth_est, depth_gt, reduction="none")  # (B)
        depth_loss = torch.sum(depth_loss_all[~is_depth_gt_available])  # scalar
        loss += depth_loss
        #       """Compute the loss function."""
        # # Add optional depth loss (for the predicted depth of the target frame vs. the ground truth depth)
        # if self.train_with_gt_depth:
        #     # Get the ground truth depth map of the target frame at the full resolution
        #     depth_gt = batch[("depth_gt", 0)]

        #     # Get the predicted depth map  of the target frame at the full resolution
        #     depth_pred = outputs[("depth", 0)]

        #     # Compute the depth loss
        #     depth_loss = nnF.l2_loss(depth_pred, depth_gt)
        #     # multiply the depth loss by a factor to match the scale of the other losses
        #     depth_loss = depth_loss * 0.5
        #     losses["depth_loss"] = depth_loss
        #     losses["loss"] += depth_loss
        #     assert depth_loss.isfinite().all(), f"depth_loss is not finite: {depth_loss}"

        # # Add optional pose loss (GT relative pose from target to reference frame vs. predicted)
        # if self.train_with_gt_pose:
        #     trans_loss_tot = 0
        #     rot_loss_tot = 0
        #     # Go over all the reference frames
        #     for shift in self.ref_frame_shifts:
        #         # Get the ground truth pose change from target to reference frame
        #         translation_gt, axisangle_gt = poses_to_md2_format(inputs[("tgt_to_ref_pose", shift)])
        #         # Get the predicted pose change from reference to target frame
        #         trans_pred = outputs[("translation", 0, shift)]
        #         rot_pred = outputs[("axisangle", 0, shift)]

        #         # Compute the pose losses:
        #         trans_loss, rot_loss = compute_pose_loss(
        #             trans_pred=trans_pred,
        #             trans_gt=translation_gt,
        #             rot_pred=rot_pred,
        #             rot_gt=axisangle_gt,
        #         )
        #         trans_loss_tot += trans_loss
        #         rot_loss_tot += rot_loss
        #     trans_loss = trans_loss_tot / self.n_ref_imgs
        #     rot_loss = rot_loss_tot / self.n_ref_imgs
        #     # multiply the pose losses by a factor to match the scale of the other losses
        #     trans_loss *= 10
        #     rot_loss *= 100
        #     losses["trans_loss"] = trans_loss
        #     losses["rot_loss"] = rot_loss
        #     losses["loss"] += trans_loss + rot_loss

        return loss

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
        losses = self.compute_loss_func(batch)
        print(f"  Batch {batch_idx}:")
        print(losses, prefix="    ")
        self.logger.writer("val/loss", losses["loss"].item(), self.global_step)

    # ---------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------

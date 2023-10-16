from pathlib import Path

import torch
from torch.utils.data import DataLoader

from colon_nav.dnn.depth_model import DepthModel
from colon_nav.dnn.egomotion_model import EgomotionModel
from colon_nav.dnn.log_utils import TensorBoardWriter, add_to_dict_vals
from colon_nav.dnn.loss_func import LossFunc
from colon_nav.dnn.model_info import ModelInfo
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
        log_freq: int = 10,  # print the running average of train losses to console every this number of batches\steps
    ):
        self.save_model_path = save_model_path
        self.train_loader = train_loader
        self.val_loader = val_loader
        train_set = self.train_loader.dataset
        depth_map_size = train_set.depth_map_size
        self.model_info = model_info
        self.n_epochs = n_epochs
        self.run_name = run_name or get_time_now_str()
        self.ref_frame_shifts = model_info.ref_frame_shifts
        self.device = get_device()
        # The loss terms to use in the loss function and their weights
        self.loss_terms_lambdas = {
            "depth_sup_L1": 1,
            "depth_sup_SSIM": 0.1,
            "trans_sup_L1_quat": 1,
            "rot_sup_L1_quat": 1,
            "rot_sup_L1_mat": 1,
        }
        self.loss_func = LossFunc(loss_terms_lambdas=self.loss_terms_lambdas, ref_frame_shifts=self.ref_frame_shifts)
        self.log_freq = log_freq  # print the running average of train losses to console every this number of batches\steps

        ### Initialize the depth model
        self.depth_model = DepthModel(
            model_info=model_info,
            load_model_path=load_depth_model_path,
            device=self.device,
            is_train=True,
            depth_map_size=depth_map_size,
        )

        ### Initial the egomotion model
        self.egomotion_model = EgomotionModel(
            model_info=model_info,
            load_model_path=load_egomotion_model_path,
            device=self.device,
            is_train=True,
        )

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
            log_freq=self.log_freq,
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
            val_loss = self.validate()
            self.lr_scheduler.step(metrics=val_loss)

            # self.checkpoint_manager.step()

            # TODO: plot example image grid + depth output + info to tensorboard - see https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html

        # Save the model
        self.depth_model.save_model(save_model_path=self.save_model_path)
        self.egomotion_model.save_model(save_model_path=self.save_model_path)
        print(f"Finished training, model saved to {self.save_model_path}")

    # ---------------------------------------------------------------------------------------------------------------------

    def train_one_epoch(self, epoch):
        """Train the network for one epoch."""
        print(f"Epoch #{epoch}:")
        self.depth_model.train()
        self.egomotion_model.train()
        running_loss = 0.0
        for _batch_idx, batch_cpu in enumerate(self.train_loader):
            # move the batch to the GPU
            batch = sample_to_gpu(batch_cpu, self.device)

            # train the networks for one batch
            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Get networks outputs
            outputs = self.compute_outputs(batch)

            # Compute the loss function
            tot_loss, loss_terms, outputs = self.loss_func(batch=batch, outputs=outputs)

            # Backpropagate the loss
            tot_loss.backward()
            self.optimizer.step()

            # add the loss of this batch to the running loss
            running_loss += tot_loss.item()

            # Log the loss
            self.logger.update_running_train_loss(
                tot_loss=tot_loss,
                loss_terms=loss_terms,
                global_step=self.global_step,
            )

            self.global_step += 1

        # Print sample images to tensorboard (every epoch)
        self.logger.plot_sample(
            sample=batch_cpu,
            is_train=True,
            outputs=outputs,
            global_step=self.global_step,
        )

    # ---------------------------------------------------------------------------------------------------------------------

    def compute_outputs(self, batch: dict) -> dict[str, torch.Tensor]:
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

        outputs = {"tgt_depth_est": tgt_depth_est, "list_tgt_to_refs_motion_est": tgt_to_refs_motion_est}
        return outputs

    # ---------------------------------------------------------------------------------------------------------------------

    def validate(self):
        """Calculate the average loss of the model on the validation set."""
        self.depth_model.eval()
        self.egomotion_model.eval()
        print("Validation:")
        tot_loss, losses_scaled = 0, {}
        with torch.no_grad():
            for batch_cpu in self.val_loader:
                batch = sample_to_gpu(batch_cpu, self.device)
                outputs = self.compute_outputs(batch)
                tot_loss_b, losses_scaled_b, outputs = self.loss_func(batch=batch, outputs=outputs)
                # Add the loss of this batch to the total loss
                tot_loss += tot_loss_b
                # Add the scaled losses of this batch to the scaled losses of all batches
                losses_scaled = add_to_dict_vals(dict_orig=losses_scaled, dict_to_add=losses_scaled_b)
            # average the loss over all batches
            tot_loss /= len(self.val_loader)
            # average the scaled losses over all batches
            for loss_name in losses_scaled:
                losses_scaled[loss_name] /= len(self.val_loader)
        print(f"  val/tot_loss: {tot_loss:.3f}")
        print("  val/losses_scaled: ", {loss_name: f"{loss_val:.3f}" for loss_name, loss_val in losses_scaled.items()})
        # Write the losses to the the TensorBoard writer
        self.logger.writer.add_scalar("val/loss", tot_loss.item(), global_step=self.global_step)
        for loss_name, loss_val in losses_scaled.items():
            self.logger.writer.add_scalar(f"val/{loss_name}", loss_val.item(), global_step=self.global_step)
        return tot_loss

    # ---------------------------------------------------------------------------------------------------------------------
    def write_train_log(
        self,
        tot_loss: torch.Tensor,
        losses_scaled: dict[str, torch.Tensor],
        print_to_console: bool = True,
    ):
        """Write the loss to the the TensorBoard writer and print to console."""
        lr = self.optimizer.param_groups[0]["lr"]
        self.logger.writer.add_scalar("train/loss", tot_loss.item(), global_step=self.global_step)
        self.logger.writer.add_scalar("train/lr", lr, global_step=self.global_step)
        for loss_name, loss_val in losses_scaled.items():
            self.logger.writer.add_scalar(f"train/{loss_name}", loss_val.item(), global_step=self.global_step)
        if print_to_console:
            print(f"  global_step: {self.global_step}")
            print(f"  train/loss: {tot_loss.item():.3f}")
            print(
                "  Loss terms:  ",
                {loss_name: f"{loss_val.item():.3f}" for loss_name, loss_val in losses_scaled.items()},
            )

    # ---------------------------------------------------------------------------------------------------------------------

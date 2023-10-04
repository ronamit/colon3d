from pathlib import Path

import torch
from torch.utils.data import DataLoader

from colon_nav.nets.fcb_former_model import FCBFormer
from colon_nav.nets.resnet_model import get_resnet_model
from colon_nav.nets.train_utils import TensorBoardLogger
from colon_nav.util.general_util import get_time_now_str


class NetTrainer:
    def __init__(
        self,
        save_path: Path,  # path to save the trained model
        train_loader: DataLoader,  # training data loader
        val_loader: DataLoader,  # validation data loader
        depth_model_name: str = "fcb_former",  # name of the depth model
        egomotion_model_name: str = "resnet50",  # name of the egomotion model
        load_depth_model_path: Path | None = None,  # path to load a pretrained depth model
        load_egomotion_model_path: Path | None = None,  # path to load a pretrained egomotion model
        n_epochs: int = 300,  # number of epochs to train
        run_name: str = "",  # name of the run
    ):
        self.save_path = save_path
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_epochs = n_epochs
        self.egomotion_model_name = egomotion_model_name
        self.run_name = run_name or get_time_now_str()
        self.train_dataset_meta = self.train_loader.dataset.dataset_meta

        ### Initialize the depth model
        if depth_model_name == "fcb_former":
            self.depth_model = FCBFormer()
        else:
            raise ValueError(f"Unknown depth model name: {depth_model_name}")
        
        ### Load pretrained depth model
        if load_depth_model_path is not None:
            self.depth_model.load_state_dict(torch.load(load_depth_model_path))

        ### Initial the egomotion model
        # TODO: change the input layer for our needs (n_channels=3*(n_ref + 1))
        # TODO: change the output layer for our needs
        self.egomotion_model = get_resnet_model(self.egomotion_model_name)

        ### Load pretrained egomotion model
        if load_egomotion_model_path is not None:
            self.egomotion_model.load_state_dict(torch.load(load_egomotion_model_path))

        ### Initialize the optimizer
        depth_model_params = list(self.depth_model.parameters())
        egomotion_model_params = list(self.egomotion_model.parameters())
        self.optimizer = torch.optim.AdamW(lr=1e-4, params=depth_model_params + egomotion_model_params)

        ### Initialize the learning-rate scheduler
        # LR will be halved when the validation loss plateaus for 10 epochs
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            verbose=True,
        )

        ### Initialize the tensorboard writer
        self.tb_logger = TensorBoardLogger(
            log_dir=self.save_path / "logs" / self.run_name,
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
        for epoch in range(self.n_epochs):
            self.train_one_epoch(epoch)
            val_loss = self.validate(epoch)
            self.scheduler.step(metrics=val_loss)
            self.checkpoint_manager.step()
            self.logger.write("")

            # TODO: plot example image grid + depth output + info to tensorboard - see https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html

    # ---------------------------------------------------------------------------------------------------------------------

    def train_one_epoch(self, epoch):
        """Train the network for one epoch."""
        self.model.train()
        self.logger.write(f"Training epoch {epoch}:")
        for batch_idx, batch in enumerate(self.train_loader):
            self.train_one_batch(batch_idx, batch)

    # ---------------------------------------------------------------------------------------------------------------------

    def train_one_batch(self, batch_idx, batch):
        """Train the network for one batch."""
        self.optimizer.zero_grad()
        losses = self.compute_losses(batch)
        losses["loss"].backward()
        self.optimizer.step()
        self.logger.write(f"  Batch {batch_idx}:")
        self.logger.write_losses(losses, prefix="    ")
        self.writer.add_scalar("train/loss", losses["loss"].item(), self.global_step)
        self.global_step += 1

    # ---------------------------------------------------------------------------------------------------------------------

    def compute_losses(self, batch):
        """Compute the loss function."""
        losses = {}
        losses["loss"] = self.loss(batch)
        return losses

    # ---------------------------------------------------------------------------------------------------------------------

    def validate(self, epoch):
        """Validate the network."""
        self.model.eval()
        self.logger.write(f"Validation epoch {epoch}:")
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                self.validate_one_batch(batch_idx, batch)
        self.logger.write("")

    # ---------------------------------------------------------------------------------------------------------------------

    def validate_one_batch(self, batch_idx, batch):
        """Validate the network for one batch."""
        losses = self.compute_losses(batch)
        self.logger.write(f"  Batch {batch_idx}:")
        self.logger.write_losses(losses, prefix="    ")
        self.writer.add_scalar("val/loss", losses["loss"].item(), self.global_step)

    # ---------------------------------------------------------------------------------------------------------------------

    def loss(self, batch):
        """Compute the loss function."""
        raise NotImplementedError

    # ---------------------------------------------------------------------------------------------------------------------

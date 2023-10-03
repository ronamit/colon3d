from pathlib import Path

import attrs
import torch
from torch.utils.data import DataLoader

from colon_nav.nets.fcb_former_model import FCBFormer
from colon_nav.nets.resnet import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
from colon_nav.nets.train_utils import DatasetMeta
from colon_nav.nets.data_transforms import get_train_transform, get_val_transform

@attrs.define
class NetTrainer:
    save_path: Path  # path to save the trained model
    train_loader: DataLoader  # training data loader
    val_loader: DataLoader  # validation data loader
    n_epochs: int  # number of epochs to train
    egomotion_model_name = attrs.attrib(default="resnet50", validator=attrs.validators.in_(["resnet18", "resnet50"]))
    # parameters that will be set by __attrs_post_init__
    depth_model: torch.nn.Module
    egomotion_model: torch.nn.Module
    train_dataset_meta: DatasetMeta = attrs.field(init=False)

    # ---------------------------------------------------------------------------------------------------------------------

    def __attrs_post_init__(self):
        self.train_dataset_meta = self.train_loader.dataset.dataset_meta

        ### Initialize the depth model
        self.depth_model = FCBFormer()

        ### Initial the egomotion model
        if self.egomotion_model_name == "resnet18":
            self.egomotion_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif self.egomotion_model_name == "resnet50":
            self.egomotion_model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            raise NotImplementedError

        ### Initialize the optimizer

        ### Initialize the learning-rate scheduler

        ### Initialize the loss function

        ### Initialize the logger

        ### Initialize the tensorboard writer

        ### Initialize the checkpoint manager

    # ---------------------------------------------------------------------------------------------------------------------

    def train(self):
        """Train the network."""
        for epoch in range(self.n_epochs):
            self.train_one_epoch(epoch)
            self.validate(epoch)
            self.scheduler.step()
            self.checkpoint_manager.step()
            self.logger.write("")

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

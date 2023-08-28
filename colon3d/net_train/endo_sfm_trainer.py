import csv
import time
from pathlib import Path

import attrs
import torch
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from colon3d.util.general_util import Tee, create_empty_folder, set_rand_seed
from colon3d.util.torch_util import get_device
from endo_sfm.logger import AverageMeter
from endo_sfm.loss_functions import compute_photo_and_geometry_loss, compute_smooth_loss
from endo_sfm.models_def.DispResNet import DispResNet
from endo_sfm.models_def.PoseResNet import PoseResNet
from endo_sfm.utils import save_checkpoint, save_model_info


# ---------------------------------------------------------------------------------------------------------------------
@attrs.define
class TrainRunner:
    save_path: Path = None  # Path to save checkpoints and training outputs
    train_loader: torch.utils.data.DataLoader = None  # Loader for training set
    validation_loader: torch.utils.data.DataLoader = None  # Loader for validation set
    pretrained_disp: str = ""  # path to pre-trained DispNet model (disparity=1/depth),
    # if empty then training from scratch
    pretrained_pose: str = ""  # path to pre-trained PoseNet model, if empty then training from scratch
    with_pretrain: bool = True  # in case training from scratch -  do we use ImageNet pretrained weights or not
    n_epochs: int = 100  # number of epochs to train
    epoch_size: int = 0  # manual epoch size (will match dataset size if not set)
    learning_rate: float = 1e-4  # initial learning rate
    momentum: float = 0.9  # momentum for sgd, alpha parameter for adam
    beta: float = 0.999  # beta parameters for adam
    weight_decay: float = 1e-4  # weight decay
    print_freq: int = 10  # print frequency
    seed: int = 0  # seed for random functions, and network initialization
    log_summary: str = "progress_log_summary.csv"  # csv where to save per-epoch train and valid stats
    log_full: str = "progress_log_full.csv"  # csv where to save per-gradient descent train stats
    log_output: bool = False  # will log dispnet outputs at validation step
    disp_resnet_layers: int = 18  # number of ResNet layers for disparity estimation.
    pose_resnet_layers: int = 18  #    # only 18 is supported for now
    num_scales: int = 1  # the number of scales
    photo_loss_weight: float = 1  # weight for photometric loss
    smooth_loss_weight: float = 0.1  # weight for disparity smoothness loss
    geometry_consistency_weight: float = 0.5  # weight for depth consistency loss
    with_ssim: bool = True  # with ssim or not
    with_mask: bool = True  # with the the mask for moving objects and occlusions or not
    with_auto_mask: bool = False  # with the the mask for stationary points
    padding_mode: str = "zeros"  # padding mode for image warping : this is important for photometric differenciation when going outside target image.
    #  "zeros" will null gradients outside target image.
    #  "border" will only null gradients of the coordinate outside (x or y),
    save_overwrite: bool = True  # overwrite save path if already exists

    # ---------------------------------------------------------------------------------------------------------------------

    def run(self):
        device = get_device()
        torch.autograd.set_detect_anomaly(True)
        is_created = create_empty_folder(self.save_path, save_overwrite=self.save_overwrite)
        if not is_created:
            print(f"{self.save_path} already exists, skipping...\n" + "-" * 50)
            return

        print(f"Outputs will be saved to {self.save_path}")

        with Tee(self.save_path / "prints_log.txt"):  # save the prints to a file
            ### inits
            best_error = -1
            n_iter = 0
            set_rand_seed(self.seed)

            # loggers
            training_writer = SummaryWriter(self.save_path)
            output_writers = []
            if self.log_output:
                for i in range(3):
                    output_writers.append(SummaryWriter(self.save_path / "valid" / str(i)))

            if self.epoch_size == 0:
                self.epoch_size = len(self.train_loader)

            # get the metadata of some scene (we assume that all scenes have the same metadata)
            train_set = self.train_loader.dataset
            scene_metadata = train_set.get_scene_metadata()

            # save model_info file
            save_model_info(
                save_dir_path=self.save_path,
                scene_metadata=scene_metadata,
                disp_resnet_layers=self.disp_resnet_layers,
                pose_resnet_layers=self.pose_resnet_layers,
                overwrite=self.save_overwrite,
            )

            # create model
            print("=> creating model")
            disp_net = DispResNet(self.disp_resnet_layers, pretrained=self.with_pretrain).to(device)
            pose_net = PoseResNet(self.pose_resnet_layers, pretrained=self.with_pretrain).to(device)

            # load parameters
            if self.pretrained_disp:
                loaded_pretrained = torch.load(self.pretrained_disp)
                print("=> using pre-trained weights for DispResNet")
                disp_net.load_state_dict(loaded_pretrained["state_dict"], strict=False)
                disp_net.to(device)

            if self.pretrained_pose:
                loaded_pretrained = torch.load(self.pretrained_pose)
                print("=> using pre-trained weights for PoseResNet")
                pose_net.load_state_dict(loaded_pretrained["state_dict"], strict=False)
                pose_net.to(device)

            print("=> setting adam solver")
            optim_params = [
                {"params": disp_net.parameters(), "lr": self.learning_rate},
                {"params": pose_net.parameters(), "lr": self.learning_rate},
            ]
            optimizer = torch.optim.Adam(optim_params, betas=(self.momentum, self.beta), weight_decay=self.weight_decay)

            with (self.save_path / self.log_summary).open("w") as csvfile:
                writer = csv.writer(csvfile, delimiter="\t")
                writer.writerow(["train_loss", "validation_loss"])

            with (self.save_path / self.log_full).open("w") as csvfile:
                writer = csv.writer(csvfile, delimiter="\t")
                writer.writerow(["train_loss", "photo_loss", "smooth_loss", "geometry_consistency_loss"])

            # save initial checkpoint
            save_checkpoint(
                save_path=self.save_path,
                dispnet_state={
                    "epoch": 0,
                    "state_dict": disp_net.state_dict(),
                },
                exp_pose_state={
                    "epoch": 0,
                    "state_dict": pose_net.state_dict(),
                },
                is_best=True,
                scene_metadata=scene_metadata,
            )

            # main optimization loop
            for i_epoch in range(self.n_epochs):
                print(f"Training epoch {i_epoch+1}/{self.n_epochs}")

                # train for one epoch
                train_loss, n_iter = self.run_epoch(
                    self.save_path,
                    self.train_loader,
                    disp_net,
                    pose_net,
                    optimizer,
                    self.epoch_size,
                    training_writer,
                    n_iter,
                )
                print(f" * Avg Loss : {train_loss:.3f}")

                # evaluate on validation set
                errors, error_names = self.validate_with_gt(self.validation_loader, disp_net, output_writers)

                error_string = ", ".join(
                    f"{name} : {error:.3f}" for name, error in zip(error_names, errors, strict=True)
                )
                print(f" * Avg {error_string}")

                for error, name in zip(errors, error_names, strict=True):
                    training_writer.add_scalar(name, error, i_epoch)

                # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
                decisive_error = errors[1]
                if best_error < 0:
                    best_error = decisive_error

                # remember lowest error and save checkpoint
                is_best = decisive_error < best_error
                best_error = min(best_error, decisive_error)
                save_checkpoint(
                    save_path=self.save_path,
                    dispnet_state={
                        "epoch": i_epoch + 1,
                        "state_dict": disp_net.state_dict(),
                    },
                    exp_pose_state={
                        "epoch": i_epoch + 1,
                        "state_dict": pose_net.state_dict(),
                    },
                    is_best=is_best,
                    scene_metadata=scene_metadata,
                )

                with (self.save_path / self.log_summary).open("a") as csvfile:
                    writer = csv.writer(csvfile, delimiter="\t")
                    writer.writerow([train_loss, decisive_error])

    # ---------------------------------------------------------------------------------------------------------------------

    def run_epoch(
        self,
        save_path: Path,
        train_loader,
        disp_net,
        pose_net,
        optimizer,
        epoch_size,
        train_writer,
        n_iter: int,
    ):
        """Train for one epoch on the training set"""
        device = get_device()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(precision=4)
        w1, w2, w3 = self.photo_loss_weight, self.smooth_loss_weight, self.geometry_consistency_weight

        # switch to train mode
        disp_net.train()
        pose_net.train()

        end = time.time()

        # loop over batches
        for i, batch in enumerate(train_loader):
            log_losses = i > 0 and n_iter % self.print_freq == 0

            data_time.update(time.time() - end)  # measure data loading time

            tgt_img = batch["tgt_img"].to(device)
            ref_imgs = [img.to(device) for img in batch["ref_imgs"]]
            intrinsics = batch["intrinsics"].to(device)

            # compute output
            tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)
            poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

            loss_1, loss_3 = compute_photo_and_geometry_loss(
                tgt_img,
                ref_imgs,
                intrinsics,
                tgt_depth,
                ref_depths,
                poses,
                poses_inv,
                self.num_scales,
                self.with_ssim,
                self.with_mask,
                self.with_auto_mask,
                self.padding_mode,
            )

            loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

            loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3

            if log_losses:
                train_writer.add_scalar("photometric_error", loss_1.item(), n_iter)
                train_writer.add_scalar("disparity_smoothness_loss", loss_2.item(), n_iter)
                train_writer.add_scalar("geometry_consistency_loss", loss_3.item(), n_iter)
                train_writer.add_scalar("total_loss", loss.item(), n_iter)

            # record loss and EPE
            losses.update(loss.item(), self.batch_size)

            # compute gradient and do Adam step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            with (save_path / self.log_full).open("a") as csvfile:
                writer = csv.writer(csvfile, delimiter="\t")
                writer.writerow([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])
            if i % self.print_freq == 0:
                print(f"Train: batch-time {batch_time}, data-time {data_time}, Loss {losses}")
            if i >= epoch_size - 1:
                break

            n_iter += 1

        return losses.avg[0], n_iter

    # ---------------------------------------------------------------------------------------------------------------------

    @torch.no_grad()
    def validate_with_gt(self, val_loader, disp_net, output_writers=None):
        output_writers = output_writers or []
        device = get_device()
        batch_time = AverageMeter()
        error_names = ["abs_diff", "abs_rel"]
        errors = AverageMeter(i=len(error_names))

        # switch to evaluate mode
        disp_net.eval()

        end = time.time()
        for i, batch in enumerate(val_loader):
            tgt_img = batch["tgt_img"].to(device)
            gt_depth = batch["depth_img"].to(device)

            # compute output
            output_disp = disp_net(tgt_img)
            output_depth = 1 / output_disp[:, 0]

            # compute errors
            abs_diff = torch.mean(torch.abs(gt_depth - output_depth))
            abs_rel = torch.mean(torch.abs(gt_depth - output_depth) / gt_depth)
            errors.update([abs_diff, abs_rel])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % self.print_freq == 0:
                print(f"valid: Time {batch_time} Abs Error {errors.val[0]:.4f} ({errors.avg[0]:.4f})")

        return errors.avg, error_names

    # ---------------------------------------------------------------------------------------------------------------------


def compute_depth(disp_net, tgt_img, ref_imgs):
    tgt_depth = [1 / disp for disp in disp_net(tgt_img)]

    ref_depths = []
    for ref_img in ref_imgs:
        ref_depth = [1 / disp for disp in disp_net(ref_img)]
        ref_depths.append(ref_depth)

    return tgt_depth, ref_depths


# ---------------------------------------------------------------------------------------------------------------------


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    poses_inv = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))

    return poses, poses_inv


# ---------------------------------------------------------------------------------------------------------------------

import argparse
import csv
import random
import time
from pathlib import Path

import attrs
import torch
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from colon3d.utils.general_util import (
    ArgsHelpFormatter,
    Tee,
    bool_arg,
    create_empty_folder,
    set_rand_seed,
)
from colon3d.utils.torch_util import get_device
from endo_sfm import custom_transforms
from endo_sfm.dataset_loading import ScenesDataset
from endo_sfm.logger import AverageMeter
from endo_sfm.loss_functions import compute_photo_and_geometry_loss, compute_smooth_loss
from endo_sfm.models_def.DispResNet import DispResNet
from endo_sfm.models_def.PoseResNet import PoseResNet
from endo_sfm.utils import save_checkpoint, save_model_info

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--save_path",
        type=str,
        default="saved_models/temp",
        help="Path to save checkpoints and training outputs",
    )
    parser.add_argument(
        "--save_overwrite",
        type=bool_arg,
        default=True,
        help="overwrite save path if already exists",
    )
    parser.add_argument(
        "--dataset_path",
        help="path to training dataset of scenes",
        default="data/sim_data/TrainData22",
    )
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.1,
        help="ratio of the number of scenes in the validation set from entire training set scenes",
    )
    parser.add_argument(
        "--pretrained_disp",
        dest="pretrained_disp",
        default="saved_models/EndoSFM_orig/DispNet_best.pt",
        help="path to pre-trained DispNet model (disparity=1/depth), if empty then training from scratch",
    )
    parser.add_argument(
        "--pretrained_pose",
        dest="pretrained_pose",
        default="saved_models/EndoSFM_orig/PoseNet_best.pt",
        help="path to pre-trained PoseNet model, if empty then training from scratch",
    )

    parser.add_argument(
        "--with_pretrain",
        type=bool_arg,
        default=True,
        help="in case training from scratch -  do we use ImageNet pretrained weights or not",
    )
    parser.add_argument("--sequence_length", type=int, help="sequence length for training", default=3)
    parser.add_argument("--n_workers", default=4, type=int, help="number of data loading workers")
    parser.add_argument("--n_epochs", default=100, type=int, help="number of total epochs to run")
    parser.add_argument(
        "--epoch_size",
        default=0,
        type=int,
        help="manual epoch size (will match dataset size if not set)",
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="mini-batch size, decrease this if out of memory",
    )
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="initial learning rate")
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="momentum for sgd, alpha parameter for adam",
    )
    parser.add_argument("--beta", default=0.999, type=float, help="beta parameters for adam")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay")
    parser.add_argument("--print_freq", default=10, type=int, help="print frequency")
    parser.add_argument("--seed", default=0, type=int, help="seed for random functions, and network initialization")
    parser.add_argument(
        "--log_summary",
        default="progress_log_summary.csv",
        help="csv where to save per-epoch train and valid stats",
    )
    parser.add_argument(
        "--log_full",
        default="progress_log_full.csv",
        help="csv where to save per-gradient descent train stats",
    )
    parser.add_argument(
        "--log_output",
        type=bool_arg,
        default=False,
        help="will log dispnet outputs at validation step",
    )
    parser.add_argument(
        "--disp_resnet_layers",
        type=int,
        default=18,
        choices=[18, 50],
        help="number of ResNet layers for disparity estimation.",
    )
    parser.add_argument(
        "--num_scales",
        type=int,
        help="the number of scales",
        default=1,
    )
    parser.add_argument(
        "--photo_loss_weight",
        type=float,
        help="weight for photometric loss",
        default=1,
    )
    parser.add_argument(
        "--smooth_loss_weight",
        type=float,
        help="weight for disparity smoothness loss",
        default=0.1,
    )
    parser.add_argument(
        "--geometry_consistency_weight",
        type=float,
        help="weight for depth consistency loss",
        default=0.5,
    )
    parser.add_argument("--with_ssim", type=bool_arg, default=True, help="with ssim or not")
    parser.add_argument(
        "--with_mask",
        type=bool_arg,
        default=True,
        help="with the the mask for moving objects and occlusions or not",
    )
    parser.add_argument(
        "--with_auto_mask",
        type=bool_arg,
        default=False,
        help="with the the mask for stationary points",
    )
    parser.add_argument(
        "--padding_mode",
        type=str,
        choices=["zeros", "border"],
        default="zeros",
        help="padding mode for image warping : this is important for photometric differenciation when going outside target image."
        " zeros will null gradients outside target image."
        " border will only null gradients of the coordinate outside (x or y)",
    )

    args = parser.parse_args()
    train_runner = TrainRunner(
        save_path=Path(args.save_path),
        save_overwrite=args.save_overwrite,
        dataset_path=Path(args.dataset_path),
        validation_ratio=args.validation_ratio,
        pretrained_disp=args.pretrained_disp,
        pretrained_pose=args.pretrained_pose,
        with_pretrain=args.with_pretrain,
        sequence_length=args.sequence_length,
        n_workers=args.n_workers,
        n_epochs=args.n_epochs,
        epoch_size=args.epoch_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        momentum=args.momentum,
        beta=args.beta,
        weight_decay=args.weight_decay,
        print_freq=args.print_freq,
        seed=args.seed,
        log_summary=args.log_summary,
        log_full=args.log_full,
        log_output=args.log_output,
        disp_resnet_layers=args.disp_resnet_layers,
        num_scales=args.num_scales,
        photo_loss_weight=args.photo_loss_weight,
        smooth_loss_weight=args.smooth_loss_weight,
        geometry_consistency_weight=args.geometry_consistency_weight,
        with_ssim=args.with_ssim,
        with_mask=args.with_mask,
        with_auto_mask=args.with_auto_mask,
        padding_mode=args.padding_mode,
    )
    train_runner.run()


# ---------------------------------------------------------------------------------------------------------------------
@attrs.define
class TrainRunner:
    save_path: Path
    dataset_path: Path
    validation_ratio: float = 0.1
    pretrained_disp: str = ""
    pretrained_pose: str = ""
    with_pretrain: bool = True
    sequence_length: int = 3
    n_workers: int = 4
    n_epochs: int = 100
    epoch_size: int = 0
    batch_size: int = 8
    learning_rate: float = 1e-4
    momentum: float = 0.9
    beta: float = 0.999
    weight_decay: float = 1e-4
    print_freq: int = 10
    seed: int = 0
    log_summary: str = "progress_log_summary.csv"
    log_full: str = "progress_log_full.csv"
    log_output: bool = False
    disp_resnet_layers: int = 18
    pose_resnet_layers: int = 18  # only 18 is supported for now
    num_scales: int = 1
    photo_loss_weight: float = 1
    smooth_loss_weight: float = 0.1
    geometry_consistency_weight: float = 0.5
    with_ssim: bool = True
    with_mask: bool = True
    with_auto_mask: bool = False
    padding_mode: str = "zeros"
    save_overwrite: bool = True

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

            # dataset split
            dataset_path = Path(self.dataset_path)
            print(f"Loading dataset from {dataset_path}")
            all_scenes_paths = [
                scene_path
                for scene_path in dataset_path.iterdir()
                if scene_path.is_dir() and scene_path.name.startswith("Scene")
            ]
            random.shuffle(all_scenes_paths)
            n_all_scenes = len(all_scenes_paths)
            n_train_scenes = int(n_all_scenes * (1 - self.validation_ratio))
            n_val_scenes = n_all_scenes - n_train_scenes
            train_scenes_paths = all_scenes_paths[:n_train_scenes]
            val_scenes_paths = all_scenes_paths[n_train_scenes:]
            print(f"Number of training scenes {n_train_scenes}, validation scenes {n_val_scenes}")

            # loggers
            training_writer = SummaryWriter(self.save_path)
            output_writers = []
            if self.log_output:
                for i in range(3):
                    output_writers.append(SummaryWriter(self.save_path / "valid" / str(i)))

            # set data transforms
            chan_normalize_mean = [0.45, 0.45, 0.45]
            chan_normalize_std = [0.225, 0.225, 0.225]
            normalize = custom_transforms.Normalize(mean=chan_normalize_mean, std=chan_normalize_std)
            train_transform = custom_transforms.Compose(
                [
                    custom_transforms.RandomHorizontalFlip(),
                    custom_transforms.RandomScaleCrop(),
                    custom_transforms.ArrayToTensor(),
                    normalize,
                ],
            )
            validation_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

            # training set
            train_set = ScenesDataset(
                scenes_paths=train_scenes_paths,
                sequence_length=self.sequence_length,
                load_tgt_depth=False,
                transform=train_transform,
            )

            # validation set
            val_set = ScenesDataset(
                scenes_paths=val_scenes_paths,
                sequence_length=1,
                load_tgt_depth=True,
                transform=validation_transform,
            )

            # data loaders
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.n_workers,
                pin_memory=True,
            )
            val_loader = torch.utils.data.DataLoader(
                val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.n_workers,
                pin_memory=True,
            )

            if self.epoch_size == 0:
                self.epoch_size = len(train_loader)

            # get the metadata of some scene (we assume that all scenes have the same metadata)
            scene_metadata = train_set.get_scene_metadata(scene_index=0)

            # save model_info file
            save_model_info(
                save_dir_path=self.save_path,
                scene_metadata=scene_metadata,
                disp_resnet_layers=self.disp_resnet_layers,
                pose_resnet_layers=self.pose_resnet_layers,
                overwrite=False,
            )

            # create model
            print("=> creating model")
            disp_net = DispResNet(self.disp_resnet_layers, pretrained=self.with_pretrain).to(device)
            pose_net = PoseResNet(self.pose_resnet_layers, pretrained=self.with_pretrain).to(device)

            # load parameters
            # TODO: save both nets in the same file
            # TODO: mechanism to continue run from the last epoch - if save path not empty
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

            # TODO: make this work (for faster training):
            # disp_net = torch.nn.DataParallel(disp_net)
            # pose_net = torch.nn.DataParallel(pose_net)

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
                    train_loader,
                    disp_net,
                    pose_net,
                    optimizer,
                    self.epoch_size,
                    training_writer,
                    n_iter,
                )
                print(f" * Avg Loss : {train_loss:.3f}")

                # evaluate on validation set
                errors, error_names = self.validate_with_gt(val_loader, disp_net, output_writers)

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

if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------------------------------------------------

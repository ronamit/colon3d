import argparse
import csv
import random
import time
from pathlib import Path

import custom_transforms
import models
import torch
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from colon3d.general_util import UltimateHelpFormatter, create_empty_folder, get_time_now_str, set_rand_seed
from colon3d.torch_util import get_device
from endo_sfm_learner.dataset_loading import ScenesDataset
from endo_sfm_learner.logger import AverageMeter, TermLogger
from endo_sfm_learner.loss_functions import compute_errors, compute_photo_and_geometry_loss, compute_smooth_loss
from endo_sfm_learner.utils import save_checkpoint, tensor2array

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=UltimateHelpFormatter)
    parser.add_argument(
        "--name",
        dest="name",
        type=str,
        default="temp",
        help="name of the experiment, checkpoints are stored in checkpoints/name",
    )
    parser.add_argument(
        "--dataset_path",
        metavar="DIR",
        help="path to training dataset",
        default="data/sim_data/TrainData3",
    )
    parser.add_argument(
        "--validation_percent",
        type=float,
        default=0.2,
        help="percentage of the number of scenes in the validation set from entire training set scenes",
    )

    parser.add_argument("--sequence_length", type=int, metavar="N", help="sequence length for training", default=3)
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--epochs", default=200, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "--epoch-size",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch size (will match dataset size if not set)",
    )
    parser.add_argument("-b", "--batch_size", default=4, type=int, metavar="N", help="mini-batch size")
    parser.add_argument("--lr", "--learning_rate", default=1e-4, type=float, metavar="LR", help="initial learning rate")
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="M",
        help="momentum for sgd, alpha parameter for adam",
    )
    parser.add_argument("--beta", default=0.999, type=float, metavar="M", help="beta parameters for adam")
    parser.add_argument("--weight_decay", "--wd", default=0, type=float, metavar="W", help="weight decay")
    parser.add_argument("--print_freq", default=10, type=int, metavar="N", help="print frequency")
    parser.add_argument("--seed", default=0, type=int, help="seed for random functions, and network initialization")
    parser.add_argument(
        "--log_summary",
        default="progress_log_summary.csv",
        metavar="PATH",
        help="csv where to save per-epoch train and valid stats",
    )
    parser.add_argument(
        "--log_full",
        default="progress_log_full.csv",
        metavar="PATH",
        help="csv where to save per-gradient descent train stats",
    )
    parser.add_argument("--log_output", action="store_true", help="will log dispnet outputs at validation step")
    parser.add_argument(
        "--resnet_layers",
        type=int,
        default=18,
        choices=[18, 50],
        help="number of ResNet layers for depth estimation.",
    )
    parser.add_argument(
        "--num_scales",
        "--number_of_scales",
        type=int,
        help="the number of scales",
        metavar="W",
        default=1,
    )
    parser.add_argument(
        "-p",
        "--photo_loss_weight",
        type=float,
        help="weight for photometric loss",
        metavar="W",
        default=1,
    )
    parser.add_argument(
        "-s",
        "--smooth_loss_weight",
        type=float,
        help="weight for disparity smoothness loss",
        metavar="W",
        default=0.1,
    )
    parser.add_argument(
        "-c",
        "--geometry_consistency_weight",
        type=float,
        help="weight for depth consistency loss",
        metavar="W",
        default=0.5,
    )
    parser.add_argument("--with_ssim", type=bool, default=True, help="with ssim or not")
    parser.add_argument(
        "--with_mask",
        type=bool,
        default=True,
        help="with the the mask for moving objects and occlusions or not",
    )
    parser.add_argument("--with_auto_mask", type=bool, default=False, help="with the the mask for stationary points")
    parser.add_argument("--with_pretrain", type=bool, default=True, help="with or without imagenet pretrain for resnet")
    parser.add_argument(
        "--pretrained_disp",
        dest="pretrained_disp",
        default="pretrained/dispnet_model_best.pt",
        metavar="PATH",
        help="path to pre-trained dispnet model",
    )
    parser.add_argument(
        "--pretrained_pose",
        dest="pretrained_pose",
        default="pretrained/exp_pose_model_best.pt",
        metavar="PATH",
        help="path to pre-trained Pose net model",
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

    ### inits
    best_error = -1
    n_iter = 0
    device = get_device()
    torch.autograd.set_detect_anomaly(True)
    args = parser.parse_args()
    timestamp = get_time_now_str()
    save_path = Path(args.name)
    args.save_path = "checkpoints" / save_path / timestamp
    print(f"=> will save everything to {args.save_path}")
    create_empty_folder(args.save_path)
    set_rand_seed(args.seed)

    # dataset split
    dataset_path = Path(args.dataset_path)
    print(f"Loading dataset from {dataset_path}")
    all_scenes_paths = [
        scene_path
        for scene_path in dataset_path.iterdir()
        if scene_path.is_dir() and scene_path.name.startswith("Scene")
    ]
    random.shuffle(all_scenes_paths)
    n_all_scenes = len(all_scenes_paths)
    n_train_scenes = int(n_all_scenes * 0.8)
    n_val_scenes = n_all_scenes - n_train_scenes
    train_scenes_paths = all_scenes_paths[:n_train_scenes]
    val_scenes_paths = all_scenes_paths[n_train_scenes:]
    print(f"Number of training scenes {n_train_scenes}, validation scenes {n_val_scenes}")

    # loggers
    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path / "valid" / str(i)))

    # set data transforms
    normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
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
        sequence_length=args.sequence_length,
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
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")
    disp_net = models.DispResNet(args.resnet_layers, pretrained=args.with_pretrain).to(device)
    pose_net = models.PoseResNet(18, pretrained=args.with_pretrain).to(device)

    # load parameters
    if args.pretrained_disp:
        print("=> using pre-trained weights for DispResNet")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights["state_dict"], strict=False)
        disp_net.to(device)

    if args.pretrained_pose:
        print("=> using pre-trained weights for PoseResNet")
        weights = torch.load(args.pretrained_pose)
        pose_net.load_state_dict(weights["state_dict"], strict=False)
        pose_net.to(device)

    # TODO: fix issue with DataParallel and device ids mismatch
    # disp_net = torch.nn.DataParallel(disp_net)
    # pose_net = torch.nn.DataParallel(pose_net)

    print("=> setting adam solver")
    optim_params = [
        {"params": disp_net.parameters(), "lr": args.lr},
        {"params": pose_net.parameters(), "lr": args.lr},
    ]
    optimizer = torch.optim.Adam(optim_params, betas=(args.momentum, args.beta), weight_decay=args.weight_decay)

    with (args.save_path / args.log_summary).open("w") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(["train_loss", "validation_loss"])

    with (args.save_path / args.log_full).open("w") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(["train_loss", "photo_loss", "smooth_loss", "geometry_consistency_loss"])

    logger = TermLogger(
        n_epochs=args.epochs,
        train_size=min(len(train_loader), args.epoch_size),
        valid_size=len(val_loader),
    )
    logger.epoch_bar.start()

    # main optimization loop
    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()
        train_loss, n_iter = train(
            args,
            train_loader,
            disp_net,
            pose_net,
            optimizer,
            args.epoch_size,
            logger,
            training_writer,
            n_iter,
        )
        logger.train_writer.write(f" * Avg Loss : {train_loss:.3f}")

        # evaluate on validation set
        errors, error_names = validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers)

        error_string = ", ".join(f"{name} : {error:.3f}" for name, error in zip(error_names, errors, strict=True))
        logger.valid_writer.write(f" * Avg {error_string}")

        for error, name in zip(errors, error_names, strict=True):
            training_writer.add_scalar(name, error, epoch)

        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        decisive_error = errors[1]
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(
            args.save_path,
            {
                "epoch": epoch + 1,
                "state_dict": disp_net.module.state_dict(),
            },
            {
                "epoch": epoch + 1,
                "state_dict": pose_net.module.state_dict(),
            },
            is_best,
        )

        with (args.save_path / args.log_summary).open("a") as csvfile:
            writer = csv.writer(csvfile, delimiter="\t")
            writer.writerow([train_loss, decisive_error])
    logger.epoch_bar.finish()


# ---------------------------------------------------------------------------------------------------------------------


def train(args, train_loader, disp_net, pose_net, optimizer, epoch_size, logger, train_writer, n_iter: int):
    """Train for one epoch on the training set"""
    device = get_device()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight

    # switch to train mode
    disp_net.train()
    pose_net.train()

    end = time.time()
    logger.train_bar.start()
    # loop over batches
    for i, batch in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0

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
            args.num_scales,
            args.with_ssim,
            args.with_mask,
            args.with_auto_mask,
            args.padding_mode,
        )

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3

        if log_losses:
            train_writer.add_scalar("photometric_error", loss_1.item(), n_iter)
            train_writer.add_scalar("disparity_smoothness_loss", loss_2.item(), n_iter)
            train_writer.add_scalar("geometry_consistency_loss", loss_3.item(), n_iter)
            train_writer.add_scalar("total_loss", loss.item(), n_iter)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with (args.save_path / args.log_full).open("a") as csvfile:
            writer = csv.writer(csvfile, delimiter="\t")
            writer.writerow([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])
        logger.train_bar.update(i + 1)
        if i % args.print_freq == 0:
            logger.train_writer.write(f"Train: Time {batch_time} Data {data_time} Loss {losses}")
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0], n_iter


# ---------------------------------------------------------------------------------------------------------------------


@torch.no_grad()
def validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers=None):
    output_writers = output_writers or []
    device = get_device()
    batch_time = AverageMeter()
    error_names = ["abs_diff", "abs_rel", "sq_rel", "a1", "a2", "a3"]
    errors = AverageMeter(i=len(error_names))
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()

    end = time.time()
    logger.valid_bar.start()
    for i, batch in enumerate(val_loader):
        
        tgt_img = batch["tgt_img"].to(device)
        depth = batch["depth_img"].to(device)

        # check gt
        if depth.nelement() == 0:
            continue

        # compute output
        output_disp = disp_net(tgt_img)
        output_depth = 1 / output_disp[:, 0]

        if log_outputs and i < len(output_writers):
            if epoch == 0:
                output_writers[i].add_image("val Input", tensor2array(tgt_img[0]), 0)
                depth_to_show = depth[0]
                output_writers[i].add_image("val target Depth", tensor2array(depth_to_show, max_value=10), epoch)
                depth_to_show[depth_to_show == 0] = 1000
                disp_to_show = (1 / depth_to_show).clamp(0, 10)
                output_writers[i].add_image(
                    "val target Disparity Normalized",
                    tensor2array(disp_to_show, max_value=None, colormap="magma"),
                    epoch,
                )

            output_writers[i].add_image(
                "val Dispnet Output Normalized",
                tensor2array(output_disp[0], max_value=None, colormap="magma"),
                epoch,
            )
            output_writers[i].add_image("val Depth Output", tensor2array(output_depth[0], max_value=10), epoch)

        if depth.nelement() != output_depth.nelement():
            b, h, w = depth.size()
            output_depth = torch.nn.functional.interpolate(output_depth.unsqueeze(1), [h, w]).squeeze(1)

        errors.update(compute_errors(depth, output_depth, args.dataset))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i + 1)
        if i % args.print_freq == 0:
            logger.valid_writer.write(f"valid: Time {batch_time} Abs Error {errors.val[0]:.4f} ({errors.avg[0]:.4f})")
    logger.valid_bar.update(len(val_loader))
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

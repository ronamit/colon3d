import csv
import time
from pathlib import Path

import attrs
import torch
import torch.nn.functional as nnF
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from colon3d.net_train.endosfm_transforms import poses_to_enfosfm_format
from colon3d.net_train.loss_terms import compute_pose_losses
from colon3d.net_train.shared_transforms import sample_to_gpu
from colon3d.net_train.train_utils import DatasetMeta
from colon3d.util.general_util import Tee, create_empty_folder, set_rand_seed
from colon3d.util.torch_util import get_device
from endo_sfm.logger import AverageMeter
from endo_sfm.loss_functions import compute_photo_and_geometry_loss, compute_smooth_loss
from endo_sfm.models_def.DispResNet import DispResNet
from endo_sfm.models_def.PoseResNet import PoseResNet
from endo_sfm.utils import save_checkpoint


# ---------------------------------------------------------------------------------------------------------------------
@attrs.define
class EndoSFMTrainer:
    save_path: Path  # Path to save checkpoints and training outputs
    train_loader: torch.utils.data.DataLoader  # Loader for training set
    val_loader: torch.utils.data.DataLoader  # Loader for validation set
    pretrained_model_path: str = ""  # path to pre-trained model
    with_pretrain: bool = True  # in case training from scratch -  do we use ImageNet pretrained weights or not
    n_epochs: int = 100  # number of epochs to train
    learning_rate: float = 1e-4  # initial learning rate
    momentum: float = 0.9  # momentum for sgd, alpha parameter for adam
    beta: float = 0.999  # beta parameters for adam
    weight_decay: float = 1e-4  # weight decay
    print_freq: int = 10  # print frequency
    seed: int = 0  # seed for random functions, and network initialization
    log_summary: str = "progress_log_summary.csv"  # csv where to save per-epoch train and valid stats
    log_full: str = "progress_log_full.csv"  # csv where to save per-gradient descent train stats
    log_output: bool = False  # will log dispnet outputs at validation step
    num_layers: int = 18  # number of ResNet layers for depth and pose networks
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
    ########  fields that will be set later ######
    train_dataset_meta: DatasetMeta = attrs.field(init=False)
    feed_height: int = attrs.field(init=False)  # height of the input image
    feed_width: int = attrs.field(init=False)  # width of the input image
    train_with_gt_depth: bool = attrs.field(init=False)  # if set, will train with ground truth depth
    train_with_gt_pose: bool = attrs.field(init=False)  # if set, will train with ground truth pose
    num_input_images: int = attrs.field(init=False)  # frame ids within each sample (target = 0, reference = 1, 2, ...)
    losses_names: list = None  # list of losses names
    pretrained_disp: str = ""  # path to pre-trained DispNet model (disparity=1/depth)
    pretrained_pose: str = ""  # path to pre-trained PoseNet model
    # ---------------------------------------------------------------------------------------------------------------------

    def __attrs_post_init__(self):
        self.train_dataset_meta = self.train_loader.dataset.dataset_meta
        self.feed_height = self.train_dataset_meta.feed_height  # height of the input image
        self.feed_width = self.train_dataset_meta.feed_width  # width of the input image
        self.train_with_gt_depth = self.train_dataset_meta.load_gt_depth  # if set, will train with ground truth depth
        self.train_with_gt_pose = self.train_dataset_meta.load_gt_pose  # if set, will train with ground truth pose
        # number of frames in each sample (target + reference frames):
        self.num_input_images = self.train_dataset_meta.num_input_images
        self.losses_names = ["photo_loss", "smooth_loss", "geometry_consistency_loss"]
        if self.train_with_gt_depth:
            self.losses_names += ["depth_loss"]
        if self.train_with_gt_pose:
            self.losses_names += ["trans_loss", "rot_loss"]

        if self.pretrained_model_path:
            self.pretrained_disp = self.pretrained_model_path / "DispNet_best.pt"
            self.pretrained_pose = self.pretrained_model_path / "PoseNet_best.pt"

    # ---------------------------------------------------------------------------------------------------------------------
    def run(self):
        device = get_device()
        torch.autograd.set_detect_anomaly(True)
        is_created = create_empty_folder(self.save_path, save_overwrite=self.save_overwrite)
        if not is_created:
            print(f"{self.save_path} already exists, skipping...\n" + "-" * 50)
            return

        print(f"Outputs will be saved to {self.save_path}")

        # save params
        with (self.save_path / "params.txt").open("w") as file:
            file.write(str(self) + "\n")

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

            # get the metadata of some scene (we assume that all scenes have the same metadata)
            train_set = self.train_loader.dataset
            scene_metadata = train_set.get_scene_metadata()

            # create model
            print("=> creating model")
            disp_net = DispResNet(self.num_layers, pretrained=self.with_pretrain).to(device)
            pose_net = PoseResNet(self.num_layers, pretrained=self.with_pretrain).to(device)

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
                writer.writerow(self.losses_names)

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
                    i_epoch,
                    self.save_path,
                    self.train_loader,
                    disp_net,
                    pose_net,
                    optimizer,
                    training_writer,
                    n_iter,
                )
                print(f" * Avg Loss : {train_loss:.3f}")

                # evaluate on validation set
                errors, error_names = self.validate_with_gt(self.val_loader, disp_net, output_writers)

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
        i_epoch,
        save_path: Path,
        train_loader,
        disp_net,
        pose_net,
        optimizer,
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
        n_batches = len(train_loader)

        # loop over batches
        for i, batch_cpu in enumerate(train_loader):
            batch = sample_to_gpu(batch_cpu, device=device)
            data_time.update(time.time() - end)  # measure data loading time
            log_losses = i > 0 and n_iter % self.print_freq == 0
            # RGB of the target image (index 0) and reference images (index 1, 2, ...)
            rgb_imgs = [batch[("color", i)] for i in range(1, self.num_input_images)]
            tgt_img = rgb_imgs[0]
            ref_imgs = rgb_imgs[1:]
            intrinsics_K = batch["K"]

            # Get the disparity predictions: num_input_images x [batch_size, height, width]
            pred_disps = [disp_net(img) for img in rgb_imgs]
            pred_depths = [1 / d for d in pred_disps]

            # Get the predicted pose change from reference to target frame: num_input_images x [batch_size, 6]
            # Note: the first pose is target to target, i.e. the identity pose (no change)
            #  Predict pose (6-DoF) change from the reference frame (i-k, k=1,2,...) to the target \ source frame (i)
            pred_poses = [pose_net(tgt_img, img) for img in rgb_imgs]
            #  Predict the inverse pose (6-DoF) changes
            pred_inv_poses = [pose_net(img, tgt_img) for img in rgb_imgs]

            loss_terms = {loss_name: 0 for loss_name in self.losses_names}
            loss_weights = {"photo_loss": w1, "smooth_loss": w2, "geometry_consistency_loss": w3}

            # Compute the supervised depth loss if needed
            if self.train_with_gt_depth:
                depth_loss = 0
                for frame_id in range(self.num_input_images):
                    depth_gt = batch[("depth", frame_id)]
                    # add a supervised loss term for training the depth network
                    depth_loss += nnF.l1_loss(pred_depths[frame_id], depth_gt)
                depth_loss /= self.num_input_images
                loss_weights["depth_loss"] = 1
                assert depth_loss.isfinite(), f"depth_loss is not finite: {depth_loss}"
                # multiply the depth loss by a factor to match the scale of the other losses
                depth_loss = depth_loss * 0.1
                loss_terms["depth_loss"] = depth_loss

            # Compute the supervised pose loss if needed
            if self.train_with_gt_pose:
                # Go over all the reference frames
                for frame_id in range(self.num_input_images):
                    # convert the GT pose changes to axis-angle format
                    tgt_to_ref_pose_gt = poses_to_enfosfm_format(batch[("tgt_to_ref_pose", frame_id)])  # [batch_size, 6]
                    ref_to_tgt_pose_gt = poses_to_enfosfm_format(batch[("ref_to_tgt_pose", frame_id)]) # [batch_size, 6]
                    # the estimated pose changes  are in axis-angle format
                    ref_to_tgt_pose_pred = pred_poses[frame_id] # [batch_size, 6]
                    tgt_to_ref_pose_pred = pred_inv_poses[frame_id]  # [batch_size, 6]

                # add a supervised loss term for training the pose network

                trans_loss1, rot_loss1 = compute_pose_losses(pose_pred=tgt_to_ref_pose_pred, pose_gt=tgt_to_ref_pose_gt)
                trans_loss2, rot_loss2 = compute_pose_losses(pose_pred=ref_to_tgt_pose_pred, pose_gt=ref_to_tgt_pose_gt)
                trans_loss = (trans_loss1 + trans_loss2) / 2
                rot_loss = (rot_loss1 + rot_loss2) / 2
                loss_weights["trans_loss"] = 1
                loss_terms["trans_loss"] = trans_loss
                loss_weights["rot_loss"] = 1
                loss_terms["rot_loss"] = rot_loss

            # compute the photometric and geometry consistency losses
            tgt_depth = pred_depths[0]
            ref_depths = pred_depths[1:]
            loss_terms["photo_loss"], loss_terms["geometry_consistency_loss"] = compute_photo_and_geometry_loss(
                tgt_img,
                ref_imgs,
                intrinsics_K,
                tgt_depth,
                ref_depths,
                pred_poses,
                pred_inv_poses,
                self.num_scales,
                self.with_ssim,
                self.with_mask,
                self.with_auto_mask,
                self.padding_mode,
            )

            loss_terms["smooth_loss"] = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

            loss = sum(loss_terms[loss_name] * loss_weights[loss_name] for loss_name in self.losses_names)

            if log_losses:
                for loss_name in self.losses_names:
                    train_writer.add_scalar(loss_name, loss_terms[loss_name].item(), n_iter)
                train_writer.add_scalar("total_loss", loss.item(), n_iter)

            with (save_path / self.log_full).open("a") as csvfile:
                writer = csv.writer(csvfile, delimiter="\t")
                writer.writerow([loss.item()] + [loss_terms[loss_name].item() for loss_name in self.losses_names])

            # record loss and EPE
            losses.update(loss.item(), self.train_loader.batch_size)

            # compute gradient and do Adam step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % self.print_freq == 0:
                print(
                    f"Train: Epoch {i_epoch}/{self.n_epochs}, Batch: {i}/{n_batches}, Loss {losses}, batch-time {batch_time}, data-time {data_time},",
                )

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
        for i, batch_cpu in enumerate(val_loader):
            batch = sample_to_gpu(batch_cpu, device=device)
            tgt_img = batch["target_img"].to(device)
            gt_depth = batch["target_depth"].to(device)

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

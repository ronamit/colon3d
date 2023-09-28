# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
import json
import time
from pathlib import Path

import attrs
import numpy as np
import torch
import torch.nn.functional as nnF
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader

from colon3d.net_train.loss_terms import compute_pose_loss_aux
from colon3d.net_train.md2_transforms import poses_to_md2_format
from colon3d.net_train.shared_transforms import sample_to_gpu
from colon3d.net_train.train_utils import DatasetMeta
from colon3d.util.general_util import create_folder_if_not_exists, to_str
from colon3d.util.torch_util import get_device
from monodepth2.layers import (
    SSIM,
    BackprojectDepth,
    Project3D,
    compute_depth_errors,
    disp_to_depth,
    get_smooth_loss,
    transformation_from_parameters,
)
from monodepth2.networks.depth_decoder import DepthDecoder
from monodepth2.networks.pose_cnn import PoseCNN
from monodepth2.networks.pose_decoder import PoseDecoder
from monodepth2.networks.resnet_encoder import ResnetEncoder
from monodepth2.utils import normalize_image, sec_to_hm_str


# ---------------------------------------------------------------------------------------------------------------------
@attrs.define
class MonoDepth2Trainer:
    save_path: Path  # path to save the trained model
    train_loader: DataLoader  # training data loader
    val_loader: DataLoader  # validation data loader
    num_layers: int = 18  # number of resnet layers # choices=[18, 34, 50, 101, 152]
    n_scales: int = 4  # number of scales
    disparity_smoothness: float = 1e-3  # disparity smoothness weight
    scales: list = (0, 1, 2, 3)  # scales used in the loss
    min_depth: float = 0.1  # minimum depth
    max_depth: float = 100.0  # maximum depth
    learning_rate: float = 1e-4  # learning rate
    n_epochs: int = 200  # number of epochs
    scheduler_step_size: int = 15  # step size of the scheduler
    avg_reprojection: bool = False  # if set, uses average reprojection loss
    disable_automasking: bool = False  # if set, doesn't do auto-masking
    predictive_mask: bool = False  # if set, uses a predictive masking scheme as in Zhou et al
    no_ssim: bool = False  # if set, disables ssim in the loss
    weights_init: str = "pretrained"  # pretrained or scratch # choices=["pretrained", "scratch"]
    pose_model_input: str = "pairs"  # how many images the pose network gets # choices=["pairs", "all"]
    pose_model_type: str = "separate_resnet"  # normal or shared # choices=["posecnn", "separate_resnet"]
    pretrained_model_path: str = None  # name of model to load
    models_to_load: tuple = ("encoder", "depth", "pose_encoder", "pose")  # models to load
    log_frequency: int = 10  # number of batches between each tensorboard log
    save_frequency: int = 1  # number of epochs between each save
    disable_median_scaling: bool = False  # if set disables median scaling in evaluation
    pred_depth_scale_factor: float = 1  # if set multiplies predictions by this number
    no_eval: bool = False  # if set disables evaluation
    post_process: bool = False  # if set will perform the flipping post processing from the original monodepth paper
    save_overwrite: bool = True  # overwrite save path if already exists
    ########  fields that will be set later ######
    batch_size: int = attrs.field(init=False)
    log_path: str = attrs.field(init=False)  # path to save logs
    models: dict = attrs.field(init=False)
    parameters_to_train: list = attrs.field(init=False)
    device: torch.device = attrs.field(init=False)
    num_input_images: int = attrs.field(init=False)  # frame ids within each sample (target = 0, reference = 1, 2, ...)
    model_optimizer: optim.Optimizer = attrs.field(init=False)
    model_lr_scheduler: optim.lr_scheduler = attrs.field(init=False)
    num_total_steps: int = attrs.field(init=False)
    val_iter: iter = attrs.field(init=False)
    writers: dict = attrs.field(init=False)
    ssim: nn.Module = attrs.field(init=False)
    backproject_depth: dict = attrs.field(init=False)
    project_3d: dict = attrs.field(init=False)
    depth_metric_names: list = attrs.field(init=False)
    train_dataset_meta: DatasetMeta = attrs.field(init=False)
    feed_height: int = attrs.field(init=False)  # height of the input image
    feed_width: int = attrs.field(init=False)  # width of the input image
    train_with_gt_depth: bool = attrs.field(init=False)  # if set, will train with ground truth depth
    train_with_gt_pose: bool = attrs.field(init=False)  # if set, will train with ground truth pose
    frame_ids: list = attrs.field(init=False)
    epoch: int = 0
    step: int = 0
    start_time: float = 0
    # ---------------------------------------------------------------------------------------------------------------------

    def __attrs_post_init__(self):
        self.save_path = Path(self.save_path)
        create_folder_if_not_exists(self.save_path)
        self.log_path = self.save_path
        self.train_dataset_meta = self.train_loader.dataset.dataset_meta
        self.feed_height = self.train_dataset_meta.feed_height  # height of the input image
        self.feed_width = self.train_dataset_meta.feed_width  # width of the input image
        self.train_with_gt_depth = self.train_dataset_meta.load_gt_depth  # if set, will train with ground truth depth
        self.train_with_gt_pose = self.train_dataset_meta.load_gt_pose  # if set, will train with ground truth pose
        # number of frames in each sample (target + reference frames):
        self.num_input_images = self.train_dataset_meta.num_input_images
        self.models = {}
        self.parameters_to_train = []
        self.device = get_device()
        self.n_scales = len(self.scales)
        self.frame_ids = list(range(self.num_input_images))

        # checking height and width are multiples of 32
        assert self.feed_height % 32 == 0, "'height' must be a multiple of 32"
        assert self.feed_width % 32 == 0, "'width' must be a multiple of 32"

        # save params
        with (self.save_path / "params.txt").open("w") as file:
            file.write(str(self) + "\n")

        # Init models
        self.models["encoder"] = ResnetEncoder(self.num_layers, self.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = DepthDecoder(self.models["encoder"].num_ch_enc, self.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        if self.pose_model_type == "separate_resnet":
            self.models["pose_encoder"] = ResnetEncoder(
                self.num_layers,
                self.weights_init == "pretrained",
                num_input_images=self.num_input_images,
            )
            self.models["pose_encoder"].to(self.device)
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())

            self.models["pose"] = PoseDecoder(
                self.models["pose_encoder"].num_ch_enc,
                num_input_features=1,
                num_frames_to_predict_for=2,
            )

        elif self.pose_model_type == "shared":
            self.models["pose"] = PoseDecoder(self.models["encoder"].num_ch_enc, self.num_input_images)

        elif self.pose_model_type == "posecnn":
            self.models["pose"] = PoseCNN(self.num_input_images if self.pose_model_input == "all" else 2)

        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())

        if self.predictive_mask:
            assert (
                self.disable_automasking
            ), "When using predictive_mask, please disable automasking with --disable_automasking"

            # Our implementation of the predictive masking baseline has the the same architecture
            # as our depth decoder. We predict a separate mask for each source frame.
            self.models["predictive_mask"] = DepthDecoder(
                self.models["encoder"].num_ch_enc,
                self.scales,
                num_output_channels=(len(self.frame_ids) - 1),
            )
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.scheduler_step_size, 0.1)

        if self.pretrained_model_path is not None:
            self.load_model()

        print("Models and tensorboard events files are saved to:\n  ", self.log_path)
        print("Training is using:\n  ", self.device)

        num_train_samples = len(self.train_loader.dataset)
        num_val_samples = len(self.val_loader.dataset)
        self.batch_size = self.train_loader.batch_size
        self.num_total_steps = num_train_samples // self.batch_size * self.n_epochs

        self.val_iter = iter(self.val_loader)
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(self.log_path / mode)

        if not self.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.scales:
            h = self.feed_height // (2**scale)
            w = self.feed_width // (2**scale)

            self.backproject_depth[scale] = BackprojectDepth(self.batch_size, h, w)
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print(f"There are {num_train_samples:d} training items and {num_val_samples:d} validation items\n")

        self.save_opts()

    # ---------------------------------------------------------------------------------------------------------------------

    def set_train(self):
        """Convert all models to training mode"""
        for m in self.models.values():
            m.train()

    # ---------------------------------------------------------------------------------------------------------------------

    def set_eval(self):
        """Convert all models to testing/evaluation mode"""
        for m in self.models.values():
            m.eval()

    # ---------------------------------------------------------------------------------------------------------------------

    def run(self):
        """Run the entire training pipeline"""
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for i_epoch in range(self.n_epochs):
            self.epoch = i_epoch
            self.run_epoch()
            if (i_epoch + 1) % self.save_frequency == 0:
                self.save_model()

    # ---------------------------------------------------------------------------------------------------------------------

    def run_epoch(self):
        """Run a single epoch of training and validation"""

        print("Training")
        self.set_train()

        for batch_idx, inputs_cpu in enumerate(self.train_loader):
            before_op_time = time.time()
            inputs = sample_to_gpu(inputs_cpu, self.device)

            # Process the batch to get the outputs and losses
            # outputs fields:
            # ("disp", s) : predicted disparity (1/depth) at scale s.
            # ("translation", 0, i) : predicted translation from target (frame_id=0) to reference frame i.
            # ("axisangle", 0, i) : predicted axis-angle rotation from target (frame_id=0) to reference frame i.
            outputs, losses = self.process_batch(inputs)

            # Add optional depth loss (for the predicted depth of the target frame vs. the ground truth depth)
            if self.train_with_gt_depth:
                # Get the ground truth depth map of the target frame at the full resolution
                depth_gt = inputs[("depth_gt", 0)]
                # Get the predicted disparity map  of the target frame at the full resolution
                disp_pred = outputs[("disp", 0)]
                # Convert the predicted disparity to depth
                _, depth_pred = disp_to_depth(disp_pred, self.min_depth, self.max_depth)
                # Compute the depth loss
                depth_loss = nnF.l1_loss(depth_pred, depth_gt)
                # multiply the depth loss by a factor to match the scale of the other losses
                depth_loss = depth_loss * 0.5
                losses["depth_loss"] = depth_loss
                losses["loss"] += depth_loss
                assert depth_loss.isfinite().all(), f"depth_loss is not finite: {depth_loss}"

            # Add optional pose loss (GT relative pose from target to reference frame vs. predicted)
            if self.train_with_gt_pose:
                trans_loss_tot = 0
                rot_loss_tot = 0
                # Go over all the reference frames (1,2,..) and sum the pose loss
                for i, frame_id in enumerate(self.frame_ids[1:]):
                    # Get the ground truth pose change from target to reference frame
                    translation_gt, axisangle_gt = poses_to_md2_format(inputs[("tgt_to_ref_pose", i)])
                    # Get the predicted pose change from reference to target frame
                    trans_pred = outputs[("translation", 0, frame_id)]
                    rot_pred = outputs[("axisangle", 0, frame_id)]
                    trans_pred = trans_pred[:, i, 0, :]  # [batch_size, 3]
                    rot_pred = rot_pred[:, i, 0, :]  # [batch_size, 3] (axis-angle representation)

                    # Compute the pose losses:
                    trans_loss, rot_loss = compute_pose_loss_aux(
                        trans_pred=trans_pred,
                        trans_gt=translation_gt,
                        rot_pred=rot_pred,
                        rot_gt=axisangle_gt,
                    )
                    trans_loss_tot += trans_loss
                    rot_loss_tot += rot_loss
                n_ref_frames = len(self.frame_ids) - 1
                trans_loss = trans_loss_tot / n_ref_frames
                rot_loss = rot_loss_tot / n_ref_frames
                # multiply the pose losses by a factor to match the scale of the other losses
                trans_loss *= 10
                rot_loss *= 100
                losses["trans_loss"] = trans_loss
                losses["rot_loss"] = rot_loss
                losses["loss"] += trans_loss + rot_loss

            # Take a gradient step
            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data, losses)
                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)
                self.log("train", inputs, outputs, losses)
                self.val()
            self.step += 1
        self.model_lr_scheduler.step()

    # ---------------------------------------------------------------------------------------------------------------------

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses.
        outputs fields:
        # ("disp", s) : predicted disparity (1/depth) at scale s.
        # ("translation", 0, i) : predicted translation from target (frame_id=0) to reference frame i.
        # ("axisangle", 0, i) : predicted axis-angle rotation from target (frame_id=0) to reference frame i.
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # In MonoDepth v2 we only feed the image with frame_id 0 (target) through the depth encoder
        input_imgs = inputs[("color", 0, 0)]
        features = self.models["encoder"](input_imgs)
        outputs = self.models["depth"](features)

        if self.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)

        outputs.update(self.predict_poses(inputs))

        self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses

    # ---------------------------------------------------------------------------------------------------------------------

    def predict_poses(self, inputs):
        """Predict poses between input frames for monocular sequences.
        We predict the pose chamge from the target (frame_id=0) to each reference frame (frame_id=1,2,...).
        Outputs:
        # ("translation", 0, i) : predicted translation from target (frame_id=0) to reference frame i.
        # ("axisangle", 0, i) : predicted axis-angle rotation from target (frame_id=0) to reference frame i.
        # ("cam_T_cam", 0, i) : predicted 4x4 pose transformation from target (frame_id=0) to reference frame i.
        """
        outputs = {}

        # Here we input all frames to the pose net (and predict all poses) together
        if self.pose_model_type in ["separate_resnet", "posecnn"]:
            pose_inputs = torch.cat([inputs[("color", i, 0)] for i in self.frame_ids], 1)

            if self.pose_model_type == "separate_resnet":
                pose_inputs = [self.models["pose_encoder"](pose_inputs)]

        axisangle, translation = self.models["pose"](pose_inputs)

        for i, f_i in enumerate(self.frame_ids[1:]):
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(axisangle[:, i], translation[:, i])

        return outputs

    # ---------------------------------------------------------------------------------------------------------------------

    def val(self):
        """Validate the model on a single minibatch"""
        self.set_eval()
        try:
            inputs = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)
        inputs = sample_to_gpu(inputs, self.device)

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    # ---------------------------------------------------------------------------------------------------------------------

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.scales:
            disp = outputs[("disp", scale)]
            disp = nnF.interpolate(disp, [self.feed_height, self.feed_width], mode="bilinear", align_corners=False)
            source_scale = 0

            _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)

            outputs[("depth", 0, scale)] = depth

            for _i, frame_id in enumerate(self.frame_ids[1:]):
                T = inputs["stereo_T"] if frame_id == "s" else outputs["cam_T_cam", 0, frame_id]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0],
                        translation[:, 0] * mean_inv_depth[:, 0],
                        frame_id < 0,
                    )

                cam_points = self.backproject_depth[source_scale](depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = nnF.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border",
                    align_corners=True,
                )

                if not self.disable_automasking:
                    outputs[("color_identity", frame_id, scale)] = inputs[("color", frame_id, source_scale)]

    # ---------------------------------------------------------------------------------------------------------------------

    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images"""
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    # ---------------------------------------------------------------------------------------------------------------------

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch"""
        losses = {}
        total_loss = 0

        for scale in self.scales:
            loss = 0
            reprojection_losses = []
            source_scale = 0
            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]

            for frame_id in self.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))

            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.disable_automasking:
                identity_reprojection_losses = []
                for frame_id in self.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                if self.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.predictive_mask:
                # use the predicted mask
                mask = outputs["predictive_mask"]["disp", scale]
                mask = nnF.interpolate(mask, [self.feed_height, self.feed_width], mode="bilinear", align_corners=False)
                reprojection_losses *= mask

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
                loss += weighting_loss.mean()

            if self.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                reprojection_loss = reprojection_losses

            if not self.disable_automasking:
                # add random numbers to break ties
                identity_reprojection_loss += (
                    torch.randn(identity_reprojection_loss.shape, device=self.device) * 0.00001
                )

                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                to_optimise, idxs = torch.min(combined, dim=1)

            if not self.disable_automasking:
                outputs[f"identity_selection/{scale}"] = (idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.avg_reprojection * smooth_loss / (2**scale)
            total_loss += loss
            losses[f"loss/{scale}"] = loss

        total_loss /= self.n_scales
        losses["loss"] = total_loss
        return losses

    # ---------------------------------------------------------------------------------------------------------------------

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = depth_pred.detach().squeeze(1)  # [B, 1, H, W] -> [B, H, W]
        depth_gt = inputs["depth_gt"].squeeze(1)  # [B, 1, H, W] -> [B, H, W]

        mask = depth_gt > 0

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]

        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss: torch.Tensor, losses: dict):
        """Print a logging statement to the terminal"""
        samples_per_sec = self.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print(
            f"epoch {self.epoch:>3} | batch {batch_idx:>6} | loss: {loss:.5f}"
            f"| examples/s: {samples_per_sec:5.1f}  | time elapsed: {sec_to_hm_str(time_sofar)} | time left: { sec_to_hm_str(training_time_left)}\n",
            ">" + ", ".join([f"{k}: {v.item():.5f}" for k, v in losses.items() if k != "loss"]),
        )

    # ---------------------------------------------------------------------------------------------------------------------

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file"""
        writer = self.writers[mode]
        for loss_name, loss_val in losses.items():
            writer.add_scalar(f"{loss_name}", loss_val, self.step)

        for j in range(min(4, self.batch_size)):  # write a maxmimum of four images
            for s in self.scales:
                for frame_id in self.frame_ids:
                    writer.add_image(
                        f"color_{frame_id}_{s}/{j}",
                        inputs[("color", frame_id, s)][j].data,
                        self.step,
                    )
                    if s == 0 and frame_id != 0:
                        writer.add_image(
                            f"color_pred_{frame_id}_{s}/{j}",
                            outputs[("color", frame_id, s)][j].data,
                            self.step,
                        )

                writer.add_image(f"disp_{s}/{j}", normalize_image(outputs[("disp", s)][j]), self.step)

                if self.predictive_mask:
                    for f_idx, frame_id in enumerate(self.frame_ids[1:]):
                        writer.add_image(
                            f"predictive_mask_{frame_id}_{s}/{j}",
                            outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
                            self.step,
                        )

                elif not self.disable_automasking:
                    writer.add_image(
                        f"automask_{s}/{j}",
                        outputs[f"identity_selection/{s}"][j][None, ...],
                        self.step,
                    )

    # ---------------------------------------------------------------------------------------------------------------------

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with"""
        to_save = {k: to_str(v) for k, v in self.__getstate__().items()}
        with (self.log_path / "opt.json").open("w") as f:
            json.dump(to_save, f, indent=2)

    # ---------------------------------------------------------------------------------------------------------------------

    def save_model(self):
        """Save model weights to disk"""
        # save the last model  to log_path and also save the weights to log_path/models/weights_{epoch}
        save_folders = [self.log_path, self.log_path / "models" / f"weights_{self.epoch}"]
        for save_folder in save_folders:
            create_folder_if_not_exists(save_folder)
            for model_name, model in self.models.items():
                save_path = save_folder / f"{model_name}.pth"
                to_save = model.state_dict()
                if model_name == "encoder":
                    # save the sizes - these are needed at prediction time
                    to_save["height"] = self.feed_height
                    to_save["width"] = self.feed_width
                torch.save(to_save, save_path)
            save_path = save_folder / "adam.pth"
            torch.save(self.model_optimizer.state_dict(), save_path)

    # ---------------------------------------------------------------------------------------------------------------------

    def load_model(self):
        """Load model(s) from disk"""
        self.pretrained_model_path = Path(self.pretrained_model_path)

        assert self.pretrained_model_path.is_dir(), f"Cannot find folder {self.pretrained_model_path}"
        print(f"loading model from folder {self.pretrained_model_path}")

        for n in self.models_to_load:
            print(f"Loading {n} weights...")
            path = self.pretrained_model_path / f"{n}.pth"
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = self.pretrained_model_path / "adam.pth"
        if optimizer_load_path.is_file():
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    # ---------------------------------------------------------------------------------------------------------------------

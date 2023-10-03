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

from colon_nav.net_train.loss_terms import compute_pose_loss
from colon_nav.net_train.md2_transforms import poses_to_md2_format
from colon_nav.net_train.shared_transforms import sample_to_gpu
from colon_nav.net_train.train_utils import DatasetMeta
from colon_nav.util.general_util import create_folder_if_not_exists, to_str
from colon_nav.util.torch_util import get_device
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
from monodepth2.networks.pose_net import PoseNet
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
    no_ssim: bool = False  # if set, disables ssim in the loss
    pretrained_model_path: str = None  # name of model to load
    models_to_load: tuple = ("encoder", "depth", "pose")  # models to load
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
    n_ref_imgs: int = attrs.field(init=False)  # frame ids within each sample (target = 0, reference = 1, 2, ...)
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
    # The time shifts of the reference frames w.r.t. the target frame:
    ref_frame_shifts: list = attrs.field(init=False)
    epoch: int = 0
    step: int = 0
    start_time: float = 0
    depth_encoder: ResnetEncoder = attrs.field(init=False)
    depth_decoder: DepthDecoder = attrs.field(init=False)
    pose_net: PoseNet = attrs.field(init=False)
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
        self.ref_frame_shifts = self.train_dataset_meta.ref_frame_shifts
        # number of frames in each sample (target + reference frames):
        self.n_ref_imgs = self.train_dataset_meta.n_ref_imgs
        assert self.n_ref_imgs > 0, "n_ref_imgs must be > 0"
        self.models = {}
        self.parameters_to_train = []
        self.device = get_device()
        self.n_scales = len(self.scales)

        # checking height and width are multiples of 32
        assert self.feed_height % 32 == 0, "'height' must be a multiple of 32"
        assert self.feed_width % 32 == 0, "'width' must be a multiple of 32"

        # save params
        with (self.save_path / "params.txt").open("w") as file:
            file.write(str(self) + "\n")

        ### Init models (with ImageNet pre-trained weights)
        # Depth encoder: gets one RGB image and outputs a feature map
        self.depth_encoder = ResnetEncoder(num_layers=self.num_layers, pretrained=True, num_input_images=1).to(
            self.device,
        )
        self.parameters_to_train += list(self.depth_encoder.parameters())
        num_ch_depth_encoder = self.depth_encoder.num_ch_enc
        # Depth decoder: gets the feature map from the encoder and outputs a disparity map
        self.depth_decoder = DepthDecoder(
            num_ch_enc=num_ch_depth_encoder,
            scales=self.scales,
            num_output_channels=1,
        ).to(self.device)
        self.parameters_to_train += list(self.depth_decoder.parameters())

        # Pose network: gets the RGB images of the target and reference frames and outputs the relative pose change from target to reference frames.
        self.pose_net = PoseNet(n_ref_imgs=self.n_ref_imgs, num_layers=self.num_layers).to(self.device)
        self.parameters_to_train += list(self.pose_net.parameters())

        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.scheduler_step_size, 0.1)

        if self.pretrained_model_path is not None:
            models = {"encoder": self.depth_encoder, "depth": self.depth_decoder, "pose": self.pose_net}
            self.load_model(models)

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
                # Go over all the reference frames
                for shift in self.ref_frame_shifts:
                    # Get the ground truth pose change from target to reference frame
                    translation_gt, axisangle_gt = poses_to_md2_format(inputs[("tgt_to_ref_pose", shift)])
                    # Get the predicted pose change from reference to target frame
                    trans_pred = outputs[("translation", 0, shift)]
                    rot_pred = outputs[("axisangle", 0, shift)]

                    # Compute the pose losses:
                    trans_loss, rot_loss = compute_pose_loss(
                        trans_pred=trans_pred,
                        trans_gt=translation_gt,
                        rot_pred=rot_pred,
                        rot_gt=axisangle_gt,
                    )
                    trans_loss_tot += trans_loss
                    rot_loss_tot += rot_loss
                trans_loss = trans_loss_tot / self.n_ref_imgs
                rot_loss = rot_loss_tot / self.n_ref_imgs
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
        features = self.depth_encoder(input_imgs)
        outputs = self.depth_decoder(features)

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

        #  we input all frames to the PoseNet to predict the relative pose between the target frame (frame_id=0) and each reference frame (shifts: -n_ref_imgs, ..., -1)

        # Get the pose change from target (frame_id=0) to each reference frame
        ref_imgs = [inputs[("color", shift, 0)] for shift in self.ref_frame_shifts]
        tgt_img = inputs[("color", 0, 0)]
        axisangle_all, translation_all = self.pose_net(ref_imgs=ref_imgs,  tgt_img=tgt_img)

        for i_ref, shift in enumerate(self.ref_frame_shifts):
            # Save the predicted pose change from target (frame_id=0) to reference frame i.
            # Output rotation [batch_size, 3] (axis-angle representation)
            outputs[("axisangle", 0, shift)] = axisangle_all[:,i_ref,0,:]
            # Output translation [batch_size, 3]
            outputs[("translation", 0, shift)] = translation_all[:, i_ref, 0, :]
            outputs[("cam_T_cam", 0, shift)] = transformation_from_parameters(
                axisangle_all[:, i_ref],
                translation_all[:, i_ref],
            )
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

            # Go over all the reference frames (1,2,..) and generate the warped (reprojected) color images that estimate the target frame
            for shift in self.ref_frame_shifts:
                T = outputs["cam_T_cam", 0, shift]
                cam_points = self.backproject_depth[source_scale](depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)
                # The estimated pixel coordinates of the reference frame in the target frame
                outputs[("sample", shift, scale)] = pix_coords
                # Use grid_sample to generate the warped (reprojected) color image
                outputs[("color", shift, scale)] = nnF.grid_sample(
                    input=inputs[("color", shift, source_scale)],
                    grid=outputs[("sample", shift, scale)],
                    padding_mode="border",
                    align_corners=True,
                )

                if not self.disable_automasking:
                    outputs[("color_identity", shift, scale)] = inputs[("color", shift, source_scale)]

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

            # Go over all the reference frames (1,2,..) and compute the reprojection loss between the true target frame and the predicted target frame based on the reference frame
            for shift in self.ref_frame_shifts:
                # Get the estimated target frame based on the reference frame
                pred = outputs[("color", shift, scale)]
                # Compute the reprojection loss between the true target frame and the estimated target frame
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)

            if not self.disable_automasking:
                identity_reprojection_losses = []
                for shift in self.ref_frame_shifts:
                    # Get the estimated target frame based on the reference frame
                    pred = inputs[("color", shift, source_scale)]
                    # Compute the reprojection loss between the true target frame and the estimated target frame
                    identity_reprojection_losses.append(self.compute_reprojection_loss(pred, target))
                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                if self.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

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
                for shift in [*self.ref_frame_shifts, 0]:
                    writer.add_image(
                        f"color_{shift}_{s}/{j}",
                        inputs[("color", shift, s)][j].data,
                        self.step,
                    )
                    if s == 0 and shift != 0:
                        writer.add_image(
                            f"color_pred_{shift}_{s}/{j}",
                            outputs[("color", shift, s)][j].data,
                            self.step,
                        )
                writer.add_image(f"disp_{s}/{j}", normalize_image(outputs[("disp", s)][j]), self.step)
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

    def load_model(self, models):
        """Load model(s) from disk"""
        self.pretrained_model_path = Path(self.pretrained_model_path)

        assert self.pretrained_model_path.is_dir(), f"Cannot find folder {self.pretrained_model_path}"
        print(f"loading model from folder {self.pretrained_model_path}")

        for name in self.models_to_load:
            print(f"Loading {name} weights...")
            path = self.pretrained_model_path / f"{name}.pth"
            model_state = models[name].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_state}
            model_state.update(pretrained_dict)
            models[name].load_state_dict(model_state)

        # loading adam state
        optimizer_load_path = self.pretrained_model_path / "adam.pth"
        if optimizer_load_path.is_file():
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    # ---------------------------------------------------------------------------------------------------------------------

# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from monodepth2.datasets import KITTIOdomDataset
from monodepth2.layers import transformation_from_parameters
from monodepth2.networks.pose_net import PoseNet
from monodepth2.options import GeneralOptions
from monodepth2.utils import readlines


# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):
    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz**2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error**2)) / gtruth_xyz.shape[0]
    return rmse


def evaluate(opt):
    """Evaluate odometry on the KITTI dataset"""
    assert os.path.isdir(opt.load_weights_folder), f"Cannot find a folder at {opt.load_weights_folder}"

    assert opt.eval_split == "odom_9" or opt.eval_split == "odom_10", "eval_split should be either odom_9 or odom_10"

    sequence_id = int(opt.eval_split.split("_")[1])

    filenames = readlines(
        os.path.join(os.path.dirname(__file__), "splits", "odom", f"test_files_{sequence_id:02d}.txt"),
    )

    dataset = KITTIOdomDataset(opt.data_path, filenames, opt.height, opt.width, [0, 1], 4, is_train=False)
    dataloader = DataLoader(
        dataset,
        opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    pose_net = PoseNet(n_ref_imgs=1, num_layers=opt.num_layers)
    pose_net.load_state_dict(torch.load(Path(opt.load_weights_folder) / "pose.pth"))

    pose_net.cuda()
    pose_net.eval()

    pred_poses = []

    print("-> Computing pose predictions")

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input

    with torch.no_grad():
        for inputs in dataloader:
            for key, ipt in inputs.items():
                inputs[key] = ipt.cuda()

            axisangle, translation = pose_net(
                ref_imgs=[inputs[("color_aug", 1, 1)]],
                target_img=inputs["color_aug", 0, 0],
            )
            pred_poses.append(transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())

    pred_poses = np.concatenate(pred_poses)

    gt_poses_path = os.path.join(opt.data_path, "poses", f"{sequence_id:02d}.txt")
    gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    gt_global_poses = np.concatenate((gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    gt_global_poses[:, 3, 3] = 1
    gt_xyzs = gt_global_poses[:, :3, 3]

    gt_local_poses = []
    for i in range(1, len(gt_global_poses)):
        gt_local_poses.append(np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))

    ates = []
    num_frames = gt_xyzs.shape[0]
    track_length = 5
    for i in range(num_frames - 1):
        local_xyzs = np.array(dump_xyz(pred_poses[i : i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i : i + track_length - 1]))

        ates.append(compute_ate(gt_local_xyzs, local_xyzs))

    print(f"\n   Trajectory error: {np.mean(ates):0.3f}, std: {np.std(ates):0.3f}\n")

    save_path = os.path.join(opt.load_weights_folder, "poses.npy")
    np.save(save_path, pred_poses)
    print("-> Predictions saved to", save_path)


if __name__ == "__main__":
    options = GeneralOptions()
    evaluate(options.parse())

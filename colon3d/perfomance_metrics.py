from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from colon3d.general_util import save_plot_and_close
from colon3d.import_from_sim.simulate_tracks import TargetsInfo
from colon3d.keypoints_util import transform_tracks_points_to_cam_frame
from colon3d.slam_alg import SlamOutput
from colon3d.torch_util import np_func, to_numpy
from colon3d.transforms_util import apply_pose_change, find_pose_change

# ---------------------------------------------------------------------------------------------------------------------
""""
* based on:
- "A Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry", Zhang et. al. 2018 https://rpg.ifi.uzh.ch/docs/IROS18_Zhang.pdf
- https://github.com/uzh-rpg/rpg_trajectory_evaluation
- Sect 3.2 of VR-Caps: A Virtual Environment for Capsule Endoscopy İncetana et. al. 2021 https://arxiv.org/abs/2008.12949
- https://github.com/CapsuleEndoscope/VirtualCapsuleEndoscopy/blob/master/Tasks/Pose%20and%20Depth%20Estimation/poseError.py
"""

# ---------------------------------------------------------------------------------------------------------------------


def align_estimated_trajectory(gt_cam_poses: np.ndarray, est_cam_poses: np.ndarray) -> np.ndarray:
    """
    Align the estimated trajectory with the ground-truth trajectory using rigid-body transformation.

    Args:
        gt_poses [N x 7] ground-truth poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
        est_poses [N x 7] estimated poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
    Returns:
        aligned_est_poses [N x 7]
    Notes:
        * As both trajectories can be specified in arbitrary coordinate frames, they first need to be aligned.
        We use rigid-body transformation that maps the estimated trajectory onto the ground truth trajectory such that the first frame are aligned.
                * we assume N < N_gt_frames, i.e. the estimated trajectory is shorter than the ground-truth trajectory.
    """
    n_frames = est_cam_poses.shape[0]
    # find the alignment transformation, according to the first frame
    # rotation
    pose_align = np_func(find_pose_change)(start_pose=est_cam_poses[0], final_pose=gt_cam_poses[0])

    # apply the alignment transformation to the estimated trajectory
    aligned_est_poses = np.zeros_like(est_cam_poses)
    for i in range(n_frames):
        aligned_est_poses[i] = np_func(apply_pose_change)(start_pose=est_cam_poses[i], pose_change=pose_align)
    return aligned_est_poses


# ---------------------------------------------------------------------------------------------------------------------


def compute_ATE(gt_cam_poses: np.ndarray, est_cam_poses: np.ndarray) -> dict:
    """compute the mean and standard deviation ATE (Absolute Trajectory Error) across all frames.
    Args:
        gt_poses [N x 7] ground-truth poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
        est_poses [N x 7] estimated poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
    Notes:
        * As both trajectories can be specified in arbitrary coordinate frames, they first need to be aligned.
            We use rigid-body transformation that maps the predicted trajectory onto the ground truth trajectory such that the first frame are aligned.

        * The ATE per frame is computed as the mean Euclidean distance between the ground-truth and estimated translation vectors.
    """
    n_frames = est_cam_poses.shape[0]

    # align according to the first frame
    est_poses_aligned = align_estimated_trajectory(gt_cam_poses=gt_cam_poses, est_cam_poses=est_cam_poses)

    # compute the ATE per frame

    # The ATE_pose per frame is computed as the Euclidean distance between the ground-truth and aligned estimated translation vectors.
    ate_trans = np.zeros(n_frames)  # [mm]
    # The ATE_rot per frame is computed as the angle of rotation from the estimated-aligned rotation to the ground-truth rotation.
    ate_rot_deg = np.zeros(n_frames)

    for i in range(n_frames):
        # find the pose difference (estimation error) (order does not matter, since we take the absolute value of the change)
        delta_pose = np_func(find_pose_change)(start_pose=gt_cam_poses[i], final_pose=est_poses_aligned[i])
        delta_pose = delta_pose.squeeze()
        delta_loc = delta_pose[:3]
        delta_rot = delta_pose[3:]

        # translation error
        ate_trans[i] = np.linalg.norm(delta_loc)  # [mm]
        # angle of rotation of the unit-quaternion
        ate_rot_deg[i] = np.rad2deg(np.abs(2 * np.arccos(delta_rot[0])))
        # take the 360-degree complement if the angle is greater than 180 degrees
        ate_rot_deg[i] = min(ate_rot_deg[i], 360 - ate_rot_deg[i])

    return {"ate_trans": ("Translation ATE [mm]", ate_trans), "ate_rot_deg": ("Rotation ATE [deg]", ate_rot_deg)}


# ---------------------------------------------------------------------------------------------------------------------


def compute_RPE(gt_poses: np.ndarray, est_poses: np.ndarray) -> dict:
    """compute the mean and standard deviation RPE (Relative Pose Error) across all frames.
    Args:
        gt_poses [N x 7] ground-truth poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
        est_poses [N x 7] estimated poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
    """
    n_frames = est_poses.shape[0]

    # the RPE_trans per-frame:
    rpe_trans = np.zeros(n_frames - 1)

    # the RPE_rot per-frame:
    rpe_rot_deg = np.zeros(n_frames - 1)

    for i in range(n_frames - 1):
        # Find the pose change from the current frame to the next frame according to the ground-truth trajectory
        delta_pose_gt = np_func(find_pose_change)(start_pose=gt_poses[i], final_pose=gt_poses[i + 1])
        # Find the pose change from the current frame to the next frame according to the estimated trajectory
        delta_pose_est = np_func(find_pose_change)(start_pose=est_poses[i], final_pose=est_poses[i + 1])

        # find the relative pose change between the estimated and ground-truth poses (order doesn't matter - since we take the magnitude of the rotation and translation)
        delta_pose = np_func(find_pose_change)(start_pose=delta_pose_gt, final_pose=delta_pose_est)

        delta_pose = delta_pose.squeeze()
        delta_loc = delta_pose[:3]
        delta_rot = delta_pose[3:]
        rpe_trans[i] = np.linalg.norm(delta_loc)  # [mm]
        # The angle of rotation of the unit-quaternion
        rpe_rot_deg[i] = np.rad2deg(np.abs(2 * np.arccos(delta_rot[0])))
        # take the 360-degree complement if the angle is greater than 180 degrees
        rpe_rot_deg[i] = min(rpe_rot_deg[i], 360 - rpe_rot_deg[i])
        

    return {
        "rpe_trans": ("Translation RPE [mm]", rpe_trans),
        "rpe_rot_deg": ("Rotation RPE [deg]", rpe_rot_deg),
    }


# ---------------------------------------------------------------------------------------------------------------------


def calc_nav_aid_metrics(gt_cam_poses: np.ndarray, est_cam_poses: np.ndarray, gt_targets_info: dict, online_est_track_world_loc: list) -> dict:
    """Calculates the navigation-aid metrics.
    Args:
        gt_cam_poses [N x 7] ground-truth camera poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
        est_cam_poses [N x 7] estimated camera poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
    """
    eps = 1e-20  # to avoid division by zero
    n_frames = est_cam_poses.shape[0]

    # Transform to the camera system of each frame (according the estimated camera poses)
    online_est_track_cam_loc = np_func(transform_tracks_points_to_cam_frame)(
        tracks_world_locs=online_est_track_world_loc,
        cam_poses=est_cam_poses,
    )
    
    # the targets are static in world system (same for all frames)
    n_frames_gt = gt_cam_poses.shape[0]
    gt_tracks_world_loc = [{track_id: gt_targets_info.points3d[track_id] for track_id in range(gt_targets_info.n_targets)} for _ in range(n_frames_gt)]
    # find their GT locations in the GT camera system per frame:
    gt_targets_cam_loc = np_func(transform_tracks_points_to_cam_frame)(
        tracks_world_locs=gt_tracks_world_loc,
        cam_poses=gt_cam_poses,
    )
        
    # the navigation-aid metrics per-frame:
    angle_err_deg = np.zeros(n_frames)
    z_err_mm = np.zeros(n_frames)
    
    for i in range(n_frames):
        # the estimated 3d position (in camera system) of the tracked polyps in the seen in the current frame (units: mm)
        track_loc_est_cam = online_est_track_cam_loc[i]
        
        # the GT 3d position of the center KP of the tracked polyps in the seen in the current frame (units: mm)  (in GT camera system)
        track_loc_gt_cam = gt_targets_cam_loc[i]
        
        # go over all ths tracks that have been their location estimated in the cur\rent frame
        for track_id in track_loc_est_cam:
            gt_p3d_cam = track_loc_gt_cam[track_id].reshape(3)  # [mm]
            est_p3d_cam = track_loc_est_cam[track_id].reshape(3)  # [mm]
            
            # The locations on the z-axis of the tracked polyps in the current frame (in camera system)
            gt_z_dist = gt_p3d_cam[2]  # [mm]
            est_z_dist = est_p3d_cam[2]  # [mm]
            
            # the angle in degrees between the z-axis and the ray from the camera center to the tracked polyp
            gt_angle_rad = np.arccos(gt_z_dist / max(np.linalg.norm(gt_p3d_cam), eps))  # [rad]
            est_angle_rad = np.arccos(est_z_dist / max(np.linalg.norm(est_p3d_cam), eps))  # [rad]
            
            gt_angle_deg = np.rad2deg(gt_angle_rad)  # [deg]
            est_angle_deg = np.rad2deg(est_angle_rad)  # [deg]

            angle_err_deg[i] = gt_angle_deg - est_angle_deg  # [deg]
            
            z_err_mm = gt_z_dist - est_z_dist  # [mm]
    return  {"angle_err_deg": ("Angle error [deg]", angle_err_deg), "z_err_mm": ("Z error [mm]", z_err_mm)}


# ---------------------------------------------------------------------------------------------------------------------


def calc_performance_metrics(gt_cam_poses: np.ndarray, gt_targets_info: TargetsInfo, slam_out: SlamOutput) -> dict:
    """Calculate the SLAM performance metrics w.r.t ground truth."""

    # Extract the estimation results from the SLAM output:
    # Note: in each frame we take the online estimation (and not the final post-hoc estimation)

    # Get the online estimated camera poses per frame (in world coordinate)
    est_cam_poses = slam_out.online_logger.cam_pose_estimates
    
    n_frames = est_cam_poses.shape[0] # number of frames in the estimated trajectory
    
    # take the subsets of the GT camera poses that corresponds to the frames in the estimated trajectory
    gt_cam_poses = gt_cam_poses[:n_frames]

    #  List of the estimated 3D locations of each track's KPs (in the world system) per frame
    online_est_track_world_loc = to_numpy(slam_out.online_est_track_world_loc)

    # Compute SLAM metrics
    ate_metrics = compute_ATE(gt_cam_poses=gt_cam_poses, est_cam_poses=est_cam_poses)
    rpe_metrics = compute_RPE(gt_poses=gt_cam_poses, est_poses=est_cam_poses)

    # Compute navigation-aid metrics
    nav_metrics = calc_nav_aid_metrics(
        gt_cam_poses=gt_cam_poses,
        gt_targets_info=gt_targets_info,
        est_cam_poses=est_cam_poses,
        online_est_track_world_loc=online_est_track_world_loc,
    )
    err_metrics = ate_metrics | rpe_metrics | nav_metrics
    return err_metrics


# ---------------------------------------------------------------------------------------------------------------------


def plot_trajectory_metrics(err_metrics: dict, save_path: Path):
    n_metrics = len(err_metrics)
    fig = plt.figure()
    for i_plot, metric_name in enumerate(err_metrics):
        # add subplot
        n_cols = 2
        ax = fig.add_subplot(n_metrics  // n_cols + 1, n_cols, i_plot + 1)
        metric_title, metric_values = err_metrics[metric_name]
        ax.plot(metric_values, label=metric_title)
        ax.set_xlabel('Frame index')
        ax.set_title(metric_title)
        ax.grid(True)
    fig.tight_layout()
    save_plot_and_close(save_path)

# ---------------------------------------------------------------------------------------------------------------------
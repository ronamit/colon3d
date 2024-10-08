from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from colon_nav.data_import.simulate_tracks import TargetsInfo
from colon_nav.slam.tracks_loader import DetectionsTracker, get_track_angle_from_cam_sys_loc
from colon_nav.util.data_util import SceneLoader
from colon_nav.util.general_util import save_current_figure_and_close
from colon_nav.util.perf_metrics_SimCol import calc_sim_col_metrics
from colon_nav.util.pose_transforms import (
    compose_poses,
    find_rigid_registration,
    get_pose_delta,
    transform_tracks_points_to_cam_frame,
)
from colon_nav.util.rotations_util import get_rotation_angles, normalize_quaternions
from colon_nav.util.torch_util import np_func, to_numpy

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
        * we assume N < N_gt_frames, i.e. the estimated trajectory is shorter than the ground-truth trajectory.
    """
    n_frames = est_cam_poses.shape[0]

    # find the rigid transformation that best aligns the estimated trajectory with the ground-truth trajectory
    # I.e find an 6DOF alignment A s.t, P_{est} @ A ~= P_{gt}
    rigid_align = np_func(find_rigid_registration)(
        poses1=est_cam_poses[:n_frames, :],
        poses2=gt_cam_poses[:n_frames, :],
    )

    # apply the alignment transformation to the estimated trajectory
    # I.e., P_{est_aligned} = P_{est} @ inv(A)
    aligned_est_poses = np.zeros_like(est_cam_poses)
    for i in range(n_frames):
        aligned_est_poses[i] = np_func(compose_poses)(pose1=rigid_align, pose2=est_cam_poses[i])

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

    # find the pose difference (estimation error) (order does not matter, since we take the absolute value of the change)
    delta_poses = np_func(get_pose_delta)(pose1=gt_cam_poses[:n_frames], pose2=est_poses_aligned[:n_frames])
    delta_locs = delta_poses[:, :3]
    delta_rots = delta_poses[:, 3:]
    # angle of rotation of the unit-quaternion  [rad] in the range [-pi, pi]
    delta_rots_rad = np_func(get_rotation_angles)(delta_rots)

    # translation error
    ate_trans = np.linalg.norm(delta_locs, axis=-1)  # [mm]

    # take the absolute value of the angle (since the rotation can be clockwise or counter-clockwise)
    ate_rot_deg = np.rad2deg(np.abs(delta_rots_rad))  # [deg]
    # take into account the fact that the rotation is cyclic
    ate_rot_deg = np.minimum(ate_rot_deg, 360.0 - ate_rot_deg)  # [deg]

    metrics_per_frame = {
        "Translation ATE [mm]": ate_trans,
        "Rotation ATE [deg]": ate_rot_deg,
    }

    # Calculate RMSE
    rmse_ate_trans = np.sqrt(np.mean(ate_trans**2))
    rmse_ate_rot_deg = np.sqrt(np.mean(ate_rot_deg**2))
    metrics_stats = {
        "Trans_ATE_RMSE_mm": rmse_ate_trans,
        "Rot_ATE_RMSE_Deg": rmse_ate_rot_deg,
    }

    return metrics_per_frame, metrics_stats


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

    # Find the pose change from the current frame to the next frame according to the ground-truth trajectory
    delta_pose_gt = np_func(get_pose_delta)(pose1=gt_poses[:-1, :], pose2=gt_poses[1:, :])
    # Find the pose change from the current frame to the next frame according to the estimated trajectory
    delta_pose_est = np_func(get_pose_delta)(pose1=est_poses[:-1, :], pose2=est_poses[1:, :])

    # find the relative pose change between the estimated and ground-truth poses (order doesn't matter - since we take the magnitude of the rotation and translation)
    delta_pose = np_func(get_pose_delta)(pose1=delta_pose_gt, pose2=delta_pose_est)
    delta_trans = delta_pose[:, :3]
    delta_rot = delta_pose[:, 3:]
    rpe_trans = np.linalg.norm(delta_trans, axis=-1)  # [mm]
    # The angle of rotation of the unit-quaternion
    delta_rot = np_func(normalize_quaternions)(delta_rot)  # normalize the quaternions (avoid numerical issues)
    delta_rot_rad = np_func(get_rotation_angles)(delta_rot)  # [rad] in the range [-pi, pi]
    # take the absolute value of the angle
    rpe_rot_deg = np.rad2deg(np.abs(delta_rot_rad))  # [deg]
    # take into account the fact that the rotation is cyclic
    rpe_rot_deg = np.minimum(rpe_rot_deg, 360.0 - rpe_rot_deg)  # [deg]

    metrics_per_frame = {
        "Translation RPE [mm]": rpe_trans,
        "Rotation RPE [deg]": rpe_rot_deg,
    }

    # calculate RMSE (root mean squared error) of the RPE
    rmse_rpe_trans = np.sqrt(np.mean(rpe_trans**2))
    rase_rpe_rot_deg = np.sqrt(np.mean(rpe_rot_deg**2))

    metrics_stats = {
        "Trans_RPE_RMSE_mm": rmse_rpe_trans,
        "Rot_RPE_RMSE_Deg": rase_rpe_rot_deg,
    }

    return metrics_per_frame, metrics_stats


# ---------------------------------------------------------------------------------------------------------------------


def calc_nav_aid_metrics(
    gt_cam_poses: np.ndarray,
    est_cam_poses: np.ndarray,
    gt_targets_info: dict,
    online_est_track_cam_loc: list,
    online_est_track_angle: list,
    detections_tracker: DetectionsTracker,
    scene_loader: SceneLoader,
) -> dict:
    """Calculates the navigation-aid metrics.
    Args:
        gt_cam_poses [N x 7] ground-truth camera poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
        est_cam_poses [N x 7] estimated camera poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
        online_est_track_cam_loc: [list per frame of dict per track of 3D location] the estimated track location in the camera frame.
    """
    n_frames = est_cam_poses.shape[0]
    n_targets = gt_targets_info.n_targets
    alg_view_pix_normalizer = scene_loader.alg_view_pix_normalizer
    alg_cam_info = scene_loader.alg_cam_info

    deg_err_thresh = 15  # [deg] the threshold for the angular error to consider a target as "detected"

    # The GT target location in the GT world system is static (same for all frames)
    n_frames_gt = gt_cam_poses.shape[0]
    gt_tracks_world_loc = [
        {track_id: gt_targets_info.points3d[track_id] for track_id in range(n_targets)} for _ in range(n_frames_gt)
    ]
    # Find the GT location in the camera system per frame
    gt_targets_cam_loc = np_func(transform_tracks_points_to_cam_frame)(
        tracks_world_locs=gt_tracks_world_loc,
        cam_poses=gt_cam_poses,
    )

    ### the navigation-aid metrics per-frame:

    # True if the target is seen being tracked the current frame (i.e. it was seen in some past frame and is still being tracked in the current frame)
    is_tracked = np.zeros((n_frames, n_targets), dtype=bool)

    # True if the target track is currently in view of the camera (i.e. it is seen in the current frame)
    is_in_view = np.zeros((n_frames, n_targets), dtype=bool)

    # True if the estimated location of the target is in front of the camera (in z-axis)
    is_front_est = np.zeros((n_frames, n_targets), dtype=bool)

    angle_err_deg = np.ones((n_frames, n_targets)) * np.nan
    z_err_mm = np.ones((n_frames, n_targets)) * np.nan
    # 1 if the sign of the error is different than the sign of the GT z-distance, 0 otherwise
    z_sign_err = np.ones((n_frames, n_targets), dtype=bool) * np.nan

    for i in range(n_frames):
        track_ids_in_frame = detections_tracker.get_tracks_in_frame(i)
        # mark the in-view tracks in the current frame
        for track_id in track_ids_in_frame:
            is_in_view[i, track_id] = track_id in track_ids_in_frame

        # the estimated 3d position (in camera system) of the tracks w.r.t. current estimated cam system (units: mm)
        track_loc_est_cam = online_est_track_cam_loc[i]

        # the GT 3d position of the center KP of the tracks w.r.t. current ground-truth cam system  (units: mm)
        track_loc_gt_cam = gt_targets_cam_loc[i]

        # go over all ths tracks that have been their location estimated in the current frame
        for track_id in track_loc_est_cam:
            is_tracked[i, track_id] = True  # mark the track_id as being tracked in the current frame
            gt_p3d_cam = track_loc_gt_cam[track_id].reshape(3)  # [mm]
            est_p3d_cam = track_loc_est_cam[track_id].reshape(3)  # [mm]
            # The locations on the z-axis of the tracked polyps in the current frame (in camera system)
            gt_z_cam = gt_p3d_cam[2]  # [mm]
            est_z_cam = est_p3d_cam[2]  # [mm]
            z_err_mm[i, track_id] = np.abs(gt_z_cam - est_z_cam)  # [mm]
            z_sign_err[i, track_id] = np.sign(gt_z_cam) != np.sign(est_z_cam)
            if est_z_cam > 0:
                # in this case, the estimated location of the track is in front of the camera (in z-axis)
                is_front_est[i, track_id] = True

                # get the angle of the 2D arrow vector which is the projection of the vector from the camera origin to the camera system XY plane
                gt_angle_rad = get_track_angle_from_cam_sys_loc(
                    track_p3d_cam=gt_p3d_cam,
                    cx=alg_cam_info.cx,
                    cy=alg_cam_info.cy,
                    alg_view_pix_normalizer=alg_view_pix_normalizer,
                )

                # est_angle_rad = np.arctan2(est_p3d_cam[1], est_p3d_cam[0])  # [rad]
                est_angle_rad = online_est_track_angle[i][track_id]  # [rad]

                gt_angle_deg = np.rad2deg(gt_angle_rad)  # [deg] in the range [-180, 180]
                est_angle_deg = np.rad2deg(est_angle_rad)  # [deg] in the range [-180, 180]

            angle_abs_diff = np.abs(gt_angle_deg - est_angle_deg)  # [deg]
            # take into account the fact that the angle is cyclic
            angle_err_deg[i, track_id] = np.minimum(angle_abs_diff, 360 - angle_abs_diff)  # [deg] in the range [0, 180]

    # for each frame calculate the average absolute error over the tracked targets
    z_err_mm_per_frame = np.ones(n_frames) * np.nan
    z_sign_err_per_frame = np.ones(n_frames) * np.nan
    angle_err_deg_per_frame = np.ones(n_frames) * np.nan

    # we draw the navigation-aid arrow when the track went out of the algorithm view, and is estimated to be in front of the camera
    is_out_of_view = is_tracked & ~is_in_view
    is_nav_arrow = is_out_of_view & is_front_est

    for i in range(n_frames):
        # for each frame calculate the average absolute error over the tracked targets
        if np.any(is_out_of_view[i]):
            z_err_mm_per_frame[i] = np.nanmean(np.abs(z_err_mm[i][is_out_of_view[i]]))
            z_sign_err_per_frame[i] = np.nanmean(np.abs(z_sign_err[i][is_out_of_view[i]]))
        # for angle error - consider only tracks that went out of view and are estimated to be in front of the camera
        if np.any(is_nav_arrow[i]):
            angle_err_deg_per_frame[i] = np.nanmean(np.abs(angle_err_deg[i][is_nav_arrow[i]]))

    # calculate the RMSE over frames (only frames in which the target went out of view are considered)
    if np.any(is_out_of_view[:]):
        z_err_mm_rmse = np.sqrt(np.nanmean(z_err_mm_per_frame**2))
        z_sign_err_ratio = np.nanmean(z_sign_err_per_frame)
    else:
        print("No targets went out of view !!!!!")
        z_err_mm_rmse = np.nan
        z_sign_err_ratio = np.nan

    if np.any(is_nav_arrow[:]):
        # calculate the percentage of arrows in frames in which the angle error is less than a threshold
        large_angle_err_ratio = np.mean(angle_err_deg_per_frame > deg_err_thresh)
        angle_err_deg_rmse = np.sqrt(np.nanmean(angle_err_deg_per_frame**2))
    else:
        large_angle_err_ratio = np.nan
        angle_err_deg_rmse = np.nan

    metrics_per_frame = {
        "Nav_Angle_Err_Deg": angle_err_deg_per_frame,
        "Nav_Z_Err_mm": z_err_mm_per_frame,
        "Nav_Z_Sign_Err": z_sign_err_per_frame,
    }
    metrics_stats = {
        "Nav_Z_Err_RMSE_mm": z_err_mm_rmse,
        "Nav_Z_Sign_Err_Percent": 100 * z_sign_err_ratio,
        "Nav_Arrow_Err_RMSE_Deg": angle_err_deg_rmse,
        f"Nav_Arrow_Err_Over_{deg_err_thresh}_Deg_Percent": large_angle_err_ratio * 100,
    }
    return metrics_per_frame, metrics_stats


# ---------------------------------------------------------------------------------------------------------------------


def calc_performance_metrics(
    gt_cam_poses: np.ndarray,
    gt_targets_info: TargetsInfo | None,
    slam_out: dict,
    scene_loader: SceneLoader,
) -> dict:
    """Calculate the SLAM performance metrics w.r.t ground truth."""

    # Extract the estimation results from the SLAM output:
    # Note: in each frame we take the online estimation (and not the final post-hoc estimation)

    # Get the online estimated camera poses per frame (in world coordinate)
    est_cam_poses = slam_out["online_logger"].cam_pose_estimates

    n_frames = est_cam_poses.shape[0]  # number of frames in the estimated trajectory

    # take the subsets of the GT camera poses that corresponds to the frames in the estimated trajectory
    gt_cam_poses = gt_cam_poses[:n_frames]

    # ensure the rotations unit quaternion are normalized
    gt_cam_poses[:, 3:] = np_func(normalize_quaternions)(gt_cam_poses[:, 3:])

    #  List of the estimated 3D locations of each track's KPs (in the camera system) per frame
    online_est_track_cam_loc = to_numpy(slam_out["online_est_track_cam_loc"])
    online_est_track_angle = to_numpy(slam_out["online_est_track_angle"])

    # Compute SLAM metrics
    ate_metrics_per_frame, ate_metrics_stats = compute_ATE(gt_cam_poses=gt_cam_poses, est_cam_poses=est_cam_poses)
    rpe_metrics_per_frame, rpe_metrics_stats = compute_RPE(gt_poses=gt_cam_poses, est_poses=est_cam_poses)

    sim_col_metrics = calc_sim_col_metrics(
        gt_cam_poses=gt_cam_poses,
        est_cam_poses=est_cam_poses,
        is_synthetic=True,
    )

    metrics_per_frame = ate_metrics_per_frame | rpe_metrics_per_frame
    metrics_stats = ate_metrics_stats | rpe_metrics_stats | sim_col_metrics

    # Compute navigation-aid metrics, if targets info is available
    if gt_targets_info is not None:
        nav_metrics_per_frame, nav_metrics_stats = calc_nav_aid_metrics(
            gt_cam_poses=gt_cam_poses,
            gt_targets_info=gt_targets_info,
            est_cam_poses=est_cam_poses,
            online_est_track_cam_loc=online_est_track_cam_loc,
            online_est_track_angle=online_est_track_angle,
            detections_tracker=slam_out["detections_tracker"],
            scene_loader=scene_loader,
        )
        # add the navigation-aid metrics
        metrics_per_frame = metrics_per_frame | nav_metrics_per_frame
        metrics_stats = metrics_stats | nav_metrics_stats
    metrics_stats["Num. frames"] = n_frames
    return metrics_per_frame, metrics_stats


# ---------------------------------------------------------------------------------------------------------------------


def plot_trajectory_metrics(metrics_per_frame: dict, save_path: Path):
    n_metrics = len(metrics_per_frame)
    fig = plt.figure()
    for i_plot, metric_name in enumerate(metrics_per_frame):
        # add subplot
        n_cols = 2
        ax = fig.add_subplot(n_metrics // n_cols + 1, n_cols, i_plot + 1)
        metric_values = metrics_per_frame[metric_name]
        ax.plot(metric_values, label=metric_name)
        ax.set_xlabel("Frame index")
        ax.set_title(metric_name)
        ax.grid(True)
    fig.tight_layout()
    save_current_figure_and_close(save_path)


# --------------------------------------------------------------------------------------------------------------------

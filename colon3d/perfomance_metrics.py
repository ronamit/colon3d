import numpy as np

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
        gt_poses [N_gt_frames x 7] ground-truth poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
        est_poses [N x 7] estimated poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
    Notes:
        * As both trajectories can be specified in arbitrary coordinate frames, they first need to be aligned.
            We use rigid-body transformation that maps the predicted trajectory onto the ground truth trajectory such that the first frame are aligned.

        * The ATE per frame is computed as the mean Euclidean distance between the ground-truth and estimated translation vectors.
        * we assume N < N_gt_frames, i.e. the estimated trajectory is shorter than the ground-truth trajectory, and the error is computed only for the frames where the estimated trajectory is defined.
    """
    n_frames = est_cam_poses.shape[0]

    # align according to the first frame
    est_poses_aligned = align_estimated_trajectory(gt_cam_poses=gt_cam_poses, est_cam_poses=est_cam_poses)

    # compute the ATE per frame

    # The ATE_pose per frame is computed as the Euclidean distance between the ground-truth and aligned estimated translation vectors.
    ate_trans_all = np.zeros(n_frames)  # [mm]
    # The ATE_rot per frame is computed as the angle of rotation from the estimated-aligned rotation to the ground-truth rotation.
    ate_rot_all = np.zeros(n_frames)  # [rad]

    for i in range(n_frames):
        # find the pose difference (estimation error) (order does not matter, since we take the absolute value of the change)
        delta_pose = np_func(find_pose_change)(start_pose=gt_cam_poses[i], final_pose=est_poses_aligned[i])
        delta_pose = delta_pose.squeeze()
        delta_loc = delta_pose[:3]
        delta_rot = delta_pose[3:]
        assert np.allclose(np.linalg.norm(delta_rot), 1), "delta_rot is not a unit-quaternion"
        delta_rot = np.clip(delta_rot, -1, 1)  # to avoid numerical errors
        ate_rot_all[i] = np.abs(2 * np.arccos(delta_rot[0]))  # [rad] (angle of rotation of the unit-quaternion)
        ate_trans_all[i] = np.linalg.norm(delta_loc)  # [mm]

    # To quantify the quality of the whole trajectory, take the average over frames
    ate_trans_mean = np.mean(ate_trans_all)  # [mm]
    ate_trans_std = np.std(ate_trans_all)  # [mm]
    ate_rot_mean = np.mean(ate_rot_all)  # [rad]
    ate_rot_std = np.std(ate_rot_all)  # [rad]
    return {
        "ate_trans_mean": ate_trans_mean,
        "ate_trans_std": ate_trans_std,
        "ate_rot_mean": ate_rot_mean,
        "ate_rot_std": ate_rot_std,
    }


# ---------------------------------------------------------------------------------------------------------------------


def compute_RPE(gt_poses: np.ndarray, est_poses: np.ndarray) -> dict:
    """compute the mean and standard deviation RPE (Relative Pose Error) across all frames.
    Args:
        gt_poses [N_gt_frames x 7] ground-truth poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
        est_poses [N x 7] estimated poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
    Notes:
            * we assume N < N_gt_frames, i.e. the estimated trajectory is shorter than the ground-truth trajectory, and the error is computed only for the frames where the estimated trajectory is defined.
    """
    n_frames = est_poses.shape[0]

    # the RPE_trans per-frame:
    rpe_trans_all = np.zeros(n_frames - 1)

    # the RPE_rot per-frame:
    rpe_rot_all = np.zeros(n_frames - 1)

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
        rpe_trans_all[i] = np.linalg.norm(delta_loc)  # [mm]
        rpe_rot_all[i] = np.abs(2 * np.arccos(delta_rot[0]))  # [rad] (angle of rotation of the unit-quaternion)

    # To quantify the quality of the whole trajectory, take the average over frames
    rpe_trans_mean = np.mean(rpe_trans_all)
    rpe_trans_std = np.std(rpe_trans_all)
    rpe_rot_mean = np.mean(rpe_rot_all)
    rpe_rot_std = np.std(rpe_rot_all)
    return {
        "rpe_trans_mean": rpe_trans_mean,
        "rpe_trans_std": rpe_trans_std,
        "rpe_rot_mean": rpe_rot_mean,
        "rpe_rot_std": rpe_rot_std,
    }


# ---------------------------------------------------------------------------------------------------------------------


def calc_nav_aid_metrics(est_cam_poses: np.ndarray, online_est_track_world_loc: list) -> dict:
    """Calculates the navigation-aid metrics.
    Args:
        est_cam_poses [N x 7] estimated camera poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
    """
    eps = 1e-20  # to avoid division by zero
    n_frames = est_cam_poses.shape[0]

    # Transform to the camera system of each frame (according the estimated camera poses)
    online_est_track_cam_loc = np_func(transform_tracks_points_to_cam_frame)(
        online_est_track_world_loc,
        est_cam_poses,
    )

    for i in range(n_frames):
        # the estimated 3d position (in camera system) of the tracked polyps in the seen in the current frame (units: mm)
        tracks_kps_loc_est = online_est_track_cam_loc[i]
        # go over all ths tracks that have been their location estimated in the cur\rent frame
        for _track_id, cur_track_kps_loc_est in tracks_kps_loc_est.items():
            # the estimated 3d position of the center KP of the current track in the current frame (units: mm)  (in est. camera system)
            p3d_cam = cur_track_kps_loc_est[0] # [mm]
            z_dist = p3d_cam[2]  # [mm]
            # compute  the track position angle with the z axis
            ray_dist = np.linalg.norm(p3d_cam[0:3], axis=-1)  # [mm]
            angle_rad = np.arccos(z_dist / max(ray_dist, eps))  # [rad]
            angle_deg = np.rad2deg(angle_rad)  # [deg]
            print("angle:", angle_deg)

    return {}


# ---------------------------------------------------------------------------------------------------------------------


def calc_performance_metrics(gt_poses: np.ndarray, slam_out: SlamOutput) -> dict:
    """Calculate the SLAM performance metrics w.r.t ground truth."""

    # Extract the estimation results from the SLAM output:
    # Note: in each frame we take the online estimation (and not the final post-hoc estimation)

    # Get the online estimated camera poses per frame (in world coordinate)
    est_cam_poses = slam_out.online_logger.cam_pose_estimates

    #  List of the estimated 3D locations of each track's KPs (in the world system) per frame
    online_est_track_world_loc = to_numpy(slam_out.online_est_track_world_loc)

    # Compute SLAM metrics
    ate_metrics = compute_ATE(gt_cam_poses=gt_poses, est_cam_poses=est_cam_poses)
    rpe_metrics = compute_RPE(gt_poses=gt_poses, est_poses=est_cam_poses)

    # Compute navigation-aid metrics
    calc_nav_aid_metrics(
        est_cam_poses=est_cam_poses,
        online_est_track_world_loc=online_est_track_world_loc,
    )

    return ate_metrics | rpe_metrics


# ---------------------------------------------------------------------------------------------------------------------

import numpy as np

from colon3d.torch_util import np_func
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


def align_estimated_trajectory(gt_poses: np.ndarray, est_poses: np.ndarray) -> np.ndarray:
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
    """
    n_frames = gt_poses.shape[0]
    # find the alignment transformation, according to the first frame
    # rotation
    pose_align = find_pose_change(start_pose=est_poses[0], final_pose=gt_poses[0])

    # apply the alignment transformation to the estimated trajectory
    aligned_est_poses = np.zeros_like(est_poses)
    for i in range(n_frames):
        aligned_est_poses[i] = apply_pose_change(start_pose=est_poses[i], pose_change=pose_align[i])
    return aligned_est_poses


# ---------------------------------------------------------------------------------------------------------------------


def compute_ATE(gt_poses: np.ndarray, est_poses: np.ndarray):
    """compute the mean and standard deviation ATE (Absolute Trajectory Error) across all frames.
    Args:
        gt_poses [N x 7] ground-truth poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
        est_poses [N x 7] estimated poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
    Notes:
        * As both trajectories can be specified in arbitrary coordinate frames, they first need to be aligned.
            We use rigid-body transformation that maps the predicted trajectory onto the ground truth trajectory such that the first frame are aligned.

        * The ATE per frame is computed as the mean Euclidean distance between the ground-truth and estimated translation vectors.
    """
    n_frames = gt_poses.shape[0]

    # align according to the first frame
    est_poses_aligned = align_estimated_trajectory(gt_poses=gt_poses, est_poses=est_poses)

    # compute the ATE per frame

    # The ATE_pose per frame is computed as the Euclidean distance between the ground-truth and aligned estimated translation vectors.
    ate_trans_all = np.zeros(n_frames)  # [mm]
    # The ATE_rot per frame is computed as the angle of rotation from the estimated-aligned rotation to the ground-truth rotation.
    ate_rot_all = np.zeros(n_frames)  # [rad]

    for i in range(n_frames):
        # find the pose difference (estimation error) (order does not matter, since we take the absolute value of the change)
        delta_pose = find_pose_change(start_pose=est_poses_aligned[i], final_pose=gt_poses[i])
        delta_loc = delta_pose[:3]
        delta_rot = delta_pose[3:]
        ate_rot_all[i] = np.abs(2 * np.arccos(delta_rot[0]))  # [rad] (angle of rotation of the unit-quaternion)
        ate_trans_all[i] = np.linalg.norm(delta_loc)  # [mm]

    # To quantify the quality of the whole trajectory, take the average over frames
    ate_trans_avg = np.mean(ate_trans_all)  # [mm]
    ate_rot_avg = np.mean(ate_rot_all)  # [rad]
    return ate_trans_avg, ate_rot_avg


# ---------------------------------------------------------------------------------------------------------------------


def compute_RPE(gt_poses: np.ndarray, est_poses: np.ndarray):
    """compute the mean and standard deviation RPE (Relative Pose Error) across all frames.
    Args:
        gt_poses [N x 7] ground-truth poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
        est_poses [N x 7] estimated poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
    """
    n_frames = gt_poses.shape[0]

    # the RPE_trans per-frame:
    rpe_trans_all = np.zeros(n_frames)

    # the RPE_rot per-frame:
    rpe_rot_all = np.zeros(n_frames)

    for i in range(n_frames - 1):
        # find the relative pose change between the estimated and ground-truth poses (order doesn't matter - since we take the magnitude of the rotation and translation)
        delta_pose = np_func(find_pose_change)(start_pose=est_poses[i], final_pose=gt_poses[i])
        delta_loc = delta_pose[:3]
        delta_rot = delta_pose[3:]
        rpe_trans_all[i] = np.linalg.norm(delta_loc)  # [mm]
        rpe_rot_all[i] = np.abs(2 * np.arccos(delta_rot[0]))  # [rad] (angle of rotation of the unit-quaternion)

    # To quantify the quality of the whole trajectory, take the average over frames
    rpe_trans_avg = np.mean(rpe_trans_all)
    rpe_rot_avg = np.mean(rpe_rot_all)
    return rpe_trans_avg, rpe_rot_avg


# ---------------------------------------------------------------------------------------------------------------------


def calc_trajectory_err_metrics(gt_poses: np.ndarray, est_poses: np.ndarray) -> dict:
    """Calculate the SLAM performance metrics w.r.t ground truth tracks info.
    Args:
        slam_out: the SLAM output
        gt_targets_info: the ground truth tracks info
    Returns:
        metrics: the metrics
    """
    ate_trans_avg, ate_rot_avg = compute_ATE(gt_poses=gt_poses, est_poses=est_poses)
    rpe_trans_avg, rpe_rot_avg = compute_RPE(gt_poses=gt_poses, est_poses=est_poses)
    return {
        "ate_trans_avg": ate_trans_avg,
        "ate_rot_avg": ate_rot_avg,
        "rpe_trans_avg": rpe_trans_avg,
        "rpe_rot_avg": rpe_rot_avg,
    }


# ---------------------------------------------------------------------------------------------------------------------

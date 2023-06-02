import numpy as np

from colon3d.rotations_util import (
    infer_egomotions_np,
    invert_rotation_np,
    quaternion_raw_multiply_np,
    rotate_np,
    transform_between_poses_np,
)

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
    Align the estimated trajectory with the ground-truth trajectory.

    Args:
        gt_poses [N x 7] ground-truth poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
        est_poses [N x 7] estimated poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
    Returns:
        aligned_est_poses [N x 7]
    Notes:
        * As both trajectories can be specified in arbitrary coordinate frames, they first need to be aligned.
        We use rigid-body transformation that maps the predicted trajectory onto the ground truth trajectory such that the first frame are aligned.
    """
    n_frames = gt_poses.shape[0]
    # find the alignment transformation, according to the first frame
    # rotation
    gt_rot_0 = gt_poses[0, 3:][np.newaxis, :]
    est_rot_0 = est_poses[0, 3:][np.newaxis, :]
    est_rot_0_inv = invert_rotation_np(est_rot_0)
    align_rot = quaternion_raw_multiply_np(gt_rot_0, est_rot_0_inv)
    # translation
    gt_trans_0 = gt_poses[0, :3]
    est_trans_0 = est_poses[0, :3]
    align_trans = gt_trans_0 - rotate_np(points3d=est_trans_0, rot_vecs=align_rot)

    # apply the alignment transformation to the estimated trajectory
    aligned_est_poses = np.zeros_like(est_poses)
    for i in range(n_frames):
        aligned_est_poses[i, 3:] = quaternion_raw_multiply_np(align_rot, est_poses[i, 3:])
        aligned_est_poses[i, :3] = rotate_np(points3d=est_poses[i, :3], rot_vecs=align_rot) + align_trans

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
    gt_trans = gt_poses[:, :3]
    gt_rot = gt_poses[:, 3:]
    # align according to the first frame
    est_poses_aligned = align_estimated_trajectory(gt_poses=gt_poses, est_poses=est_poses)
    est_trans_aligned = est_poses_aligned[:, :3]
    est_rot_aligned = est_poses_aligned[:, 3:]

    # compute the ATE per frame

    # The ATE_pose per frame is computed as the Euclidean distance between the ground-truth and aligned estimated translation vectors.
    ate_trans_all = np.zeros(n_frames)  # [mm]
    # The ATE_rot per frame is computed as the angle of rotation from the estimated-aligned rotation to the ground-truth rotation.
    ate_rot_all = np.zeros(n_frames)  # [rad]

    for i in range(n_frames):
        # delta_rot = gt_rot * (est_rot_aligned)^-1
        delta_rot = quaternion_raw_multiply_np(gt_rot[i], invert_rotation_np(est_rot_aligned[i].unsqueeze(0)))
        ate_rot_all[i] = np.abs(2 * np.arccos(delta_rot[0]))  # [rad] (angle of rotation of the unit-quaternion)
        delta_trans = gt_trans[i] - rotate_np(points3d=est_trans_aligned[i], delta_rot=gt_rot[i])
        ate_trans_all[i] = np.linalg.norm(delta_trans)  # [mm]

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
    gt_egomotions = infer_egomotions_np(gt_poses)[1:]
    est_egomotions = infer_egomotions_np(est_poses)[1:]

    # the RPE_trans per-frame:
    rpe_trans_all = np.zeros(n_frames)

    # the RPE_rot per-frame:
    rpe_rot_all = np.zeros(n_frames)

    for i in range(n_frames - 1):
        rel_egomotion = transform_between_poses_np(gt_egomotions[i], est_egomotions[i])
        rel_trans = rel_egomotion[:3]
        rel_quant = rel_egomotion[3:]
        rpe_trans_all[i] = np.linalg.norm(rel_trans)  # [mm]
        rpe_rot_all[i] = np.abs(2 * np.arccos(rel_quant[0]))  # [rad] (angle of rotation of the unit-quaternion)

    # To quantify the quality of the whole trajectory, take the average over frames
    rpe_trans_avg = np.mean(rpe_trans_all)
    rpe_rot_avg = np.mean(rpe_rot_all)
    return rpe_trans_avg, rpe_rot_avg


# ---------------------------------------------------------------------------------------------------------------------


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

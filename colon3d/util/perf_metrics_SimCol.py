import numpy as np

from colon3d.util.pose_transforms import (
    get_relative_4x4_poses,
    poses_to_4x4_matrices,
)

# ---------------------------------------------------------------------------------------------------------------------
""""
Compute the SLAM performance metrics defined in the SimCol3D challenge.

Sources:
* https://www.researchgate.net/publication/372547675_SimCol3D_--_3D_Reconstruction_during_Colonoscopy_Challenge
* https://github.com/anitarau/simcol/blob/main/evaluation/eval_synthetic_pose.py
* https://github.com/anitarau/simcol/blob/main/SimCol_evaluation_guide.pdf
"""

# ---------------------------------------------------------------------------------------------------------------------


def calc_sim_col_metrics(gt_cam_poses: np.ndarray, est_cam_poses: np.ndarray, is_synthetic: bool) -> np.ndarray:
    """
    Compute the SLAM performance metrics defined in the SimCol3D challenge.

    Args:
        gt_poses [N x 7] ground-truth poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
        est_poses [N x 7] estimated poses per frame (in world coordinate) where first 3 coordinates are x,y,z [mm] and the rest are unit-quaternions (real part first)
    Returns:
        metrics [dict] a dictionary with the following metrics:
            * ATE [cm] Absolute Trajectory Error
            * RTE [cm] Relative Trajectory Error
            * ROT [deg] Rotation Error
    Notes:
        * we assume N < N_gt_frames, i.e. the estimated trajectory is shorter than the ground-truth trajectory.
    """
    assert is_synthetic, "Not implemented for real data yet."
    n_frames = est_cam_poses.shape[0]
    gt_cam_poses = gt_cam_poses[:n_frames, :]

    # convert the poses to 4x4 pose matrices per frame (in world coordinate) where the translation coordinates are in [cm].
    gt_abs_poses = poses_to_4x4_matrices(gt_cam_poses)
    pred_abs_poses = poses_to_4x4_matrices(est_cam_poses)
    # convert the translation from mm to cm:
    gt_abs_poses[:, :3, -1] /= 10
    pred_abs_poses[:, :3, -1] /= 10

    # Get the relative poses:
    gt_rel_poses = get_relative_4x4_poses(gt_abs_poses)
    preds = get_relative_4x4_poses(pred_abs_poses)

    scale = get_scale(np.array(gt_rel_poses), np.array(preds))

    pred_abs_poses[:, :3, -1] = pred_abs_poses[:, :3, -1] * scale
    ATE, RTE, errs, ROT, gt_rot_mag = compute_errors(gt_abs_poses, pred_abs_poses)
    print("------------------------")
    print(f"Scale {scale:8.4f}")
    print(f"ATE {ATE:10.4f} cm")
    print(f"RTE {RTE:10.4f} cm")
    print(f"ROT {ROT:10.4f} degrees")
    return {"ATE_SimCol_cm": ATE, "RTE_SimCol_cm": RTE, "ROT_SimCol_deg": ROT}


# ---------------------------------------------------------------------------------------------------------------------


def get_scale(gt, pred):
    scale_factor = np.sum(gt[:, :3, -1] * pred[:, :3, -1]) / np.sum(pred[:, :3, -1] ** 2)
    return scale_factor


# ---------------------------------------------------------------------------------------------------------------------


def compute_errors(gt, pred, delta=1):
    errs = []
    rot_err = []
    rot_gt = []
    trans_gt = []
    for i in range(pred.shape[0] - delta):
        Q = np.linalg.inv(gt[i, :, :]) @ gt[i + delta, :, :]
        P = np.linalg.inv(pred[i, :, :]) @ pred[i + delta, :, :]
        E = np.linalg.inv(Q) @ P
        t = E[:3, -1]
        t_gt = Q[:3, -1]
        trans = np.linalg.norm(t, ord=2)
        errs.append(trans)
        tr = np.arccos((np.trace(E[:3, :3]).clip(-3, 3) - 1) / 2)
        gt_tr = np.arccos((np.trace(Q[:3, :3]) - 1) / 2)
        rot_err.append(tr)
        rot_gt.append(gt_tr)
        trans_gt.append(np.linalg.norm(t_gt, ord=2))

    errs = np.array(errs)

    ATE = np.median(np.linalg.norm((gt[:, :, -1] - pred[:, :, -1]), ord=2, axis=1))
    RTE = np.median(errs)
    ROT = np.median(rot_err)

    return ATE, RTE, errs, ROT * 180 / np.pi, np.mean(rot_gt) * 180 / np.pi


# ---------------------------------------------------------------------------------------------------------------------

import os
import time

import numpy as np
import torch
import torchmin  # https://github.com/rfeinman/pytorch-minimize # type: ignore  # noqa: PGH003

from colon_nav.slam.alg_settings import AlgSettings
from colon_nav.slam.constraints_terms import SoftConstraints
from colon_nav.slam.keypoints_util import KeyPointsLog
from colon_nav.util.general_util import print_if
from colon_nav.util.pose_transforms import project_world_to_image_normalized_coord
from colon_nav.util.rotations_util import find_rotation_delta, get_rotation_angle, normalize_quaternions
from colon_nav.util.torch_util import concat_list_to_tensor, get_device, get_val, is_finite, pseudo_huber_loss_on_x_sqr

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # prevent cuda out of memory error

# --------------------------------------------------------------------------------------------------------------------


def compute_cost_function(
    x: torch.Tensor,
    cam_poses: torch.Tensor,
    points_3d: torch.Tensor,
    frames_opt_flag: np.ndarray,
    p3d_opt_flag: np.ndarray,
    kp_frame_idx_u: np.ndarray,
    kp_p3d_idx_u: np.ndarray,
    kp_nrm_u: torch.Tensor,
    kp_weights_u: torch.Tensor,
    n_cam_poses_opt: int,
    n_points_3d_opt: int,
    penalizer: SoftConstraints,
    alg_prm: AlgSettings,
    scene_metadata: dict,
):
    """
    Compute the cost function for the given optimization vector x.
    (re-projection errors + additional penalties)
    Args:
        x = torch.cat([cam_poses_opt, points_3d_opt]), where cam_poses_opt is a flattened vector the of the optimized camera poses per optimized frame
            where each cam_pose has 7 elements (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy, qz) is the unit-quaternion of the rotation
            abd points_3d_opt is a flattened vector of the optimized 3D points (each point has x,y,z elements) (units: mm).
        cam_poses [n_frames x 7] = [x, y, z, q0, qx, qy, qz] where (x, y, z) is the translation [mm] and  (q0, qx, qy, qz) is the unit-quaternion of the rotation
        points_3d [n_points_3d x 3] = [x, y, z] (units: mm) current 3D points estimate
    Returns:
        cost:  the cost function value for the given optimization vector x.
    """
    assert torch.isfinite(input=x).all(), "x is not finite"
    fx = scene_metadata["fx"]
    fy = scene_metadata["fy"]

    frames_opt_inds = np.where(frames_opt_flag)[0].tolist()
    cam_poses_opt = x[: n_cam_poses_opt * 7].reshape(n_cam_poses_opt, 7)
    points_3d_opt = x[n_cam_poses_opt * 7 :].reshape(n_points_3d_opt, 3)
    cam_loc_opt = cam_poses_opt[:, 0:3]
    cam_rot_opt = cam_poses_opt[:, 3:7]

    # normalize the rotation parameters to get a standard unit-quaternion (the optimization steps may take the parameters out of the domain)
    cam_rot_opt = normalize_quaternions(cam_rot_opt)

    # The full (all frames) camera poses tensor where we substitute the currently optimized (in the optimized frames elements)
    cam_poses_full_opt = cam_poses.clone()
    for i, frame_idx in enumerate(frames_opt_inds):
        cam_poses_full_opt[frame_idx, 0:3] = cam_loc_opt[i]
        cam_poses_full_opt[frames_opt_flag, 3:7] = cam_rot_opt[i]

    # the current 3d points are the optimized ones, except for the ones that are not optimized
    points_3d_c = points_3d.clone()
    points_3d_c[p3d_opt_flag] = points_3d_opt

    # take the subsets that are used in the optimization objective:
    cam_poses_per_kp = cam_poses_full_opt[kp_frame_idx_u]
    point3d_per_kp = points_3d_c[kp_p3d_idx_u]

    # Project the 3D points to to normalized coordinates in the undistorted image system (focal length is 1 and the optical center is at (0,0))
    projected_point_nrm = project_world_to_image_normalized_coord(
        points3d_world=point3d_per_kp,
        cam_poses=cam_poses_per_kp,
    )

    # re-projection error between the normalized coordinates of the projected 3D points and the normalized coordinates of the keypoints
    reproject_diff = projected_point_nrm - kp_nrm_u  # [dim: n_kps x 2] normalized coordinates.
    # convert the re-projection error to pixel units  [n_kps x 2]
    reproject_diff_pix = reproject_diff * torch.tensor([fx, fy], device=get_device())
    reproject_dists_sqr = torch.sum(torch.square(reproject_diff_pix), dim=1)  # [dim: n_kps] [pix^2]

    # compute the weighted sum of re-projection errors
    cost_kps = torch.sum(pseudo_huber_loss_on_x_sqr(reproject_dists_sqr, delta=1.0) * kp_weights_u)
    # cost_kps = torch.sum(torch.square(projected_point_nrm - kp_nrm_u) * kp_weights_u.unsqueeze(1))

    # add a log-barrier penalties for violating the bounds on the camera poses rate of change
    penalty_lim_angular_vel = 0
    penalty_lim_vel = 0
    penalty_cam_rot = 0
    penalty_cam_trans = 0
    penalty_p3d_change = 0
    if alg_prm.add_penalties:
        for i, frame_idx in enumerate(frames_opt_inds):
            cur_loc = cam_loc_opt[i]  # the estimated 3D cam location
            cur_rot = cam_rot_opt[i]  # the estimated unit-quaternion of the cam rotation

            prev_frame_idx = frame_idx - 1
            # the estimated cam-pose parameters the frame before the i-th optimized frame:
            # note: we use for prev_pose the pre-optimization values, for stability reasons
            prev_pose = cam_poses[prev_frame_idx, 0:7]
            prev_loc = prev_pose[0:3]  # the estimated 3D cam location
            prev_rot = prev_pose[3:7]  # the estimated unit-quaternion of the cam rotation

            # the L2 distance between the i-th optimized frame from the previous frame
            cam_trans_sqr = torch.sum(torch.square(cur_loc - prev_loc))

            # penalty term of the l2 norm of the camera translation (from previous frame)
            penalty_cam_trans += cam_trans_sqr * alg_prm.w_cam_trans

            # penalty term of the 3D points change (from initial values)
            penalty_p3d_change + alg_prm.w_p3d_change * torch.sum(torch.square(points_3d_opt - points_3d[p3d_opt_flag]))

            # add a log-barrier penalty for violating the max position change bound
            penalty_lim_vel += penalizer.get_penalty(constraint_name="lim_vel", val=cam_trans_sqr)

            # Get the rotation change angle between the current frame and the previous frame [rad]
            rot_change = find_rotation_delta(rot1=prev_rot, rot2=cur_rot)
            rot_change_angle = get_rotation_angle(rot_change)

            # penalty term of the camera rotation (from previous frame)
            penalty_cam_rot += rot_change_angle * alg_prm.w_cam_rot

            # add a log-barrier penalty for violating the max angle change bound:
            penalty_lim_angular_vel += penalizer.get_penalty(
                constraint_name="lim_angular_vel",
                val=rot_change_angle,
            )

        # log the cost components
        cost_components = {
            "cost_kps": cost_kps,
            "penalty_max_vel": penalty_lim_vel,
            "penalty_max_angular_vel": penalty_lim_angular_vel,
            "penalty_cam_rot": penalty_cam_rot,
            "penalty_cam_trans": penalty_cam_trans,
            "penalty_p3d_change": penalty_p3d_change,
        }
    else:
        cost_components = {
            "cost_kps": cost_kps,
        }
    # sum the cost components
    tot_cost = 0
    for cost_name, cost_term in cost_components.items():
        if is_finite(cost_term):
            tot_cost += cost_term
        else:
            raise Warning(f"cost term is not finite: {cost_name}={cost_term}")

    return tot_cost, cost_components, reproject_dists_sqr


# --------------------------------------------------------------------------------------------------------------------


def run_bundle_adjust(
    cam_poses: torch.Tensor,
    points_3d: torch.Tensor,
    kp_log: KeyPointsLog,
    p3d_inds_per_frame: list,
    frames_inds_to_opt: list,
    earliest_frame_to_use: int,
    alg_prm: AlgSettings,
    fps: float,
    scene_metadata: dict,
    verbose: int = 2,
    print_now: bool = False,
):
    """
    Run bundle adjustment on the given data
    Args:
        cam_poses: saves per-frame of the 6DOF camera pose arrays of size 7 (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy , qz) is the unit-quaternion of the rotation.
        points_3d: saves of 3d points in world coordinates  [List of arrays of size 3] (units: mm)
        p3d_inds_in_frame: list of lists of 3d point indexes for each frame [List of size n_frames of list of lengths n_points_in_frame]
        frames_inds_to_opt: the frame indexes to set the optimization variables (cam poses and 3D points)
        earliest_frame_to_use: the first frame index to use for the loss terms, if -1, then use all history
        verbose: verbosity level
    Returns:
        cam_poses_opt: camera poses after bundle adjustment optimization [saves per-frame of arrays of size 7]
        points_3d_opt:  3d points after bundle adjustment optimization [saves per-point of arrays of size 3]
    """
    n_3d_pts = points_3d.shape[0]
    n_frames = cam_poses.shape[0]

    device = get_device()
    time_interval = 1 / fps  # sec
    max_cam_trans = time_interval * alg_prm.max_vel
    max_cam_trans_sqr = max_cam_trans**2
    max_angle_change = time_interval * alg_prm.max_angular_vel

    # create flag vector of the frame indexes and p3d indexes we are optimizing now
    p3d_opt_flag = np.zeros(n_3d_pts, dtype=bool)
    frames_opt_flag = np.zeros(n_frames, dtype=bool)
    for i_frame in frames_inds_to_opt:
        frames_opt_flag[i_frame] = True
        # find what 3d points to optimize - those that are seen from frames in frames_inds_to_opt
        p3d_opt_flag[list(p3d_inds_per_frame[i_frame])] = True

    # Find what KP to use in the optimized loss function: all KPs that are associated with the optimized frames or the optimized world-3D points
    kp_opt_ids = kp_log.get_kp_ids_in_frames_or_p3d_ids(
        frame_inds=frames_inds_to_opt,
        p3d_flag_vec=p3d_opt_flag,
        earliest_frame_to_use=earliest_frame_to_use,
    )

    # take the subsets that are used in the optimization objective:
    if len(kp_opt_ids) == 0:
        print_if(print_now, "No keypoints to be used in the optimization objective... skipping optimization")
        return cam_poses, points_3d, kp_log, 0

    kp_frame_idx_u = concat_list_to_tensor([kp_id[0] for kp_id in kp_opt_ids], num_type="int")
    kp_p3d_idx_u = concat_list_to_tensor([kp_log.get_kp_p3d_idx(kp_id) for kp_id in kp_opt_ids], num_type="int")
    kp_nrm_u = concat_list_to_tensor([kp_log.get_kp_norm_coord(kp_id) for kp_id in kp_opt_ids]).to(device)
    n_kp_used = kp_nrm_u.shape[0]
    kp_type_u = concat_list_to_tensor([kp_log.get_kp_type(kp_id) for kp_id in kp_opt_ids], num_type="int").to(device)
    kp_weights_u = torch.ones(n_kp_used, device=device)
    kp_weights_u[kp_type_u == -1] = alg_prm.w_salient_kp
    kp_weights_u[kp_type_u != -1] = alg_prm.w_track_kp

    # take the subset of the vectors that participate in the optimization objective:
    cam_poses_opt = torch.as_tensor(cam_poses[frames_opt_flag]).requires_grad_()
    points_3d_opt = torch.as_tensor(points_3d[p3d_opt_flag]).requires_grad_()
    n_cam_poses_opt = cam_poses_opt.shape[0]
    n_points_3d_opt = points_3d_opt.shape[0]
    print_if(
        print_now,
        f"Optimizing cam-pose of {n_cam_poses_opt} frames, and {n_points_3d_opt} 3D-points, using {n_kp_used} keypoints.",
    )

    # Set constraints on the optimization
    constraints = {}
    constraints["lim_vel"] = {"lim_type": "upper", "lim": max_cam_trans_sqr, "weight": alg_prm.w_lim_vel}
    constraints["lim_angular_vel"] = {
        "lim_type": "upper",
        "lim": max_angle_change,
        "weight": alg_prm.w_lim_angular_vel,
        "margin_ratio": 0.001,
    }
    penalizer = SoftConstraints(constraints)

    # set the initial value of the optimization parameters
    x0 = torch.cat((cam_poses_opt.flatten(), points_3d_opt.flatten()))

    # create the cost function
    cost_fun_kwargs = {
        "cam_poses": cam_poses,
        "points_3d": points_3d,
        "frames_opt_flag": frames_opt_flag,
        "p3d_opt_flag": p3d_opt_flag,
        "kp_frame_idx_u": kp_frame_idx_u,
        "kp_p3d_idx_u": kp_p3d_idx_u,
        "kp_nrm_u": kp_nrm_u,
        "kp_weights_u": kp_weights_u,
        "n_cam_poses_opt": n_cam_poses_opt,
        "n_points_3d_opt": n_points_3d_opt,
        "penalizer": penalizer,
        "alg_prm": alg_prm,
        "scene_metadata": scene_metadata,
    }

    def cost_function(x: torch.Tensor) -> torch.Tensor:
        cost, _, _ = compute_cost_function(
            x=x,
            **cost_fun_kwargs,
        )
        return cost

    # run the optimization
    print_if(print_now, f"Running optimization for frames: {frames_inds_to_opt}...")
    t0 = time.time()
    if alg_prm.opt_method in {"bfgs", "l-bfgs", "cg", "newton-cg", "newton-exact", "trust-ncg"}:
        result = torchmin.minimize(
            fun=cost_function,
            x0=x0,
            method=alg_prm.opt_method,
            max_iter=alg_prm.opt_max_iter,
            tol=alg_prm.opt_x_tol,
            disp=verbose,
            options={"xtol": alg_prm.opt_x_tol, "gtol": alg_prm.opt_g_tol, "line_search": alg_prm.opt_line_search},
        )
    else:
        raise ValueError(f"Unknown opt_method: {alg_prm.opt_method}")
    print_if(print_now, f"Optimization took {time.time() - t0:.1f} seconds")
    with torch.no_grad():
        tot_cost, cost_components, reproject_dists_sqr = compute_cost_function(result.x, **cost_fun_kwargs)
        # mark the salient keypoints (type==-1) that have a large re-projection error as invalid
        kp_reproject_err_threshold = alg_prm.kp_reproject_err_threshold
        is_kp_invalid = (reproject_dists_sqr > kp_reproject_err_threshold**2) & (kp_type_u == -1)
        is_kp_invalid = is_kp_invalid.cpu().numpy()
        print_if(print_now, f"Total final cost: {get_val(tot_cost):1.6g}")
        print_if(
            print_now,
            "Cost components: " + ", ".join([f"{k}={get_val(v):1.6g}" for k, v in cost_components.items()]),
        )

    # Discard the invalid keypoints
    n_invalid_kps = is_kp_invalid.sum()
    print_if(print_now, f"Number of invalid keypoints to discard: {n_invalid_kps}")
    kp_ids_to_discard = [kp_id for i_kp, kp_id in enumerate(kp_opt_ids) if is_kp_invalid[i_kp]]
    kp_log.discard_keypoints(kp_ids_to_discard)

    cam_poses_optimized = result.x[: n_cam_poses_opt * 7].reshape(n_cam_poses_opt, 7).detach()
    points_3d_optimized = result.x[n_cam_poses_opt * 7 :].reshape(n_points_3d_opt, 3).detach()

    # normalize the rotation parameters to get a standard unit-quaternion (the optimization steps may take the parameters out of the domain)
    cam_poses_optimized[:, 3:7] = normalize_quaternions(cam_poses_optimized[:, 3:7])

    points_3d[p3d_opt_flag] = points_3d_optimized
    cam_poses[frames_opt_flag] = cam_poses_optimized

    return cam_poses, points_3d, kp_log, n_invalid_kps


# --------------------------------------------------------------------------------------------------------------------

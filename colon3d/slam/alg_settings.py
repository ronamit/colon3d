from dataclasses import dataclass

import numpy as np


@dataclass
class AlgorithmParam:
    n_last_frames_to_opt: int = 1
    optimize_each_n_frames: int = 1
    add_penalties = True  # if True, calculate and add penalties to the cost function of the bundle adjustment (otherwise, only the reprojection error is used)
    max_vel: float = 150  # mm/s
    max_angular_vel: float = 2.0 * np.pi  # rad/s
    w_salient_kp: float = 0.3  # default weighting for the bundle adjustment cost function for salient KPs
    w_track_kp: float = 1.0  # default  weighting for the bundle adjustment cost function for track KPs
    w_cam_trans: float = 1e-6  # default weighting for the penalty term of the l2 norm of the camera translation (from previous frame) in the bundle adjustment cost function
    w_cam_rot: float = 1e-5  # default weighting for the penalty term of the camera rotation (from previous frame) in the bundle adjustment cost function
    w_p3d_change: float = 1e-1  # default weighting for the penalty term of the change in the 3D points in the bundle adjustment cost function
    # default weighting for the max-velocity penalty term in the bundle adjustment cost function:
    w_lim_vel: float = 1e-3
    # default weighting for the max-angular-velocity penalty term in the bundle adjustment cost function:
    w_lim_angular_vel: float = 1e-3
    depth_upper_bound: float = 2000  # upper bound to clip the the z-depth estimation (units: mm)
    depth_lower_bound: float = 0.0  # lower bound to clip the the z-depth estimation (units: mm)
    depth_default: float = 5  # default z-depth estimation, when no other data is available (units: mm)
    opt_method: str = "bfgs"  #  Optimization method, options:  "bfgs", "l-bfgs", "cg", "newton-cg",  "newton-exact", "trust-ncg", "trust-krylov", "trust-exact", "trust-constr"]
    opt_max_iter: int = 100  # maximum number of iterations for the optimization
    opt_x_tol: float = 1e-3  # Optimization termination tolerance on function/parameter changes.
    opt_g_tol: float = 1e-6  # Optimization termination tolerance on the gradient.
    # opt_lr: float = 10.  # initial learning rate for the optimization
    detect_bb_kps_ratios = np.array([(0, 0)])  # , (-0.1, 0), (0.1, 0), (0, -0.1), (0, +0.1)],
    # note: do not make the ratio 0.25 or larger, otherwise boxes that are on the edge of the view might cause wrong KP matchings

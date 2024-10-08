from dataclasses import dataclass

import numpy as np


@dataclass
class AlgSettings:
    use_bundle_adjustment: bool = True
    n_last_frames_to_opt: int = 1  # number of last frames to set camera poses and 3D points as optimization variables
    # number of last frames to use for the bundle adjustment loss term, if -1, then use all history:
    n_last_frames_to_use: int = -1
    optimize_each_n_frames: int = 1  # optimize each n frames
    add_penalties = False  # if True, calculate and add penalties to the cost function of the bundle adjustment (otherwise, only the reprojection error is used)
    max_vel: float = 200  # mm/s
    max_angular_vel: float = 3.0 * np.pi  # rad/s
    w_salient_kp: float = 0.01  # default weighting for the bundle adjustment cost function for salient KPs
    w_track_kp: float = 1.0  # default  weighting for the bundle adjustment cost function for track KPs
    w_cam_trans: float = 1e-5  # default weighting for the penalty term of the l2 norm of the camera translation (from previous frame) in the bundle adjustment cost function
    w_cam_rot: float = 1e-4  # default weighting for the penalty term of the camera rotation (from previous frame) in the bundle adjustment cost function
    w_p3d_change: float = 1e-3  # default weighting for the penalty term of the change in the 3D points in the bundle adjustment cost function
    # default weighting for the max-velocity penalty term in the bundle adjustment cost function:
    w_lim_vel: float = 1e-2
    # default weighting for the max-angular-velocity penalty term in the bundle adjustment cost function:
    w_lim_angular_vel: float = 1e-2
    depth_default: float = 15  # default z-depth estimation, when no other data is available (units: mm)
    opt_method: str = "bfgs"  #  Optimization method, options:  "bfgs", "l-bfgs", "cg", "newton-cg",  "newton-exact", "trust-ncg", "trust-krylov", "trust-exact", "trust-constr"]
    opt_max_iter: int = 500  # maximum number of iterations for the optimization
    opt_x_tol: float = 1e-4  # Optimization termination tolerance on function/parameter changes.
    opt_g_tol: float = 1e-7  # Optimization termination tolerance on the gradient.
    # opt_lr: float = 10.  # initial learning rate for the optimization
    opt_line_search = "strong-wolfe"  # line search method for the optimization, options: none, strong-wolfe (slow)
    ransac_reprojection_err_threshold: float = 2.0  # reprojection error threshold for RANSAC (units: pixels)
    max_initial_keypoints: int = 300  # maximum number of keypoints the KPs detector will return.
    kp_descriptor_patch_size: int = 51  # patch size for the keypoint descriptor (units: pixels)
    kp_descriptor_edge_threshold: int = 26  # edge threshold for the keypoint descriptor (units: pixels)
    min_n_matches_to_filter: int = 10  # minimum number of keypoints matches to filter out outliers.
    # maximum distance from the identity matrix to consider a homography as valid
    hom_dist_from_identity_threshold: float = 50.0
    # maximum distance between two keypoints to consider them as a match (units: pixels)
    max_match_pix_dist: float = 30.0
    orb_fast_thresh: int = 5  # FAST threshold for ORB keypoints detector
    # reprojection error threshold for marking keypoints as "invalid" (units: pixels)
    kp_reproject_err_threshold: float = 5.0
    # if True then use the trivial solution yo navigation aid - if track goes out of view, use the last navigation angle:
    use_trivial_nav_aid: bool = False
    max_bundle_adjustment_repeats: int = 3  # maximum number of bundle adjustment repeats

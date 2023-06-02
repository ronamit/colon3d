import cv2
import numpy as np
import torch

from colon3d.rotations_util import invert_rotation, rotate

# --------------------------------------------------------------------------------------------------------------------


@torch.jit.script  # disable this for debugging
def project_world_to_normalized_coord(
    cur_points_3d: torch.Tensor,
    cur_cam_poses: torch.Tensor,
) -> torch.Tensor:
    """Convert 3-D points to 2-D by projecting onto images.
    assumes camera parameters set a rectilinear image transform from 3d to 2d (i.e., fisheye undistorting was done)
    Args:
        cur_points_3d (torch.Tensor): [n_points x 3] (units: mm)
        cur_cam_poses (torch.Tensor): [n_points x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy , qz) is the unit-quaternion of the rotation.
    Returns:
        points_2d (torch.Tensor): [n_points x 2]  (units: normalized image coordinates)
    """
    assert cur_points_3d.shape[1] == 3, f"Points are not in 3D, {cur_points_3d.shape}."
    assert cur_cam_poses.shape[1] == 7, f"Cam poses are not in 7D, {cur_cam_poses.shape}."
    # Rotate & translate to camera system
    eps = 1e-20
    cam_loc = cur_cam_poses[:, 0:3]  # [n_points x 3]
    cam_rot = cur_cam_poses[:, 3:7]  # [n_points x 4]
    inv_cam_rot = invert_rotation(cam_rot)  # [n_points x 4]
    points_cam_sys = rotate(cur_points_3d - cam_loc, inv_cam_rot)
    # Perspective transform to 2d image-plane
    # (this transforms to normalized image coordinates, i.e., fx=1, fy=1, cx=0, cy=0)
    z_cam_sys = points_cam_sys[:, 2]  # [n_points x 1]
    x_wrt_axis = points_cam_sys[:, 0] / (z_cam_sys + eps)  # [n_points x 1]
    y_wrt_axis = points_cam_sys[:, 1] / (z_cam_sys + eps)  # [n_points x 1]
    points_2d_nrm = torch.stack((x_wrt_axis, y_wrt_axis), dim=1)  # [n_points x 2]
    return points_2d_nrm


# --------------------------------------------------------------------------------------------------------------------


def unproject_normalized_coord_to_world(
    points_nrm: torch.Tensor,
    z_depth: torch.Tensor,
    cam_poses: torch.Tensor,
) -> torch.Tensor:
    """Transforms a normalized coordinate and a given z depth to a 3D point in world coordinates.
    Args:
        points_nrm (torch.Tensor): [n_points x 2] (units: normalized image coordinates)
        z_depth (np.ndarray): [n_points] (units: mm)
        cam_poses (torch.Tensor): [n_points x 7]  [n_points x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy, qz) is the unit-quaternion of the rotation.
    Returns:
        points_3d (np.ndarray): [n_points x 3]   (units: mm)
    """
    assert points_nrm.shape[1] == 2, f"Points are not in 2D, {points_nrm.shape}."
    assert cam_poses.shape[1] == 7, f"Cam poses are not in 7D, {cam_poses.shape}."
    cam_loc = cam_poses[:, 0:3]  # [n_points x 3] (units: mm)
    cam_rot = cam_poses[:, 3:7]  # [n_points x 4]

    # normalized coordinate corresponds to fx=1, fy=1, cx=0, cy=0, so we can just multiply by z_depth to get 3d point in the camera system
    z_cam_sys = z_depth
    x_cam_sys = points_nrm[:, 0] * z_depth
    y_cam_sys = points_nrm[:, 1] * z_depth
    points_3d_cam_sys = torch.stack((x_cam_sys, y_cam_sys, z_cam_sys), dim=1)
    #  translate & rotate to world system
    points_3d = cam_loc + rotate(points_3d_cam_sys, cam_rot)
    return points_3d


# --------------------------------------------------------------------------------------------------------------------


def unproject_normalized_coord_to_world_np(
    points_nrm: np.ndarray,
    z_depth: np.ndarray,
    cam_poses: np.ndarray,
) -> np.ndarray:
    """Transforms a normalized coordinate and a given z depth to a 3D point in world coordinates.
    Args:
        points_nrm (np.ndarray): [n_points x 2] (units: normalized image coordinates)
        z_depth  (np.ndarray): [n_points] (units: mm)
        cam_poses (torch.Tensor): [n_points x 7]  [n_points x 7] each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy, qz) is the unit-quaternion of the rotation.
    Returns:
        points_3d  (np.ndarray): [n_points x 3]   (units: mm)
    """
    return unproject_normalized_coord_to_world(
        torch.from_numpy(points_nrm),
        torch.from_numpy(z_depth),
        torch.from_numpy(cam_poses),
    ).numpy()


# --------------------------------------------------------------------------------------------------------------------


def get_normalized_pixels_np(pixels_x: np.ndarray, pixels_y: np.ndarray, cam_K: np.ndarray) -> np.ndarray:
    """Transforms pixel coordinates to normalized image coordinates.
    Assumes the camera is rectilinear with a given K matrix.
        pixels_x (np.ndarray): [n_points] (units: pixels)
        pixels_y (np.ndarray): [n_points] (units: pixels)
        cam_K: [3 x 3] camera intrinsics matrix (we assume it is of the form [fx, 0, cx; 0, fy, cy; 0, 0, 1])
    """

    x = (pixels_x - cam_K[0, 2]) / cam_K[0, 0]
    y = (pixels_y - cam_K[1, 2]) / cam_K[1, 1]
    points_nrm = np.stack((x, y), axis=1)
    return points_nrm


# --------------------------------------------------------------------------------------------------------------------
def get_frame_point_cloud(z_depth_frame: np.ndarray, K_of_depth_map: np.ndarray, cam_pose: np.ndarray):
    """Returns a point cloud for a given depth map and camera pose. The point cloud is in the world coordinate system.
        Note: the order of the points is by (y,x) and not (x,y).
    """
    frame_width, frame_height = z_depth_frame.shape
    n_pix = frame_width * frame_height

    # find the world coordinates that each pixel in the depth map corresponds to:
    pixels_y, pixels_x = np.meshgrid(np.arange(frame_height), np.arange(frame_width))
    # notice that the depth image coordinates are (y,x) not (x,y).
    z_depths = z_depth_frame.flatten()
    pixels_x = pixels_x.flatten()
    pixels_y = pixels_y.flatten()

    points_nrm = get_normalized_pixels_np(pixels_x=pixels_x, pixels_y=pixels_y, cam_K=K_of_depth_map)
    points3d = unproject_normalized_coord_to_world_np(
        points_nrm=points_nrm,
        z_depth=z_depths,
        cam_poses=np.tile(cam_pose, (n_pix, 1)),
    )
    return points3d


# --------------------------------------------------------------------------------------------------------------------


def kp_to_p3d(kp_per_frame, map_kp_to_p3d_idx, points_3d):
    """
    Transform keypoints to corresponding estimated 3d points
    Parameters:
        kp_per_frame: list of list of keypoints
        map_kp_to_p3d_idx: dict {(frame_idx, kp_x, kp_y): p3d_idx}
        points_3d: array of 3d points
    Return:
        p3d_per_frame: list of array of 3d points
    """
    n_frames = len(kp_per_frame)
    p3d_per_frame = []
    for i_frame in range(n_frames):
        p3d_per_frame.append(None)
        kp_list = kp_per_frame[i_frame]
        p3d_inds = [map_kp_to_p3d_idx[(i_frame, kp[0], kp[1])] for kp in kp_list]
        if p3d_inds:
            p3d_per_frame[i_frame] = points_3d[p3d_inds, :]
    return p3d_per_frame


# --------------------------------------------------------------------------------------------------------------------


def transform_tracks_points_to_cam_frame(tracks_kps_world_loc_per_frame: list, cam_poses: torch.Tensor) -> list:
    """
    Transform 3D points of from world system to the camera system per frame
    Parameters:
        tracks_kps_world_loc_per_frame: list per frame of dict with key=track_id, value = Tensor[,n_points, 3]  array of 3D coordinates of the track's KPs (in world system) (units: mm)
        cam_poses: Tensor[n_frames ,7], each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy, qz) is the unit-quaternion of the rotation.
    Return:
        cam_p3d_per_frame_per_track: transformed to the camera system (units: mm)
    """
    assert cam_poses.shape[1] == 7, f"Cam poses are not in 7D, {cam_poses.shape}."
    n_frames = cam_poses.shape[0]
    cam_p3d_per_frame_per_track = [{} for _ in range(n_frames)]
    for i_frame in range(n_frames):
        world_tracks_p3d = tracks_kps_world_loc_per_frame[i_frame]
        for track_id, track_world_p3d in world_tracks_p3d.items():
            n_points = track_world_p3d.shape[0]
            # Rotate & translate to camera view
            cams_locs = cam_poses[i_frame, 0:3].expand(n_points, 3)
            cams_rots = cam_poses[i_frame, 3:7].expand(n_points, 4)
            inv_cam_rpts = invert_rotation(cams_rots)
            track_cam_p3d = rotate(track_world_p3d - cams_locs, inv_cam_rpts)
            cam_p3d_per_frame_per_track[i_frame][track_id] = track_cam_p3d
    return cam_p3d_per_frame_per_track


# --------------------------------------------------------------------------------------------------------------------


def get_tracks_keypoints(tracks_in_frame, alg_prm):
    """
    Args:
        curr_bb: current bounding box
    Returns:
        polyps_kps keypoints of the polyp in the bounding box
    """
    detect_bb_kps_ratios = alg_prm.detect_bb_kps_ratios
    tracks_kps_in_frame = {}
    for track_id, track_info in tracks_in_frame.items():
        x_min = track_info["xmin"]
        y_min = track_info["ymin"]
        x_max = track_info["xmax"]
        y_max = track_info["ymax"]
        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2
        x_len = x_max - x_min
        y_len = y_max - y_min
        kps = [(x_mid + x_ratio * x_len, y_mid + y_ratio * y_len) for (x_ratio, y_ratio) in detect_bb_kps_ratios]
        tracks_kps_in_frame[track_id] = kps
    return tracks_kps_in_frame


# --------------------------------------------------------------------------------------------------------------------


def get_kp_matchings(
    keypoints_A,
    keypoints_B,
    descriptors_A,
    descriptors_B,
    kp_matcher,
    min_n_matches_to_filter=10,
):
    """
    Parameters:
        keypoints_A: list of keypoints in frame A
        keypoints_B: list of keypoints in frame B
        descriptors_A: list of descriptors for the KPs in frame A
        descriptors_B: list of descriptors for the KPs in frame B
        kp_matcher: keypoints matcher
        min_n_matches_to_filter: minimum number of matches to use RANSAC filter
    """
    if len(keypoints_A) == 0 or len(keypoints_B) == 0:
        print("No keypoints to match...")
        return [], []
    matches = kp_matcher.knnMatch(descriptors_A, descriptors_B, k=1)
    # list of matches from the first image to the second one, which means that kp1[i] has
    # a corresponding point in kp2[matches[i]] .
    # we used k=1 (take only the best match)- so each element is a tuple of size 1..
    # Each match element is
    # DMatch.distance - Distance between descriptors. The lower, the better it is.
    # DMatch.trainIdx - Index of the descriptor in train descriptors (des2)
    # DMatch.queryIdx - Index of the descriptor in query descriptors (des1)
    matches = [m[0] for m in matches if m]
    n_matches = len(matches)
    if n_matches > min_n_matches_to_filter:
        print(f"Matched {n_matches} salient keypoint pairs...")
        # If enough matches are found, we extract the locations of matched keypoints in both the images.
        # They are passed to find the perspective transformation. Once we get this 3x3 transformation matrix,
        src_pts = np.float32([keypoints_A[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_B[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M_hom, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        good_matches = [match for i_match, match in enumerate(matches) if matches_mask[i_match]]
        print(f"After RANSAC, we kept only {len(good_matches)} good matches ...")
    else:
        good_matches = matches
        print(f"Too few matches for RANSAC {len(good_matches)}/{min_n_matches_to_filter}, using all matches...")
        matches_mask = None

    matched_A_kps = []
    matched_B_kps = []
    for match in good_matches:
        matched_A_kps.append(keypoints_A[match.queryIdx].pt)
        matched_B_kps.append(keypoints_B[match.trainIdx].pt)
    return matched_A_kps, matched_B_kps


# --------------------------------------------------------------------------------------------------------------------

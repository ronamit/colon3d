import cv2
import numpy as np
import torch

from colon3d.utils.torch_util import get_default_dtype
from colon3d.utils.transforms_util import transform_points_in_world_sys_to_cam

np_dtype = get_default_dtype("numpy")
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


def transform_tracks_points_to_cam_frame(tracks_world_locs: list, cam_poses: torch.Tensor) -> list:
    """
    Transform 3D points of from world system to the camera system per frame.
    Parameters:
        tracks_world_locs: list per frame of dict with key=track_id, value = 3D coordinates of the track's KP (in world system) (units: mm)

        cam_poses: Tensor[n_frames ,7], each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, qx, qy, qz) is the unit-quaternion of the rotation.
    Return:
        cam_p3d_per_frame_per_track: transformed to the per-frame camera system (units: mm)
    """
    n_frames = cam_poses.shape[0]
    if cam_poses.ndim == 1:
        # if only one camera pose is given, repeat it for all frames
        cam_poses = cam_poses.expand(n_frames, -1)
    cam_p3d_per_frame_per_track = [{} for _ in range(n_frames)]
    for i_frame in range(n_frames):
        cam_pose = cam_poses[i_frame]
        for track_id, track_world_p3d in tracks_world_locs[i_frame].items():
            # Rotate & translate to camera view (of the current frame camera pose)
            track_cam_p3d = transform_points_in_world_sys_to_cam(
                points_3d_world=track_world_p3d,
                cam_poses=cam_pose,
            )
            cam_p3d_per_frame_per_track[i_frame][track_id] = track_cam_p3d
    return cam_p3d_per_frame_per_track


# --------------------------------------------------------------------------------------------------------------------


def get_tracks_keypoints(tracks_in_frame):
    """
    Args:
        curr_bb: current bounding box
    Returns:
        polyps_kps keypoints of the polyp in the bounding box
    """
    tracks_kps_in_frame = {}
    for track_id, track_info in tracks_in_frame.items():
        if "target_x_pix" in track_info and np.isfinite(track_info["target_x_pix"]):
            # in this case we have the pixel coordinates of the target physical center point:
            kp = (track_info["target_x_pix"], track_info["target_y_pix"])
        else:
            # in this case we have the pixel coordinates of target, so we need to calculate the center point of the bounding box:
            x_min = track_info["xmin"]
            y_min = track_info["ymin"]
            x_max = track_info["xmax"]
            y_max = track_info["ymax"]
            x_mid = (x_min + x_max) / 2
            y_mid = (y_min + y_max) / 2
            kp = (x_mid, y_mid)
        tracks_kps_in_frame[track_id] = kp
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
        src_pts = np.array([keypoints_A[m.queryIdx].pt for m in matches], dtype=np_dtype).reshape(-1, 1, 2)
        dst_pts = np.array([keypoints_B[m.trainIdx].pt for m in matches], dtype=np_dtype).reshape(-1, 1, 2)
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
        kp_A = np.round(keypoints_A[match.queryIdx].pt).astype(int)
        kp_B = np.round(keypoints_B[match.trainIdx].pt).astype(int)
        matched_A_kps.append(kp_A)
        matched_B_kps.append(kp_B)
    return matched_A_kps, matched_B_kps


# --------------------------------------------------------------------------------------------------------------------

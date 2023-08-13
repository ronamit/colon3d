import cv2
import numpy as np
import torch

from colon3d.slam.alg_settings import AlgorithmParam
from colon3d.util.general_util import to_str
from colon3d.util.torch_util import get_default_dtype
from colon3d.util.transforms_util import transform_points_world_to_cam

np_dtype = get_default_dtype("numpy")

# --------------------------------------------------------------------------------------------------------------------

class KeyPointsLog:
    """ Saves the keypoint information.
    Note: invalid keypoints are discarded.
    """
    def __init__(self, pix_normalizer) -> None:
        self.map_kp_to_p3d_idx = {} # maps a keypoint (frame_idx, x, y,) its 3D point index
        self.map_kp_to_type = {} # maps a keypoint (frame_idx, x, y,) its type (-1 indicates a salient keypoint, >=0 indicates the track id of the keypoint)
        self.pix_normalizer = pix_normalizer # pixel normalizer object
        
    # --------------------------------------------------------------------------------------------------------------------
    
    def is_kp_coord_valid(self, pix_coord):
        # check if the KP is too close to the image border, and so its undistorted coordinates are invalid
        nrm_coords, is_valid = self.pix_normalizer.get_normalized_coords(pix_coord)
        return is_valid
    # --------------------------------------------------------------------------------------------------------------------

    def add_kp(self, frame_idx, pix_coord, kp_type, p3d_id):
        """ Add a keypoint to the log.
        Args:
            frame_idx: frame index of the keypoint
            pix_coord: pixel coordinates of the keypoint (units: pixels)
            kp_type: keypoint type (-1 indicates a salient keypoint, >=0 indicates the track id of the keypoint)
            p3d_id: 3D point index of the keypoint
        """
        if not self.is_kp_coord_valid(pix_coord):
            return False
        kp_id = (frame_idx, pix_coord[0], pix_coord[1])
        self.map_kp_to_p3d_idx[kp_id] = p3d_id
        self.map_kp_to_type[kp_id] = kp_type
        return True
    # --------------------------------------------------------------------------------------------------------------------

    def get_kp_p3d_idx(self, kp_id: tuple):
        if kp_id in self.map_kp_to_p3d_idx:
            return self.map_kp_to_p3d_idx[kp_id]
        return None
    
    # --------------------------------------------------------------------------------------------------------------------
    
    def get_kp_norm_coord(self, kp_id: tuple):
        """ Get the normalized coordinates of the given keypoint.
        """
        assert isinstance(kp_id, tuple) and len(kp_id) == 3
        norm_coord, is_valid = self.pix_normalizer.get_normalized_coords(kp_id[1:])
        return norm_coord[0]
    # --------------------------------------------------------------------------------------------------------------------

    def get_kp_type(self, kp_id: tuple):
        if kp_id in self.map_kp_to_type:
            return self.map_kp_to_type[kp_id]
        return None
    

    # --------------------------------------------------------------------------------------------------------------------

    def get_kp_ids_in_frame_inds(self, frame_inds: list) -> list:
        """ Get all the keypoint ids in the given frames.
        Args:
            frame_inds: list of frame indexes
        """
        kp_ids = []
        frame_inds_set = set(frame_inds)
        for kp_id in self.map_kp_to_p3d_idx:
            if kp_id[0] in frame_inds_set:
                kp_ids.append(kp_id)
        return kp_ids

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
            track_cam_p3d = transform_points_world_to_cam(
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
    alg_prm: AlgorithmParam,
):
    """
    Parameters:
        keypoints_A: list of keypoints in frame A
        keypoints_B: list of keypoints in frame B
        descriptors_A: list of descriptors for the KPs in frame A
        descriptors_B: list of descriptors for the KPs in frame B
        kp_matcher: keypoints matcher
    """
    min_n_matches_to_filter = alg_prm.min_n_matches_to_filter
    ransac_reprojection_err_threshold = alg_prm.ransac_reprojection_err_threshold
    max_match_pix_dist = alg_prm.max_match_pix_dist

    if len(keypoints_A) == 0 or len(keypoints_B) == 0:
        print("No keypoints to match...")
        return [], []

    matches = kp_matcher.knnMatch(descriptors_A, descriptors_B, k=1)
    # we get list of matches from the first image to the second one, which means that kp1[i] has
    # a corresponding point in kp2[matches[i]] .
    # we used k=1 (take only the best match)- so each element is a tuple of size 1..
    # Each match element is
    # DMatch.distance - Distance between descriptors. The lower, the better it is.
    # DMatch.trainIdx - Index of the descriptor in train descriptors (des2)
    # DMatch.queryIdx - Index of the descriptor in query descriptors (des1)
    matches = [m[0] for m in matches if m]
    n_matches = len(matches)
    # check minimum number of matches to use RANSAC filter
    if n_matches > min_n_matches_to_filter:
        print(f"Matched {n_matches} salient keypoint pairs...")
        # If enough matches are found, we extract the locations of matched keypoints in both the images.
        # They are passed to find the perspective transformation. Once we get this 3x3 transformation matrix,
        src_pts = np.array([keypoints_A[m.queryIdx].pt for m in matches], dtype=np_dtype).reshape(-1, 1, 2)
        dst_pts = np.array([keypoints_B[m.trainIdx].pt for m in matches], dtype=np_dtype).reshape(-1, 1, 2)

        # Create a mask that indicates KP pairs that are too far from each other as outliers
        # the mask is used to filter out outliers from the matches pre-RANSAC
        pre_mask = np.ones((n_matches,1), dtype=np.uint8)
        for i_match in range(n_matches):
            pre_mask[i_match] = np.linalg.norm(src_pts[i_match] - dst_pts[i_match]) < max_match_pix_dist
        pre_mask = pre_mask
        
        M_hom, post_mask = cv2.findHomography(
            srcPoints=src_pts,
            dstPoints=dst_pts,
            method=cv2.RANSAC,
            mask=pre_mask,
            ransacReprojThreshold=ransac_reprojection_err_threshold,
        )
        if M_hom is None or M_hom.shape == ():
            print("RANSAC failed to find a homography...")
            good_matches = []
        else:
            # find the (Frobenius norm) distance of the estimated homography matrix from the identity matrix
            hom_dist_from_identity = np.linalg.norm(M_hom - np.eye(3), ord="fro")
            if hom_dist_from_identity > alg_prm.hom_dist_from_identity_threshold:
                print(
                    f"RANSAC failed to find a homography... (distance from identity is {to_str(hom_dist_from_identity)})",
                )
                good_matches = []
            else:
                matches_mask = post_mask.ravel().tolist()
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

from dataclasses import dataclass

import numpy as np
import pandas as pd

from colon_nav.util.torch_util import get_default_dtype, np_func
from colon_nav.util.pose_transforms import (
    get_frame_point_cloud,
    transform_rectilinear_image_pixel_coords_to_normalized,
    unproject_image_normalized_coord_to_world,
)

# --------------------------------------------------------------------------------------------------------------------


@dataclass
class TargetsInfo:
    n_targets: int
    points3d: np.ndarray  # shape: (n_targets, 3) the 3D points of the targets centers in world coordinate system
    radiuses: np.ndarray  # shape: (n_targets,) the radius of the targets in mm
    init_targ_depth_mm: float  # the depth of the target in the first frame (minimal z in cam sys) [mm]


# --------------------------------------------------------------------------------------------------------------------


def generate_targets(
    n_targets: int,
    gt_depth_maps: np.ndarray,
    gt_cam_poses: np.ndarray,
    rng: np.random.Generator,
    cam_K: np.ndarray,
    cases_params: dict,
) -> TargetsInfo | None:
    """generate random 3D points on the surface of the colon, which will be used as the center of the tracks/"""

    min_target_radius_mm = cases_params["min_target_radius_mm"]
    max_target_radius_mm = cases_params["max_target_radius_mm"]
    min_dist_from_center_ratio = cases_params["min_dist_from_center_ratio"]
    max_dist_from_center_ratio = cases_params["max_dist_from_center_ratio"]
    min_visible_frames = cases_params["min_visible_frames"]
    min_non_visible_frames = cases_params["min_non_visible_frames"]
    min_initial_pixels_in_bb = cases_params["min_initial_pixels_in_bb"]
    max_init_depth_mm = cases_params["max_init_depth_mm"]
    dtype = get_default_dtype("numpy")
    dtype_int = get_default_dtype("numpy", num_type="int")

    n_frames, frame_width, frame_height = gt_depth_maps.shape
    frame_radius = min(frame_width / 2, frame_height / 2)
    target_center_radius_max = max_dist_from_center_ratio * frame_radius
    target_center_radius_min = min_dist_from_center_ratio * frame_radius

    max_attempts = 200  # maximal number of attempts to generate a valid track
    i_attempt = 0

    # we are going to sample the points that are seen from the first frame of the scene:

    while i_attempt < max_attempts:
        i_attempt += 1
        print(f"attempt {i_attempt}/{max_attempts}")

        # we are choose the targets centers from pixels in the first frame
        inspected_frame_idx = 0

        # set the targets centers:
        # randomly sample pixels inside a circle with radius max_radius around the center of the image (per target):
        target_center_radius = rng.uniform(target_center_radius_min, target_center_radius_max, size=n_targets).astype(
            dtype,
        )
        angle = rng.uniform(0, 2 * np.pi, size=n_targets).astype(dtype)
        pixels_x = ((frame_width / 2) + target_center_radius * np.cos(angle)).astype(dtype_int)
        pixels_y = ((frame_height / 2) + target_center_radius * np.sin(angle)).astype(dtype_int)

        # get of the depth target center in the first frame (per target):
        pixels_depth = gt_depth_maps[inspected_frame_idx][pixels_y, pixels_x]

        # ensure that the target is not too far from the camera:
        init_targ_depth_mm = pixels_depth.min()
        print(f"Initial target depth: {init_targ_depth_mm} [mm]]")
        if init_targ_depth_mm > max_init_depth_mm:
            continue

        # get the 3D point in the world coordinate system of the target center (per target):
        targets_centers_nrm = transform_rectilinear_image_pixel_coords_to_normalized(
            pixels_x=pixels_x,
            pixels_y=pixels_y,
            cam_K=cam_K,
        )
        targets_centers_3d = np_func(unproject_image_normalized_coord_to_world)(
            points_nrm=targets_centers_nrm,
            z_depths=pixels_depth,
            cam_poses=gt_cam_poses[inspected_frame_idx],
        )
        # Determine the size of the ball around the target center:
        targets_radiuses = rng.uniform(min_target_radius_mm, max_target_radius_mm, size=n_targets)

        targets_info = TargetsInfo(
            n_targets=n_targets,
            points3d=targets_centers_3d,
            radiuses=targets_radiuses,
            init_targ_depth_mm=init_targ_depth_mm,
        )

        # Ensure the targets are valid:

        tracks = create_tracks_per_frame(
            gt_depth_maps=gt_depth_maps,
            gt_cam_poses=gt_cam_poses,
            cam_K=cam_K,
            targets_info=targets_info,
        )

        initial_pixels_in_bb = np.zeros(
            n_targets,
            dtype=int,
        )  # number of pixels in the bounding box of each target in the first frame
        n_vis_frames_per_target = np.zeros(n_targets, dtype=int)  # number of frames in which the target is visible

        for i_target in range(n_targets):
            targ_tracks = tracks[tracks["track_id"] == i_target]
            # get the number of frames in which the target is visible:
            vis_frames_inds = targ_tracks["frame_idx"].unique()
            n_vis_frames_per_target[i_target] = vis_frames_inds.shape[0]

            # get the number of pixels in the bounding box of each target in the first frame:
            cur_frame_tracks = targ_tracks[targ_tracks["frame_idx"] == 0]
            if len(cur_frame_tracks) > 0:
                initial_pixels_in_bb[i_target] = cur_frame_tracks.n_pix_in_bb.to_numpy()[0]

        # ensure that the tracks are in-view long enough:
        if np.any(n_vis_frames_per_target < min_visible_frames):
            continue

        # ensure that the tracks are outside the field of view long enough:
        n_non_vis_frames_per_target = n_frames - n_vis_frames_per_target
        if np.any(n_non_vis_frames_per_target < min_non_visible_frames):
            continue

        # ensure minimum number of pixels in the bounding box of each target in the first frame:
        if np.any(initial_pixels_in_bb < min_initial_pixels_in_bb):
            continue

        break  # we found a valid target

    if i_attempt == max_attempts:
        print("Could not create valid targets after {max_attempts} attempts")
        return None
    print(f"Found valid targets after {i_attempt} attempts")
    print(f"n_vis_frames_per_target: {n_vis_frames_per_target}")
    return targets_info


# --------------------------------------------------------------------------------------------------------------------


def create_tracks_per_frame(
    gt_depth_maps: np.ndarray,
    gt_cam_poses: np.ndarray,
    cam_K: np.ndarray,
    targets_info: TargetsInfo,
    min_pixels_in_bb: int = 10,
) -> pd.DataFrame:
    """Returns the detections bounding boxes of the tracks in each frame.
    Args:
        gt_depth_maps: the depth maps of the scene
        gt_cam_poses: the camera poses of the scene
        targets_info: a TargetsInfo object containing the targets information
    Returns:
        a dataframe containing the bounding boxes of the tracks in each frame
    """
    targets_p3d_world = targets_info.points3d
    targets_radiuses = targets_info.radiuses
    n_targets = targets_p3d_world.shape[0]
    n_frames, frame_width, frame_height = gt_depth_maps.shape
    # pixel coordinates of all the pixels in the image - we use (y, x) since this is the index order of the depth image
    pixels_y, pixels_x = np.meshgrid(np.arange(frame_height), np.arange(frame_width), indexing="ij")
    pixels_x = pixels_x.flatten()
    pixels_y = pixels_y.flatten()
    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []
    target_x_list = []
    target_y_list = []
    frame_idx_list = []
    track_id_list = []
    n_pix_in_bb_list = []

    for i_frame in range(n_frames):
        cam_pose = gt_cam_poses[i_frame].reshape(1, 7)
        depth_map = gt_depth_maps[i_frame]
        # get the point cloud that the camera views (in world coordinates)
        point_cloud_world = get_frame_point_cloud(z_depth_frame=depth_map, cam_K=cam_K, cam_pose=cam_pose)
        for i_trg in range(n_targets):
            trg_cent_world = targets_p3d_world[i_trg]
            trg_radius = targets_radiuses[i_trg]
            # find the distance of each point in the point cloud from the target center:
            dists_to_trg = np.linalg.norm(point_cloud_world - trg_cent_world, axis=-1)
            # find the pixels that are inside the ball around each target center:
            is_inside = dists_to_trg <= trg_radius
            pixels_x_inside = pixels_x[is_inside]
            pixels_y_inside = pixels_y[is_inside]
            n_inside = np.sum(is_inside)
            if n_inside == 0:
                # if the target is not visible in the current frame
                if i_frame == 0:
                    # we only consider targets that are visible in the first frame
                    # in this case the function will return an empty dataframe (failure)
                    break
                # if the target is not visible in the current frame - just skip this frame
                continue
            n_pix_in_bb = (pixels_x_inside.max() - pixels_x_inside.min()) * (
                pixels_y_inside.max() - pixels_y_inside.min()
            )
            if n_pix_in_bb < min_pixels_in_bb:
                # if the number of pixels in the bounding box is too small - ignore this detection
                continue

            # find bounding box of the pixels that are inside the ball:
            x_min = pixels_x_inside.min()
            y_min = pixels_y_inside.min()
            x_max = pixels_x_inside.max()
            y_max = pixels_y_inside.max()

            # find the pixel coordinate that is closest to the target center:
            pix_ind = np.argmin(dists_to_trg)
            x_trg = pixels_x[pix_ind]
            y_trg = pixels_y[pix_ind]
            target_x_list.append(x_trg)
            target_y_list.append(y_trg)
            # print(f"i_frame: {i_frame}, trg_pos: {(x_trg, y_trg)}, bbbox center: {(x_min + x_max) / 2, (y_min + y_max) / 2}")

            ## create a new record for this track in this frame:
            xmin_list.append(x_min)
            ymin_list.append(y_min)
            xmax_list.append(x_max)
            ymax_list.append(y_max)
            n_pix_in_bb_list.append(n_pix_in_bb)
            frame_idx_list.append(i_frame)
            track_id_list.append(i_trg)
    tracks_per_frame = pd.DataFrame(
        {
            "frame_idx": np.array(frame_idx_list),
            "track_id": np.array(track_id_list),
            "xmin": np.array(xmin_list),
            "ymin": np.array(ymin_list),
            "xmax": np.array(xmax_list),
            "ymax": np.array(ymax_list),
            "target_x_pix": np.array(target_x_list),
            "target_y_pix": np.array(target_y_list),
            "n_pix_in_bb": np.array(n_pix_in_bb_list),
        },
    ).astype({"frame_idx": int, "track_id": int})
    return tracks_per_frame


# --------------------------------------------------------------------------------------------------------------------

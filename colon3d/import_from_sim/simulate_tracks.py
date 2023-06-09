import numpy as np
import pandas as pd

from colon3d.torch_util import np_func
from colon3d.transforms_util import (
    get_frame_point_cloud,
    transform_rectilinear_image_pixel_coords_to_normalized,
    unproject_image_normalized_coord_to_world,
)

# --------------------------------------------------------------------------------------------------------------------


def generate_targets(
    n_targets: int,
    gt_depth_maps: np.ndarray,
    gt_cam_poses: np.ndarray,
    rng: np.random.Generator,
    depth_info: dict,
    min_target_radius_mm: float,
    max_target_radius_mm: float,
    max_dist_from_center_ratio: float,
):
    """generate random 3D points on the surface of the colon, which will be used as the center of the tracks/"""
    n_frames, frame_width, frame_height = gt_depth_maps.shape
    K_of_depth_map = depth_info["K_of_depth_map"]
    target_center_radius_max = max_dist_from_center_ratio * min(frame_width / 2, frame_height / 2)

    max_attempts = 100
    i_attempt = 0
    min_target_pixels = 10 # minimum number of pixels that the target should cover

    # we are going to sample the points that are seen from the first frame of the sequence:
    i_frame = 0
    cam_pose = gt_cam_poses[i_frame][np.newaxis, :]
    depth_map = gt_depth_maps[i_frame]

    # get the point cloud of the first frame (in world coordinate:)
    points3d = get_frame_point_cloud(z_depth_frame=depth_map, K_of_depth_map=K_of_depth_map, cam_pose=cam_pose)

    while i_attempt < max_attempts:
        # randomly sample pixels inside a circle with radius max_radius around the center of the image:
        target_center_radius = rng.uniform(0, target_center_radius_max, size=n_targets)
        angle = rng.uniform(0, 2 * np.pi, size=n_targets)
        pixel_x = ((frame_width / 2) + target_center_radius * np.cos(angle)).astype(int)
        pixel_y = ((frame_height / 2) + target_center_radius * np.sin(angle)).astype(int)
        z_depth = depth_map[pixel_y, pixel_x]  # notice that the depth image coordinates are (y,x) not (x,y).

        # get the 3D point in the world coordinate system of the target center
        target_center_nrm = transform_rectilinear_image_pixel_coords_to_normalized(
            pixels_x=pixel_x,
            pixels_y=pixel_y,
            cam_K=K_of_depth_map,
        )
        target_center_3d = np_func(unproject_image_normalized_coord_to_world)(
            points_nrm=target_center_nrm,
            z_depths=z_depth,
            cam_poses=cam_pose,
            )
        # Determine the size of the ball around the target center:
        target_radius = rng.uniform(min_target_radius_mm, max_target_radius_mm, size=n_targets)

        # check how many points are inside the ball around each target center:
        n_points_inside= np.sum(
            np.linalg.norm(points3d - target_center_3d[:, np.newaxis], axis=-1) < target_radius,
            axis=-1,
        )
        print(f"i_attempt={i_attempt}, n_points_inside={n_points_inside}")
        if n_points_inside > min_target_pixels:
            break
        i_attempt += 1

    targets_info = {
        "points3d": target_center_3d,
        "radiuses": target_radius,
        "seen_at_pixel": (i_frame, pixel_x, pixel_y),
    }
    return targets_info


# --------------------------------------------------------------------------------------------------------------------


def create_tracks_per_frame(
    gt_depth_maps: np.ndarray,
    gt_cam_poses: np.ndarray,
    depth_info: dict,
    targets_info: dict,
) -> pd.DataFrame:
    """Returns the detections bounding boxes of the tracks in each frame."""
    tracks_point3d = targets_info["points3d"]
    tracks_radiuses = targets_info["radiuses"]
    n_tracks = tracks_point3d.shape[0]
    n_frames, frame_width, frame_height = gt_depth_maps.shape
    K_of_depth_map = depth_info["K_of_depth_map"]
    tracks_initial_area = {}  # a dictionary that will contain the initial area of each track in each frame [pixels^2]
    # pixel coordinates of all the pixels in the image - we use (y, x) since this is the index order of the depth image
    pixels_y, pixels_x = np.meshgrid(np.arange(frame_height), np.arange(frame_width), indexing="ij")
    pixels_x = pixels_x.flatten()
    pixels_y = pixels_y.flatten()
    xmin_list = []
    ymin_list = []
    xmax_list = []
    ymax_list = []
    frame_idx_list = []
    track_id_list = []
    discard_track_ratio = 0.5
    for i_frame in range(n_frames):
        cam_pose = gt_cam_poses[i_frame].reshape(1, 7)
        depth_map = gt_depth_maps[i_frame]
        # get the point cloud of the first frame (in world coordinate:)
        points3d = get_frame_point_cloud(z_depth_frame=depth_map, K_of_depth_map=K_of_depth_map, cam_pose=cam_pose)
        for i_track in range(n_tracks):
            # find the pixels that are inside the ball around each track center:
            track_center = tracks_point3d[i_track]
            track_radius = tracks_radiuses[i_track]
            is_inside = np.linalg.norm(points3d - track_center, axis=-1) < track_radius
            pixels_x_inside = pixels_x[is_inside]
            pixels_y_inside = pixels_y[is_inside]
            n_inside = np.sum(is_inside)
            if n_inside == 0:
                continue
            if i_frame == 0:
                assert n_inside > 0, "the first frame should contain at least one pixel inside the ball"
                tracks_initial_area[i_track] = n_inside
            elif n_inside / tracks_initial_area[i_track] < discard_track_ratio:
                # if the ratio of the area of the track in the current frame to the initial area is too low - discard it:
                continue
            # create a new detection of the track:
            # find bounding box of the pixels that are inside the ball:
            xmin_list.append(pixels_x_inside.min())
            ymin_list.append(pixels_y_inside.min())
            xmax_list.append(pixels_x_inside.max())
            ymax_list.append(pixels_y_inside.max())
            frame_idx_list.append(i_frame)
            track_id_list.append(i_track)
    detections = pd.DataFrame(
        {
            "frame_idx": np.array(frame_idx_list),
            "track_id": np.array(track_id_list),
            "xmin": np.array(xmin_list),
            "ymin": np.array(ymin_list),
            "xmax": np.array(xmax_list),
            "ymax": np.array(ymax_list),
        },
    ).astype({"frame_idx": int, "track_id": int})
    return detections


# --------------------------------------------------------------------------------------------------------------------

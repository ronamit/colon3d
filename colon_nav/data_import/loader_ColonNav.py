import json
from pathlib import Path

import cv2
import numpy as np

from colon_nav.util.general_util import path_to_str
from colon_nav.util.rotations_util import normalize_quaternions
from colon_nav.util.torch_util import np_func

# --------------------------------------------------------------------------------------------------------------------

# In this simulator, 1 unity distance unit = 100 mm = 1 cm.
UNITY_TO_MM = 100  # multiply the loaded distance values by this factor to get mm units.
# --------------------------------------------------------------------------------------------------------------------


def load_sim_raw(load_base_path: Path, limit_n_scenes: int, limit_n_frames, fps_override: float):
    """Load the raw data saved using Unity's Perception Package camera captures.
    Args:
        load_base_path: the path to the folder that contains the "Dataset*" folder
        limit_n_scenes: limit the number of scenes to load (for debugging) if 0 then no limit
        limit_n_frames: limit the number of frames to load (for debugging) if 0 then no limit
        fps_override: override the FPS value in the metadata (if 0 then use the value in the metadata)
    Returns:
        scenes_names: list of scene names
        metadata_per_scene: list of metadata per scene
        rgb_frames_paths_per_scene: list of lists of the RGB frames paths per scene
        cam_poses_per_scene: list of lists of the camera poses per scene
        depth_frames_paths_per_scene: list of lists of the depth frames paths per scene
    """

    # gather all the "capture" files
    assert load_base_path.is_dir(), "The input path should be a directory"
    paths = [p for p in load_base_path.glob("Dataset*") if p.is_dir()]
    assert len(paths) == 1, f"there should be exactly one Dataset* sub-folder in load_base_path={load_base_path}"
    metadata_dir_path = paths[0]
    print(f"Loading dataset metadata from {metadata_dir_path}")
    captures = []
    file_index = 0
    while True:
        filename = f"captures_{str(file_index).zfill(3)}.json"
        f_path = metadata_dir_path / filename
        if not f_path.exists():
            break
        with f_path.open("r") as f:
            metadata = json.load(f)
            captures += metadata["captures"]
        file_index += 1

    ### extract the data from the captures ###
    scenes_names = []  # list of scene names
    rgb_frames_paths_per_scene = []  # list of lists
    depth_frames_paths_per_scene = []  # list of lists
    # list of lists of the camera translation per frame per scene, before changing to our format:
    raw_trans_per_scene = []
    # list of lists of the camera rotation per frame per scene, before changing to our format:
    raw_rot_per_scene = []
    seen_rgb_dirs = {}  # set of seen rgb dirs
    metadata_per_scene = []  # list of metadata per scene
    scene_idx = -1
    frame_idx = -1
    n_scenes = 0
    print(f"Extracting data from the captures in {metadata_dir_path}")
    for capture in captures:
        frame_idx += 1
        rgb_file_path = load_base_path / capture["filename"]
        # check if we started a new scene (i.e. a new folder of RGB images)
        rgb_dir_name = rgb_file_path.parent.name
        if rgb_dir_name not in seen_rgb_dirs:
            # we found a new scene
            scene_idx += 1
            seen_rgb_dirs[rgb_dir_name] = scene_idx
            # check if we reached the limit of scenes
            if limit_n_scenes > 0 and scene_idx >= limit_n_scenes:
                break
            n_scenes = scene_idx + 1
            scene_name = "Scene_" + str(scene_idx).zfill(5)
            metadata = get_scene_metadata(
                capture=capture,
                raw_data_source=rgb_file_path.parent,
                fps_override=fps_override,
            )
            metadata_per_scene.append(metadata)
            scenes_names.append(scene_name)
            rgb_frames_paths_per_scene.append([])
            depth_frames_paths_per_scene.append([])
            raw_trans_per_scene.append([])
            raw_rot_per_scene.append([])
        elif limit_n_frames > 0 and frame_idx >= limit_n_frames:
            # check if we reached or passe te limit of frames per scene
            # # in this case, just skip the current capture... until we reach the next scene
            continue
        # extract the current frame data
        rgb_file_path = load_base_path / capture["filename"]
        raw_trans_per_scene[-1].append(np.array(capture["sensor"]["translation"]))
        raw_rot_per_scene[-1].append(np.array(capture["sensor"]["rotation"]))
        annotations = capture["annotations"]
        depth_annotation = next(a for a in annotations if a["@type"] == "type.unity.com/unity.solo.DepthAnnotation")
        depth_file_path = load_base_path / depth_annotation["filename"]
        # store the current frame data:
        rgb_frames_paths_per_scene[-1].append(rgb_file_path)
        depth_frames_paths_per_scene[-1].append(depth_file_path)
    # end for capture in captures
    n_scenes = len(scenes_names)
    print(f"Number of extracted scenes: {n_scenes}")

    cam_poses_per_scene = []
    for i_scene in range(n_scenes):
        translations = np.stack(raw_trans_per_scene[i_scene], axis=0)
        rotations = np.stack(raw_rot_per_scene[i_scene], axis=0)
        cam_poses_per_scene.append(get_cam_pose(cam_trans=translations, cam_rot=rotations))

    return (
        scenes_names,
        metadata_per_scene,
        rgb_frames_paths_per_scene,
        cam_poses_per_scene,
        depth_frames_paths_per_scene,
    )


# --------------------------------------------------------------------------------------------------------------------


def get_scene_metadata(capture: dict, raw_data_source: Path, fps_override: float) -> dict:
    cam_intrinsic = next(a for a in capture["annotations"] if a["@type"] == "camDataDef")
    frame_width = cam_intrinsic["pixelWidth"]  # [pixels]
    frame_height = cam_intrinsic["pixelHeight"]  # [pixels]
    # the physical specs of the camera are given in 0.001 Unity distance units
    total_sensor_size_x_mm = cam_intrinsic["sensorSizeX"] * 0.001 * UNITY_TO_MM  # [mm]
    total_sensor_size_y_mm = cam_intrinsic["sensorSizeY"] * 0.001 * UNITY_TO_MM  # [mm]
    focal_length_mm = cam_intrinsic["focalLength"] * 0.001 * UNITY_TO_MM  # [mm]
    min_vis_z_mm = cam_intrinsic["nearClipPlane"] * UNITY_TO_MM  # [mm]
    assert cam_intrinsic["lensShiftX"] == 0 and cam_intrinsic["lensShiftY"] == 0, "lens shift is not supported"
    frame_time_interval = cam_intrinsic["simulationDeltaTime"]  # [sec]

    fps = 1 / frame_time_interval  # [Hz]
    if fps_override != 0:
        fps = fps_override

    # per-pixel sensor size
    sx = total_sensor_size_x_mm / frame_width  # [millimeter/pixel]
    sy = total_sensor_size_y_mm / frame_height  # [millimeter/pixel]
    # focal-length in pixel units
    fx = focal_length_mm / sx
    fy = focal_length_mm / sy
    # the optical center pixel location
    cx = frame_width / 2
    cy = frame_height / 2
    metadata = {
        "frame_width": frame_width,  # [pixels]
        "frame_height": frame_height,  # [pixels]
        "fx": float(fx),  # focal length in x-axis [pixels]
        "fy": float(fy),  # focal length in y-axis [pixels]
        "cx": cx,  # optical center in x-axis [pixels]
        "cy": cy,  # optical center in y-axis [pixels]
        "fps": fps,  # frame rate [Hz]
        "distort_pram": None,  # simulated images are not distorted
        "min_vis_z_mm": min_vis_z_mm,  # in the simulation,
        "sim_info": {
            "total_sensor_size_x_mm": total_sensor_size_x_mm,
            "total_sensor_size_y_mm": total_sensor_size_y_mm,
            "focal_length_mm": float(focal_length_mm),
        },
        "raw_data_source": raw_data_source,
    }
    return metadata


# --------------------------------------------------------------------------------------------------------------------
def get_cam_pose(cam_trans: np.ndarray, cam_rot: np.ndarray) -> np.ndarray:
    """
    save the 6-DOF camera poses in the world coordinates system.
    the 7 values are: x, y, z, q_w, q_x, q_y, q_z
    (x, y, z) is the camera position in the world system (in mm)
    (q_w, q_x, q_y, q_z) is a  unit-quaternion has the real part as the first value, that represents the camera rotation w.r.t. the world system.
    Args:
        raw_trans: 3D translation vectors (x, y, z) in Unity units
        raw_rot: rotation unit-quaternion in Unity units
    Returns:
        cam_poses: the camera-poses in out format, a numpy array with shape (N, 7) where N is the number of frames.
    See also:
    - https://gamedev.stackexchange.com/a/201978
    - https://github.com/zsustc/colon_reconstruction_dataset
    """
    # change quaternion to real-first format (q_w, q_x, q_y, q_z)
    cam_rot = cam_rot[:, [3, 0, 1, 2]]
    # change the units from Unity units to mm
    cam_trans *= UNITY_TO_MM
    # Change the camera transform from left-handed to right-handed coordinate system.
    # y <-> -y
    cam_trans[:, 1] *= -1
    # change the rotation accordingly: qy <-> -qy
    cam_rot[:, 2] *= -1
    # change the rotation to direction (since the switch y <-> -y took a mirror image of the world)
    cam_rot[:, 1:] *= -1
    # ensure that the quaternion is unit
    cam_rot = np_func(normalize_quaternions)(cam_rot)
    cam_poses = np.concatenate((cam_trans, cam_rot), axis=1)
    return cam_poses


# --------------------------------------------------------------------------------------------------------------------


def get_ground_truth_depth(
    depth_frames_paths: list,
    metadata: dict,
):
    # Get the depth maps
    n_frames = len(depth_frames_paths)
    frame_height = metadata["frame_height"]
    frame_width = metadata["frame_width"]
    z_depth_frames = np.zeros((n_frames, frame_height, frame_width), dtype=np.float64)
    for i_frame in range(n_frames):
        depth_file_path = depth_frames_paths[i_frame]
        print(f"Loading depth frame {i_frame}/{n_frames}", end="\r")
        # All 3 channels are the same (depth), so we only need to read one
        depth_img = cv2.imread(path_to_str(depth_file_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        R_channel_idx = 2  # OpenCV reads in BGR order
        z_depth_mm = UNITY_TO_MM * depth_img[:, :, R_channel_idx]  # z-depth is stored in the R channel
        z_depth_frames[i_frame] = z_depth_mm

    return z_depth_frames


# --------------------------------------------------------------------------------------------------------------------

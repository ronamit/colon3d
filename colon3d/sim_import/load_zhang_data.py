from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np

from colon3d.util.general_util import find_between_str, path_to_str
from colon3d.util.rotations_util import normalize_quaternions
from colon3d.util.torch_util import np_func

"""
Instructions for importing the data from the Zhang22 simulator (not needed if you use the pre-processed data)):

* Download the dataset from [Zhang22](https://github.com/zsustc/colon_reconstruction_dataset).
If download fails, try to download each case folder separately.

* Extract all the Case `<number>`  directories (1 to 15)  to a some path

* Run the import script:

python -m colon3d.sim_import.import_dataset --sim_name "Zhang22"  --raw_sim_data_path PATH --path_to_save_data PATH

"""
# --------------------------------------------------------------------------------------------------------------------


def load_sim_raw(
    input_data_path: Path,
    limit_n_scenes: int,
    limit_n_frames: int,
    fps_override: float,
    cam_to_load: str,
):
    """extract the data from the cases folders generated by the Zhang22 simulator"""
    case_dir_names = [Path(p.name) for p in input_data_path.glob("Case*") if p.is_dir()]
    case_dir_names.sort()
    if limit_n_scenes > 0:
        case_dir_names = case_dir_names[:limit_n_scenes]
    n_scenes = len(case_dir_names)
    rgb_frames_paths_per_scene = []  # list of lists
    cam_poses_per_scene = []
    metadata_per_scene = []  # list of metadata per scene

    for i_scene in range(n_scenes):
        case_path = case_dir_names[i_scene]
        # Get the RGB frames file paths (relative to input_data_path):
        rgb_frames_paths = list((input_data_path / case_path / cam_to_load).glob("*.png"))
        rgb_frames_paths = [Path(case_path) / cam_to_load / p.name for p in rgb_frames_paths]
        rgb_frames_paths.sort()
        if limit_n_frames > 0:
            rgb_frames_paths = rgb_frames_paths[:limit_n_frames]
        assert len(rgb_frames_paths) > 0, f"no RGB frames found in {case_path}"
        metadata_per_scene.append(get_scene_metadata(fps_override))
        rgb_frames_paths_per_scene.append(rgb_frames_paths)
        cam_poses_per_scene.append(
            get_cam_poses(
                scene_path=input_data_path / case_path,
                limit_n_frames=limit_n_frames,
                cam_to_load=cam_to_load,
            ),
        )
    scene_names = ["Scene_" + str(i_scene).zfill(5) for i_scene in range(n_scenes)]
    return (
        scene_names,
        metadata_per_scene,
        rgb_frames_paths_per_scene,
        cam_poses_per_scene,
    )


# --------------------------------------------------------------------------------------------------------------------


def get_scene_metadata(fps_override) -> dict:
    # The values are based on https://github.com/zsustc/colon_reconstruction_dataset
    focal_length_mm = 4.969783  # [mm]
    total_sensor_size_x_mm = 10.26  # [mm]
    total_sensor_size_y_mm = 7.695  # [mm]
    frame_width = 480  # [pixels]
    frame_height = 640  # [pixels]
    min_vis_z_mm = 0  # [mm] not relevant for this simulator
    fps = 3  # [Hz]
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
    }

    return metadata


# --------------------------------------------------------------------------------------------------------------------
def get_cam_poses(scene_path: Path, limit_n_frames: int, cam_to_load: str) -> np.ndarray:
    cam_name_letter = cam_to_load.capitalize()[0]
    pos_file_path = next(scene_path.glob(f"*_Camera Position {cam_name_letter} Data.txt"))
    i = 0
    # The first frame is always at the origin of the world coordinate system
    pos_x = [0.0]
    pos_y = [0.0]
    pos_z = [0.0]
    loaded_translation_to_mm = 10  # multiply the loaded translation values by this factor to get mm units.
    # (based on the readme of https://github.com/zsustc/colon_reconstruction_dataset)

    with pos_file_path.open() as file:
        lines = file.readlines()
        for line in lines:
            # there seems to be two different formats for the position file, the code below handles both
            if line.startswith("id tx ty tz"):
                continue  # skip the header line
            if "Frame" in line:
                frame_ind = int(find_between_str(line, "Frame ", " "))
                x = float(find_between_str(line, "X=", ","))
                y = float(find_between_str(line, "Y=", ","))
                z = float(find_between_str(line, "Z=", " "))
            else:
                s = clean_line_string(line)
                frame_ind = int(s[:6]) - 1
                x, y, z = (float(v) for v in s[6:].replace(" ", "").split(","))
            assert frame_ind == i, f"frame index mismatch in {pos_file_path}"
            pos_x.append(x)
            pos_y.append(y)
            pos_z.append(z)
            i += 1
            if i == limit_n_frames:
                break
    rot_file_path = next(scene_path.glob("*_Camera Quaternion Rotation Data.txt"))
    i = 0
    # The first frame is always the identity quaternion
    quat_x = [0.0]
    quat_y = [0.0]
    quat_z = [0.0]
    quat_w = [1.0]
    with rot_file_path.open() as file:
        lines = file.readlines()
        for line in lines:
            # there seems to be two different formats for the position file, the code below handles both
            if line.startswith("id qx qy qz qw"):
                continue  # skip the header line
            if "Frame" in line:
                frame_ind = int(find_between_str(line, "Frame ", " "))
                qx = float(find_between_str(line, "X=", ","))
                qy = float(find_between_str(line, "Y=", ","))
                qz = float(find_between_str(line, "Z=", ","))
                qw = float(find_between_str(line, "W=", " "))
            else:
                s = clean_line_string(line)
                frame_ind = int(s[:6]) - 1
                qx, qy, qz, qw = (float(v) for v in s[6:].replace(" ", "").split(","))
            assert i == frame_ind
            quat_x.append(qx)
            quat_y.append(qy)
            quat_z.append(qz)
            quat_w.append(qw)
            i += 1
            if i == limit_n_frames:
                break
    # save the 6-DOF camera poses in the world coordinates as a numpy array with shape (N, 7) where N is the number of frames
    # the 7 values are: x, y, z, q_w, q_x, q_y, q_z
    # the world coordinate system is defined by the camera coordinate system at the first frame (the optical axis of the camera is the z-axis)
    # (x, y, z) is the camera position in the world system (in mm)
    cam_loc = np.column_stack((pos_x, pos_y, pos_z)) * loaded_translation_to_mm
    # (q_w, q_x, q_y, q_z) is a  unit-quaternion has the real part as the first value, that represents the camera rotation w.r.t. the world system
    cam_rot = np.column_stack((quat_w, quat_x, quat_y, quat_z))
    # transform from Unity's left handed space to a right handed space  (see readme.md of https://github.com/zsustc/colon_reconstruction_dataset)
    # y -> -y
    cam_loc[:, 1] *= -1
    # change the rotation accordingly: qy -> -qy
    cam_rot[:, 2] *= -1
    # change the rotation to direction (since the switch y -> -y took a mirror image of the world)
    cam_rot[:, 1:] *= -1
    cam_rot = np_func(normalize_quaternions)(cam_rot)
    cam_poses = np.column_stack((cam_loc, cam_rot))
    return cam_poses


# --------------------------------------------------------------------------------------------------------------------


def clean_line_string(line: str):
    s = deepcopy(line)
    # delete all spaces in the start of the line:
    while s[0] == " ":
        s = s[1:]
    s.replace("\n", "")
    return s


# --------------------------------------------------------------------------------------------------------------------


def read_depth_exr_file(exr_path):
    # the  values in the loaded EXR files should be multiplied by this value to get the actual depth in mm
    # according to https://github.com/zsustc/colon_reconstruction_dataset
    EXR_DEPTH_SCALE = 5.0
    z_depth_img = (
        EXR_DEPTH_SCALE * cv2.imread(path_to_str(exr_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
    )
    return z_depth_img


# --------------------------------------------------------------------------------------------------------------------

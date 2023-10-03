from pathlib import Path

import cv2
import numpy as np

from colon_nav.util.rotations_util import normalize_quaternions
from colon_nav.util.torch_util import np_func

"""
Instructions for importing the data from the SimCol3D dataset (not needed if you use the pre-processed data):

* Download the dataset from https://rdr.ucl.ac.uk/articles/dataset/Simcol3D_-_3D_Reconstruction_during_Colonoscopy_Challenge_Dataset/24077763/1?file=42248541
* Extract all folders and files to a single folder (raw_sim_data_path)

* Run the import script:
python -m colon3d.sim_import.import_sim_dataset --sim_name "SimCol3D"  --raw_sim_data_path PATH --path_to_save_data PATH


** Info on the raw dataset:
This dataset comprises three synthetic colonoscopy scenes (I-III) and a folder with miscellaneous files.

.
├── misc
├── SyntheticColon_I
│   ├── Frames_S1
...
* Each subfolder "Frames_xx" consists of RGB images and depth maps.
* The greyscale depth images in the range [0, 1] correspond to [0, 20] cm in world space.
* The respective camera poses are provided in .txt files inside the scene folders.
* "SavedPosition" contains the camera translations in the following format: tx, ty, tz. [in cm]
* "SavedRotationQuaternion" includes the camera rotations as quaternions in the following format: qx, qy, qz, qw. Here, t denotes the translation, and q the rotation as quaternions. Each text file corresponds to one folder and can be matched based on the trajectory index (1-15).
* Note that the camera poses are provided in a left-handed coordinate system while the Rotation package assumes right-handedness.
Please find more details in the misc/read_poses.py file.

* The misc/train_file.txt and misc/test_file.txt files contain the names of the training and test scenes, respectively.

*details on how to load and visualize the data:
https://github.com/anitarau/simcol/blob/main/data_helpers/visualize_3D_data.py


"""
# --------------------------------------------------------------------------------------------------------------------


def load_sim_raw(
    input_data_path: Path,
    split_name: str,
    limit_n_scenes: int,
    limit_n_frames: int,
    fps_override: float,
):
    # Get the train\test split from the misc folder:
    split_file_path = input_data_path / "misc" / f"{split_name.lower()}_file.txt"
    with split_file_path.open() as file:
        lines = file.readlines()
        scenes_paths = [Path(line.strip()) for line in lines]
    scenes_paths.sort()
    if limit_n_scenes > 0:
        scenes_paths = scenes_paths[:limit_n_scenes]
    n_scenes = len(scenes_paths)

    depth_frames_paths_per_scene = []  # list of lists
    rgb_frames_paths_per_scene = []  # list of lists
    cam_poses_per_scene = []
    metadata_per_scene = []  # list of metadata per scene

    for i_scene in range(n_scenes):
        scene_path = scenes_paths[i_scene]

        # Get scene metadata
        metadata_per_scene.append(get_scene_metadata(input_data_path / scene_path, fps_override))

        # Get the RGB frames file paths (relative to input_data_path):
        rgb_frames_paths = list((input_data_path / scene_path).glob("FrameBuffer*.png"))
        rgb_frames_paths = [Path(scene_path) / p.name for p in rgb_frames_paths]
        rgb_frames_paths.sort()
        if limit_n_frames > 0:
            rgb_frames_paths = rgb_frames_paths[:limit_n_frames]
        assert len(rgb_frames_paths) > 0, f"no RGB frames found in {scene_path}"
        rgb_frames_paths_per_scene.append(rgb_frames_paths)

        # Get depth frames paths:
        depth_frames_paths = list((input_data_path / scene_path).glob("Depth_*.png"))
        depth_frames_paths = [Path(scene_path) / p.name for p in depth_frames_paths]
        depth_frames_paths.sort()
        if limit_n_frames > 0:
            depth_frames_paths = depth_frames_paths[:limit_n_frames]
        assert len(depth_frames_paths) > 0, f"no depth frames found in {scene_path}"
        depth_frames_paths_per_scene.append(depth_frames_paths)

        # Get camera poses:
        cam_poses_per_scene.append(
            get_cam_poses(
                scene_path=input_data_path / scene_path,
                limit_n_frames=limit_n_frames,
            ),
        )
    scenes_paths = ["Scene_" + str(i_scene).zfill(5) for i_scene in range(n_scenes)]
    return (
        scenes_paths,
        metadata_per_scene,
        rgb_frames_paths_per_scene,
        cam_poses_per_scene,
        depth_frames_paths_per_scene,
    )


# --------------------------------------------------------------------------------------------------------------------


def get_scene_metadata(scene_full_path: Path, fps_override: float = 0) -> dict:
    # The values are based on https://github.com/anitarau/simcol/blob/main/data_helpers/visualize_3D_data.py

    frame_width = 475  # [pixels]
    frame_height = 475  # [pixels]
    min_vis_z_mm = 0  # [mm] not relevant for this simulator
    fps = 20  # [Hz]  the paper doesn't mention the frame rate, so I assume it is 5 Hz (used for video showing + maximum speed penalty)
    if fps_override != 0:
        fps = fps_override

    # load the 3x3 camera matrix from the cam.txt file:
    cam_file_path = scene_full_path.parent / "cam.txt"
    with cam_file_path.open() as file:
        line = file.readlines()
        assert len(line) == 1
        cam_values = np.array([float(v) for v in line[0].split(" ")])
    # note: the order of the values is not clear to me, but it seems that the first value is the focal length in pixels

    # focal-length in pixel units
    fx = cam_values[0]
    fy = cam_values[0]
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
        "raw_scene_path": scene_full_path,
    }

    return metadata


# --------------------------------------------------------------------------------------------------------------------
def get_cam_poses(scene_path: Path, limit_n_frames: int) -> np.ndarray:
    # based on SimCol3D/misc/read_poses.py and https://github.com/anitarau/simcol/blob/main/data_helpers/visualize_3D_data.py
    scene_name = scene_path.name
    scene_codename = scene_name.split("_")[1]
    pos_file_path = scene_path.parent / f"SavedPosition_{scene_codename}.txt"
    rot_file_path = scene_path.parent / f"SavedRotationQuaternion_{scene_codename}.txt"

    locations = []
    rotations = []
    for i, line in enumerate(pos_file_path.open()):
        locations.append(list(map(float, line.split())))
        if limit_n_frames > 0 and i >= limit_n_frames:
            break

    for i, line in enumerate(rot_file_path.open()):
        rotations.append(list(map(float, line.split())))
        if limit_n_frames > 0 and i >= limit_n_frames:
            break

    locations = np.array(locations)  # in cm
    rotations = np.array(rotations)

    assert locations.shape[0] == rotations.shape[0]
    assert locations.shape[1] == 3
    assert rotations.shape[1] == 4  # quaternion

    # save the 6-DOF camera poses in the world coordinates as a numpy array with shape (N, 7) where N is the number of frames
    # the 7 values are: x, y, z, q_w, q_x, q_y, q_z
    # the world coordinate system is defined by the camera coordinate system at the first frame (the optical axis of the camera is the z-axis)
    # (x, y, z) is the camera position in the world system (in mm)
    loaded_translation_to_mm = 10  # multiply the loaded translation values by this factor to get mm units.
    cam_loc = locations * loaded_translation_to_mm
    # (q_w, q_x, q_y, q_z) is a  unit-quaternion has the real part as the first value, that represents the camera rotation w.r.t. the world system
    cam_rot = rotations[:, [3, 0, 1, 2]]  #  change to real first
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


def read_depth_file(depth_map_path: Path) -> np.ndarray:
    # * The greyscale depth images in the range [0, 1] correspond to [0, 20] cm in world space.
    # based on https://github.com/anitarau/simcol/blob/main/data_helpers/visualize_3D_data.py
    depth_map = cv2.imread(str(depth_map_path), cv2.IMREAD_UNCHANGED)
    # convert to [0,1]
    depth_map = depth_map.astype(np.float32) / 256 / 255
    # convert to [0, 200] [mm]
    depth_map = depth_map * 200
    return depth_map


# --------------------------------------------------------------------------------------------------------------------


def get_ground_truth_depth(
    input_data_path: Path,
    depth_frames_paths: list,
    metadata: dict,
):
    # Get the depth maps
    n_frames = len(depth_frames_paths)
    frame_height = metadata["frame_height"]
    frame_width = metadata["frame_width"]
    z_depth_frames = np.zeros((n_frames, frame_height, frame_width), dtype=np.float32)
    for i_frame in range(n_frames):
        depth_file_path = input_data_path / depth_frames_paths[i_frame]
        print(f"Loading depth frame {i_frame}/{n_frames}", end="\r")
        z_depth_mm = read_depth_file(depth_file_path)
        z_depth_frames[i_frame] = z_depth_mm

    return z_depth_frames


# --------------------------------------------------------------------------------------------------------------------

import os
import pickle
import shutil
from pathlib import Path

import cv2
import h5py
import matplotlib
import numpy as np
import yaml
from matplotlib import pyplot as plt

from colon3d.general_util import (
    create_empty_folder,
    find_between_str,
    find_in_file_between_str,
    path_to_str,
    save_plot_and_close,
)

matplotlib.use("Agg")  # use a non-interactive backend
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # for reading EXR files


# --------------------------------------------------------------------------------------------------------------------


def get_seq_id_from_path(seq_in_path: Path):
    """
    get the 3 letter prefix of the files
    """
    first_file_name = next(seq_in_path.glob("*.png")).name
    seq_id = first_file_name[0:3]
    return seq_id


# --------------------------------------------------------------------------------------------------------------------
def read_depth_exr_file(exr_path, metadata):
    # the  values in the loaded EXR files should be multiplied by this value to get the actual depth in mm:
    EXR_DEPTH_SCALE = metadata["sim_info"]["EXR_DEPTH_SCALE"]
    z_depth_img = (
        EXR_DEPTH_SCALE * cv2.imread(path_to_str(exr_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:, :, 0]
    )
    return z_depth_img


# --------------------------------------------------------------------------------------------------------------------


class NewSimImporter:
    # --------------------------------------------------------------------------------------------------------------------
    def __init__(self, raw_sim_data_path: str, path_to_save_sequence: str, limit_n_frames: int, fps_override: float):
        seq_in_path = Path(raw_sim_data_path)
        seq_out_path = Path(path_to_save_sequence)
        print("Raw simulated sequences will be be loaded from: ", seq_in_path)
        create_empty_folder(seq_out_path, ask_overwrite=True)
        print(f"The processed sequence will be saved to {seq_out_path}")
        self.seq_in_path = seq_in_path
        self.seq_out_path = seq_out_path
        self.limit_n_frames = limit_n_frames
        self.fps_override = fps_override

    # --------------------------------------------------------------------------------------------------------------------

    def import_sequence(self):
        metadata, n_frames, fps = self.create_metadata()
        if self.limit_n_frames > 0:
            n_frames = min(self.limit_n_frames, n_frames)
            print(f"Only {n_frames} frames will be processed")
        else:
            print(f"All {n_frames} frames will be processed")
        self.save_rgb_video(
            vid_file_name="Video",
            n_frames=n_frames,
            fps=fps,
        )
        self.save_ground_truth_depth_and_cam_poses(
            metadata=metadata,
            n_frames=n_frames,
        )

    # --------------------------------------------------------------------------------------------------------------------

    def create_metadata(self):
        seq_id = get_seq_id_from_path(self.seq_in_path)
        sim_settings_path = self.seq_in_path / "MySettings.set"
        # copy the settings file to the dataset folder
        shutil.copy2(sim_settings_path, self.seq_out_path / "Sim_GUI_Settings.set")
        # Extract the settings from the settings file
        # camera FOV [deg]:
        camFOV_deg = float(find_in_file_between_str(sim_settings_path, '"camFOV":"float(', ')"'))
        np.deg2rad(camFOV_deg)  #  [rad]
        frame_width = int(find_in_file_between_str(sim_settings_path, '"shotResX":"float(', ')"'))  # [pixels]
        frame_height = int(find_in_file_between_str(sim_settings_path, '"shotResY":"float(', ')"'))  # [pixels]
        if self.fps_override == 0:
            fps = float(find_in_file_between_str(sim_settings_path, '"shotPerSec":"float(', ')"'))  # [Hz]
        else:
            fps = self.fps_override

        # Load intrinsics data:
        intrinsics_data_path = self.seq_in_path / (seq_id + "_Intrinsic Data.txt")
        shutil.copy2(intrinsics_data_path, self.seq_out_path / "intrinsic_data.txt")
        focal_length_mm = float(
            find_in_file_between_str(
                intrinsics_data_path,
                before_str=": ",
                after_str=" \n",
                line_prefix="Focal Length (mm) :",
            ),
        )
        total_sensor_size_x_mm = float(
            find_in_file_between_str(
                intrinsics_data_path,
                before_str=" X=",
                after_str=" ",
                line_prefix="Sensor Size (mm) :",
            ),
        )
        total_sensor_size_y_mm = float(
            find_in_file_between_str(
                intrinsics_data_path,
                before_str=" Y=",
                after_str=" ",
                line_prefix="Sensor Size (mm) :",
            ),
        )
        n_frames = int(
            find_in_file_between_str(
                intrinsics_data_path,
                before_str=": ",
                after_str=" \n",
                line_prefix="Total Images : ",
            ),
        )
        EXR_DEPTH_SCALE = (
            5.0  # the  values in the loaded EXR files should be multiplied by this value to get the actual
        )
        # depth in mm, according to https://github.com/zsustc/colon_reconstruction_dataset
        sx_mm = total_sensor_size_x_mm / frame_width  # the with of each pixel's sensor [millimeter/pixel]
        sy_mm = total_sensor_size_y_mm / frame_height  # the height of each pixel's sensor [millimeter/pixel]
        fx = focal_length_mm / sx_mm  # [pixels]
        fy = focal_length_mm / sy_mm  # [pixels]
        cx = frame_width / 2.0  # middle of the image in x-axis [pixels]
        cy = frame_height / 2.0  # middle of the image in y-axis [pixels]

        metadata = {
            "frame_width": frame_width,  # [pixels]
            "frame_height": frame_height,  # [pixels]
            "fx": float(fx),  # focal length in x-axis [pixels]
            "fy": float(fy),  # focal length in y-axis [pixels]
            "cx": cx,  # optical center in x-axis [pixels]
            "cy": cy,  # optical center in y-axis [pixels]
            "fps": fps,  # frame rate [Hz]
            "distort_pram": None,  # simulated images are not distorted
            "min_vis_z_mm": 0.0,  # in the simulation, the minimal visible z-depth is 0.0
            "sim_info": {
                "EXR_DEPTH_SCALE": EXR_DEPTH_SCALE,  # the  values in the loaded EXR files should be multiplied by this value to get the actual depth in mm(
                "total_sensor_size_x_mm": total_sensor_size_x_mm,
                "total_sensor_size_y_mm": total_sensor_size_y_mm,
                "sx_mm": sx_mm,
                "sy_mm": sy_mm,
                "focal_length_mm": float(focal_length_mm),
            },
        }
        file_name = "meta_data.yaml"
        metadata_path = self.seq_out_path / file_name
        with metadata_path.open("w") as file:
            yaml.dump(metadata, file)

        # copy any screenshots of the simulator GUI, if exist
        for file_path in self.seq_in_path.glob("Screenshot*.png"):
            shutil.copy2(file_path, self.seq_out_path / file_path.name)

        return metadata, n_frames, fps

    # --------------------------------------------------------------------------------------------------------------------
    def save_rgb_video(self, vid_file_name: str, n_frames: int, fps: float):
        output_path = self.seq_out_path / (vid_file_name + ".mp4")
        seq_id = get_seq_id_from_path(self.seq_in_path)
        frames_paths = list(self.seq_in_path.glob(f"{seq_id}_*.png"))
        frames_paths.sort()
        assert (
            len(frames_paths) >= n_frames
        ), f"Only {len(frames_paths)} frames were found in {self.seq_in_path}, but {n_frames} are required"
        frames_paths = frames_paths[:n_frames]
        frame_size = cv2.imread(path_to_str(frames_paths[0])).shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out_video = cv2.VideoWriter(
            filename=path_to_str(output_path),
            fourcc=fourcc,
            fps=fps,
            frameSize=frame_size,
            isColor=True,
        )
        for i_frame, frame_path in enumerate(frames_paths):
            print(f"Writing RGB frame {i_frame+1}/{n_frames} to video")
            assert frame_path.name.startswith(
                f"{seq_id}_{i_frame:05d}",
            ), f" {frame_path.name} does not match the expected name"
            im = cv2.imread(path_to_str(frame_path))
            out_video.write(im)
        out_video.release()
        print(f"Video saved to: {output_path}")

    # --------------------------------------------------------------------------------------------------------------------

    def load_camera_motion(self, seq_in_path: Path, n_frames: int):
        """Load the camera motion from the file 'seq_id_Camera Position Data.txt'
        Based on the data format description in https://github.com/zsustc/colon_reconstruction_dataset
        """
        seq_id = get_seq_id_from_path(seq_in_path)
        pos_file_path = seq_in_path / (seq_id + "_Camera Position Data.txt")
        i = 0
        pos_x = []
        pos_y = []
        pos_z = []
        cm_to_mm = 10
        with pos_file_path.open() as file:
            lines = file.readlines()
            for line in lines:
                frame_ind = int(find_between_str(line, "Frame ", " "))
                assert i == frame_ind
                pos_x.append(cm_to_mm * float(find_between_str(line, "X=", ",")))
                pos_y.append(cm_to_mm * float(find_between_str(line, "Y=", ",")))
                pos_z.append(cm_to_mm * float(find_between_str(line, "Z=", " ")))
                i += 1
                if i == n_frames:
                    break
        print(f"Camera positions were loaded for {len(pos_x)} frames.")
        rot_file_path = seq_in_path / (seq_id + "_Camera Quaternion Rotation Data.txt")
        i = 0
        quat_x = []
        quat_y = []
        quat_z = []
        quat_w = []
        with rot_file_path.open() as file:
            lines = file.readlines()
            for line in lines:
                frame_ind = int(find_between_str(line, "Frame ", " "))
                assert i == frame_ind
                quat_x.append(float(find_between_str(line, "X=", ",")))
                quat_y.append(float(find_between_str(line, "Y=", ",")))
                quat_z.append(float(find_between_str(line, "Z=", ",")))
                quat_w.append(float(find_between_str(line, "W=", " ")))
                i += 1
                if i == n_frames:
                    break
        print(f"Camera rotations were loaded for {len(quat_x)} frames.")
        # save the 6-DOF camera poses in the world coordinates as a numpy array with shape (N, 7) where N is the number of frames
        # the 7 values are: x, y, z, q_w, q_x, q_y, q_z
        # the world coordinate system is defined by the camera coordinate system at the first frame (the optical axis of the camera is the z-axis)

        # (x, y, z) is the camera position in the world system (in mm)
        cam_loc = np.column_stack((pos_x, pos_y, pos_z))

        # (q_w, q_x, q_y, q_z) is a  unit-quaternion has the real part as the first value, that represents the camera rotation w.r.t. the world system
        cam_rot = np.column_stack((quat_w, quat_x, quat_y, quat_z))

        # transform from the unity left handed space to a right handed space  (see readme.md of https://github.com/zsustc/colon_reconstruction_dataset)
        # [x,-y,z]-->[x,y,z]
        cam_loc[:, 1] *= -1

        #  [qw, qx, qy, qz]-->[ qw, -qx, qy, -qz]
        cam_rot[:, 1] *= -1
        cam_rot[:, 3] *= -1

        cam_poses = np.column_stack((cam_loc, cam_rot))
        return cam_poses

    # --------------------------------------------------------------------------------------------------------------------
    def save_ground_truth_depth_and_cam_poses(
        self,
        metadata: dict,
        n_frames: int,
    ):
        """
        Load a sequence of depth images from a folder
        """

        seq_id = get_seq_id_from_path(self.seq_in_path)
        depth_files_paths = list(self.seq_in_path.glob(f"{seq_id}_depth*.exr"))
        depth_files_paths.sort()
        depth_files_paths = depth_files_paths[:n_frames]
        n_frames = len(depth_files_paths)
        print(f"Number of depth frames: {n_frames}")

        # Get the depth maps
        # find the frame size
        z_depth_img = read_depth_exr_file(depth_files_paths[0], metadata)
        depth_map_size = z_depth_img.shape[:2]
        z_depth_frames = np.zeros((n_frames, depth_map_size[0], depth_map_size[1]), dtype=np.float32)
        for i_frame, exr_path in enumerate(depth_files_paths):
            print(f"Loading depth frame {i_frame} from {exr_path}")
            assert exr_path.name.startswith(
                f"{seq_id}_depth{i_frame:05d}",
            ), f"Depth file name is not correct: {exr_path.name}"
            # All 3 channels are the same (depth), so we only need to read one
            z_depth_img = read_depth_exr_file(exr_path, metadata)
            z_depth_frames[i_frame] = z_depth_img

        # Get the egomotion (transformation of the camera pose from the previous frame to the current frame.)
        cam_poses = self.load_camera_motion(self.seq_in_path, n_frames)

        # save as h5 file
        output_path = self.seq_out_path / "gt_depth_and_cam_poses.h5"
        print(f"Saving depth frames and camera poses to: {output_path}")
        with h5py.File(output_path, "w") as hf:
            hf.create_dataset("z_depth_map", data=z_depth_frames, compression="gzip")
            hf.create_dataset("cam_poses", data=cam_poses)

        assert (
            z_depth_frames.shape[0] == cam_poses.shape[0]
        ), "Number of depth frames and camera poses should be the same."
        # print histogram
        print(f"Z-Depth frames saved to: {output_path}")
        plt.hist(z_depth_frames.flatten(), bins="auto")
        save_plot_and_close(self.seq_out_path / "z_depth_histogram.png")

        # The simulator generates depth maps with the same camera intrinsics as the RGB images.
        fx = metadata["fx"]
        fy = metadata["fy"]
        cx = metadata["cx"]
        cy = metadata["cy"]
        K_of_depth_map = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # save metadata for the depth maps
        depth_info = {
            "K_of_depth_map": K_of_depth_map,
            "n_frames": n_frames,
            "depth_map_size": {"width": depth_map_size[1], "height": depth_map_size[0]},
        }

        with (self.seq_out_path / "depth_info.pkl").open("wb") as file:
            pickle.dump(depth_info, file)

    # --------------------------------------------------------------------------------------------------------------------

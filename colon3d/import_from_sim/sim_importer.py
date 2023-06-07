import json
import os
import pickle
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml

from colon3d.general_util import (
    create_empty_folder,
    path_to_str,
    save_video_from_func,
)
from colon3d.torch_util import np_func
from colon3d.transforms_util import infer_egomotions
from colon3d.visuals.plot_depth_video import plot_depth_video

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # for reading EXR files

# --------------------------------------------------------------------------------------------------------------------


def change_cam_transform_to_right_handed(cam_trans: np.ndarray, cam_rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforms the camera transform from left-handed to right-handed coordinate system.
    Args:
        cam_trans: (N, 3) array of camera translations
        cam_rot: (N, 4) array of camera rotations (quaternions) in the format (qw, qx, qy, qz)
    Notes:
        see - https://gamedev.stackexchange.com/a/201978
            - https://github.com/zsustc/colon_reconstruction_dataset
    """
    # y <-> -y
    cam_trans[:, 1] *= -1
    # # change the rotation accordingly: qy <-> -qy
    cam_rot[:, 2] *= -1
    # # since we switched from LH to RH space, we need to flip the rotation angle sign.
    cam_rot[:, 1:] *= -1

    return cam_trans, cam_rot


# --------------------------------------------------------------------------------------------------------------------


def change_image_to_right_handed(image: np.ndarray) -> np.ndarray:
    """
    Transforms the image from left-handed to right-handed coordinate system.
    Args:
        image: (H, W, ..) array of image pixels
    """
    # rotate the first two dimensions by negative 90 degrees:
    image = np.rot90(image, k=-1, axes=(0, 1))
    return image


# --------------------------------------------------------------------------------------------------------------------


class SimImporter:
    # --------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        raw_sim_data_path: str,
        processed_sim_data_path: str,
        limit_n_sequences: int,
        limit_n_frames: int,
        fps_override: float,
        ask_overwrite: bool = True,
    ):
        input_data_path = Path(raw_sim_data_path)
        output_data_path = Path(processed_sim_data_path)
        print("Raw simulated sequences will be be loaded from: ", input_data_path)
        assert input_data_path.exists(), "The input path does not exist"
        print("Processed simulated sequences will be saved to: ", output_data_path)
        create_empty_folder(output_data_path, ask_overwrite=ask_overwrite)
        self.input_data_path = input_data_path
        self.output_data_path = output_data_path
        self.limit_n_frames = limit_n_frames
        self.limit_n_sequences = limit_n_sequences
        self.fps_override = fps_override
        # In our models, 1 Unity distance unit = 100 mm
        self.UNITY_TO_MM = 100

    # --------------------------------------------------------------------------------------------------------------------

    def import_data(self):
        # gather all the "capture" files
        paths = [p for p in self.input_data_path.glob("Dataset*") if p.is_dir()]
        assert len(paths) == 1, "there should be exactly one Dataset* sub-folder"
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

        # extract the data from the captures

        rgb_frames_paths_per_seq = []  # list of lists
        depth_frames_paths_per_seq = []  # list of lists
        # list of lists of the camera translation per frame per sequence, before changing to our format:
        raw_trans_per_seq = []
        # list of lists of the camera rotation per frame per sequence, before changing to our format:
        raw_rot_per_seq = []
        seen_rgb_dirs = {}  # set of seen rgb dirs
        seq_names = []  # list of sequence names
        metadata_per_seq = []  # list of metadata per sequence
        seq_idx = -1
        frame_idx = -1
        for capture in captures:
            frame_idx += 1
            rgb_file_path = capture["filename"]
            # check if we started a new sequence (i.e. a new folder of RGB images)
            rgb_dir_name = rgb_file_path.split("/")[-2]
            if rgb_dir_name not in seen_rgb_dirs:
                # we found a new sequence
                seq_idx += 1
                seen_rgb_dirs[rgb_dir_name] = seq_idx
                # check if we reached the limit of sequences
                if self.limit_n_sequences > 0 and seq_idx >= self.limit_n_sequences:
                    break
                seq_name = "Seq_" + str(seq_idx).zfill(5)
                seq_path = self.output_data_path / seq_name
                create_empty_folder(seq_path, ask_overwrite=False)
                print(f"Saving a new sequence to {seq_path}")
                metadata = self.get_sequence_metadata(capture)
                metadata_per_seq.append(metadata)
                seq_names.append(seq_name)
                rgb_frames_paths_per_seq.append([])
                depth_frames_paths_per_seq.append([])
                raw_trans_per_seq.append([])
                raw_rot_per_seq.append([])
            elif self.limit_n_frames > 0 and frame_idx >= self.limit_n_frames:
                # check if we reached or passe te limit of frames per sequence
                # # in this case, just skip the current capture... until we reach the next sequence
                continue
            # extract the current frame data
            rgb_file_path = capture["filename"]
            translation = np.array(capture["sensor"]["translation"])
            rotation = np.array(capture["sensor"]["rotation"])
            annotations = capture["annotations"]
            depth_annotation = [a for a in annotations if a["@type"] == "type.unity.com/unity.solo.DepthAnnotation"][0]
            depth_file_path = depth_annotation["filename"]
            # store the current frame data:
            rgb_frames_paths_per_seq[-1].append(rgb_file_path)
            depth_frames_paths_per_seq[-1].append(depth_file_path)
            raw_trans_per_seq[-1].append(translation)
            raw_rot_per_seq[-1].append(rotation)
        # end for capture in captures
        n_seq = seq_idx + 1
        print(f"Number of extracted sequences: {n_seq}")

        # save the camera poses and depth frames for each sequence
        for i_seq in range(n_seq):
            seq_path = self.output_data_path / seq_names[i_seq]
            metadata = metadata_per_seq[i_seq]
            print(f"Saving sequence #{i_seq} to {seq_path}... ")
            n_frames = len(rgb_frames_paths_per_seq[i_seq])
            time_length = n_frames / metadata["fps"]
            print(f"Number of frames: {n_frames}, Length {time_length:.2f} seconds")

            # save metadata
            metadata_path = seq_path / "meta_data.yaml"
            with metadata_path.open("w") as file:
                yaml.dump(metadata, file)
            print(f"Saved metadata to {metadata_path}")

            # extract the camera poses in our format
            cam_poses = self.get_sequence_cam_poses(
                raw_trans=raw_trans_per_seq[i_seq],
                raw_rot=raw_rot_per_seq[i_seq],
            )

            # infer the egomotions (camera pose changes) from the camera poses:
            egomotions = np_func(infer_egomotions)(cam_poses)

            # extract the depth frames
            z_depth_frames, depth_info = self.get_ground_truth_depth(
                depth_frames_paths=depth_frames_paths_per_seq[i_seq],
                metadata=metadata,
            )

            # Save RGB video
            self.save_rgb_frames(
                seq_path=seq_path,
                rgb_frames_paths=rgb_frames_paths_per_seq[i_seq],
                metadata=metadata,
                save_video=True,
            )
            # save depth info
            with (seq_path / "gt_depth_info.pkl").open("wb") as file:
                pickle.dump(depth_info, file)

            # save depth video
            plot_depth_video(
                depth_frames=z_depth_frames,
                save_path=seq_path / "gt_depth_video",
                fps=metadata["fps"],
            )

            # save h5 file of depth frames and camera poses
            file_path = seq_path / "gt_depth_and_egomotion.h5"
            print(f"Saving depth-maps and camera poses to: {file_path}")
            with h5py.File(file_path, "w") as hf:
                hf.create_dataset("z_depth_map", data=z_depth_frames, compression="gzip")
                hf.create_dataset("cam_poses", data=cam_poses)
                hf.create_dataset("egomotions", data=egomotions)

        print("Done.")

    # --------------------------------------------------------------------------------------------------------------------

    def get_sequence_metadata(self, capture: dict):
        cam_intrinsic = [a for a in capture["annotations"] if a["@type"] == "camDataDef"][0]
        frame_width = cam_intrinsic["pixelWidth"]  # [pixels]
        frame_height = cam_intrinsic["pixelHeight"]  # [pixels]
        # the physical specs of the camera are given in 0.001 Unity distance units
        total_sensor_size_x_mm = cam_intrinsic["sensorSizeX"] * 0.001 * self.UNITY_TO_MM  # [mm]
        total_sensor_size_y_mm = cam_intrinsic["sensorSizeY"] * 0.001 * self.UNITY_TO_MM  # [mm]
        focal_length_mm = cam_intrinsic["focalLength"] * 0.001 * self.UNITY_TO_MM  # [mm]
        min_vis_z_mm = cam_intrinsic["nearClipPlane"] * self.UNITY_TO_MM  # [mm]
        assert cam_intrinsic["lensShiftX"] == 0 and cam_intrinsic["lensShiftY"] == 0, "lens shift is not supported"
        frame_time_interval = cam_intrinsic["simulationDeltaTime"]  # [sec]
        fps = 1 / frame_time_interval  # [Hz]
        if self.fps_override != 0:
            fps = self.fps_override

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
    def get_sequence_cam_poses(self, raw_trans: list, raw_rot: list) -> np.ndarray:
        """
        save the 6-DOF camera poses in the world coordinates as a numpy array with shape (N, 7) where N is the number of frames.
        the 7 values are: x, y, z, q_w, q_x, q_y, q_z
        (x, y, z) is the camera position in the world system (in mm)
        (q_w, q_x, q_y, q_z) is a  unit-quaternion has the real part as the first value, that represents the camera rotation w.r.t. the world system.
        Args:
            raw_trans: a list of 3D translation vectors (x, y, z) in Unity units
            raw_rot: a list of 4D rotation unit-quaternion in Unity units
        Returns:
            cam_poses: the camera-poses in out format, a numpy array with shape (N, 7) where N is the number of frames.
        """

        cam_trans = np.row_stack(raw_trans)
        cam_rot = np.row_stack(raw_rot)

        # change quaternion to real-first format:
        cam_rot = cam_rot[:, [3, 0, 1, 2]]

        # change the units from Unity units to mm
        cam_trans *= self.UNITY_TO_MM

        cam_trans, cam_rot = change_cam_transform_to_right_handed(cam_trans, cam_rot)

        cam_poses = np.concatenate((cam_trans, cam_rot), axis=1)
        return cam_poses

    # --------------------------------------------------------------------------------------------------------------------
    def save_rgb_frames(self, rgb_frames_paths: list, seq_path: Path, metadata: dict, save_video: bool = True):
        n_frames = len(rgb_frames_paths)

        # frame loader function
        def load_rgb_frame(i_frame) -> np.ndarray:
            frame_path = rgb_frames_paths[i_frame]
            im = cv2.imread(path_to_str(self.input_data_path / frame_path))
            im = change_image_to_right_handed(im)
            return im
        
        # copy all the rgb frames to the output directory
        frames_out_path = seq_path / "RGB_Frames"
        create_empty_folder(frames_out_path, ask_overwrite=False)
        n_frames = len(rgb_frames_paths)
        for i_frame in range(n_frames):
            im = load_rgb_frame(i_frame)
            frame_name = f"{i_frame:06d}.png"
            # save the image
            cv2.imwrite(
                filename=path_to_str(frames_out_path / frame_name),
                img=im,
            )
        if save_video:
            output_vid_path = seq_path / "Video"
            save_video_from_func(save_path=output_vid_path, make_frame=load_rgb_frame, n_frames=n_frames, fps=metadata["fps"])

    # --------------------------------------------------------------------------------------------------------------------

    def get_ground_truth_depth(
        self,
        depth_frames_paths: list,
        metadata: dict,
    ):
        # Get the depth maps
        n_frames = len(depth_frames_paths)
        frame_height = metadata["frame_height"]
        frame_width = metadata["frame_width"]
        z_depth_frames = np.zeros((n_frames, frame_height, frame_width), dtype=np.float32)
        for i_frame in range(n_frames):
            depth_file_path = self.input_data_path / depth_frames_paths[i_frame]
            print(f"Loading depth frame {i_frame}/{n_frames}", end="\r")
            # All 3 channels are the same (depth), so we only need to read one
            depth_img = cv2.imread(path_to_str(depth_file_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            depth_img = change_image_to_right_handed(depth_img)
            R_channel_idx = 2  # OpenCV reads in BGR order
            z_depth_mm = self.UNITY_TO_MM * depth_img[:, :, R_channel_idx]  # z-depth is stored in the R channel
            z_depth_frames[i_frame] = z_depth_mm

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
            "depth_map_size": {"width": frame_width, "height": frame_height},
        }
        return z_depth_frames, depth_info

    # --------------------------------------------------------------------------------------------------------------------

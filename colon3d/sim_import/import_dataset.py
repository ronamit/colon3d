import argparse
import os
from pathlib import Path

import h5py
import numpy as np

from colon3d.sim_import import loader_ColonNav, loader_SimCol3D, loader_Zhang22
from colon3d.util.data_util import get_all_scenes_paths_in_dir
from colon3d.util.general_util import (
    ArgsHelpFormatter,
    bool_arg,
    create_empty_folder,
    load_rgb_image,
    save_depth_video,
    save_dict_to_yaml,
    save_rgb_image,
    save_video_from_func,
)
from colon3d.util.torch_util import np_func, to_default_type
from colon3d.util.transforms_util import compose_poses, get_identity_pose, infer_egomotions
from colon3d.visuals.plots_3d_scene import plot_3d_trajectories

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # for reading EXR files

# --------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--raw_sim_data_path",
        type=str,
        default="data_gcp/raw_datasets/SimCol3D",  #  ColonNav/Test | ColonNav/Train | Zhang22| SimCol3D |
        help="The path to the folder with the raw simulated scenes to load (generated by Unity).",
    )
    parser.add_argument(
        "--path_to_save_data",
        type=str,
        default="data/datasets/SimCol3D",  #  ColonNav/Test | ColonNav/Train | Zhang22| SimCol3D |
        help="The path to the folder where the processed simulated scenes will be saved.",
    )
    parser.add_argument(
        "--sim_name",
        type=str,
        choices=["Zhang22", "ColonNav", "SimCol3D"],
        default="SimCol3D",
        help="The name of the simulator that generated the data.",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        choices=["Train", "Test"],
        default="test",
        help="The name of the split to load (Train or Test). relevant for  SimCold3D dataset.",
    )
    parser.add_argument(
        "--limit_n_scenes",
        type=int,
        default=0,
        help="The number maximal number of scenes to process, if 0 all scenes will be processed",
    )
    parser.add_argument(
        "--limit_n_frames",
        type=int,
        default=0,
        help="The number maximal number of frames to take from each scenes, if 0 all frames will be processed",
    )
    parser.add_argument(
        "--fps_override",
        type=float,
        default=0.0,
        help="If >0 then the FPS of the output videos will be set to this value",
    )
    parser.add_argument(
        "--world_sys_to_first_cam",
        type=bool_arg,
        default=False,
        help="If True, the world coordinate system will be set to the first camera pose (the first camera pose will be at zero location and unit rotation)",
    )
    parser.add_argument(
        "--save_overwrite",
        type=bool_arg,
        default=True,
        help="If True,the output folder will be overwritten if it already exists",
    )
    args = parser.parse_args()
    print(f"args={args}")
    sim_importer = SimImporter(
        load_path=args.raw_sim_data_path,
        split_name=args.split_name,
        save_path=args.path_to_save_data,
        limit_n_scenes=args.limit_n_scenes,
        limit_n_frames=args.limit_n_frames,
        world_sys_to_first_cam=args.world_sys_to_first_cam,
        fps_override=args.fps_override,
        save_overwrite=args.save_overwrite,
        sim_name=args.sim_name,
    )
    sim_importer.run()


# -------------------------------------------------------------------------------------------------------------------


class SimImporter:
    # --------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        load_path: str,
        save_path: str,
        split_name: str = "Test",
        limit_n_scenes: int = 0,
        limit_n_frames: int = 0,
        world_sys_to_first_cam: bool = False,
        fps_override: float = 0.0,
        save_overwrite: bool = True,
        sim_name: str = "ColonNav",
    ):
        """Imports the simulated scenes from the raw data to the processed data format.
        Args:
            raw_sim_data_path: The path to the folder with the raw simulated scenes to load (generated by Unity).
            path_to_save_data: The path to the folder where the processed simulated scenes will be saved.
            limit_n_scenes: The number maximal number of scenes to process, if 0 all scenes will be processed.
            limit_n_frames: The number maximal number of frames to take from each scenes, if 0 all frames will be processed
            world_sys_to_first_cam: If True, the world coordinate system will be set to the first camera pose.
            fps_override: frame rate in Hz of the output videos, if 0 the frame rate will be extracted from the settings file.
        """
        self.raw_sim_data_path = Path(load_path)
        self.path_to_save_data = Path(save_path)
        print("Raw simulated scenes will be be loaded from: ", self.path_to_save_data)
        print("Path to save scenes: ", self.path_to_save_data)
        self.limit_n_frames = limit_n_frames
        self.limit_n_scenes = limit_n_scenes
        self.world_sys_to_first_cam = world_sys_to_first_cam
        self.fps_override = fps_override
        self.save_overwrite = save_overwrite
        self.sim_name = sim_name
        self.split_name = split_name
        print("Simulator: ", self.sim_name)

    # --------------------------------------------------------------------------------------------------------------------

    def run(self):
    
        is_created = create_empty_folder(self.path_to_save_data, save_overwrite=self.save_overwrite)
        if not is_created:
            # in this case the dataset already exists, so we will load it instead of creating it
            scenes_paths = get_all_scenes_paths_in_dir(self.path_to_save_data, with_targets=True)
            print(f"Loading existing dataset from {self.path_to_save_data}...\n" + "-" * 50)
            return scenes_paths

        if self.sim_name == "ColonNav":
            (
                scenes_names,
                metadata_per_scene,
                rgb_frames_paths_per_scene,
                cam_poses_per_scene,
                depth_frames_paths_per_scene,
            ) = loader_ColonNav.load_sim_raw(
                input_data_path=self.raw_sim_data_path,
                limit_n_scenes=self.limit_n_scenes,
                limit_n_frames=self.limit_n_frames,
                fps_override=self.fps_override,
            )
        elif self.sim_name == "Zhang22":
            (
                scenes_names,
                metadata_per_scene,
                rgb_frames_paths_per_scene,
                cam_poses_per_scene,
            ) = loader_Zhang22.load_sim_raw(
                input_data_path=self.raw_sim_data_path,
                limit_n_scenes=self.limit_n_scenes,
                limit_n_frames=self.limit_n_frames,
                fps_override=self.fps_override,
                cam_to_load="left",  # "left" or "right" camera (we only use one camera)
            )
            depth_frames_paths_per_scene = None  # the Zhang22 dataset does not have GT depth frames

        elif self.sim_name == "SimCol3D":
            (
                scenes_names,
                metadata_per_scene,
                rgb_frames_paths_per_scene,
                cam_poses_per_scene,
                depth_frames_paths_per_scene,
            ) = loader_SimCol3D.load_sim_raw(
                input_data_path=self.raw_sim_data_path,
                split_name=self.split_name,
                limit_n_scenes=self.limit_n_scenes,
                limit_n_frames=self.limit_n_frames,
                fps_override=self.fps_override,
            )
        else:
            raise NotImplementedError(f"Unknown sim_name={self.sim_name}")

        n_scenes = len(scenes_names)
        # save the camera poses and depth frames for each scene
        scenes_paths = []
        for i_scene in range(n_scenes):
            scene_path = self.path_to_save_data / scenes_names[i_scene]
            create_empty_folder(scene_path, save_overwrite=True)
            print(f"Saving a new scene to {scene_path}")
            scenes_paths.append(scene_path)
            metadata = metadata_per_scene[i_scene]
            cam_poses = cam_poses_per_scene[i_scene]
            rgb_frames_paths = rgb_frames_paths_per_scene[i_scene]
            n_frames = len(rgb_frames_paths)
            print(f"Saving scene #{i_scene} to {scene_path}... ")
            time_length = n_frames / metadata["fps"]
            print(f"Number of frames: {n_frames}, Length {time_length:.2f} seconds")

            # save metadata
            save_dict_to_yaml(save_path=scene_path / "meta_data.yaml", dict_to_save=metadata)

            if self.world_sys_to_first_cam:
                print("Rotating world system to set first cam pose at zero location and unit rotation.")
                # infer the egomotions (camera pose changes) from the camera poses:
                egomotions = np_func(infer_egomotions)(cam_poses)

                # get the new camera poses, by composing the egomotions starting from identity pose:
                cam_poses[0] = np_func(get_identity_pose)()

                for i_frame in range(1, n_frames):
                    cam_poses[i_frame] = np_func(compose_poses)(pose1=cam_poses[i_frame - 1], pose2=egomotions[i_frame])

            # infer the egomotions (camera pose changes) from the camera poses:
            egomotions = np_func(infer_egomotions)(cam_poses)

            # Save RGB frames and video
            save_rgb_frames(
                raw_sim_data_path=self.raw_sim_data_path,
                scene_path=scene_path,
                rgb_frames_paths=rgb_frames_paths,
                fps=metadata["fps"],
                save_video=True,
            )

            # Get the GT depth frames (if available) and save them
            if self.sim_name in ["ColonNav", "SimCol3D"]:
                # extract the GT depth frames
                if self.sim_name == "ColonNav":
                    z_depth_frames = loader_ColonNav.get_ground_truth_depth(
                        input_data_path=self.raw_sim_data_path,
                        depth_frames_paths=depth_frames_paths_per_scene[i_scene],
                        metadata=metadata,
                    )
                else:
                    z_depth_frames = loader_SimCol3D.get_ground_truth_depth(
                        input_data_path=self.raw_sim_data_path,
                        depth_frames_paths=depth_frames_paths_per_scene[i_scene],
                        metadata=metadata,
                    )
                # save depth video
                save_depth_video(
                    depth_frames=z_depth_frames,
                    save_path=scene_path / "gt_depth_video",
                    fps=metadata["fps"],
                )
            else:
                z_depth_frames = None
                print("No ground-truth depth data available.")

            # save h5 file of depth frames and camera poses
            file_path = scene_path / "gt_3d_data.h5"
            print(f"Saving depth-maps and camera poses to: {file_path}")
            with h5py.File(file_path.resolve(), "w") as hf:
                hf.create_dataset("cam_poses", data=to_default_type(cam_poses))
                hf.create_dataset("egomotions", data=to_default_type(egomotions))
                if z_depth_frames is not None:
                    hf.create_dataset(
                        "z_depth_map",
                        data=to_default_type(z_depth_frames, num_type="float_m"),
                        compression="gzip",
                    )

            # plot the camera trajectory
            plot_3d_trajectories(
                trajectories=cam_poses,
                save_path=scene_path / "gt_cam_trajectory.html",
            )

        print(f"Done creating {n_scenes} scenes in {self.path_to_save_data}")
        return scenes_paths


# --------------------------------------------------------------------------------------------------------------------


def save_rgb_frames(
    raw_sim_data_path: Path,
    rgb_frames_paths: list,
    scene_path: Path,
    fps: float = 0.1,
    save_video: bool = True,
):
    n_frames = len(rgb_frames_paths)

    # frame loader function
    def load_rgb_frame(i_frame) -> np.ndarray:
        frame_path = raw_sim_data_path / rgb_frames_paths[i_frame]
        assert frame_path.exists(), f"File {frame_path} does not exist"
        im = load_rgb_image(frame_path)
        return im

    # copy all the rgb frames to the output directory
    frames_out_path = scene_path / "RGB_Frames"
    create_empty_folder(frames_out_path, save_overwrite=False)
    n_frames = len(rgb_frames_paths)
    for i_frame in range(n_frames):
        im = load_rgb_frame(i_frame)
        frame_name = f"{i_frame:06d}.png"
        save_rgb_image(img=im, save_path=frames_out_path / frame_name)

    if save_video:
        output_vid_path = scene_path / "Video"
        save_video_from_func(
            save_path=output_vid_path,
            make_frame=load_rgb_frame,
            n_frames=n_frames,
            fps=fps,
        )


# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------------------------

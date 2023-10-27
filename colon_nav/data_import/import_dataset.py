import argparse
import os
from pathlib import Path

import h5py
import numpy as np

from colon_nav.data_import import loader_ColonNav, loader_SimCol3D
from colon_nav.data_import.create_target_cases import TargetCasesCreator
from colon_nav.util.data_util import get_all_scenes_paths_in_dir
from colon_nav.util.general_util import (
    ArgsHelpFormatter,
    bool_arg,
    create_empty_folder,
    load_rgb_image,
    save_depth_video,
    save_dict_to_yaml,
    save_rgb_image,
    save_video_from_func,
)
from colon_nav.util.pose_transforms import compose_poses, get_identity_pose, infer_egomotions
from colon_nav.util.torch_util import np_func, to_default_type
from colon_nav.visuals.plots_3d_scene import plot_3d_trajectories

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # for reading EXR files

# --------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--sim_name",
        type=str,
        default="ColonNav",
        choices=["ColonNav", "SimCol3D"],
        help="The name of the simulator that generated the data.",
    )
    parser.add_argument(
        "--load_dataset_path",
        type=str,
        default="data/raw_datasets/ColonNav",  #  ColonNav  | SimCol3D |
        help="The path to the folder with the raw simulated scenes to load.",
    )
    parser.add_argument(
        "--save_dataset_path",
        type=str,
        default="data/datasets/ColonNav",  #  ColonNav | SimCol3D |
        help="The path to the folder where the processed dataset will be saved.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="Train,Test",
        help="The splits subfolders to load and save (Train, Test, or both), separated by comma.",
    )
    parser.add_argument(
        "--n_cases_per_scene",
        type=int,
        default=5,
        help="The number of cases with targets to create for each scene (for the ColonNav dataset).",
    )
    parser.add_argument(
        "--rand_seed",
        type=int,
        default=0,
        help="The random seed to use for creating the cases with targets (for the ColonNav dataset).",
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
        default=True,
        help="If True, the world coordinate system will be set to the first camera pose (the first camera pose will be at zero location and unit rotation)",
    )
    parser.add_argument(
        "--debug_mode",
        type=bool_arg,
        default=False,
        help="If True, the script will run in debug mode",
    )
    args = parser.parse_args()
    # --------------------------------------------------------------------------------------------------------------------

    save_dataset_path = Path(args.save_dataset_path)
    limit_n_scenes = 0  # num cases to run the algorithm on
    limit_n_frames = 0  # num frames to run the algorithm on from each scene.
    n_cases_per_scene = args.n_cases_per_scene

    if args.debug_mode:
        print("Running in debug mode!!!!")
        limit_n_scenes = 1  # num cases to run the algorithm on
        limit_n_frames = 0  # num frames to run the algorithm on from each scene.
        n_cases_per_scene = 1
        save_dataset_path = save_dataset_path / "debug"

    print(f"args={args}")
    sim_importer = SimImporter(
        sim_name=args.sim_name,
        load_dataset_path=args.load_dataset_path,
        save_dataset_path=save_dataset_path,
        splits=args.splits,
        n_cases_per_scene=n_cases_per_scene,
        rand_seed=args.rand_seed,
        limit_n_scenes=limit_n_scenes,
        limit_n_frames=limit_n_frames,
        world_sys_to_first_cam=args.world_sys_to_first_cam,
        fps_override=args.fps_override,
    )
    save_scene_paths_per_split = sim_importer.run()
    print(f"Done importing the splits: {list(save_scene_paths_per_split.keys())}.")


# -------------------------------------------------------------------------------------------------------------------


class SimImporter:
    # --------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        sim_name: str,
        load_dataset_path: str,
        save_dataset_path: str,
        splits: str = "Train,Test",
        n_cases_per_scene: int = 5,
        rand_seed: int = 0,
        limit_n_scenes: int = 0,
        limit_n_frames: int = 0,
        world_sys_to_first_cam: bool = True,
        fps_override: float = 0.0,
    ):
        """Imports the simulated scenes from the raw data to the processed data format.
        Args:
            load_dataset_path: The path to the folder with the raw simulated scenes to load (generated by Unity).
            save_dataset_path: The path to the folder where the processed simulated scenes will be saved.
            limit_n_scenes: The number maximal number of scenes to process, if 0 all scenes will be processed.
            limit_n_frames: The number maximal number of frames to take from each scenes, if 0 all frames will be processed
            world_sys_to_first_cam: If True, the world coordinate system will be set to the first camera pose.
            fps_override: frame rate in Hz of the output videos, if 0 the frame rate will be extracted from the settings file.
        """
        self.load_dataset_path = Path(load_dataset_path)
        self.save_dataset_path = Path(save_dataset_path)
        self.limit_n_frames = limit_n_frames
        self.limit_n_scenes = limit_n_scenes
        self.world_sys_to_first_cam = world_sys_to_first_cam
        self.fps_override = fps_override
        self.sim_name = sim_name
        self.n_cases_per_scene = n_cases_per_scene
        self.rand_seed = rand_seed
        self.splits_list = splits.split(",")
        assert len(self.splits_list) > 0, "No splits were given"
        assert all(
            split_name in ["Train", "Test"] for split_name in self.splits_list
        ), f"Unknown split name in {self.splits_list}"
        print(f"Data source: {self.sim_name}, splits: {self.splits_list}")
        print("Raw dataset will be loaded from: ", self.load_dataset_path)
        print("Processed dataset will be saved to: ", self.save_dataset_path)

    # --------------------------------------------------------------------------------------------------------------------

    def run(self):
        save_scene_paths_per_split = {}
        for split_name in self.splits_list:
            print(f"Processing split: {split_name}")
            save_scene_paths_per_split[split_name] = self.import_split(split_name)

        if self.sim_name == "ColonNav":
            print(f"Creating cases with targets for each scene in the ColonNav dataset at {self.save_dataset_path}.")
            cases_creator = TargetCasesCreator(
                sim_data_path=self.save_dataset_path,
                n_cases_per_scene=self.n_cases_per_scene,
                rand_seed=self.rand_seed,
            )
            cases_creator.run()

        return save_scene_paths_per_split

    # --------------------------------------------------------------------------------------------------------------------

    def import_split(self, split_name):
        # The path to save this split of the dataset
        save_split_path = self.save_dataset_path / split_name
        is_created = create_empty_folder(save_split_path, save_overwrite=True)
        if not is_created:
            # in this case the dataset already exists, so we will load it instead of creating it
            scenes_save_paths = get_all_scenes_paths_in_dir(save_split_path, with_targets=True)
            print(f"Loading existing dataset from {save_split_path}...\n" + "-" * 50)
            return scenes_save_paths

        # Extract the data from the raw data folder
        if self.sim_name == "ColonNav":
            load_split_path = self.load_dataset_path / split_name
            (
                scenes_names,
                metadata_per_scene,
                load_rgb_frames_paths_per_scene,
                cam_poses_per_scene,
                depth_frames_paths_per_scene,
            ) = loader_ColonNav.load_sim_raw(
                load_base_path=load_split_path,
                limit_n_scenes=self.limit_n_scenes,
                limit_n_frames=self.limit_n_frames,
                fps_override=self.fps_override,
            )
        elif self.sim_name == "SimCol3D":
            (
                scenes_names,
                metadata_per_scene,
                load_rgb_frames_paths_per_scene,
                cam_poses_per_scene,
                depth_frames_paths_per_scene,
            ) = loader_SimCol3D.load_sim_raw(
                load_dataset_path=self.load_dataset_path,
                split_name=split_name,
                limit_n_scenes=self.limit_n_scenes,
                limit_n_frames=self.limit_n_frames,
                fps_override=self.fps_override,
            )
        else:
            raise NotImplementedError(f"Unknown sim_name={self.sim_name}")

        # save the camera poses and depth frames for each scene
        n_scenes = len(scenes_names)
        # The paths to save the scenes
        scenes_save_paths = [save_split_path / scene_name for scene_name in scenes_names]
        for i_scene, save_scene_path in enumerate(scenes_save_paths):
            create_empty_folder(save_scene_path, save_overwrite=True)
            print(f"Saving scene #{i_scene} to {save_scene_path}... ")
            metadata = metadata_per_scene[i_scene]
            cam_poses = cam_poses_per_scene[i_scene]
            load_rgb_frames_paths = load_rgb_frames_paths_per_scene[i_scene]
            n_frames = len(load_rgb_frames_paths)
            time_length = n_frames / metadata["fps"]
            print(f"Number of frames: {n_frames}, Length {time_length:.2f} [sec]")
            save_dict_to_yaml(save_path=save_scene_path / "meta_data.yaml", dict_to_save=metadata)

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
                load_rgb_frames_paths=load_rgb_frames_paths,
                save_scene_path=save_scene_path,
                fps=metadata["fps"],
                save_video=True,
            )

            # Get the GT depth frames and save them
            if self.sim_name == "ColonNav":
                z_depth_frames = loader_ColonNav.get_ground_truth_depth(
                    depth_frames_paths=depth_frames_paths_per_scene[i_scene],
                    metadata=metadata,
                )
            elif self.sim_name == "SimCol3D":
                z_depth_frames = loader_SimCol3D.get_ground_truth_depth(
                    depth_frames_paths=depth_frames_paths_per_scene[i_scene],
                    metadata=metadata,
                )
            else:
                raise NotImplementedError(f"Unknown sim_name={self.sim_name}")
            # save depth video
            save_depth_video(
                depth_frames=z_depth_frames,
                save_path=save_scene_path / "gt_depth_video",
                fps=metadata["fps"],
                save_plots=True,
            )

            # save h5 file of depth frames and camera poses
            file_path = save_scene_path / "gt_3d_data.h5"
            print(f"Saving depth-maps and camera poses to: {file_path}")
            with h5py.File(file_path.resolve(), "w") as hf:
                hf.create_dataset("cam_poses", data=to_default_type(cam_poses))
                hf.create_dataset("egomotions", data=to_default_type(egomotions))
                if z_depth_frames is not None:
                    hf.create_dataset(
                        "z_depth_map",
                        data=to_default_type(z_depth_frames, num_type="float_m"),
                    )

            # plot the camera trajectory
            plot_3d_trajectories(
                trajectories=cam_poses,
                save_path=save_scene_path / "gt_cam_trajectory.html",
            )

        print(f"Done creating {n_scenes} scenes in {save_split_path}")
        # if self.debug_mode:
        #     example_frame_idx = 55
        #     shifts = [-2, -1, 0]
        #     save_rgb_and_depth_subplots(
        #         rgb_imgs=[load_rgb_image(load_rgb_frames_paths[example_frame_idx + shift]) for shift in shifts],
        #         depth_imgs=[z_depth_frames[example_frame_idx + shift] for shift in shifts],
        #         frame_names=[f"Shift: {shift}" for shift in shifts],
        #         save_path=Path("temp") / "rgb_depth_example.png",
        #     )
        #     raise ValueError("DEBUG MODE")
        return scenes_save_paths


# --------------------------------------------------------------------------------------------------------------------


def save_rgb_frames(
    load_rgb_frames_paths: list,
    save_scene_path: Path,
    fps: float,
    save_video: bool = True,
):
    """
    Save the RGB frames of a scene to a folder and optionally save a video of the frames.
    Args:
        load_dataset_path:  The path to the raw dataset to load.
        load_rgb_frames_paths: The paths to the RGB frames of this scene.
        save_scene_path: The path to the folder where the scene will be saved.
        fps: The frame rate of the output video, if 0 the frame rate will be extracted from the settings file.
        save_video: If True, a video of the frames will be saved.
    """
    n_frames = len(load_rgb_frames_paths)

    # frame loader function
    def load_rgb_frame(i_frame) -> np.ndarray:
        load_frame_path = load_rgb_frames_paths[i_frame]
        assert load_frame_path.exists(), f"File {load_frame_path} does not exist"
        im = load_rgb_image(load_frame_path)
        return im

    # copy all the rgb frames to the output directory
    save_frames_path = save_scene_path / "RGB_Frames"
    create_empty_folder(save_frames_path, save_overwrite=False)
    n_frames = len(load_rgb_frames_paths)
    for i_frame in range(n_frames):
        im = load_rgb_frame(i_frame)
        frame_name = f"{i_frame:06d}.png"
        save_rgb_image(img=im, save_path=save_frames_path / frame_name)

    if save_video:
        save_video_path = save_scene_path / "Video"
        save_video_from_func(
            save_path=save_video_path,
            make_frame=load_rgb_frame,
            n_frames=n_frames,
            fps=fps,
        )


# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------------------------

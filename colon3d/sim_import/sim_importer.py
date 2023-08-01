import argparse
import os
import pickle
from pathlib import Path

import cv2
import h5py
import numpy as np

from colon3d.sim_import import sim_load_colon_nav_sim, sim_load_zhang
from colon3d.util.general_util import (
    ArgsHelpFormatter,
    bool_arg,
    create_empty_folder,
    path_to_str,
    save_depth_video,
    save_dict_to_yaml,
    save_video_from_func,
)
from colon3d.util.torch_util import np_func, to_default_type
from colon3d.util.transforms_util import compose_poses, get_identity_pose, infer_egomotions

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # for reading EXR files

# --------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--raw_sim_data_path",
        type=str,
        default="data/raw_sim_data/TestData21",
        help="The path to the folder with the raw simulated scenes to load (generated by Unity).",
    )
    parser.add_argument(
        "--processed_sim_data_path",
        type=str,
        default="data/sim_data/TestData21",
        help="The path to the folder where the processed simulated scenes will be saved.",
    )
    parser.add_argument(
        "--sim_name",
        type=str,
        choices=["Zhang22", "ColonNavSim"],
        default="ColonNavSim",
        help="The name of the simulator that generated the data.",
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
        default=True,
        help="If True, the world coordinate system will be set to the first camera pose",
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
        raw_sim_data_path=args.raw_sim_data_path,
        processed_sim_data_path=args.processed_sim_data_path,
        limit_n_scenes=args.limit_n_scenes,
        limit_n_frames=args.limit_n_frames,
        fps_override=args.fps_override,
        save_overwrite=args.save_overwrite,
        sim_name=args.sim_name,
    )
    sim_importer.run()


# --------------------------------------------------------------------------------------------------------------------


class SimImporter:
    # --------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        raw_sim_data_path: str,
        processed_sim_data_path: str,
        limit_n_scenes: int = 0,
        limit_n_frames: int = 0,
        world_sys_to_first_cam: bool = True,
        fps_override: float = 0.0,
        save_overwrite: bool = True,
        sim_name: str = "ColonNavSim",
    ):
        """Imports the simulated scenes from the raw data to the processed data format.
        Args:
            raw_sim_data_path: The path to the folder with the raw simulated scenes to load (generated by Unity).
            processed_sim_data_path: The path to the folder where the processed simulated scenes will be saved.
            limit_n_scenes: The number maximal number of scenes to process, if 0 all scenes will be processed.
            limit_n_frames: The number maximal number of frames to take from each scenes, if 0 all frames will be processed
            world_sys_to_first_cam: If True, the world coordinate system will be set to the first camera pose.
            fps_override: frame rate in Hz of the output videos, if 0 the frame rate will be extracted from the settings file.
        """
        input_data_path = Path(raw_sim_data_path)
        output_data_path = Path(processed_sim_data_path)
        print("Raw simulated scenes will be be loaded from: ", input_data_path)
        self.input_data_path = input_data_path
        self.output_data_path = output_data_path
        print("Path to save simulated scenes will be saved to: ", self.output_data_path)
        self.limit_n_frames = limit_n_frames
        self.limit_n_scenes = limit_n_scenes
        self.world_sys_to_first_cam = world_sys_to_first_cam
        self.fps_override = fps_override
        self.save_overwrite = save_overwrite
        self.sim_name = sim_name
        print("Simulator: ", self.sim_name)

    # --------------------------------------------------------------------------------------------------------------------

    def run(self):
        is_created = create_empty_folder(self.output_data_path, save_overwrite=self.save_overwrite)
        # check if the dataset already exists
        if not is_created:
            # in this case the dataset already exists, so we will load it instead of creating it
            scenes_paths = [p for p in self.output_data_path.glob("Scene*") if p.is_dir()]
            print(f"Loading existing dataset from {self.output_data_path}...\n" + "-" * 50)
            return scenes_paths

        if self.sim_name == "ColonNavSim":
            (
                scenes_names,
                metadata_per_scene,
                rgb_frames_paths_per_scene,
                cam_poses_per_scene,
                depth_frames_paths_per_scene,
            ) = sim_load_colon_nav_sim.load_sim_raw(
                input_data_path=self.input_data_path,
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
            ) = sim_load_zhang.load_sim_raw(
                input_data_path=self.input_data_path,
                limit_n_scenes=self.limit_n_scenes,
                limit_n_frames=self.limit_n_frames,
                fps_override=self.fps_override,
                cam_to_load="left",  # "left" or "right" camera (we only use one camera)
            )
            depth_frames_paths_per_scene = None  # this dataset does not have GT depth frames
        else:
            raise NotImplementedError(f"Unknown sim_name={self.sim_name}")

        n_scenes = len(scenes_names)
        # save the camera poses and depth frames for each scene
        scenes_paths = []
        for i_scene in range(n_scenes):
            scene_path = self.output_data_path / scenes_names[i_scene]
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
                # infer the egomotions (camera pose changes) from the camera poses:
                egomotions = np_func(infer_egomotions)(cam_poses)

                # get the new camera poses, by composing the egomotions starting from identity pose:
                cam_poses[0] = np_func(get_identity_pose)()

                for i_frame in range(1, n_frames):
                    cam_poses[i_frame] = np_func(compose_poses)(pose1=cam_poses[i_frame - 1], pose2=egomotions[i_frame])

            # infer the egomotions (camera pose changes) from the camera poses:
            egomotions = np_func(infer_egomotions)(cam_poses)

            # Save RGB video
            self.save_rgb_frames(
                scene_path=scene_path,
                rgb_frames_paths=rgb_frames_paths,
                metadata=metadata,
                save_video=True,
            )

            if self.sim_name == "ColonNavSim":
                # extract the depth frames
                z_depth_frames, depth_info = sim_load_colon_nav_sim.get_ground_truth_depth(
                    input_data_path=self.input_data_path,
                    depth_frames_paths=depth_frames_paths_per_scene[i_scene],
                    metadata=metadata,
                )
                # save depth info
                with (scene_path / "gt_depth_info.pkl").open("wb") as file:
                    pickle.dump(depth_info, file)
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

        print(f"Done creating {n_scenes} scenes in {self.output_data_path}")
        return scenes_paths

    # --------------------------------------------------------------------------------------------------------------------
    def save_rgb_frames(self, rgb_frames_paths: list, scene_path: Path, metadata: dict, save_video: bool = True):
        n_frames = len(rgb_frames_paths)

        # frame loader function
        def load_rgb_frame(i_frame) -> np.ndarray:
            frame_path = self.input_data_path / rgb_frames_paths[i_frame]
            assert frame_path.exists(), f"File {frame_path} does not exist"
            im = cv2.imread(path_to_str(frame_path))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            return im

        # copy all the rgb frames to the output directory
        frames_out_path = scene_path / "RGB_Frames"
        create_empty_folder(frames_out_path, save_overwrite=False)
        n_frames = len(rgb_frames_paths)
        for i_frame in range(n_frames):
            im = load_rgb_frame(i_frame)
            frame_name = f"{i_frame:06d}.png"
            # save the image
            cv2.imwrite(
                filename=path_to_str(frames_out_path / frame_name),
                img=cv2.cvtColor(im, cv2.COLOR_RGB2BGR),
            )
        if save_video:
            output_vid_path = scene_path / "Video"
            save_video_from_func(
                save_path=output_vid_path,
                make_frame=load_rgb_frame,
                n_frames=n_frames,
                fps=metadata["fps"],
            )


# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------------------------

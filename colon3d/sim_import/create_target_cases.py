import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np
import yaml
from numpy.random import default_rng

from colon3d.sim_import.simulate_tracks import create_tracks_per_frame, generate_targets
from colon3d.util.data_util import get_all_scenes_paths_in_dir
from colon3d.util.general_util import ArgsHelpFormatter, create_empty_folder, to_str
from colon3d.util.rotations_util import get_random_rot_quat
from colon3d.util.torch_util import np_func, to_default_type, to_numpy
from colon3d.util.transforms_util import compose_poses
from colon3d.visuals.plots_2d import save_frames_with_tracks, save_video_with_tracks

# --------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--sim_data_path",
        type=str,
        default="data/sim_data/SimData18",
        help="The path to the folder with processed simulated scenes to load",
    )
    parser.add_argument(
        "--n_cases_per_scene",
        type=int,
        default=5,
        help="The number of cases with random targets to generate from each scene (with random polyp locations, estimation noise etc.)",
    )
    parser.add_argument("--rand_seed", type=int, default=0, help="The random seed.")

    args = parser.parse_args()
    print(f"args={args}")
    cases_creator = TargetCasesCreator(
        sim_data_path=args.sim_data_path,
        n_cases_per_scene=args.n_cases_per_scene,
        rand_seed=args.rand_seed,
    )
    cases_creator.run()


# --------------------------------------------------------------------------------------------------------------------


class TargetCasesCreator:
    def __init__(
        self,
        sim_data_path: str,
        n_cases_per_scene: int = 5,
        rand_seed: int = 0,
        min_pixels_in_bb: int = 20,
        min_target_radius_mm: float = 1.0,
        max_target_radius_mm: float = 3.0,
        max_dist_from_center_ratio: float = 1.0,
        min_dist_from_center_ratio: float = 0.0,
        min_visible_frames: int = 5,
        min_non_visible_frames: int = 20,
        min_initial_pixels_in_bb: int = 20,
        simulate_depth_and_egomotion_estimation: bool = False,
        depth_noise_std_mm: float = 0.0,
        cam_motion_loc_std_mm: float = 0.0,
        cam_motion_rot_std_deg: float = 0.0,
        save_overwrite: bool = True,
    ):
        """Create cases with random targets (polyps) for each scene in the dataset.
            The target cases of each scene are saved in a subfolder of the scene folder.
        Args:
            sim_data_path: The path to the folder with processed simulated scenes to load.
            n_cases_per_scene: The number of cases with random targets to generate from each scene (with random polyp locations, estimation noise etc.).
            rand_seed: The random seed.
            min_pixels_in_bb: The minimum number of pixels in the bounding box of a target detection.
            min_target_radius_mm: The minimum radius of the simulated targets (polyps).
            max_target_radius_mm: The maximum radius of the simulated targets (polyps).
            max_dist_from_center_ratio: The maximum distance of the polyp from the center of the image (in ratio to the image size).
            min_dist_from_center_ratio: The minimum distance of the polyp from the center of the image (in ratio to the image size).
            min_visible_frames: The minimum number of frames that the polyp is visible in the scene.
            min_non_visible_frames: The minimum number of frames that the polyp is not visible in the scene.
            min_initial_pixels_in_bb: The minimum number of pixels in the bounding box of the first detection of the polyp.
            simulate_depth_and_egomotion_estimation: If True, the depth maps and camera egomotion estimations will be simulated by adding noise to the ground-truth.
            depth_noise_std_mm: The standard deviation of the noise to add to the depth maps (in mm).
            cam_motion_loc_std_mm: The standard deviation of the noise to add to the camera location (in mm).
            cam_motion_rot_std_deg: The standard deviation of the noise to add to the camera rotation (in degrees).
            save_overwrite: If True, the existing folder with the generated cases will be overwritten.
        Returns:
            None
        """

        self.sim_data_path = Path(sim_data_path)
        self.save_overwrite = save_overwrite
        self.rand_seed = rand_seed
        self.n_cases_per_scene = n_cases_per_scene
        self.cases_params = {
            "simulate_depth_and_egomotion_estimation": simulate_depth_and_egomotion_estimation,
            "rand_seed": self.rand_seed,
            "n_cases_per_scene": n_cases_per_scene,
            "min_target_radius_mm": min_target_radius_mm,
            "max_target_radius_mm": max_target_radius_mm,
            "max_dist_from_center_ratio": max_dist_from_center_ratio,
            "min_dist_from_center_ratio": min_dist_from_center_ratio,
            "min_visible_frames": min_visible_frames,
            "min_non_visible_frames": min_non_visible_frames,
            "min_initial_pixels_in_bb": min_initial_pixels_in_bb,
            "min_pixels_in_bb": min_pixels_in_bb,
        }
        if simulate_depth_and_egomotion_estimation:
            print(
                "The depth maps and camera egomotion estimations will be simulated by adding noise to the ground-truth.",
            )
            self.cases_params = self.cases_params | {
                "depth_noise_std_mm": depth_noise_std_mm,
                "cam_motion_loc_std_mm": cam_motion_loc_std_mm,
                "cam_motion_rot_std_deg": cam_motion_rot_std_deg,
            }

    # ----------------------------------------------------------------------------------------------------------------

    def run(self):
        rng = default_rng(self.rand_seed)

        print("The simulated scenes will be be loaded from: ", self.sim_data_path)
        scenes_paths_list = get_all_scenes_paths_in_dir(self.sim_data_path, with_targets=False)
        print(f"Found {len(scenes_paths_list)} scenes.")
        for scene_path in scenes_paths_list:
            generate_cases_from_scene(
                scene_path=scene_path,
                n_cases_per_scene=self.n_cases_per_scene,
                cases_params=self.cases_params,
                save_overwrite=self.save_overwrite,
                rng=rng,
            )


# --------------------------------------------------------------------------------------------------------------------


def generate_cases_from_scene(
    scene_path: Path,
    n_cases_per_scene: int,
    cases_params: dict,
    save_overwrite: bool,
    rng,
    save_images_of_tracks: bool = False,
):
    print(f"Generating cases from scene {scene_path}")
    assert scene_path.is_dir() and scene_path.name.startswith("Scene_"), "Invalid scene path"
    path_to_save_cases = scene_path / "Target_Cases"
    is_created = create_empty_folder(path_to_save_cases, save_overwrite=save_overwrite)
    if not is_created:
        print(f"{path_to_save_cases} already exists, skipping...\n" + "-" * 50)
        return

    # load the ground truth depth maps and camera poses:
    with h5py.File((scene_path / "gt_3d_data.h5").resolve(), "r") as h5f:
        gt_depth_maps = to_numpy(h5f["z_depth_map"][:], num_type="float_m")
        gt_cam_poses = to_numpy(h5f["cam_poses"][:])
        gt_egomotions = to_numpy(h5f["egomotions"][:])

    n_frames = gt_depth_maps.shape[0]
    assert n_frames == gt_cam_poses.shape[0], "The number of frames in the depth maps and camera poses is not equal"
    with (scene_path / "gt_depth_info.pkl").open("rb") as file:
        depth_info = to_numpy(pickle.load(file))

    for i_case in range(n_cases_per_scene):
        scene_name = scene_path.name
        case_name = f"Case_{i_case:04d}"
        print(f"Generating case {i_case+1}/{n_cases_per_scene} for scene {scene_name}")
        # Attempt to generate valid targets (static 3D balls in the world, with random radius and position on the surface of the colon wall)
        n_targets = 1  # we currently want only one tracked object
        print("Generating", n_targets, "targets")
        targets_info = generate_targets(
            n_targets=n_targets,
            gt_depth_maps=gt_depth_maps,
            gt_cam_poses=gt_cam_poses,
            rng=rng,
            depth_info=depth_info,
            cases_params=cases_params,
        )
        if targets_info is None:
            print(f"Failed to generate valid targets for the scene {scene_name}... skipping this scene.")
            break

        # create subfolder for the case:
        case_path = path_to_save_cases / case_name
        create_empty_folder(case_path)
        print(f"Generating cases {i_case+1}/{n_cases_per_scene} for scene {scene_name} to save in {case_path}")

        if cases_params["simulate_depth_and_egomotion_estimation"]:
            print("Generating egomotion and depth estimations")
            est_depth_maps, est_egomotions = get_egomotion_and_depth_estimations(
                gt_depth_maps=gt_depth_maps,
                gt_egomotions=gt_egomotions,
                cases_prams=cases_params,
                rng=rng,
            )
            # save the estimated depth maps and egomotions to a file:
            with h5py.File((case_path / "est_depth_and_egomotion.h5").resolve(), "w") as hf:
                hf.create_dataset(
                    "z_depth_map",
                    data=to_default_type(est_depth_maps, num_type="float_m"),
                    compression="gzip",
                )
                hf.create_dataset("egomotions", data=to_default_type(est_egomotions))
            # save depth info to a file (unchanged from the ground truth):
            with (case_path / "est_depth_info.pkl").open("wb") as file:
                pickle.dump(depth_info, file)

        # get the FPS:
        with (scene_path / "meta_data.yaml").open("r") as file:
            fps = yaml.load(file, Loader=yaml.FullLoader)["fps"]

        print("Targets info:", to_str(targets_info))
        with (case_path / "targets_info.pkl").open("wb") as file:
            pickle.dump(targets_info, file)
        # also save the targets info to a text file:
        with (case_path / "targets_info.txt").open("w") as file:
            file.write("Targets info: " + to_str(targets_info) + "\n")

        # simulate the tracks (bounding boxes) of the targets in the 2D images:
        tracks = create_tracks_per_frame(
            gt_depth_maps=gt_depth_maps,
            gt_cam_poses=gt_cam_poses,
            depth_info=depth_info,
            targets_info=targets_info,
            min_pixels_in_bb=cases_params["min_pixels_in_bb"],
        )
        # save tracks to a csv file:
        tracks.to_csv(case_path / "Tracks.csv", encoding="utf-8-sig", index=False)

        # save video with the tracks bounding boxes (for visualization)
        save_video_with_tracks(
            frames_folder_path=scene_path / "RGB_Frames",
            tracks=tracks,
            path_to_save=case_path / "Video_with_tracks",
            fps=fps,
        )
        # save frames with the tracks bounding boxes (for visualization)
        if save_images_of_tracks:
            save_frames_with_tracks(
                frames_folder_path=scene_path / "RGB_Frames",
                tracks=tracks,
                path_to_save=case_path / "Frames_with_tracks",
            )
    print("Done generating cases from scene", scene_path.name)


# --------------------------------------------------------------------------------------------------------------------


def get_egomotion_and_depth_estimations(
    gt_depth_maps: np.ndarray,
    gt_egomotions: np.ndarray,
    cases_prams: dict,
    rng: np.random.Generator,
):
    depth_noise_std_mm = cases_prams["depth_noise_std_mm"]
    cam_motion_loc_std_mm = cases_prams["cam_motion_loc_std_mm"]
    cam_motion_rot_std_deg = cases_prams["cam_motion_rot_std_deg"]
    n_frames = gt_depth_maps.shape[0]
    if depth_noise_std_mm > 0:
        # add noise to the depth maps:
        est_depth_maps = rng.normal(
            loc=gt_depth_maps,
            scale=depth_noise_std_mm,
        )
    else:
        est_depth_maps = gt_depth_maps
    if cam_motion_loc_std_mm == 0 and cam_motion_rot_std_deg == 0:
        # if no noise is added to the egomotions, then the estimated egomotions are the ground truth egomotions:
        return est_depth_maps, gt_egomotions
    # add noise to the location component of the egomotions:
    loc_err = rng.standard_normal(size=(n_frames, 3)) * cam_motion_loc_std_mm
    # add noise to the rotation component of the egomotions:
    rot_err_quat = get_random_rot_quat(rng=rng, angle_std_deg=cam_motion_rot_std_deg, n_vecs=n_frames)
    err_egomotions = np.concatenate([loc_err, rot_err_quat], axis=1)
    # create the estimated egomotions by applying the error egomotions to the ground truth egomotions:
    est_egomotions = np_func(compose_poses)(pose1=gt_egomotions, pose2=err_egomotions)
    return est_depth_maps, est_egomotions


# --------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()

# --------------------------------------------------------------------------------------------------------------------

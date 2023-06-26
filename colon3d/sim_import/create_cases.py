import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import yaml
from numpy.random import default_rng

from colon3d.sim_import.simulate_tracks import create_tracks_per_frame, generate_targets
from colon3d.utils.general_util import ArgsHelpFormatter, create_empty_folder, save_run_info, to_str
from colon3d.utils.rotations_util import get_random_rot_quat
from colon3d.utils.torch_util import np_func, to_numpy
from colon3d.utils.transforms_util import apply_pose_change
from colon3d.visuals.plots_2d import save_video_with_tracks

# --------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--sim_data_path",
        type=str,
        default="data/sim_data/SimData17",
        help="The path to the folder with processed simulated scenes to load",
    )
    parser.add_argument(
        "--path_to_save_scenes",
        type=str,
        default="data/sim_data/SimData17_cases",
        help="The path to the folder where the generated scenes with targets will be saved",
    )
    parser.add_argument(
        "--n_cases_per_scene",
        type=int,
        default=5,
        help="The number of cases with random targets to generate from each scene (with random polyp locations, estimation noise etc.)",
    )
    parser.add_argument("--rand_seed", type=int, default=0, help="The random seed.")
    parser.add_argument(
        "--min_pixels_in_bb",
        type=int,
        default=20,
        help="The minimum number of pixels in the bounding box of a target detection",
    )
    ## Parameters for simulated targets (polyps) generation
    parser.add_argument(
        "--min_target_radius_mm",
        type=float,
        default=1,
        help="The minimum radius of the simulated targets (polyps)",
    )
    parser.add_argument(
        "--max_target_radius_mm",
        type=float,
        default=3,
        help="The maximum radius of the simulated targets (polyps)",
    )
    parser.add_argument(
        "--max_dist_from_center_ratio",
        type=float,
        default=1.0,
        help="Number in the range [0,1] that determines the maximum distance of the targets (polyps) from the center of the FOV",
    )
    parser.add_argument(
        "--min_dist_from_center_ratio",
        type=float,
        default=0.0,
        help="Number in the range [0,1] that determines the minimum distance of the targets (polyps) from the center of the FOV",
    )
    parser.add_argument(
        "--min_visible_frames",
        type=int,
        default=5,
        help="The minimum number of frames in which the target (polyp) is visible",
    )
    parser.add_argument(
        "--min_non_visible_frames",
        type=int,
        default=20,
        help="The minimum number of frames in which the target (polyp) is not visible",
    )
    parser.add_argument(
        "--min_initial_pixels_in_bb",
        type=int,
        default=15,
        help="The minimum number of pixels in the bounding box of the target (polyp) in the first frame",
    )
    ## Parameters for simulated depth and egomotion estimation
    parser.add_argument(
        "--simulate_depth_and_egomotion_estimation",
        type=bool,
        default=False,
        help="If True, the depth maps and camera egomotion estimations will be simulated by adding noise to the ground-truth",
    )
    parser.add_argument(
        "--depth_noise_std_mm",
        type=float,
        default=0,
        help="The standard deviation of the estimation error added to the depth maps",
    )
    parser.add_argument(
        "--cam_motion_loc_std_mm",
        type=float,
        default=0,
        help="The standard deviation of the estimation error added to the location change component of the camera motion",
    )
    parser.add_argument(
        "--cam_motion_rot_std_deg",
        type=float,
        default=0,
        help="The standard deviation of the estimation error added to the rotation change component of the camera motion",
    )

    args = parser.parse_args()
    n_cases_per_scene = args.n_cases_per_scene
    sim_data_path = Path(args.sim_data_path)
    path_to_save_scenes = Path(args.path_to_save_scenes)
    print(f"The generated cases will be saved to {path_to_save_scenes}")
    create_empty_folder(path_to_save_scenes, ask_overwrite=False)
    save_run_info(path_to_save_scenes, args)
    rng = default_rng(args.rand_seed)
    cases_params = {
        "simulate_depth_and_egomotion_estimation": args.simulate_depth_and_egomotion_estimation,
        "rand_seed": args.rand_seed,
        "n_cases_per_scene": n_cases_per_scene,
        "min_target_radius_mm": args.min_target_radius_mm,
        "max_target_radius_mm": args.max_target_radius_mm,
        "max_dist_from_center_ratio": args.max_dist_from_center_ratio,
        "min_dist_from_center_ratio": args.min_dist_from_center_ratio,
        "min_visible_frames": args.min_visible_frames,
        "min_non_visible_frames": args.min_non_visible_frames,
        "min_initial_pixels_in_bb": args.min_initial_pixels_in_bb,
        "min_pixels_in_bb": args.min_pixels_in_bb,
    }
    if args.simulate_depth_and_egomotion_estimation:
        print("The depth maps and camera egomotion estimations will be simulated by adding noise to the ground-truth.")
        cases_params = cases_params | {
            "depth_noise_std_mm": args.depth_noise_std_mm,
            "cam_motion_loc_std_mm": args.cam_motion_loc_std_mm,
            "cam_motion_rot_std_deg": args.cam_motion_rot_std_deg,
        }

    print("The simulated scnes will be be loaded from: ", sim_data_path)
    scenes_paths_list = [
        scene_path
        for scene_path in sim_data_path.iterdir()
        if scene_path.is_dir() and scene_path.name.startswith("Scene_")
    ]
    scenes_paths_list.sort()
    print(f"Found {len(scenes_paths_list)} scenes.")
    for scene_path in scenes_paths_list:
        print(f"Generating cases from scene {scene_path}")
        generate_cases_from_scene(
            scene_path=scene_path,
            n_cases_per_scene=n_cases_per_scene,
            cases_params=cases_params,
            path_to_save_scenes=path_to_save_scenes,
            rng=rng,
        )
    # save the cases parameters to a json file:
    with (path_to_save_scenes / "cases_prams.json").open("w") as file:
        json.dump(cases_params, file, indent=4)


# --------------------------------------------------------------------------------------------------------------------


def generate_cases_from_scene(
    scene_path: Path,
    n_cases_per_scene: int,
    cases_params: dict,
    path_to_save_scenes: Path,
    rng,
):
    # load the ground truth depth maps and camera poses:
    with h5py.File(scene_path / "gt_depth_and_egomotion.h5", "r") as h5f:
        gt_depth_maps = h5f["z_depth_map"][:]
        gt_cam_poses = to_numpy(h5f["cam_poses"][:])
        gt_egomotions = to_numpy(h5f["egomotions"][:])

    n_frames = gt_depth_maps.shape[0]
    assert n_frames == gt_cam_poses.shape[0], "The number of frames in the depth maps and camera poses is not equal"
    with (scene_path / "gt_depth_info.pkl").open("rb") as file:
        depth_info = to_numpy(pickle.load(file))

    for i_case in range(n_cases_per_scene):
        scene_name = scene_path.name
        case_name = f"{scene_name}_{i_case:04d}"
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
        case_path = path_to_save_scenes / case_name
        create_empty_folder(case_path)
        print(
            f"Generating cases {i_case+1}/{n_cases_per_scene} for scene {scene_name} to save in {case_path}",
            flush=True,
        )
        # create symbolic links in the case folder:
        for file_name in [
            "meta_data.yaml",
            "Video.mp4",
            "gt_depth_info.pkl",
            "gt_depth_and_egomotion.h5",
            "RGB_Frames",
            "gt_depth_video.mp4",
        ]:
            (case_path / file_name).symlink_to((scene_path / file_name).resolve())

        if cases_params["simulate_depth_and_egomotion_estimation"]:
            print("Generating egomotion and depth estimations")
            est_depth_maps, est_egomotions = get_egomotion_and_depth_estimations(
                gt_depth_maps=gt_depth_maps,
                gt_egomotions=gt_egomotions,
                cases_prams=cases_params,
                rng=rng,
            )
            # save the estimated depth maps and egomotions to a file:
            with h5py.File(case_path / "est_depth_and_egomotion.h5", "w") as hf:
                hf.create_dataset("z_depth_map", data=est_depth_maps, compression="gzip")
                hf.create_dataset("egomotions", data=est_egomotions)
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
            rgb_frames_path=case_path / "RGB_Frames",
            tracks=tracks,
            path_to_save=case_path / "Video_with_tracks",
            fps=fps,
        )
        print("Done generating cases from scene", scene_path.name)
    print("Done generating cases from all scenes")


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
    est_egomotions = np_func(apply_pose_change)(start_pose=gt_egomotions, pose_change=err_egomotions)
    return est_depth_maps, est_egomotions


# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

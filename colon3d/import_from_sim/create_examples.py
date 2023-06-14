import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import yaml
from numpy.random import default_rng

from colon3d.general_util import create_empty_folder, to_str
from colon3d.import_from_sim.simulate_tracks import create_tracks_per_frame, generate_targets
from colon3d.rotations_util import get_random_rot_quat
from colon3d.torch_util import np_func
from colon3d.transforms_util import apply_pose_change
from colon3d.visuals.plots_2d import save_video_with_tracks

# --------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sim_data_path",
        type=str,
        default="data/sim_data/SimData8",
        help="The path to the folder with processed simulated sequences to load",
    )
    parser.add_argument(
        "--path_to_save_examples",
        type=str,
        default="data/sim_data/SimData8/Examples",
        help="The path to the folder where the generated examples will be saved",
    )
    parser.add_argument(
        "--n_examples_per_sequence",
        type=int,
        default=5,
        help="The number of examples to generate from each sequence (with random polyp locations, estimation noise etc.)",
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
        default=0.,
        help="Number in the range [0,1] that determines the minimum distance of the targets (polyps) from the center of the FOV",
    )
    parser.add_argument(
        "--min_visible_frames",
        type=int,
        default=1,
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
        default=10,
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
    n_examples_per_sequence = args.n_examples_per_sequence
    sim_data_path = Path(args.sim_data_path)
    path_to_save_examples = Path(args.path_to_save_examples)
    print(f"The generated examples will be saved to {path_to_save_examples}")
    create_empty_folder(path_to_save_examples, ask_overwrite=False)
    rng = default_rng(args.rand_seed)
    examples_prams = {
        "simulate_depth_and_egomotion_estimation": args.simulate_depth_and_egomotion_estimation,
        "rand_seed": args.rand_seed,
        "n_examples_per_sequence": n_examples_per_sequence,
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
        examples_prams = examples_prams | {
            "depth_noise_std_mm": args.depth_noise_std_mm,
            "cam_motion_loc_std_mm": args.cam_motion_loc_std_mm,
            "cam_motion_rot_std_deg": args.cam_motion_rot_std_deg,
        }

    print("The simulated sequences will be be loaded from: ", sim_data_path)
    sequences_paths_list = [
        sequence_path
        for sequence_path in sim_data_path.iterdir()
        if sequence_path.is_dir() and sequence_path.name.startswith("Seq_")
    ]
    sequences_paths_list.sort()
    print(f"Found {len(sequences_paths_list)} sequences.")
    for sequence_path in sequences_paths_list:
        print(f"Generating examples from sequence {sequence_path}")
        generate_examples_from_sequence(
            sequence_path=sequence_path,
            n_examples_per_sequence=n_examples_per_sequence,
            examples_prams=examples_prams,
            path_to_save_examples=path_to_save_examples,
            rng=rng,
        )
    # save the examples parameters to a json file:
    with (path_to_save_examples / "examples_prams.json").open("w") as file:
        json.dump(examples_prams, file, indent=4)


# --------------------------------------------------------------------------------------------------------------------


def generate_examples_from_sequence(
    sequence_path: Path,
    n_examples_per_sequence: int,
    examples_prams: dict,
    path_to_save_examples: Path,
    rng,
):
    # load the ground truth depth maps and camera poses:
    with h5py.File(sequence_path / "gt_depth_and_egomotion.h5", "r") as h5f:
        gt_depth_maps = h5f["z_depth_map"][:]
        gt_cam_poses = h5f["cam_poses"][:]
        gt_egomotions = h5f["egomotions"][:]
    n_frames = gt_depth_maps.shape[0]
    assert n_frames == gt_cam_poses.shape[0], "The number of frames in the depth maps and camera poses is not equal"
    with (sequence_path / "gt_depth_info.pkl").open("rb") as file:
        depth_info = pickle.load(file)

    for i_example in range(n_examples_per_sequence):
        sequence_name = sequence_path.name
        example_name = f"{sequence_name}_{i_example:04d}"
        print(f"Generating example {i_example}/{n_examples_per_sequence} for sequence {sequence_name}")
        # Attempt to generate valid targets (static 3D balls in the world, with random radius and position on the surface of the colon wall)
        n_targets = 1  # we currently want only one tracked object
        print("Generating", n_targets, "targets")
        targets_info = generate_targets(
            n_targets=n_targets,
            gt_depth_maps=gt_depth_maps,
            gt_cam_poses=gt_cam_poses,
            rng=rng,
            depth_info=depth_info,
            examples_prams=examples_prams,
        )
        if targets_info is None:
            print("Failed to generate valid targets for the sequence {sequence_name}... skipping this sequence.")
            break
        
        # create subfolder for the example:
        example_path = path_to_save_examples / example_name
        create_empty_folder(example_path)
        print(
            f"Generating example {i_example}/{n_examples_per_sequence} for sequence {sequence_name} to save in {example_path}",
            flush=True,
        )
        # create symbolic links in the example folder:
        for file_name in [
            "meta_data.yaml",
            "Video.mp4",
            "gt_depth_info.pkl",
            "gt_depth_and_egomotion.h5",
            "RGB_Frames",
            "gt_depth_video.mp4",
        ]:
            (example_path / file_name).symlink_to((sequence_path / file_name).resolve())

        if examples_prams["simulate_depth_and_egomotion_estimation"]:
            print("Generating egomotion and depth estimations")
            est_depth_maps, est_egomotions = get_egomotion_and_depth_estimations(
                gt_depth_maps=gt_depth_maps,
                gt_egomotions=gt_egomotions,
                examples_prams=examples_prams,
                rng=rng,
            )
            # save the estimated depth maps and egomotions to a file:
            with h5py.File(example_path / "est_depth_and_egomotion.h5", "w") as hf:
                hf.create_dataset("z_depth_map", data=est_depth_maps, compression="gzip")
                hf.create_dataset("egomotions", data=est_egomotions)
            # save depth info to a file (unchanged from the ground truth):
            with (example_path / "est_depth_info.pkl").open("wb") as file:
                pickle.dump(depth_info, file)

        # get the FPS:
        with (sequence_path / "meta_data.yaml").open("r") as file:
            fps = yaml.load(file, Loader=yaml.FullLoader)["fps"]

        print("Targets info:", to_str(targets_info))
        with (example_path / "targets_info.pkl").open("wb") as file:
            pickle.dump(targets_info, file)
        # also save the targets info to a text file:
        with (example_path / "targets_info.txt").open("w") as file:
            file.write("Targets info: " + to_str(targets_info) + "\n")

        # simulate the tracks (bounding boxes) of the targets in the 2D images:
        tracks = create_tracks_per_frame(
            gt_depth_maps=gt_depth_maps,
            gt_cam_poses=gt_cam_poses,
            depth_info=depth_info,
            targets_info=targets_info,
            min_pixels_in_bb = examples_prams["min_pixels_in_bb"],
        )
        # save tracks to a csv file:
        tracks.to_csv(example_path / "Tracks.csv", encoding="utf-8-sig", index=False)

        # save video with the tracks bounding boxes (for visualization)
        save_video_with_tracks(
            rgb_frames_path=example_path / "RGB_Frames",
            tracks=tracks,
            path_to_save=example_path / "Video_with_tracks",
            fps=fps,
        )
        print("Done generating examples from sequence", sequence_path.name)
    print("Done generating examples from all sequences")


# --------------------------------------------------------------------------------------------------------------------


def get_egomotion_and_depth_estimations(
    gt_depth_maps: np.ndarray,
    gt_egomotions: np.ndarray,
    examples_prams: dict,
    rng: np.random.Generator,
):
    depth_noise_std_mm = examples_prams["depth_noise_std_mm"]
    cam_motion_loc_std_mm = examples_prams["cam_motion_loc_std_mm"]
    cam_motion_rot_std_deg = examples_prams["cam_motion_rot_std_deg"]
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

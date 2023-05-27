import argparse
import pickle
from pathlib import Path

import h5py
import numpy as np
from numpy.random import default_rng

from colon3d.general_util import create_empty_folder
from colon3d.import_from_sim.simulate_tracks import generate_tracks_gt_3d_loc, get_tracks_detections_per_frame
from colon3d.rotations_util import apply_egomotions_np, get_random_rot_quat, infer_egomotions_np

# --------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence_path",
        type=str,
        default="data/sim_data/Seq_00009_short",
        help="The path to the processed simulated sequence",
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=1,
        help="The number of examples to generate from the sequence",
    )
    parser.add_argument(
        "--depth_noise_std_mm",
        type=float,
        default=0,
        help="The standard deviation of the noise added to the depth maps",
    )
    parser.add_argument(
        "--cam_motion_loc_std_mm",
        type=float,
        default=0,
        help="The standard deviation of the noise added to the location change component of the camera motion",
    )
    parser.add_argument(
        "--cam_motion_rot_std_deg",
        type=float,
        default=0,
        help="The standard deviation of the noise added to the rotation change component of the camera motion",
    )
    parser.add_argument("--rand_seed", type=int, default=0, help="The random seed to use")

    args = parser.parse_args()
    n_examples = args.n_examples
    sequence_path = Path(args.sequence_path)
    rng = default_rng(args.rand_seed)
    print("The simulated sequence will be be loaded from: ", sequence_path)
    examples_dir_path = sequence_path / "Examples"
    print(f"The generated examples will be saved to {examples_dir_path}")
    # load the ground truth depth maps and camera poses:
    with h5py.File(sequence_path / "gt_depth_and_cam_poses.h5", "r") as h5f:
        gt_depth_maps = h5f["z_depth_map"][:]
        gt_cam_poses = h5f["cam_poses"][:]
    n_frames = gt_depth_maps.shape[0]
    assert n_frames == gt_cam_poses.shape[0], "The number of frames in the depth maps and camera poses is not equal"
    with (sequence_path / "depth_info.pkl").open("rb") as file:
        depth_info = pickle.load(file)
    # infer the egomotions (camera pose changes) from the camera poses:
    gt_egomotions = infer_egomotions_np(gt_cam_poses)
    for i_example in range(n_examples):
        # create subfolder for the example:
        example_path = examples_dir_path / f"{i_example:04d}"
        create_empty_folder(example_path)
        print(f"Generating example {i_example} in {example_path}")
        # create symbolic links in the example folder:
        for file_name in ["meta_data.yaml", "Video.mp4", "depth_info.pkl", "gt_depth_and_cam_poses.h5"]:
            (example_path / file_name).symlink_to((sequence_path / file_name).resolve())
        print("Generating egomotion and depth estimations")
        est_depth_maps, est_egomotions = get_egomotion_and_depth_estimations(
            gt_depth_maps=gt_depth_maps,
            gt_egomotions=gt_egomotions,
            depth_noise_std_mm=args.depth_noise_std_mm,
            cam_motion_loc_std_mm=args.cam_motion_loc_std_mm,
            cam_motion_rot_std_deg=args.cam_motion_rot_std_deg,
            rng=rng,
        )
        # save the estimated depth maps and egomotions to a file:
        with h5py.File(example_path / "depth_and_egomotion.h5", "w") as hf:
            hf.create_dataset("z_depth_map", data=est_depth_maps, compression="gzip")
            hf.create_dataset("egomotions", data=est_egomotions)

        # simulate the true locations of the tracks (polyps) in the 3D world:
        n_tracks = 1
        print("Generating", n_tracks, "tracks")
        tracks_info = generate_tracks_gt_3d_loc(
            n_tracks=n_tracks,
            gt_depth_maps=gt_depth_maps,
            gt_cam_poses=gt_cam_poses,
            rng=rng,
            depth_info=depth_info,
        )
        print("Tracks info:", tracks_info)
        with (example_path / "tracks_info.pkl").open("wb") as file:
            pickle.dump(tracks_info, file)

        # simulate the detections of the tracks in the 2D images:
        detections = get_tracks_detections_per_frame(
            gt_depth_maps=gt_depth_maps,
            gt_cam_poses=gt_cam_poses,
            depth_info=depth_info,
            tracks_info=tracks_info,
        )
        detections.to_csv(example_path / "Detections.csv", encoding="utf-8-sig", index=False)


# --------------------------------------------------------------------------------------------------------------------
def get_egomotion_and_depth_estimations(
    gt_depth_maps: np.ndarray,
    gt_egomotions: np.ndarray,
    depth_noise_std_mm: float,
    cam_motion_loc_std_mm: float,
    cam_motion_rot_std_deg: float,
    rng: np.random.Generator,
):
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
    est_egomotions = apply_egomotions_np(gt_egomotions, err_egomotions)
    return est_depth_maps, est_egomotions


# --------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

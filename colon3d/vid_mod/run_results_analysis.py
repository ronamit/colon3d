import argparse
import pickle
from pathlib import Path

from colon3d.util.general_util import ArgsHelpFormatter

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--loaded_results_path",
        type=str,
        default="data/results/real_videos/Mod_Vids",
        help="path to the load the save outputs",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/results/real_videos/Mod_Vids/Analysis",
        help="path to the save outputs",
    )
    args = parser.parse_args()
    loaded_results_path = Path(args.loaded_results_path)
    save_path = Path(args.save_path)

    #  get the per-scale scene folders paths
    all_results_paths = sorted(loaded_results_path.glob("Scale_*"))
    print(f"Found {len(all_results_paths)} scale result folders")

    for result_path in all_results_paths:

        # load the estimated track location in the camera system per frame
        with (result_path / "slam_results.pkl").open("rb") as f:
            slam_results = pickle.load(f)

        online_est_track_world_loc = slam_results["online_est_track_world_loc"]
        online_est_track_cam_loc = slam_results["online_est_track_cam_loc"]
        online_est_track_angle = slam_results["online_est_track_angle"]
        scene_info = slam_results["scene_info"]
        seg_scale = scene_info["seg_scale"]
        new_segments = scene_info["new_segments"]
        is_in_view_new = scene_info["is_in_view_new"]
        alg_fov_ratio = scene_info["alg_fov_ratio"]

        n_frames = len(online_est_track_world_loc)

        for i_seg, seg in enumerate(new_segments):
            pass

        print("-" * 100)
        print(f"Scale {seg_scale} results")
        print(f"New segments: {new_segments}")

# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
# ---------------------------------------------------------------------------------------------------------------------

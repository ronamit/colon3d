import argparse
import pickle
from pathlib import Path

from colon3d.run_on_scene import SlamRunner
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

    for result_path in all_results_paths:
        # get info about the scene:
        with (result_path / "out_of_view_info.pkl").open("rb") as f:
            scene_info = pickle.load(f)
            scale = scene_info["scale"]
            # new_segments = info_dict["new_segments"]
            is_in_view_new = scene_info["is_in_view_new"]
            alg_fov_ratio = scene_info["alg_fov_ratio"] if "alg_fov_ratio" in scene_info else 0.8
 
        # load the estimated track location in the camera system per frame
        with pickle.open(result_path / "slam_results.pkl", "rb") as f:
            slam_results = pickle.load(f)
        online_est_track_world_loc = slam_results["online_est_track_world_loc"]
        online_est_track_cam_loc = slam_results["online_est_track_cam_loc"]
        online_est_track_angle = slam_results["online_est_track_angle"]
 
# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
# ---------------------------------------------------------------------------------------------------------------------

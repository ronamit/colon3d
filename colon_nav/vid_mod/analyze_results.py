import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from colon_nav.alg.tracks_loader import DetectionsTracker
from colon_nav.util.data_util import SceneLoader
from colon_nav.util.general_util import ArgsHelpFormatter, save_plot_and_close

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
        "--loaded_scenes_path",
        type=str,
        default="data/datasets/real_videos/Mod_Vids",
        help="base path to the load the modified scenes",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="data/results/real_videos/Mod_Vids/Analysis",
        help="path to the save outputs",
    )
    args = parser.parse_args()
    loaded_results_path = Path(args.loaded_results_path)
    loaded_scenes_path = Path(args.loaded_scenes_path)
    save_path = Path(args.save_path)

    #  get the per-scale scene folders paths
    all_results_paths = sorted(loaded_results_path.glob("Scale_*"))
    print(f"Found {len(all_results_paths)} scale result folders")

    out_of_view_avg_seg_time = []
    avg_err_deg = []
    std_err_deg = []
    n_seg = []
    for result_path in all_results_paths:
        scene_name = result_path.name
        scene_path = loaded_scenes_path / scene_name
        _avg_err_deg, _std_err_deg, _n_seg, _out_of_view_avg_seg_time = analyze_scene_result(result_path, scene_path)
        avg_err_deg.append(_avg_err_deg)
        std_err_deg.append(_std_err_deg)
        n_seg.append(_n_seg)
        out_of_view_avg_seg_time.append(_out_of_view_avg_seg_time)

    # plot the average error per scale with 95% confidence interval

    avg_err_deg = np.array(avg_err_deg)
    std_err_deg = np.array(std_err_deg)
    n_seg = np.array(n_seg)
    out_of_view_avg_seg_time = np.array(out_of_view_avg_seg_time)
    err_confidence = 1.96 * std_err_deg / np.sqrt(n_seg)

    plt.figure()
    plt.errorbar(out_of_view_avg_seg_time, avg_err_deg, yerr=err_confidence, fmt="o")
    plt.xlabel("Avg. out-of-view segment length [Sec]")
    plt.ylabel("Average error [Deg]")
    plt.title(f"Average estimation error of the target angle before return to view. \n over {int(n_seg.mean())} segments (95% CI)")
    plt.grid()
    save_plot_and_close(save_path / "avg_err_per_scale.png")


# ---------------------------------------------------------------------------------------------------------------------


def analyze_scene_result(result_path: Path, scene_path: Path):
    # load the estimated track location in the camera system per frame
    with (result_path / "slam_results.pkl").open("rb") as f:
        slam_results = pickle.load(f)
    online_est_track_angle = slam_results["online_est_track_angle"]
    scene_info = slam_results["scene_info"]
    seg_scale = scene_info["seg_scale"]
    new_segments = scene_info["new_segments"]
    alg_fov_ratio = scene_info["alg_fov_ratio"]
    seg_scale = scene_info["seg_scale"]
    is_in_view = scene_info["is_in_view_new"]
    n_segments = len(new_segments)

    # get the ground-truth tracks info in the full frame
    scene_loader = SceneLoader(scene_path=scene_path, alg_fov_ratio=alg_fov_ratio)
    detections_tracker = DetectionsTracker(scene_path=scene_path, scene_loader=scene_loader)
    fps = scene_loader.fps

    n_frames_out_of_view = (~is_in_view).sum()
    out_of_view_tot_time = n_frames_out_of_view / fps  # [Sec]
    out_of_view_avg_seg_time = out_of_view_tot_time / n_segments  # [Sec]

    track_id = 0  # we only have one track in the scene
    err_per_seg = []

    for _i_seg, seg in enumerate(new_segments):
        # get the estimated angle of the track in the camera system, in the last frame of the out-of-view segment
        i_last = seg["last"]

        # Get the ground-truth angle of the track in the camera system, in the last frame of the out-of-view segment
        # (before the track re-enters the field of view)
        gt_angle_per_track = detections_tracker.get_tracks_angles_in_frame(i_last, frame_type="full_view")

        if len(gt_angle_per_track) == 0:
            # we only use segments where the track is visible in the full frame before  the out-of-algorithm-view segment
            continue

        gt_angle = gt_angle_per_track[track_id]

        # The estimated angle of the track in the camera system, in the last frame of the out-of-view segment
        # (before the track re-enters the field of view)
        est_angle = online_est_track_angle[i_last][track_id]

        err_per_seg.append(abs(est_angle - gt_angle))

    err_per_seg = np.array(err_per_seg)
    n_seg = len(err_per_seg)
    avg_err_deg = np.mean(err_per_seg) * 180 / np.pi
    std_err_deg = np.std(err_per_seg) * 180 / np.pi
    print(f"Scale {seg_scale} results")
    print(f"Average error: {avg_err_deg} [deg]")
    print(f"Std error: {std_err_deg}  [deg]")
    print(
        "Number of out-of-algorithm-view segment) (where the track is visible in the full-frame before the segment) ",
        n_seg,
    )

    return avg_err_deg, std_err_deg, n_seg, out_of_view_avg_seg_time


# ---------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
# ---------------------------------------------------------------------------------------------------------------------

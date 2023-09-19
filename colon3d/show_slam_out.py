import argparse
import pickle
from pathlib import Path

import h5py

from colon3d.alg.slam_out_analysis import plot_z_dist_from_cam
from colon3d.util.data_util import get_origin_scene_path
from colon3d.util.general_util import ArgsHelpFormatter, Tee, create_folder_if_not_exists
from colon3d.util.torch_util import to_default_type, to_torch
from colon3d.visuals.plot_nav_aid import draw_aided_nav
from colon3d.visuals.plots_2d import draw_keypoints_and_tracks
from colon3d.visuals.plots_3d_scene import plot_3d_trajectories, plot_camera_sys_per_frame, plot_world_sys_per_frame

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--example_path",
        type=str,
        default="data_gcp/datasets/real_videos/Example_4",
        help="path to the video",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="data/results/real_videos/Example_4",
        help="path to the SLAM algorithm outputs",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/results/real_videos/Example_4/Show",
        help="path to the save outputs",
    )

    args = parser.parse_args()
    print(f"args={args}")
    save_path = Path(args.save_path)

    with Tee(save_path / "log_run_slam.txt"):  # save the prints to a file
        # Load saved results
        with (Path(args.results_path) / "out_variables.pkl").open("rb") as fid:
            slam_out = pickle.load(fid)
        # Show the results
        save_slam_plots(slam_out, args.save_path, args.example_path)


# ---------------------------------------------------------------------------------------------------------------------


def save_slam_plots(
    slam_out,
    save_path,
    scene_path,
    start_frame=0,
    stop_frame=None,
    max_anim_frames=10,
):
    save_path = Path(save_path)
    create_folder_if_not_exists(save_path)
    # Extract the relevant variables
    scene_loader = slam_out["scene_loader"]
    scene_loader.example_path = scene_path
    n_frames_orig = scene_loader.n_frames
    assert n_frames_orig > 0, "No frames were loaded!"
    tracks_p3d_inds = slam_out["tracks_p3d_inds"]
    detections_tracker = slam_out["detections_tracker"]
    online_logger = slam_out["online_logger"]
    # Get the online estimated camera poses
    est_cam_poses = to_torch(online_logger.cam_pose_estimates)
    tracks_ids = tracks_p3d_inds.keys()
    #  List of the estimated 3D locations of each track's KPs (in the world system):
    online_est_track_cam_loc = slam_out["online_est_track_world_loc"]
    online_est_track_world_loc = slam_out["online_est_track_world_loc"]
    online_est_track_angle = slam_out["online_est_track_angle"]
    alg_prm = slam_out["alg_prm"]

    online_est_salient_kp_world_loc = slam_out["online_est_salient_kp_world_loc"]
    if stop_frame is None:
        stop_frame = n_frames_orig
    fps = scene_loader.fps  # frames per second
    t_interval_sec = 1 / fps  # sec

    # ---- Plot the estimated tracks z-coordinate in the camera system per fame
    if tracks_ids:
        plot_z_dist_from_cam(
            tracks_ids,
            start_frame,
            stop_frame,
            online_est_track_cam_loc,
            t_interval_sec,
            save_path,
        )

    online_logger.plot_cam_pose_changes(save_path, t_interval_sec)

    # Draw local-aided navigation video
    if tracks_ids:
        draw_aided_nav(
            scene_loader=scene_loader,
            detections_tracker=detections_tracker,
            online_est_track_cam_loc=online_est_track_cam_loc,
            online_est_track_angle=online_est_track_angle,
            alg_prm=alg_prm,
            start_frame=start_frame,
            stop_frame=stop_frame,
            save_path=save_path,
        )
    else:
        print("Skipping draw_aided_nav...")

    # Draw the keypoints and detections on the video
    if tracks_ids or alg_prm.use_bundle_adjustment:
        draw_keypoints_and_tracks(
            scene_loader=scene_loader,
            detections_tracker=detections_tracker,
            kp_log=slam_out["kp_log"],
            save_path=save_path,
        )
    else:
        print("Skipping draw_keypoints_and_tracks...")

    # plot the estimated tracks positions in the camera system per fame
    stop_frame_anim = min(stop_frame, start_frame + max_anim_frames)
    plot_camera_sys_per_frame(
        tracks_kps_cam_loc_per_frame=online_est_track_cam_loc,
        detections_tracker=detections_tracker,
        start_frame=start_frame,
        stop_frame=stop_frame_anim,
        fps=fps,
        cam_fov_deg=scene_loader.alg_fov_deg,
        show_fig=False,
        save_path=save_path,
    )

    # plot the camera trajectory and the track estimated positions in world system (animated plot)
    plot_world_sys_per_frame(
        online_est_track_world_loc=online_est_track_world_loc,
        online_est_salient_kp_world_loc=online_est_salient_kp_world_loc,
        cam_poses=est_cam_poses,
        start_frame=start_frame,
        stop_frame=stop_frame_anim,
        fps=fps,
        cam_fov_deg=scene_loader.alg_fov_deg,
        detections_tracker=detections_tracker,
        save_path=save_path,
    )

    # plot the camera trajectory and the track estimated positions (and ground-truth, if available) in world system (simple plot)
    trajectories = {"Est": est_cam_poses}
    gt_data_path = get_origin_scene_path(scene_path) / "gt_3d_data.h5"
    if gt_data_path.exists():
        with h5py.File(gt_data_path.resolve(), "r") as hf:
            gt_cam_poses = to_default_type(hf["cam_poses"][:])  # load the ground-truth camera poses into memory
        trajectories["GT"] = gt_cam_poses
    plot_3d_trajectories(
        trajectories=trajectories,
        save_path=save_path / "trajectory_world.html",
    )


# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
# ---------------------------------------------------------------------------------------------------------------------

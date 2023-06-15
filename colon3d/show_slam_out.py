import argparse
import pickle
from pathlib import Path

from colon3d.general_util import Tee, create_folder_if_not_exists
from colon3d.keypoints_util import transform_tracks_points_to_cam_frame
from colon3d.slam_out_analysis import plot_z_dist_from_cam
from colon3d.visuals.aided_nav_plot import draw_aided_nav
from colon3d.visuals.plots_2d import draw_keypoints_and_tracks
from colon3d.visuals.plots_3d_scene import plot_camera_sys_per_frame, plot_world_sys_per_frame

# ---------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--example_path",
        type=str,
        default="data/my_videos/Example_4",
        help="path to the video",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default="data/my_videos/Example_4/Results",
        help="path to the SLAM algorithm outputs",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/my_videos/Example_4/Results/Show",
        help="path to the save outputs",
    )

    args = parser.parse_args()
    save_path = Path(args.save_path).expanduser()

    with Tee(save_path / "log_run_slam.txt"):  # save the prints to a file
        # Load saved results
        with (Path(args.results_path) / "out_variables.pkl").open("rb") as fid:
            slam_out = pickle.load(fid)
        # Show the results
        save_slam_out_plots(slam_out, args.save_path, args.example_path)


# ---------------------------------------------------------------------------------------------------------------------


def save_slam_out_plots(
    slam_out,
    save_path,
    example_path,
    start_frame=0,
    stop_frame=None,
    plot_names=None,
    max_anim_frames=10,
):
    save_path = Path(save_path)
    create_folder_if_not_exists(save_path)
    # Extract the relevant variables
    frames_loader = slam_out.frames_loader
    # over-ride the example folder (in case it was moved)
    frames_loader.example_path = example_path
    n_frames_orig = frames_loader.n_frames
    assert n_frames_orig > 0, "No frames were loaded!"
    kp_frame_idx_all = slam_out.kp_frame_idx_all
    kp_px_all = slam_out.kp_px_all
    tracks_p3d_inds = slam_out.tracks_p3d_inds
    kp_id_all = slam_out.kp_id_all
    detections_tracker = slam_out.detections_tracker
    online_logger = slam_out.online_logger
    # Get the online estimated camera poses
    est_cam_poses = online_logger.cam_poses
    tracks_ids = tracks_p3d_inds.keys()
    #  List of the estimated 3D locations of each track's KPs (in the world system):
    online_est_track_world_loc = slam_out.online_est_track_world_loc
    online_est_salient_kp_world_loc = slam_out.online_est_salient_kp_world_loc
    if stop_frame is None:
        stop_frame = n_frames_orig
    fps = frames_loader.fps  # frames per second
    t_interval_sec = 1 / fps  # sec

    # ---- Get track estimated location of the track KPs w.r.t. camera system in each frame
    online_est_track_cam_loc = transform_tracks_points_to_cam_frame(online_est_track_world_loc, est_cam_poses)

    # ---- Plot the estimated tracks z-coordinate in the camera system per fame
    if plot_names is None or "z_dist_from_cam" in plot_names:
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
    if plot_names is None or "aided_nav" in plot_names:
        draw_aided_nav(
            frames_loader=frames_loader,
            detections_tracker=detections_tracker,
            online_est_track_cam_loc=online_est_track_cam_loc,
            start_frame=start_frame,
            stop_frame=stop_frame,
            save_path=save_path,
        )

    # Draw the keypoints and detections on the video
    if plot_names is None or "keypoints_and_tracks" in plot_names:
        draw_keypoints_and_tracks(
            frames_loader=frames_loader,
            detections_tracker=detections_tracker,
            kp_frame_idx_all=kp_frame_idx_all,
            kp_px_all=kp_px_all,
            kp_id_all=kp_id_all,
            save_path=save_path,
        )

    # plot the estimated tracks positions in the camera system per fame
    stop_frame_anim = min(stop_frame, start_frame + max_anim_frames)
    if plot_names is None or "camera_sys" in plot_names:
        plot_camera_sys_per_frame(
            tracks_kps_cam_loc_per_frame=online_est_track_cam_loc,
            detections_tracker=detections_tracker,
            start_frame=start_frame,
            stop_frame=stop_frame_anim,
            fps=fps,
            cam_fov_deg=frames_loader.alg_fov_deg,
            show_fig=False,
            save_path=save_path,
        )

    # plot the camera trajectory and the track estimated positions in world system
    if plot_names is None or "world_sys" in plot_names:
        plot_world_sys_per_frame(
            online_est_track_world_loc=online_est_track_world_loc,
            online_est_salient_kp_world_loc=online_est_salient_kp_world_loc,
            cam_poses=est_cam_poses,
            start_frame=start_frame,
            stop_frame=stop_frame_anim,
            fps=fps,
            cam_fov_deg=frames_loader.alg_fov_deg,
            detections_tracker=detections_tracker,
            save_path=save_path,
        )


# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from colon3d.slam.alg_settings import AlgorithmParam
from colon3d.util.general_util import save_plot_and_close
from colon3d.util.rotations_util import find_rotation_delta, get_rotation_angle
from colon3d.util.torch_util import np_func, to_numpy

# ---------------------------------------------------------------------------------------------------------------------


class AnalysisLogger:
    def __init__(self, alg_prm: AlgorithmParam):
        # saves the per frame initial guess of the camera pose (location and rotation) from which the optimizer starts:
        self.cam_pose_guesses = np.empty((0, 7))
        # saves the per frame final optimized camera pose (location and rotation):
        self.cam_pose_estimates = np.empty((0, 7))
        self.alg_prm = alg_prm

    # ---------------------------------------------------------------------------------------------------------------------

    def save_cam_pose_guess(self, cam_pose_guess):
        cam_pose_guess = deepcopy(to_numpy(cam_pose_guess)[np.newaxis, :])
        self.cam_pose_guesses = np.append(self.cam_pose_guesses, cam_pose_guess, axis=0)

    # ---------------------------------------------------------------------------------------------------------------------

    def save_cam_pose_estimate(self, cam_pose_est):
        cam_pose_est = deepcopy(to_numpy(cam_pose_est)[np.newaxis, :])
        self.cam_pose_estimates = np.append(self.cam_pose_estimates, cam_pose_est, axis=0)

    # ---------------------------------------------------------------------------------------------------------------------

    def plot_cam_pose_changes(self, save_path, t_interval_sec):
        cam_pose_estimates = self.cam_pose_estimates
        cam_pose_guesses = self.cam_pose_guesses

        n_frames = cam_pose_estimates.shape[0]
        frame_inds = np.arange(n_frames)
        time_vec = frame_inds * t_interval_sec  # seconds
        max_vel = self.alg_prm.max_vel  # mm/sec
        max_angular_vel = self.alg_prm.max_angular_vel  # rad/sec
        max_dist_between_frames_mm = max_vel * t_interval_sec  # mm
        max_angle_change_between_frames_deg = np.rad2deg(max_angular_vel * t_interval_sec)

        fig, axs = plt.subplots(2, 2)
        fig.suptitle("Camera pose estimation")
        for ax in axs.flat:
            ax.set(xlabel="t [sec])")
            ax.grid(True)
            # ax.legend()

        # The distance between the post-optimization estimated camera location to the initial guess
        dist_loc_guess_to_est = np.linalg.norm(cam_pose_estimates[:, :3] - cam_pose_guesses[:, :3], axis=1)
        axs[0, 0].plot(time_vec, dist_loc_guess_to_est, label="loc. change from guess")
        axs[0, 0].set(title="Loc. dist. from initial guess", ylabel="[mm]")

        # the distance of current camera location to the previous camera location
        dist_loc_to_prev_loc = np.linalg.norm(cam_pose_estimates[1:, :3] - cam_pose_estimates[:-1, :3], axis=1)
        axs[0, 1].plot(time_vec[1:], dist_loc_to_prev_loc, label="loc. change from prev. frame")
        # axs[0, 1].plot(time_vec[1:],  max_dist_between_frames_mm * np.ones(n_frames - 1), label="max allowed dist.")
        axs[0, 1].set(title=f"dist. from prev. frame\ndist_limit={max_dist_between_frames_mm:.2f}[mm]", ylabel="[mm]")

        # find the angle between the current camera rotation estimate and the initial guess
        angle_opt_change = np.zeros(n_frames)
        for i_frame in range(1, n_frames):
            rot_est = cam_pose_estimates[i_frame, 3:]
            rot_guess = cam_pose_guesses[i_frame, 3:]
            rot_diff = np_func(find_rotation_delta)(rot1=rot_guess, rot2=rot_est)
            rot_diff_angle = np_func(get_rotation_angle)(rot_diff)
            angle_opt_change[i_frame] = rot_diff_angle
        axs[1, 0].plot(time_vec[1:], np.rad2deg(angle_opt_change[1:]), label="diff. from guess")
        axs[1, 0].set(title="diff. from guess", ylabel="[Deg.]")

        # find the angle between the current camera rotation estimate and the previous camera rotation
        angle_change = np.zeros(n_frames)
        for i_frame in range(1, n_frames):
            cur_rot = cam_pose_estimates[i_frame, 3:]
            prev_rot = cam_pose_estimates[i_frame - 1, 3:]
            rot_change = np_func(find_rotation_delta)(rot1=prev_rot, rot2=cur_rot)
            rot_change_angle = np_func(get_rotation_angle)(rot_change)
            angle_change[i_frame] = rot_change_angle
        axs[1, 1].plot(
            time_vec[1:],
            np.rad2deg(angle_change[1:]),
            label="diff. from prev. frame",
        )
        # axs[1, 1].plot(time_vec[1:],  max_angle_change_between_frames_deg * np.ones(n_frames - 1), label="max allowed angle change")
        axs[1, 1].set(
            title=f"diff. from prev. frame\nangle_limit={max_angle_change_between_frames_deg:.2f}[Deg.]",
            ylabel="[Deg.]",
        )
        fig.suptitle("SLAM out per frame")
        fig.tight_layout()
        save_plot_and_close(save_path / "slam_analysis.png")


# ---------------------------------------------------------------------------------------------------------------------


def plot_z_dist_from_cam(tracks_ids, start_frame, stop_frame, online_est_track_cam_loc, t_interval_sec, save_path):
    plt.figure()
    for track_id in tracks_ids:
        track_est_3d_cam = np.zeros((0, 3))
        vis_frames = []  # the frames in which the track was visible
        for i_frame in range(start_frame, stop_frame):
            if track_id in online_est_track_cam_loc[i_frame]:
                cur_track_kp_loc = to_numpy(online_est_track_cam_loc[i_frame][track_id])
                track_est_3d_cam = np.concatenate((track_est_3d_cam, cur_track_kp_loc[np.newaxis, :]))
                vis_frames.append(i_frame)
        plt.plot(
            np.array(vis_frames) * t_interval_sec,
            track_est_3d_cam[:, 2],
            label=f"Track #{track_id}",
        )
    plt.xlabel("t [sec]")
    plt.ylabel("Z [mm]")
    plt.title("Tracks (polyps) center KP estimated location in camera system")
    plt.legend()
    plt.grid(True)
    save_plot_and_close(save_path / "polyps_z.png")


# ---------------------------------------------------------------------------------------------------------------------

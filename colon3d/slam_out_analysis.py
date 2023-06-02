from copy import deepcopy
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

from colon3d.alg_settings import AlgorithmParam
from colon3d.camera_util import FishEyeUndistorter
from colon3d.data_util import FramesLoader
from colon3d.depth_util import DepthAndEgoMotionLoader
from colon3d.general_util import save_plot_and_close
from colon3d.rotations_util import get_smallest_angle_between_rotations_np
from colon3d.torch_util import to_numpy
from colon3d.tracks_util import DetectionsTracker

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
            angle_opt_change[i_frame] = get_smallest_angle_between_rotations_np(rot_est, rot_guess)
        axs[1, 0].plot(time_vec[1:], np.rad2deg(angle_opt_change[1:]), label="diff. from guess")
        axs[1, 0].set(title="diff. from guess", ylabel="[Deg.]")

        # find the angle between the current camera rotation estimate and the previous camera rotation
        angle_change = np.zeros(n_frames)
        for i_frame in range(1, n_frames):
            cur_rot = cam_pose_estimates[i_frame, 3:]
            prev_rot = cam_pose_estimates[i_frame - 1, 3:]
            angle_change[i_frame] = get_smallest_angle_between_rotations_np(cur_rot, prev_rot)
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


def plot_z_dist_from_cam(tracks_ids, start_frame, stop_frame, tracks_kps_cam_loc_per_frame, t_interval_sec, save_path):
    plt.figure()
    for track_id in tracks_ids:
        track_est_3d_cam = np.zeros((0, 3))
        vis_frames = []  # the frames in which the track was visible
        for i_frame in range(start_frame, stop_frame):
            if track_id in tracks_kps_cam_loc_per_frame[i_frame]:
                cur_track_kps_locs = tracks_kps_cam_loc_per_frame[i_frame][track_id].numpy(force=True)
                cur_track_center_kp_loc = cur_track_kps_locs[0]  # the center KP
                track_est_3d_cam = np.concatenate((track_est_3d_cam, cur_track_center_kp_loc[np.newaxis, :]))
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

# create a dataclass for the SLAM algorithm output


@dataclass
class SlamOutput:
    alg_prm: AlgorithmParam  # algorithm parameters
    cam_poses: torch.Tensor  # (N, 7) tensor of camera poses (quaternion + translation)
    points_3d: torch.Tensor  # (M, 3) tensor of 3D keypoints
    kp_frame_idx_all: list[int]  # list of length M, each element is the frame index of the keypoint
    kp_px_all: torch.Tensor  # (M, 2) tensor of 2D keypoints
    kp_nrm_all: torch.Tensor  # (M, 2) tensor of 2D keypoints normalized image coordinates
    kp_p3d_idx_all: torch.Tensor  # (M,) tensor of 3D points corresponding to each keypoint
    tracks_p3d_inds: list[list[int]]  #  maps a track_id to its associated 3D world points indexes
    kp_id_all: torch.Tensor  # (M,) tensor of keypoint ids
    p3d_inds_in_frame: list[list[int]]  # maps a frame index to its associated 3D world points indexes
    map_kp_to_p3d_idx: list[int]  # maps a keypoint index to its associated 3D world point index
    frames_loader: FramesLoader  # frames loader object
    detections_tracker: DetectionsTracker  # detections tracker object
    cam_undistorter: FishEyeUndistorter  # camera undistorter object
    depth_estimator: DepthAndEgoMotionLoader  # depth estimator object
    analysis_logger: AnalysisLogger  # analysis logger object
    salient_kps_world_loc_per_frame: list  # List of the per-step estimates of the 3D locations of the saliency KPs (in the world system)
    tracks_kps_world_loc_per_frame: list  # List of the per-step estimates of the 3D locations of the tracks KPs (in the world system)


# ---------------------------------------------------------------------------------------------------------------------

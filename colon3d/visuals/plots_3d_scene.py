from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from colon3d.alg.tracks_loader import DetectionsTracker
from colon3d.util.torch_util import to_numpy
from colon3d.visuals.animate_util import create_animation_video, create_interactive_3d_animation
from colon3d.visuals.create_3d_obj import plot_fov_cone

# --------------------------------------------------------------------------------------------------------------------


def plot_world_sys_per_frame(
    online_est_track_world_loc: list,
    online_est_salient_kp_world_loc: list,
    cam_poses,
    start_frame,
    stop_frame,
    fps: float,
    cam_fov_deg: float,
    detections_tracker: DetectionsTracker,
    speedup_factor: float = 5.0,
    show_salient_kps: bool = False,
    save_path=None,
    save_as_video: bool = False,
):
    """Plots the 3D scene in world system, with the camera trajectory and the 3D points of the track.
    The figure has a slider that controls the frame shown.

    Args:
    ----
        online_est_track_world_loc: list of dicts, each dict is a frame, key = track_id, value = 3D position of the track in the world coordinate system
        cam_poses: camera poses in world system
        start_frame: start frame
        stop_frame: stop frame
        save_path: path to save the figure
        show_fig: show figure
        dist_lim: maximum distance from the first frame to plot
        fps: frames per second [Hz]
        show_current_est: whether to show the per-frame estimation of the tracks world coordinates or the final estimation
    """
    frame_inds = np.arange(start_frame, stop_frame, dtype=int)
    # the 6DOF pose of the camera from the world origin (per frame):
    cam_poses = to_numpy(cam_poses)
    frame_inds = np.arange(start_frame, stop_frame, dtype=int)
    t_interval = 1 / fps  # time interval between frames [sec]
    step_duration_sec = t_interval / speedup_factor  # [sec] the duration of each frame in the animation
    n_steps = len(frame_inds)

    #### create the objects that are dynamics (per-frame) in the scene
    per_step_objs_lists = [[] for _ in range(n_steps)]  # list of lists of object to show per step\frame
    max_dist = 1

    if show_salient_kps:
        for i_step, i_frame in enumerate(frame_inds):
            salient_kps = to_numpy(online_est_salient_kp_world_loc[i_frame]).astype(np.float32)
            if salient_kps.shape[0] == 0:
                continue
            per_step_objs_lists[i_step].append(
                go.Scatter3d(
                    x=salient_kps[:, 0],  # salient_kps[-max_points:, 0],
                    y=salient_kps[:, 1],
                    z=salient_kps[:, 2],
                    mode="markers",
                    marker={"size": 1, "color": "coral"},
                    name="Salient KPs",
                    showlegend=False,
                ),
            )
            max_dist = max(max_dist, np.linalg.norm(salient_kps, axis=1).max())

    # plot the tracks\polyps estimated 3D location in the world system per frame:
    for i_step, i_frame in enumerate(frame_inds):
        for track_id, track_p3d_tensor in online_est_track_world_loc[i_frame].items():
            track_p3d_world = to_numpy(track_p3d_tensor)  # location of the track in the world system
            # check if the track is in the view of the algorithm
            is_track_in_view = detections_tracker.is_track_in_alg_view(track_id, [i_frame])[0]
            color = "green" if is_track_in_view else "red"
            #  plot the track KP in the world system
            per_step_objs_lists[i_step].append(
                go.Scatter3d(
                    x=[track_p3d_world[0]],
                    y=[track_p3d_world[1]],
                    z=[track_p3d_world[2]],
                    mode="markers",
                    marker={"size": 3, "color": color, "opacity": 0.9},
                    text=f"track_id: {track_id}",
                    name=f"Track #{track_id}",
                    showlegend=False,
                ),
            )
            # find the distance of the camera from a track point
            max_dist = max(max_dist, np.linalg.norm(track_p3d_world, axis=-1).max())
    # plot the camera estimated 3D location and FOV cone, in the world system per frame:
    for i_step, i_frame in enumerate(frame_inds):
        # the camera trajectory up to the current frame (including the current frame)
        cam_traj = cam_poses[: i_frame + 1, 0:3]
        # the current camera rotation in world system (unit quaternion)
        cam_cur_rot = cam_poses[i_frame, 3:7]
        # add the camera 3D trajectory up to the current frame
        per_step_objs_lists[i_step].append(
            go.Scatter3d(
                x=cam_traj[:, 0],
                y=cam_traj[:, 1],
                z=cam_traj[:, 2],
                mode="lines",
                line={"color": "darkblue", "width": 1},
                name="Cam. path",
            ),
        )
        cone_objs = plot_fov_cone(
            cone_height=max_dist,
            fov_deg=cam_fov_deg,
            cam_loc=cam_traj[i_frame, 0:3],
            cam_rot=cam_cur_rot,
            label="Alg. FOV",
        )
        per_step_objs_lists[i_step] += cone_objs
    # create the figure:
    title_per_step = [f"Frame {i_frame}, t={i_frame * t_interval:.2f}[sec]" for i_frame in frame_inds]
    range_lim = max_dist * 1.5
    axes_ranges = {"x": [-range_lim, range_lim], "y": [-range_lim, range_lim], "z": [-range_lim, range_lim]}
    layout_args = {
        "scene": {"xaxis_title": "X [mm]", "yaxis_title": "Y [mm]", "zaxis_title": "Z [mm]"},
        "title": "World View",
        "hoverlabel": {"font_size": 12, "font_family": "Rockwell"},
    }
    if save_as_video:
        create_animation_video(
            axes_ranges=axes_ranges,
            static_objs_list=[],
            per_step_objs_lists=per_step_objs_lists,
            title_per_step=title_per_step,
            step_duration_sec=step_duration_sec,
            layout_args=layout_args,
            file_name="world_view",
            save_path=save_path,
        )
    else:
        fig = create_interactive_3d_animation(
            axes_ranges=axes_ranges,
            static_objs_list=[],
            per_step_objs_lists=per_step_objs_lists,
            title_per_step=title_per_step,
            step_duration_sec=step_duration_sec,
            layout_args=layout_args,
        )
        file_path = save_path / "world_view.html"
        fig.write_html(file_path)
        print("Saved figure at ", file_path)


# --------------------------------------------------------------------------------------------------------------------


def plot_camera_sys_per_frame(
    tracks_kps_cam_loc_per_frame: list,
    detections_tracker: DetectionsTracker,
    start_frame: int,
    stop_frame: int,
    fps: float,
    cam_fov_deg: float,
    show_fig=False,
    save_path=None,
    dist_lim: float = 3000,
    speedup_factor: float = 1,
):
    """
    Plot the 3D scene in the camera coordinate system
    Args:
        tracks_est_loc_wrt_cam: dict, key = track_id, value = estimated 3D position of the track KPs per frame in the camera coordinate system
        detections_tracker: detections tracker
        start_frame: start frame
        stop_frame: stop frame
        show_fig: show figure
        save_path: save path
        dist_lim: maximal distance to plot (units: mm).
    """
    frame_inds = np.arange(start_frame, stop_frame)
    t_interval = 1 / fps  # time interval between frames [sec]
    step_duration_sec = t_interval / speedup_factor  # [sec] the duration of each frame in the animation

    max_dist = 1

    n_steps = len(frame_inds)
    #### create the objects that are dynamics (per-frame) in the scene
    per_step_objs_lists = [[] for _ in range(n_steps)]  # list of lists of object to show per step\frame
    for i_step, i_frame in enumerate(frame_inds):
        tracks_est_loc_wrt_cam = tracks_kps_cam_loc_per_frame[i_frame]
        # Plot 3D position of each tracks in the current frame
        for track_id in tracks_est_loc_wrt_cam:
            kp_p3d = to_numpy(tracks_est_loc_wrt_cam[track_id])
            # check if the track is in the view of the algorithm
            is_track_in_view = detections_tracker.is_track_in_alg_view(track_id, [i_frame])[0]
            # draw only points in the distance limit
            dist_sqr = np.square(kp_p3d).sum(axis=-1)
            max_dist = max(max_dist, np.sqrt(dist_sqr))
            if dist_sqr > dist_lim**2:
                continue  # skip this point
            color = "green" if is_track_in_view else "red"
            opacity = 0.9
            per_step_objs_lists[i_step].append(
                go.Scatter3d(
                    x=[kp_p3d[0]],
                    y=[kp_p3d[1]],
                    z=[kp_p3d[2]],
                    text=f"Track {track_id}",
                    mode="markers",
                    marker={"size": 3, "color": color, "opacity": opacity},
                    showlegend=False,
                ),
            )

    ########## create the static plot objects ##########
    #   the camera cone plot objects are static:
    static_objs_list = plot_fov_cone(cone_height=max_dist, fov_deg=cam_fov_deg, label="Alg. FOV")

    # create the figure:
    title_per_step = [f"Frame {i_frame}, t={i_frame * t_interval:.2f}[sec]" for i_frame in frame_inds]
    range_lim = max_dist * 1.5
    axes_ranges = {"x": [-range_lim, range_lim], "y": [-range_lim, range_lim], "z": [-range_lim, range_lim]}
    layout_args = {
        "scene": {"xaxis_title": "X [mm]", "yaxis_title": "Y [mm]", "zaxis_title": "Z [mm]"},
        "title": "World View",
        "hoverlabel": {"font_size": 12, "font_family": "Rockwell"},
    }
    fig = create_interactive_3d_animation(
        axes_ranges=axes_ranges,
        static_objs_list=static_objs_list,
        per_step_objs_lists=per_step_objs_lists,
        title_per_step=title_per_step,
        step_duration_sec=step_duration_sec,
        layout_args=layout_args,
    )
    if save_path:
        file_path = save_path / "camera_view.html"
        fig.write_html(file_path)
        print("Saved figure at ", file_path)
    if show_fig:
        fig.show()


# --------------------------------------------------------------------------------------------------------------------


def plot_3d_trajectories(
    trajectories: np.ndarray | dict,
    save_path: Path,
    start_frame: int = 0,
    stop_frame: int | None = None,
):
    """
    Plot the 3D trajectory of the camera in the world system.
    Args:
        cam_poses: camera poses in world system [n_frames, 7]: [x,y,z, qw, qx, qy, qz] (or a dict with several trajectories to plot)
        start_frame: start frame
        stop_frame: stop frame
    """
    if isinstance(trajectories, np.ndarray):
        trajectories = {"cam": trajectories}

    colors = [
        "darkblue",
        "darkred",
        "darkgreen",
        "darkorange",
        "darkmagenta",
        "darkcyan",
        "darkgoldenrod",
        "darkviolet",
    ]

    fig = go.Figure()

    # plot all the trajectories:
    for i_traj, traj_name in enumerate(trajectories):
        poses = trajectories[traj_name]
        color = colors[i_traj % len(colors)]
        if stop_frame is None:
            stop_frame = poses.shape[0]
        locs = to_numpy(poses[start_frame:stop_frame, 0:3])
        fig.add_trace(
            go.Scatter3d(
                x=locs[:, 0],
                y=locs[:, 1],
                z=locs[:, 2],
                mode="lines+markers",
                marker={"color": color, "symbol": "circle", "size": 1},
                line={"color": color, "width": 1},
                name=traj_name,
            ),
        )
        # Plot the first point in larger size:
        fig.add_trace(
            go.Scatter3d(
                x=[locs[0, 0]],
                y=[locs[0, 1]],
                z=[locs[0, 2]],
                mode="markers",
                marker={"color": color, "symbol": "circle", "size": 3},
                name=traj_name,
            ),
        )

    fig.update_layout(
        scene={"xaxis_title": "X [mm]", "yaxis_title": "Y [mm]", "zaxis_title": "Z [mm]"},
        title="Camera Trajectory",
        hoverlabel={"font_size": 12, "font_family": "Rockwell"},
    )
    fig.write_html(save_path)
    print("Saved figure at ", save_path)


# --------------------------------------------------------------------------------------------------------------------

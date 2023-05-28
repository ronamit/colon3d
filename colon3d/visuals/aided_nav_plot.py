import numpy as np
from cv2 import cv2

from colon3d.data_util import VideoLoader
from colon3d.detections_util import DetectionsTracker
from colon3d.general_util import colors_platte, put_unicode_text_on_img, save_video
from colon3d.visuals.plots_2d import draw_alg_view_in_the_full_frame, draw_detections_on_frame

# --------------------------------------------------------------------------------------------------------------------


def draw_aided_nav(
    video_loader: VideoLoader,
    detections_tracker: DetectionsTracker,
    tracks_kps_cam_loc_per_frame: list,
    start_frame: int,
    stop_frame: int,
    save_path=None,
):
    """Draws the aided navigation on the video.
    Args:
        video_loader: VideoLoader
        detections_tracker: DetectionsTracker
        cam_undistorter: FishEyeUndistorter
        tracks_kps_cam_loc_per_frame: list of the estimated 3d position (in camera system) of the tracked polyps in the seen in each frame (units: mm)
        start_frame: the frame to start the aided navigation video
        stop_frame: the frame to stop the aided navigation video
        save_path: str, the path to save the video
    """
    fps = video_loader.fps  # [Hz]
    frames_generator = video_loader.frames_generator(frame_type="full")
    alg_view_cropper = video_loader.alg_view_cropper
    orig_cam_undistorter = video_loader.orig_cam_undistorter
    alg_cam_info = video_loader.alg_cam_info
    alg_view_radius = alg_view_cropper.view_radius
    cx_orig = alg_view_cropper.cx_orig
    cy_orig = alg_view_cropper.cy_orig
    orig_im_center = np.array([cx_orig, cy_orig])  # [px]

    nav_vis_frames = []
    ### Draw the aided navigation for each frame
    for i_frame, frame in enumerate(frames_generator):
        if start_frame > i_frame or i_frame >= stop_frame:
            continue
        vis_frame = np.copy(frame)
        # draw the algorithm view circle
        vis_frame = draw_alg_view_in_the_full_frame(vis_frame, video_loader)
        # draw bounding boxes for the original detections in the full frame
        orig_detections = detections_tracker.get_tracks_in_frame(i_frame, frame_type="full_view")
        for track_id, cur_detect in orig_detections.items():
            vis_frame = draw_detections_on_frame(
                vis_frame,
                cur_detect,
                track_id,
                alg_view_cropper=alg_view_cropper,
                convert_from_alg_view_to_full=False,
                color=colors_platte(track_id),
            )
        # draw bounding boxes for the detections in the algorithm view (on top of the original detections)
        alg_view_detections = detections_tracker.get_tracks_in_frame(i_frame, frame_type="alg_view")
        for track_id, cur_detect in alg_view_detections.items():
            vis_frame = draw_detections_on_frame(
                vis_frame,
                cur_detect,
                track_id,
                alg_view_cropper=alg_view_cropper,
                convert_from_alg_view_to_full=True,
            )
        # the estimated 3d position (in camera system) of the tracked polyps in the seen in the current frame (units: mm)
        tracks_kps_loc_est = tracks_kps_cam_loc_per_frame[i_frame]

        # go over all ths tracks that have been their location estimated in the cur\rent frame
        for track_id, cur_track_kps_loc_est in tracks_kps_loc_est.items():
            # the estimated 3d position of the center KP of the current track in the current frame (units: mm)
            p3d_cam = cur_track_kps_loc_est[0].numpy(force=True)  # [mm]
            z_dist = p3d_cam[2]  # [mm]
            # compute  the track position angle with the z axis
            ray_dist = np.linalg.norm(p3d_cam[0:3], axis=-1)  # [mm]
            angle_rad = np.arccos(z_dist / (ray_dist + 1e-20))  # [rad]
            angle_deg = np.rad2deg(angle_rad)  # [deg]
            xy_dist = np.linalg.norm(p3d_cam[0:2], axis=-1)  # [mm]
            xy_dir = p3d_cam[0:2] / xy_dist
            orient_arrow_base = orig_im_center + alg_view_radius * xy_dir  # [px]
            orient_arrow_len = 50
            orient_arrow_tip = orig_im_center + (alg_view_radius + orient_arrow_len) * xy_dir  # [px]
            # project the 3d point in camera system to the full\original image pixel coordinates  (projective transform + distortion)
            track_coord_in_orig_im, is_track_in_orig_im = orig_cam_undistorter.project_from_cam_sys_to_pixels(
                points3d=p3d_cam,
            )
            track_coord_in_orig_im = track_coord_in_orig_im.squeeze(0)
            is_track_in_orig_im = is_track_in_orig_im.squeeze(0)

            # check if the track center is in the algorithm view:
            is_track_center_in_alg_view = (
                np.linalg.norm(track_coord_in_orig_im - orig_im_center, axis=-1) < alg_view_radius
            )

            # draw text at top left corner:
            extra_text = " (in front of cam.)" if z_dist > 0 else " (behind cam.)"
            vis_frame = put_unicode_text_on_img(
                vis_frame,
                text=f"z={round(z_dist)} [mm], \u03B1={round(angle_deg)}\xb0" + extra_text,
                pos=(5, 5 + track_id * 30),
                font_size=30,
                fill_color=colors_platte(track_id),
                stroke_width=2,
                stroke_fill="black",
            )
            # draw orientation arrow, if the estimated position is outside the algorithm view
            if z_dist > 0 and not is_track_center_in_alg_view:
                vis_frame = cv2.drawMarker(
                    vis_frame,
                    np.round(orient_arrow_base).astype(int),
                    color=colors_platte(track_id),
                    markerType=cv2.MARKER_DIAMOND,
                    markerSize=15,
                    thickness=2,
                )
                vis_frame = cv2.arrowedLine(
                    vis_frame,
                    np.round(orient_arrow_base).astype(int),
                    np.round(orient_arrow_tip).astype(int),
                    color=colors_platte(track_id),
                    thickness=4,
                )
                # draw text near the arrow
                delta_z = z_dist - alg_cam_info.min_vis_z_mm
                alg_fov = 2 * np.arctan(alg_view_cropper.view_radius / max(alg_cam_info.fx, alg_cam_info.fy))
                delta_alpha = angle_deg - 0.5 * np.rad2deg(alg_fov)
                vis_frame = put_unicode_text_on_img(
                    vis_frame,
                    text=f"{round(delta_z):+g}mm\n{round(delta_alpha):+g}\xb0",
                    pos=np.round(orient_arrow_base + 4 * (orient_arrow_tip - orient_arrow_base)).astype(int),
                    font_size=40,
                    fill_color=colors_platte(track_id),
                    stroke_width=2,
                    stroke_fill="black",
                )
            if is_track_in_orig_im and z_dist > 0:
                # draw estimated detection indicator on top of the image
                vis_frame = cv2.drawMarker(
                    vis_frame,
                    (round(track_coord_in_orig_im[0]), round(track_coord_in_orig_im[1])),
                    color=colors_platte(track_id),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=15,
                    thickness=2,
                )
        nav_vis_frames.append(vis_frame)
    # end for
    if save_path:
        save_video(
            save_path,
            file_name="local_aided_nav.mp4",
            frames=nav_vis_frames,
            fps=fps,
        )
    return nav_vis_frames


# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------------

import os
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from colon3d.util.data_util import RadialImageCropper, SceneLoader
from colon3d.util.general_util import (
    colors_platte,
    coord_to_cv2kp,
    create_empty_folder,
    save_plot_and_close,
    save_rgb_image,
    save_video_from_frames_list,
    save_video_from_func,
)
from colon3d.alg.keypoints_util import KeyPointsLog
from colon3d.alg.tracks_loader import DetectionsTracker

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# --------------------------------------------------------------------------------------------------------------------


def draw_kps(vis_frame, kps, color):
    for kp in kps:
        vis_frame = cv2.circle(
            vis_frame,
            (round(kp.pt[0]), round(kp.pt[1])),
            radius=2,
            color=color,
            thickness=-1,
        )
    return vis_frame


# --------------------------------------------------------------------------------------------------------------------


def draw_track_box_on_frame(
    rgb_frame: np.ndarray,
    track: dict,
    track_id: int,
    alg_view_cropper: RadialImageCropper | None = None,
    convert_from_alg_view_to_full=False,
    color=None,
):
    vis_frame = np.copy(rgb_frame)
    top_left = (round(track["xmin"]), round(track["ymin"]))
    bottom_right = (round(track["xmax"]), round(track["ymax"]))
    if alg_view_cropper is not None and convert_from_alg_view_to_full:
        top_left = alg_view_cropper.convert_coord_in_crop_to_full(point2d=top_left)
        bottom_right = alg_view_cropper.convert_coord_in_crop_to_full(point2d=bottom_right)
    if color is None:
        color = colors_platte(track_id)
    # draw bounding bounding box
    vis_frame = cv2.rectangle(
        vis_frame,
        top_left,
        bottom_right,
        color=color,
        thickness=2,
    )
    return vis_frame


# --------------------------------------------------------------------------------------------------------------------


def save_video_with_tracks(frames_folder_path: Path, path_to_save: Path, tracks: pd.DataFrame, fps: float):
    frames_paths = sorted(frames_folder_path.glob("*.png"))
    n_frames = len(frames_paths)

    def get_frame_with_tracks(i_frame):
        frame = cv2.imread(str(frames_paths[i_frame]))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        vis_frame = np.copy(frame)
        tracks_in_frame = tracks.loc[tracks["frame_idx"] == i_frame].to_dict("records")
        for track in tracks_in_frame:
            vis_frame = draw_track_box_on_frame(
                rgb_frame=vis_frame,
                track=track,
                track_id=track["track_id"],
            )
        return vis_frame

    save_video_from_func(save_path=path_to_save, make_frame=get_frame_with_tracks, n_frames=n_frames, fps=fps)


# --------------------------------------------------------------------------------------------------------------------


def save_frames_with_tracks(frames_folder_path: Path, tracks: pd.DataFrame, path_to_save: Path):
    frames_paths = sorted(frames_folder_path.glob("*.png"))
    create_empty_folder(path_to_save)
    n_frames = len(frames_paths)
    for i_frame in range(n_frames):
        tracks_in_frame = tracks.loc[tracks["frame_idx"] == i_frame].to_dict("records")
        if tracks_in_frame:
            frame = cv2.imread(str(frames_paths[i_frame]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            vis_frame = np.copy(frame)
            for track in tracks_in_frame:
                vis_frame = draw_track_box_on_frame(
                    rgb_frame=vis_frame,
                    track=track,
                    track_id=track["track_id"],
                )
            save_rgb_image(vis_frame, path_to_save / f"{i_frame:06d}.png")


# --------------------------------------------------------------------------------------------------------------------


def draw_keypoints_and_tracks(
    scene_loader: SceneLoader,
    detections_tracker: DetectionsTracker,
    kp_log: KeyPointsLog,
    save_path=None,
):
    """Draw keypoints and detections on the full frame.
    Args:
        scene_loader: VideoLoader object.
        detections_tracker: DetectionsTracker object.
        save_path: str, the path to save the visualization.
    """
    frames_generator = scene_loader.frames_generator(frame_type="full")
    alg_view_cropper = scene_loader.alg_view_cropper  # RadialImageCropper or None
    fps = scene_loader.fps
    vis_frames = []
    ### Draw each frame
    for i_frame, frame in enumerate(frames_generator):
        vis_frame = np.copy(frame)
        # draw the algorithm view circle
        vis_frame = draw_alg_view_in_the_full_frame(vis_frame, scene_loader)
        # get the current frame keypoints
        kp_ids = kp_log.get_kp_ids_in_frame_inds([i_frame])
        # draw bounding boxes for the original tracks in the full frame
        orig_tracks = detections_tracker.get_tracks_in_frame(i_frame, frame_type="full_view")
        for track_id in orig_tracks:
            cur_track = orig_tracks[track_id]
            vis_frame = draw_track_box_on_frame(
                rgb_frame=vis_frame,
                track=cur_track,
                track_id=track_id,
                alg_view_cropper=alg_view_cropper,
                convert_from_alg_view_to_full=False,
                color=[0, 0, 127],
            )
        # draw bounding boxes for the tracks in the algorithm view
        cur_tracks = detections_tracker.get_tracks_in_frame(i_frame, frame_type="alg_view")
        for track_id in cur_tracks:
            cur_track = cur_tracks[track_id]
            vis_frame = draw_track_box_on_frame(
                rgb_frame=vis_frame,
                track=cur_track,
                track_id=track_id,
                alg_view_cropper=alg_view_cropper,
                convert_from_alg_view_to_full=True,
            )
        for kp_id in kp_ids:
            _, kp_x, kp_y = kp_id
            kp_type = kp_log.get_kp_type(kp_id)
            kp_xy = np.array([kp_x, kp_y])
            if alg_view_cropper is not None:
                kp_xy = alg_view_cropper.convert_coord_in_crop_to_full(point2d=kp_xy)
            kp_color = colors_platte(kp_type + 1)
            #  draw keypoint
            vis_frame = cv2.drawMarker(
                vis_frame,
                (int(kp_xy[0]), int(kp_xy[1])),
                color=kp_color,
                markerType=cv2.MARKER_DIAMOND,
                markerSize=6,
                thickness=1,
                line_type=cv2.LINE_AA,
            )
            # end for kp
        # end if rect
        vis_frames.append(vis_frame)
    # end for

    if save_path:
        save_video_from_frames_list(
            save_path=save_path / "draw_keypoints_and_tracks",
            frames=vis_frames,
            fps=fps,
        )
    return vis_frames


# --------------------------------------------------------------------------------------------------------------------


def draw_kp_on_img(
    img,
    salient_KPs,
    track_KPs,
    curr_tracks,
    alg_view_cropper: RadialImageCropper | None,
    save_path,
    i_frame,
):
    vis_frame = np.copy(img)  # RGB
    for track_id in curr_tracks:
        cur_track = curr_tracks[track_id]
        vis_frame = draw_track_box_on_frame(
            rgb_frame=vis_frame,
            track=cur_track,
            track_id=track_id,
            alg_view_cropper=alg_view_cropper,
        )
    vis_frame = draw_kps(vis_frame, salient_KPs, color=colors_platte(0))
    for track_id, track_kp in track_KPs.items():
        track_kp_for_plot = coord_to_cv2kp(track_kp)
        vis_frame = cv2.drawKeypoints(
            vis_frame,
            [track_kp_for_plot],
            None,
            color=colors_platte(track_id + 1),
            flags=0,
        )
    display_image_in_actual_size(vis_frame)
    if save_path:
        save_plot_and_close(save_path / f"{i_frame:06d}_kp.png")


# --------------------------------------------------------------------------------------------------------------------


def draw_matches(
    img_A,
    img_B,
    matched_A_kps,
    matched_B_kps,
    track_KPs_A,
    track_KPs_B,
    i_frameA,
    i_frameB,
    tracks_in_frameA,
    tracks_in_frameB,
    alg_view_cropper: RadialImageCropper | None,
    save_path,
):
    """Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        imgA: RGB image ndarray
        imgB: RGB image ndarray
        matched_A_kps: A list of cv2.KeyPoint objects for imgA.
        matched_B_kps: A list of cv2.KeyPoint objects for imgB.
        SOURCE : https://gist.github.com/woolpeeker/d7e1821e1b5c556b32aafe10b7a1b7e8
    """
    n_matches = len(matched_A_kps)

    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    new_shape = (
        max(img_A.shape[0], img_B.shape[0]),
        img_A.shape[1] + img_B.shape[1],
        3,
    )
    new_img = np.zeros(new_shape, type(img_A.flat[0]))
    img_A_vis = np.copy(img_A)
    img_B_vis = np.copy(img_B)
    # draw detections bounding boxes
    for track_id, cur_track in tracks_in_frameA.items():
        img_A_vis = draw_track_box_on_frame(
            rgb_frame=img_A_vis,
            track=cur_track,
            track_id=track_id,
            alg_view_cropper=alg_view_cropper,
        )
    for track_id, cur_track in tracks_in_frameB.items():
        img_B_vis = draw_track_box_on_frame(
            rgb_frame=img_B_vis,
            track=cur_track,
            track_id=track_id,
            alg_view_cropper=alg_view_cropper,
        )
    # Place images onto the new image.
    new_img[0 : img_A.shape[0], 0 : img_A.shape[1]] = img_A_vis
    new_img[0 : img_B.shape[0], img_A.shape[1] : img_A.shape[1] + img_B.shape[1]] = img_B_vis

    # Draw lines fro the salient KPs matches.
    for i_match in range(n_matches):
        kp_A = matched_A_kps[i_match]
        kp_B = matched_B_kps[i_match]
        end1 = (int(kp_A[0]), int(kp_A[1]))
        end2 = (int(kp_B[0]) + img_A.shape[1], int(kp_B[1]))
        cv2.line(new_img, end1, end2, color=[0, 127, 255], thickness=1)

    # draw salient-KPs
    for i_match in range(n_matches):
        kp_A = matched_A_kps[i_match]
        kp_B = matched_B_kps[i_match]
        end1 = (int(kp_A[0]), int(kp_A[1]))
        end2 = (int(kp_B[0]) + img_A.shape[1], int(kp_B[1]))
        cv2.circle(new_img, end1, radius=1, color=colors_platte(0), thickness=1)
        cv2.circle(new_img, end2, radius=1, color=colors_platte(0), thickness=1)
    # draw tracks KPs in the first image
    for track_id, track_kp in track_KPs_A.items():
        cv2.circle(
            new_img,
            center=(round(track_kp[0]), round(track_kp[1])),
            radius=2,
            color=colors_platte(track_id + 1),
            thickness=1,
        )
    # draw tracks KPs in the second image
    for track_id, track_kp in track_KPs_B.items():
        cv2.circle(
            new_img,
            center=(round(track_kp[0]) + img_A.shape[1], round(track_kp[1])),
            radius=2,
            color=colors_platte(track_id + 1),
            thickness=1,
        )
    plt.figure()
    display_image_in_actual_size(new_img)
    if save_path:
        save_plot_and_close(save_path / f"{i_frameA:06d}_{i_frameB:06d}_match.png")


# --------------------------------------------------------------------------------------------------------------------


def display_image_in_actual_size(im_data, hide_axis=True):
    # https://stackoverflow.com/questions/28816046/displaying-different-images-with-actual-size-in-matplotlib-subplot

    dpi = matplotlib.rcParams["figure.dpi"]
    height, width, n_chan = im_data.shape
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)
    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    # Hide spines, ticks, etc.
    if hide_axis:
        ax.axis("off")
    # Display the image.
    ax.imshow(im_data, cmap="gray")


# --------------------------------------------------------------------------------------------------------------------


def draw_alg_view_in_the_full_frame(frame, scene_loader: SceneLoader):
    # check if the algorithm view is smaller than the full frame
    if scene_loader.alg_fov_ratio > 0:
        # get the FOV of the alg view
        alg_view_radius = round(scene_loader.alg_view_radius)
        orig_cam_info = scene_loader.orig_cam_info
        orig_im_center = np.array([orig_cam_info.cx, orig_cam_info.cy]).round().astype(int)  # [px]
        color = colors_platte(color_name="gray")
        frame = cv2.circle(frame, orig_im_center, alg_view_radius, color=color, thickness=2)

    return frame


# --------------------------------------------------------------------------------------------------------------------

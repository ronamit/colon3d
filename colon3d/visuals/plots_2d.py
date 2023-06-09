import os
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from colon3d.data_util import FramesLoader, RadialImageCropper
from colon3d.general_util import (
    colors_platte,
    coord_to_cv2kp,
    create_empty_folder,
    save_plot_and_close,
    save_video_from_frames_list,
)
from colon3d.tracks_util import DetectionsTracker

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


def draw_tracks_on_frame(
    vis_frame,
    cur_detect,
    track_id: int,
    alg_view_cropper: RadialImageCropper | None = None,
    convert_from_alg_view_to_full=False,
    color=None,
):
    top_left = (round(cur_detect["xmin"]), round(cur_detect["ymin"]))
    bottom_right = (round(cur_detect["xmax"]), round(cur_detect["ymax"]))
    if alg_view_cropper is not None and convert_from_alg_view_to_full:
        top_left = alg_view_cropper.convert_coord_in_crop_to_full(point2d=top_left)
        bottom_right = alg_view_cropper.convert_coord_in_crop_to_full(point2d=bottom_right)
    if color is None:
        color = colors_platte(track_id + 1, "BGR")
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


def save_frames_with_tracks(rgb_frames_path: Path, path_to_save: Path, tracks: pd.DataFrame):
    frames_paths = sorted(rgb_frames_path.glob("*.png"))
    n_frames = len(frames_paths)
    create_empty_folder(path_to_save)

    def get_frame_with_tracks(i_frame):
        frame = cv2.imread(str(frames_paths[i_frame]))
        vis_frame = np.copy(frame)
        tracks_in_frame = tracks.loc[tracks["frame_idx"] == i_frame].to_dict("records")
        if len(tracks_in_frame) == 0:
            return None
        for cur_detect in tracks_in_frame:
            vis_frame = draw_tracks_on_frame(
                vis_frame=vis_frame,
                cur_detect=cur_detect,
                track_id=cur_detect["track_id"],
            )
        return vis_frame

    for i_frame in range(n_frames):
        vis_frame = get_frame_with_tracks(i_frame)
        if vis_frame is not None:
            # in case there tracks in the frame - save it
            cv2.imwrite(str(path_to_save / f"{i_frame:06d}.png"), vis_frame)


# --------------------------------------------------------------------------------------------------------------------


def draw_keypoints_and_tracks(
    frames_loader: FramesLoader,
    detections_tracker: DetectionsTracker,
    kp_frame_idx_all,
    kp_px_all,
    kp_id_all,
    save_path=None,
):
    """Draw keypoints and detections on the full frame.
    Args:
        frames_loader: VideoLoader object.
        detections_tracker: DetectionsTracker object.
        kp_frame_idx_all: list of int, the frame index of each keypoint.
        kp_px_all: list of np.array, the pixel coordinates of each keypoint.
        kp_id_all: list of int, the track id of each keypoint.
        save_path: str, the path to save the visualization.
    """
    kp_frame_idx_all = np.array(kp_frame_idx_all)
    kp_px_all = np.stack(kp_px_all, axis=0)
    frames_generator = frames_loader.frames_generator(frame_type="full")
    alg_view_cropper = frames_loader.alg_view_cropper # RadialImageCropper or None
    fps = frames_loader.fps
    vis_frames = []
    kp_id_all = np.array(kp_id_all)
    ### Draw each frame
    for i_frame, frame in enumerate(frames_generator):
        vis_frame = np.copy(frame)
        # draw the algorithm view circle
        vis_frame = draw_alg_view_in_the_full_frame(vis_frame, frames_loader)
        keypoints = kp_px_all[kp_frame_idx_all == i_frame]
        keypoints_ids = kp_id_all[kp_frame_idx_all == i_frame]
        # draw bounding boxes for the original tracks in the full frame
        orig_tracks = detections_tracker.get_tracks_in_frame(i_frame, frame_type="full_view")
        for track_id in orig_tracks:
            cur_detect = orig_tracks[track_id]
            vis_frame = draw_tracks_on_frame(
                vis_frame,
                cur_detect,
                track_id,
                alg_view_cropper=alg_view_cropper,
                convert_from_alg_view_to_full=False,
                color=[0, 0, 127],
            )
        # draw bounding boxes for the tracks in the algorithm view
        cur_tracks = detections_tracker.get_tracks_in_frame(i_frame, frame_type="alg_view")
        for track_id in cur_tracks:
            cur_detect = cur_tracks[track_id]
            vis_frame = draw_tracks_on_frame(
                vis_frame,
                cur_detect,
                track_id,
                alg_view_cropper=alg_view_cropper,
                convert_from_alg_view_to_full=True,
            )
        for i_kp, kp in enumerate(keypoints):
            kp_x = round(kp[0].item())
            kp_y = round(kp[1].item())
            kp_xy = (kp_x, kp_y)
            if alg_view_cropper is not None:
                kp_xy = alg_view_cropper.convert_coord_in_crop_to_full(point2d=kp_xy)
            kp_id = keypoints_ids[i_kp]
            kp_color = colors_platte(kp_id + 1, "BGR")
            #  draw keypoint
            vis_frame = cv2.drawMarker(
                vis_frame,
                kp_xy,
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
    vis_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for track_id in curr_tracks:
        cur_detect = curr_tracks[track_id]
        vis_frame = draw_tracks_on_frame(
            vis_frame=vis_frame,
            cur_detect=cur_detect,
            track_id=track_id,
            alg_view_cropper=alg_view_cropper,
        )
    vis_frame = draw_kps(vis_frame, salient_KPs, color=colors_platte(0, "RGB"))
    for track_id, track_kps in track_KPs.items():
        track_kps_for_plot = [coord_to_cv2kp(kp) for kp in track_kps]
        vis_frame = cv2.drawKeypoints(
            vis_frame,
            track_kps_for_plot,
            None,
            color=colors_platte(track_id + 1, "RGB"),
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
    alg_view_cropper : RadialImageCropper | None,
    save_path,
):
    """Draws lines between matching keypoints of two images.
    Keypoints not in a matching pair are not drawn.
    Places the images side by side in a new image and draws circles
    around each keypoint, with line segments connecting matching pairs.
    You can tweak the r, thickness, and figsize values as needed.
    Args:
        img1: An openCV image ndarray in a grayscale or color format.
        kp1: A list of cv2.KeyPoint objects for img1.
        img2: An openCV image ndarray of the same format and with the same
        element type as img1.
        kp2: A list of cv2.KeyPoint objects for img2.
        matches: A list of DMatch objects whose trainIdx attribute refers to
        img1 keypoints and whose queryIdx attribute refers to img2 keypoints.
        color: The color of the circles and connecting lines drawn on the images.
        A 3-tuple for color images, a scalar for grayscale images.  If None, these
        values are randomly generated.

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
    img_A_vis = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
    img_B_vis = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
    # draw detections bounding boxes
    for track_id, cur_detect in tracks_in_frameA.items():
        img_A_vis = draw_tracks_on_frame(
            vis_frame=img_A_vis,
            cur_detect=cur_detect,
            track_id=track_id,
            alg_view_cropper=alg_view_cropper,
        )
    for track_id, cur_detect in tracks_in_frameB.items():
        img_B_vis = draw_tracks_on_frame(
            vis_frame=img_B_vis,
            cur_detect=cur_detect,
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
        cv2.circle(new_img, end1, radius=1, color=colors_platte(0, "RGB"), thickness=1)
        cv2.circle(new_img, end2, radius=1, color=colors_platte(0, "RGB"), thickness=1)
    # draw tracks KPs
    for track_id, track_kps in track_KPs_A.items():
        for kp in track_kps:
            cv2.circle(
                new_img,
                center=(round(kp[0]), round(kp[1])),
                radius=2,
                color=colors_platte(track_id + 1, "RGB"),
                thickness=1,
            )
    for track_id, track_kps in track_KPs_B.items():
        for kp in track_kps:
            cv2.circle(
                new_img,
                center=(round(kp[0]) + img_A.shape[1], round(kp[1])),
                radius=2,
                color=colors_platte(track_id + 1, "RGB"),
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


def draw_alg_view_in_the_full_frame(frame, frames_loader):
    # get the FOV of the alg view
    alg_view_radius = round(frames_loader.alg_view_radius)
    orig_cam_info = frames_loader.orig_cam_info
    orig_im_center = np.array([orig_cam_info.cx, orig_cam_info.cy]).round().astype(int)  # [px]
    frame = cv2.circle(frame, orig_im_center, alg_view_radius, color=(255, 51, 51), thickness=2)

    return frame


# --------------------------------------------------------------------------------------------------------------------

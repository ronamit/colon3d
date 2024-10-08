from pathlib import Path

import numpy as np
import pandas as pd

from colon_nav.util.data_util import SceneLoader

# --------------------------------------------------------------------------------------------------------------------


class DetectionsTracker:
    """This class is used to track the detections of the polyp detector over time."""

    def __init__(self, scene_path, scene_loader: SceneLoader):
        self.scene_path = Path(scene_path)
        tracks_file_path = self.scene_path / "Tracks.csv"
        # Get the polyp detector results for this video
        if not tracks_file_path.is_file():
            print("No tracks file found at ", tracks_file_path)
            self.tracks_original = pd.DataFrame(
                {"frame_idx": [], "track_id": [], "xmin": [], "ymin": [], "xmax": [], "ymax": []},
            )
        else:
            self.tracks_original = pd.read_csv(tracks_file_path)
            self.tracks_original = self.tracks_original.astype({"frame_idx": int, "track_id": int})
            print("Loaded detections from: ", tracks_file_path)
        self.frame_inds = self.tracks_original["frame_idx"].unique()
        self.n_frames = len(self.frame_inds)
        self.fps = scene_loader.fps
        self.alg_view_cropper = scene_loader.alg_view_cropper
        self.truncate_tracks_for_alg_view()
    # ------------------------------------------------------------------------------------------------------------------


    def get_tracks_in_frame(self, frame_idx: int, frame_type: str = "alg_view"):
        """Gets the detections for the current algorithm view or full frame.

        Args:
            frame_idx (int): the frame index

        Returns:
            tracks_in_frame (dict): a dictionary of the detections in the current frame, where the key is the track_id
        """
        if frame_type == "alg_view":
            tracks = self.tracks_for_alg
        elif frame_type == "full_view":
            tracks = self.tracks_original
        else:
            raise ValueError("Unknown frame_type: " + frame_type)
        cur_tracks = tracks[tracks["frame_idx"] == frame_idx]
        track_ids = cur_tracks.track_id.unique()
        tracks_in_frame = {}
        for track_id in track_ids:
            tracks_in_frame[track_id] = cur_tracks[cur_tracks["track_id"] == track_id].to_dict("records")
            assert len(tracks_in_frame[track_id]) == 1, "There should be only one detection per (track,frame) pair"
            tracks_in_frame[track_id] = tracks_in_frame[track_id][0]  # take the first (and only) element
        return tracks_in_frame
    # ------------------------------------------------------------------------------------------------------------------
    
    def get_tracks_angles_in_frame(self, frame_idx: int, frame_type: str):
        tracks_in_frame = self.get_tracks_in_frame(frame_idx, frame_type)
        # get the center of the algorithm view
        if frame_type == "alg_view":
            cx = self.alg_view_cropper.cx_new
            cy = self.alg_view_cropper.cy_new
        elif frame_type == "full_view":
            cx = self.alg_view_cropper.cx_orig
            cy = self.alg_view_cropper.cy_orig
        else:
            raise ValueError("Unknown frame_type: " + frame_type)
        # get the angle of each track in the current frame
        tracks_angles = {}
        for track_id, track_info in tracks_in_frame.items():
            x_min = track_info["xmin"]
            x_max = track_info["xmax"]
            y_min = track_info["ymin"]
            y_max = track_info["ymax"]
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            tracks_angles[track_id] = get_track_angle_from_pixel_loc(x_center, y_center, cx, cy)
        return tracks_angles


    # ------------------------------------------------------------------------------------------------------------------

    def is_track_in_alg_view(self, track_id, frame_inds):
        """Checks if a track is in the current algorithm view.

        Args:
            track_id (int): the track id
            frame_inds (list): the frame indices

        Returns:
            is_track_in_view (np.array): a boolean array of the same length as frame_inds, where True means that the track is in the current algorithm view
        """
        is_track_in_view = np.zeros(len(frame_inds), dtype=bool)
        for i, i_frame in enumerate(frame_inds):
            is_track_in_view[i] = track_id in self.get_tracks_in_frame(i_frame)
        return is_track_in_view

    # ------------------------------------------------------------------------------------------------------------------

    def truncate_tracks_for_alg_view(self):
        """Truncates the detections to the defined algorithm view"""
        if self.alg_view_cropper is None:
            # no algorithm view cropping was defined - using original tracks as is.
            self.tracks_for_alg = self.tracks_original
            return
        x_min = self.alg_view_cropper.x_min
        y_min = self.alg_view_cropper.y_min
        view_radius = self.alg_view_cropper.view_radius
        cx_orig = self.alg_view_cropper.cx_orig
        cy_orig = self.alg_view_cropper.cy_orig
        n_tracks_orig = len(self.tracks_original)
        new_xmin = np.zeros(n_tracks_orig)
        new_xmax = np.zeros(n_tracks_orig)
        new_ymin = np.zeros(n_tracks_orig)
        new_ymax = np.zeros(n_tracks_orig)
        new_target_x_pix = np.ones(n_tracks_orig) * np.nan
        new_target_y_pix = np.ones(n_tracks_orig) * np.nan
        is_valid = np.zeros(n_tracks_orig, dtype=bool)
        for i, cur_track in self.tracks_original.iterrows():
            # get the o pixel coordinates inside the original track box in the full frame
            x_coords = np.arange(round(cur_track.xmin), round(cur_track.xmax) + 1)
            y_coords = np.arange(round(cur_track.ymin), round(cur_track.ymax) + 1)
            x_orig, y_orig = np.meshgrid(x_coords, y_coords, indexing="ij")
            # what pixels are inside the alg view FOV circle
            is_in_box = (x_orig - cx_orig) ** 2 + (y_orig - cy_orig) ** 2 <= view_radius**2
            n_pix_in_in_alg_view = is_in_box.flatten().sum()
            n_pix_in_orig_box = is_in_box.size
            if n_pix_in_in_alg_view <= n_pix_in_orig_box * 0.5:
                # too few pixels in the track box are in the alg-view FOV circle
                # so we determine the track box is not in the alg-view
                continue
            is_valid[i] = True
            # get the new track box coordinates in the cropped area
            new_xmin[i] = np.min(x_orig[is_in_box]) - x_min
            new_xmax[i] = np.max(x_orig[is_in_box]) - x_min
            new_ymin[i] = np.min(y_orig[is_in_box]) - y_min
            new_ymax[i] = np.max(y_orig[is_in_box]) - y_min

            # get the location of the target physical center in the cropped area
            if "target_x_pix" in cur_track:
                x_trg, y_trg = cur_track["target_x_pix"], cur_track["target_y_pix"]
                if (x_trg - cx_orig) ** 2 + (y_trg - cy_orig) ** 2 <= view_radius**2:
                    new_target_x_pix[i] = x_trg - x_min
                    new_target_y_pix[i] = y_trg - y_min

        detections_new = pd.DataFrame(
            {
                "frame_idx": self.tracks_original["frame_idx"][is_valid],
                "track_id": self.tracks_original["track_id"][is_valid],
                "xmin": new_xmin[is_valid],
                "ymin": new_ymin[is_valid],
                "xmax": new_xmax[is_valid],
                "ymax": new_ymax[is_valid],
                "target_x_pix": new_target_x_pix[is_valid],
                "target_y_pix": new_target_y_pix[is_valid],
            },
        ).astype({"frame_idx": int, "track_id": int})
        self.tracks_for_alg = detections_new

    # ------------------------------------------------------------------------------------------------------------------

def get_track_angle_from_pixel_loc(track_x_pix, track_y_pix, cx, cy):
    """ Gets track pixel location and the coordinates of the optical center and returns the angle of the track in the camera system.
    Args:
        track_x_pix, track_y_pix: coordinates of the track center in pixels
        cx, cy: coordinates of the optical center in pixels
    Returns:
        angle: angle of the track w.r.t. the optical center in the image  [rad]
    """
    dx = track_x_pix - cx
    dy = track_y_pix - cy
    angle = np.arctan2(dy, dx)
    return angle

# --------------------------------------------------------------------------------------------------------------------
def get_track_angle_from_cam_sys_loc(track_p3d_cam, cx, cy, alg_view_pix_normalizer):
    """ Gets track location in the camera system and the coordinates of the optical center and returns the angle of the track in the camera system.
    Args:
        track_p3d_cam: 3D coordinates of the track center in the camera system
        cx, cy: coordinates of the optical center in the camera system
    Returns:
        angle: angle of the track w.r.t. the optical center in the image  [rad]
    """
    track_pix_loc, _ = alg_view_pix_normalizer.project_from_cam_sys_to_pixels(track_p3d_cam)
    track_x_pix, track_y_pix = track_pix_loc.squeeze(0)
    return get_track_angle_from_pixel_loc(track_x_pix, track_y_pix, cx, cy)

import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

from colon3d.general_util import is_equal_dicts, save_plot_and_close


# --------------------------------------------------------------------------------------------------------------------
@dataclass
class CamInfo:
    """
    Metadata about the view
    """

    width: int  # image width (unit: pixels)
    height: int  # image height (units: pixels)
    cx: float  # optical center x position in the image (units: pixels)
    cy: float  # optical center y position in the image (units: pixels)
    fx: float  # focal length, normalized by the x-size of a pixel's sensor  (units: pixels)
    fy: float  # focal length, normalized by the y-size of a pixel's sensor  (units: pixels)
    distort_pram: np.ndarray  # Fisheye distortion parameters (cv2 format)
    fps: float  # frames per second (units: Hz)
    min_vis_z_mm: float  # the minimal z distance from the focal point that can be seen (units: mm)


# --------------------------------------------------------------------------------------------------------------------


class FishEyeUndistorter:
    """Class to undistort pixel coordinates in fisheye camera images to normalized oordinates in a rectilinear image.
    (target camera has zero distortion and a matrix K = unit matrix (fx = fy = 1, cx = cy = 0)
    """

    def __init__(self, cam_info: CamInfo):
        self.cam_K_mat = np.array([[cam_info.fx, 0, cam_info.cx], [0, cam_info.fy, cam_info.cy], [0, 0, 1]])
        self.cam_distort_param = cam_info.distort_pram
        self.frame_width = cam_info.width
        self.frame_height = cam_info.height
        self.fps = cam_info.fps
        self.undistort_points_lut = self.get_undistort_points_lut()

    # --------------------------------------------------------------------------------------------------------------------

    def get_undistort_points_lut(self):
        """
            get  a lookup table for undistorting points
        #  transforms normalized coordinates in the undistorted image  (transform to rectilinear camera with focal length is 1 and the optical center is at (0,0))
        """

        undistort_config = {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "cam_K_mat": self.cam_K_mat,
            "cam_distort_param": self.cam_distort_param,
        }
        cur_dir = Path(os.path.realpath(__file__)).parent
        luts_folder = cur_dir.parents[0] / "Undistort_LUTs"
        if not luts_folder.exists():
            luts_folder.mkdir(parents=True)
        #  Try to load existing LUT
        lut_file_prefix = f"Undistort_LUT_{self.frame_width}X{self.frame_height}_"
        file_names = [
            fpath for fpath in os.listdir(luts_folder) if fpath.startswith(lut_file_prefix) and fpath.endswith(".npy")
        ]
        n_existing_files = len(file_names)
        i_file = 0
        found = False
        while not found and i_file < n_existing_files:
            file_name = f"{lut_file_prefix}{i_file}"
            # check that the LUT is for the same camera
            with (luts_folder / f"{file_name}.pkl").open("rb") as file:
                lut_config = pickle.load(file)
            if is_equal_dicts(lut_config, undistort_config):
                found = True
                lut = np.load(luts_folder / f"{file_name}.npy")
            else:
                i_file += 1
        if not found:
            # create a new LUT
            lut = create_undistortion_lut(undistort_config)
            # save the LUT and camera parameters
            np.save(luts_folder / f"{lut_file_prefix}{i_file}.npy", lut)
            with (luts_folder / f"{lut_file_prefix}{i_file}.pkl").open("wb") as file:
                pickle.dump(undistort_config, file)
            # save boolean image of not NaN values
            plt.figure()
            plt.imshow(~np.isnan(lut[:, :, 0]))
            plt.title("valid points")
            save_plot_and_close(luts_folder / f"{lut_file_prefix}{i_file}.png")
        return lut

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def undistort_points(self, points2d):
        """returns the corresponding normalized coordinates in the undistorted image  (transform to rectilinear camera with focal length is 1 and the optical center is at (0,0))
        Args:
            points2d: 2D points in the distorted image (shape: (N,2), units: pixels)
        Returns:
            undistorted: 2D points in the undistorted image (shape: (N,2))
            is_valid: boolean array indicating if the point is valid (not outside the image)
        """
        undistorted = self.undistort_points_lut[points2d[:, 1], points2d[:, 0]]
        is_valid = np.any(~np.isnan(undistorted), axis=-1)
        return undistorted, is_valid

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def undistort_point(self, point2d):
        """returns the corresponding normalized coordinates in the undistorted image  (transform to rectilinear camera with focal length is 1 and the optical center is at (0,0))
        Args:
            point2d: 2D point in the distorted image (shape: (2), units: pixels)
        Returns:
            undistorted: 2D points in the undistorted image (shape: (2))
        """
        assert isinstance(point2d, tuple)
        assert len(point2d) == 2
        undistorted = self.undistort_points_lut[int(point2d[1]), int(point2d[0]), :]
        is_valid = np.any(~np.isnan(undistorted), axis=-1)
        return undistorted, is_valid

    # --------------------------------------------------------------------------------------------------------------------

    def project_from_cam_sys_to_pixels(self, points3d: np.ndarray):
        """
        Projects points in the 3D camera system to 2D pixels in the distorted image.
        Args:
            points3d: 3D points  in the camera system (shape: (N,3), units: mm)
        Returns:
            points2d: 2D points in the distorted image (shape: (2), units: pixels)
        """
        # set the translation & rotation to zero
        rot = np.array([[[0.0, 0.0, 0.0]]])
        trans = np.array([[[0.0, 0.0, 0.0]]])
        if points3d.ndim == 1:
            points3d = points3d[np.newaxis, :]
        points3d = points3d[:, np.newaxis, :]  # (N_points,1,3) to fit cv2.fisheye.projectPoints input
        # project points to image plane &  convert to pixels position in distorted image:
        points2d, _ = cv2.fisheye.projectPoints(points3d, rot, trans, self.cam_K_mat, self.cam_distort_param)
        points2d = np.squeeze(points2d, axis=1)  # (N,2)
        is_in_frame = (
            (points2d[:, 0] >= 0)
            & (points2d[:, 0] < self.frame_width)
            & (points2d[:, 1] >= 0)
            & (points2d[:, 1] < self.frame_height)
        )
        return points2d, is_in_frame


# --------------------------------------------------------------------------------------------------------------------


def create_undistortion_lut(undistort_config):
    """
        create  lookup table for undistorting points
    #  transforms pixel coordinates to  normalized coordinates in the undistorted image  (transform to rectilinear camera with focal length is 1 and the optical center is at (0,0))

    """
    frame_width = undistort_config["frame_width"]
    frame_height = undistort_config["frame_height"]
    # create list of all points in the original image
    x = np.arange(frame_width)
    y = np.arange(frame_height)
    xv, yv = np.meshgrid(x, y)
    points2d = np.stack([xv.flatten(), yv.flatten()], axis=1)
    # undistort the points
    undistorted, is_valid = run_undistort_points(undistort_config, points2d)
    lut = np.ones((frame_height, frame_width, 2), dtype=np.float32) * np.nan
    xv = xv.flatten()
    yv = yv.flatten()
    lut[yv[is_valid], xv[is_valid], 0] = undistorted[is_valid, 0]
    lut[yv[is_valid], xv[is_valid], 1] = undistorted[is_valid, 1]
    return lut


# --------------------------------------------------------------------------------------------------------------------

def run_undistort_points(undistort_config, points2d):
    #  transforms  normalized coordinates in the undistorted image  (transform to rectilinear camera with focal length is 1 and the optical center is at (0,0))
    cam_K_mat = undistort_config["cam_K_mat"]
    cam_distort_param = undistort_config["cam_distort_param"]
    #  the desired K matrix is a unit matrix (fx = fy = 1 and cx = cy = 0)
    K_undistort = np.eye(3)
    points2d_in = points2d.copy()[:, np.newaxis, :].astype(float)
    undistorted = cv2.fisheye.undistortPoints(
        distorted=points2d_in,
        K=cam_K_mat,
        D=cam_distort_param,
        R=None,
        P=K_undistort,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-6),
    )
    undistorted = np.squeeze(undistorted, axis=1)
    # sometimes the function cv2.fisheye.undistortPoints gives invalid values
    upper_lim = 1e4
    is_valid = np.sum(np.abs(undistorted), axis=1) < upper_lim
    return undistorted, is_valid


# --------------------------------------------------------------------------------------------------------------------

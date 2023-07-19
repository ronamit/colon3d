import cv2
import numpy as np

from colon3d.util.camera_info import CamInfo

# --------------------------------------------------------------------------------------------------------------------


class PixelCoordNormalizer:
    """
    The class is used to map pixel coordinates to normalized coordinates.
    Normalized coordinates correspond to rectilinear camera with focal length 1 and optical center at (0,0).
    """

    def __init__(self, cam_info: CamInfo):
        self.cam_K_mat = np.array([[cam_info.fx, 0, cam_info.cx], [0, cam_info.fy, cam_info.cy], [0, 0, 1]])
        self.cam_K_mat_inv = np.linalg.inv(self.cam_K_mat)
        self.cam_distort_param = cam_info.distort_pram
        self.frame_width = cam_info.frame_width
        self.frame_height = cam_info.frame_height
        self.is_fish_eye = self.cam_distort_param is not None and not np.allclose(self.cam_distort_param, 0)

    # --------------------------------------------------------------------------------------------------------------------

    def get_normalized_coords(self, points2d):
        """returns the corresponding normalized coordinates in the undistorted image  (transform to rectilinear camera with focal length is 1 and the optical center is at (0,0))
        Args:
            points2d: 2D pixel locations in the (original) distorted image (shape: (N,2), units: pixels)
        Returns:
            undistorted: normalized 2D points in the undistorted image (shape: (N,2), units: normalized coordinates)
            is_valid: boolean array indicating if the point is valid (not outside the image)
        Notes:
            * Normalized coordinates correspond to the rectilinear camera with focal length is 1 and the optical center is at (0,0)
        """
        n_points = points2d.shape[0]
        width = self.frame_width
        height = self.frame_height
        if self.is_fish_eye:
            # set accuracy criteria (to be used in fish-eye undistortion)
            criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 30, 1e-3)
            # change points2d to cv2 format
            points2d_cv2 = points2d[:, np.newaxis, :].astype(np.float32)
            nrm_coords = cv2.fisheye.undistortPoints(
                distorted=points2d_cv2,
                K=self.cam_K_mat,
                D=self.cam_distort_param,
                criteria=criteria,
            )
            # remove the extra dimension
            nrm_coords = np.squeeze(nrm_coords, axis=1)
            # sometimes the function cv2.fisheye.undistortPoints gives invalid values
            upper_lim = width + height  # some large number
            is_valid = np.sum(np.abs(nrm_coords), axis=1) < upper_lim
        else:
            # in case of rectilinear camera, the normalized coordinates are:
            points2d_ext = np.hstack((points2d, np.ones((n_points, 1))))
            nrm_coords = self.cam_K_mat_inv @ points2d_ext.T
            nrm_coords = nrm_coords[:2, :].T
            is_valid = np.ones(n_points, dtype=bool)

        return nrm_coords, is_valid

    # --------------------------------------------------------------------------------------------------------------------

    def get_normalized_coord(self, point2d):
        """returns the corresponding normalized coordinates in the undistorted image  (transform to rectilinear camera with focal length is 1 and the optical center is at (0,0))
        Args:
            points2d: 2D pixel locations in the (original) distorted image (shape: (2), units: pixels)
        Returns:
            undistorted: normalized 2D points in the undistorted image (shape: (2), units: normalized coordinates)
            is_valid: boolean array indicating if the point is valid (not outside the image)
        Notes:
            * Normalized coordinates correspond to the rectilinear camera with focal length is 1 and the optical center is at (0,0)
        """
        assert len(point2d) == 2
        nrm_coords, is_valid = self.get_normalized_coords(np.array([point2d]))
        return nrm_coords.squeeze(0), is_valid.squeeze(0)

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

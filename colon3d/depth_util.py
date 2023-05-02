import pickle
from pathlib import Path

import h5py
import numpy as np
import torch

from colon3d.slam_util import unproject_normalized_coord_to_world

# --------------------------------------------------------------------------------------------------------------------


class DepthEstimator:
    def __init__(self, example_path):
        """Wrapper for the depth & ego-motion estimation model.

        Args:
            example_path: path to the example folder
        """
        self.example_path = Path(example_path).expanduser()
        # Get the monocular depth and egomotion estimation results file path
        file_path = self.example_path / "depth_and_egomotion.h5"
        assert file_path.is_file(), "Not found depth file: " + str(file_path)
        print("Using egomotion and z-depth maps from: ", file_path)
        # load the depth estimation info\metadata
        with (self.example_path / "depth_info.pkl").open("rb") as file:
            depth_info = pickle.load(file)
        self.depth_map_size = depth_info["depth_map_size"]
        self.de_K = depth_info["K_of_depth_map"] # the camera matrix of the depth map images
        self.file_path = file_path
        self.n_frames = depth_info["n_frames"]
        with h5py.File(self.file_path, "r") as h5f:
            self.loaded_egomotions = h5f["egomotions"][:]  # load into memory

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_egomotions_at_frames(
        self,
        frame_indexes: np.array,
    ):
        """Get the egomotion estimation at a given frame.
            The egomotion is the 6-DOF current camera pose w.r.t. the previous camera pose.
            The egomotion is represented as a 7D vector: (x, y, z, q0, qx, qy, qz)
            x,y,z are the translation in mm.
            The quaternion (q0, qx, qy, qz) represents the rotation.
        """
        egomotions_est = self.loaded_egomotions[frame_indexes]
        assert egomotions_est.shape[1] == 7  # (x, y, z, q0, qx, qy, qz)
        egomotions_est[:, :3] = egomotions_est[:, :3]  # convert units to mm
        return egomotions_est

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def get_depth_at_nrm_points(
        self,
        frame_indexes: np.array,
        queried_points_nrm: torch.Tensor,
        z_depth_upper_bound: float,
        z_depth_lower_bound: float,
    ):
        """Get the depth estimation at a given point in the image.

        Args:
            frame_idx: the frame index per point [n_points]
            queried_points_2d: the normalized coordinates in the (undistorted) image  [n_points x 2]

        Returns:
            depth_z_est: the depth estimation at the queried point [n_points] (units: mm)
        """
        n_points = queried_points_nrm.shape[0]
        device = queried_points_nrm.device
        dtype = queried_points_nrm.dtype
        # the depth  map size
        de_width = self.depth_map_size["width"]
        de_height = self.depth_map_size["height"]
        # transform the query points from normalized coords (rectilinear with  K=I) to the depth estimation map coordinates (rectilinear with a given K matrix)
        x = queried_points_nrm[:, 0]
        y = queried_points_nrm[:, 1]
        x = x * self.de_K[0, 0] + self.de_K[0, 2]
        y = y * self.de_K[1, 1] + self.de_K[1, 2]
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        x = x.round().astype(int)
        y = y.round().astype(int)
        x = np.clip(x, 0, de_width - 1)
        y = np.clip(y, 0, de_height - 1)
        # get the depth estimation at the queried point from the saved depth maps
        depth_z_est = torch.zeros((n_points), device=device, dtype=dtype)
        with h5py.File(self.file_path, "r") as h5f:
            for frame_idx in np.unique(frame_indexes):
                loaded_depth_maps = h5f["z_depth_map"][frame_idx]
                depth_out = loaded_depth_maps[y[frame_indexes == frame_idx], x[frame_indexes == frame_idx]]
                depth_out = torch.as_tensor(depth_out, device=device, dtype=dtype)
                depth_z_est[frame_indexes == frame_idx] =  depth_out

        del loaded_depth_maps, h5f
        # clip the depth
        depth_z_est[depth_z_est > z_depth_upper_bound] = torch.as_tensor(
            z_depth_upper_bound, device=device, dtype=dtype,
        )
        depth_z_est[depth_z_est < z_depth_lower_bound] = torch.as_tensor(
            z_depth_lower_bound, device=device, dtype=dtype,
        )
        return depth_z_est

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def estimate_3d_points(
        self,
        frame_indexes: list,
        cam_poses: torch.Tensor,
        queried_points_nrm: torch.Tensor,
        z_depth_upper_bound: float,
        z_depth_lower_bound: float,
    ):
        """Estimate the 3D point in the world coordinate system.

        Args:
            frame_idx (list): the frame indexes for each point [n_points]
            cam_poses (np.ndarray): the camera poses [n_points x 7], each row is (x, y, z, q0, qx, qy, qz) where (x, y, z) is the translation [mm] and (q0, q1, q2 , q3) is the unit-quaternion of the rotation.
            queried_points_nrm np.ndarray: the point in the image [n_points x 2] (normalized coordinates)

        Returns:
            p3d_est: the estimated 3D point in the world coordinate system [n_points x 3] (units: mm)
        """
        depth_z_est = self.get_depth_at_nrm_points(
            frame_indexes=frame_indexes,
            queried_points_nrm=queried_points_nrm,
            z_depth_upper_bound=z_depth_upper_bound,
            z_depth_lower_bound=z_depth_lower_bound,
        )
        p3d_est = unproject_normalized_coord_to_world(
            points_nrm=queried_points_nrm,
            z_depth=depth_z_est,
            cam_poses=cam_poses,
        )
        return p3d_est

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

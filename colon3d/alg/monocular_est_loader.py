import queue
from pathlib import Path

import h5py
import numpy as np
import torch

from colon3d.alg.depth_and_ego_models import DepthModel, EgomotionModel
from colon3d.util.data_util import SceneLoader, get_origin_scene_path
from colon3d.util.pose_transforms import transform_rectilinear_image_norm_coords_to_pixel
from colon3d.util.rotations_util import get_identity_quaternion, normalize_quaternions
from colon3d.util.torch_util import get_default_dtype, get_device, to_default_type, to_torch

# --------------------------------------------------------------------------------------------------------------------


class DepthAndEgoMotionLoader:
    def __init__(
        self,
        scene_path: Path,
        scene_loader: SceneLoader,
        depth_maps_source: str,
        egomotions_source: str,
        model_path: str | None = None,
        depth_lower_bound: float | None = None,
        depth_upper_bound: float | None = None,
        depth_default: float | None = None,
    ):
        """Wrapper for the depth & ego-motion estimation model.

        Args:
            example_path: path to the scene folder
            depth_maps_source: the source of the depth maps, if 'ground_truth' then the ground truth depth maps will be loaded,
                if 'online_estimates' then the depth maps will be estimated online by the algorithm
                if 'loaded_estimates' then the depth maps estimations will be loaded,
                if 'none' then no depth maps will not be used,
            egomotions_source: the source of the egomotion, if 'ground_truth' then the ground truth egomotion will be loaded,
                if 'online_estimates' then the egomotion will be estimated online by the algorithm
                if 'loaded_estimates' then the egomotion estimations will be loaded,
                if 'none' then no egomotion will not be used,
        """
        self.orig_scene_path = get_origin_scene_path(scene_path)
        self.depth_maps_source = depth_maps_source
        self.egomotions_source = egomotions_source
        self.depth_lower_bound = depth_lower_bound if depth_lower_bound is not None else 1e-2
        self.depth_upper_bound = depth_upper_bound if depth_upper_bound is not None else 2e3
        self.depth_default = depth_default
        self.device = get_device()

        # Initialize egomotions
        if egomotions_source == "online_estimates":
            print("Using online egomotion estimator")
            self.egomotion_estimator = EgomotionModel(
                model_path=model_path,
            )
            # # number of reference images to use as input to the egomotion estimator:
            self.n_ref_imgs = self.egomotion_estimator.n_ref_imgs
            
        elif egomotions_source == "ground_truth":
            print("Using loaded ground-truth egomotions")
            self.init_loaded_egomotions("gt_3d_data.h5")
            self.n_ref_imgs = 0  # no need for reference images

        elif egomotions_source == "loaded_estimates":
            print("Using loaded estimated egomotions")
            self.init_loaded_egomotions("est_depth_and_egomotion.h5")
            self.n_ref_imgs = 0  # no need for reference images
        else:
            raise ValueError(f"Unknown egomotions source: {egomotions_source}")

        if depth_maps_source == "online_estimates":
            self.depth_estimator = DepthModel(
                depth_lower_bound=self.depth_lower_bound,
                depth_upper_bound=self.depth_upper_bound,
                model_path=model_path,
            )
            print("Using online depth estimation")

        elif depth_maps_source == "ground_truth":
            print("Using loaded ground-truth depth maps")
            self.init_loaded_depth("gt_3d_data.h5")

        elif depth_maps_source == "loaded_estimates":
            print("Using loaded estimated depth maps")
            self.init_loaded_depth("est_depth_and_egomotion.h")

        elif depth_maps_source == "none":
            assert depth_default is not None
        else:
            raise ValueError(f"Unknown depth maps source: {depth_maps_source}")

        self.alg_cam_info = scene_loader.alg_cam_info

    # --------------------------------------------------------------------------------------------------------------------

    def init_loaded_egomotions(self, egomotions_file_name: str):
        # load the pre-computed egomotions into buffer
        with h5py.File((self.orig_scene_path / egomotions_file_name).resolve(), "r") as h5f:
            self.egomotions_buffer = to_default_type(h5f["egomotions"][:])  # load all into memory
        n_frames = self.egomotions_buffer.shape[0]
        self.egomotions_buffer_frame_inds = list(range(n_frames))

    # --------------------------------------------------------------------------------------------------------------------
    def init_loaded_depth(self, depth_maps_file_name: str):
        """Load the pre-computed depth maps into buffer."""
        self.depth_maps_path = self.orig_scene_path / depth_maps_file_name
        # load the depth maps
        with h5py.File(self.depth_maps_path.resolve(), "r") as h5f:
            self.depth_maps_buffer = to_torch(
                h5f["z_depth_map"][:],
                num_type="float_m",
                device="default",
            )  # load all into memory
        n_frames = self.depth_maps_buffer.shape[0]
        self.depth_maps_buffer_frame_inds = list(range(n_frames))

    # --------------------------------------------------------------------------------------------------------------------

    def get_depth_map_at_frame(
        self,
        frame_idx: int,
        rgb_frame: np.ndarray | None = None,
    ):
        """Get the estimated z-depth map at a given frame.
        Notes: we assume process_new_frame was called before this function on this frame index.
        Returns:
            depth_map: the depth estimation map [depth_map_width x depth_map_height] (units: mm)
        Note:
            the size of the depth_map might be different than the size of the RGB image.
        """
        if self.depth_maps_source == "none":
            # return the default depth map
            depth_map = torch.ones(*rgb_frame.shape[:2], device=get_device()) * self.depth_default

        elif self.depth_maps_source in ["ground_truth", "loaded_estimates"]:
            buffer_idx = self.depth_maps_buffer_frame_inds.index(frame_idx)
            depth_map = self.depth_maps_buffer[buffer_idx]

        elif self.depth_maps_source == "online_estimates":
            depth_map = self.depth_estimator.estimate_depth_map(rgb_frame)

        else:
            raise ValueError(f"Unknown depth maps source: {self.depth_maps_source}")
        return depth_map

    # --------------------------------------------------------------------------------------------------------------------

    def get_egomotions_at_frame(
        self,
        curr_frame_idx: int,
        cur_rgb_frame: np.ndarray | None = None,
        prev_rgb_frames: queue.Queue[np.ndarray] | None = None,
    ) -> torch.Tensor:
        """Get the egomotion at a given frame.
        The egomotion is the pose change from the previous frame to the current frame, defined in the previous frame system.
        The egomotion is represented as a 7D vector: (x, y, z, q0, qx, qy, qz)
        x,y,z are the translation in mm.
        The quaternion (q0, qx, qy, qz) represents the rotation.
        Args:
            curr_frame_idx: the current frame index,
            curr_frame: the current RGB frame,
            prev_frames: the previous RGB frames (in a queue with max length of n_ref_imgs),
        Returns:
            egomotion: the egomotion (units: mm, mm, mm, -, -, -, -)
        Notes: we assume process_new_frame was called before this function on this frame index.
        """
        dtype = get_default_dtype()
        n_ref_imgs = self.n_ref_imgs
        n_prev_rgb_frames = prev_rgb_frames.qsize()

        # By default, the egomotion is the identity (no motion)
        # In case we haven't seen yet enough frames to feed the PoseNet, we output the identity egomotion.
        if self.egomotions_source == "none" or n_prev_rgb_frames < n_ref_imgs:
            # default value = identity egomotion (no motion)
            egomotion = torch.zeros((7), dtype=dtype, device=self.device)
            egomotion[3:] = get_identity_quaternion()

        # In case we use pre-computed egomotions, we just load the egomotion from the buffer
        elif self.egomotions_source in ["ground_truth", "loaded_estimates"]:
            buffer_idx = self.egomotions_buffer_frame_inds.index(curr_frame_idx)
            egomotion = self.egomotions_buffer[buffer_idx]
            egomotion = to_torch(egomotion, device="default")
            # normalize the quaternion (in case it is not normalized)
            egomotion[3:] = normalize_quaternions(egomotion[3:])

        # In case we estimate the egomotion online, we use the egomotion estimator
        elif self.egomotions_source == "online_estimates":
            egomotion = self.egomotion_estimator.estimate_egomotion(
                cur_rgb_frame=cur_rgb_frame,
                prev_rgb_frames=prev_rgb_frames,
            )
            egomotion = to_torch(egomotion, device="default")
            # normalize the quaternion (in case it is not normalized)
            egomotion[3:] = normalize_quaternions(egomotion[3:])

        else:
            raise ValueError(f"Unknown egomotions source: {self.egomotions_source}")
        return egomotion

    # --------------------------------------------------------------------------------------------------------------------

    def get_z_depth_at_pixels(
        self,
        kp_norm_coords: torch.Tensor,
        kp_frame_inds: int,
        cur_frame_idx: int,
        cur_depth_frame,
        prev_depth_frame,
    ):
        """Get the depth estimation at given key-points.
        Args:
            kp_norm_coords: the key-points normalized coordinates [n_points, 2] (units: normalized)
            kp_frame_inds: the frame index of each key-point [n_points]
            cur_frame_idx: the current frame index,
            cur_depth_frame: the current depth frame,
            prev_depth_frame: the previous depth frame,

        Returns:
            depth_z_est: the depth estimation at the queried point [n_points] (units: mm)

        Notes: - Normalized coordinates correspond to a rectilinear camera with focal length is 1 and the optical center is at (0,0))
        """
        n_points = kp_norm_coords.shape[0]
        device = kp_norm_coords.device
        dtype = kp_norm_coords.dtype
        assert cur_depth_frame.ndim == 2 and (prev_depth_frame is None or prev_depth_frame.ndim == 2)

        if self.depth_maps_source == "none":
            return torch.ones((n_points), device=device, dtype=dtype) * self.depth_default
        # transform the query points from normalized coords (rectilinear with  K=I) to pixel coordinates.

        pixels_cord = transform_rectilinear_image_norm_coords_to_pixel(
            points_nrm=kp_norm_coords,
            cam_info=self.alg_cam_info,
        )

        x = pixels_cord[:, 0]
        y = pixels_cord[:, 1]

        # get the depth estimation at the queried point from the saved depth maps
        z_depths = torch.zeros((n_points), device=device, dtype=dtype)
        kp_frame_inds = np.array(kp_frame_inds)
        kps_in_cur = kp_frame_inds == cur_frame_idx
        kps_in_prev = kp_frame_inds == cur_frame_idx - 1
        assert kps_in_cur.sum() + kps_in_prev.sum() == n_points, "sanity check XOR should be true for all points"
        if np.any(kps_in_prev):
            z_depths[kps_in_prev] = prev_depth_frame[y[kps_in_prev], x[kps_in_prev]]
        if np.any(kps_in_cur):
            z_depths[kps_in_cur] = cur_depth_frame[y[kps_in_cur], x[kps_in_cur]]

        # clip the depth
        if self.depth_lower_bound is not None or self.depth_upper_bound is not None:
            z_depths = torch.clamp(z_depths, min=self.depth_lower_bound, max=self.depth_upper_bound)
        return z_depths

    # --------------------------------------------------------------------------------------------------------------------
    def process_frames_sequence(self, scene_loader: SceneLoader):
        """Process a sequence of frames, and saves all estimations to the buffers."""
        frames_generator = scene_loader.get_frames_generator()
        prev_rgb_frame = None
        for i_frame, cur_rgb_frame in enumerate(frames_generator):
            self.process_new_frame(
                i_frame=i_frame,
                cur_rgb_frame=cur_rgb_frame,
                prev_rgb_frame=prev_rgb_frame,
            )
            prev_rgb_frame = cur_rgb_frame


# --------------------------------------------------------------------------------------------------------------------

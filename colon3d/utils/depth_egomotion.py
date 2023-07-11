import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from colon3d.utils.data_util import SceneLoader
from colon3d.utils.rotations_util import get_identity_quaternion, normalize_quaternions
from colon3d.utils.torch_util import get_device, resize_images, to_default_type, to_numpy, to_torch
from colon3d.utils.transforms_util import (
    transform_rectilinear_image_norm_coords_to_pixel,
    unproject_image_normalized_coord_to_world,
)
from endo_sfm.models_def.DispResNet import DispResNet
from endo_sfm.models_def.PoseResNet import PoseResNet

# --------------------------------------------------------------------------------------------------------------------


class DepthAndEgoMotionLoader:
    def __init__(
        self,
        scene_path: Path,
        depth_maps_source: str,
        egomotions_source: str,
        depth_and_egomotion_model_path: str | None = None,
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
        self.scene_path = scene_path
        self.depth_maps_source = depth_maps_source
        self.egomotions_source = egomotions_source
        self.depth_lower_bound = depth_lower_bound
        self.depth_upper_bound = depth_upper_bound
        self.depth_default = depth_default
        self.device = get_device()

        # Init empty buffers
        self.depth_maps_buffer = []
        self.egomotions_buffer = []
        self.depth_maps_buffer_frame_inds = []
        self.egomotions_buffer_frame_inds = []

        # Initialize egomotions

        if egomotions_source == "online_estimates":
            print("Using online egomotion estimator")
            self.egomotion_estimator = EgomotionModel(depth_and_egomotion_model_path)

        elif egomotions_source == "ground_truth":
            print("Using loaded ground-truth egomotions")
            self.init_loaded_egomotions("gt_depth_and_egomotion.h5")

        elif egomotions_source == "loaded_estimates":
            print("Using loaded estimated egomotions")
            self.init_loaded_egomotions("est_depth_and_egomotion.h5")

        if depth_maps_source == "online_estimates":
            self.depth_estimator = DepthModel(
                depth_lower_bound=self.depth_lower_bound,
                depth_upper_bound=self.depth_upper_bound,
                depth_and_egomotion_model_path=depth_and_egomotion_model_path,
            )
            print("Using online depth estimation")

        elif depth_maps_source == "ground_truth":
            print("Using loaded ground-truth depth maps")
            self.init_loaded_depth("gt_depth_and_egomotion.h5", "gt_depth_info.pkl")

        elif depth_maps_source == "loaded_estimates":
            print("Using loaded estimated depth maps")
            self.init_loaded_depth("est_depth_and_egomotion.h5", "est_depth_info.pkl")

        elif depth_maps_source == "none":
            assert depth_default is not None
        else:
            raise ValueError(f"Unknown depth maps source: {depth_maps_source}")

    # --------------------------------------------------------------------------------------------------------------------

    def init_loaded_egomotions(self, egomotions_file_name: str):
        with h5py.File(self.scene_path / egomotions_file_name, "r") as h5f:
            self.egomotions_buffer = to_default_type(h5f["egomotions"][:])  # load all into memory
        n_frames = self.egomotions_buffer.shape[0]
        self.egomotions_buffer_frame_inds = list(range(n_frames))

    # --------------------------------------------------------------------------------------------------------------------
    def init_loaded_depth(self, depth_maps_file_name: str, depth_info_file_name: str):
        """Initialize the loaded depth maps from a given file.
        The depth maps are loaded into memory.
        """
        self.depth_maps_path = self.scene_path / depth_maps_file_name
        # load the depth estimation info\metadata
        with (self.scene_path / depth_info_file_name).open("rb") as file:
            self.depth_info = to_numpy(pickle.load(file))
        # load the depth maps
        with h5py.File(self.depth_maps_path, "r") as h5f:
            self.depth_maps_buffer = to_default_type(h5f["z_depth_map"][:], num_type="float_m")  # load all into memory
        n_frames = self.depth_maps_buffer.shape[0]
        self.depth_maps_buffer_frame_inds = list(range(n_frames))
        self.loaded_depth_map_size = self.depth_info["depth_map_size"]
        self.loaded_depth_map_K = self.depth_info["K_of_depth_map"]  # the camera matrix of the depth map images
        self.n_frames = self.depth_info["n_frames"]

    # --------------------------------------------------------------------------------------------------------------------
    def process_new_frame(self, i_frame: int, cur_rgb_frame: np.ndarray, prev_rgb_frame: np.ndarray):
        """Process the current frame and add the estimated depth map to the buffer.
        If the previous frame is also given, then the egomotion will be estimated and added to the buffer.
        """
        if self.depth_maps_source == "none" or i_frame in self.depth_maps_buffer_frame_inds:
            pass  # no need to estimate

        elif self.depth_maps_source == "online_estimates":
            self.depth_maps_buffer.append(self.depth_estimator.estimate_depth_map(cur_rgb_frame))
            self.depth_maps_buffer_frame_inds.append(i_frame)

        if self.egomotions_source == "none" or i_frame in self.egomotions_buffer_frame_inds or i_frame == 0:
            pass  # no need to estimate

        elif self.egomotions_source == "online_estimates" and prev_rgb_frame is not None:
            self.egomotions_buffer.append(
                self.egomotion_estimator.estimate_egomotion(
                    prev_frame=prev_rgb_frame,
                    curr_frame=cur_rgb_frame,
                ),
            )
            self.egomotions_buffer_frame_inds.append(i_frame)

    # --------------------------------------------------------------------------------------------------------------------

    def get_depth_map_at_frame(
        self,
        frame_idx: int,
        rgb_frame: np.ndarray | None = None,
    ):
        """Get the depth estimation at a given frame.
        Notes: we assume process_new_frame was called before this function on this frame index.
        Returns:
            depth_map: the depth estimation map (units: mm)
        """
        if self.depth_maps_source == "none":
            # return the default depth map
            return torch.ones(self.loaded_depth_map_size) * self.depth_default
        # if the depth map is already in the buffer, return it, otherwise estimate it first
        if frame_idx not in self.depth_maps_buffer_frame_inds:
            assert rgb_frame is not None
            self.process_new_frame(i_frame=frame_idx, cur_rgb_frame=rgb_frame, prev_rgb_frame=None)
        buffer_idx = self.depth_maps_buffer_frame_inds.index(frame_idx)
        return self.depth_maps_buffer[buffer_idx]

    # --------------------------------------------------------------------------------------------------------------------

    def get_egomotions_at_frame(
        self,
        curr_frame_idx: int,
        cur_rgb_frame: np.ndarray | None = None,
        prev_rgb_frame: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Get the egomotion at a given frame.
        The egomotion is the 6-DOF current camera pose w.r.t. the previous camera pose.
        The egomotion is represented as a 7D vector: (x, y, z, q0, qx, qy, qz)
        x,y,z are the translation in mm.
        The quaternion (q0, qx, qy, qz) represents the rotation.
        Args:
            curr_frame_idx: the current frame index,
            prev_frame: the previous RGB frame,
            curr_frame: the current RGB frame,
        Returns:
            egomotion: the egomotion (units: mm, mm, mm, -, -, -, -)
        Notes: we assume process_new_frame was called before this function on this frame index.
        """
        if self.egomotions_source == "none" or curr_frame_idx == 0:
            # default value = identity egomotion (no motion)
            egomotion = torch.zeros((7), dtype=torch.float32, device=self.device)
            egomotion[3:] = get_identity_quaternion()
            return egomotion
        # if the egomotion is already in the buffer, return it, otherwise estimate it first
        if curr_frame_idx not in self.egomotions_buffer_frame_inds:
            assert cur_rgb_frame is not None and prev_rgb_frame is not None
            self.process_new_frame(i_frame=curr_frame_idx, cur_rgb_frame=cur_rgb_frame, prev_rgb_frame=prev_rgb_frame)
        buffer_idx = self.egomotions_buffer_frame_inds.index(curr_frame_idx)
        egomotion = self.egomotions_buffer[buffer_idx]
        egomotion = to_torch(egomotion)
        # normalize the quaternion (in case it is not normalized)
        egomotion[3:] = normalize_quaternions(egomotion[3:])

        return egomotion

    # --------------------------------------------------------------------------------------------------------------------

    def get_depth_at_nrm_points(
        self,
        queried_points_nrm: torch.Tensor,
        frame_indexes: np.ndarray | None = None,
    ):
        """Get the depth estimation at a given point in the image.

        Args:
            frame_idx: the frame index per point [n_points]
            queried_points_2d: the normalized coordinates in the (undistorted) image  [n_points x 2]

        Returns:
            depth_z_est: the depth estimation at the queried point [n_points] (units: mm)

        Notes: - Normalized coordinates correspond to a rectilinear camera with focal length is 1 and the optical center is at (0,0))
        """
        n_points = queried_points_nrm.shape[0]
        device = queried_points_nrm.device
        dtype = queried_points_nrm.dtype

        if self.depth_maps_source == "none":
            # just return the default depth for all points
            z_depths = self.depth_default * torch.ones((n_points), dtype=dtype, device=device)
            return z_depths

        # we need to transform the query points from normalized coords (rectilinear with  K=I) to the depth estimation map coordinates (rectilinear with a given K matrix)
        if self.depth_maps_source in ["ground_truth", "loaded_estimates"]:
            # the depth map size of the loaded depth maps
            depth_map_width = self.loaded_depth_map_size["width"]
            depth_map_height = self.loaded_depth_map_size["height"]
            depth_map_K = self.loaded_depth_map_K
        elif self.depth_maps_source == "online_estimates":
            depth_map_width = self.depth_estimator.depth_map_width
            depth_map_height = self.depth_estimator.depth_map_height
            depth_map_K = self.depth_estimator.depth_map_K

        # transform the query points from normalized coords (rectilinear with  K=I) to the depth estimation map coordinates (rectilinear with a given K matrix)
        # that the depth estimation map was created with
        pixels_cord = transform_rectilinear_image_norm_coords_to_pixel(
            points_nrm=queried_points_nrm,
            cam_K=depth_map_K,
            im_height=depth_map_height,
            im_width=depth_map_width,
        )
        x = pixels_cord[:, 0]
        y = pixels_cord[:, 1]

        # get the depth estimation at the queried point from the saved depth maps
        z_depths = torch.zeros((n_points), device=device, dtype=dtype)

        for frame_idx in np.unique(frame_indexes):
            buffer_idx = self.depth_maps_buffer_frame_inds.index(frame_idx)

            # notice that the depth image coordinates are (y,x) not (x,y).
            depth_out = self.depth_maps_buffer[buffer_idx][
                y[frame_indexes == frame_idx],
                x[frame_indexes == frame_idx],
            ]
            depth_out = torch.as_tensor(depth_out, device=device, dtype=dtype)
            z_depths[frame_indexes == frame_idx] = depth_out

        # clip the depth
        if self.depth_lower_bound is not None or self.depth_upper_bound is not None:
            z_depths = torch.clamp(z_depths, min=self.depth_lower_bound, max=self.depth_upper_bound)
        return z_depths

    # --------------------------------------------------------------------------------------------------------------------

    def estimate_3d_points(
        self,
        cam_poses: torch.Tensor,
        queried_points_nrm: torch.Tensor,
        frame_indexes: list | None = None,
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
            queried_points_nrm=queried_points_nrm,
            frame_indexes=frame_indexes,
        )
        p3d_est = unproject_image_normalized_coord_to_world(
            points_nrm=queried_points_nrm,
            z_depths=depth_z_est,
            cam_poses=cam_poses,
        )
        return p3d_est

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


def imgs_to_net_in(
    imgs: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    depth_map_height: int,
    depth_map_width: int,
) -> torch.Tensor:
    """Transform the input images to the network input format.
    Args:
        imgs: the input images [n_imgs x height x width x n_channels]
    Returns:
        imgs: the input images in the network format [n_imgs x n_channels x depth_map_height x depth_map_width]
    """
    if imgs.ndim == 3:
        imgs = np.expand_dims(imgs, axis=-1)  # add channel dimension
    n_imgs, height, width, n_channels = imgs.shape
    assert n_channels in [1, 3]
    if (height, width) != (depth_map_height, depth_map_width):
        # resize the images
        imgs = resize_images(imgs, new_height=depth_map_height, new_width=depth_map_width)

    # transform to channels first
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    imgs = to_torch(imgs, device=device, dtype=dtype)
    # normalize the images to fit the pre-trained weights (based on https://github.com/CapsuleEndoscope/EndoSLAM/blob/master/EndoSfMLearner/run_inference.py)
    imgs = (imgs / 255 - 0.45) / 0.225
    return imgs


# --------------------------------------------------------------------------------------------------------------------


def get_model_info(model_dir_path: Path):
    model_info_path = model_dir_path / "model_info.yaml"
    assert model_info_path.is_file(), f"Model info file not found at {model_info_path}"
    model_info = yaml.safe_load(model_info_path.open("r"))
    if "net_out_to_mm" not in model_info:
        print("net_out_to_mm not found in model info, using default value of 1.0")
        model_info["net_out_to_mm"] = 1.0

    return model_info


# --------------------------------------------------------------------------------------------------------------------


def get_camera_matrix(model_info: dict) -> np.ndarray:
    fx = model_info["fx"]
    fy = model_info["fy"]
    cx = model_info["cx"]
    cy = model_info["cy"]
    distort_pram = model_info["distort_pram"]
    assert distort_pram is None, "we assume no distortion"
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


# --------------------------------------------------------------------------------------------------------------------
class DepthModel:
    """
    The depth estimation network.
    Note that we use a network that estimates the disparity and then convert it to depth by taking 1/disparity.
    """

    def __init__(self, depth_lower_bound: float, depth_upper_bound: float, depth_and_egomotion_model_path: str) -> None:
        self.depth_lower_bound = depth_lower_bound
        self.depth_upper_bound = depth_upper_bound

        model_dir_path = Path(depth_and_egomotion_model_path)
        self.model_info = get_model_info(model_dir_path)
        # load the Disparity network
        self.disp_net_path = model_dir_path / "DispNet_best.pt"
        assert self.disp_net_path.is_file(), f"File not found: {self.disp_net_path}"
        print(f"Using pre-trained weights for DispResNet from {self.disp_net_path}")
        self.resnet_layers = self.model_info["DispResNet_layers"]
        # the dimensions of the input images to the network
        self.model_frame_height = self.model_info["frame_height"]
        self.model_frame_width = self.model_info["frame_width"]
        # the dimensions of the output depth maps are the same as the input images
        self.depth_map_width = self.model_frame_width
        self.depth_map_height = self.model_frame_height
        # the output of the network (translation part) needs to be multiplied by this number to get the depth\ego-translations in mm (based on the analysis of sample data in examine_depths.py):
        self.net_out_to_mm = self.model_info["net_out_to_mm"]
        # the camera matrix corresponding to the depth maps.
        self.depth_map_K = get_camera_matrix(self.model_info)
        self.device = get_device()
        self.dtype = torch.float64
        weights = torch.load(self.disp_net_path)
        self.disp_net = DispResNet(self.resnet_layers, pretrained=True).to(self.device)
        self.disp_net.load_state_dict(weights["state_dict"], strict=False)
        self.disp_net.to(self.device)
        self.disp_net.eval()  # switch to evaluate mode

    # --------------------------------------------------------------------------------------------------------------------
    def estimate_depth_maps(self, imgs: torch.Tensor) -> torch.Tensor:
        """Estimate the depth map from the image.

        Args:
            img (torch.Tensor): the input images [N x H x W x 3]
        Returns:
            depth_map (torch.Tensor): the estimated depth maps [N X H x W] (units: mm)
        """
        # get the original shape of the images:
        n_imgs, height, width, n_channels = imgs.shape

        # resize and change dimension order of the images to fit the network input format  # [N x 3 x H x W]
        imgs = imgs_to_net_in(imgs, self.device, self.dtype, self.depth_map_height, self.depth_map_width)
        with torch.no_grad():
            disparity_maps = self.disp_net(imgs)

        # remove the n_channels dimension
        disparity_maps.squeeze_(dim=1)  # [N x H x W]

        # convert the disparity to depth
        depth_maps = 1 / disparity_maps

        # multiply by the scale factor to get the depth in mm
        depth_maps *= self.net_out_to_mm

        # clip the depth if needed
        if self.depth_lower_bound is not None or self.depth_upper_bound is not None:
            depth_maps = torch.clamp(depth_maps, self.depth_lower_bound, self.depth_upper_bound)

        # resize the output to the original size (since the network works with a fixed size as input that might be different from the original size of the images)
        depth_maps = resize_images(depth_maps, new_height=height, new_width=width)

        return depth_maps

    # --------------------------------------------------------------------------------------------------------------------
    def estimate_depth_map(self, img: torch.Tensor) -> torch.Tensor:
        """Estimate the depth map using a single RGB image.

        Args:
            img (torch.Tensor): the input images [H x W x 3]
        Returns:
            depth_map (torch.Tensor): the estimated depth maps [H x W] (units: mm)
        """
        assert img.ndim == 3
        assert img.shape[2] == 3  # RGB
        imgs = np.expand_dims(img, axis=0)  # add n_imgs dimension
        depth_maps = self.estimate_depth_maps(imgs)
        return depth_maps[0]  # remove the n_imgs dimension


# --------------------------------------------------------------------------------------------------------------------


class EgomotionModel:
    def __init__(self, depth_and_egomotion_model_path: str) -> None:
        model_dir_path = Path(depth_and_egomotion_model_path)
        self.model_info = get_model_info(model_dir_path)
        self.pose_net_path = model_dir_path / "PoseNet_best.pt"
        assert self.pose_net_path.is_file(), f"File not found: {self.pose_net_path}"
        print(f"Using pre-trained weights for PoseNet from {self.pose_net_path}")
        self.device = get_device()
        self.dtype = torch.float64
        self.resnet_layers = self.model_info["PoseResNet_layers"]
        self.model_im_height = self.model_info["frame_height"]
        self.model_im_width = self.model_info["frame_width"]
        # the output of the network (translation part) needs to be multiplied by this number to get the depth\ego-translations in mm (based on the analysis of sample data in examine_depths.py):
        self.net_out_to_mm = self.model_info["net_out_to_mm"]
        # the camera matrix corresponding to the depth maps.
        self.depth_map_K = get_camera_matrix(self.model_info)
        self.pose_net = PoseResNet(self.resnet_layers, pretrained=True).to(self.device)
        weights = torch.load(self.pose_net_path)
        self.pose_net.load_state_dict(weights["state_dict"], strict=False)
        self.pose_net.to(self.device)
        self.pose_net.eval()  # switch to evaluate mode

        # --------------------------------------------------------------------------------------------------------------------

    def estimate_egomotions(self, from_imgs: np.ndarray, to_imgs: np.ndarray) -> torch.Tensor:
        """Estimate the 6DOF egomotion from the from image to to image.
        Args:
            from_imgs: the 'from' images (target) [N x 3 x H x W] where N is the number of image pairs
            to_imgs: the corresponding 'to' images  (reference) [N X 3 x H x W]
        Returns:
            egomotions: the estimated egomotions [N x 7] 6DoF pose parameters from from_imgs to to_imgs, in the format:
                (x,y,z,qw,qx,qy,qz) where (x, y, z) is the translation [mm] and (qw, qx, qy , qz) is the unit-quaternion of the rotation.
        """
        n_imgs = len(from_imgs)
        assert len(to_imgs) == n_imgs
        from_imgs = imgs_to_net_in(from_imgs, self.device, self.dtype, self.model_im_height, self.model_im_width)
        to_imgs = imgs_to_net_in(to_imgs, self.device, self.dtype, self.model_im_height, self.model_im_width)
        with torch.no_grad():
            pose_out = self.pose_net(from_imgs, to_imgs)
        # this returns the estimated egomotion [N x 6] 6DoF pose parameters from target to reference  in the order of tx, ty, tz, rx, ry, rz
        trans = pose_out[:, :3]
        # to get the a unit-quaternion of the rotation, concatenate a 1 at the beginning of the vector and normalize
        rot_quat = torch.ones((n_imgs, 4), device=self.device, dtype=self.dtype)
        rot_quat[:, 1:] = pose_out[:, 3:]
        rot_quat = rot_quat / torch.norm(rot_quat, dim=1, keepdim=True)
        # multiply the translation by the conversion factor to get mm units
        trans *= self.net_out_to_mm
        egomotions = torch.cat((trans, rot_quat), dim=1)
        return egomotions

    # --------------------------------------------------------------------------------------------------------------------

    def estimate_egomotion(self, prev_frame: np.ndarray, curr_frame: np.ndarray):
        """Estimate the 6DOF egomotion from the previous frame to the current frame."""
        assert prev_frame.ndim == 3
        assert curr_frame.ndim == 3
        egomotion = self.estimate_egomotions(
            from_imgs=np.expand_dims(prev_frame, axis=0),
            to_imgs=np.expand_dims(curr_frame, axis=0),
        )[0]
        return egomotion


# --------------------------------------------------------------------------------------------------------------------

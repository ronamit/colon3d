import pickle
from pathlib import Path

import h5py
import numpy as np
import torch

from colon3d.rotations_util import get_identity_quaternion, normalize_quaternions
from colon3d.torch_util import get_device, to_numpy
from colon3d.transforms_util import unproject_image_normalized_coord_to_world
from endo_sfm_learner.models.DispResNet import DispResNet
from endo_sfm_learner.models.PoseResNet import PoseResNet

# --------------------------------------------------------------------------------------------------------------------


class DepthAndEgoMotionLoader:
    def __init__(
        self,
        scene_path: Path,
        depth_maps_source: str,
        egomotions_source: str,
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

        # Initialize egomotions
        if egomotions_source == "online_estimates":
            print("Using online egomotion estimator")
            # initialize the egomotion estimator
            self.egomotion_estimator = PoseNet()
        elif egomotions_source == "ground_truth":
            print("Using loaded ground-truth egomotions")
            with h5py.File( scene_path / "gt_depth_and_egomotion.h5", "r") as h5f:
                self.loaded_egomotions = h5f["egomotions"][:]  # load into memory
        elif egomotions_source == "loaded_estimates":
            print("Using loaded estimated egomotions")
            with h5py.File(scene_path / "est_depth_and_egomotion.h5", "r") as h5f:
                self.loaded_egomotions = h5f["egomotions"][:]  # load into memory
        # Init depth maps.
        if depth_maps_source == "online_estimates":
            # initialize the depth estimator
            self.depth_estimator = DepthNet(
                depth_lower_bound=self.depth_lower_bound,
                depth_upper_bound=self.depth_upper_bound,
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

    # --------------------------------------------------------------------------------------------------------------------

    def init_loaded_depth(self, depth_maps_file_name: str, depth_info_file_name: str):
        """Initialize the loaded depth maps from a given file.
        The depth maps are loaded into memory.
        """
        with h5py.File(self.scene_path / depth_maps_file_name, "r") as h5f:
            self.loaded_depth_maps = h5f["z_depth_map"][:]
        # load the depth estimation info\metadata
        with (self.scene_path / depth_info_file_name).open("rb") as file:
            self.depth_info = to_numpy(pickle.load(file))
        self.depth_map_size = self.depth_info["depth_map_size"]
        self.de_K = self.depth_info["K_of_depth_map"]  # the camera matrix of the depth map images
        self.n_frames = self.depth_info["n_frames"]

    # --------------------------------------------------------------------------------------------------------------------

    def get_depth_map_at_frame(
        self,
        frame_idx: int,
        rgb_frame: torch.Tensor | None = None,
    ):
        """Get the depth estimation at a given frame.
        Returns:
            depth_map: the depth estimation map (units: mm)
        """
        if self.depth_maps_source in ["ground_truth", "loaded_estimates"]:
            return self.loaded_depth_maps[frame_idx]
        if self.depth_maps_source == "online_estimates":
            return self.depth_estimator.get_depth_map(rgb_frame)
        return None

    # --------------------------------------------------------------------------------------------------------------------

    def get_egomotions_at_frame(
        self,
        curr_frame_idx: int,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
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
        """
        if self.egomotions_source in ["ground_truth", "loaded_estimates"]:
            egomotion = self.loaded_egomotions[curr_frame_idx]
            egomotion = torch.from_numpy(egomotion).to(self.device)
            # normalize the quaternion (in case it is not normalized)
            egomotion[3:] = normalize_quaternions(egomotion[3:])
        elif self.egomotions_source == "online_estimates":
            egomotion = self.egomotion_estimator.get_egomotions(
                tgt_imgs=np.expand_dims(prev_frame, axis=0),
                ref_imgs=np.expand_dims(curr_frame, axis=0),
            )[0]
        else:
            # default value = identity egomotion (no motion)
            egomotion = torch.zeros((7), dtype=torch.float32)
            egomotion[3:] = get_identity_quaternion()
        assert len(egomotion) == 7  # (x, y, z, qw, qx, qy, qz)
        return egomotion

    # --------------------------------------------------------------------------------------------------------------------

    def get_depth_at_nrm_points(
        self,
        frame_indexes: np.ndarray,
        queried_points_nrm: torch.Tensor,
    ):
        """Get the depth estimation at a given point in the image.

        Args:
            frame_idx: the frame index per point [n_points]
            queried_points_2d: the normalized coordinates in the (undistorted) image  [n_points x 2]

        Returns:
            depth_z_est: the depth estimation at the queried point [n_points] (units: mm)

        Note: Normalized coordinates correspond to a rectilinear camera with focal length is 1 and the optical center is at (0,0))
        """
        if self.depth_maps_source in ["ground_truth", "loaded_estimates"]:
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
            z_depths = torch.zeros((n_points), device=device, dtype=dtype)
            # with h5py.File(self.file_path, "r") as h5f:
            for frame_idx in np.unique(frame_indexes):
                # notice that the depth image coordinates are (y,x) not (x,y).
                depth_out = self.loaded_depth_maps[frame_idx][
                    y[frame_indexes == frame_idx],
                    x[frame_indexes == frame_idx],
                ]
                depth_out = torch.as_tensor(depth_out, device=device, dtype=dtype)
                z_depths[frame_indexes == frame_idx] = depth_out
            # clip the depth
            if self.depth_lower_bound is not None or self.depth_upper_bound is not None:
                z_depths = torch.clamp(z_depths, min=self.depth_lower_bound, max=self.depth_upper_bound)
        else:
            # return the default depth
            z_depths = self.depth_default * torch.ones_like(queried_points_nrm[:, 0])
        return z_depths

    # --------------------------------------------------------------------------------------------------------------------

    def estimate_3d_points(
        self,
        frame_indexes: list,
        cam_poses: torch.Tensor,
        queried_points_nrm: torch.Tensor,
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
        )
        p3d_est = unproject_image_normalized_coord_to_world(
            points_nrm=queried_points_nrm,
            z_depths=depth_z_est,
            cam_poses=cam_poses,
        )
        return p3d_est


# --------------------------------------------------------------------------------------------------------------------


def imgs_to_net_in(
    imgs: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    net_input_height: int,
    net_input_width: int,
) -> torch.Tensor:
    """Transform the input images to the network input format.
    Args:
        imgs (np.ndarray): the input images [n_imgs x height x width x n_channels]
    Returns:
        imgs_net_in (torch.Tensor): the input images in the network format [n_imgs x n_channels x net_input_height x net_input_width]
    """
    if imgs.ndim == 3:
        imgs = np.expand_dims(imgs, axis=-1)  # add channel dimension
    n_imgs, height, width, n_channels = imgs.shape
    assert n_channels in [1, 3]
    assert (height, width) == (net_input_height, net_input_width)
    # transform to channels first
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    imgs = torch.as_tensor(imgs, device=device, dtype=dtype)
    # normalize the images to fit the pre-trained weights
    imgs = (imgs / 255 - 0.45) / 0.225
    return imgs


# --------------------------------------------------------------------------------------------------------------------
class DepthNet:
    def __init__(self, depth_lower_bound: float, depth_upper_bound: float) -> None:
        self.depth_lower_bound = depth_lower_bound
        self.depth_upper_bound = depth_upper_bound
        # load the disparity-estimation network (disparity = 1/depth)
        self.pretrained_disp = Path("pretrained/dispnet_model_best.pt")
        self.resnet_layers = 18
        self.net_input_height = 256
        self.net_input_width = 256
        self.net_out_to_mm = 41.1334 # the output of the depth network needs to be multiplied by this number to get the depth in mm (based on the analysis of sample data in examine_depths.py)
        assert self.pretrained_disp.is_file()
        self.device = get_device()
        self.dtype = torch.float32
        print(f"Using pre-trained weights for DispResNet from {self.pretrained_disp}")
        weights = torch.load(self.pretrained_disp)
        self.disp_net = DispResNet(self.resnet_layers, pretrained=False).to(self.device)
        self.disp_net.load_state_dict(weights["state_dict"], strict=False)
        self.disp_net.to(self.device)
        self.disp_net.eval()  # switch to evaluate mode

    # --------------------------------------------------------------------------------------------------------------------
    def get_depth_maps(self, imgs: torch.Tensor) -> torch.Tensor:
        """Estimate the depth map from the image.

        Args:
            img (torch.Tensor): the input images [N x H x W x 3]
        Returns:
            depth_map (torch.Tensor): the estimated depth maps [N X H x W] (units: mm)
        """
        imgs = imgs_to_net_in(imgs, self.device, self.dtype, self.net_input_height, self.net_input_width)
        with torch.no_grad():
            output_disparity = self.disp_net(imgs)
        # remove the n_channels dimension
        output_disparity.squeeze_(dim=1) # [N x H x W]
        output_depth = 1 / output_disparity
        # clip the depth
        if self.depth_lower_bound is not None or self.depth_upper_bound is not None:
            output_depth = torch.clamp(output_depth, self.depth_lower_bound, self.depth_upper_bound)
        # make sure the output is detached from the graph
        output_depth = output_depth.detach()
        # multiply by the scale factor to get the depth in mm
        output_depth *= self.net_out_to_mm
        return output_depth

    # --------------------------------------------------------------------------------------------------------------------
    def get_depth_map(self, img: torch.Tensor) -> torch.Tensor:
        """Estimate the depth map using a single RGB image.

        Args:
            img (torch.Tensor): the input images [H x W x 3]
        Returns:
            depth_map (torch.Tensor): the estimated depth maps [H x W] (units: mm)
        """
        assert img.ndim == 3
        assert img.shape[2] == 3 # RGB
        imgs = np.expand_dims(img, axis=0) # add n_imgs dimension
        depth_maps = self.get_depth_maps(imgs)
        return depth_maps[0] # remove the n_imgs dimension


# --------------------------------------------------------------------------------------------------------------------


class PoseNet:
    def __init__(self) -> None:
        self.device = get_device()
        self.dtype = torch.float32
        self.resnet_layers = 18
        self.net_input_height = 256
        self.net_input_width = 256
        self.pretrained_pose = Path("pretrained/exp_pose_model_best.pt")
        self.net_out_to_mm = 41.1334 # the output of the network (translation part) needs to be multiplied by this number to get the depth in mm (based on the analysis of sample data in examine_depths.py)
        assert self.pretrained_pose.is_file()
        print(f"Using pre-trained weights for PoseNet from {self.pretrained_pose}")
        self.pose_net = PoseResNet(self.resnet_layers, pretrained=False).to(self.device)
        weights = torch.load(self.pretrained_pose)
        self.pose_net.load_state_dict(weights["state_dict"], strict=False)
        self.pose_net.to(self.device)
        self.pose_net.eval()  # switch to evaluate mode

        # --------------------------------------------------------------------------------------------------------------------

    def get_egomotions(self, tgt_imgs: np.ndarray, ref_imgs: np.ndarray) -> torch.Tensor:
        """Estimate the egomotion from the target images to the reference images.
        Args:
            tgt_img: the target images (from) [N x 3 x H x W]
            ref_imgs: the reference images (to) [N X 3 x H x W]
        Returns:
            egomotions: the estimated egomotion [N x 7] 6DoF pose parameters from target to reference, in the format:
                        (x,y,z,qw,qx,qy,qz) where (x, y, z) is the translation [mm] and (qw, qx, qy , qz) is the unit-quaternion of the rotation.
        """
        n_imgs = len(tgt_imgs)
        assert len(ref_imgs) == n_imgs
        tgt_imgs = imgs_to_net_in(tgt_imgs, self.device, self.dtype, self.net_input_height, self.net_input_width)
        ref_imgs = imgs_to_net_in(ref_imgs, self.device, self.dtype, self.net_input_height, self.net_input_width)
        with torch.no_grad():
            pose_out = self.pose_net(tgt_imgs, ref_imgs)
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

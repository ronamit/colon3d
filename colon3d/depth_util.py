import pickle
from pathlib import Path

import h5py
import numpy as np
import torch

import endo_sfm_learner
from colon3d.alg_settings import AlgorithmParam
from colon3d.rotations_util import get_identity_quaternion, normalize_quaternions
from colon3d.torch_util import get_device, np_func, to_numpy
from colon3d.transforms_util import unproject_image_normalized_coord_to_world

# --------------------------------------------------------------------------------------------------------------------


class DepthAndEgoMotionLoader:
    def __init__(self, example_path: Path, depth_maps_source: str, egomotions_source: str, alg_prm: AlgorithmParam):
        """Wrapper for the depth & ego-motion estimation model.

        Args:
            example_path: path to the example folder
            depth_maps_source: the source of the depth maps, if 'ground_truth' then the ground truth depth maps will be loaded,
                if 'online_estimates' then the depth maps will be estimated online by the algorithm
                if 'loaded_estimates' then the depth maps estimations will be loaded,
                if 'none' then no depth maps will not be used,
            egomotions_source: the source of the egomotion, if 'ground_truth' then the ground truth egomotion will be loaded,
                if 'online_estimates' then the egomotion will be estimated online by the algorithm
                if 'loaded_estimates' then the egomotion estimations will be loaded,
                if 'none' then no egomotion will not be used,
        """
        self.depth_maps_source = depth_maps_source
        self.egomotions_source = egomotions_source
        self.z_depth_lower_bound = alg_prm.z_depth_lower_bound
        self.z_depth_upper_bound = alg_prm.z_depth_upper_bound
        self.z_depth_default = alg_prm.z_depth_default
        self.example_path = Path(example_path).expanduser()

        # Load egomotions
        if egomotions_source == "ground_truth":
            self.egomotions_file_path = self.example_path / "gt_depth_and_egomotion.h5"
            print("Using ground-truth egomotions from: ", self.egomotions_file_path)
        elif egomotions_source == "loaded_estimates":
            self.egomotions_file_path = self.example_path / "est_depth_and_egomotion.h5"
            print("Using estimated egomotions from: ", self.egomotions_file_path)
        elif egomotions_source == "online_estimates":
            # initialize the egomotion estimator
            self.egomotion_estimator = PoseNet()
        elif egomotions_source == "none":
            self.egomotions_file_path = None
        else:
            raise ValueError("Invalid egomotions_source: ", egomotions_source)
        if self.egomotions_file_path is not None:
            with h5py.File(self.egomotions_file_path, "r") as h5f:
                self.loaded_egomotions = h5f["egomotions"][:]  # load into memory

        # Load depth maps
        if depth_maps_source == "ground_truth":
            self.depth_maps_file_path = self.example_path / "gt_depth_and_egomotion.h5"
            self.depth_info_file_path = self.example_path / "gt_depth_info.pkl"
            print("Using ground-truth depth maps from: ", self.depth_maps_file_path)
        elif depth_maps_source == "loaded_estimates":
            self.depth_maps_file_path = self.example_path / "est_depth_and_egomotion.h5"
            self.depth_info_file_path = self.example_path / "est_depth_info.pkl"
            print("Using estimated depth maps from: ", self.depth_maps_file_path)
        elif depth_maps_source == "online_estimates":
            # initialize the depth estimator
            self.depth_estimator = DepthNet()
        elif depth_maps_source == "none":
            self.depth_maps_file_path = None
        else:
            raise ValueError("Invalid depth_maps_source: ", depth_maps_source)
        if self.depth_maps_file_path is not None:
            with h5py.File(self.depth_maps_file_path, "r") as h5f:
                self.loaded_depth_maps = h5f["z_depth_map"][:]  # load into memory
            # load the depth estimation info\metadata
            with (self.depth_info_file_path).open("rb") as file:
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

    def get_egomotions_at_frames(
        self,
        frame_indexes: np.ndarray,
        pre_rgb_frames: torch.Tensor | None = None,
        cur_rgb_frames: torch.Tensor | None = None,
    ):
        """Get the egomotion at a given frame.
        The egomotion is the 6-DOF current camera pose w.r.t. the previous camera pose.
        The egomotion is represented as a 7D vector: (x, y, z, q0, qx, qy, qz)
        x,y,z are the translation in mm.
        The quaternion (q0, qx, qy, qz) represents the rotation.
        """
        if self.egomotions_source in ["ground_truth", "loaded_estimates"]:
            egomotions = self.loaded_egomotions[frame_indexes]
            assert egomotions.shape[1] == 7  # (x, y, z, q0, qx, qy, qz)
            # normalize the quaternion (in case it is not normalized)
            egomotions[:, 3:] = np_func(normalize_quaternions)(egomotions[:, 3:])
        elif self.egomotions_source == "online_estimates":
            egomotions = self.egomotion_estimator.get_egomotions(tgt_imgs=pre_rgb_frames, ref_imgs=cur_rgb_frames)
        else:
            # default value = identity egomotion (no motion)
            egomotions = torch.zeros((len(frame_indexes), 7), dtype=torch.float32)
            egomotions[:, 3:] = get_identity_quaternion()
            egomotions = to_numpy(egomotions)
        return egomotions

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
            z_depths = torch.clamp(z_depths, self.z_depth_lower_bound, self.z_depth_upper_bound)
        else:
            # return the default depth
            z_depths = self.z_depth_default * torch.ones_like(queried_points_nrm[:, 0])
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


# --------------------------------------------------------------------------------------------------------------------
class DepthNet:
    def __init__(self, z_depth_lower_bound: float, z_depth_upper_bound: float) -> None:
        self.z_depth_lower_bound = z_depth_lower_bound
        self.z_depth_upper_bound = z_depth_upper_bound
        # load the disparity-estimation network (disparity = 1/depth)
        self.pretrained_disp = Path("pretrained/dispnet_model_best.pt")
        self.resnet_layers = 18
        assert self.pretrained_disp.is_file()
        self.device = get_device()
        print(f"Using pre-trained weights for DispResNet from {self.pretrained_disp}")
        weights = torch.load(self.pretrained_disp)
        self.disp_net = endo_sfm_learner.models.DispResNet(self.resnet_layers, pretrained=False).to(self.device)
        self.disp_net.load_state_dict(weights["state_dict"], strict=False)
        self.disp_net.to(self.device)
        self.disp_net.eval()  # switch to evaluate mode

    # --------------------------------------------------------------------------------------------------------------------
    def estimate_depth_maps(self, imgs: torch.Tensor) -> torch.Tensor:
        """Estimate the depth map from the image.

        Args:
            img (torch.Tensor): the input images [N x 3 x H x W]
        Returns:
            depth_map (torch.Tensor): the estimated depth maps [N X H x W]
        """
        assert imgs.shape[0] == 4
        output_disparity = self.disp_net(imgs)
        output_depth = 1 / output_disparity
        # clip the depth
        output_depth = torch.clamp(output_depth, self.z_depth_lower_bound, self.z_depth_upper_bound)
        return output_depth

    # --------------------------------------------------------------------------------------------------------------------
    def estimate_depth_map(self, img: torch.Tensor) -> torch.Tensor:
        """Estimate the depth map from the image.

        Args:
            img (torch.Tensor): the input images [3 x H x W]
        Returns:
            depth_map (torch.Tensor): the estimated depth maps [H x W]
        """
        assert img.shape[0] == 3
        return self.estimate_depth_maps(img.unsqueeze(0))[0]


# --------------------------------------------------------------------------------------------------------------------


class PoseNet:
    def __init__(self) -> None:
        self.device = get_device()
        self.resnet_layers = 18
        self.pretrained_pose = Path("pretrained/exp_pose_model_best.pt")
        assert self.pretrained_pose.is_file()
        print(f"Using pre-trained weights for PoseNet from {self.pretrained_pose}")
        self.pose_net = endo_sfm_learner.models.PoseResNet(self.resnet_layers, pretrained=False).to(self.device)
        weights = torch.load(self.pretrained_pose)
        self.pose_net.load_state_dict(weights["state_dict"], strict=False)
        self.pose_net.to(self.device)
        self.pose_net.eval()  # switch to evaluate mode

        # --------------------------------------------------------------------------------------------------------------------

    def get_egomotions(self, tgt_imgs: torch.Tensor, ref_imgs: torch.Tensor) -> torch.Tensor:
        """Estimate the egomotion from the target images to the reference images.
        Args:
            tgt_img (torch.Tensor): the target images (from) [N x 3 x H x W]
            ref_imgs (torch.Tensor): the reference images (to) [N X 3 x H x W]
        Returns:
            egomotions: the estimated egomotion [N x 7] 6DoF pose parameters from target to reference, in the format:
                        (x,y,z,qw,qx,qy,qz) where (x, y, z) is the translation [mm] and (qw, qx, qy , qz) is the unit-quaternion of the rotation.
        """
        assert tgt_imgs.shape[0] == 4
        pose_out = self.pose_net(tgt_imgs, ref_imgs)
        # this returns the estimated egomotion [N x 6] 6DoF pose parameters from target to reference  in the order of tx, ty, tz, rx, ry, rz
        # to get a a unit-quaternion of the rotation, use the following
        rot_quat = torch.cat(torch.ones_like(pose_out[:, 0]), pose_out[:, 3:], dim=1)
        rot_quat = rot_quat / torch.norm(rot_quat, dim=1, keepdim=True)
        trans = pose_out[:, :3]
        egomotions = torch.cat((trans, rot_quat), dim=1)
        return egomotions


# --------------------------------------------------------------------------------------------------------------------

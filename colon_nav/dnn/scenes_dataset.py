import random
from pathlib import Path

import h5py
import numpy as np
import yaml
from PIL import Image
from torch.utils import data

from colon_nav.dnn.data_transforms import get_train_transform, get_val_transform
from colon_nav.dnn.model_info import ModelInfo
from colon_nav.util.data_util import get_all_scenes_paths_in_dir
from colon_nav.util.general_util import save_rgb_and_depth_subplots
from colon_nav.util.torch_util import to_default_type, to_torch

# ---------------------------------------------------------------------------------------------------------------------


class ScenesDataset(data.Dataset):
    # ---------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        scenes_paths: list,
        model_info: ModelInfo,
        split_name: str,
        n_sample_lim: int = 0,
        load_gt_depth: bool = False,
        load_gt_pose: bool = False,
        n_scenes_lim: int = 0,
        rgb_img_height = 475,
        rgb_img_width = 475,
        depth_map_height = 475,
        depth_map_width = 475,
    ):
        r"""Initialize the DatasetLoader class
        Args:
            scenes_paths (list): List of paths to the scenes
            n_scenes_lim (int): Limit the number of scenes to load (for debugging) if 0 then no limit
            load_gt_depth (bool): Whether to add the depth map of the target frame to each sample (default: False)
            load_gt_pose (bool): Whether to add the ground-truth pose change between from target to the reference frames,  each sample (default: False)
            transforms: transforms to apply to each sample  (in order)
            n_sample_lim (int): Limit the number of samples to load (for debugging) if 0 then no limit
            rgb_img_height, rgb_img_width (int): The height and width of the RGB images are resized to to keep the training batches uniform
            depth_map_height, depth_map_width (int): The height and width of the depth map that the GT depth maps and the DepthNet output are resized to
        """
        self.scenes_paths = scenes_paths
        if n_scenes_lim > 0:
            self.scenes_paths = self.scenes_paths[:n_scenes_lim]
        print(f"Dataset type; {split_name}, n_scenes: {len(self.scenes_paths)}")
        self.model_info = model_info
        self.split_name = split_name
        self.load_gt_depth = load_gt_depth
        self.load_gt_pose = load_gt_pose
        self.rgb_img_size = (rgb_img_height, rgb_img_width)
        self.depth_map_size = (depth_map_height, depth_map_width)
        #  ref_frame_shifts (list[int]) The time shifts of the reference frames w.r.t. the target frame
        self.ref_frame_shifts = model_info.ref_frame_shifts
        # List of all target frames (each target frame is a dict with the scene index and the target frame index)
        self.target_ids = []
        # the first frame that can be a target (since we need to have enough reference frames before it)
        min_tgt_frame_idx = -max(self.ref_frame_shifts)

        # go over all the scenes in the dataset
        self.frames_paths_per_scene = []
        n_frames_per_scene = []
        for scene_path in self.scenes_paths:
            # Get the list of all the frames' paths in the scene, sorted by their index
            frames_paths = list((scene_path / "RGB_Frames").glob("*.png"))
            frames_paths.sort(key=lambda x: int(x.stem))
            if n_sample_lim > 0:
                frames_paths = frames_paths[:n_sample_lim]
            self.frames_paths_per_scene.append(frames_paths)
            n_frames_per_scene.append(len(frames_paths))

        for i_scene, frames_paths in enumerate(self.frames_paths_per_scene):
            n_frames = len(frames_paths)
            assert n_frames > 0, f"Scene {i_scene} has no frames"
            # set the scene index and the target frame index for each sample (later we will set the reference frames indices)
            for i_frame in range(min_tgt_frame_idx, n_frames):
                self.target_ids.append({"scene_idx": i_scene, "target_frame_idx": i_frame})

        # Create transforms
        if self.split_name == "train":
            self.transform = get_train_transform(model_info=model_info, rgb_img_size=self.rgb_img_size, depth_map_size=self.depth_map_size)
        elif self.split_name == "val":
            self.transform = get_val_transform(model_info=model_info, rgb_img_size=self.rgb_img_size, depth_map_size=self.depth_map_size)
        else:
            raise ValueError(f"Unknown dataset type: {self.split_name}")

    # ---------------------------------------------------------------------------------------------------------------------

    def __len__(self):
        return len(self.target_ids)

    # ---------------------------------------------------------------------------------------------------------------------

    def __getitem__(self, index: int, debug=False) -> dict:
        """Get a sample from the dataset
        Args:
            index (int): The index of the sample in the dataset.
            debug (bool): Whether to plot the sample data (default: False)
        Returns:
            sample (dict): A sample from the dataset. with fields:
                "scene_index" (int): The index of the scene in the dataset.
                "target_frame_idx" (int): The index of the target frame in the scene.
                "color" (list[torch.Tensor]): The RGB images of the reference and target frame (each frame is a tensor of shape [H, W, 3])
                "depth_gt" (list[torch.Tensor]): The ground-truth depth map of the reference and target frame (each frame is a tensor of shape [H, W])
                "abs_pose" (list[torch.Tensor]): The ground-truth pose change between the target and the reference frames (each pose is a tensor of shape [7])
                "K" (torch.Tensor): The camera intrinsics matrix (shape [3, 3])
        """
        sample = {}
        target_id = self.target_ids[index]
        scene_index = target_id["scene_idx"]
        target_frame_idx = target_id["target_frame_idx"]
        scene_path = self.scenes_paths[scene_index]
        scene_frames_paths = self.frames_paths_per_scene[scene_index]
        sample["scene_index"] = scene_index  # for debugging
        sample["target_frame_idx"] = target_frame_idx  # for debugging

        # get the camera intrinsics matrix
        with (scene_path / "meta_data.yaml").open() as file:
            scene_metadata = yaml.safe_load(file)
            intrinsics_orig = get_camera_matrix(scene_metadata)

        sample["K"] = to_torch(intrinsics_orig)

        # load the RGB images
        all_frames_shift = [*self.ref_frame_shifts, 0]
        for shift in all_frames_shift:
            frame_idx = target_frame_idx + shift
            frame_path = scene_frames_paths[frame_idx]
            sample[("color", shift)] = Image.open(frame_path)

        # load the ground-truth depth map and the ground-truth poses (if available)
        if (self.load_gt_depth or self.load_gt_pose) and (scene_path / "gt_3d_data.h5").exists():
            with h5py.File((scene_path / "gt_3d_data.h5").resolve(), "r") as h5f:
                for shift in all_frames_shift:
                    frame_idx = target_frame_idx + shift
                    if self.load_gt_depth:
                        depth_gt = h5f["z_depth_map"][frame_idx]
                        depth_gt = to_default_type(depth_gt)
                        sample[("depth_gt", shift)] = depth_gt
                    if self.load_gt_pose:
                        abs_pose = h5f["cam_poses"][frame_idx]
                        abs_pose = to_default_type(abs_pose)
                        sample[("abs_pose", shift)] = abs_pose
        if debug:
            self.debug_plot_sample(sample, all_frames_shift)

        # apply the transform
        if self.transform:
            sample = self.transform(sample)
        return sample

    # ---------------------------------------------------------------------------------------------------------------------
    def debug_plot_sample(self, sample: dict, all_frames_shift) -> None:
        # DEBUG: plot the sample data
        rgb_imgs = [sample[("color", shift)] for shift in all_frames_shift]
        depth_imgs = [sample[("depth_gt", shift)] for shift in all_frames_shift]
        save_path = Path("temp") / "sample.png"
        save_rgb_and_depth_subplots(
            rgb_imgs=rgb_imgs,
            depth_imgs=depth_imgs,
            frame_names=[f"Shift:{shift}" for shift in all_frames_shift],
            save_path=save_path,
        )
        raise ValueError(f"Debugging: sample saved to {save_path}")

    # ---------------------------------------------------------------------------------------------------------------------

    def get_scene_metadata(self, i_scene: int = 0) -> dict:
        scene_path = self.scenes_paths[i_scene]
        with (scene_path / "meta_data.yaml").open() as file:
            metadata = yaml.safe_load(file)
        return metadata


# ---------------------------------------------------------------------------------------------------------------------


def get_camera_matrix(scene_metadata: dict) -> np.ndarray:
    cam_K = np.zeros((3, 3), dtype=np.float32)
    cam_K[0, 0] = scene_metadata["fx"]
    cam_K[1, 1] = scene_metadata["fy"]
    cam_K[0, 2] = scene_metadata["cx"]
    cam_K[1, 2] = scene_metadata["cy"]
    cam_K[2, 2] = 1
    return cam_K


# ---------------------------------------------------------------------------------------------------------------------


def get_scenes_dataset_random_split(dataset_path: Path, validation_ratio: float):
    all_scenes_paths = get_all_scenes_paths_in_dir(dataset_path, with_targets=False)
    random.shuffle(all_scenes_paths)
    n_all_scenes = len(all_scenes_paths)
    n_train_scenes = int(n_all_scenes * (1 - validation_ratio))
    train_scenes_paths = all_scenes_paths[:n_train_scenes]
    val_scenes_paths = all_scenes_paths[n_train_scenes:]
    return train_scenes_paths, val_scenes_paths


# ---------------------------------------------------------------------------------------------------------------------

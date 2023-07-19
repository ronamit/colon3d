import random

import h5py
import numpy as np
import yaml
from imageio import imread
from torch.utils import data

from colon3d.util.torch_util import to_default_type
from endo_sfm.custom_transforms import Compose

# ---------------------------------------------------------------------------------------------------------------------


def load_as_float(path):
    return imread(path).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------------------


def get_camera_matrix(metadata: dict) -> np.ndarray:
    cam_K = np.zeros((3, 3), dtype=np.float32)
    cam_K[0, 0] = metadata["fx"]
    cam_K[1, 1] = metadata["fy"]
    cam_K[0, 2] = metadata["cx"]
    cam_K[1, 2] = metadata["cy"]
    cam_K[2, 2] = 1
    return cam_K


# ---------------------------------------------------------------------------------------------------------------------


class ScenesDataset(data.Dataset):
    # ---------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        scenes_paths: list,
        sequence_length: int,
        load_tgt_depth: bool,
        transform: Compose | None,
        skip_frames: int = 1,
    ):
        """Initialize the DatasetLoader class
        Args:
            scenes_paths (list): List of paths to the scenes
            sequence_length (int): Number of consecutive frames in each sample (the reference frame is in the middle)
            load_tgt_depth (bool): Whether to add the depth map of the target frame to each sample (default: False)
            transform (Compose | None): transform to apply to each sample (default: None)
            skip_frames (int): Number of frames to skip between samples.
                e.g., if sample_frames_len==3 and skip_frames==1, then the samples will be: [0,1,2], [1,2,3], [2,3,4], ...
        """
        self.scenes_paths = scenes_paths
        self.load_tgt_depth = load_tgt_depth
        self.sequence_length = sequence_length
        # each sample is a list of n consecutive frames, where the reference frame is in the middle:
        target_ind_in_sample = (self.sequence_length - 1) // 2
        self.transform = transform
        self.samples = []
        self.frame_paths_per_scene = []
        # go over all the scenes in the dataset
        for scene_index, scene_path in enumerate(self.scenes_paths):
            frames_paths = [
                frame_path
                for frame_path in (scene_path / "RGB_Frames").iterdir()
                if frame_path.is_file() and frame_path.name.endswith(".png")
            ]
            self.frame_paths_per_scene.append(frames_paths)
            frames_paths.sort()
            n_frames = len(frames_paths)
            for seq_start_idx in range(0, n_frames - self.sequence_length + 1, skip_frames):
                seq_inds = list(range(seq_start_idx, seq_start_idx + self.sequence_length))
                target_ind = seq_inds[target_ind_in_sample]
                ref_inds = seq_inds[:target_ind_in_sample] + seq_inds[target_ind_in_sample + 1 :]
                sample = {"scene_index": scene_index, "target_frame_index": target_ind, "ref_frames_inds": ref_inds}
                self.samples.append(sample)
        random.shuffle(self.samples)

    # ---------------------------------------------------------------------------------------------------------------------

    def get_scene_metadata(self, scene_index: int) -> dict:
        """Get the metadata of a scene
        Args:
            scene_index (int): Index of the scene
        Returns:
            dict: Dictionary containing the metadata of the scene
        """
        scene_path = self.scenes_paths[scene_index]
        with (scene_path / "meta_data.yaml").open() as file:
            metadata = yaml.load(file, Loader=yaml.FullLoader)
        return metadata

    # ---------------------------------------------------------------------------------------------------------------------

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        scene_index = sample["scene_index"]
        scene_path = self.scenes_paths[scene_index]
        scene_frames_paths = self.frame_paths_per_scene[scene_index]
        # get the camera intrinsics matrix
        with (scene_path / "meta_data.yaml").open() as file:
            metadata = yaml.load(file, Loader=yaml.FullLoader)
            intrinsics_orig = get_camera_matrix(metadata)
        # load the target frame
        target_frame_ind = sample["target_frame_index"]
        target_frame_path = scene_frames_paths[target_frame_ind]
        tgt_img = load_as_float(target_frame_path)
        # load the reference frames
        ref_imgs = []
        for ref_frame_ind in sample["ref_frames_inds"]:
            ref_frame_path = scene_frames_paths[ref_frame_ind]
            ref_frame = load_as_float(ref_frame_path)
            ref_imgs.append(ref_frame)

        # list of images to apply the transforms on (the target frame and the reference frames)
        imgs = [tgt_img, *ref_imgs]

        # apply the transforms on the images and on the camera intrinsics matrix
        if self.transform is not None:
            imgs, intrinsics = self.transform(imgs, np.copy(intrinsics_orig))
        tgt_img = imgs[0]
        ref_imgs = imgs[1 : self.sequence_length]
        inv_intrinsics = np.linalg.inv(intrinsics)
        sample = {"tgt_img": tgt_img, "ref_imgs": ref_imgs, "intrinsics": intrinsics, "inv_intrinsics": inv_intrinsics}

        if self.load_tgt_depth:
            # load the depth map of the target frame and return it as part of the sample (As is, without any transformation)
            with h5py.File(scene_path / "gt_depth_and_egomotion.h5", "r") as h5f:
                depth_img = to_default_type(h5f["z_depth_map"][target_frame_ind], num_type="float_m")
                sample["depth_img"] = depth_img
        return sample

    # ---------------------------------------------------------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)


# ---------------------------------------------------------------------------------------------------------------------

from pathlib import Path

import h5py

from colon3d.alg.monocular_est_loader import DepthAndEgoMotionLoader
from colon3d.alg.tracks_loader import DetectionsTracker
from colon3d.util.data_util import SceneLoader, get_origin_scene_path
from colon3d.util.torch_util import np_func, to_torch
from colon3d.util.pose_transforms import compose_poses, get_identity_pose, get_pose_delta

scene_path = Path("data_gcp/datasets/ColonNav/Test/Scene_00000")

scene_loader = SceneLoader(scene_path=scene_path, n_frames_lim=0, alg_fov_ratio=0)
detections_tracker = DetectionsTracker(scene_path=scene_path, scene_loader=scene_loader)
depth_estimator = DepthAndEgoMotionLoader(
    scene_path=scene_path,
    scene_loader=scene_loader,
    depth_maps_source="ground_truth",
    egomotions_source="ground_truth",
)
n_frames = scene_loader.n_frames

# initial the cam_pose to be the unit pose
cum_pose = get_identity_pose()
for i in range(n_frames):
    egomotion = depth_estimator.get_egomotions_at_frame(i)
    cum_pose = compose_poses(pose1=cum_pose, pose2=egomotion)
print("cum_pose=\n", cum_pose)
print(cum_pose)


# get the ground truth poses
origin_scene_path = get_origin_scene_path(scene_path)
with h5py.File(origin_scene_path / "gt_3d_data.h5", "r") as h5f:
    gt_cam_poses = h5f["cam_poses"][:]
    gt_egomotions = to_torch(h5f["egomotions"][:])


# find the delta pose between first and last frame

delta_pose = (np_func)(get_pose_delta)(pose1=gt_cam_poses[0], pose2=gt_cam_poses[-1])

print("delta_pose=\n", delta_pose)

err = (np_func)(get_pose_delta)(pose1=cum_pose, pose2=delta_pose)
print("err=\n", err)


# initial the cam_pose to be the unit pose
cum_pose2 = get_identity_pose()
for i in range(n_frames):
    egomotion = gt_egomotions[i]
    cum_pose2 = compose_poses(pose1=cum_pose2, pose2=egomotion)
print("cum_pose2=\n", cum_pose2)
print(cum_pose2)


err2 = (np_func)(get_pose_delta)(pose1=cum_pose2, pose2=delta_pose)
print("err2=\n", err2)

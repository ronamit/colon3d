from pathlib import Path

from colon3d.utils.data_util import SceneLoader
from colon3d.utils.depth_egomotion import DepthAndEgoMotionLoader
from colon3d.utils.tracks_util import DetectionsTracker

scene_path = Path("data/sim_data/TestData21_cases/Scene_00000_0000")

scene_loader = SceneLoader(scene_path=scene_path, n_frames_lim=0, alg_fov_ratio=0)
detections_tracker = DetectionsTracker(scene_path=scene_path, scene_loader=scene_loader)
depth_estimator = DepthAndEgoMotionLoader(
    scene_path=scene_path,
    depth_maps_source="ground_truth",
    egomotions_source="ground_truth",
)
pass
n_frames = scene_loader.get_n_frames()
depth_estimator.get_egomotions_at_frame(0)

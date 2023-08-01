import argparse
import pickle
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from colon3d.util.data_util import SceneLoader
from colon3d.util.depth_egomotion import DepthAndEgoMotionLoader
from colon3d.util.general_util import ArgsHelpFormatter, create_empty_folder, save_plot_and_close, save_rgb_image
from colon3d.util.torch_util import np_func, to_default_type, to_numpy
from colon3d.util.tracks_util import DetectionsTracker
from colon3d.util.transforms_util import get_frame_point_cloud, transform_points_world_to_cam
from colon3d.visuals.create_3d_obj import plot_cam_and_point_cloud
from colon3d.visuals.plots_2d import draw_track_box_on_frame


# --------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(formatter_class=ArgsHelpFormatter)
    parser.add_argument(
        "--scene_path",
        type=str,
        default="data/sim_data/TestData21_cases/Scene_00005_0000",
        help="Path to the scene folder",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="results/plot_example_frame/Scene_00005_0000",
        help="Path to save the results",
    )
    parser.add_argument(
        "--frame_time",
        type=float,
        default=-1,
        help="The time of the frame to plot [sec], if -1 then the frame_index will be used",
    )
    parser.add_argument(
        "--frame_index",
        type=float,
        default=0,
        help="The index of the frame to plot, if frame_time is not -1 then  frame_time will be used instead",
    )
    parser.add_argument(
        "--depth_maps_source",
        type=str,
        default="ground_truth",
        choices=["ground_truth", "loaded_estimates", "online_estimates", "none"],
        help="The source of the depth-map",
    )
    parser.add_argument(
        "--egomotions_source",
        type=str,
        default="ground_truth",
        choices=["ground_truth", "loaded_estimates", "online_estimates", "none"],
        help="The source of the egomotions",
    )
    parser.add_argument(
        "--depth_and_egomotion_model_path",
        type=str,
        default="saved_models/EndoSFM_orig",
        help="path to the saved depth and egomotion model (PoseNet and DepthNet) to be used for online estimation",
    )
    args = parser.parse_args()
    print(f"args={args}")
    scene_path = Path(args.scene_path)

    scene_loader = SceneLoader(
        scene_path=scene_path,
    )
    depth_loader = DepthAndEgoMotionLoader(
        scene_path=scene_path,
        depth_maps_source=args.depth_maps_source,
        egomotions_source=args.egomotions_source,
        depth_and_egomotion_model_path=Path(args.depth_and_egomotion_model_path),
    )
    detections_tracker = DetectionsTracker(scene_path=scene_path, scene_loader=scene_loader)

    save_path = Path(args.save_path)
    create_empty_folder(save_path, save_overwrite=True)
    fps = scene_loader.fps
    frame_idx = args.frame_index if args.frame_time == -1 else int(args.frame_time * fps)
    frame_name = f"Frame{frame_idx:04d}"

    # save the color frame
    rgb_frame = scene_loader.get_frame_at_index(frame_idx, color_type="RGB", frame_type="full")
    file_path = save_path / f"{frame_name}.png"
    save_rgb_image(rgb_frame, file_path)
    print(f"Saved color frame to {file_path}")

    # get the depth map
    z_depth_frame = depth_loader.get_depth_map_at_frame(frame_idx=frame_idx, rgb_frame=rgb_frame)
    z_depth_frame = to_numpy(z_depth_frame)

    plt.figure()
    plt.imshow(z_depth_frame, aspect="equal")
    plt.xlabel("x [pixels]")
    plt.ylabel("y [pixels]")
    save_plot_and_close(save_path / f"{frame_name}_depth_{args.depth_maps_source}.png")

    depth_info = depth_loader.depth_info
    K_of_depth_map = depth_info["K_of_depth_map"]
    fx, fy = K_of_depth_map[0, 0], K_of_depth_map[1, 1]
    frame_height, frame_width = z_depth_frame.shape[:2]
    fov_deg = 2 * np.rad2deg(np.arctan(0.5 * min(frame_width / fx, frame_height / fy)))

    # if ground-truth camera pose is available, use it for the point cloud plot
    if args.depth_maps_source == "ground_truth":
        with h5py.File((scene_path / "gt_3d_data.h5").resolve(), "r") as h5f:
            gt_cam_poses = to_default_type(h5f["cam_poses"][:])
            cam_pose = gt_cam_poses[frame_idx]
            print("Using ground-truth camera pose for the point cloud plot")
            print(f"(x,y,z)={cam_pose[:3]} [mm]\n (qw,qx,qy,qz)={cam_pose[3:]}")

        # load the  ground truth targets info
        targets_info_path = scene_path / "targets_info.pkl"
        if targets_info_path.exists():
            with targets_info_path.open("rb") as file:
                gt_targets_info = pickle.load(file)
            target_index = 0  # we are plotting only one target
            target_p3d_world = gt_targets_info.points3d[target_index]
            target_p3d_cam = np_func(transform_points_world_to_cam)(target_p3d_world, cam_pose).squeeze()
            target_radius = gt_targets_info.radiuses[target_index]
            tracks_in_frame = detections_tracker.get_tracks_in_frame(frame_idx)
            for track_id, cur_track in tracks_in_frame.items():
                rgb_frame = draw_track_box_on_frame(
                    rgb_frame=rgb_frame,
                    track=cur_track,
                    track_id=track_id,
                )
            # save the color frame with the detections
            file_path = save_path / f"{frame_name}_tracks.png"
            save_rgb_image(rgb_frame, file_path)
            print(f"Saved color frame + tracks to {file_path}")

        else:
            target_p3d_world = None
            print("No targets info file found...")

        print("Frame index:", frame_idx)
        # get the point cloud (in the world coordinate system)
        print("World system:")
        points3d = get_frame_point_cloud(z_depth_frame=z_depth_frame, K_of_depth_map=K_of_depth_map, cam_pose=cam_pose)
        plot_cam_and_point_cloud(
            points3d=points3d,
            cam_pose=cam_pose,
            cam_fov_deg=fov_deg,
            target_p3d=target_p3d_world,
            target_radius=target_radius,
            verbose=True,
            save_path=save_path / f"{frame_name}_world_sys_3d_{args.depth_maps_source}.html",
        )
        # get the point cloud (in the camera coordinate system)
        print("Camera system:")
        points3d = get_frame_point_cloud(z_depth_frame=z_depth_frame, K_of_depth_map=K_of_depth_map, cam_pose=None)
        plot_cam_and_point_cloud(
            points3d=points3d,
            cam_pose=None,
            cam_fov_deg=fov_deg,
            target_p3d=target_p3d_cam,
            target_radius=target_radius,
            verbose=True,
            save_path=save_path / f"{frame_name}_cam_sys_3d_{args.depth_maps_source}.html",
        )
    else:
        print("No ground-truth camera pose is available, so the point cloud plot will not be saved")


# --------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
# --------------------------------------------------------------------------------------------------------------------

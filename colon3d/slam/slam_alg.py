import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch

from colon3d.slam.alg_settings import AlgorithmParam
from colon3d.slam.bundle_adjust import run_bundle_adjust
from colon3d.slam.slam_out_analysis import AnalysisLogger
from colon3d.util.data_util import RadialImageCropper, SceneLoader
from colon3d.util.depth_egomotion import DepthAndEgoMotionLoader
from colon3d.util.general_util import convert_sec_to_str, get_time_now_str
from colon3d.util.keypoints_util import KeyPointsLog, get_kp_matchings, get_tracks_keypoints
from colon3d.util.rotations_util import get_identity_quaternion
from colon3d.util.torch_util import get_default_dtype, get_device, to_numpy
from colon3d.util.tracks_util import DetectionsTracker
from colon3d.util.transforms_util import (
    compose_poses,
    transform_points_world_to_cam,
    unproject_image_normalized_coord_to_world,
)
from colon3d.visuals.plots_2d import draw_kp_on_img, draw_matches

torch.set_default_dtype(get_default_dtype())
# ---------------------------------------------------------------------------------------------------------------------


class SlamAlgRunner:
    """Run the SLAM algorithm."""

    def __init__(
        self,
        alg_prm: AlgorithmParam,
        scene_loader: SceneLoader,
        detections_tracker: DetectionsTracker,
        depth_estimator: DepthAndEgoMotionLoader,
        save_path: Path | None = None,
        draw_interval: int = 0,
        verbose_print_interval: int = 0,
    ):
        self.alg_prm = alg_prm
        self.scene_loader = scene_loader
        self.detections_tracker = detections_tracker
        self.depth_estimator = depth_estimator
        self.save_path = save_path
        self.draw_interval = draw_interval
        self.verbose_print_interval = verbose_print_interval

        #  ---- Algorithm hyperparameters  ----
        # ---- ORB feature detector and descriptor (https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html)
        self.kp_detector = cv2.ORB_create(
            nfeatures=alg_prm.max_initial_keypoints,  # maximum number of features (keypoints) to retain
            scaleFactor=1.2,  # Pyramid decimation ratio, greater than 1.
            nlevels=8,  # The number of pyramid levels.
            edgeThreshold=alg_prm.kp_descriptor_patch_size,  # This is size of the border where the features are not detected.
            firstLevel=0,  # The level of pyramid to put source image to.
            WTA_K=2,  # The number of points that produce each element of the oriented BRIEF descriptor.
            scoreType=cv2.ORB_HARRIS_SCORE,  # 	The default HARRIS_SCORE means that Harris algorithm is used to rank features
            patchSize=alg_prm.kp_descriptor_patch_size,  # size of the patch used by the oriented BRIEF descriptor.
            fastThreshold=alg_prm.orb_fast_thresh,  # Threshold on difference between intensity of the central pixel and pixels of a circle around this pixel.
        )
        # Create FLANN Matcher
        self.kp_matcher = cv2.FlannBasedMatcher(
            {
                "algorithm": 6,  # FLANN_INDEX_LSH, suited for ORB
                "table_number": 12,  # 12
                "key_size": 20,  # 20
                "multi_probe_level": 2,
            },
            {"checks": 50},
        )

    # ---------------------------------------------------------------------------------------------------------------------

    def init_algorithm(self, scene_metadata: dict):
        self.device = get_device()

        # The (matched) keypoints (KPs) data is saved in a dataframe
        # note that KPs might get discarded later on, if found to be invalid
        self.alg_view_pix_normalizer = self.scene_loader.alg_view_pix_normalizer
        self.kp_log = KeyPointsLog(self.alg_view_pix_normalizer)

        #  List of the per-step estimates of the 3D locations of each track's KPs in the world system:
        self.online_est_track_world_loc = []
        #  List of the per-step estimates of the 3D locations of each track's KPs in the camera system:
        self.online_est_track_cam_loc = []
        # list of the per-frame estimate of the angle in the camera YZ plane of each track (in radians):
        self.online_est_track_angle = []
        #  List of the per-step estimates oft the 3D locations of the saliency KPs (in the world system):
        self.online_est_salient_kp_world_loc = []
        """
        cam_poses = saves for each frame of the estimated 6DOF *change* of the camera pose from its initial pose (in world system)
        Note: the world system is defined such that its origin is in frame #0 is the camera focal point, and its z axis is the initial camera orientation.
        We represent 6DOF as a vector with 7 parameters (x, y, z, q0, qx, qy, qz)
        where
        * (x,y,z) is the translation vector from the initial (frame #0) position of the camera's focal point (units: mm).
        * (q0, qx, qy, qz) is the unit-quaternion that represents the rotation from the initial (frame #0) direction of the camera's optical (z) axis.
            We use the Hamilton convention for quaternions (q0 is the scalar part).
            We always keep the quaternion normalized (i.e. q0^2 + q1^2 + q2^2 + q3^2 = 1).  also we always to a standard form: one in which the real
        part is non negative.
        """

        # saves for each frame the most up-to-date estimate of the 6DOF camera pose change from the initial pose:
        self.cam_poses = torch.full((1, 7), torch.nan, device=self.device)
        self.cam_poses[0, :] = torch.cat((torch.tensor([0, 0, 0]), get_identity_quaternion()))
        #  saves the identified world 3D point - the currently optimized coordinates  (units: mm):
        self.points_3d = torch.full((0, 3), torch.nan, device=self.device)
        self.p3d_inds_per_frame = []  # for each frame - a list of the 3D points indexes seen in that frame
        self.tracks_p3d_inds = {}  # maps a track_id to its associated 3D world points index
        self.n_world_points = 0  # number of 3D world points identified so far
        self.online_logger = AnalysisLogger(self.alg_prm)
        # saves data about the previous frame:
        self.prev_rgb_frame = None
        self.descriptors_A = None
        self.salient_KPs_A = None
        self.track_KPs_A = None
        self.tracks_in_frameA = None
        self.scene_metadata = scene_metadata
        print("Algorithm parameters: \n", self.alg_prm)

    # ---------------------------------------------------------------------------------------------------------------------

    def run(
        self,
    ) -> dict:
        frames_generator = self.scene_loader.frames_generator(frame_type="alg_input")
        alg_view_pix_normalizer = self.scene_loader.alg_view_pix_normalizer
        alg_view_cropper = self.scene_loader.alg_view_cropper
        scene_metadata = self.scene_loader.metadata
        n_frames = self.scene_loader.n_frames
        fps = self.scene_loader.fps

        # initialize the algorithm
        self.init_algorithm(scene_metadata)

        # ---- Run algorithm (on-line)  ----
        print("-" * 50 + f"\nRunning SLAM algorithm. Time now: {get_time_now_str()}...\n" + "-" * 50)
        print(f"{self.alg_prm}\n" + "-" * 50)
        print(f"Processing {n_frames} frames...")
        runtime_start = time.time()
        for i_frame in range(n_frames):
            print("-" * 50 + f"\ni_frame: {i_frame}/{n_frames-1}")
            # Get the RGB frame:
            cur_rgb_frame = frames_generator.__next__()
            # Get the targets tracks in the current frame:
            curr_tracks = self.detections_tracker.get_tracks_in_frame(i_frame)
            # Get the depth and ego-motion estimation for the current frame:
            self.depth_estimator.process_new_frame(
                i_frame,
                cur_rgb_frame=cur_rgb_frame,
                prev_rgb_frame=self.prev_rgb_frame,
            )

            self.run_on_new_frame(
                cur_rgb_frame=cur_rgb_frame,
                i_frame=i_frame,
                curr_tracks=curr_tracks,
                alg_view_cropper=alg_view_cropper,
                depth_estimator=self.depth_estimator,
                fps=fps,
                draw_interval=self.draw_interval,
                verbose_print_interval=self.verbose_print_interval,
                save_path=self.save_path,
            )
        print("-" * 50, f"\nSLAM algorithm run finished. Time now: {get_time_now_str()}")
        print("Elapsed time: ", convert_sec_to_str(time.time() - runtime_start))
        # ---- Save outputs ----
        slam_out = {
            "alg_prm": self.alg_prm,
            "cam_poses": self.cam_poses,
            "points_3d": self.points_3d,
            "kp_log": self.kp_log,
            "tracks_p3d_inds": self.tracks_p3d_inds,
            "p3d_inds_per_frame": self.p3d_inds_per_frame,
            "scene_loader": self.scene_loader,
            "detections_tracker": self.detections_tracker,
            "alg_view_pix_normalizer": alg_view_pix_normalizer,
            "depth_estimator": self.depth_estimator,
            "online_est_track_world_loc": self.online_est_track_world_loc,
            "online_est_track_cam_loc": self.online_est_track_cam_loc,
            "online_est_salient_kp_world_loc": self.online_est_salient_kp_world_loc,
            "online_est_track_angle": self.online_est_track_angle,
            "online_logger": self.online_logger,
        }
        return slam_out

    # ---------------------------------------------------------------------------------------------------------------------

    def run_on_new_frame(
        self,
        cur_rgb_frame: np.array,
        i_frame: int,
        curr_tracks: dict,
        alg_view_cropper: RadialImageCropper | None,
        depth_estimator: DepthAndEgoMotionLoader,
        fps: float,
        draw_interval: int,
        verbose_print_interval: int,
        save_path: Path | None = None,
    ):
        """Run the algorithm on a new frame.
        1. Detect salient keypoints in the new frame
        2. Match the salient keypoints to the keypoints in the previous frame
        3. Estimate the 3D location of the matched keypoints
        5. Run bundle-adjustment to update the 3D points and the camera poses estimates.
        Args:
            curr_frame: the current RGB frame
            i_frame: the index of the current frame
            curr_tracks: the tracks in the current frame (bounding boxes)
            curr_egomotion_est: the initially estimated egomotion of the camera between the current frame and the previous frame
            cam_undistorter: the camera undistorter
            alg_view_cropper: the view cropper
            depth_estimator: the depth estimator
            fps: the FPS of the video [Hz]
            draw_interval: the interval (in frames) in which to draw the results (0 for no drawing)
            verbose_print_interval: the interval (in frames) in which to print the results (0 for no printing)
            save_path: the path to save the results and plots

        Note:
            we call the current frame "frame B" and the previous frame "frame A"
        """
        alg_prm = self.alg_prm

        # Keep  the (frame_idx, x, y) of keypoints that are associated with a newly discovered 3D point (in the current frame)
        new_world_point_kp_ids = []

        # Initialize the set of 3D points seen  with the tracks in the current frame
        self.p3d_inds_per_frame.append(set())
        self.online_est_track_world_loc.append({})
        self.online_est_track_cam_loc.append({})
        self.online_est_track_angle.append({})

        #  Guess the cam pose for current frame, based on the last estimate and the egomotion estimation
        # note: for i_frame=0, the cam pose is not optimized, since it set as the frame of reference
        if i_frame > 0:
            # Get the previous est. cam pose (w.r.t. the world system)
            prev_cam_pose = self.cam_poses[i_frame - 1, :]
            # Get the estimated cam pose change from the previous frame to the current frame (egomotion) w.r.t. the previous frame
            #  i.e., Egomotion_{i} = inv(Pose_{i-1}^{world}) @ Pose_i^{world}
            curr_egomotion_est = depth_estimator.get_egomotions_at_frame(
                curr_frame_idx=i_frame,
            )
            # Get the estimated cam pose for the current frame (w.r.t. the world system)
            #  Pose_i^{world} = Pose_{i-1}^{world} @ Egomotion_{i}
            cur_guess_cam_pose = compose_poses(
                pose1=curr_egomotion_est.unsqueeze(0),
                pose2=prev_cam_pose.unsqueeze(0),
            )
            # concat the current guess to the cam_poses tensor
            self.cam_poses = torch.cat((self.cam_poses, cur_guess_cam_pose), dim=0)

        if self.alg_prm.use_bundle_adjustment:
            # in this case - we need to find KPs matches for the bundle adjustment
            # find salient keypoints with ORB
            salient_KPs_B = self.kp_detector.detect(cur_rgb_frame, None)
            # compute the descriptors with ORB
            salient_KPs_B, descriptors_B = self.kp_detector.compute(cur_rgb_frame, salient_KPs_B)
        else:
            # in this case - we don't need salient KPs, since we don't run bundle adjustment (we only need the tracks)
            salient_KPs_B = []
            descriptors_B = []

        # Find the track-keypoints according to the detection bounding box
        tracks_in_frameB = curr_tracks
        track_KPs_B = get_tracks_keypoints(tracks_in_frameB)
        if draw_interval and i_frame % draw_interval == 0:
            draw_kp_on_img(
                img=cur_rgb_frame,
                salient_KPs=salient_KPs_B,
                track_KPs=track_KPs_B,
                curr_tracks=tracks_in_frameB,
                alg_view_cropper=alg_view_cropper,
                save_path=save_path,
                i_frame=i_frame,
            )
        # ----Starting from second frame, run matching of keypoints with pervious frame
        if i_frame > 0:
            matched_A_kps, matched_B_kps = get_kp_matchings(
                keypoints_A=self.salient_KPs_A,
                keypoints_B=salient_KPs_B,
                descriptors_A=self.descriptors_A,
                descriptors_B=descriptors_B,
                kp_matcher=self.kp_matcher,
                alg_prm=self.alg_prm,
            )
            # -----Draw the matches
            if draw_interval and i_frame % draw_interval == 0:
                # draw our inliers (if RANSAC was done) or all good matching keypoints
                draw_matches(
                    img_A=self.prev_rgb_frame,
                    img_B=cur_rgb_frame,
                    matched_A_kps=matched_A_kps,
                    matched_B_kps=matched_B_kps,
                    track_KPs_A=self.track_KPs_A,
                    track_KPs_B=track_KPs_B,
                    i_frameA=i_frame - 1,
                    i_frameB=i_frame,
                    tracks_in_frameA=self.tracks_in_frameA,
                    tracks_in_frameB=tracks_in_frameB,
                    alg_view_cropper=alg_view_cropper,
                    save_path=save_path,
                )
            # -----Update the inputs to the bundle-adjustment using the salient KPs
            n_matches = len(matched_A_kps)
            for i_match in range(n_matches):
                # Get KP A (from previous frame) and KP B (from current frame
                kpA_xy = matched_A_kps[i_match]
                kpB_xy = matched_B_kps[i_match]
                if not self.kp_log.is_kp_coord_valid(kpB_xy) or not self.kp_log.is_kp_coord_valid(kpA_xy):
                    # in case one of the KPs is too close to the image border, and its undistorted coordinates are invalid
                    continue
                kpA_frame_idx = i_frame - 1
                kpB_frame_idx = i_frame
                # check if KP A (which was detected in the previous frame) already has an associated 3d world point
                kp_A_id = (kpA_frame_idx, kpA_xy[0], kpA_xy[1])
                if self.kp_log.get_kp_p3d_idx(kp_A_id) is not None:
                    # In this case, KP B will be associated wit the same 3D point (since there is a match)
                    p3d_id = self.kp_log.get_kp_p3d_idx(kp_A_id)
                    self.kp_log.add_kp(frame_idx=kpB_frame_idx, pix_coord=kpB_xy, kp_type=-1, p3d_id=p3d_id)
                else:
                    # otherwise, associate both KPs (A and B) with as a newly identified 3d world point
                    # (so that the bundle adjustment can use the two view points of the same 3D point)
                    p3d_id = deepcopy(self.n_world_points)  # index of the new 3d point
                    self.n_world_points += 1
                    self.kp_log.add_kp(frame_idx=kpA_frame_idx, pix_coord=kpA_xy, kp_type=-1, p3d_id=p3d_id)
                    self.kp_log.add_kp(frame_idx=kpB_frame_idx, pix_coord=kpB_xy, kp_type=-1, p3d_id=p3d_id)

                    # We use kp A to initialize the guess of the coordinates of the new 3D point  since its cam pose was optimized already in the previous bundle adjustment, and thus is more accurate:
                    new_world_point_kp_ids.append((kpA_frame_idx, kpA_xy[0], kpA_xy[1]))
                self.p3d_inds_per_frame[-1].add(p3d_id)

        # ----- Update the inputs to the bundle-adjustment using the tracks (polyps) KPs
        for track_id, kpB_xy in track_KPs_B.items():
            # in case we have not seen this track before, we need to create a new 3d points for it
            register_new_p3d = track_id not in self.tracks_p3d_inds
            if not self.kp_log.is_kp_coord_valid(kpB_xy):
                # in case one of the KPs is too close to the image border, and its undistorted coordinates are invalid
                continue
            kpB_frame_idx = i_frame
            if register_new_p3d:
                # Register a new world 3d point
                p3d_id = deepcopy(self.n_world_points)  # index of the new 3d point
                self.n_world_points += 1
                # KP B will be used to guess the 3d location of the new world point
                new_world_point_kp_ids.append((kpB_frame_idx, kpB_xy[0], kpB_xy[1]))
                self.tracks_p3d_inds[track_id] = p3d_id
            else:
                # in this case we already have a 3D point for this track, so we use the same p3d index
                p3d_id = self.tracks_p3d_inds[track_id]

            self.kp_log.add_kp(frame_idx=kpB_frame_idx, pix_coord=kpB_xy, kp_type=track_id, p3d_id=p3d_id)
            self.p3d_inds_per_frame[-1].add(p3d_id)

        # Use the depth estimation to guess the 3d location of the newly found world points
        n_new_kps = len(new_world_point_kp_ids)
        if n_new_kps > 0:
            # get the frame indexes of the new KPs
            frame_inds_of_new = [kp_id[0] for kp_id in new_world_point_kp_ids]
            # get the camera poses corresponding to the new KPs
            cam_poses_of_new = self.cam_poses[frame_inds_of_new]
            # get the normalized pixel coordinates of the new KPs
            kp_nrm_of_new = [self.kp_log.get_kp_norm_coord(kp_id) for kp_id in new_world_point_kp_ids]
            # convert to torch
            kp_nrm_of_new = torch.tensor(np.array(kp_nrm_of_new), device=self.device)
            # estimate the 3d points of the new KPs (using the depth estimator)
            # note: the depth estimator already process the frame index until the current frame, and saved the results in a buffer
            z_depths_of_new = depth_estimator.get_z_depth_at_pixels(
                queried_pix_nrm=kp_nrm_of_new,
                frame_indexes=frame_inds_of_new,
            )
            new_p3d_est = unproject_image_normalized_coord_to_world(
                points_nrm=kp_nrm_of_new,
                z_depths=z_depths_of_new,
                cam_poses=cam_poses_of_new,
            )
            self.points_3d = torch.cat((self.points_3d, new_p3d_est), dim=0)

        # save the current camera pose guess
        self.online_logger.save_cam_pose_guess(self.cam_poses[i_frame, :])

        # ---- Run bundle-adjustment:
        if i_frame > 0 and self.alg_prm.use_bundle_adjustment and i_frame % alg_prm.optimize_each_n_frames == 0:
            verbose = 2 if (verbose_print_interval and i_frame % verbose_print_interval == 0) else 0
            frames_inds_to_opt = list(range(max(0, i_frame - alg_prm.n_last_frames_to_opt + 1), i_frame + 1))

            # Loop that runs the bundle adjustment until no more KPs are discarded
            n_invalid_kps = -1
            i_repeat = 0
            while n_invalid_kps != 0:
                print(f"Running bundle adjustment. Repeat #{i_repeat}")
                self.cam_poses, self.points_3d, self.kp_log, n_invalid_kps = run_bundle_adjust(
                    cam_poses=self.cam_poses,
                    points_3d=self.points_3d,
                    kp_log=self.kp_log,
                    p3d_inds_per_frame=self.p3d_inds_per_frame,
                    frames_inds_to_opt=frames_inds_to_opt,
                    alg_prm=alg_prm,
                    fps=fps,
                    scene_metadata=self.scene_metadata,
                    verbose=verbose,
                )
                i_repeat += 1

        # ----  Save online-estimates for the current frame
        # save the currently estimated 3D KP of the tracks that have been seen until now

        for track_id, track_kps_p3d_ind in self.tracks_p3d_inds.items():
            track_world_p3d = self.points_3d[track_kps_p3d_ind]
            track_cam_p3d = transform_points_world_to_cam(
                points_3d_world=track_world_p3d,
                cam_poses=self.cam_poses[i_frame],
            )
            # in the world system
            self.online_est_track_world_loc[i_frame][track_id] = to_numpy(track_world_p3d)
            self.online_est_track_cam_loc[i_frame][track_id] = to_numpy(track_cam_p3d)
            # estimate the direction of the track in he camera frame XT plane (for navigation arrow)
            self.online_est_track_angle[i_frame][track_id] = self.estimate_track_angle_in_frame(
                track_id=track_id,
                i_frame=i_frame,
                cur_track_KPs=track_KPs_B,
            )

        # save the currently estimated 3D world system location of the salient KPs
        self.online_est_salient_kp_world_loc.append(deepcopy(self.points_3d))

        # Save the currently estimated camera pose.
        self.online_logger.save_cam_pose_estimate(self.cam_poses[i_frame, :])

        # ----- update variables for the next frame
        self.salient_KPs_A = salient_KPs_B
        self.descriptors_A = descriptors_B
        self.prev_rgb_frame = cur_rgb_frame
        self.track_KPs_A = track_KPs_B
        self.tracks_in_frameA = tracks_in_frameB

    # ---------------------------------------------------------------------------------------------------------------------

    def estimate_track_angle_in_frame(self, track_id: int, i_frame: int, cur_track_KPs: dict):
        """estimate the direction of the track in he camera frame XY plane (for navigation arrow) [rad]"""

        if track_id in cur_track_KPs:
            # in this case - we see the track in the current frame, so we use its pixel coordinates
            track_loc_pix = cur_track_KPs[track_id]
            # transform to normalized image coordinates:
            track_loc_nrm, _ = self.alg_view_pix_normalizer.get_normalized_coord(track_loc_pix)
            # transform from pixel coordinates to camera system coordinates:
            angle_rad = np.arctan2(track_loc_nrm[1], track_loc_nrm[0])

        elif self.alg_prm.use_trivial_nav_aid:
            # the track is out-of-view.
            # use the naive navigation aid - just using the same angle as in previous frame
            angle_rad = self.online_est_track_angle[i_frame - 1][track_id]

        else:
            # the track is out-of-view.
            # use the estimated location of the track in the current frame
            track_loc_cam = self.online_est_track_cam_loc[i_frame][track_id]
            # get the angle of the 2D arrow vector which is the projection of the vector from the camera origin to the camera system XY plane
            angle_rad = np.arctan2(track_loc_cam[1], track_loc_cam[0])

        return angle_rad

    #


# ---------------------------------------------------------------------------------------------------------------------

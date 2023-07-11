import time
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch

from colon3d.slam.alg_settings import AlgorithmParam
from colon3d.slam.bundle_adjust import run_bundle_adjust
from colon3d.slam.slam_out_analysis import AnalysisLogger, SlamOutput
from colon3d.utils.camera_util import FishEyeUndistorter
from colon3d.utils.data_util import RadialImageCropper, SceneLoader
from colon3d.utils.depth_egomotion import DepthAndEgoMotionLoader
from colon3d.utils.general_util import convert_sec_to_str, get_time_now_str
from colon3d.utils.keypoints_util import get_kp_matchings, get_tracks_keypoints
from colon3d.utils.rotations_util import get_identity_quaternion
from colon3d.utils.torch_util import get_default_dtype, get_device
from colon3d.utils.tracks_util import DetectionsTracker
from colon3d.utils.transforms_util import compose_poses
from colon3d.visuals.plots_2d import draw_kp_on_img, draw_matches

torch.set_default_dtype(get_default_dtype())
# ---------------------------------------------------------------------------------------------------------------------


class SlamAlgRunner:
    """Run the SLAM algorithm."""

    def __init__(
        self,
        alg_prm: AlgorithmParam,
    ):
        #  ---- Algorithm hyperparameters  ----
        self.alg_prm = alg_prm
        # ---- ORB feature detector and descriptor (https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html)
        self.kp_detector = cv2.ORB_create(
            nfeatures=300,  # maximum number of features (keypoints) to retain
            scaleFactor=1.2,  # Pyramid decimation ratio, greater than 1.
            nlevels=8,  # The number of pyramid levels.
            edgeThreshold=31,  # This is size of the border where the features are not detected.
            firstLevel=0,  # The level of pyramid to put source image to.
            WTA_K=2,  # The number of points that produce each element of the oriented BRIEF descriptor.
            scoreType=cv2.ORB_HARRIS_SCORE,  # 	The default HARRIS_SCORE means that Harris algorithm is used to rank features
            patchSize=31,  # size of the patch used by the oriented BRIEF descriptor.
            fastThreshold=20,  # Threshold on difference between intensity of the central pixel and pixels of a circle around this pixel.
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

    def init_algorithm(self):
        self.device = get_device()
        self.kp_px_all = []  # List of keypoints, each element is the (x,y) pixel coordinates of a keypoint in the image
        #  the (x,y) normalized coordinates of a keypoint in some image:
        self.kp_nrm_all = torch.full((0, 2), torch.nan, device=self.device)
        # List of identifier numbers for each keypoint (-1 indicates a salient keypoint, >=0 indicates the track id of the keypoint):
        self.kp_id_all = []
        self.SALIENT_KP_ID = -1
        self.kp_frame_idx_all = []  #  List of the keypoint's frame index
        self.kp_p3d_idx_all = []  #  List of the keypoint's associated 3D point index
        #  List of the per-step estimates oft the 3D locations of each track's KPs (in the world system):
        self.online_est_track_world_loc = []
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
        initial_position = torch.tensor([0, 0, 0])  # zero translation
        initial_rotation = get_identity_quaternion()  # the identity quaternion. It represents no rotation.
        # saves for each frame the most up-to-date estimate of the 6DOF camera pose change from the initial pose:
        self.cam_poses = torch.full((1, 7), torch.nan, device=self.device)
        self.cam_poses[0, :] = torch.cat((initial_position, initial_rotation))
        #  saves the identified world 3D point - the currently optimized coordinates  (units: mm):
        self.points_3d = torch.full((0, 3), torch.nan, device=self.device)
        self.p3d_inds_in_frame = []  # for each frame - a list of the 3D points indexes seen in that frame
        self.map_kp_to_p3d_idx = {}  # maps a KP (i_frame, i_cord, j_cord) to a an index of a 3D world point
        self.tracks_p3d_inds = {}  # maps a track_id to its associated 3D world points index
        self.n_world_points = 0  # number of 3D world points identified so far
        self.online_logger = AnalysisLogger(self.alg_prm)
        # saves data about the previous frame:
        self.prev_rgb_frame = None
        self.descriptors_A = None
        self.salient_KPs_A = None
        self.track_KPs_A = None
        self.tracks_in_frameA = None

        print("Algorithm parameters: \n", self.alg_prm)

    # ---------------------------------------------------------------------------------------------------------------------

    def run(
        self,
        scene_loader: SceneLoader,
        detections_tracker: DetectionsTracker,
        depth_estimator: DepthAndEgoMotionLoader,
        use_bundle_adjustment: bool = True,
        save_path: Path | None = None,
        draw_interval: int = 0,
        verbose_print_interval: int = 0,
    ) -> SlamOutput:
        frames_generator = scene_loader.frames_generator(frame_type="alg_input")
        cam_undistorter = scene_loader.alg_cam_undistorter
        alg_view_cropper = scene_loader.alg_view_cropper
        n_frames = scene_loader.n_frames
        fps = scene_loader.fps

        # initialize the algorithm
        self.init_algorithm()

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
            curr_tracks = detections_tracker.get_tracks_in_frame(i_frame)
            # Get the depth and ego-motion estimation for the current frame:
            depth_estimator.process_new_frame(i_frame, cur_rgb_frame=cur_rgb_frame, prev_rgb_frame=self.prev_rgb_frame)

            self.run_on_new_frame(
                cur_rgb_frame=cur_rgb_frame,
                i_frame=i_frame,
                curr_tracks=curr_tracks,
                cam_undistorter=cam_undistorter,
                alg_view_cropper=alg_view_cropper,
                depth_estimator=depth_estimator,
                fps=fps,
                use_bundle_adjustment=use_bundle_adjustment,
                draw_interval=draw_interval,
                verbose_print_interval=verbose_print_interval,
                save_path=save_path,
            )
        print("-" * 50, f"\nSLAM algorithm run finished. Time now: {get_time_now_str()}")
        print("Elapsed time: ", convert_sec_to_str(time.time() - runtime_start))
        # ---- Save outputs ----
        slam_out = SlamOutput(
            alg_prm=self.alg_prm,
            cam_poses=self.cam_poses,
            points_3d=self.points_3d,
            kp_frame_idx_all=self.kp_frame_idx_all,
            kp_px_all=self.kp_px_all,
            kp_nrm_all=self.kp_nrm_all,
            kp_p3d_idx_all=self.kp_p3d_idx_all,
            tracks_p3d_inds=self.tracks_p3d_inds,
            kp_id_all=self.kp_id_all,
            p3d_inds_in_frame=self.p3d_inds_in_frame,
            map_kp_to_p3d_idx=self.map_kp_to_p3d_idx,
            scene_loader=scene_loader,
            detections_tracker=detections_tracker,
            cam_undistorter=cam_undistorter,
            depth_estimator=depth_estimator,
            online_est_track_world_loc=self.online_est_track_world_loc,
            online_est_salient_kp_world_loc=self.online_est_salient_kp_world_loc,
            online_logger=self.online_logger,
        )
        return slam_out

    # ---------------------------------------------------------------------------------------------------------------------

    def run_on_new_frame(
        self,
        cur_rgb_frame: np.array,
        i_frame: int,
        curr_tracks: dict,
        cam_undistorter: FishEyeUndistorter,
        alg_view_cropper: RadialImageCropper | None,
        depth_estimator: DepthAndEgoMotionLoader,
        fps: float,
        use_bundle_adjustment: bool,
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
        # Keep track of the keypoints that are associated with a newly discovered 3D point
        kp_inds_of_new = []
        # Initialize the 3D points associated with the tracks in the current frame
        self.p3d_inds_in_frame.append(set())
        self.online_est_track_world_loc.append({})

        #  guess of cam pose for current frame, based on the last estimate and the egomotion estimation
        # note: for i_frame=0, the cam pose is not optimized, since it set as the frame of reference
        if i_frame > 0:
            # get the estimated 6DOF cam pose change from the previous frame (egomotion)
            prev_cam_pose = self.cam_poses[i_frame - 1, :]
            curr_egomotion_est = depth_estimator.get_egomotions_at_frame(
                curr_frame_idx=i_frame,
            )
            cur_guess_cam_pose = compose_poses(
                pose1=prev_cam_pose.unsqueeze(0),
                pose2=curr_egomotion_est.unsqueeze(0),
            )
            self.cam_poses = torch.cat((self.cam_poses, cur_guess_cam_pose), dim=0)  # extend the cam_poses tensor

        if use_bundle_adjustment:
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
                self.salient_KPs_A,
                salient_KPs_B,
                self.descriptors_A,
                descriptors_B,
                self.kp_matcher,
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
                kpB_nrm, is_validB = cam_undistorter.undistort_point(kpB_xy)
                kpA_nrm, is_validA = cam_undistorter.undistort_point(kpA_xy)
                if not is_validB or not is_validA:
                    # in case one of the KPs is too close to the image border, and its undistorted coordinates are invalid
                    continue
                kpA_nrm = torch.as_tensor(kpA_nrm, device=self.device).unsqueeze(0)
                kpB_nrm = torch.as_tensor(kpB_nrm, device=self.device).unsqueeze(0)
                kpA_frame_idx = i_frame - 1
                kpB_frame_idx = i_frame
                kp_A_id = (kpA_frame_idx, kpA_xy[0], kpA_xy[1])
                kp_B_id = (kpB_frame_idx, kpB_xy[0], kpB_xy[1])
                # check if KP A (which was detected in the previous frame) already has an associated 3d world point
                if kp_A_id in self.map_kp_to_p3d_idx:
                    # In this case, KP B will be associated wit the same 3D point (since there is a match)
                    p3d_id = self.map_kp_to_p3d_idx[kp_A_id]
                    self.kp_px_all.append(kpB_xy)
                    self.kp_nrm_all = torch.cat((self.kp_nrm_all, kpB_nrm), dim=0)
                    self.kp_id_all.append(self.SALIENT_KP_ID)
                    self.kp_p3d_idx_all.append(p3d_id)
                    self.kp_frame_idx_all.append(kpB_frame_idx)
                else:
                    # otherwise, associate both KPs (A and B) with as a newly identified 3d world point
                    # (so that the bundle adjustment can use the two view points of the same 3D point)
                    p3d_id = deepcopy(self.n_world_points)  # index of the new 3d point
                    self.n_world_points += 1
                    self.kp_px_all += [kpA_xy, kpB_xy]
                    self.kp_nrm_all = torch.cat((self.kp_nrm_all, kpA_nrm, kpB_nrm), dim=0)
                    self.idx_kp_A = len(self.kp_px_all) - 2
                    self.kp_id_all += [self.SALIENT_KP_ID, self.SALIENT_KP_ID]
                    self.kp_p3d_idx_all += [p3d_id, p3d_id]
                    self.kp_frame_idx_all += [kpA_frame_idx, kpB_frame_idx]
                    # the 3d point will be guessed using KP A later on (we use A since its cam pose was optimized already in the previous bundle adjustment):
                    kp_inds_of_new.append(self.idx_kp_A)
                self.map_kp_to_p3d_idx[kp_A_id] = p3d_id
                self.map_kp_to_p3d_idx[kp_B_id] = p3d_id
                self.p3d_inds_in_frame[-1].add(p3d_id)

        # ----- Update the inputs to the bundle-adjustment using the tracks(polyps) KPs
        for track_id, kpB_xy in track_KPs_B.items():
            # in case we have not seen this track before, we need to create a new 3d points for it
            register_new_p3d = track_id not in self.tracks_p3d_inds
            kpB_nrm, is_validB = cam_undistorter.undistort_point(kpB_xy)
            if not is_validB:
                # in case one of the KPs is too close to the image border, and its undistorted coordinates are invalid
                continue
            kpB_nrm = torch.as_tensor(kpB_nrm, device=self.device).unsqueeze(0)
            kpB_frame_idx = i_frame
            self.kp_frame_idx_all.append(kpB_frame_idx)
            self.kp_px_all.append(kpB_xy)
            self.kp_nrm_all = torch.cat((self.kp_nrm_all, kpB_nrm), dim=0)
            idx_kp_B = len(self.kp_px_all) - 1
            if register_new_p3d:
                # Register a new world 3d point
                p3d_id = deepcopy(self.n_world_points)  # index of the new 3d point
                self.n_world_points += 1
                # KP B will be used to guess the 3d location of the new world point
                kp_inds_of_new.append(idx_kp_B)
                self.tracks_p3d_inds[track_id] = p3d_id
            else:
                # in this case we already have a 3D point for this track, so we use the same p3d index
                p3d_id = self.tracks_p3d_inds[track_id]
            self.kp_id_all.append(track_id)
            self.kp_p3d_idx_all.append(p3d_id)
            self.p3d_inds_in_frame[-1].add(p3d_id)

        # Use the depth estimation to guess the 3d location of the newly found world points
        n_new_kps = len(kp_inds_of_new)
        if n_new_kps > 0:
            frame_inds_of_new = [self.kp_frame_idx_all[i_kp] for i_kp in kp_inds_of_new]
            cam_poses_of_new = self.cam_poses[frame_inds_of_new]
            kp_nrm_of_new = torch.stack([self.kp_nrm_all[i_kp] for i_kp in kp_inds_of_new], dim=0)
            new_p3d_est = depth_estimator.estimate_3d_points(
                cam_poses=cam_poses_of_new,
                queried_points_nrm=kp_nrm_of_new,
                frame_indexes=frame_inds_of_new,
            )
            self.points_3d = torch.cat((self.points_3d, new_p3d_est), dim=0)

        # save the current camera pose guess
        self.online_logger.save_cam_pose_guess(self.cam_poses[i_frame, :])

        # ---- Run bundle-adjustment:
        if i_frame > 0 and use_bundle_adjustment and i_frame % alg_prm.optimize_each_n_frames == 0:
            verbose = 2 if (verbose_print_interval and i_frame % verbose_print_interval == 0) else 0
            frames_inds_to_opt = list(range(max(0, i_frame - alg_prm.n_last_frames_to_opt + 1), i_frame + 1))
            self.cam_poses, self.points_3d = run_bundle_adjust(
                cam_poses=self.cam_poses,
                points_3d=self.points_3d,
                kp_frame_idx_all=self.kp_frame_idx_all,
                kp_p3d_idx_all=self.kp_p3d_idx_all,
                kp_nrm_all=self.kp_nrm_all,
                kp_id_all=self.kp_id_all,
                p3d_inds_in_frame=self.p3d_inds_in_frame,
                frames_inds_to_opt=frames_inds_to_opt,
                alg_prm=alg_prm,
                fps=fps,
                SALIENT_KP_ID=self.SALIENT_KP_ID,
                verbose=verbose,
            )

        # ----  Save online-estimates for the current frame
        # save the currently estimated 3D KP of the tracks that have been seen until now
        for track_id, track_kps_p3d_ind in self.tracks_p3d_inds.items():
            self.online_est_track_world_loc[i_frame][track_id] = self.points_3d[track_kps_p3d_ind]

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

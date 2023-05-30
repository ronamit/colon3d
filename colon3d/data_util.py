from pathlib import Path

import cv2
import numpy as np
import yaml

from colon3d.camera_util import CamInfo, FishEyeUndistorter


# --------------------------------------------------------------------------------------------------------------------
class FramesLoader:
    # --------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        sequence_path: str,
        alg_fov_ratio=1.0,
        n_frames_lim=0,
        fps=None,
    ):
        """Initialize the video loader.

        Args:
            sequence_path: path to the example folder
            alg_fov_ratio: The FOV ratio (in the range [0,1]) used for the SLAM algorithm, out of the original FOV, the rest is hidden and only used for validation
            n_frames_lim: limit the number of frames to load
            fps: if not None, the video will be resampled to this fps
        """
        assert 0 <= alg_fov_ratio <= 1, "alg_fov_ratio must be in the range [0,1]"
        # ---- Load video and metadata:
        self.sequence_path = Path(sequence_path).expanduser()

        if (self.sequence_path / "RGB_Frames").is_dir():
            self.rgb_frames_path = self.sequence_path / "RGB_Frames"
            self.image_files_paths = sorted(
                Path(self.sequence_path / "RGB_Frames").glob("*.png"),
                key=lambda path: int(path.stem),
            )
            self.rgb_source = "Images"
        else:
            self.video_path = self.sequence_path / "Video.mp4"
            assert self.video_path.is_file(), f"Video file not found at {self.video_path}"
            self.rgb_source = "Video"

        self.alg_fov_ratio = alg_fov_ratio
        self.n_frames_lim = n_frames_lim
        metadata_path = self.sequence_path / "meta_data.yaml"
        print(f"Loading meta-data from {metadata_path}")
        with (self.sequence_path / "meta_data.yaml").open() as file:
            metadata = yaml.load(file, Loader=yaml.FullLoader)
        if fps is None:
            fps = metadata["fps"]
        self.fps = fps
        self.frame_interval = 1 / fps  # in seconds
        print(f"Frames per second: {fps}")
        distort_pram = metadata["distort_pram"]
        distort_pram = np.zeros(4) if distort_pram is None else np.array(distort_pram)
        # get the camera info of the original loaded video:
        self.orig_cam_info = CamInfo(
            frame_width=metadata["frame_width"],
            frame_height=metadata["frame_height"],
            cx=metadata["cx"],  # optical center x-coordinate in pixels,
            cy=metadata["cy"],  # optical center y-coordinate in pixels
            fx=metadata["fx"],  # focal length in pixels in the x direction
            fy=metadata["fy"],  # focal length in pixels in the y direction,
            distort_pram=distort_pram,
            fps=fps,
            min_vis_z_mm=metadata["min_vis_z_mm"],
        )
        self.alg_view_cropper = RadialImageCropper(self.orig_cam_info, alg_fov_ratio)
        # get the camera info of the alg-viewed video:
        # note:  # since the distort params depend only on the radial distance from optical center, then they are not affected by the crop by a radial crop around the center
        # the fx and fy value doesn't change, since  the focal length divided by sensor size is a physical property of the camera, and not a function of the image size, same for the distortion parameters
        self.alg_cam_info = CamInfo(
            frame_width=self.alg_view_cropper.width_new,
            frame_height=self.alg_view_cropper.height_new,
            cx=self.alg_view_cropper.cx_new,
            cy=self.alg_view_cropper.cy_new,
            fx=metadata["fx"],
            fy=metadata["fy"],
            distort_pram=distort_pram,
            fps=fps,
            min_vis_z_mm=metadata["min_vis_z_mm"],
        )
        self.alg_fov_deg = 2 * np.rad2deg(
            np.arctan(self.alg_view_cropper.view_radius / max(self.alg_cam_info.fx, self.alg_cam_info.fy)),
        )
        self.alg_cam_undistorter = FishEyeUndistorter(self.alg_cam_info)
        self.orig_cam_undistorter = FishEyeUndistorter(self.orig_cam_info)
        self.metadata = metadata
        n_frames = 0
        for _ in self.frames_generator():
            n_frames += 1
        self.n_frames = n_frames
        if n_frames_lim == 0:
            print(f"Using all {n_frames} frames of the video...")
        else:
            print(f"Using only the first {n_frames} frames of the video...")

    # --------------------------------------------------------------------------------------------------------------------

    def frames_generator(self, color_type="bgr", frame_type="alg_input"):
        if self.rgb_source == "Video":
            video_path = Path(self.sequence_path) / "Video.mp4"
            try:
                vidcap = cv2.VideoCapture(str(video_path))
                i_frame = 0
                while vidcap.isOpened():
                    if self.n_frames_lim != 0 and i_frame >= self.n_frames_lim:
                        break
                    frame_exists, curr_frame = vidcap.read()
                    if not frame_exists:
                        break
                    frame = self.adjust_frame(curr_frame, frame_type, color_type)
                    yield frame
                    i_frame += 1
            finally:
                vidcap.release()
        elif self.rgb_source == "Images":
            # sort the image files by the number in their name:
            for i_frame, image_file in enumerate(self.image_files_paths):
                if self.n_frames_lim != 0 and i_frame >= self.n_frames_lim:
                    break
                frame = self.adjust_frame(cv2.imread(str(image_file)), frame_type, color_type)
                yield frame

    # --------------------------------------------------------------------------------------------------------------------
    def adjust_frame(self, frame, frame_type, color_type):
        # crop image for algorithm input, if needed:
        if frame_type == "alg_input":
            frame = self.alg_view_cropper.crop_img(frame)
        elif frame_type == "full":
            pass
        else:
            raise ValueError(f"Unknown frame_type: {frame_type}")
        if color_type == "bgr":
            pass  # do nothing
        elif color_type == "gray":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif color_type == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unknown color_type: {color_type}")
        return frame

    # --------------------------------------------------------------------------------------------------------------------
    def get_frame_at_index(self, i_frame: int, color_type="bgr", frame_type="alg_input"):
        if self.rgb_source == "Video":
            for i, frame in enumerate(self.frames_generator(color_type=color_type, frame_type=frame_type)):
                if i == i_frame:
                    return frame
        else:
            image_file = self.image_files_paths[i_frame]
            frame = self.adjust_frame(cv2.imread(str(image_file)), frame_type, color_type)
            return frame
        return None


# --------------------------------------------------------------------------------------------------------------------


def rotate_img(img, angle_deg):
    # rotate the img by angle_deg degrees around the center of the image
    height, width = img.shape[:2]
    rotation_center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(rotation_center, angle_deg, 1)
    rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_image


class RadialImageCropper:
    """Crop an image to a circular region around the optical center of the camera. The cropped image is resized to a square image where areas outside the circle are filled with zeros."""

    def __init__(self, orig_cam_info: CamInfo, alg_fov_ratio: float):
        self.cx_orig = orig_cam_info.cx
        self.cy_orig = orig_cam_info.cy
        self.view_radius = 0.5 * min(orig_cam_info.frame_width, orig_cam_info.frame_height) * alg_fov_ratio
        # the box to be cropped (that contains the FOV circle):
        self.x_min = np.floor(self.cx_orig - self.view_radius).astype(int)
        self.x_max = np.ceil(self.cx_orig + self.view_radius).astype(int)
        self.y_min = np.floor(self.cy_orig - self.view_radius).astype(int)
        self.y_max = np.ceil(self.cy_orig + self.view_radius).astype(int)
        self.cx_new = self.cx_orig - self.x_min
        self.cy_new = self.cy_orig - self.y_min
        self.width_new = self.x_max - self.x_min
        self.height_new = self.y_max - self.y_min
        print(f"Original image size: {orig_cam_info.frame_width}x{orig_cam_info.frame_height}")
        print(f"Algorithm-view image size: {self.width_new}x{self.height_new}")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def crop_img(self, frame):
        # crop and set the image to black outside the FOV circle:
        mask = np.zeros(frame.shape[:-1]).astype(np.uint8)
        mask_value = 255
        mask = cv2.circle(mask, (round(self.cx_orig), round(self.cy_orig)), round(self.view_radius), mask_value, -1)
        frame = cv2.bitwise_and(frame, frame, mask=mask)
        frame = frame[self.y_min : self.y_max, self.x_min : self.x_max]
        return frame

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def convert_coord_in_crop_to_full(self, point2d):
        """Convert a point in the cropped image (the frame the algorithm sees) to the full image."""
        x = point2d[0]
        y = point2d[1]
        return np.array([x + self.x_min, y + self.y_min])


# --------------------------------------------------------------------------------------------------------------------

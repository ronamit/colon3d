import attrs
import numpy as np

# --------------------------------------------------------------------------------------------------------------------


@attrs.define
class CamInfo:
    """
    Metadata about the view
    """

    frame_width: int  # image width (unit: pixels)
    frame_height: int  # image height (units: pixels)
    cx: float  # optical center x position in the image (units: pixels)
    cy: float  # optical center y position in the image (units: pixels)
    fx: float  # focal length, normalized by the x-size of a pixel's sensor  (units: pixels)
    fy: float  # focal length, normalized by the y-size of a pixel's sensor  (units: pixels)
    distort_pram: np.ndarray  # Fisheye distortion parameters (cv2 format)
    fps: float | None  # frames per second (units: Hz)
    min_vis_z_mm: float  # the minimal z distance from the focal point that can be seen (units: mm)
    K: np.ndarray | None = None # the camera intrinsics matrix (units: pixels)

    def __attrs_post_init__(self):
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])


# --------------------------------------------------------------------------------------------------------------------

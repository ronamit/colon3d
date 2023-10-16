import attrs

# ---------------------------------------------------------------------------------------------------------------------


@attrs.define
class ModelInfo:
    depth_model_name: str
    egomotion_model_name: str
    ref_frame_shifts: list[int]  # The time shifts of the reference frames w.r.t. the target frame
    depth_calib_type: str = "none"
    depth_calib_a: float = 1.0
    depth_calib_b: float = 0.0
    # lower bound to clip the the z-depth estimation (units: mm). If None - then no lower bound is used:
    depth_lower_bound: float | None = None
    # upper bound to clip the the z-depth estimation (units: mm). If None - then no upper bound is used:
    depth_upper_bound: float | None = None
    model_description: str = ""


# ---------------------------------------------------------------------------------------------------------------------

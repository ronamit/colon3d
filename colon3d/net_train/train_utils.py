from dataclasses import dataclass

# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class DatasetMeta:
    feed_height: int  # The height of the input images to the network
    feed_width: int  # The width of the input images to the network
    load_gt_depth: bool  # Whether to add the depth map of the target frame to each sample (default: False)
    load_gt_pose: bool  # Whether to add the ground-truth pose change between from target to the reference frames,  each sample (default: False)
    num_input_images: int  # number of frames in each sample (target + reference frames)


# ---------------------------------------------------------------------------------------------------------------------


def get_default_model_info(model_name: str) -> dict:
    if model_name in {"EndoSFM", "MonoDepth2"}:
        return {
            "feed_height": 320,  # The image size the network receives as input, in pixels
            "feed_width": 320,
            "net_out_to_mm": 1.0,  # the output of the depth network needs to be multiplied by this number to get the depth in mm
            "num_layers": 18,  # number of ResNet layers in the PoseNet
        }
    raise ValueError(f"Unknown model name: {model_name}")


# ---------------------------------------------------------------------------------------------------------------------


def get_method_info(method: str) -> str:
    if method in {"EndoSFM", "EndoSFM_GTD", "EndoSFM_GTPD"}:
        model_name = "EndoSFM"
    elif method in {"MonoDepth2", "MonoDepth2_GTD", "MonoDepth2_GTPD"}:
        model_name = "MonoDepth2"
    else:
        raise ValueError(f"Unknown method: {method}")

    # methods that use ground-truth depth labels:
    load_gt_depth = method in {"EndoSFM_GTD", "EndoSFM_GTPD", "MonoDepth2_GTD", "MonoDepth2_GTPD"}

    # methods that use ground-truth pose labels:
    load_gt_pose = method in {"EndoSFM_GTPD", "MonoDepth2_GTPD"}

    return model_name, load_gt_depth, load_gt_pose


# ---------------------------------------------------------------------------------------------------------------------

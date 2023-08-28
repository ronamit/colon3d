from torchvision.transforms import Compose

from colon3d.net_train.custom_transforms import AddInvIntrinsics, RandomFlip, RandomScaleCrop, ToTensors

# ---------------------------------------------------------------------------------------------------------------------


def get_train_transform() -> Compose:
    # set data transforms
    transform_list = [
        RandomFlip(flip_x_p=0.5, flip_y_p=0.5),
        RandomScaleCrop(max_scale=1.15),
        ToTensors(img_normalize_mean=0.45, img_normalize_std=0.225),
        AddInvIntrinsics(),
    ]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------


def get_validation_transform():
    transform_list = [ToTensors(img_normalize_mean=0.45, img_normalize_std=0.225)]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------

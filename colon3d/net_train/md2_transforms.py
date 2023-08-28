from torchvision.transforms import Compose

from colon3d.net_train.custom_transforms import NormalizeCamIntrinsicMat

# ---------------------------------------------------------------------------------------------------------------------


def get_train_transform(num_scales: int = 4):
    # composed = transforms.Compose([Rescale(256),
    #                            RandomCrop(224)])
    # TODO: create num_scales=4
    transform_list = [
        NormalizeCamIntrinsicMat(),
    ]
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------


def get_validation_transform():
    transform_list = []
    return Compose(transform_list)


# ---------------------------------------------------------------------------------------------------------------------

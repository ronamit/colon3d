# from colon3d.net_train.custom_transforms import 

from torchvision.transforms import Compose

def get_train_transforms():
    # set data transforms
    chan_normalize_mean = [0.45, 0.45, 0.45]
    chan_normalize_std = [0.225, 0.225, 0.225]
    normalize = custom_transforms.Normalize(mean=chan_normalize_mean, std=chan_normalize_std)
    train_transform =  Compose([
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.RandomScaleCrop(),
            custom_transforms.ArrayToTensor(),
            normalize,
        ])
    # TODO: add scales and use them in the loss fuction - seems like EndoSFM uses only scale=1
    # TODO: ad more transforms
    return train_transform

def get_validation_transforms():
    validation_transform = [custom_transforms.ArrayToTensor(), normalize]
    return validation_transform


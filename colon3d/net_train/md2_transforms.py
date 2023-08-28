from colon3d.net_train.md2_transforms import Rescale, RandomCrop, ToTensor, NormalizeCamIntrinsicMat

from torchvision.transforms import Compose

def get_train_transform(num_scales: int=4):
    # composed = transforms.Compose([Rescale(256),
    #                            RandomCrop(224)])
    # TODO: create num_scales=4
    pass



def get_validation_transform():
    pass
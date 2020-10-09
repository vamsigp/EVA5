from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize,Rotate, Cutout, PadIfNeeded
from albumentations.pytorch import ToTensor
import numpy as np


class album_compose():
    def __init__(self):
        self.albumentation_transforms = Compose([
            Rotate((-7.0, 7.0)),
            Cutout(),
#             CoarseDropout(),
            # RandomSizedCrop':{'height':32,'width':32,'min_max_height':[28,28]},
            # RandomCrop(10,10),
            HorizontalFlip(),
            Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ), ToTensor()])

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentation_transforms(image=img)['image']
        return img
    
class Week11Albumentations():
    def __init__(self):
        self.albumentation_transforms = Compose([
            PadIfNeeded(36,36),
            RandomCrop(32,32),
            HorizontalFlip(),
            Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            ), ToTensor()])

    def __call__(self, img):
        img = np.array(img)
        img = self.albumentation_transforms(image=img)['image']
        return img

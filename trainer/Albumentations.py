from albumentations import Compose, RandomCrop, Normalize, HorizontalFlip, Resize,Rotate, Cutout, CoarseDropout
from albumentations.pytorch import ToTensor
import numpy as np


class album_compose():

    def __init__(self):
        self.albumentation_transforms = Compose([
            Rotate((-7.0, 7.0)),
            Cutout(),
            CoarseDropout(),
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


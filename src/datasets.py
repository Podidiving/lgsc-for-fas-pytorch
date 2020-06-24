import os
import numpy as np
from PIL import Image
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor


def get_train_augmentations():
    return A.Compose(
        [
            A.LongestMaxSize(512),
            A.CoarseDropout(20),
            A.Rotate(30),
            A.RandomCrop(224, 224, p=0.5),
            A.LongestMaxSize(224),
            A.PadIfNeeded(224, 224, 0),
            A.Normalize(),
            ToTensor(),
        ]
    )


def get_test_augmentations():
    return A.Compose(
        [A.LongestMaxSize(224), A.PadIfNeeded(224, 224, 0), A.Normalize(), ToTensor()]
    )


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, root, transforms):
        self.df = df
        self.root = root
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        path = os.path.join(self.root, self.df.iloc[item].path)
        file = np.random.choice(os.listdir(path))
        full_path = os.path.join(path, file)

        image = np.array(Image.open(full_path))
        image = self.transforms(image=image)["image"]

        target = self.df.iloc[item].target

        return image, target

# Authors: Son Hai Nguyen, Miroslav Karpíšek
# Logins: xnguye16, xkarpi05
# Project: Neural network pruning
# Course: Convolutional Neural Networks
# Year: 2021


import os
import torchvision.datasets as datasets


class TinyImageNet(datasets.ImageFolder):
    IMG_SIZE = 56

    def __init__(self, root: str, train=True, **kwargs):
        root = os.path.join(root, "train" if train else "test")
        super().__init__(
            root,
            **kwargs
        )

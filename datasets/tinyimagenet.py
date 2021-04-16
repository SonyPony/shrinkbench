import os
import numpy as np
import torchvision.datasets as datasets


class TinyImageNet(datasets.ImageFolder):
    def __init__(self, root: str, train=True, **kwargs):
        root = os.path.join(root, "train" if train else "test")
        super().__init__(
            root,
            **kwargs
        )

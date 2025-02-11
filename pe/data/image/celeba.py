import torchvision
from torchvision import transforms as T
import tempfile
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

from pe.data import Data
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.constant.data import IMAGE_DATA_COLUMN_NAME

CELEBA_ATTRIBUTE_NAMES = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]


class target_transform(object):
    """From: https://github.com/fjxmlzn/DPImageBench/blob/main/data/preprocess_dataset.py"""

    def __init__(self, attr_index):
        self.idx = attr_index

    def __call__(self, attrs):
        return attrs[self.idx]

    def __repr__(self):
        return self.__class__.__name__


class CelebA(Data):
    """The CelebA dataset."""

    def __init__(self, root_dir, res=32, attr_name="Male", split="train"):
        """Constructor.

        :param root_dir: The root directory of the CelebA dataset
        :type root_dir: str
        :param res: The resolution of the image, defaults to 32
        :type res: int, optional
        :param attr_name: The attribute name to use as the label, defaults to "Male"
        :type attr_name: str, optional
        :param split: The split of the dataset, default is "train"
        :type split: str, optional
        """
        if root_dir is None:
            root_dir = tempfile.gettempdir()
        if os.path.isdir(os.path.join(root_dir, "celeba", "img_align_celeba")):
            download = False
        else:
            download = True
        transform = T.Compose(
            [
                T.Resize(res),
                T.CenterCrop(res),
            ]
        )
        attr_index = CELEBA_ATTRIBUTE_NAMES.index(attr_name)
        dataset = torchvision.datasets.CelebA(
            root=root_dir,
            split=split,
            target_type="attr",
            download=download,
            target_transform=target_transform(attr_index),
            transform=transform,
        )
        images = []
        labels = []
        for i in tqdm(range(len(dataset))):
            image, label = dataset[i]
            images.append(np.array(image))
            labels.append(int(label.numpy()))
        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: images,
                LABEL_ID_COLUMN_NAME: labels,
            }
        )
        metadata = {"label_info": [{"name": f"{attr_name}_{i}"} for i in [0, 1]]}
        super().__init__(data_frame=data_frame, metadata=metadata)

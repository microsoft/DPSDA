import pandas as pd
from wilds import get_dataset
from tqdm import tqdm
import numpy as np
import torchvision.transforms as T

from pe.data import Data
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.constant.data import IMAGE_DATA_COLUMN_NAME

CAMELYON17_LABEL_NAMES = [
    "no_tumor",
    "tumor",
]


class Camelyon17(Data):
    """The Camelyon17 dataset."""

    def __init__(self, split="train", root_dir="data", res=64):
        """Constructor.

        :param split: The split of the dataset. It should be either "train", "val", or "test", defaults to "train"
        :type split: str, optional
        :param root_dir: The root directory to save the dataset, defaults to "data"
        :type root_dir: str, optional
        :param res: The resolution of the images, defaults to 64
        :type res: int, optional
        :raises ValueError: If the split is invalid
        """
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Invalid split: {split}")
        dataset = get_dataset(dataset="camelyon17", download=True, root_dir=root_dir)
        data = dataset.get_subset(split)
        transform = T.Resize(res)

        images = []
        labels = []
        for i in tqdm(range(len(data))):
            image, label, _ = data[i]
            images.append(np.array(transform(image)))
            labels.append(label.item())
        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: images,
                LABEL_ID_COLUMN_NAME: labels,
            }
        )
        metadata = {"label_info": [{"name": n} for n in CAMELYON17_LABEL_NAMES]}
        super().__init__(data_frame=data_frame, metadata=metadata)

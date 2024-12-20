import torchvision
import tempfile
import pandas as pd

from pe.data import Data
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.constant.data import IMAGE_DATA_COLUMN_NAME

CIFAR10_LABEL_NAMES = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


class Cifar10(Data):
    """The CIFAR10 dataset."""

    def __init__(self, split="train"):
        """Constructor.

        :param split: The split of the dataset. It should be either "train" or "test", defaults to "train"
        :type split: str, optional
        :raises ValueError: If the split is invalid
        """
        if split not in ["train", "test"]:
            raise ValueError(f"Invalid split: {split}")
        train = split == "train"
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = torchvision.datasets.CIFAR10(root=tmp_dir, train=train, download=True)
        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: list(dataset.data),
                LABEL_ID_COLUMN_NAME: dataset.targets,
            }
        )
        metadata = {"label_info": [{"name": n} for n in CIFAR10_LABEL_NAMES]}
        super().__init__(data_frame=data_frame, metadata=metadata)

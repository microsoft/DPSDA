import pandas as pd
import torchvision.datasets
import torchvision.transforms as T
from tqdm import tqdm
import torch

from pe.data import Data
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.constant.data import IMAGE_DATA_COLUMN_NAME


class ImageNet(Data):
    """The ImageNet dataset."""

    def __init__(self, root_dir, conditional=False, split="train", res=32, batch_size=1000, num_workers=10):
        """Constructor.

        :param root_dir: The root directory of the dataset.
        :param conditional: Whether to use conditional ImageNet. Defaults to False
        :type conditional: bool, optional
        :param split: The split of the dataset, defaults to "train"
        :type split: str, optional
        :param res: The resolution of the images, defaults to 32
        :type res: int, optional
        :param batch_size: The batch size to load the images, defaults to 1000
        :type batch_size: int, optional
        :param num_workers: The number of workers to load the images, defaults to 10
        :type num_workers: int, optional
        """
        transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.Resize(res), T.PILToTensor()])
        dataset = torchvision.datasets.ImageNet(
            root=root_dir,
            split=split,
            transform=transform,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
        )

        images = []
        for batch in tqdm(data_loader, desc="Loading ImageNet", unit="batch"):
            images.append(batch[0])
        images = torch.cat(images, dim=0)
        images = images.permute(0, 2, 3, 1).numpy()

        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: list(images),
                LABEL_ID_COLUMN_NAME: dataset.targets if conditional else [0] * len(images),
            }
        )
        if conditional:
            metadata = {"label_info": [{"name": n} for n in map(str, dataset.classes)]}
        else:
            metadata = {"label_info": [{"name": "none"}]}
        super().__init__(data_frame=data_frame, metadata=metadata)

import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import zipfile
from PIL import Image
import torchvision.transforms as T
from collections import defaultdict

from pe.data import Data
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.constant.data import IMAGE_DATA_COLUMN_NAME
from pe.util import download

CAT_LABEL_NAMES = [
    "cookie",
    "doudou",
]


class Cat(Data):
    """The Cat dataset."""

    #: The URL of the dataset
    URL = "https://www.kaggle.com/api/v1/datasets/download/fjxmlzn/cat-cookie-doudou"

    def __init__(self, root_dir="data", res=512):
        """Constructor.

        :param root_dir: The root directory to save the dataset, defaults to "data"
        :type root_dir: str, optional
        :param res: The resolution of the images, defaults to 512
        :type res: int, optional
        """
        self._zip_path = os.path.join(root_dir, "cat-cookie-doudou.zip")
        self._download()
        data = self._read_data()
        transform = T.Resize(res)

        images = []
        labels = []
        for label, sub_images in data.items():
            for image in tqdm(sub_images, desc=f"Processing {label} images"):
                image = Image.fromarray(image)
                image = transform(image)
                image = np.array(image)
                images.append(image)
                labels.append(CAT_LABEL_NAMES.index(label))
        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: images,
                LABEL_ID_COLUMN_NAME: labels,
            }
        )
        metadata = {"label_info": [{"name": n} for n in CAT_LABEL_NAMES]}
        super().__init__(data_frame=data_frame, metadata=metadata)

    def _download(self):
        """Download the dataset if it does not exist."""
        if not os.path.exists(self._zip_path):
            os.makedirs(os.path.dirname(self._zip_path), exist_ok=True)
            download(url=self.URL, fname=self._zip_path)

    def _read_data(self):
        """Read the data from the zip file."""
        data = defaultdict(list)
        with zipfile.ZipFile(self._zip_path) as z:
            for name in tqdm(z.namelist(), desc="Reading zip file"):
                with z.open(name) as f:
                    image = Image.open(f)
                    label = name.split("/")[0]
                    data[label].append(np.array(image))
        return data

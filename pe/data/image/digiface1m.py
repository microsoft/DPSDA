import pandas as pd
import os
import numpy as np
from PIL import Image
import glob
from tqdm.contrib.concurrent import process_map

from pe.data import Data
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.constant.data import IMAGE_DATA_COLUMN_NAME


class DigiFace1M(Data):
    """The DigiFace1M dataset (https://github.com/microsoft/DigiFace1M/)."""

    def __init__(self, root_dir, res=32, num_processes=50, chunksize=1000):
        """Constructor.

        :param root_dir: The root directory of the dataset, defaults to "data"
        :type root_dir: str, optional
        :param res: The resolution of the images, defaults to 32
        :type res: int, optional
        :param num_processes: The number of processes to use for parallel processing, defaults to 50
        :type num_processes: int, optional
        :param chunksize: The chunk size to use for parallel processing, defaults to 1000
        :type chunksize: int, optional
        :raises ValueError: If the number of images in ``root_dir`` is not 1,219,995
        """
        self._res = res
        files = glob.glob(os.path.join(root_dir, "*", "*.png"))
        if len(files) != 1219995:
            raise ValueError(
                f"Expected 1,219,995 images, but found {len(files)}. Please download the dataset from "
                "https://github.com/microsoft/DigiFace1M, unzip the images into a folder, and pass the path to the "
                "folder as ``root_dir``."
            )
        images = process_map(self._read_and_process_image, files, max_workers=num_processes, chunksize=chunksize)
        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: images,
                LABEL_ID_COLUMN_NAME: [0] * len(images),
            }
        )
        metadata = {"label_info": [{"name": "None"}]}
        super().__init__(data_frame=data_frame, metadata=metadata)

    def _read_and_process_image(self, path):
        """Read and process an image.

        :param path: The path to the image
        :type path: str
        :return: The processed image
        :rtype: np.ndarray
        """
        image = Image.open(path)
        image = image.convert("RGB")
        image = image.resize((self._res, self._res))
        return np.array(image)

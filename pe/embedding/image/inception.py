import tempfile
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm

from cleanfid.inception_torchscript import InceptionV3W
from cleanfid.resize import build_resizer
from cleanfid.resize import make_resizer

from pe.embedding import Embedding
from pe.constant.data import IMAGE_DATA_COLUMN_NAME
from pe.logging import execution_logger


def to_uint8(x, min, max):
    x = (x - min) / (max - min)
    x = np.around(np.clip(x * 255, a_min=0, a_max=255)).astype(np.uint8)
    return x


class Inception(Embedding):
    """Compute the Inception embedding of images."""

    def __init__(self, res, device="cuda", batch_size=2000):
        """Constructor.

        :param res: The resolution of the images. The images will be resized to (res, res) before computing the
            embedding
        :type res: int
        :param device: The device to use for computing the embedding, defaults to "cuda"
        :type device: str, optional
        :param batch_size: The batch size to use for computing the embedding, defaults to 2000
        :type batch_size: int, optional
        """
        super().__init__()
        self._temp_folder = tempfile.TemporaryDirectory()
        self._device = device
        self._inception = InceptionV3W(path=self._temp_folder.name, download=True, resize_inside=False).to(device)
        self._resize_pre = make_resizer(
            library="PIL",
            quantize_after=False,
            filter="bicubic",
            output_size=(res, res),
        )
        self._resizer = build_resizer("clean")
        self._batch_size = batch_size

    def compute_embedding(self, data):
        """Compute the Inception embedding of images.

        :param data: The data object containing the images
        :type data: :py:class:`pe.data.Data`
        :return: The data object with the computed embedding
        :rtype: :py:class:`pe.data.Data`
        """
        uncomputed_data = self.filter_uncomputed_rows(data)
        if len(uncomputed_data.data_frame) == 0:
            execution_logger.info(f"Embedding: {self.column_name} already computed")
            return data
        execution_logger.info(
            f"Embedding: computing {self.column_name} for {len(uncomputed_data.data_frame)}/{len(data.data_frame)}"
            " samples"
        )
        x = np.stack(uncomputed_data.data_frame[IMAGE_DATA_COLUMN_NAME].values, axis=0)
        if x.shape[3] == 1:
            x = np.repeat(x, 3, axis=3)
        embeddings = []
        for i in tqdm(range(0, len(x), self._batch_size)):
            transformed_x = []
            for j in range(i, min(i + self._batch_size, len(x))):
                image = x[j]
                image = self._resize_pre(image)
                image = to_uint8(image, min=0, max=255)
                image = self._resizer(image)
                transformed_x.append(image)
            transformed_x = np.stack(transformed_x, axis=0).transpose((0, 3, 1, 2))
            embeddings.append(self._inception(torch.from_numpy(transformed_x).to(self._device)))
        embeddings = torch.cat(embeddings, dim=0)
        embeddings = embeddings.cpu().detach().numpy()
        uncomputed_data.data_frame[self.column_name] = pd.Series(
            list(embeddings), index=uncomputed_data.data_frame.index
        )
        execution_logger.info(
            f"Embedding: finished computing {self.column_name} for "
            f"{len(uncomputed_data.data_frame)}/{len(data.data_frame)} samples"
        )
        return self.merge_computed_rows(data, uncomputed_data)

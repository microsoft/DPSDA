import numpy as np
import torch
import pandas as pd

from fld.features.InceptionFeatureExtractor import InceptionFeatureExtractor

from cleanfid.resize import make_resizer

from pe.embedding import Embedding
from pe.constant.data import IMAGE_DATA_COLUMN_NAME
from pe.logging import execution_logger


def to_uint8(x, min, max):
    x = (x - min) / (max - min)
    x = np.around(np.clip(x * 255, a_min=0, a_max=255)).astype(np.uint8)
    return x


class FLDInception(Embedding):
    """Compute the Inception embedding of images using FLD library."""

    def __init__(self, res=None):
        """Constructor.

        :param res: The resolution of the images. The images will be resized to (res, res) before computing the
            embedding. If None, the images will not be resized. Defaults to None
        :type res: int, optional
        """
        super().__init__()
        self._feature_extractor = InceptionFeatureExtractor()
        if res is not None:
            self._resize_pre = make_resizer(
                library="PIL",
                quantize_after=False,
                filter="bicubic",
                output_size=(res, res),
            )
        else:
            self._resize_pre = None

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
        if self._resize_pre is not None:
            x = [self._resize_pre(image) for image in x]
            x = np.stack(x, axis=0)
            x = to_uint8(x, min=0, max=255)
        x = np.transpose(x, (0, 3, 1, 2))
        x = torch.from_numpy(x)
        embeddings = self._feature_extractor.get_tensor_features(x)
        embeddings = embeddings.cpu().detach().numpy()
        uncomputed_data.data_frame[self.column_name] = pd.Series(
            list(embeddings), index=uncomputed_data.data_frame.index
        )
        execution_logger.info(
            f"Embedding: finished computing {self.column_name} for "
            f"{len(uncomputed_data.data_frame)}/{len(data.data_frame)} samples"
        )
        return self.merge_computed_rows(data, uncomputed_data)

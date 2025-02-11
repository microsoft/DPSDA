import numpy as np
import pandas as pd

from pe.api import API
from pe.logging import execution_logger
from pe.data import Data
from pe.constant.data import IMAGE_DATA_COLUMN_NAME
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.api.util import ConstantList


DATA_ID_COLUMN_NAME = "PE.NEAREST_IMAGE.DATA_ID"


def _to_constant_list_if_needed(value):
    if not isinstance(value, list):
        value = ConstantList(value)
    return value


class NearestImage(API):
    """The API that generates synthetic images by randomly drawing an image from the given dataset as the RANDOM_API
    and finding the nearest images in the given dataset as the VARIATION_API."""

    def __init__(
        self,
        data,
        embedding,
        nearest_neighbor_mode,
        variation_degrees,
        nearest_neighbor_backend="auto",
    ):
        """Constructor.

        :param data: The data object that contains the images
        :type data: :py:class:`pe.data.Data`
        :param embedding: The embedding object that computes the embeddings of the images
        :type embedding: :py:class:`pe.embedding.Embedding`
        :param nearest_neighbor_mode: The distance metric to use for finding the nearest neighbors. It should be one
            of the following: "l2" (l2 distance), "cos_sim" (cosine similarity), "ip" (inner product). Not all backends
            support all modes
        :type nearest_neighbor_mode: str
        :param variation_degrees: The variation degrees utilized at each PE iteration. If a single value is provided,
            the same variation degree will be used for all iterations. The value means the number of nearest neighbors
            to consider for the VARIAITON_API
        :type variation_degrees: int or list[int]
        :param nearest_neighbor_backend: The backend to use for finding the nearest neighbors. It should be one of the
            following: "faiss" (FAISS), "sklearn" (scikit-learn), "auto" (using FAISS if available, otherwise
            scikit-learn). Defaults to "auto". FAISS supports GPU and is much faster when the number of samples is
            large. It requires the installation of `faiss-gpu` or `faiss-cpu` package. See https://faiss.ai/
        :type nearest_neighbor_backend: str, optional
        :raises ValueError: If the `nearest_neighbor_backend` is unknown
        """
        super().__init__()
        self._data = data
        self._embedding = embedding
        self._nearest_neighbor_mode = nearest_neighbor_mode
        self._nearest_neighbor_backend = nearest_neighbor_backend
        self._variation_degrees = _to_constant_list_if_needed(variation_degrees)
        self._max_variation_degree = (
            self._variation_degrees[0] if isinstance(variation_degrees, ConstantList) else max(self._variation_degrees)
        )

        if nearest_neighbor_backend.lower() == "faiss":
            from pe.histogram.nearest_neighbor_backend.faiss import search

            self._search = search
        elif nearest_neighbor_backend.lower() == "sklearn":
            from pe.histogram.nearest_neighbor_backend.sklearn import search

            self._search = search
        elif nearest_neighbor_backend.lower() == "auto":
            from pe.histogram.nearest_neighbor_backend.auto import search

            self._search = search
        else:
            raise ValueError(f"Unknown backend: {nearest_neighbor_backend}")

        self._build_nearest_neighbor_graph()

    def _build_nearest_neighbor_graph(self):
        """Finding the nearest neighbor for each sample in the given dataset."""
        self._data = self._embedding.compute_embedding(self._data)
        embedding = np.stack(self._data.data_frame[self._embedding.column_name].values, axis=0).astype(np.float32)
        distances, ids = self._search(
            syn_embedding=embedding,
            priv_embedding=embedding,
            num_nearest_neighbors=self._max_variation_degree,
            mode=self._nearest_neighbor_mode,
        )
        sorted_indices = np.argsort(distances, axis=1)
        self._nearest_neighbor_ids = ids[np.arange(len(ids))[:, None], sorted_indices]
        self._nearest_neighbor_distances = distances[np.arange(len(distances))[:, None], sorted_indices]

    def random_api(self, label_info, num_samples):
        """Generating random synthetic data by randomly drawing images from the given dataset.

        :param label_info: The info of the label, not utilized in this API
        :type label_info: omegaconf.dictconfig.DictConfig
        :param num_samples: The number of random samples to generate
        :type num_samples: int
        :return: The data object of the generated synthetic data
        :rtype: :py:class:`pe.data.Data`
        """
        label_name = label_info.name
        execution_logger.info(f"RANDOM API: creating {num_samples} samples for label {label_name}")

        data_ids = np.random.choice(len(self._data.data_frame), num_samples, replace=False)
        images = self._data.data_frame.iloc[data_ids][IMAGE_DATA_COLUMN_NAME].values
        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: images,
                DATA_ID_COLUMN_NAME: data_ids,
                LABEL_ID_COLUMN_NAME: 0,
            }
        )
        metadata = {"label_info": [label_info]}
        execution_logger.info(f"RANDOM API: finished creating {num_samples} samples for label {label_name}")
        return Data(data_frame=data_frame, metadata=metadata)

    def variation_api(self, syn_data):
        """Generating variations of the synthetic data by finding the nearest images in the given dataset.

        :param syn_data: The data object of the synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        :return: The data object of the variation of the input synthetic data
        :rtype: :py:class:`pe.data.Data`
        """
        execution_logger.info(f"VARIATION API: creating variations for {len(syn_data.data_frame)} samples")
        original_data_ids = list(syn_data.data_frame[DATA_ID_COLUMN_NAME].values)
        iteration = getattr(syn_data.metadata, "iteration", -1)
        variation_degree = self._variation_degrees[iteration + 1]

        execution_logger.info(f"VARIATION API parameters: variation_degree={variation_degree}")

        nn_ids = np.random.choice(variation_degree, len(original_data_ids), replace=True)
        new_data_ids = self._nearest_neighbor_ids[original_data_ids, nn_ids]
        images = self._data.data_frame.iloc[new_data_ids][IMAGE_DATA_COLUMN_NAME].values

        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: images,
                DATA_ID_COLUMN_NAME: new_data_ids,
                LABEL_ID_COLUMN_NAME: syn_data.data_frame[LABEL_ID_COLUMN_NAME].values,
            }
        )
        execution_logger.info(f"VARIATION API: finished creating variations for {len(syn_data.data_frame)} samples")
        return Data(data_frame=data_frame, metadata=syn_data.metadata)

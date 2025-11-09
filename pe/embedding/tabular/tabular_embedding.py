import pandas as pd
import numpy as np

from pe.embedding import Embedding
from pe.logging import execution_logger
from pe.constant.data import TABULAR_DATA_COLUMN_NAME
from pe.data.tabular.tabular_csv import TabularColumnType


class TabularEmbedding(Embedding):
    """Compute the tabular embedding."""

    def __init__(self, info, cat_weight=1 / 3, num_weight=1):
        """Constructor.

        :param info: The information (categories and numerical bounds) of the private data
        :type info: dict
        :param cat_weight: The weight for the categorical columns, defaults to 1/3
        :type cat_weight: float, optional
        :param num_weight: The weight for the numerical columns, defaults to 1
        :type num_weight: float, optional
        """
        super().__init__()
        self._info = info
        self._cat_weight = cat_weight
        self._num_weight = num_weight

    def compute_embedding(self, data):
        """Compute the tabular embedding. (the embedding is computed using the features only, not the labels)
        Vectorization per column is implemented to improve the performance.

        :param data: The data object containing the tabular data
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

        cat_columns = data.metadata["cat_columns"]
        num_columns = data.metadata["int_columns"] + data.metadata["float_columns"]
        feature_columns = data.metadata["feature_columns"]

        features_list = uncomputed_data.data_frame[TABULAR_DATA_COLUMN_NAME].tolist()
        features_df = pd.DataFrame(features_list, columns=feature_columns)

        # Build embedding vectors
        embedding_vectors = []
        num_samples = len(features_df)

        for col in num_columns:
            if col in self._info and self._info[col]["type"] in [TabularColumnType.INTEGER, TabularColumnType.FLOAT]:
                col_values = features_df[col].values
                min_val = self._info[col]["min"]
                max_val = self._info[col]["max"]
                normalized = (col_values - min_val) * self._num_weight / (max_val - min_val)
                embedding_vectors.append(normalized.reshape(-1, 1))
            else:
                raise ValueError(f"Tabular Embedding: No info for numerical column {col}, cannot proceed.")

        for col in cat_columns:
            if col in self._info and self._info[col]["type"] == TabularColumnType.CATEGORICAL:
                categories = self._info[col]["categories"]
                num_categories = len(categories)
                col_values = features_df[col].values

                # Get indices for each sample using vectorized lookup
                category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
                indices = pd.Series(col_values).map(category_to_idx).fillna(0).astype(int).values

                # Create one-hot-like vectors
                one_hot = np.zeros((num_samples, num_categories))
                one_hot[np.arange(num_samples), indices] = self._cat_weight
                embedding_vectors.append(one_hot)
            else:
                raise ValueError(f"Tabular Embedding: No info for categorical column {col}, cannot proceed.")

        # Concatenate all vectors
        embeddings = np.concatenate(embedding_vectors, axis=1)

        # Convert to list of lists and store
        uncomputed_data.data_frame[self.column_name] = pd.Series(
            [emb.tolist() for emb in embeddings], index=uncomputed_data.data_frame.index
        )

        execution_logger.info(
            f"Embedding: finished computing {self.column_name} for "
            f"{len(uncomputed_data.data_frame)}/{len(data.data_frame)} samples"
        )
        return self.merge_computed_rows(data, uncomputed_data)

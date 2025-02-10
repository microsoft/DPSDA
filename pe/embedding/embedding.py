from abc import ABC, abstractmethod

from pe.constant.data import EMBEDDING_COLUMN_NAME
from pe.data import Data


class Embedding(ABC):
    """The abstract class that computes the embedding of samples."""

    @property
    def column_name(self):
        """The column name to be used in the data frame."""
        return f"{EMBEDDING_COLUMN_NAME}.{type(self).__name__}"

    @abstractmethod
    def compute_embedding(self, data):
        """Compute the embedding of samples.

        :param data: The data to compute the embedding
        :type data: :py:class:`pe.data.Data`
        """
        ...

    def filter_uncomputed_rows(self, data):
        data_frame = data.data_frame
        if self.column_name in data_frame.columns:
            data_frame = data_frame[data_frame[self.column_name].isna()]
        return Data(data_frame=data_frame, metadata=data.metadata)

    def merge_computed_rows(self, data, computed_data):
        data_frame = data.data_frame
        computed_data_frame = computed_data.data_frame
        data_frame.update(computed_data_frame)
        return Data(data_frame=data_frame, metadata=data.metadata)

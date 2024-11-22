from abc import ABC, abstractmethod

from pe.constant.data import EMBEDDING_COLUMN_NAME


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
        :type data: :py:class:`pe.data.data.Data`
        """
        pass

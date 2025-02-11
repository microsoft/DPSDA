import numpy as np
from fld.metrics.PrecisionRecall import PrecisionRecall
import torch

from pe.callback.callback import Callback
from pe.metric_item import FloatMetricItem
from pe.logging import execution_logger


class ComputePrecisionRecall(Callback):
    """The callback that computes precision and recall metrics (https://arxiv.org/abs/1904.06991) between the private
    and synthetic data."""

    def __init__(self, priv_data, embedding, num_precision_neighbors=4, num_recall_neighbors=5, filter_criterion=None):
        """Constructor.

        :param priv_data: The private data
        :type priv_data: :py:class:`pe.data.Data`
        :param embedding: The embedding to compute the FID
        :type embedding: :py:class:`pe.embedding.Embedding`
        :param num_precision_neighbors: The number of neighbors to use for computing precision, defaults to 4
            following https://github.com/marcojira/fld/tree/main
        :type num_precision_neighbors: int, optional
        :param num_recall_neighbors: The number of neighbors to use for computing recall, defaults to 5
            following https://github.com/marcojira/fld/tree/main
        :type num_recall_neighbors: int, optional
        :param filter_criterion: Only computes the metric based on samples satisfying the criterion. None means no
            filtering. Defaults to None
        :type filter_criterion: dict, optional
        """
        self._priv_data = priv_data
        self._embedding = embedding
        self._num_precision_neighbors = num_precision_neighbors
        self._num_recall_neighbors = num_recall_neighbors
        self._filter_criterion = filter_criterion
        self._filter_criterion_str = str(filter_criterion).replace(" ", "")
        self._precision_metric_name = (
            f"precision_{self._embedding.column_name}_{self._filter_criterion_str}"
            if filter_criterion
            else f"precision_{self._embedding.column_name}"
        )
        self._recall_metric_name = (
            f"recall_{self._embedding.column_name}_{self._filter_criterion_str}"
            if filter_criterion
            else f"recall_{self._embedding.column_name}"
        )

        self._priv_data = self._embedding.compute_embedding(self._priv_data)
        priv_embedding = np.stack(self._priv_data.data_frame[self._embedding.column_name].values, axis=0).astype(
            np.float32
        )
        self._priv_embedding = priv_embedding

        self._precision = PrecisionRecall(mode="Precision", num_neighbors=self._num_precision_neighbors)
        self._recall = PrecisionRecall(mode="Recall", num_neighbors=self._num_recall_neighbors)

    def __call__(self, syn_data):
        """This function is called after each PE iteration that computes the FID between the private and synthetic
        data.

        :param syn_data: The synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        :return: The FID between the private and synthetic data
        :rtype: list[:py:class:`pe.metric_item.FloatMetricItem`]
        """
        execution_logger.info(
            f"Computing precision and recall ({self._embedding.column_name}, {self._filter_criterion_str})"
        )
        syn_data = syn_data.filter(self._filter_criterion)
        execution_logger.info(f"Number of samples after filtering: {len(syn_data.data_frame)}")
        syn_data = self._embedding.compute_embedding(syn_data)
        syn_embedding = np.stack(syn_data.data_frame[self._embedding.column_name].values, axis=0).astype(np.float32)
        priv_embedding_torch = torch.from_numpy(self._priv_embedding)
        syn_embedding_torch = torch.from_numpy(syn_embedding)
        precision = self._precision.compute_metric(priv_embedding_torch, None, syn_embedding_torch)
        recall = self._recall.compute_metric(priv_embedding_torch, None, syn_embedding_torch)
        precision_metric_item = FloatMetricItem(name=self._precision_metric_name, value=precision)
        recall_metric_item = FloatMetricItem(name=self._recall_metric_name, value=recall)
        execution_logger.info(
            f"Finished computing precision and recall ({self._embedding.column_name}, {self._filter_criterion_str})"
        )
        return [precision_metric_item, recall_metric_item]

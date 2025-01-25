import numpy as np
import cleanfid.fid

from pe.callback.callback import Callback
from pe.metric_item import FloatMetricItem
from pe.logging import execution_logger


class ComputeFID(Callback):
    """The callback that computes the Frechet Inception Distance (FID) between the private and synthetic data."""

    def __init__(self, priv_data, embedding, filter_criterion=None):
        """Constructor.

        :param priv_data: The private data
        :type priv_data: :py:class:`pe.data.Data`
        :param embedding: The embedding to compute the FID
        :type embedding: :py:class:`pe.embedding.Embedding`
        :param filter_criterion: Only computes the metric based on samples satisfying the criterion. None means no
            filtering. Defaults to None
        :type filter_criterion: dict, optional
        """
        self._priv_data = priv_data
        self._embedding = embedding
        self._filter_criterion = filter_criterion
        self._filter_criterion_str = str(filter_criterion).replace(" ", "")
        self._metric_name = (
            f"fid_{self._embedding.column_name}_{self._filter_criterion_str}"
            if filter_criterion
            else f"fid_{self._embedding.column_name}"
        )

        self._priv_data = self._embedding.compute_embedding(self._priv_data)
        priv_embedding = np.stack(self._priv_data.data_frame[self._embedding.column_name].values, axis=0).astype(
            np.float32
        )
        self._real_mu = np.mean(priv_embedding, axis=0)
        self._real_sigma = np.cov(priv_embedding, rowvar=False)

    def __call__(self, syn_data):
        """This function is called after each PE iteration that computes the FID between the private and synthetic
        data.

        :param syn_data: The synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        :return: The FID between the private and synthetic data
        :rtype: list[:py:class:`pe.metric_item.FloatMetricItem`]
        """
        execution_logger.info(f"Computing FID ({self._embedding.column_name}, {self._filter_criterion_str})")
        syn_data = syn_data.filter(self._filter_criterion)
        execution_logger.info(f"Number of samples after filtering: {len(syn_data.data_frame)}")
        syn_data = self._embedding.compute_embedding(syn_data)
        syn_embedding = np.stack(syn_data.data_frame[self._embedding.column_name].values, axis=0).astype(np.float32)
        syn_mu = np.mean(syn_embedding, axis=0)
        syn_sigma = np.cov(syn_embedding, rowvar=False)
        fid = cleanfid.fid.frechet_distance(
            mu1=self._real_mu,
            sigma1=self._real_sigma,
            mu2=syn_mu,
            sigma2=syn_sigma,
        )
        metric_item = FloatMetricItem(name=self._metric_name, value=fid)
        execution_logger.info(f"Finished computing FID ({self._embedding.column_name}, {self._filter_criterion_str})")
        return [metric_item]

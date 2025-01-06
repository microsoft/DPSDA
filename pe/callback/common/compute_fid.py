import numpy as np
import cleanfid.fid

from pe.callback.callback import Callback
from pe.metric_item import FloatMetricItem
from pe.logging import execution_logger


class ComputeFID(Callback):
    """The callback that computes the Frechet Inception Distance (FID) between the private and synthetic data."""

    def __init__(self, priv_data, embedding):
        """Constructor.

        :param priv_data: The private data
        :type priv_data: :py:class:`pe.data.data.Data`
        :param embedding: The embedding to compute the FID
        :type embedding: :py:class:`pe.embedding.embedding.Embedding`
        """
        self._priv_data = priv_data
        self._embedding = embedding

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
        :type syn_data: :py:class:`pe.data.data.Data`
        :return: The FID between the private and synthetic data
        :rtype: list[:py:class:`pe.metric_item.FloatMetricItem`]
        """
        execution_logger.info(f"Computing FID ({type(self._embedding).__name__})")
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
        metric_item = FloatMetricItem(name=f"fid_{self._embedding.column_name}", value=fid)
        execution_logger.info(f"Finished computing FID ({type(self._embedding).__name__})")
        return [metric_item]

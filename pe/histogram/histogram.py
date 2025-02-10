from abc import ABC, abstractmethod


class Histogram(ABC):
    """The abstract class for computing the histogram over synthetic samples. The histogram values indicate how good
    each synthetic sample is in terms their closeness to the private data.
    """

    @abstractmethod
    def compute_histogram(self, priv_data, syn_data):
        """Compute the histogram over the synthetic data using the private data.

        :param priv_data: The private data
        :type priv_data: :py:class:`pe.data.Data`
        :param syn_data: The synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        """
        ...

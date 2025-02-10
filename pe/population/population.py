from abc import ABC, abstractmethod


class Population(ABC):
    """The abstract class that generates synthetic data."""

    @abstractmethod
    def initial(self, label_info, num_samples):
        """Generate the initial synthetic data.

        :param label_info: The label info
        :type label_info: omegaconf.dictconfig.DictConfig
        :param num_samples: The number of samples to generate
        :type num_samples: int
        """
        ...

    @abstractmethod
    def next(self, syn_data, num_samples):
        """Generate the next synthetic data.

        :param syn_data: The synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        :param num_samples: The number of samples to generate
        :type num_samples: int
        """
        ...

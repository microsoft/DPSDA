from abc import ABC, abstractmethod


class API(ABC):
    """The abstract class that defines the APIs for the synthetic data generation."""

    @abstractmethod
    def random_api(self, label_info, num_samples):
        """The abstract method that generates random synthetic data.

        :param label_info: The info of the label
        :type label_info: omegaconf.dictconfig.DictConfig
        :param num_samples: The number of random samples to generate
        :type num_samples: int
        """
        ...

    @abstractmethod
    def variation_api(self, syn_data):
        """The abstract method that generates variations of the synthetic data.

        :param syn_data: The data object of the synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        """
        ...

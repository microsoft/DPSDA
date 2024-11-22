from abc import ABC, abstractmethod


class Population(ABC):
    """The abstract class that generates synthetic data."""

    @abstractmethod
    def initial(self, label_name, num_samples):
        """Generate the initial synthetic data.

        :param label_name: The label name
        :type label_name: str
        :param num_samples: The number of samples to generate
        :type num_samples: int
        """
        pass

    @abstractmethod
    def next(self, syn_data, num_samples):
        """Generate the next synthetic data.

        :param syn_data: The synthetic data
        :type syn_data: :py:class:`pe.data.data.Data`
        :param num_samples: The number of samples to generate
        :type num_samples: int
        """
        pass

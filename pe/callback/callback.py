from abc import ABC, abstractmethod


class Callback(ABC):
    """The abstract class that defines the callback for the synthetic data generation. These callbacks can be
    configured to be called after each PE iteration.
    """

    @abstractmethod
    def __call__(self, syn_data):
        """This function is called after each PE iteration.

        :param syn_data: The :py:class:`pe.data.Data` object of the synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        """
        ...

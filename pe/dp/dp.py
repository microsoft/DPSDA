from abc import ABC, abstractmethod


class DP(ABC):
    """The abstract class for Differential Privacy (DP) histogram mechanism."""

    @abstractmethod
    def set_epsilon_and_delta(self, num_iterations, epsilon, delta, noise_multiplier):
        """Set the epsilon and delta for the DP mechanism. Either epsilon or noise_multiplier should be None.

        :param num_iterations: The number of PE iterations
        :type num_iterations: int
        :param epsilon: The epsilon value of DP
        :type epsilon: float or None
        :param delta: The delta value of DP
        :type delta: float
        :param noise_multiplier: The noise multiplier of the DP mechanism
        :type noise_multiplier: float or None
        """
        ...

    @abstractmethod
    def add_noise(self, syn_data):
        """Add noise to the histogram of synthetic data.

        :param syn_data: The synthetic data to add noise
        :type syn_data: :py:class:`pe.data.Data`
        """
        ...

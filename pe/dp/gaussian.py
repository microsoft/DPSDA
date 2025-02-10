import scipy.optimize
from scipy.optimize import root_scalar
import numpy as np

from pe.dp import DP
from pe.logging import execution_logger
from pe.constant.data import CLEAN_HISTOGRAM_COLUMN_NAME
from pe.constant.data import DP_HISTOGRAM_COLUMN_NAME


def delta_Gaussian(eps, mu):
    """Compute delta of Gaussian mechanism with shift mu or equivalently noise scale 1/mu.

    :param eps: The epsilon value
    :type eps: float
    :param mu: The mu value
    :type mu: float
    :return: The delta value
    :rtype: float
    """
    if mu == 0:
        return 0
    if np.isinf(np.exp(eps)):
        return 0
    return scipy.stats.norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * scipy.stats.norm.cdf(-eps / mu - mu / 2)


def eps_Gaussian(delta, mu, max_epsilon):
    """Compute eps of Gaussian mechanism with shift mu or equivalently noise scale 1/mu.

    :param delta: The delta value
    :type delta: float
    :param mu: The mu value
    :type mu: float
    :param max_epsilon: The maximum epsilon value to search for
    :type max_epsilon: float
    """

    def f(x):
        return delta_Gaussian(x, mu) - delta

    return root_scalar(f, bracket=[0, max_epsilon], method="brentq").root


def compute_epsilon(noise_multiplier, num_steps, delta, max_epsilon=1e7):
    """Compute epsilon of Gaussian mechanism.

    :param noise_multiplier: The noise multiplier
    :type noise_multiplier: float
    :param num_steps: The number of steps
    :type num_steps: int
    :param delta: The delta value
    :type delta: float
    :param max_epsilon: The maximum epsilon value to search for, defaults to 1e7
    :type max_epsilon: float, optional
    :return: The epsilon value.
    :rtype: float
    """
    if noise_multiplier == 0:
        execution_logger.warning("Since noise_multiplier is 0, epsilon is INF.")
        return np.inf
    return eps_Gaussian(delta=delta, mu=np.sqrt(num_steps) / noise_multiplier, max_epsilon=max_epsilon)


def get_noise_multiplier(
    epsilon,
    num_steps,
    delta,
    min_noise_multiplier=1e-1,
    max_noise_multiplier=500,
    max_epsilon=1e7,
):
    """Get noise multiplier of Gaussian mechanism.

    :param epsilon: The epsilon value
    :type epsilon: float
    :param num_steps: The number of steps
    :type num_steps: int
    :param delta: The delta value
    :type delta: float
    :param min_noise_multiplier: The minimum noise multiplier to search for, defaults to 1e-1
    :type min_noise_multiplier: float, optional
    :param max_noise_multiplier: The maximum noise multiplier to search for, defaults to 500
    :type max_noise_multiplier: float, optional
    :param max_epsilon: The maximum epsilon value to search for, defaults to 1e7
    :type max_epsilon: float, optional
    """

    if epsilon == np.inf:
        return 0.0

    def objective(x):
        return (
            compute_epsilon(
                noise_multiplier=x,
                num_steps=num_steps,
                delta=delta,
                max_epsilon=max_epsilon,
            )
            - epsilon
        )

    output = root_scalar(objective, bracket=[min_noise_multiplier, max_noise_multiplier], method="brentq")

    if not output.converged:
        raise ValueError("Failed to converge")

    return output.root


class Gaussian(DP):
    """The Gaussian mechanism for Differential Privacy (DP) histogram."""

    def set_epsilon_and_delta(self, num_iterations, epsilon, delta, noise_multiplier):
        """Set the epsilon and delta for the Gaussian mechanism.

        :param num_iterations: The number of PE iterations
        :type num_iterations: int
        :param epsilon: The epsilon value of DP
        :type epsilon: float
        :param delta: The delta value of DP
        :type delta: float
        :param noise_multiplier: The noise multiplier of the DP mechanism
        :type noise_multiplier: float
        :raises ValueError: If delta is None
        :raises ValueError: If both epsilon and noise_multiplier are None or not None
        """
        if delta is None:
            raise ValueError("Delta should not be None")
        if (epsilon is None) == (noise_multiplier is None):
            raise ValueError("Either epsilon or noise multiplier should be None")

        self._delta = delta
        if epsilon is not None:
            self._epsilon = epsilon
            if num_iterations == 0:
                self._noise_multiplier = 0
                execution_logger.warning(
                    "Since num_iterations is 0, noise_multiplier is set to 0, and epsilon is ignored."
                )
            else:
                self._noise_multiplier = get_noise_multiplier(
                    epsilon=epsilon,
                    num_steps=num_iterations,
                    delta=delta,
                )
        else:
            self._noise_multiplier = noise_multiplier
            if num_iterations == 0:
                self._epsilon = 0
                execution_logger.warning(
                    "Since num_iterations is 0, epsilon is set to 0, and noise_multiplier is ignored."
                )
            else:
                self._epsilon = compute_epsilon(
                    noise_multiplier=noise_multiplier,
                    num_steps=num_iterations,
                    delta=delta,
                )
        execution_logger.info(
            f"DP epsilon={self._epsilon}, delta={self._delta}, noise_multiplier={self._noise_multiplier}, "
            f"num_iterations={num_iterations}."
        )

    def add_noise(self, syn_data):
        """Add noise to the histogram of synthetic data.

        :param syn_data: The synthetic data to add noise. The synthetic data should have the
            :py:const:`pe.constant.data.CLEAN_HISTOGRAM_COLUMN_NAME` column
        :type syn_data: :py:class:`pe.data.Data`
        :return: The synthetic data with noise added to the histogram. The noisy histogram is stored in the
            :py:const:`pe.constant.data.DP_HISTOGRAM_COLUMN_NAME` column
        :rtype: :py:class:`pe.data.Data`
        """
        syn_data.data_frame[DP_HISTOGRAM_COLUMN_NAME] = syn_data.data_frame[
            CLEAN_HISTOGRAM_COLUMN_NAME
        ] + np.random.normal(scale=self._noise_multiplier, size=len(syn_data.data_frame))
        return syn_data

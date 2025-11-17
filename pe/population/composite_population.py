from .population import Population
from pe.logging import execution_logger


class CompositePopulation(Population):
    """Using different population algorithms at different PE iterations."""

    def __init__(self, populations):
        """Constructor.

        :param populations: The list of populations to use at each PE iteration
        :type populations: list[:py:class:`pe.population.Population`]
        """
        super().__init__()
        self._populations = populations

    def initial(self, label_info, num_samples):
        """Generate the initial synthetic data.

        :param label_info: The label info
        :type label_info: omegaconf.dictconfig.DictConfig
        :param num_samples: The number of samples to generate
        :type num_samples: int
        :return: The initial synthetic data
        :rtype: :py:class:`pe.data.Data`
        """
        execution_logger.info(
            f"Composite population: generating {num_samples} initial synthetic samples for label {label_info.name}"
        )
        data = self._populations[0].initial(label_info=label_info, num_samples=num_samples)
        execution_logger.info(
            f"Composite population: finished generating {num_samples} initial "
            f"synthetic samples for label {label_info.name}"
        )
        return data

    def next(self, syn_data, num_samples):
        """Generate the next synthetic data.

        :param syn_data: The synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        :param num_samples: The number of samples to generate
        :type num_samples: int
        :return: The next synthetic data
        :rtype: :py:class:`pe.data.Data`
        """
        execution_logger.info(f"Composite population: generating {num_samples} next synthetic samples")
        iteration = syn_data.metadata.iteration + 1
        if iteration >= len(self._populations):
            raise ValueError(f"No population defined for iteration {iteration}")
        data = self._populations[iteration].next(syn_data=syn_data, num_samples=num_samples)
        execution_logger.info(f"Composite population: finished generating {num_samples} next synthetic samples")
        return data

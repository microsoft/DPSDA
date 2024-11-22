from abc import ABC, abstractmethod


class Logger(ABC):
    """The abstract class for logging the metrics"""

    @abstractmethod
    def log(self, iteration, metric_items):
        """Log the metrics.

        :param iteration: The PE iteration number
        :type iteration: int
        :param metric_items: The metrics to log
        :type metric_items: list[:py:class:`pe.metric_item.MetricItem`]
        """
        ...

    def clean_up(self):
        """Clean up the logger."""
        ...

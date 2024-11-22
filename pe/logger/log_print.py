from .logger import Logger
from pe.metric_item import FloatMetricItem, FloatListMetricItem
from pe.logging import execution_logger


class LogPrint(Logger):
    """The logger that prints the metrics to the console/file using :py:const:`pe.logging.execution_logger`."""

    def __init__(self, log_iteration_freq=1):
        """Constructor.

        :param log_iteration_freq: The frequency to log the metrics, defaults to 1
        :type log_iteration_freq: int, optional
        """
        self._log_iteration_freq = log_iteration_freq

    def log(self, iteration, metric_items):
        """Log the metrics to the console/file.

        :param iteration: The PE iteration number
        :type iteration: int
        :param metric_items: The metrics to log
        :type metric_items: list[:py:class:`pe.metric_item.FloatMetricItem` or
            :py:class:`pe.metric_item.FloatListMetricItem`]
        """
        if iteration % self._log_iteration_freq != 0:
            return
        metric_items = [item for item in metric_items if isinstance(item, (FloatMetricItem, FloatListMetricItem))]
        if len(metric_items) == 0:
            return
        execution_logger.info(f"Iteration: {iteration}")
        for item in metric_items:
            if isinstance(item, FloatMetricItem):
                value = [item.value]
            else:
                value = item.value
            value = ",".join([f"{v:.8f}" for v in value])
            execution_logger.info(f"\t{item.name}: {value}")

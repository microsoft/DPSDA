import os
import csv
import torch
import numpy as np
from collections import defaultdict

from .logger import Logger
from pe.metric_item import FloatMetricItem
from pe.metric_item import FloatListMetricItem


class CSVPrint(Logger):
    """The logger that prints the metrics to CSV files."""

    def __init__(
        self,
        output_folder,
        path_separator="-",
        float_format=".8f",
        flush_iteration_freq=1,
    ):
        """Constructor.

        :param output_folder: The output folder that will be used to save the CSV files
        :type output_folder: str
        :param path_separator: The string that will be used to replace '\' and '/' in log names, defaults to "-"
        :type path_separator: str, optional
        :param float_format: The format of the floating point numbers, defaults to ".8f"
        :type float_format: str, optional
        :param flush_iteration_freq: The frequency to flush the logs, defaults to 1
        :type flush_iteration_freq: int, optional
        """
        self._output_folder = output_folder
        os.makedirs(self._output_folder, exist_ok=True)
        self._path_separator = path_separator
        self._float_format = float_format
        self._flush_iteration_freq = flush_iteration_freq
        self._clear_logs()

    def _clear_logs(self):
        """Clear the logs."""
        self._logs = defaultdict(list)

    def _get_log_path(self, iteration, item):
        """Get the log path.

        :param iteration: The PE iteration number
        :type iteration: int
        :param item: The metric item
        :type item: :py:class:`pe.metric_item.MetricItem`
        :return: The log path
        :rtype: str
        """
        log_path = item.name
        log_path = log_path.replace("/", self._path_separator)
        log_path = log_path.replace("\\", self._path_separator)
        log_path = os.path.join(self._output_folder, log_path + ".csv")
        return log_path

    def _flush(self):
        """Flush the logs."""
        for path in self._logs:
            with open(path, "a") as f:
                writer = csv.writer(f)
                writer.writerows(self._logs[path])

    def _log_float(self, log_path, iteration, item):
        """Log a float metric item.

        :param log_path: The path of the log file
        :type log_path: str
        :param iteration: The PE iteration number
        :type iteration: int
        :param item: The float metric item
        :type item: :py:class:`pe.metric_item.FloatMetricItem` or :py:class:`pe.metric_item.FloatListMetricItem`
        """
        str_iteration = str(iteration)
        str_value = item.value
        if isinstance(item.value, torch.Tensor):
            str_value = item.value.cpu().detach().numpy()
        if isinstance(str_value, np.ndarray):
            str_value = str_value.tolist()
        if isinstance(str_value, list):
            str_value = ",".join([format(v, self._float_format) for v in str_value])
        else:
            str_value = format(str_value, self._float_format)
        self._logs[log_path].append([str_iteration, str_value])

    def log(self, iteration, metric_items):
        """Log the metrics.

        :param iteration: The PE iteration number
        :type iteration: int
        :param metric_items: The metrics to log
        :type metric_items: list[:py:class:`pe.metric_item.MetricItem`]
        """
        for item in metric_items:
            if not isinstance(item, (FloatMetricItem, FloatListMetricItem)):
                continue
            log_path = self._get_log_path(iteration, item)
            self._log_float(log_path, iteration, item)
        if iteration % self._flush_iteration_freq == 0:
            self._flush()
            self._clear_logs()

    def clean_up(self):
        """Clean up the logger."""
        self._flush()
        self._clear_logs()

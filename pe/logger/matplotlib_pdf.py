import os

from .logger import Logger
from pe.metric_item import MatplotlibMetricItem


class MatplotlibPDF(Logger):
    """The logger that saves Matplotlib figures to PDF files."""

    def __init__(
        self,
        output_folder,
        path_separator="-",
        iteration_format="09d",
    ):
        """Constructor.

        :param output_folder: The output folder that will be used to save the PDF files
        :type output_folder: str
        :param path_separator: The string that will be used to replace '\' and '/' in log names, defaults to "-"
        :type path_separator: str, optional
        :param iteration_format: The format of the iteration number, defaults to "09d"
        :type iteration_format: str, optional
        """
        self._output_folder = output_folder
        os.makedirs(self._output_folder, exist_ok=True)
        self._path_separator = path_separator
        self._iteration_format = iteration_format

    def log(self, iteration, metric_items):
        """Log the Matplotlib figures to PDF files.

        :param iteration: The PE iteration number
        :type iteration: int
        :param metric_items: The Matplotlib figures to log
        :type metric_items: list[:py:class:`pe.metric_item.MatplotlibMetricItem`]
        """
        for item in metric_items:
            if not isinstance(item, (MatplotlibMetricItem,)):
                continue
            pdf_path = self._get_pdf_path(iteration, item)
            item.value.savefig(pdf_path)

    def _get_pdf_path(self, iteration, item):
        """Get the PDF save path.

        :param iteration: The PE iteration number
        :type iteration: int
        :param item: The Matplotlib figure metric item
        :type item: :py:class:`pe.metric_item.MatplotlibMetricItem`
        :return: The PDF save path
        :rtype: str
        """
        image_name = item.name
        image_name = image_name.replace("/", self._path_separator)
        image_name = image_name.replace("\\", self._path_separator)
        image_folder = os.path.join(self._output_folder, image_name)
        os.makedirs(image_folder, exist_ok=True)
        iteration_string = format(iteration, self._iteration_format)
        image_file_name = f"{iteration_string}.pdf"
        image_path = os.path.join(
            image_folder,
            image_file_name,
        )
        return image_path

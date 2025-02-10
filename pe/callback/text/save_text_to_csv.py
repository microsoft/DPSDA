import os
import pandas as pd

from pe.callback.callback import Callback
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.constant.data import TEXT_DATA_COLUMN_NAME
from pe.logging import execution_logger


class SaveTextToCSV(Callback):
    """The callback that saves the synthetic text to a CSV file."""

    def __init__(
        self,
        output_folder,
        iteration_format="09d",
    ):
        """Constructor.

        :param output_folder: The output folder that will be used to save the CSV files
        :type output_folder: str
        :param iteration_format: The format of the iteration part of the CSV paths, defaults to "09d"
        :type iteration_format: str, optional
        """
        self._output_folder = output_folder
        self._iteration_format = iteration_format

    def _get_csv_path(self, iteration):
        """Get the CSV path.

        :param iteration: The PE iteration number
        :type iteration: int
        :return: The CSV path
        :rtype: str
        """
        os.makedirs(self._output_folder, exist_ok=True)
        iteration_string = format(iteration, self._iteration_format)
        csv_path = os.path.join(
            self._output_folder,
            f"{iteration_string}.csv",
        )
        return csv_path

    def __call__(self, syn_data):
        """This function is called after each PE iteration that saves the synthetic text to a CSV file.

        :param syn_data: The :py:class:`pe.data.Data` object of the synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        """
        execution_logger.info("Saving the synthetic text to a CSV file")
        samples = syn_data.data_frame[TEXT_DATA_COLUMN_NAME].tolist()
        label_ids = syn_data.data_frame[LABEL_ID_COLUMN_NAME].tolist()
        columns = {syn_data.metadata.text_column: samples}
        for i in range(len(syn_data.metadata.label_columns)):
            column_name = syn_data.metadata.label_columns[i]
            columns[column_name] = [
                syn_data.metadata.label_info[label_id].column_values[column_name] for label_id in label_ids
            ]

        data_frame = pd.DataFrame(columns)
        csv_path = self._get_csv_path(syn_data.metadata.iteration)
        data_frame.to_csv(csv_path, index=False)

        execution_logger.info("Finished saving the synthetic text to a CSV file")

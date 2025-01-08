from pe.data import Data
import pandas as pd
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.constant.data import TEXT_DATA_COLUMN_NAME


class TextCSV(Data):
    """The text dataset in CSV format."""

    def __init__(self, csv_path, label_columns=[], text_column="text", num_samples=None):
        """Constructor.

        :param csv_path: The path to the CSV file
        :type csv_path: str
        :param label_columns: The names of the columns that contain the labels, defaults to []
        :type label_columns: list, optional
        :param text_column: The name of the column that contains the text data, defaults to "text"
        :type text_column: str, optional
        :param num_samples: The number of samples to load from the CSV file. If None, load all samples. Defaults to
            None
        :type num_samples: int, optional
        :raises ValueError: If the label columns or text column does not exist in the CSV file
        """
        data_frame = pd.read_csv(csv_path, dtype=str)
        if num_samples is not None:
            data_frame = data_frame[:num_samples]
        for column in label_columns + [text_column]:
            if column not in data_frame.columns:
                raise ValueError(f"Column {column} does not exist in the CSV file")
        labels = data_frame.apply(lambda row: tuple([row[col] for col in label_columns]), axis=1).tolist()
        label_set = list(sorted(set(labels)))
        label_id_map = {label: i for i, label in enumerate(label_set)}
        label_ids = [label_id_map[label] for label in labels]
        data_frame[LABEL_ID_COLUMN_NAME] = label_ids
        label_info = [
            {
                "name": " | ".join(f"{label_columns[i]}: {label[i]}" for i in range(len(label_columns))),
                "column_values": {label_columns[i]: label[i] for i in range(len(label_columns))},
            }
            for label in label_set
        ]
        metadata = {"label_columns": label_columns, "text_column": text_column, "label_info": label_info}
        data_frame = data_frame.rename(columns={text_column: TEXT_DATA_COLUMN_NAME})
        super().__init__(data_frame=data_frame, metadata=metadata)

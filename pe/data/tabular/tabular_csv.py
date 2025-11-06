from pe.data import Data
import pandas as pd
import numpy as np
import json, requests
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.constant.data import TABULAR_DATA_COLUMN_NAME
import sys
import math
from enum import Enum


class TabularColumnType(Enum):
    """The type of the tabular column."""

    CATEGORICAL = "cat"
    INTEGER = "int"
    FLOAT = "float"


class TabularCSV(Data):
    """The tabular dataset in CSV format."""

    def __init__(self, csv_path, metadata_path):
        """Constructor.

        :param csv_path: The path to the CSV file or the URL to the CSV file
        :type csv_path: str
        :param metadata_path: The path to the metadata file or the URL to the metadata file
        :type metadata_path: str
        """
        raw_tab_df = pd.read_csv(csv_path)  # pd.reac_csv can work with both local path and http path

        if "https://" in metadata_path:
            metadata = json.loads(requests.get(metadata_path).text)  # read metadata from http path
        else:
            with open(metadata_path, "r") as file:
                metadata = json.load(file)  # read metadata from local path

        cat_columns = metadata["cat_columns"]
        int_columns = metadata["int_columns"]
        float_columns = metadata["float_columns"]
        label_columns = metadata["label_columns"]

        # Create label IDs following the same logic as TextCSV
        labels = raw_tab_df.apply(lambda row: tuple([row[col] for col in label_columns]), axis=1).tolist()
        label_set = list(sorted(set(labels)))
        label_id_map = {label: i for i, label in enumerate(label_set)}
        label_ids = [label_id_map[label] for label in labels]

        def to_builtin(x):
            if pd.isna(x):
                return None
            if isinstance(x, np.generic):
                return x.item()
            return x

        label_info = [
            {
                "name": " | ".join(f"{label_columns[i]}: {to_builtin(label[i])}" for i in range(len(label_columns))),
                "column_values": {label_columns[i]: to_builtin(label[i]) for i in range(len(label_columns))},
            }
            for label in label_set
        ]

        # Merge tabular features into a single column
        feature_columns = cat_columns + int_columns + float_columns
        merged_features = raw_tab_df[feature_columns].apply(lambda row: row.tolist(), axis=1)

        data_frame = pd.DataFrame(
            {
                TABULAR_DATA_COLUMN_NAME: merged_features,
                LABEL_ID_COLUMN_NAME: label_ids,
            }
        )

        metadata = {
            "label_columns": label_columns,
            "label_info": label_info,
            "cat_columns": cat_columns,
            "int_columns": int_columns,
            "float_columns": float_columns,
            "feature_columns": feature_columns,
        }

        super().__init__(data_frame=data_frame, metadata=metadata)

    def get_tab_info(self):
        """Get the information of the private data.

        :param priv_data: The data object containing the training tabular data
        :type priv_data: :py:class:`pe.data.Data`
        :return: The information (categories and numerical bounds) of the private data
        :rtype: dict
        """

        info = {}
        features_columns = self.metadata["feature_columns"]
        features_df = pd.DataFrame(self.data_frame[TABULAR_DATA_COLUMN_NAME].tolist(), columns=features_columns)
        for column in features_columns:
            if column in self.metadata["cat_columns"]:
                info[column] = {
                    "categories": list(features_df[column].unique()),
                    "type": TabularColumnType.CATEGORICAL,
                }
            elif column in self.metadata["int_columns"]:
                info[column] = {
                    "min": math.floor(features_df[column].min()),
                    "max": math.ceil(features_df[column].max()),
                    "type": TabularColumnType.INTEGER,
                }
            elif column in self.metadata["float_columns"]:
                info[column] = {
                    "min": features_df[column].min(),
                    "max": features_df[column].max(),
                    "type": TabularColumnType.FLOAT,
                }

        return info

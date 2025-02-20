import os
from omegaconf import OmegaConf
import pandas as pd
import numpy as np
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.constant.data import CLIENT_ID_COLUMN_NAME


class Data:
    """The class that holds the private data or synthetic data from PE."""

    def __init__(self, data_frame=None, metadata={}):
        """Constructor.

        :param data_frame: A pandas dataframe that holds the data, defaults to None
        :type data_frame: :py:class:`pandas.DataFrame`, optional
        :param metadata: the metadata of the data, defaults to {}
        :type metadata: dict, optional
        """
        self.data_frame = data_frame
        self.metadata = OmegaConf.create(metadata)
        self._data_frame_file_name = "data_frame.pkl"
        self._metadata_file_name = "metadata.yaml"

    def __str__(self):
        return f"Metadata:\n{self.metadata}\nData frame:\n{self.data_frame}"

    def save_checkpoint(self, path):
        """Save the data to a checkpoint.

        :param path: The folder to save the checkpoint
        :type path: str
        :raises ValueError: If the path is None
        :raises ValueError: If the data frame is empty
        """
        if path is None:
            raise ValueError("Path is None")
        if self.data_frame is None:
            raise ValueError("Data frame is empty")
        os.makedirs(path, exist_ok=True)
        self.data_frame.to_pickle(os.path.join(path, self._data_frame_file_name))
        with open(os.path.join(path, self._metadata_file_name), "w") as file:
            file.write(OmegaConf.to_yaml(self.metadata))

    def load_checkpoint(self, path):
        """Load data from a checkpoint

        :param path: The folder that contains the checkpoint
        :type path: str
        :return: Whether the checkpoint is loaded successfully
        :rtype: bool
        """
        data_frame_path = os.path.join(path, self._data_frame_file_name)
        metadata_path = os.path.join(path, self._metadata_file_name)
        if not os.path.exists(data_frame_path) or not os.path.exists(metadata_path):
            return False
        self.data_frame = pd.read_pickle(data_frame_path)
        with open(metadata_path, "r") as file:
            self.metadata = OmegaConf.create(file.read())
        return True

    def filter_label_id(self, label_id):
        """Filter the data frame according to a label id

        :param label_id: The label id that is used to filter the data frame
        :type label_id: int
        :return: :py:class:`pe.data.Data` object with the filtered data frame
        :rtype: :py:class:`pe.data.Data`
        """
        return Data(
            data_frame=self.data_frame[self.data_frame[LABEL_ID_COLUMN_NAME] == label_id],
            metadata=self.metadata,
        )

    def set_label_id(self, label_id):
        """Set the label id for the data frame

        :param label_id: The label id to set
        :type label_id: int
        """
        self.data_frame[LABEL_ID_COLUMN_NAME] = label_id

    def truncate(self, num_samples):
        """Truncate the data frame to a certain number of samples

        :param num_samples: The number of samples to truncate
        :type num_samples: int
        :return: A new :py:class:`pe.data.Data` object with the truncated data frame
        :rtype: :py:class:`pe.data.Data`
        """
        return Data(data_frame=self.data_frame[:num_samples], metadata=self.metadata)

    def random_truncate(self, num_samples):
        """Randomly truncate the data frame to a certain number of samples

        :param num_samples: The number of samples to randomly truncate
        :type num_samples: int
        :return: A new :py:class:`pe.data.Data` object with the randomly truncated data frame
        :rtype: :py:class:`pe.data.Data`
        """
        data_frame = self.data_frame.sample(n=num_samples)
        return Data(data_frame=data_frame, metadata=self.metadata)

    def random_split(self, num_samples_list, seed=0):
        """Randomly split the data frame into multiple data frames

        :param num_samples_list: The list of numbers of samples for each data frame
        :type num_samples_list: list[int]
        :param seed: The seed for the random number generator, defaults to 0
        :type seed: int, optional
        :raises ValueError: If the sum of num_samples_list is not equal to the number of samples
        :return: The list of :py:class:`pe.data.Data` objects with the splited data
        :rtype: list[:py:class:`pe.data.Data`]
        """
        if sum(num_samples_list) != len(self.data_frame):
            raise ValueError("The sum of num_samples_list must be equal to the number of samples")
        data_frame_list = (
            self.data_frame.sample(frac=1, random_state=seed)
            .reset_index(drop=True)
            .groupby(
                pd.cut(
                    range(len(self.data_frame)),
                    bins=[0] + list(np.cumsum(num_samples_list)),
                    right=False,
                    labels=False,
                )
            )
        )
        return [
            Data(data_frame=data_frame.reset_index(drop=True), metadata=self.metadata)
            for _, data_frame in data_frame_list
        ]

    def merge(self, data):
        """Merge the data object with another data object

        :param data: The data object to merge
        :type data: :py:class:`pe.data.Data`
        :raises ValueError: If the metadata of `data` is not the same as the metadata of the current object
        :return: The merged data object
        :rtype: :py:class:`pe.data.Data`
        """
        if self.metadata != data.metadata:
            raise ValueError("Metadata must be the same")
        cols_to_use = data.data_frame.columns.difference(self.data_frame.columns)
        if len(cols_to_use) == 0:
            return self
        data_frame = self.data_frame.join(data.data_frame[cols_to_use])
        return Data(data_frame=data_frame, metadata=self.metadata)

    def filter(self, filter_criteria):
        """Filter the data object according to a filter criteria

        :param filter_criteria: The filter criteria. None means no filter
        :type filter_criteria: dict
        :return: The filtered data object
        :rtype: :py:class:`pe.data.Data`
        """
        if filter_criteria is None:
            return self
        data_frame = self.data_frame
        for column, value in filter_criteria.items():
            data_frame = data_frame[data_frame[column] == value]
        return Data(data_frame=data_frame, metadata=self.metadata)

    @classmethod
    def concat(cls, data_list, metadata=None):
        """Concatenate the data frames of a list of data objects

        :param data_list: The list of data objects to concatenate
        :type data_list: list[:py:class:`pe.data.Data`]
        :param metadata: The metadata of the concatenated data. When None, the metadata of the list of data objects
            must be the same and will be used. Defaults to None
        :type metadata: dict, optional
        :raises ValueError: If the metadata of the data objects are not the same
        :return: The concatenated data object
        :rtype: :py:class:`pe.data.Data`
        """
        data_frame_list = [data.data_frame for data in data_list]
        if metadata is None:
            metadata_list = [data.metadata for data in data_list]
            # Check that all metadata are the same.
            if len(set(metadata_list)) != 1:
                raise ValueError("Metadata must be the same")
            metadata = metadata_list[0]
        return Data(data_frame=pd.concat(data_frame_list), metadata=metadata)

    def split_by_client(self):
        """Split the data frame by client ID

        :raises ValueError: If the client ID column is not in the data frame
        :return: The list of data objects with the splited data
        :rtype: list[:py:class:`pe.data.Data`]
        """
        if CLIENT_ID_COLUMN_NAME not in self.data_frame.columns:
            raise ValueError(f"{CLIENT_ID_COLUMN_NAME} not in data frame")
        grouped_data_frame = self.data_frame.groupby(CLIENT_ID_COLUMN_NAME)
        return [Data(data_frame=data_frame, metadata=self.metadata) for _, data_frame in grouped_data_frame]

    def split_by_index(self):
        """Split the data frame by index

        :return: The list of data objects with the splited data
        :rtype: list[:py:class:`pe.data.Data`]
        """
        grouped_data_frame = self.data_frame.groupby(self.data_frame.index)
        return [Data(data_frame=data_frame, metadata=self.metadata) for _, data_frame in grouped_data_frame]

    def reset_index(self, **kwargs):
        """Reset the index of the data frame

        :param kwargs: The keyword arguments to pass to the pandas reset_index function
        :type kwargs: dict
        :return: A new :py:class:`pe.data.Data` object with the reset index data frame
        :rtype: :py:class:`pe.data.Data`
        """
        return Data(data_frame=self.data_frame.reset_index(**kwargs), metadata=self.metadata)

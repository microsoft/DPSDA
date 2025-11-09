import pandas as pd
import numpy as np

from pe.api import API
from pe.logging import execution_logger
from pe.data import Data
from pe.constant.data import TABULAR_DATA_COLUMN_NAME
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.data import TabularColumnType


class TabularAPI(API):
    """The tabular API that perturbs the original tabular data to generate variations of the synthetic data."""

    def __init__(
        self,
        info: dict,
        mutation_rate_init: float = 0.5,
        mutation_rate_final: float = 0.01,
        decay_type: str = "polynomial",
        gamma: float = 0.2,
        num_iterations: int = 15,
    ):
        """Constructor.

        :param info: The information (categories and numerical bounds) of the private data
        :type info: dict
        :param mutation_rate_init: The initial mutation rate, defaults to 0.5
        :type mutation_rate_init: float, optional
        :param mutation_rate_final: The final mutation rate, defaults to 0.01
        :type mutation_rate_final: float, optional
        :param decay_type: The type of decay, defaults to "polynomial"
        :type decay_type: str, optional
        :param gamma: The gamma parameter for the polynomial decay, defaults to 0.2
        :type gamma: float, optional
        :param num_iterations: The number of PE iterations, defaults to 15
        :type num_iterations: int, optional
        """
        super().__init__()

        self._info = info
        self._mutation_rate_init = mutation_rate_init
        self._mutation_rate_final = mutation_rate_final
        self._decay_type = decay_type
        self._gamma = gamma
        self._num_iterations = num_iterations

    def random_api(self, label_info, num_samples) -> Data:
        """Generating random synthetic data.

        :param label_info: The info of the label
        :type label_info: omegaconf.dictconfig.DictConfig
        :param num_samples: The number of random samples to generate
        :type num_samples: int
        :return: The data object of the generated synthetic data
        :rtype: :py:class:`pe.data.Data`
        """
        label_name = label_info.name
        execution_logger.info(f"RANDOM API: creating {num_samples} samples for label {label_name}")
        metadata = {"label_info": [label_info]}
        feature_columns = list(self._info.keys())

        # Vectorization per column
        column_data = {}
        for column in feature_columns:
            if self._info[column]["type"] == TabularColumnType.CATEGORICAL:
                column_data[column] = np.random.choice(self._info[column]["categories"], size=num_samples)
            elif self._info[column]["type"] == TabularColumnType.INTEGER:
                column_data[column] = np.random.randint(
                    int(self._info[column]["min"]), int(self._info[column]["max"]), size=num_samples
                )
            elif self._info[column]["type"] == TabularColumnType.FLOAT:
                column_data[column] = np.random.uniform(
                    self._info[column]["min"], self._info[column]["max"], size=num_samples
                )
            else:
                raise ValueError(f"Invalid type: {self._info[column]['type']}")

        # Combine columns into rows
        rows = np.column_stack([column_data[column] for column in feature_columns]).tolist()

        data_frame = pd.DataFrame({TABULAR_DATA_COLUMN_NAME: rows, LABEL_ID_COLUMN_NAME: 0})
        execution_logger.info(f"RANDOM API: finished creating {num_samples} samples for label {label_name}")
        return Data(data_frame=data_frame, metadata=metadata)

    def _get_mutation_rate(self, iteration) -> float:
        """Get the mutation rate for the given iteration.

        :param iteration: The iteration
        :type iteration: int
        :return: The mutation rate for the given iteration
        :rtype: float
        """
        if iteration <= 0:
            return self._mutation_rate_init
        elif iteration >= self._num_iterations:
            return self._mutation_rate_final

        t = iteration / self._num_iterations
        base = self._mutation_rate_init
        floor = self._mutation_rate_final
        if self._decay_type == "polynomial":
            mutation_rate = base - (base - floor) * (t**self._gamma)
            return mutation_rate
        elif self._decay_type == "linear":
            mutation_rate = base - (base - floor) * t
            return mutation_rate
        else:
            raise ValueError(f"Invalid decay type: {self._decay_type}")

    def variation_api(self, syn_data) -> Data:
        """Generating variations of the synthetic data.

        :param syn_data: The data object of the synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        :return: The data object of the variation of the input synthetic data
        :rtype: :py:class:`pe.data.Data`
        """
        execution_logger.info(f"VARIATION API: creating variations for {len(syn_data.data_frame)} samples")
        feature_columns = self._info.keys()
        features_df = pd.DataFrame(syn_data.data_frame[TABULAR_DATA_COLUMN_NAME].tolist(), columns=feature_columns)
        label_ids = syn_data.data_frame[LABEL_ID_COLUMN_NAME].tolist()

        iteration = getattr(syn_data.metadata, "iteration", -1)
        mutation_rate = self._get_mutation_rate(iteration)

        # Vectorization per column
        for column in feature_columns:
            if self._info[column]["type"] == TabularColumnType.CATEGORICAL:
                mask = np.random.rand(len(features_df)) < mutation_rate
                if mask.any():
                    new_values = np.random.choice(self._info[column]["categories"], size=mask.sum())
                    features_df.loc[mask, column] = new_values
            elif self._info[column]["type"] in [TabularColumnType.INTEGER, TabularColumnType.FLOAT]:
                current_values = features_df[column].to_numpy()
                feature_min = self._info[column]["min"]
                feature_max = self._info[column]["max"]
                feature_range = feature_max - feature_min
                deltas = np.random.uniform(-mutation_rate, mutation_rate, size=len(features_df)) * feature_range
                updated_values = current_values + deltas

                if self._info[column]["type"] == TabularColumnType.INTEGER:  # round to nearest integer
                    updated_values = np.round(updated_values)
                # clamp to [min, max]
                updated_values = np.clip(updated_values, feature_min, feature_max)
                features_df[column] = updated_values
            else:
                raise ValueError(f"Invalid column type: {self._info[column]['type']}")

        data_frame = pd.DataFrame(
            {
                TABULAR_DATA_COLUMN_NAME: features_df.values.tolist(),
                LABEL_ID_COLUMN_NAME: label_ids,
            }
        )
        execution_logger.info(f"VARIATION API: finished creating variations for {len(syn_data.data_frame)} samples")
        return Data(data_frame=data_frame, metadata=syn_data.metadata)

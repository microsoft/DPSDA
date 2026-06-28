import ot
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import LabelEncoder

from pe.callback.callback import Callback
from pe.metric_item import FloatMetricItem
from pe.logging import execution_logger
from pe.constant.data import TABULAR_DATA_COLUMN_NAME
from pe.constant.data import LABEL_ID_COLUMN_NAME


class ComputeWSD(Callback):
    """The callback that computes the Wasserstein Distance (WSD) between the private and synthetic data."""

    def __init__(self, priv_data, degree, num_samples=None, seed=42, filter_criterion=None):
        """Constructor.

        :param priv_data: The private data
        :type priv_data: :py:class:`pe.data.Data`
        :param degree: The degree of the WSD (e.g., 2 for 2-way WSD)
        :type degree: int
        :param num_samples: The number of samples to use for the WSD for both private and synthetic data for computation efficiency. If None, all samples are used..
        :type num_samples: int, optional
        :param seed: The seed to use for for sampling the data.
        :type seed: int, optional
        :param filter_criterion: Only computes the metric based on samples satisfying the criterion. None means no
            filtering. Defaults to None
        :type filter_criterion: dict, optional
        """
        self._priv_data = priv_data
        self._filter_criterion = filter_criterion
        self._filter_criterion_str = str(filter_criterion).replace(" ", "")
        self._degree = degree
        self._num_samples = num_samples
        self._seed = seed
        self._metric_name = (
            f"{degree}way-wsd_{self._num_samples}samples_{self._seed}seed_{self._filter_criterion_str}"
            if filter_criterion
            else f"{degree}way-wsd_{self._num_samples}samples_{self._seed}seed"
        )
        self._cat_columns = priv_data.metadata["cat_columns"]
        self._int_columns = priv_data.metadata["int_columns"]
        self._float_columns = priv_data.metadata["float_columns"]
        self._label_columns = priv_data.metadata["label_columns"]
        self._feature_columns = self._cat_columns + self._int_columns + self._float_columns
        self._priv_features_df = self._get_features_df(priv_data)

    def _get_features_df(self, data):
        """Get the features DataFrame from the data.

        :param data: The data
        :type data: :py:class:`pe.data.Data`
        :return: The features DataFrame
        :rtype: :py:class:`pandas.DataFrame`
        """
        if self._num_samples is not None and self._num_samples < len(data.data_frame):
            data, _ = data.random_split([self._num_samples, len(data.data_frame) - self._num_samples], self._seed)
        label_ids = data.data_frame[LABEL_ID_COLUMN_NAME].tolist()
        features_df = pd.DataFrame(data.data_frame[TABULAR_DATA_COLUMN_NAME].tolist(), columns=self._feature_columns)
        for i in range(len(data.metadata.label_columns)):
            column_name = data.metadata.label_columns[i]
            features_df[column_name] = [
                data.metadata.label_info[label_id].column_values[column_name] for label_id in label_ids
            ]
        return features_df

    def _compute_wsd(self, syn_features_df, priv_features_df):
        """Compute the multiple-way WSD between the synthetic and private features.

        :param syn_features_df: The synthetic features DataFrame
        :type syn_features_df: :py:class:`pandas.DataFrame`
        :param priv_features_df: The private features DataFrame
        :type priv_features_df: :py:class:`pandas.DataFrame`
        :return: The multiple-way WSD
        :rtype: float
        """
        df1 = syn_features_df.copy()
        # Sample the synthetic data if num_samples is provided
        if self._num_samples is not None and self._num_samples < len(df1):
            df1 = df1.sample(n=self._num_samples, random_state=self._seed)
        df2 = priv_features_df.copy()

        all_cols = self._feature_columns + self._label_columns

        # Encode categorical and label columns numerically
        for col in self._cat_columns + self._label_columns:
            le = LabelEncoder()
            all_values = pd.concat([df2[col], df1[col]]).dropna().unique()
            le.fit(all_values)
            df1[col] = le.transform(df1[col].astype(str))
            df2[col] = le.transform(df2[col].astype(str))

        # Normalize all columns to [0, 1] using private data range
        for col in all_cols:
            col_min = df2[col].min()
            col_max = df2[col].max()
            col_range = col_max - col_min
            if col_range == 0:
                col_range = 1.0
            df1[col] = (df1[col] - col_min) / col_range
            df2[col] = (df2[col] - col_min) / col_range

        combos = list(combinations(all_cols, self._degree))

        if not combos:
            return 0.0

        wasserstein_distances = []
        for group in combos:
            group_list = list(group)

            values1 = df1[group_list].values
            values2 = df2[group_list].values

            # Remove rows with NaN values
            mask1 = ~np.isnan(values1).any(axis=1)
            mask2 = ~np.isnan(values2).any(axis=1)
            values1 = values1[mask1]
            values2 = values2[mask2]

            if len(values1) == 0 or len(values2) == 0:
                continue

            if self._degree == 1:
                wd = wasserstein_distance(values1.flatten(), values2.flatten())
            else:
                M = ot.dist(values1, values2, metric="euclidean")
                a = np.ones(len(values1)) / len(values1)
                b = np.ones(len(values2)) / len(values2)
                wd = ot.emd2(a, b, M)

            wasserstein_distances.append(wd)

        if not wasserstein_distances:
            return 0.0

        return sum(wasserstein_distances) / len(wasserstein_distances)

    def __call__(self, syn_data):
        """This function is called after each PE iteration that computes the multiple-way WSD between the private and
        synthetic data.

        :param syn_data: The synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        :return: The multiple-way WSD between the private and synthetic data
        :rtype: list[:py:class:`pe.metric_item.FloatMetricItem`]
        """
        execution_logger.info(f"Computing {self._degree}way-WSD ({self._filter_criterion_str})")
        syn_data = syn_data.filter(self._filter_criterion)
        execution_logger.info(f"Number of samples after filtering: {len(syn_data.data_frame)}")
        if len(syn_data.data_frame) == 0:
            execution_logger.warning(
                f"No samples satisfy the filter criterion {self._filter_criterion_str}. Skipping computation."
            )
            return []
        syn_features_df = self._get_features_df(syn_data)
        wsd = self._compute_wsd(syn_features_df, self._priv_features_df)
        metric_item = FloatMetricItem(name=self._metric_name, value=wsd)
        execution_logger.info(f"Finished computing {self._degree}way-WSD ({self._filter_criterion_str})")
        return [metric_item]

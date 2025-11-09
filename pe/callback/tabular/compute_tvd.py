import numpy as np
from pe.callback.callback import Callback
from pe.metric_item import FloatMetricItem
from pe.logging import execution_logger
import pandas as pd
from pe.constant.data import TABULAR_DATA_COLUMN_NAME
from pe.constant.data import LABEL_ID_COLUMN_NAME
from itertools import combinations


class ComputeTVD(Callback):
    """The callback that computes the Total Variation Distance (TVD) between the private and synthetic data."""

    def __init__(self, priv_data, degree, num_bins=20, filter_criterion=None):
        """Constructor.

        :param priv_data: The private data
        :type priv_data: :py:class:`pe.data.Data`
        :param degree: The degree of the TVD (e.g., 2 for 2-way TVD)
        :type degree: int
        :param num_bins: The number of bins to compute the TVD, defaults to 20
        :type num_bins: int, optional
        :param filter_criterion: Only computes the metric based on samples satisfying the criterion. None means no
            filtering. Defaults to None
        :type filter_criterion: dict, optional
        """
        self._priv_data = priv_data
        self._filter_criterion = filter_criterion
        self._filter_criterion_str = str(filter_criterion).replace(" ", "")
        self._degree = degree
        self._num_bins = num_bins
        self._metric_name = (
            f"{degree}way-tvd_{num_bins}bins_{self._filter_criterion_str}"
            if filter_criterion
            else f"{degree}way-tvd_{num_bins}bins"
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
        label_ids = data.data_frame[LABEL_ID_COLUMN_NAME].tolist()
        features_df = pd.DataFrame(data.data_frame[TABULAR_DATA_COLUMN_NAME].tolist(), columns=self._feature_columns)
        # merge label columns into features DataFrame
        for i in range(len(data.metadata.label_columns)):
            column_name = data.metadata.label_columns[i]
            features_df[column_name] = [
                data.metadata.label_info[label_id].column_values[column_name] for label_id in label_ids
            ]
        return features_df

    def _compute_tvd(self, syn_features_df, priv_features_df):
        """Compute the TVD between the synthetic and private features.

        :param syn_features_df: The synthetic features DataFrame
        :type syn_features_df: :py:class:`pandas.DataFrame`
        :param priv_features_df: The private features DataFrame
        :type priv_features_df: :py:class:`pandas.DataFrame`
        :return: The TVD
        :rtype: float
        """
        df1 = syn_features_df.copy()
        df2 = priv_features_df.copy()

        for col in self._int_columns + self._float_columns:
            # Use private data range for binning
            col_min = df2[col].min()
            col_max = df2[col].max()
            # Handle edge case where column is constant
            if col_min == col_max:
                edges = np.array([col_min, col_max + 1e-10])
            else:
                edges = np.linspace(col_min, col_max, self._num_bins)
            df1[col] = pd.cut(df1[col], bins=edges, include_lowest=True).astype("category")
            df2[col] = pd.cut(df2[col], bins=edges, include_lowest=True).astype("category")

        for col in self._cat_columns + self._label_columns:
            # Use private data categories
            all_categories = df2[col].dropna().unique()
            cat_type = pd.api.types.CategoricalDtype(categories=all_categories, ordered=True)
            df1[col] = df1[col].astype(cat_type)
            df2[col] = df2[col].astype(cat_type)

        combos = list(combinations(self._feature_columns + self._label_columns, self._degree))

        if not combos:
            return 0.0

        tvds = []
        for group in combos:
            group_list = list(group)
            p = df1.value_counts(subset=group_list, normalize=True, sort=False)
            q = df2.value_counts(subset=group_list, normalize=True, sort=False)
            union_index = p.index.union(q.index)
            p_aligned = p.reindex(union_index, fill_value=0.0)
            q_aligned = q.reindex(union_index, fill_value=0.0)
            tvd = 0.5 * (p_aligned - q_aligned).abs().sum()
            tvds.append(tvd)

        return sum(tvds) / len(tvds)

    def __call__(self, syn_data):
        """This function is called after each PE iteration that computes the TVD between the private and
        synthetic data.

        :param syn_data: The synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        :return: The TVD between the private and synthetic data
        :rtype: list[:py:class:`pe.metric_item.FloatMetricItem`]
        """
        execution_logger.info(f"Computing {self._degree}way-TVD ({self._filter_criterion_str})")
        syn_data = syn_data.filter(self._filter_criterion)
        execution_logger.info(f"Number of samples after filtering: {len(syn_data.data_frame)}")
        syn_features_df = self._get_features_df(syn_data)
        tvd = self._compute_tvd(syn_features_df, self._priv_features_df)
        metric_item = FloatMetricItem(name=self._metric_name, value=tvd)
        execution_logger.info(f"Finished computing {self._degree}way-TVD ({self._filter_criterion_str})")
        return [metric_item]

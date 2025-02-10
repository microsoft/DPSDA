import numpy as np

from .population import Population
from pe.data import Data
from pe.constant.data import DP_HISTOGRAM_COLUMN_NAME
from pe.constant.data import POST_PROCESSED_DP_HISTOGRAM_COLUMN_NAME
from pe.constant.data import PARENT_SYN_DATA_INDEX_COLUMN_NAME
from pe.constant.data import FROM_LAST_FLAG_COLUMN_NAME
from pe.constant.data import VARIATION_API_FOLD_ID_COLUMN_NAME
from pe.logging import execution_logger


class PEPopulation(Population):
    """The default population algorithm for Private Evolution."""

    def __init__(
        self,
        api,
        histogram_threshold=None,
        initial_variation_api_fold=0,
        next_variation_api_fold=1,
        keep_selected=False,
        selection_mode="sample",
    ):
        """Constructor.

        :param api: The API object that contains the random and variation APIs
        :type api: :py:class:`pe.api.API`
        :param histogram_threshold: The threshold for clipping the histogram. None means no clipping. Defaults to None
        :type histogram_threshold: float, optional
        :param initial_variation_api_fold: The number of variations to apply to the initial synthetic data, defaults to
            0
        :type initial_variation_api_fold: int, optional
        :param next_variation_api_fold: The number of variations to apply to the next synthetic data, defaults to 1
        :type next_variation_api_fold: int, optional
        :param keep_selected: Whether to keep the selected data in the next synthetic data, defaults to False
        :type keep_selected: bool, optional
        :param selection_mode: The selection mode for selecting the data. It should be one of the following: "sample" (
            random sampling proportional to the histogram), "rank" (select the top samples according to the histogram).
            Defaults to "sample"
        :type selection_mode: str, optional
        :raises ValueError: If next_variation_api_fold is 0 and keep_selected is False
        """
        super().__init__()
        self._api = api
        self._histogram_threshold = histogram_threshold
        self._initial_variation_api_fold = initial_variation_api_fold
        self._next_variation_api_fold = next_variation_api_fold
        self._keep_selected = keep_selected
        self._selection_mode = selection_mode
        if self._next_variation_api_fold == 0 and not self._keep_selected:
            raise ValueError(
                "next_variation_api_fold should be greater than 0 or keep_selected should be True. Otherwise, next "
                "synthetic data will be empty."
            )

    def initial(self, label_info, num_samples):
        """Generate the initial synthetic data.

        :param label_info: The label info
        :type label_info: omegaconf.dictconfig.DictConfig
        :param num_samples: The number of samples to generate
        :type num_samples: int
        :return: The initial synthetic data
        :rtype: :py:class:`pe.data.Data`
        """
        execution_logger.info(
            f"Population: generating {num_samples}*{self._initial_variation_api_fold + 1} initial "
            f"synthetic samples for label {label_info.name}"
        )
        random_data = self._api.random_api(label_info=label_info, num_samples=num_samples)
        random_data.data_frame[VARIATION_API_FOLD_ID_COLUMN_NAME] = -1
        variation_data_list = []
        for variation_api_fold_id in range(self._initial_variation_api_fold):
            variation_data = self._api.variation_api(syn_data=random_data)
            variation_data.data_frame[VARIATION_API_FOLD_ID_COLUMN_NAME] = variation_api_fold_id
            variation_data_list.append(variation_data)
        data = Data.concat([random_data] + variation_data_list)
        execution_logger.info(
            f"Population: finished generating {num_samples}*{self._initial_variation_api_fold + 1} initial "
            f"synthetic samples for label {label_info.name}"
        )
        return data

    def _post_process_histogram(self, syn_data):
        """Post process the histogram of synthetic data (e.g., clipping).

        :param syn_data: The synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        :return: The synthetic data with post-processed histogram in the column
            :py:const:`pe.constant.data.POST_PROCESSED_DP_HISTOGRAM_COLUMN_NAME`
        :rtype: :py:class:`pe.data.Data`
        """
        count = syn_data.data_frame[DP_HISTOGRAM_COLUMN_NAME].to_numpy()
        if self._histogram_threshold is not None:
            clipped_count = np.clip(count, a_min=self._histogram_threshold, a_max=None)
            clipped_count -= self._histogram_threshold
        else:
            clipped_count = count
        syn_data.data_frame[POST_PROCESSED_DP_HISTOGRAM_COLUMN_NAME] = clipped_count
        return syn_data

    def _select_data(self, syn_data, num_samples):
        """Select data from the synthetic data according to `selection_mode`.

        :param syn_data: The synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        :param num_samples: The number of samples to select
        :type num_samples: int
        :raises ValueError: If the selection mode is not supported
        :return: The selected data
        :rtype: :py:class:`pe.data.Data`
        """
        if self._selection_mode == "sample":
            count = syn_data.data_frame[POST_PROCESSED_DP_HISTOGRAM_COLUMN_NAME].to_numpy()
            prob = count / count.sum()
            indices = np.random.choice(len(syn_data.data_frame), size=num_samples, p=prob)
            new_data_frame = syn_data.data_frame.iloc[indices]
            new_data_frame[PARENT_SYN_DATA_INDEX_COLUMN_NAME] = syn_data.data_frame.index[indices]
            return Data(data_frame=new_data_frame, metadata=syn_data.metadata)
        elif self._selection_mode == "rank":
            count = syn_data.data_frame[POST_PROCESSED_DP_HISTOGRAM_COLUMN_NAME].to_numpy()
            indices = np.argsort(count)[::-1][:num_samples]
            new_data_frame = syn_data.data_frame.iloc[indices]
            new_data_frame[PARENT_SYN_DATA_INDEX_COLUMN_NAME] = syn_data.data_frame.index[indices]
            return Data(data_frame=new_data_frame, metadata=syn_data.metadata)
        else:
            raise ValueError(f"Selection mode {self._selection_mode} is not supported")

    def next(self, syn_data, num_samples):
        """Generate the next synthetic data.

        :param syn_data: The synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        :param num_samples: The number of samples to generate
        :type num_samples: int
        :return: The next synthetic data
        :rtype: :py:class:`pe.data.Data`
        """
        execution_logger.info(
            f"Population: generating {num_samples}*{self._next_variation_api_fold} " "next synthetic samples"
        )
        syn_data = self._post_process_histogram(syn_data)
        selected_data = self._select_data(syn_data, num_samples)
        selected_data.data_frame[FROM_LAST_FLAG_COLUMN_NAME] = 1
        selected_data.data_frame[VARIATION_API_FOLD_ID_COLUMN_NAME] = -1
        variation_data_list = []
        for variation_api_fold_id in range(self._next_variation_api_fold):
            variation_data = self._api.variation_api(syn_data=selected_data)
            variation_data.data_frame[PARENT_SYN_DATA_INDEX_COLUMN_NAME] = selected_data.data_frame[
                PARENT_SYN_DATA_INDEX_COLUMN_NAME
            ].values
            variation_data.data_frame[FROM_LAST_FLAG_COLUMN_NAME] = 0
            variation_data.data_frame[VARIATION_API_FOLD_ID_COLUMN_NAME] = variation_api_fold_id
            variation_data_list.append(variation_data)
        new_syn_data = Data.concat(variation_data_list + ([selected_data] if self._keep_selected else []))
        execution_logger.info(
            f"Population: finished generating {num_samples}*{self._next_variation_api_fold} " "next synthetic samples"
        )
        return new_syn_data

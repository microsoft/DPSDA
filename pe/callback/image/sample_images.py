import numpy as np

from pe.callback.callback import Callback
from pe.constant.data import IMAGE_DATA_COLUMN_NAME
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.metric_item import ImageListMetricItem


class SampleImages(Callback):
    """The callback that samples images from the synthetic data."""

    def __init__(self, num_images_per_class=10):
        """Constructor.

        :param num_images_per_class: number of images to sample per class, defaults to 10
        :type num_images_per_class: int, optional
        """
        self._num_images_per_class = num_images_per_class

    def __call__(self, syn_data):
        """This function is called after each PE iteration that samples images from the synthetic data.

        :param syn_data: The :py:class:`pe.data.Data` object of the synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        :return: A metric item with the list of sampled images
        :rtype: list[:py:class:`pe.metric_item.ImageListMetricItem`]
        """
        all_image_list = []
        num_classes = len(syn_data.metadata.label_info)
        for class_id in range(num_classes):
            image_list = syn_data.data_frame[syn_data.data_frame[LABEL_ID_COLUMN_NAME] == class_id][
                IMAGE_DATA_COLUMN_NAME
            ]
            image_list = image_list.sample(min(self._num_images_per_class, len(image_list))).tolist()
            all_image_list.extend(image_list)
            assert len(image_list) > 0
            if len(image_list) < self._num_images_per_class:
                all_image_list.extend([np.zeros_like(image_list[0])] * (self._num_images_per_class - len(image_list)))
        metric_item = ImageListMetricItem(
            name="image_sample",
            value=all_image_list,
            num_images_per_row=None if num_classes == 1 else self._num_images_per_class,
        )
        return [metric_item]

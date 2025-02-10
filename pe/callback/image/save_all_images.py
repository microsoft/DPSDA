import imageio
import os
from tqdm import tqdm

from pe.callback.callback import Callback
from pe.constant.data import IMAGE_DATA_COLUMN_NAME
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.logging import execution_logger


class SaveAllImages(Callback):
    """The callback that saves all images."""

    def __init__(
        self,
        output_folder,
        path_format="{iteration:09d}/{label_id}_{label_name}/{index}.png",
        tqdm_enabled=True,
    ):
        """Constructor.

        :param output_folder: The output folder that will be used to save the images
        :type output_folder: str
        :param path_format: The format of the image paths, defaults to
            "{iteration:09d}/{label_id}_{label_name}/{index}.png"
        :type path_format: str, optional
        :param tqdm_enabled: Whether to show tqdm progress bar when saving the images, defaults to True
        :type tqdm_enabled: bool, optional
        """
        self._output_folder = output_folder
        self._path_format = path_format
        self._tqdm_enabled = tqdm_enabled

    def _save_image(self, image, label_name, label_id, index, iteration):
        """A helper function that saves an image."""
        path = os.path.join(
            self._output_folder,
            self._path_format.format(
                iteration=iteration,
                label_id=label_id,
                label_name=label_name,
                index=index,
            ),
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.imsave(path, image)

    def __call__(self, syn_data):
        """This function is called after each PE iteration that saves all images.

        :param syn_data: The :py:class:`pe.data.Data` object of the synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        """
        execution_logger.info("Saving all images")
        iterator = range(len(syn_data.data_frame))
        if self._tqdm_enabled:
            iterator = tqdm(iterator)
        for i in iterator:
            image = syn_data.data_frame[IMAGE_DATA_COLUMN_NAME][i]
            label_id = int(syn_data.data_frame[LABEL_ID_COLUMN_NAME][i])
            label_name = syn_data.metadata.label_info[label_id].name
            index = syn_data.data_frame.index[i]
            self._save_image(
                image=image,
                label_name=label_name,
                label_id=label_id,
                index=index,
                iteration=syn_data.metadata.iteration,
            )
        execution_logger.info("Finished saving all images")

import os
import imageio
import math
import torch
import numpy as np
from torchvision.utils import make_grid

from .logger import Logger
from pe.metric_item import ImageMetricItem, ImageListMetricItem


class ImageFile(Logger):
    """The logger that saves images to files."""

    def __init__(
        self,
        output_folder,
        path_separator="-",
        iteration_format="09d",
    ):
        """Constructor.

        :param output_folder: The output folder that will be used to save the images
        :type output_folder: str
        :param path_separator: The string that will be used to replace '\' and '/' in log names, defaults to "-"
        :type path_separator: str, optional
        :param iteration_format: The format of the iteration number, defaults to "09d"
        :type iteration_format: str, optional
        """
        self._output_folder = output_folder
        self._path_separator = path_separator
        self._iteration_format = iteration_format

    def log(self, iteration, metric_items):
        """Log the images.

        :param iteration: The PE iteration number
        :type iteration: int
        :param metric_items: The images to log
        :type metric_items: list[:py:class:`pe.metric_item.ImageMetricItem` or
            :py:class:`pe.metric_item.ImageListMetricItem`]
        """
        for item in metric_items:
            if not isinstance(item, (ImageMetricItem, ImageListMetricItem)):
                continue
            image_path = self._get_image_path(iteration, item)
            if isinstance(item, ImageMetricItem):
                self._log_image(image_path, item)
            elif isinstance(item, ImageListMetricItem):
                self._log_image_list(image_path, item)

    def _get_image_path(self, iteration, item):
        """Get the image save path.

        :param iteration: The PE iteration number
        :type iteration: int
        :param item: The image metric item
        :type item: :py:class:`pe.metric_item.ImageMetricItem` or :py:class:`pe.metric_item.ImageListMetricItem`
        :return: The image save path
        :rtype: str
        """
        os.makedirs(self._output_folder, exist_ok=True)
        image_name = item.name
        image_name = image_name.replace("/", self._path_separator)
        image_name = image_name.replace("\\", self._path_separator)
        image_folder = os.path.join(self._output_folder, image_name)
        os.makedirs(image_folder, exist_ok=True)
        iteration_string = format(iteration, self._iteration_format)
        image_file_name = f"{iteration_string}.png"
        image_path = os.path.join(
            image_folder,
            image_file_name,
        )
        return image_path

    def _log_image(self, image_path, item):
        """Log a single image.

        :param image_path: The path to save the image
        :type image_path: str
        :param item: The image metric item
        :type item: :py:class:`pe.metric_item.ImageMetricItem`
        """
        image = item.value
        if isinstance(image, torch.Tensor):
            image = image.cpu().detach().numpy()
        imageio.imwrite(image_path, image)

    def _log_image_list(self, image_path, item):
        """Log a list of images.

        :param image_path: The path to save the image
        :type image_path: str
        :param item: The image list metric item
        :type item: :py:class:`pe.metric_item.ImageListMetricItem`
        """
        images = item.value
        num_images_per_row = item.num_images_per_row
        if num_images_per_row is None:
            num_images_per_row = int(math.sqrt(len(images)))

        if isinstance(images[0], np.ndarray):
            images = [torch.from_numpy(image.transpose(2, 0, 1)) for image in images]

        image = make_grid(images, nrow=num_images_per_row).cpu().detach().numpy()
        image = image.transpose((1, 2, 0))
        imageio.imwrite(image_path, image)

import numpy as np
import pandas as pd
import random
import python_avatars as pa
import cairosvg
import io
from PIL import Image
from tqdm.contrib.concurrent import process_map
from functools import partial

from pe.api import API
from pe.logging import execution_logger
from pe.data import Data
from pe.constant.data import IMAGE_DATA_COLUMN_NAME
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.api.util import ConstantList


AVATAR_PARAMS_COLUMN_NAME = "PE.AVATAR.PARAMS"


def _to_constant_list_if_needed(value):
    if not isinstance(value, list):
        value = ConstantList(value)
    return value


class Avatar(API):
    """The API that uses the python_avatars library to generate synthetic avatar images."""

    def __init__(self, res, variation_degrees, crop=(40, 40, 264 - 40, 280 - 56), num_processes=50, chunksize=100):
        """Constructor.

        :param res: The resolution of the generated images
        :type res: int
        :param variation_degrees: The variation degrees utilized at each PE iteration. If a single value is provided,
            the same variation degree will be used for all iterations. The value means the probability of changing a
            parameter to a random value.
        :type variation_degrees: float or list[float]
        :param crop: The crop of the generated images from the python_avatars library, defaults to
            (40, 40, 264 - 40, 280 - 56)
        :type crop: tuple, optional
        :param num_processes: The number of processes to use for parallel generation, defaults to 50
        :type num_processes: int, optional
        :param chunksize: The chunksize for parallel generation, defaults to 100
        :type chunksize: int, optional
        """
        super().__init__()
        self._res = res
        self._crop = crop
        self._variation_degrees = _to_constant_list_if_needed(variation_degrees)
        self._num_processes = num_processes
        self._chunksize = chunksize

    def _svg_to_numpy(self, svg):
        """Converts an SVG string to an image in numpy array format.

        :param svg: The SVG string
        :type svg: str
        :return: The image in numpy array format
        :rtype: np.ndarray
        """
        mem = io.BytesIO()
        cairosvg.svg2png(bytestring=svg, write_to=mem)
        image = Image.open(mem)
        image = image.convert("RGB")
        if self._crop is not None:
            image = image.crop(self._crop)
        image = image.resize((self._res, self._res))
        return np.array(image)

    def _get_params_from_avatar(self, avatar):
        """Get the parameters of an avatar.

        :param avatar: The avatar
        :type avatar: python_avatars.Avatar
        :return: The parameters of the avatar
        :rtype: dict
        """
        return {
            "style": avatar.style,
            "background_color": avatar.background_color,
            "top": avatar.top,
            "hat_color": avatar.hat_color,
            "eyebrows": avatar.eyebrows,
            "eyes": avatar.eyes,
            "nose": avatar.nose,
            "mouth": avatar.mouth,
            "facial_hair": avatar.facial_hair,
            "skin_color": avatar.skin_color,
            "hair_color": avatar.hair_color,
            "facial_hair_color": avatar.facial_hair_color,
            "accessory": avatar.accessory,
            "clothing": avatar.clothing,
            "clothing_color": avatar.clothing_color,
            "shirt_graphic": avatar.shirt_graphic,
        }

    def _get_random_image(self, _):
        """Get a random image and its parameters.

        :param _: The index of the sample
        :type _: int
        :return: The image and its parameters
        :rtype: tuple[np.ndarray, dict]
        """
        avatar = pa.Avatar.random()
        image = self._svg_to_numpy(avatar.render())
        params = self._get_params_from_avatar(avatar)
        return image, params

    def random_api(self, label_info, num_samples):
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

        results = process_map(
            self._get_random_image, range(num_samples), max_workers=self._num_processes, chunksize=self._chunksize
        )
        images = [result[0] for result in results]
        params = [result[1] for result in results]

        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: images,
                AVATAR_PARAMS_COLUMN_NAME: params,
                LABEL_ID_COLUMN_NAME: 0,
            }
        )
        metadata = {"label_info": [label_info]}
        execution_logger.info(f"RANDOM API: finished creating {num_samples} samples for label {label_name}")
        return Data(data_frame=data_frame, metadata=metadata)

    def _get_variation_image(
        self,
        params,
        variation_degree,
    ):
        """Get a variation image and its parameters.

        :param params: The parameters of the avatar
        :type params: dict
        :param variation_degree: The degree of variation
        :type variation_degree: float
        :return: The image of the avatar and its parameters
        :rtype: tuple[np.ndarray, dict]
        """
        avatar = pa.Avatar.random()
        for name, value in params.items():
            if random.random() > variation_degree:
                setattr(avatar, name, value)
        image = self._svg_to_numpy(avatar.render())
        params = self._get_params_from_avatar(avatar)
        return image, params

    def variation_api(self, syn_data):
        """Creating variations of the synthetic data.

        :param syn_data: The data object of the synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        :return: The data object of the variation of the input synthetic data
        :rtype: :py:class:`pe.data.Data`
        """
        execution_logger.info(f"VARIATION API: creating variations for {len(syn_data.data_frame)} samples")
        original_params = list(syn_data.data_frame[AVATAR_PARAMS_COLUMN_NAME].values)
        iteration = getattr(syn_data.metadata, "iteration", -1)
        variation_degree = self._variation_degrees[iteration + 1]

        execution_logger.info(f"VARIATION API parameters: variation_degree={variation_degree}")

        results = process_map(
            partial(self._get_variation_image, variation_degree=variation_degree),
            original_params,
            max_workers=self._num_processes,
            chunksize=self._chunksize,
        )
        variations = [result[0] for result in results]
        params = [result[1] for result in results]

        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: variations,
                AVATAR_PARAMS_COLUMN_NAME: params,
                LABEL_ID_COLUMN_NAME: syn_data.data_frame[LABEL_ID_COLUMN_NAME].values,
            }
        )
        execution_logger.info(f"VARIATION API: finished creating variations for {len(syn_data.data_frame)} samples")
        return Data(data_frame=data_frame, metadata=syn_data.metadata)

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import glob
import random
import os
import traceback
from collections import defaultdict

from pe.api import API
from pe.logging import execution_logger
from pe.data import Data
from pe.constant.data import IMAGE_DATA_COLUMN_NAME
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.api.util import ConstantList


TEXT_PARAMS_COLUMN_NAME = "PE.DRAW_TEXT.PARAMS"


def _to_constant_list_if_needed(value):
    if not isinstance(value, list):
        value = ConstantList(value)
    return value


class DrawText(API):
    """The API that uses the PIL library to generate synthetic images with text on them."""

    def __init__(
        self,
        font_root_path,
        font_variation_degrees,
        font_size_variation_degrees,
        rotation_degree_variation_degrees,
        stroke_width_variation_degrees,
        text_list=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        width=28,
        height=28,
        background_color=(0, 0, 0),
        text_color=(255, 255, 255),
        font_size_list=range(10, 30),
        stroke_width_list=[0, 1, 2],
        rotation_degree_list=range(-30, 31, 1),
    ):
        """Constructor.

        :param font_root_path: The root path that contains the font files in .ttf format
        :type font_root_path: str
        :param font_variation_degrees: The variation degrees for font utilized at each PE iteration. If a single value
            is provided, the same variation degree will be used for all iterations. The value means the probability of
            changing the font to a random font.
        :type font_variation_degrees: float or list[float]
        :param font_size_variation_degrees: The variation degrees for font size utilized at each PE iteration. If a
            single value is provided, the same variation degree will be used for all iterations. The value means
            the maximum possible variation in font size.
        :type font_size_variation_degrees: int or list[int]
        :param rotation_degree_variation_degrees: The variation degrees for rotation degree utilized at each PE
            iteration. If a single value is provided, the same variation degree will be used for all iterations. The
            value means the maximum possible variation in rotation degree.
        :type rotation_degree_variation_degrees: int or list[int]
        :param stroke_width_variation_degrees: The variation degrees for stroke width utilized at each PE iteration. If
            a single value is provided, the same variation degree will be used for all iterations. The value means the
            maximum possible variation in stroke width.
        :type stroke_width_variation_degrees: int or list[int]
        :param text_list: The texts to be used in the synthetic images. It can be a dictionary that maps label_names to
            a list of strings, meaning the texts to be used for each label_name. If a list is provided, the same texts
            will be used for all label_names. Defaults to ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        :type text_list: list or dict, optional
        :param width: The width of the synthetic images, defaults to 28
        :type width: int, optional
        :param height: The height of the synthetic images, defaults to 28
        :type height: int, optional
        :param background_color: The background color of the synthetic images, defaults to (0, 0, 0)
        :type background_color: tuple, optional
        :param text_color: The text color of the synthetic images, defaults to (255, 255, 255)
        :type text_color: tuple, optional
        :param font_size_list: The feasible set of font sizes to be used in the synthetic images, defaults to
            range(10, 30)
        :type font_size_list: list, optional
        :param stroke_width_list: The feasible set of stroke widths to be used in the synthetic images, defaults to
            [0, 1, 2]
        :type stroke_width_list: list, optional
        :param rotation_degree_list: The feasible set of rotation degrees to be used in the synthetic images, defaults
            to range(-30, 31, 1)
        :type rotation_degree_list: list, optional
        """
        super().__init__()
        self._font_root_path = font_root_path
        self._font_variation_degrees = _to_constant_list_if_needed(font_variation_degrees)
        self._font_size_variation_degrees = _to_constant_list_if_needed(font_size_variation_degrees)
        self._rotation_degree_variation_degrees = _to_constant_list_if_needed(rotation_degree_variation_degrees)
        self._stroke_width_variation_degrees = _to_constant_list_if_needed(stroke_width_variation_degrees)
        if isinstance(text_list, list):
            self._text_list = defaultdict(lambda: text_list)
        else:
            self._text_list = text_list
        self._width = width
        self._height = height
        self._background_color = background_color
        self._text_color = text_color
        self._font_size_list = font_size_list
        self._stroke_width_list = stroke_width_list
        self._rotation_degree_list = rotation_degree_list

        self._font_files = glob.glob(os.path.join(self._font_root_path, "**", "*.ttf"), recursive=True)
        execution_logger.info(f"Found {len(self._font_files)} font files in {self._font_root_path}")

    def _create_image(self, font_size, font_file, text, stroke_width, rotation_degree):
        """Create an image with text on it.

        :param font_size: The font size
        :type font_size: int
        :param font_file: The font file
        :type font_file: str
        :param text: The text
        :type text: str
        :param stroke_width: The stroke width
        :type stroke_width: int
        :param rotation_degree: The rotation degree
        :type rotation_degree: int
        :return: The image with text on it
        :rtype: np.ndarray
        """
        try:
            before_rotation_image = Image.new("RGB", (self._width, self._height), self._background_color)
            after_rotation_image = Image.new("RGB", (self._width, self._height), self._background_color)

            draw = ImageDraw.Draw(before_rotation_image)
            font = ImageFont.truetype(font_file, font_size)
            _, _, w, h = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
            draw.text(
                ((self._width - w) / 2, (self._height - h) / 2),
                text,
                font=font,
                fill=self._text_color,
                stroke_width=stroke_width,
                stroke_fill=self._text_color,
            )
            after_rotation_image.paste(before_rotation_image.rotate(rotation_degree, expand=False), (0, 0))
            return np.array(after_rotation_image)
        except Exception as e:
            execution_logger.error(f"Error when creating image: {e}")
            execution_logger.error(
                f"font_size={font_size}, font_file={font_file}, text={text}, stroke_width={stroke_width}, "
                f"rotation_degree={rotation_degree}"
            )
            execution_logger.error(traceback.format_exc())
            return None

    def _get_random_image(self, label_name):
        """Get a random image and its parameters.

        :param label_name: The label name
        :type label_name: str
        :return: The image and its parameters
        :rtype: tuple[np.ndarray, dict]
        """
        font_size = random.choice(self._font_size_list)
        font_file = random.choice(self._font_files)
        text = random.choice(self._text_list[label_name])
        stroke_width = random.choice(self._stroke_width_list)
        rotation_degree = random.choice(self._rotation_degree_list)

        image = self._create_image(
            font_size=font_size,
            font_file=font_file,
            text=text,
            stroke_width=stroke_width,
            rotation_degree=rotation_degree,
        )
        params = {
            "font_size": font_size,
            "font_file": font_file,
            "text": text,
            "stroke_width": stroke_width,
            "rotation_degree": rotation_degree,
        }
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

        images = []
        params = []
        cnt = 0
        bar = tqdm(total=num_samples)
        while cnt < num_samples:
            image, param = self._get_random_image(label_name=label_name)
            if image is not None:
                images.append(image)
                params.append(param)
                cnt += 1
                bar.update(1)
        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: images,
                TEXT_PARAMS_COLUMN_NAME: params,
                LABEL_ID_COLUMN_NAME: 0,
            }
        )
        metadata = {"label_info": [label_info]}
        execution_logger.info(f"RANDOM API: finished creating {num_samples} samples for label {label_name}")
        return Data(data_frame=data_frame, metadata=metadata)

    def _get_variation_image(
        self,
        font_size,
        font_file,
        text,
        stroke_width,
        rotation_degree,
        font_size_variation_degree,
        font_variation_degree,
        stroke_width_variation_degree,
        rotation_degree_variation_degree,
    ):
        """Get a variation image and its parameters.

        :param font_size: The font size
        :type font_size: int
        :param font_file: The font file
        :type font_file: str
        :param text: The text
        :type text: str
        :param stroke_width: The stroke width
        :type stroke_width: int
        :param rotation_degree: The rotation degree
        :type rotation_degree: int
        :param font_size_variation_degree: The degree of variation in font size
        :type font_size_variation_degree: int
        :param font_variation_degree: The degree of variation in font
        :type font_variation_degree: float
        :param stroke_width_variation_degree: The degree of variation in stroke width
        :type stroke_width_variation_degree: int
        :param rotation_degree_variation_degree: The degree of variation in rotation degree
        :type rotation_degree_variation_degree: int
        :return: The image of the avatar and its parameters
        :rtype: tuple[np.ndarray, dict]
        """
        do_font_variation = random.random() < font_variation_degree
        if do_font_variation:
            font_file = random.choice(self._font_files)

        font_size += random.randint(-font_size_variation_degree, font_size_variation_degree)
        font_size = max(min(font_size, max(self._font_size_list)), min(self._font_size_list))

        stroke_width += random.randint(-stroke_width_variation_degree, stroke_width_variation_degree)
        stroke_width = max(min(stroke_width, max(self._stroke_width_list)), min(self._stroke_width_list))

        rotation_degree += random.randint(-rotation_degree_variation_degree, rotation_degree_variation_degree)
        rotation_degree = max(min(rotation_degree, max(self._rotation_degree_list)), min(self._rotation_degree_list))

        image = self._create_image(
            font_size=font_size,
            font_file=font_file,
            text=text,
            stroke_width=stroke_width,
            rotation_degree=rotation_degree,
        )
        params = {
            "font_size": font_size,
            "font_file": font_file,
            "text": text,
            "stroke_width": stroke_width,
            "rotation_degree": rotation_degree,
        }
        return image, params

    def variation_api(self, syn_data):
        """Creating variations of the synthetic data.

        :param syn_data: The data object of the synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        :return: The data object of the variation of the input synthetic data
        :rtype: :py:class:`pe.data.Data`
        """
        execution_logger.info(f"VARIATION API: creating variations for {len(syn_data.data_frame)} samples")
        original_params = list(syn_data.data_frame[TEXT_PARAMS_COLUMN_NAME].values)
        original_images = np.stack(syn_data.data_frame[IMAGE_DATA_COLUMN_NAME].values)
        iteration = getattr(syn_data.metadata, "iteration", -1)
        font_variation_degree = self._font_variation_degrees[iteration + 1]
        font_size_variation_degree = self._font_size_variation_degrees[iteration + 1]
        rotation_variation_degree = self._rotation_degree_variation_degrees[iteration + 1]
        stroke_width_variation_degree = self._stroke_width_variation_degrees[iteration + 1]

        execution_logger.info(
            f"VARIATION API parameters: font_size_variation_degree={font_size_variation_degree}, "
            f"font_variation_degree={font_variation_degree}, rotation_variation_degree={rotation_variation_degree}, "
            f"stroke_width_variation_degree={stroke_width_variation_degree}"
        )

        variations = []
        params = []
        for i in tqdm(range(len(syn_data.data_frame))):
            original_param = original_params[i]
            image, param = self._get_variation_image(
                font_size_variation_degree=font_size_variation_degree,
                font_variation_degree=font_variation_degree,
                rotation_degree_variation_degree=rotation_variation_degree,
                stroke_width_variation_degree=stroke_width_variation_degree,
                **original_param,
            )
            if image is not None:
                variations.append(image)
                params.append(param)
            else:
                variations.append(original_images[i])
                params.append(original_param)
        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: variations,
                TEXT_PARAMS_COLUMN_NAME: params,
                LABEL_ID_COLUMN_NAME: syn_data.data_frame[LABEL_ID_COLUMN_NAME].values,
            }
        )
        execution_logger.info(f"VARIATION API: finished creating variations for {len(syn_data.data_frame)} samples")
        return Data(data_frame=data_frame, metadata=syn_data.metadata)

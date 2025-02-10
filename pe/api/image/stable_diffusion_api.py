import torch
import numpy as np
import pandas as pd
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionImg2ImgPipeline
import json
from tqdm import tqdm

from pe.api import API
from pe.logging import execution_logger
from pe.data import Data
from pe.constant.data import IMAGE_DATA_COLUMN_NAME
from pe.constant.data import IMAGE_PROMPT_COLUMN_NAME
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.api.util import ConstantList


def _to_constant_list_if_needed(value):
    if not isinstance(value, list):
        value = ConstantList(value)
    return value


def _round_to_uint8(image):
    return np.around(np.clip(image * 255, a_min=0, a_max=255)).astype(np.uint8)


class StableDiffusion(API):
    """The API that uses the Stable Diffusion model to generate synthetic data."""

    def __init__(
        self,
        prompt,
        variation_degrees,
        width=512,
        height=512,
        random_api_checkpoint="CompVis/stable-diffusion-v1-4",
        random_api_guidance_scale=7.5,
        random_api_num_inference_steps=50,
        random_api_batch_size=10,
        variation_api_checkpoint="CompVis/stable-diffusion-v1-4",
        variation_api_guidance_scale=7.5,
        variation_api_num_inference_steps=50,
        variation_api_batch_size=10,
    ):
        """Constructor.

        :param prompt: The prompt used for each label name. It can be either a string or a dictionary. If it is a
            string, it should be the path to a JSON file that contains the prompt for each label name. If it is a
            dictionary, it should be a dictionary that maps each label name to its prompt
        :type prompt: str or dict
        :param variation_degrees: The variation degrees utilized at each PE iteration. If a single float is provided,
            the same variation degree will be used for all iterations.
        :type variation_degrees: float or list[float]
        :param width: The width of the generated images, defaults to 512
        :type width: int, optional
        :param height: The height of the generated images, defaults to 512
        :type height: int, optional
        :param random_api_checkpoint: The checkpoint of the random API, defaults to "CompVis/stable-diffusion-v1-4"
        :type random_api_checkpoint: str, optional
        :param random_api_guidance_scale: The guidance scale of the random API, defaults to 7.5
        :type random_api_guidance_scale: float, optional
        :param random_api_num_inference_steps: The number of inference steps of the random API, defaults to 50
        :type random_api_num_inference_steps: int, optional
        :param random_api_batch_size: The batch size of the random API, defaults to 10
        :type random_api_batch_size: int, optional
        :param variation_api_checkpoint: The checkpoint of the variation API, defaults to
            "CompVis/stable-diffusion-v1-4"
        :type variation_api_checkpoint: str, optional
        :param variation_api_guidance_scale: The guidance scale of the variation API utilized at each PE iteration. If
            a single float is provided, the same guidance scale will be used for all iterations. Defaults to 7.5
        :type variation_api_guidance_scale: float or list[float], optional
        :param variation_api_num_inference_steps: The number of inference steps of the variation API utilized at each
            PE iteration. If a single int is provided, the same number of inference steps will be used for all
            iterations. Defaults to 50
        :type variation_api_num_inference_steps: int or list[int], optional
        :param variation_api_batch_size: The batch size of the variation API, defaults to 10
        :type variation_api_batch_size: int, optional
        :raises ValueError: If the prompt is neither a string nor a dictionary
        """
        super().__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(prompt, str):
            with open(prompt, "r") as f:
                self._prompt = json.load(f)
        elif isinstance(prompt, dict):
            self._prompt = prompt
        else:
            raise ValueError("Prompt must be either a string or a dictionary")

        self._width = width
        self._height = height

        self._random_api_checkpoint = random_api_checkpoint
        self._random_api_guidance_scale = random_api_guidance_scale
        self._random_api_num_inference_steps = random_api_num_inference_steps
        self._random_api_batch_size = random_api_batch_size

        self._variation_api_checkpoint = variation_api_checkpoint
        self._variation_api_guidance_scale = _to_constant_list_if_needed(variation_api_guidance_scale)
        self._variation_api_num_inference_steps = _to_constant_list_if_needed(variation_api_num_inference_steps)
        self._variation_api_batch_size = variation_api_batch_size

        self._variation_degrees = _to_constant_list_if_needed(variation_degrees)

        self._random_api_pipe = StableDiffusionPipeline.from_pretrained(
            self._random_api_checkpoint, torch_dtype=torch.float16
        )
        self._random_api_pipe.safety_checker = None
        self._random_api_pipe = self._random_api_pipe.to(self._device)

        self._variation_api_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self._variation_api_checkpoint, torch_dtype=torch.float16
        )
        self._variation_api_pipe.safety_checker = None
        self._variation_api_pipe = self._variation_api_pipe.to(self._device)

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

        prompt = self._prompt[label_name]
        max_batch_size = self._random_api_batch_size
        images = []
        num_iterations = int(np.ceil(float(num_samples) / max_batch_size))
        for iteration in tqdm(range(num_iterations)):
            batch_size = min(max_batch_size, num_samples - iteration * max_batch_size)
            images.append(
                self._random_api_pipe(
                    prompt=prompt,
                    width=self._width,
                    height=self._height,
                    num_inference_steps=self._random_api_num_inference_steps,
                    guidance_scale=self._random_api_guidance_scale,
                    num_images_per_prompt=batch_size,
                    output_type="np",
                ).images
            )
        images = _round_to_uint8(np.concatenate(images, axis=0))
        torch.cuda.empty_cache()
        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: list(images),
                IMAGE_PROMPT_COLUMN_NAME: prompt,
                LABEL_ID_COLUMN_NAME: 0,
            }
        )
        metadata = {"label_info": [label_info]}
        execution_logger.info(f"RANDOM API: finished creating {num_samples} samples for label {label_name}")
        return Data(data_frame=data_frame, metadata=metadata)

    def variation_api(self, syn_data):
        """Generating variations of the synthetic data.

        :param syn_data: The data object of the synthetic data
        :type syn_data: :py:class:`pe.data.Data`
        :return: The data object of the variation of the input synthetic data
        :rtype: :py:class:`pe.data.Data`
        """
        execution_logger.info(f"VARIATION API: creating variations for {len(syn_data.data_frame)} samples")
        images = np.stack(syn_data.data_frame[IMAGE_DATA_COLUMN_NAME].values)
        prompts = list(syn_data.data_frame[IMAGE_PROMPT_COLUMN_NAME].values)
        iteration = getattr(syn_data.metadata, "iteration", -1)
        variation_degree = self._variation_degrees[iteration + 1]
        guidance_scale = self._variation_api_guidance_scale[iteration + 1]
        num_inference_steps = self._variation_api_num_inference_steps[iteration + 1]

        images = images.astype(np.float32) / 127.5 - 1.0
        images = np.transpose(images, (0, 3, 1, 2))
        images = torch.Tensor(images).to(self._device)
        max_batch_size = self._variation_api_batch_size

        variations = []
        num_iterations = int(np.ceil(float(images.shape[0]) / max_batch_size))
        for iteration in tqdm(range(num_iterations)):
            variations.append(
                self._variation_api_pipe(
                    prompt=prompts[iteration * max_batch_size : (iteration + 1) * max_batch_size],
                    image=images[iteration * max_batch_size : (iteration + 1) * max_batch_size],
                    num_inference_steps=num_inference_steps,
                    strength=variation_degree,
                    guidance_scale=guidance_scale,
                    num_images_per_prompt=1,
                    output_type="np",
                ).images
            )
        variations = _round_to_uint8(np.concatenate(variations, axis=0))

        torch.cuda.empty_cache()
        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: list(variations),
                IMAGE_PROMPT_COLUMN_NAME: prompts,
                LABEL_ID_COLUMN_NAME: syn_data.data_frame[LABEL_ID_COLUMN_NAME].values,
            }
        )
        execution_logger.info(f"VARIATION API: finished creating variations for {len(syn_data.data_frame)} samples")
        return Data(data_frame=data_frame, metadata=syn_data.metadata)

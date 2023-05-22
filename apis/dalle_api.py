import openai
import numpy as np
from imageio.v2 import imread
import requests
from io import BytesIO
from PIL import Image
from tqdm import tqdm
import os
import logging

from .api import API

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


class DALLEAPI(API):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._openai_api_key = os.environ['OPENAI_API_KEY']
        openai.api_key = self._openai_api_key

    @staticmethod
    def command_line_parser():
        return super(DALLEAPI, DALLEAPI).command_line_parser()

    def image_random_sampling(self, num_samples, size, prompts):
        """
        Generates a specified number of random image samples based on a given
        prompt and size using OpenAI's Image API.

        Args:
            num_samples (int):
                The number of image samples to generate.
            size (str, optional):
                The size of the generated images in the format
                "widthxheight". Options include "256x256", "512x512", and
                "1024x1024".
            prompts (List[str]):
                The text prompts to generate images from. Each promot will be
                used to generate num_samples/len(prompts) number of samples.

        Returns:
            numpy.ndarray:
                A numpy array of shape [num_samples x width x height x
                channels] with type np.uint8 containing the generated image
                samples as numpy arrays.
            numpy.ndarray:
                A numpy array with length num_samples containing prompts for
                each image.
        """
        max_batch_size = 10
        images = []
        return_prompts = []
        for prompt_i, prompt in enumerate(prompts):
            num_samples_for_prompt = (num_samples + prompt_i) // len(prompts)
            num_iterations = int(np.ceil(
                float(num_samples_for_prompt) / max_batch_size))
            for iteration in tqdm(range(num_iterations)):
                batch_size = min(
                    max_batch_size,
                    num_samples_for_prompt - iteration * max_batch_size)
                images.append(_dalle2_random_sampling(
                    prompt=prompt, num_samples=batch_size, size=size))
            return_prompts.extend([prompt] * num_samples_for_prompt)
        return np.concatenate(images, axis=0), np.array(return_prompts)

    def image_variation(self, images, additional_info,
                        num_variations_per_image, size, variation_degree=None):
        """
        Generates a specified number of variations for each image in the input
        array using OpenAI's Image Variation API.

        Args:
            images (numpy.ndarray):
                A numpy array of shape [num_samples x width x height
                x channels] containing the input images as numpy arrays of type
                uint8.
            additional_info (numpy.ndarray):
                A numpy array with the first dimension equaling to
                num_samples containing prompts provided by
                image_random_sampling.
            num_variations_per_image (int):
                The number of variations to generate for each input image.
            size (str):
                The size of the generated image variations in the
                format "widthxheight". Options include "256x256", "512x512",
                and "1024x1024".

        Returns:
            numpy.ndarray:
                A numpy array of shape [num_samples x num_variations_per_image
                x width x height x channels] containing the generated image
                variations as numpy arrays of type uint8.
        """
        if variation_degree is not None:
            logging.info(f'Ignoring variation degree {variation_degree}')
        if additional_info is not None:
            logging.info('Ignoring additional info')
        max_batch_size = 10
        variations = []
        for iteration in tqdm(range(int(np.ceil(
                float(num_variations_per_image) / max_batch_size)))):
            batch_size = min(
                max_batch_size,
                num_variations_per_image - iteration * max_batch_size)
            sub_variations = []
            for image in tqdm(images, leave=False):
                sub_variations.append(_dalle2_image_variation(
                    image=image,
                    num_variations_per_image=batch_size,
                    size=size))
            sub_variations = np.array(sub_variations, dtype=np.uint8)
            variations.append(sub_variations)
        return np.concatenate(variations, axis=1)


# Decorator is retry logic to get around rate limits. COMMENT OUT WHEN
# DEBUGGING! Otherwise it will constantly retry on errors.
@retry(
    retry=retry_if_not_exception_type(openai.error.InvalidRequestError),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(15),
)
def _dalle2_random_sampling(prompt, num_samples=10, size="1024x1024"):
    """
    Generates a specified number of random image samples based on a given
    prompt and size using OpenAI's Image API.

    Args:
        prompt (str): The text prompt to generate images from.
        num_samples (int, optional): The number of image samples to generate.
            Default is 10. Max of 10.
        size (str, optional): The size of the generated images in the format
            "widthxheight". Default is "1024x1024". Options include "256x256",
            "512x512", and "1024x1024".

    Returns:
        numpy.ndarray: A numpy array of shape [num_samples x image size x 
            image size x channels] containing the generated image samples as
            numpy arrays.
    """
    response = openai.Image.create(prompt=prompt, n=num_samples, size=size)
    image_urls = [image["url"] for image in response["data"]]
    images = [
        imread(BytesIO(requests.get(url).content)) for url in image_urls
    ]  # Store as np array
    return np.array(images)


@retry(
    retry=retry_if_not_exception_type(openai.error.InvalidRequestError),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(15),
)
def _dalle2_image_variation(image, num_variations_per_image,
                            size="1024x1024"):
    """
    Generates a specified number of variations for one image in the input
    array using OpenAI's Image Variation API.

    Args:
        images (numpy.ndarray): A numpy array of shape [channels
            x image size x image size] containing the input images as numpy
            arrays of type uint8.
        num_variations_per_image (int): The number of variations to generate
            for each input image.
        size (str, optional): The size of the generated image variations in the
            format "widthxheight". Default is "1024x1024". Options include
            "256x256", "512x512", and "1024x1024".

    Returns:
        numpy.ndarray: A numpy array of shape [
            num_variations_per_image x image size x image size x channels]
            containing the generated image variations as numpy arrays of type
            uint8.
    """
    im = Image.fromarray(image)
    with BytesIO() as buffer:
        im.save(buffer, format="PNG")
        buffer.seek(0)
        response = openai.Image.create_variation(
            image=buffer, n=num_variations_per_image, size=size
        )
    image_urls = [image["url"] for image in response["data"]]
    image_variations = [
        imread(BytesIO(requests.get(url).content)) for url in image_urls
    ]

    return np.array(image_variations, dtype=np.uint8)

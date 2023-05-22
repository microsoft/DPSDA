import torch
import numpy as np
from tqdm import tqdm
import logging

from .api import API
from dpsda.arg_utils import str2bool

from .improved_diffusion.unet import create_model
from improved_diffusion import dist_util
from improved_diffusion.script_util import NUM_CLASSES
from .improved_diffusion.gaussian_diffusion import create_gaussian_diffusion


def _round_to_uint8(image):
    return np.around(np.clip(image, a_min=0, a_max=255)).astype(np.uint8)


class ImprovedDiffusionAPI(API):
    def __init__(self, model_image_size, num_channels, num_res_blocks,
                 learn_sigma, class_cond, use_checkpoint,
                 attention_resolutions, num_heads, num_heads_upsample,
                 use_scale_shift_norm, dropout, diffusion_steps, sigma_small,
                 noise_schedule, use_kl, predict_xstart, rescale_timesteps,
                 rescale_learned_sigmas, timestep_respacing, model_path,
                 batch_size, use_ddim, clip_denoised, use_data_parallel,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = create_model(
            image_size=model_image_size,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            learn_sigma=learn_sigma,
            class_cond=class_cond,
            use_checkpoint=use_checkpoint,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            dropout=dropout)
        self._diffusion = create_gaussian_diffusion(
            steps=diffusion_steps,
            learn_sigma=learn_sigma,
            sigma_small=sigma_small,
            noise_schedule=noise_schedule,
            use_kl=use_kl,
            predict_xstart=predict_xstart,
            rescale_timesteps=rescale_timesteps,
            rescale_learned_sigmas=rescale_learned_sigmas,
            timestep_respacing=timestep_respacing)
        self._model.load_state_dict(
            dist_util.load_state_dict(model_path, map_location="cpu"))
        self._model.to(dist_util.dev())
        self._model.eval()
        self._sampler = Sampler(model=self._model, diffusion=self._diffusion)
        if use_data_parallel:
            self._sampler = torch.nn.DataParallel(self._sampler)
        self._batch_size = batch_size
        self._use_ddim = use_ddim
        self._image_size = model_image_size
        self._clip_denoised = clip_denoised
        self._class_cond = class_cond

    @staticmethod
    def command_line_parser():
        parser = super(
            ImprovedDiffusionAPI, ImprovedDiffusionAPI).command_line_parser()
        parser.description = (
            'See https://github.com/openai/improved-diffusion for the details'
            ' of the arguments.')
        parser.add_argument(
            '--model_image_size',
            type=int,
            default=64)
        parser.add_argument(
            '--num_channels',
            type=int,
            default=128)
        parser.add_argument(
            '--num_res_blocks',
            type=int,
            default=2)
        parser.add_argument(
            '--learn_sigma',
            type=str2bool,
            default=False)
        parser.add_argument(
            '--class_cond',
            type=str2bool,
            default=False)
        parser.add_argument(
            '--use_checkpoint',
            type=str2bool,
            default=False)
        parser.add_argument(
            '--attention_resolutions',
            type=str,
            default='16,8')
        parser.add_argument(
            '--num_heads',
            type=int,
            default=4)
        parser.add_argument(
            '--num_heads_upsample',
            type=int,
            default=-1)
        parser.add_argument(
            '--use_scale_shift_norm',
            type=str2bool,
            default=True)
        parser.add_argument(
            '--dropout',
            type=float,
            default=0.0)
        parser.add_argument(
            '--diffusion_steps',
            type=int,
            default=1000)
        parser.add_argument(
            '--sigma_small',
            type=str2bool,
            default=False)
        parser.add_argument(
            '--noise_schedule',
            type=str,
            default='linear')
        parser.add_argument(
            '--use_kl',
            type=str2bool,
            default=False)
        parser.add_argument(
            '--predict_xstart',
            type=str2bool,
            default=False)
        parser.add_argument(
            '--rescale_timesteps',
            type=str2bool,
            default=True)
        parser.add_argument(
            '--rescale_learned_sigmas',
            type=str2bool,
            default=True)
        parser.add_argument(
            '--timestep_respacing',
            type=str,
            default='')
        parser.add_argument(
            '--model_path',
            type=str,
            required=True)
        parser.add_argument(
            '--batch_size',
            type=int,
            default=100)
        parser.add_argument(
            '--use_ddim',
            type=str2bool,
            default=False)
        parser.add_argument(
            '--clip_denoised',
            type=str2bool,
            default=True)
        parser.add_argument(
            '--use_data_parallel',
            type=str2bool,
            default=True,
            help='Whether to use DataParallel to speed up sampling')
        return parser

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
                A numpy array with length num_samples containing labels for
                each image.
        """
        width, height = list(map(int, size.split('x')))
        if width != self._image_size or height != self._image_size:
            raise ValueError(
                f'width and height must be equal to {self._image_size}')
        samples, labels = sample(
            sampler=self._sampler,
            start_t=0,
            num_samples=num_samples,
            batch_size=self._batch_size,
            use_ddim=self._use_ddim,
            image_size=self._image_size,
            clip_denoised=self._clip_denoised,
            class_cond=self._class_cond)
        samples = _round_to_uint8((samples + 1.0) * 127.5)
        samples = samples.transpose(0, 2, 3, 1)
        torch.cuda.empty_cache()
        return samples, labels

    def image_variation(self, images, additional_info,
                        num_variations_per_image, size, variation_degree):
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
                num_samples containing labels provided by
                image_random_sampling.
            num_variations_per_image (int):
                The number of variations to generate for each input image.
            size (str):
                The size of the generated image variations in the
                format "widthxheight". Options include "256x256", "512x512",
                and "1024x1024".
            variation_degree (int):
                The diffusion step to add noise to the images to before running
                the denoising steps. The value should between 0 and
                timestep_respacing-1. 0 means the step that is closest to
                noise. timestep_respacing-1 means the step that is closest to
                clean image. A smaller value will result in more variation.

        Returns:
            numpy.ndarray:
                A numpy array of shape [num_samples x num_variations_per_image
                x width x height x channels] containing the generated image
                variations as numpy arrays of type uint8.
        """
        width, height = list(map(int, size.split('x')))
        if width != self._image_size or height != self._image_size:
            raise ValueError(
                f'width and height must be equal to {self._image_size}')
        images = images.astype(np.float32) / 127.5 - 1.0
        images = images.transpose(0, 3, 1, 2)
        variations = []
        for _ in tqdm(range(num_variations_per_image)):
            sub_variations = self._image_variation(
                images=images,
                labels=additional_info,
                variation_degree=variation_degree)
            variations.append(sub_variations)
        variations = np.stack(variations, axis=1)

        variations = _round_to_uint8((variations + 1.0) * 127.5)
        variations = variations.transpose(0, 1, 3, 4, 2)
        torch.cuda.empty_cache()
        return variations

    def _image_variation(self, images, labels, variation_degree):
        samples, _ = sample(
            sampler=self._sampler,
            start_t=variation_degree,
            start_image=torch.Tensor(images).to(dist_util.dev()),
            labels=(None if not self._class_cond
                    else torch.LongTensor(labels).to(dist_util.dev())),
            num_samples=images.shape[0],
            batch_size=self._batch_size,
            use_ddim=self._use_ddim,
            image_size=self._image_size,
            clip_denoised=self._clip_denoised,
            class_cond=self._class_cond)
        return samples


def sample(sampler, num_samples, start_t, batch_size, use_ddim,
           image_size, clip_denoised, class_cond,
           start_image=None, labels=None):
    all_images = []
    all_labels = []
    batch_cnt = 0
    cnt = 0
    while cnt < num_samples:
        current_batch_size = \
            (batch_size if start_image is None
             else min(batch_size,
                      start_image.shape[0] - batch_cnt * batch_size))
        shape = (current_batch_size, 3, image_size, image_size)
        model_kwargs = {}
        if class_cond:
            if labels is None:
                classes = torch.randint(
                    low=0, high=NUM_CLASSES, size=(current_batch_size,),
                    device=dist_util.dev()
                )
            else:
                classes = labels[batch_cnt * batch_size:
                                 (batch_cnt + 1) * batch_size]
            model_kwargs["y"] = classes
        sample = sampler(
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            start_t=max(start_t, 0),
            start_image=(None if start_image is None
                         else start_image[batch_cnt * batch_size:
                                          (batch_cnt + 1) * batch_size]),
            use_ddim=use_ddim,
            noise=torch.randn(*shape, device=dist_util.dev()),
            image_size=image_size)
        batch_cnt += 1

        all_images.append(sample.detach().cpu().numpy())

        if class_cond:
            all_labels.append(classes.detach().cpu().numpy())

        cnt += sample.shape[0]
        logging.info(f"Created {cnt} samples")

    all_images = np.concatenate(all_images, axis=0)
    all_images = all_images[: num_samples]

    if class_cond:
        all_labels = np.concatenate(all_labels, axis=0)
        all_labels = all_labels[: num_samples]
    else:
        all_labels = np.zeros(shape=(num_samples,))
    return all_images, all_labels


class Sampler(torch.nn.Module):
    """
    A wrapper around the model and diffusion modules that handles the entire
    sampling process, so as to reduce the communiation rounds between GPUs when
    using DataParallel.
    """
    def __init__(self, model, diffusion):
        super().__init__()
        self._model = model
        self._diffusion = diffusion

    def forward(self, clip_denoised, model_kwargs, start_t, start_image,
                use_ddim, noise, image_size):
        sample_fn = (
            self._diffusion.p_sample_loop if not use_ddim
            else self._diffusion.ddim_sample_loop)
        sample = sample_fn(
            self._model,
            (noise.shape[0], 3, image_size, image_size),
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            start_t=max(start_t, 0),
            start_image=start_image,
            noise=noise,
            device=noise.device)
        return sample

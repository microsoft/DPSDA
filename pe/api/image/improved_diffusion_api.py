import torch
import numpy as np
import pandas as pd
import tempfile
import os

from pe.api import API
from pe.logging import execution_logger
from pe.data import Data
from pe.constant.data import IMAGE_DATA_COLUMN_NAME
from pe.constant.data import IMAGE_MODEL_LABEL_COLUMN_NAME
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.api.util import ConstantList
from pe.util import download

from improved_diffusion.script_util import NUM_CLASSES
from .improved_diffusion_lib.unet import create_model
from .improved_diffusion_lib.gaussian_diffusion import create_gaussian_diffusion


class ImprovedDiffusion(API):
    """The image API that utilizes improved diffusion models from https://arxiv.org/abs/2102.09672."""

    def __init__(
        self,
        variation_degrees,
        model_path,
        model_image_size=64,
        num_channels=192,
        num_res_blocks=3,
        learn_sigma=True,
        class_cond=True,
        use_checkpoint=False,
        attention_resolutions="16,8",
        num_heads=4,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0.0,
        diffusion_steps=4000,
        sigma_small=False,
        noise_schedule="cosine",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="100",
        batch_size=2000,
        use_ddim=True,
        clip_denoised=True,
        use_data_parallel=True,
    ):
        """Constructor.
        See https://github.com/openai/improved-diffusion for the explanation of the parameters not listed here.

        :param variation_degrees: The variation degrees utilized at each PE iteration. If a single int is provided, the
            same variation degree will be used for all iterations.
        :type variation_degrees: int or list[int]
        :param model_path: The path of the model checkpoint
        :type model_path: str
        :param diffusion_steps: The total number of diffusion steps, defaults to 4000
        :type diffusion_steps: int, optional
        :param timestep_respacing: The step configurations for image generation utilized at each PE iteration. If a
            single str is provided, the same step configuration will be used for all iterations. Defaults to "100"
        :type timestep_respacing: str or list[str], optional
        :param batch_size: The batch size for image generation, defaults to 2000
        :type batch_size: int, optional
        :param use_data_parallel: Whether to use data parallel during image generation, defaults to True
        :type use_data_parallel: bool, optional
        """
        super().__init__()
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
            dropout=dropout,
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self._model.to(self._device)
        self._model.eval()
        all_timestep_respacing = (
            set(timestep_respacing) if isinstance(timestep_respacing, list) else {timestep_respacing}
        )
        self._timestep_respacing_to_diffusion = {}
        self._timestep_respacing_to_sampler = {}
        for sub_timestep_respacing in all_timestep_respacing:
            self._timestep_respacing_to_diffusion[sub_timestep_respacing] = create_gaussian_diffusion(
                steps=diffusion_steps,
                learn_sigma=learn_sigma,
                sigma_small=sigma_small,
                noise_schedule=noise_schedule,
                use_kl=use_kl,
                predict_xstart=predict_xstart,
                rescale_timesteps=rescale_timesteps,
                rescale_learned_sigmas=rescale_learned_sigmas,
                timestep_respacing=sub_timestep_respacing,
            )
            self._timestep_respacing_to_sampler[sub_timestep_respacing] = Sampler(
                model=self._model, diffusion=self._timestep_respacing_to_diffusion[sub_timestep_respacing]
            )
            if use_data_parallel:
                self._timestep_respacing_to_sampler[sub_timestep_respacing] = torch.nn.DataParallel(
                    self._timestep_respacing_to_sampler[sub_timestep_respacing]
                )
        if isinstance(timestep_respacing, str):
            self._timestep_respacing = ConstantList(timestep_respacing)
        else:
            self._timestep_respacing = timestep_respacing
        self._batch_size = batch_size
        self._use_ddim = use_ddim
        self._image_size = model_image_size
        self._clip_denoised = clip_denoised
        self._class_cond = class_cond
        if isinstance(variation_degrees, int):
            self._variation_degrees = ConstantList(variation_degrees)
        else:
            self._variation_degrees = variation_degrees

    def random_api(self, label_info, num_samples):
        """Generating random synthetic data.

        :param label_info: The info of the label, not utilized in this API
        :type label_info: omegaconf.dictconfig.DictConfig
        :param num_samples: The number of random samples to generate
        :type num_samples: int
        :return: The data object of the generated synthetic data
        :rtype: :py:class:`pe.data.Data`
        """
        label_name = label_info.name
        execution_logger.info(f"RANDOM API: creating {num_samples} samples for label {label_name}")
        samples, labels = sample(
            sampler=self._timestep_respacing_to_sampler[self._timestep_respacing[0]],
            start_t=0,
            num_samples=num_samples,
            batch_size=self._batch_size,
            use_ddim=self._use_ddim,
            image_size=self._image_size,
            clip_denoised=self._clip_denoised,
            class_cond=self._class_cond,
            device=self._device,
        )
        samples = _round_to_uint8((samples + 1.0) * 127.5)
        samples = samples.transpose(0, 2, 3, 1)
        torch.cuda.empty_cache()
        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: list(samples),
                IMAGE_MODEL_LABEL_COLUMN_NAME: list(labels),
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
        labels = np.array(syn_data.data_frame[IMAGE_MODEL_LABEL_COLUMN_NAME].values)
        iteration = getattr(syn_data.metadata, "iteration", -1)
        variation_degree = self._variation_degrees[iteration + 1]
        timestep_respacing = self._timestep_respacing[iteration + 1]

        execution_logger.info(
            f"VARIATION API parameters: variation_degree={variation_degree}, timestep_respacing={timestep_respacing}, "
            f"iteration={iteration}"
        )

        images = images.astype(np.float32) / 127.5 - 1.0
        images = images.transpose(0, 3, 1, 2)
        variations, _ = sample(
            sampler=self._timestep_respacing_to_sampler[timestep_respacing],
            start_t=variation_degree,
            start_image=torch.Tensor(images).to(self._device),
            labels=(None if not self._class_cond else torch.LongTensor(labels).to(self._device)),
            num_samples=images.shape[0],
            batch_size=self._batch_size,
            use_ddim=self._use_ddim,
            image_size=self._image_size,
            clip_denoised=self._clip_denoised,
            class_cond=self._class_cond,
            device=self._device,
        )
        variations = _round_to_uint8((variations + 1.0) * 127.5)
        variations = variations.transpose(0, 2, 3, 1)
        torch.cuda.empty_cache()
        data_frame = pd.DataFrame(
            {
                IMAGE_DATA_COLUMN_NAME: list(variations),
                IMAGE_MODEL_LABEL_COLUMN_NAME: list(labels),
                LABEL_ID_COLUMN_NAME: syn_data.data_frame[LABEL_ID_COLUMN_NAME].values,
            }
        )
        execution_logger.info(f"VARIATION API: finished creating variations for {len(syn_data.data_frame)} samples")
        return Data(data_frame=data_frame, metadata=syn_data.metadata)


def sample(
    sampler,
    num_samples,
    start_t,
    batch_size,
    use_ddim,
    image_size,
    clip_denoised,
    class_cond,
    device,
    start_image=None,
    labels=None,
):
    all_images = []
    all_labels = []
    batch_cnt = 0
    cnt = 0
    while cnt < num_samples:
        current_batch_size = (
            batch_size if start_image is None else min(batch_size, start_image.shape[0] - batch_cnt * batch_size)
        )
        current_batch_size = min(num_samples - cnt, current_batch_size)
        shape = (current_batch_size, 3, image_size, image_size)
        model_kwargs = {}
        if class_cond:
            if labels is None:
                classes = torch.randint(
                    low=0,
                    high=NUM_CLASSES,
                    size=(current_batch_size,),
                    device=device,
                )
            else:
                classes = labels[batch_cnt * batch_size : (batch_cnt + 1) * batch_size]
            model_kwargs["y"] = classes
        sample = sampler(
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            start_t=max(start_t, 0),
            start_image=(
                None if start_image is None else start_image[batch_cnt * batch_size : (batch_cnt + 1) * batch_size]
            ),
            use_ddim=use_ddim,
            noise=torch.randn(*shape, device=device),
            image_size=image_size,
        )
        batch_cnt += 1

        all_images.append(sample.detach().cpu().numpy())

        if class_cond:
            all_labels.append(classes.detach().cpu().numpy())

        cnt += sample.shape[0]
        execution_logger.info(f"Created {cnt} samples")

    all_images = np.concatenate(all_images, axis=0)
    all_images = all_images[:num_samples]

    if class_cond:
        all_labels = np.concatenate(all_labels, axis=0)
        all_labels = all_labels[:num_samples]
    else:
        all_labels = np.zeros(shape=(num_samples,))
    return all_images, all_labels


class Sampler(torch.nn.Module):
    """A wrapper around the model and diffusion modules that handles the entire
    sampling process, so as to reduce the communiation rounds between GPUs when
    using DataParallel.
    """

    def __init__(self, model, diffusion):
        super().__init__()
        self._model = model
        self._diffusion = diffusion

    def forward(
        self,
        clip_denoised,
        model_kwargs,
        start_t,
        start_image,
        use_ddim,
        noise,
        image_size,
    ):
        sample_fn = self._diffusion.p_sample_loop if not use_ddim else self._diffusion.ddim_sample_loop
        sample = sample_fn(
            self._model,
            (noise.shape[0], 3, image_size, image_size),
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            start_t=max(start_t, 0),
            start_image=start_image,
            noise=noise,
            device=noise.device,
        )
        return sample


def _round_to_uint8(image):
    return np.around(np.clip(image, a_min=0, a_max=255)).astype(np.uint8)


class ImprovedDiffusion270M(ImprovedDiffusion):
    #: The URL of the checkpoint path
    CHECKPOINT_URL = "https://openaipublic.blob.core.windows.net/diffusion/march-2021/imagenet64_cond_270M_250K.pt"

    def __init__(
        self,
        variation_degrees,
        model_path=None,
        batch_size=2000,
        timestep_respacing="100",
        use_data_parallel=True,
    ):
        """The "Class-conditional ImageNet-64 model (270M parameters, trained for 250K iterations)" model from the
        Improved Diffusion paper.

        :param variation_degrees: The variation degrees utilized at each PE iteration
        :type variation_degrees: list[int]
        :param model_path: The path of the model checkpoint. If not provided, the checkpoint will be downloaded from
            the `CHECKPOINT_URL`
        :type model_path: str
        :param batch_size: The batch size for image generation, defaults to 2000
        :type batch_size: int, optional
        :param timestep_respacing: The step configuration for image generation, defaults to "100"
        :type timestep_respacing: str, optional
        :param use_data_parallel: Whether to use data parallel during image generation, defaults to True
        :type use_data_parallel: bool, optional
        """
        if model_path is None or not os.path.exists(model_path):
            model_path = self._download_checkpoint(model_path)
        super().__init__(
            variation_degrees=variation_degrees,
            model_path=model_path,
            model_image_size=64,
            num_channels=192,
            num_res_blocks=3,
            learn_sigma=True,
            class_cond=True,
            use_checkpoint=False,
            attention_resolutions="16,8",
            num_heads=4,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            dropout=0.0,
            diffusion_steps=4000,
            sigma_small=False,
            noise_schedule="cosine",
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=False,
            rescale_learned_sigmas=False,
            timestep_respacing=timestep_respacing,
            batch_size=batch_size,
            use_ddim=True,
            clip_denoised=True,
            use_data_parallel=use_data_parallel,
        )

    def _download_checkpoint(self, model_path):
        execution_logger.info(f"Downloading ImprovedDiffusion checkpoint from {self.CHECKPOINT_URL}")
        if model_path is None:
            model_path = tempfile.mktemp(suffix=".pt")
        download(url=self.CHECKPOINT_URL, fname=model_path)
        execution_logger.info(f"Finished downloading ImprovedDiffusion checkpoint to {model_path}")
        return model_path

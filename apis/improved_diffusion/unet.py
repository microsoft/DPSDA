"""
This code contains minor edits from the original code at
https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/unet.py
and
https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/script_util.py
to avoid calling self.input_blocks.parameters() in the original code, which is
not supported by DataParallel.
"""


import torch
from improved_diffusion.unet import UNetModel
from improved_diffusion.script_util import NUM_CLASSES


class FP32UNetModel(UNetModel):
    @property
    def inner_dtype(self):
        return torch.float32


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    if image_size == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return FP32UNetModel(
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )

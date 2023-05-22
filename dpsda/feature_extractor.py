import imageio
import cleanfid
import os
import shutil
import numpy as np
from tqdm import tqdm
import torch
from cleanfid.resize import make_resizer


def round_to_uint8(image):
    return np.around(np.clip(image, a_min=0, a_max=255)).astype(np.uint8)


def extract_features(
        data, mode='clean', device=torch.device('cuda'), use_dataparallel=True,
        num_workers=12, batch_size=5000, custom_fn_resize=None, description='',
        verbose=True, custom_image_tranform=None, tmp_folder='tmp_feature',
        model_name="inception_v3", res=32):
    if model_name == 'original':
        return data.reshape((data.shape[0], -1)).astype(np.float32)
    elif model_name == "inception_v3":
        feat_model = cleanfid.features.build_feature_extractor(
            mode=mode,
            device=device,
            use_dataparallel=use_dataparallel
        )
    elif model_name == "clip_vit_b_32":
        from cleanfid.clip_features import CLIP_fx, img_preprocess_clip
        clip_fx = CLIP_fx("ViT-B/32", device=device)
        feat_model = clip_fx
        custom_fn_resize = img_preprocess_clip
    else:
        raise Exception(f'Unknown model_name {model_name}')
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    else:
        shutil.rmtree(tmp_folder, ignore_errors=False, onerror=None)
        os.makedirs(tmp_folder)
    assert data.dtype == np.uint8
    resizer = make_resizer(
        library='PIL', quantize_after=False, filter='bicubic',
        output_size=(res, res))
    # TODO: in memory processing for computing features.
    for i in tqdm(range(data.shape[0])):
        image = round_to_uint8(resizer(data[i]))
        imageio.imsave(os.path.join(tmp_folder, f'{i}.png'), image)
    files = [os.path.join(tmp_folder, f'{i}.png')
             for i in range(data.shape[0])]

    np_feats = cleanfid.fid.get_files_features(
        l_files=files,
        model=feat_model,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        mode=mode,
        custom_fn_resize=custom_fn_resize,
        custom_image_tranform=custom_image_tranform,
        description=description,
        fdir=tmp_folder,
        verbose=verbose)
    return np_feats.astype(np.float32)

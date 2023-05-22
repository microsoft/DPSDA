import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import logging

from .dataset import ImageDataset


def load_data(data_dir, batch_size, image_size, class_cond,
              num_private_samples):
    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor()
    ])
    dataset = ImageDataset(folder=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=10,
                        pin_memory=torch.cuda.is_available(), drop_last=False)
    all_samples = []
    all_labels = []
    cnt = 0
    for batch, cond in loader:
        all_samples.append(batch.cpu().numpy())

        if class_cond:
            all_labels.append(cond.cpu().numpy())

        cnt += batch.shape[0]

        logging.info(f'loaded {cnt} samples')
        if batch.shape[0] < batch_size:
            logging.info('WARNING: containing incomplete batch. Please check'
                         'num_private_samples')

        if cnt >= num_private_samples:
            break

    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = all_samples[:num_private_samples]
    all_samples = np.around(np.clip(
        all_samples * 255, a_min=0, a_max=255)).astype(np.uint8)
    all_samples = np.transpose(all_samples, (0, 2, 3, 1))
    if class_cond:
        all_labels = np.concatenate(all_labels, axis=0)
        all_labels = all_labels[:num_private_samples]
    else:
        all_labels = np.zeros(shape=all_samples.shape[0], dtype=np.int64)
    return all_samples, all_labels

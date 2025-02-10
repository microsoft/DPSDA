import pandas as pd
from PIL import Image as PILImage
import blobfile as bf
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch
import numpy as np

from pe.data import Data
from pe.logging import execution_logger
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.constant.data import IMAGE_DATA_COLUMN_NAME


def _list_image_files_recursively(data_dir):
    """List all image files in a directory recursively. Adapted from
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/image_datasets.py
    """
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, folder, transform):
        super().__init__()
        self.folder = folder
        self.transform = transform

        self.local_images = _list_image_files_recursively(folder)
        self.local_class_names = [bf.basename(path).split("_")[0] for path in self.local_images]
        self.class_names = list(sorted(set(self.local_class_names)))
        self.class_name_to_id = {x: i for i, x in enumerate(self.class_names)}
        self.local_classes = [self.class_name_to_id[x] for x in self.local_class_names]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = PILImage.open(f)
            pil_image.load()

        arr = self.transform(pil_image)

        label = self.local_classes[idx]
        return arr, label


def load_image_folder(path, image_size, class_cond=True, num_images=-1, num_workers=10, batch_size=1000):
    """Load a image dataset from a folder that contains image files. The folder can be nested arbitrarily. The image
    file names must be in the format of "{class_name without '_'}_{suffix in any string}.ext". The "ext" can be "jpg",
    "jpeg", "png", or "gif". The class names will be extracted from the file names before the first "_". If class_cond
    is False, the class names will be ignored and all images will be treated as the same class with class name "None".

    :param path: The path to the root folder that contains the image files
    :type path: str
    :param image_size: The size of the images. Images will be resized to this size
    :type image_size: int
    :param class_cond: Whether to treat the loaded dataset as class conditional, defaults to True
    :type class_cond: bool, optional
    :param num_images: The number of images to load. If -1, load all images. Defaults to -1
    :type num_images: int, optional
    :param num_workers: The number of workers to use for loading the images, defaults to 10
    :type num_workers: int, optional
    :param batch_size: The batch size to use for loading the images, defaults to 1000
    :type batch_size: int, optional
    :return: The loaded data
    :rtype: :py:class:`pe.data.Data`
    """
    transform = T.Compose([T.Resize(image_size), T.CenterCrop(image_size), T.ToTensor()])
    dataset = ImageDataset(folder=path, transform=transform)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    all_samples = []
    all_labels = []
    cnt = 0
    for batch, cond in loader:
        all_samples.append(batch.cpu().numpy())

        if class_cond:
            all_labels.append(cond.cpu().numpy())

        cnt += batch.shape[0]

        execution_logger.info(f"Loaded {cnt} samples.")
        if batch.shape[0] < batch_size:
            execution_logger.info("Containing incomplete batch. Please check num_images is desired.")

        if num_images > 0 and cnt >= num_images:
            break

    all_samples = np.concatenate(all_samples, axis=0)
    if num_images <= 0:
        num_images = all_samples.shape[0]
    all_samples = all_samples[:num_images]
    all_samples = np.around(np.clip(all_samples * 255, a_min=0, a_max=255)).astype(np.uint8)
    all_samples = np.transpose(all_samples, (0, 2, 3, 1))
    if class_cond:
        all_labels = np.concatenate(all_labels, axis=0)
        all_labels = all_labels[:num_images]
    else:
        all_labels = np.zeros(shape=all_samples.shape[0], dtype=np.int64)
    data_frame = pd.DataFrame(
        {
            IMAGE_DATA_COLUMN_NAME: list(all_samples),
            LABEL_ID_COLUMN_NAME: list(all_labels),
        }
    )
    metadata = {"label_info": [{"name": n} for n in dataset.class_names] if class_cond else [{"name": "None"}]}
    return Data(data_frame=data_frame, metadata=metadata)

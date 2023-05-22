from wilds import get_dataset
from tqdm import tqdm
import os


def save(dataset, path):
    if not os.path.exists(path):
        os.makedirs(path)
    for i in tqdm(range(len(dataset))):
        image, label, _ = dataset[i]
        image.save(f'{path}/{label.item()}_{i}.png')


if __name__ == '__main__':
    dataset = get_dataset(dataset="camelyon17", download=True)
    train_data = dataset.get_subset("train")
    val_data = dataset.get_subset("val")
    test_data = dataset.get_subset("test")

    save(train_data, 'camelyon17_train')
    save(val_data, 'camelyon17_test')
    save(test_data, 'camelyon17_val')

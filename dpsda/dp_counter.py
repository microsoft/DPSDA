import faiss
import logging
import numpy as np
from collections import Counter
import torch


def dp_nn_histogram(public_features, private_features, noise_multiplier,
                    num_packing=1, num_nearest_neighbor=1, mode='L2',
                    threshold=0.0):
    assert public_features.shape[0] % num_packing == 0
    num_true_public_features = public_features.shape[0] // num_packing
    faiss_res = faiss.StandardGpuResources()
    if mode == 'L2':
        index = faiss.IndexFlatL2(public_features.shape[1])
    elif mode == 'IP':
        index = faiss.IndexFlatIP(public_features.shape[1])
    else:
        raise Exception(f'Unknown mode {mode}')
    if torch.cuda.is_available():
        index = faiss.index_cpu_to_gpu(faiss_res, 0, index)

    index.add(public_features)
    logging.info(f'Number of samples in index: {index.ntotal}')

    _, ids = index.search(private_features, k=num_nearest_neighbor)
    logging.info('Finished search')
    counter = Counter(list(ids.flatten()))
    count = np.zeros(shape=num_true_public_features)
    for k in counter:
        count[k % num_true_public_features] += counter[k]
    logging.info(f'Clean count sum: {np.sum(count)}')
    logging.info(f'Clean count num>0: {np.sum(count > 0)}')
    logging.info(f'Largest clean counters: {sorted(count)[::-1][:50]}')
    count = np.asarray(count)
    clean_count = count.copy()
    count += (np.random.normal(size=len(count)) * np.sqrt(num_nearest_neighbor)
              * noise_multiplier)
    logging.info(f'Noisy count sum: {np.sum(count)}')
    logging.info(f'Noisy count num>0: {np.sum(count > 0)}')
    logging.info(f'Largest noisy counters: {sorted(count)[::-1][:50]}')
    count = np.clip(count, a_min=threshold, a_max=None)
    count = count - threshold
    logging.info(f'Clipped noisy count sum: {np.sum(count)}')
    logging.info(f'Clipped noisy count num>0: {np.sum(count > 0)}')
    logging.info(f'Clipped largest noisy counters: {sorted(count)[::-1][:50]}')
    return count, clean_count


    


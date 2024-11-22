import faiss
import torch
import numpy as np


def search(syn_embedding, priv_embedding, num_nearest_neighbors, mode):
    """Compute the nearest neighbors of the private embedding in the synthetic embedding using FAISS.

    :param syn_embedding: The synthetic embedding
    :type syn_embedding: np.ndarray
    :param priv_embedding: The private embedding
    :type priv_embedding: np.ndarray
    :param num_nearest_neighbors: The number of nearest neighbors to search
    :type num_nearest_neighbors: int
    :param mode: The distance metric to use for finding the nearest neighbors. It should be one of the following:
            "l2" (l2 distance), "cos_sim" (cosine similarity), "ip" (inner product)
    :type mode: str
    :raises ValueError: If the mode is unknown
    :return: The distances and indices of the nearest neighbors
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    if mode.lower() == "l2":
        index = faiss.IndexFlatL2(syn_embedding.shape[1])
    elif mode.lower() == "ip":
        index = faiss.IndexFlatIP(syn_embedding.shape[1])
    elif mode.lower() == "cos_sim":
        index = faiss.IndexFlatIP(syn_embedding.shape[1])
        faiss.normalize_L2(syn_embedding)
        faiss.normalize_L2(priv_embedding)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if torch.cuda.is_available():
        ngpus = faiss.get_num_gpus()
        co = faiss.GpuMultipleClonerOptions()
        index = faiss.index_cpu_to_all_gpus(index, co, ngpus)

    index.add(syn_embedding)
    distances, ids = index.search(priv_embedding, num_nearest_neighbors)
    return np.sqrt(distances), ids

import torch
import numpy as np


def search(syn_embedding, priv_embedding, num_nearest_neighbors, mode):
    """Compute the nearest neighbors of the private embedding in the synthetic embedding using torch.

    :param syn_embedding: The synthetic embedding
    :type syn_embedding: np.ndarray
    :param priv_embedding: The private embedding
    :type priv_embedding: np.ndarray
    :param num_nearest_neighbors: The number of nearest neighbors to search
    :type num_nearest_neighbors: int
    :param mode: The distance metric to use for finding the nearest neighbors. It should be one of the following:
            "l2" (l2 distance), "cos_sim" (cosine similarity)
    :type mode: str
    :raises ValueError: If the mode is unknown
    :return: The distances and indices of the nearest neighbors
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    if mode.lower() == "l2":
        metric = "l2"
    else:
        raise ValueError(f"Torch backend only supports 'l2' distance metric")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    priv_tensor = torch.tensor(priv_embedding, dtype=torch.float32, device=device)
    syn_tensor = torch.tensor(syn_embedding, dtype=torch.float32, device=device)

    distances = torch.cdist(priv_tensor, syn_tensor, p=2)
    distances, ids = torch.topk(distances, k=num_nearest_neighbors, dim=1, largest=False)

    # convert back to numpy
    distances = distances.cpu().numpy()
    ids = ids.cpu().numpy()

    return distances, ids

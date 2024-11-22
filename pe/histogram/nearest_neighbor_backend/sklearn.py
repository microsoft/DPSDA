from sklearn.neighbors import NearestNeighbors


def search(syn_embedding, priv_embedding, num_nearest_neighbors, mode):
    """Compute the nearest neighbors of the private embedding in the synthetic embedding using sklearn.

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
    elif mode.lower() == "cos_sim":
        metric = "cosine"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    nn = NearestNeighbors(n_neighbors=num_nearest_neighbors, metric=metric, algorithm="brute", n_jobs=-1)
    nn.fit(syn_embedding)
    distances, ids = nn.kneighbors(priv_embedding)
    return distances, ids

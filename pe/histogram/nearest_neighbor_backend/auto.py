import traceback
from pe.logging import execution_logger


def search(syn_embedding, priv_embedding, num_nearest_neighbors, mode):
    """Compute the nearest neighbors of the private embedding in the synthetic embedding using Faiss. If Faiss is not
    installed or an error occurs, fall back to the sklearn backend.

    :param syn_embedding: The synthetic embedding
    :type syn_embedding: np.ndarray
    :param priv_embedding: The private embedding
    :type priv_embedding: np.ndarray
    :param num_nearest_neighbors: The number of nearest neighbors to search
    :type num_nearest_neighbors: int
    :param mode: The distance metric to use for finding the nearest neighbors. It should be one of the following:
            "l2" (l2 distance), "cos_sim" (cosine similarity), "ip" (inner product, not supported by sklearn)
    :type mode: str
    :raises ValueError: If the mode is unknown
    :return: The distances and indices of the nearest neighbors
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    try:
        execution_logger.info("Using faiss backend for nearest neighbor search")
        from pe.histogram.nearest_neighbor_backend.faiss import search

        return search(syn_embedding, priv_embedding, num_nearest_neighbors, mode)
    except Exception as e:
        execution_logger.error(f"Error using faiss backend for nearest neighbor search: {e}")
        execution_logger.error(traceback.format_exc())
        execution_logger.info(
            "Please check the installation of the Faiss library: "
            "https://microsoft.github.io/DPSDA/getting_started/installation.html#faiss"
        )
        execution_logger.info("Using sklearn backend for nearest neighbor search")
        from pe.histogram.nearest_neighbor_backend.sklearn import search

        return search(syn_embedding, priv_embedding, num_nearest_neighbors, mode)

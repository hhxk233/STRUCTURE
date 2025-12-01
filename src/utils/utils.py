import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from loguru import logger

try:
    import faiss

    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


SEED = 42


def set_seeds(seed: int = SEED) -> None:
    """
    Set seed for various random generators.

    RandomGenerators affected: ``HASHSEED``, ``random``, ``torch``, ``torch.cuda``,
    ``numpy.random``
    :param seed: Integer seed to set random generators to
    """
    if not isinstance(seed, int):
        raise ValueError(f"Expect seed to be an integer, but got {type(seed)}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_library_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def walk_and_collect(base_path: str, extensions: Sequence[str]):
    if not isinstance(base_path, str) or not isinstance(extensions, Sequence):
        raise TypeError(
            f"Expected base_path of type str or extensions of type sequence of"
            f" strings, but got {type(base_path)} and {type(extensions)}."
        )
    return [
        os.path.join(path, name)
        for path, _, files in os.walk(base_path)
        for name in files
        if any(name.endswith(s) for s in extensions)
    ]


def set_requires_grad(model: torch.nn.Module, val: bool) -> None:
    """
    Set all parameters of the module to require gradients or not.

    :param model: Model to set gradient option for
    :param val: Boolean to enable or disable autograd gradients
    """
    for p in model.parameters():
        p.requires_grad = val


def has_batchnorms(model: torch.nn.Module) -> bool:
    """
    Check if model has batch normalization layers.

    :param model: Model to check
    :return: True if batch norm layers are present, False otherwise
    """
    bn_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
    )
    for _, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False


def get_available_torch_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def are_embeddings_normalized(embeddings, epsilon=1e-6):
    """
    Check if embeddings are already normalized (have unit L2 norm).

    Args:
        embeddings: Tensor of shape (..., embedding_dim)
        epsilon: Tolerance for numerical precision

    Returns:
        bool: True if embeddings are normalized, False otherwise
    """
    # calculate the L2 norm along the embedding dimension
    norms = torch.norm(embeddings, p=2, dim=-1)
    # check if all norms are approximately 1
    return torch.all((norms - 1.0).abs() < epsilon).item()


def safe_normalize(
    embeddings: torch.Tensor, p: int = 2, dim: int = -1, epsilon: float = 1e-12
):
    """
    Normalize embeddings only if they aren't already normalized.

    Args:
        embeddings: Tensor of shape (..., embedding_dim)
        dim: Dimension along which to normalize
        epsilon: Small constant for numerical stability

    Returns:
        Normalized embeddings
    """
    if are_embeddings_normalized(embeddings):
        return embeddings
    else:
        return torch.nn.functional.normalize(embeddings, p=p, dim=dim, eps=epsilon)


def log_spherical_embedding_stats(embeddings: torch.Tensor, log_prefix: str = ""):
    # NOTE: since all embeddings are normalized we need different (spherical) measures
    # R: Norm of the mean vector (range [0, 1]); high R means vectors are aligned.
    # Spherical Variance: 1 - R; high value means more dispersion.
    # Variance of Cosine Similarities: Spread in pairwise cosine similarities (from -1 to 1).
    # Variance of Pairwise Angles: Spread in angles (in radians, range 0 to π) between vectors.
    embeddings = safe_normalize(embeddings, p=2, dim=1)

    mean_resultant = torch.mean(embeddings, dim=0)
    R = torch.norm(mean_resultant)
    spherical_variance = 1 - R

    similarity_matrix = torch.mm(embeddings, embeddings.t())
    triu_indices = torch.triu_indices(
        embeddings.shape[0], embeddings.shape[0], offset=1
    )
    pairwise_similarities = similarity_matrix[triu_indices[0], triu_indices[1]]

    cosine_similarity_variance = torch.var(pairwise_similarities)
    # Compute angles in radians by taking the arccos of the cosine similarities.
    # Clamp values to [-1, 1] to ensure numerical stability.
    pairwise_angles = torch.acos(torch.clamp(pairwise_similarities, -1.0, 1.0))
    angle_variance = torch.var(pairwise_angles)

    logger.debug(
        f"{log_prefix} Latent Stats - "
        f"Mean Resultant Length (R): {R:.4f}, "
        f"Spherical Variance: {spherical_variance:.4f}, "
        f"Variance of cosine similarities: {cosine_similarity_variance:.4f}, "
        f"Variance of pairwise angles: {angle_variance:.4f}"
    )


def set_transform_dataset(dataset, image_transform):
    def _set_transform(dataset, image_transform):
        if hasattr(dataset, "transform"):
            dataset.transform = image_transform
        if hasattr(dataset, "transforms"):
            dataset.transforms = lambda image, label: (
                image_transform(image),
                label,
            )

    _set_transform(dataset, image_transform)
    if hasattr(dataset, "dataset"):
        _set_transform(dataset.dataset, image_transform)
    if hasattr(dataset, "datasets"):
        for d in dataset.datasets:
            _set_transform(d, image_transform)


def _knn_graph(x: torch.Tensor, k: int, use_approx: bool = False) -> torch.Tensor:
    """
    Compute k-nearest neighbor indices for each point in x (excluding self).
    Supports exact (torch.cdist) or approximate (FAISS) search.

    Args:
        x (torch.Tensor): Tensor of shape (n, d).
        k (int): Number of neighbors.
        use_approx (bool): If True and FAISS is available, use FAISS GPU search.

    Returns:
        torch.Tensor: Tensor of shape (n, k) with neighbor indices.
    """
    n, d = x.shape
    device = x.device

    if use_approx and _FAISS_AVAILABLE:
        # FAISS expects CPU numpy arrays
        xb = x.cpu().numpy().astype("float32")
        index = faiss.IndexFlatL2(d)
        # send to GPU if input is on CUDA
        if device.type == "cuda":
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, device.index or 0, index)
        index.add(xb)
        distances, indices = index.search(xb, k + 1)
        # drop self (first column)
        return torch.from_numpy(indices[:, 1:]).to(device)
    else:
        # exact k-NN via pairwise distances
        dist = torch.cdist(x, x, p=2)
        idx = dist.argsort(dim=1)[:, : k + 1]
        return idx[:, 1:]


def _get_rank_matrix(dist: torch.Tensor) -> torch.Tensor:
    """
    Compute inverse rank positions for each pair from a distance matrix.

    Args:
        dist (torch.Tensor): Pairwise distance matrix of shape (n, n).

    Returns:
        torch.Tensor: Rank matrix pos where pos[i,j] = rank of j in i's distances.
    """
    n = dist.size(0)
    _, order = torch.sort(dist, dim=1)
    pos = torch.empty_like(order)
    idx = torch.arange(n, device=dist.device)
    pos[idx.unsqueeze(1), order] = idx.unsqueeze(0)
    return pos


def trustworthiness(
    X: torch.Tensor, Z: torch.Tensor, k: int, use_approx: bool = False
) -> float:
    """
    Compute the Trustworthiness metric for embeddings Z relative to X.

    Args:
        X (torch.Tensor): Original embeddings, shape (n, d).
        Z (torch.Tensor): Aligned embeddings, shape (n, d).
        k (int): Neighborhood size.
        use_approx (bool): Use approximate k-NN if True.

    Returns:
        float: Trustworthiness in [0,1], higher is better.
    """
    n = X.size(0)
    device = X.device

    # k-NN graphs
    knn_X = _knn_graph(X, k, use_approx)
    knn_Z = _knn_graph(Z, k, use_approx)

    # rank matrix in original space
    dist_X = torch.cdist(X, X, p=2)
    pos_X = _get_rank_matrix(dist_X)

    # boolean neighbor masks
    mask_X = torch.zeros((n, n), dtype=torch.bool, device=device)
    mask_Z = torch.zeros((n, n), dtype=torch.bool, device=device)
    mask_X.scatter_(1, knn_X, True)
    mask_Z.scatter_(1, knn_Z, True)

    # violations: neighbors in Z but not in X
    violations = mask_Z & (~mask_X)

    # penalty = (rank_position + 1) - (k + 1) = rank - k
    penalty = (pos_X.float() - k).clamp(min=0)

    # sum penalties over violations
    penalty_sum = penalty[violations].sum()

    # normalization factor
    denom = n * k * (2 * n - 3 * k - 1)

    return (1.0 - 2.0 * penalty_sum / denom).item()


def continuity(
    X: torch.Tensor, Z: torch.Tensor, k: int, use_approx: bool = False
) -> float:
    """
    Compute the Continuity metric for embeddings Z relative to X.
    Equivalent to Trustworthiness(Z, X).

    Args:
        X (torch.Tensor): Original embeddings, shape (n, d).
        Z (torch.Tensor): Aligned embeddings, shape (n, d).
        k (int): Neighborhood size.
        use_approx (bool): Use approximate k-NN if True.

    Returns:
        float: Continuity in [0,1], higher is better.
    """
    return trustworthiness(Z, X, k, use_approx)

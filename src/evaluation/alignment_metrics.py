from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import silhouette_score

from src.utils.utils import safe_normalize


def _as_numpy_labels(labels: Sequence) -> np.ndarray:
    labels_np = np.asarray(list(labels))
    if labels_np.ndim != 1:
        labels_np = labels_np.reshape(-1)
    return labels_np


@torch.no_grad()
def compute_purity_score(
    embeddings: torch.Tensor,
    labels: Sequence,
    batch_size: int = 512,
) -> float:
    """
    Correspondence purity based on nearest-neighbor label agreement
    in the pooled latent space.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {embeddings.ndim}D")
    labels_np = _as_numpy_labels(labels)
    if len(labels_np) != embeddings.shape[0]:
        raise ValueError(
            f"Label count ({len(labels_np)}) does not match embeddings "
            f"({embeddings.shape[0]})"
        )

    embeddings = safe_normalize(embeddings.float(), p=2, dim=1)
    device = embeddings.device
    n = embeddings.shape[0]
    correct = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = embeddings[start:end]
        sims = batch @ embeddings.T
        row_ids = torch.arange(end - start, device=device)
        col_ids = torch.arange(start, end, device=device)
        sims[row_ids, col_ids] = -1e9
        nn_idx = sims.argmax(dim=1).cpu().numpy()
        correct += (labels_np[start:end] == labels_np[nn_idx]).sum()

    return float(correct) / float(n)


@torch.no_grad()
def compute_cross_modal_purity_score(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    labels: Sequence,
    batch_size: int = 512,
) -> float:
    """
    Cross-modal purity based on nearest-neighbor label agreement.
    Computes both image->text and text->image and returns the average.
    """
    if image_embeds.ndim != 2 or text_embeds.ndim != 2:
        raise ValueError(
            f"Expected 2D embeddings, got {image_embeds.ndim}D/{text_embeds.ndim}D"
        )
    if image_embeds.shape[0] != text_embeds.shape[0]:
        raise ValueError(
            "Image/text counts differ: "
            f"{image_embeds.shape[0]} vs {text_embeds.shape[0]}"
        )
    labels_np = _as_numpy_labels(labels)
    if len(labels_np) != image_embeds.shape[0]:
        raise ValueError(
            f"Label count ({len(labels_np)}) does not match embeddings "
            f"({image_embeds.shape[0]})"
        )

    image_embeds = safe_normalize(image_embeds.float(), p=2, dim=1)
    text_embeds = safe_normalize(text_embeds.float(), p=2, dim=1)
    device = image_embeds.device
    n = image_embeds.shape[0]
    correct_i2t = 0
    correct_t2i = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        img_batch = image_embeds[start:end]
        sims = img_batch @ text_embeds.T
        nn_idx = sims.argmax(dim=1).cpu().numpy()
        correct_i2t += (labels_np[start:end] == labels_np[nn_idx]).sum()

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        txt_batch = text_embeds[start:end]
        sims = txt_batch @ image_embeds.T
        nn_idx = sims.argmax(dim=1).cpu().numpy()
        correct_t2i += (labels_np[start:end] == labels_np[nn_idx]).sum()

    return float(correct_i2t + correct_t2i) / float(2 * n)


def compute_silhouette_score(
    embeddings: torch.Tensor,
    labels: Sequence,
    metric: str = "cosine",
    sample_size: Optional[int] = None,
    random_state: int = 42,
) -> float:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got {embeddings.ndim}D")
    labels_np = _as_numpy_labels(labels)
    if len(labels_np) != embeddings.shape[0]:
        raise ValueError(
            f"Label count ({len(labels_np)}) does not match embeddings "
            f"({embeddings.shape[0]})"
        )
    if len(np.unique(labels_np)) < 2:
        logger.warning("Silhouette requires at least 2 clusters; returning NaN.")
        return float("nan")

    embeddings = safe_normalize(embeddings.float(), p=2, dim=1)
    X = embeddings.cpu().numpy()
    if sample_size is not None and sample_size < len(labels_np):
        return float(
            silhouette_score(
                X,
                labels_np,
                metric=metric,
                sample_size=sample_size,
                random_state=random_state,
            )
        )
    return float(silhouette_score(X, labels_np, metric=metric))

from typing import Dict, List, Optional

import pandas as pd
import torch

from src.utils.utils import safe_normalize


def compute_ground_truth_mapping(df, image_column="image_name"):
    """
    Constructs a mapping from each row index to the set of indices that share the same image name.

    Args:
        df (pd.DataFrame): DataFrame containing at least the image_column.
        image_column (str): Column name to group on.

    Returns:
        dict: A mapping where each key is a row index and the value is a set of indices sharing the same image name.
    """
    # Group by the image column; groups is a dict: image_name -> pd.Index of rows.
    groups = df.groupby(image_column).groups
    ground_truth = {}
    for idx in df.index:
        image_name = df.loc[idx, image_column]
        ground_truth[idx] = set(groups[image_name])
    return ground_truth


def retrieval_metrics_df(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    df: Optional[pd.DataFrame] = None,
    image_column: str = "image_name",
    k_values: List[int] = [1, 5, 10],
    batch_size: int = 64,
) -> Dict[str, float]:
    """Recall@k, Precision@k and MAP@k for image→text retrieval."""
    image_embeds = safe_normalize(image_embeds, p=2, dim=1)
    text_embeds = safe_normalize(text_embeds, p=2, dim=1)

    # build ground-truth mapping:  idx -> set(idx, …)
    N = image_embeds.size(0)
    if df is not None:
        ground_truth = compute_ground_truth_mapping(df, image_column)
    else:
        ground_truth = {i: {i} for i in range(N)}

    # every query must have ≥1 relevant item
    assert all(
        len(v) for v in ground_truth.values()
    ), "Some queries have no ground truth!"

    max_k = max(k_values)
    recall_hits = {k: 0.0 for k in k_values}
    precision_sum = {k: 0.0 for k in k_values}
    map_sum = {k: 0.0 for k in k_values}

    # batched similarity and metrics
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        sims = image_embeds[start:end] @ text_embeds.T  # [B, N]
        _, topk = torch.topk(sims, k=max_k, dim=1)  # indices

        for row, ranked_tensor in enumerate(topk):
            q_idx = start + row
            gt_set = ground_truth[q_idx]
            ranked = ranked_tensor.tolist()
            rel_flags = [idx in gt_set for idx in ranked]  # bool list

            for k in k_values:
                rel_in_k = sum(rel_flags[:k])

                # Recall@k (hit-rate)
                if rel_in_k > 0:
                    recall_hits[k] += 1

                # Precision@k
                precision_sum[k] += rel_in_k / k

                # AP@k (average precision truncated at k)
                hits, ap = 0, 0.0
                for r, is_rel in enumerate(rel_flags[:k], start=1):
                    if is_rel:
                        hits += 1
                        ap += hits / r
                ap /= min(len(gt_set), k)  # denom cap at k
                map_sum[k] += ap

    # macro-averages
    metrics = {}
    for k in k_values:
        metrics[f"R@{k}"] = recall_hits[k] / N
        metrics[f"P@{k}"] = precision_sum[k] / N
        metrics[f"MAP@{k}"] = map_sum[k] / N

    return metrics

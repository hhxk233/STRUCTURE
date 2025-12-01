import argparse
import os
from pprint import pprint
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import trange
from tqdm.auto import tqdm

import src.utils.alignment_utils as utils
import src.utils.metrics as metrics
from src.models.tasks import get_models


def prepare_features(feats, q=0.95, exact=False):
    """
    Prepare features by removing outliers and normalizing.

    Args:
        feats: a torch tensor of any share
        q: the quantile to remove outliers
    Returns:
        feats: a torch tensor of the same shape as the input
    """
    feats = metrics.remove_outliers(feats.float(), q=q, exact=exact)
    return feats.cuda()


def compute_score(
    x_feats,
    y_feats,
    metric="mutual_knn",
    topk: Union[str, int] = 10,
    normalize=True,
    show_progress: bool = True,
    **metric_kwargs,
):
    """
    Use different layer combinations of x_feats and y_feats to find the best alignment.

    Args:
        x_feats: a torch tensor of shape N x L x D
        y_feats: a torch tensor of shape N x L x D
    Returns:
        best_alignment_score: the best alignment score
        best_alignment: the indices of the best alignment
    """
    assert x_feats.shape[0] == y_feats.shape[0]
    n_samples = x_feats.shape[0]
    if type(topk) is str:
        if topk == "sqrt":
            topk = int(np.ceil(np.sqrt(n_samples)))
        elif topk == "sturges":
            topk = int(np.ceil(np.log2(n_samples) + 1))
        elif topk == "rice":
            topk = int(np.ceil(2 * n_samples ** (1 / 3)))
        else:
            raise ValueError(topk)

    best_alignment_indices = None
    best_alignment_score = -1
    alignment_list = []

    # if we only have the last layer
    if len(x_feats.shape) == 2:
        x_feats = x_feats.unsqueeze(1)
    if len(y_feats.shape) == 2:
        y_feats = y_feats.unsqueeze(1)

    for i in trange(-1, x_feats.shape[1], disable=not show_progress):
        x = x_feats.flatten(1, 2) if i == -1 else x_feats[:, i, :]

        for j in range(-1, y_feats.shape[1]):
            y = y_feats.flatten(1, 2) if j == -1 else y_feats[:, j, :]

            kwargs = {} | metric_kwargs
            if "knn" in metric:
                kwargs["topk"] = topk

            if normalize:
                x = F.normalize(x, p=2, dim=-1)
                y = F.normalize(y, p=2, dim=-1)

            score = metrics.AlignmentMetrics.measure(metric, x, y, **kwargs)
            alignment_list.append({"indices": (i, j), "alignment_score": score})

            if score > best_alignment_score:
                best_alignment_score = score
                best_alignment_indices = (i, j)

    return best_alignment_score, best_alignment_indices, alignment_list


def compute_alignment(x_feat_paths, y_feat_paths, metric, topk, precise=True):
    """
    Compute alignment of features.

    Args:
        x_feat_paths: list of paths to x features
        y_feat_paths: list of paths to y features
        metric: the metric to use
        topk: the number of nearest neighbors to use (specific to knn metrics)
        precise: if true use exact quantiling. (helpful to set to false if running on cpu)
            this is more of a feature to speed up matmul if using float32
            used in measure_alignment.py
    Returns:
        alignment_scores: a numpy array of shape len(x_feat_paths) x len(y_feat_paths)
        alignment_indices: a numpy array of shape len(x_feat_paths) x len(y_feat_paths) x 2
    """

    os.makedirs(args.output_dir, exist_ok=True)

    symmetric_metric = x_feat_paths == y_feat_paths
    if metric == "cycle_knn":
        symmetric_metric = False

    alignment_scores = np.zeros((len(x_feat_paths), len(y_feat_paths)))
    alignment_indices = np.zeros((len(x_feat_paths), len(y_feat_paths), 2))
    df_alignment = pd.DataFrame(
        columns=[
            "x_feature",
            "y_feature",
            "x_num_params",
            "y_num_params",
            "x_loss",
            "x_bpb",
            "best_score",
            "best_indices",
            "all_alignments",
            "matrix_entropies",
        ]
    )

    pbar = tqdm(total=len(y_feat_paths) * len(x_feat_paths))
    for i, x_fp in enumerate(x_feat_paths):
        all_data = torch.load(x_fp, map_location="cpu")
        x_feats = prepare_features(
            all_data["feats"].float(),
            exact=precise,
        )
        x_num_params = all_data["num_params"]
        x_loss, x_bpb = np.nan, np.nan
        if "loss" in all_data.keys() and all_data["loss"] is not None:
            x_loss = all_data["loss"].item()
        if "bpb" in all_data.keys() and all_data["bpb"] is not None:
            x_bpb = all_data["bpb"].item()

        for j, y_fp in enumerate(y_feat_paths):
            if symmetric_metric:
                if i > j:
                    pbar.update(1)
                    continue

            y_feats = prepare_features(
                torch.load(y_fp, map_location="cuda:0")["feats"].float(),
                exact=precise,
            )
            y_num_params = torch.load(y_fp, map_location="cpu")["num_params"]

            n_samples = len(x_feats)
            if type(topk) is str:
                if topk == "sqrt":
                    topk = int(np.ceil(np.sqrt(n_samples)))
                elif topk == "sturges":
                    topk = int(np.ceil(np.log2(n_samples) + 1))
                elif topk == "rice":
                    topk = int(np.ceil(2 * n_samples ** (1 / 3)))
                else:
                    raise ValueError(topk)

            best_score, best_indices, alignment_list = compute_score(
                x_feats=y_feats,
                y_feats=x_feats,
                metric=metric,
                topk=topk,
            )

            # matrix entropies for both and all layers
            d_matrix_entropies = {}
            for feat_name, feats in [("x_feats", x_feats), ("y_feats", y_feats)]:
                l_matrix_entropies = []
                for layer_idx in range(feats.shape[1]):
                    matrix_entropy = utils.matrix_entropy(feats[:, layer_idx, :])
                    l_matrix_entropies.append((layer_idx, matrix_entropy))
                d_matrix_entropies[feat_name] = l_matrix_entropies

            df_alignment.loc[len(df_alignment)] = [
                x_feat_paths[i],
                y_feat_paths[j],
                x_num_params,
                y_num_params,
                x_loss,
                x_bpb,
                best_score,
                best_indices,
                alignment_list,
                d_matrix_entropies,
            ]
            alignment_scores[i, j] = best_score
            alignment_indices[i, j] = best_indices

            if symmetric_metric:
                alignment_scores[j, i] = best_score
                alignment_indices[j, i] = best_indices[::-1]

            pbar.update(1)

            del y_feats
            torch.cuda.empty_cache()

    return alignment_scores, alignment_indices, df_alignment


if __name__ == "__main__":
    """
    recommended to use llm as modality_x since it will load each LLM features once
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="minhuh/prh")
    parser.add_argument("--subset", type=str, default="wit_1024")

    parser.add_argument(
        "--modality_x",
        type=str,
        default="all",
        choices=["vision", "language", "all"],
    )
    parser.add_argument("--prompt_x", action="store_true")
    parser.add_argument("--pool_x", type=str, default=None, choices=["avg", "cls"])

    parser.add_argument(
        "--modality_y",
        type=str,
        default="all",
        choices=["vision", "language", "all"],
    )
    parser.add_argument("--prompt_y", action="store_true")
    parser.add_argument("--pool_y", type=str, default=None, choices=["avg", "cls"])

    parser.add_argument("--modelset", type=str, default="val", choices=["val", "test"])
    parser.add_argument(
        "--metric",
        type=str,
        default="mutual_knn",
        choices=metrics.AlignmentMetrics.SUPPORTED_METRICS,
    )
    parser.add_argument("--topk", type=str, default=10)

    parser.add_argument("--input_dir", type=str, default="./results/features")
    parser.add_argument("--output_dir", type=str, default="./results/alignment")
    parser.add_argument("--precise", action="store_true")
    parser.add_argument("--force_remake", action="store_true")

    args = parser.parse_args()

    if not args.precise:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    try:
        args.topk = int(args.topk)
    except ValueError:
        pass

    save_path = utils.to_alignment_filename(
        args.output_dir,
        args.dataset,
        args.modelset,
        args.modality_x,
        args.pool_x,
        args.prompt_x,
        args.modality_y,
        args.pool_y,
        args.prompt_y,
        args.metric,
        args.topk,
    )

    if os.path.exists(save_path) and not args.force_remake:
        print(f"alignment already exists at {save_path}")
        exit()

    llm_models, lvm_models = get_models(
        args.modelset,
        modality="all",
    )
    if args.modality_x == "language":
        models_x = llm_models
    elif args.modality_x == "vision":
        models_x = lvm_models
    else:
        raise ValueError(f"Unknown modality: {args.modality_x}")

    if args.modality_y == "language":
        models_y = llm_models
    elif args.modality_y == "vision":
        models_y = lvm_models
    else:
        raise ValueError(f"Unknown modality: {args.modality_y}")

    models_x_paths = [
        utils.to_feature_filename(
            args.input_dir,
            args.dataset,
            args.subset,
            m,
            args.pool_x,
            args.prompt_x,
        )
        for m in models_x
    ]
    models_y_paths = [
        utils.to_feature_filename(
            args.input_dir,
            args.dataset,
            args.subset,
            m,
            args.pool_y,
            args.prompt_y,
        )
        for m in models_y
    ]

    for fn in models_x_paths + models_y_paths:
        assert os.path.exists(fn), fn

    print(f"dataset:\t{args.dataset}")
    print(f"metric: \t{args.metric}")
    if "knn" in args.metric:
        print(f"topk:\t{args.topk}")

    print("models_x_paths:")
    pprint(models_x_paths)
    print("\nmodels_y_paths:")
    pprint(models_y_paths)

    print("\nmeasuring alignment")
    alignment_scores, alignment_indices, df_alignment = compute_alignment(
        models_x_paths,
        models_y_paths,
        args.metric,
        args.topk,
        args.precise,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, {"scores": alignment_scores, "indices": alignment_indices})
    df_alignment.to_csv(save_path.replace(".npy", ".csv"), index=False)
    print(f"saved to {save_path}")

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _find_run_dir(results_dir: Path, run_tag: str) -> Path | None:
    matches = [p for p in results_dir.iterdir() if p.is_dir() and run_tag in p.name]
    if not matches:
        return None
    # Prefer exact suffix match (new runs use WANDB_NAME=run_tag)
    exact = [p for p in matches if p.name.endswith(run_tag)]
    candidates = exact or matches
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_result_file(run_dir: Path, keyword: str) -> Path | None:
    candidates = sorted(run_dir.glob(f"*{keyword}*.csv"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _read_metric(df: pd.DataFrame, key: str) -> float | None:
    if key not in df.columns or df.empty:
        return None
    return float(df.iloc[-1][key])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="/home/yuheng/task/STRUCTURE/results")
    p.add_argument("--out_csv", type=str, default="/home/yuheng/task/STRUCTURE/results/benchmark_summary.csv")
    p.add_argument("--subsets", nargs="+", default=["2k", "5k", "10k"])
    p.add_argument(
        "--methods",
        nargs="+",
        default=["csa", "mlp", "clip", "clip_raw", "clip_structure"],
    )
    p.add_argument("--dims", nargs="+", type=int, default=[128, 256])
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    for subset in args.subsets:
        for method in args.methods:
            if method == "clip_raw":
                run_dir = None
                run_tag = None
                for dim in args.dims:
                    candidate = f"clip_coco_{subset}_raw{dim}"
                    run_dir = _find_run_dir(results_dir, candidate)
                    if run_dir is not None:
                        run_tag = candidate
                        break
                if run_dir is None:
                    print(f"[warn] missing run dir for clip_coco_{subset}_raw")
                    continue
                model_name = f"clip_coco_{subset}_raw"
                zero_shot_file = _find_result_file(run_dir, "zero_shot_results")
                retrieval_file = _find_result_file(run_dir, "retrieval_results")

                zero_shot_df = (
                    pd.read_csv(zero_shot_file) if zero_shot_file else pd.DataFrame()
                )
                retrieval_df = (
                    pd.read_csv(retrieval_file) if retrieval_file else pd.DataFrame()
                )

                cifar_acc = _read_metric(zero_shot_df, "cifar10/top1_acc_micro")
                tiny_acc = _read_metric(zero_shot_df, "tiny_imagenet/top1_acc_micro")
                acc_vals = [v for v in (cifar_acc, tiny_acc) if v is not None]
                accuracy = sum(acc_vals) / len(acc_vals) if acc_vals else None

                r1_avg = _read_metric(retrieval_df, "coco/R@1-avg")
                if r1_avg is None:
                    i2t = _read_metric(retrieval_df, "coco/I2T-R@1")
                    t2i = _read_metric(retrieval_df, "coco/T2I-R@1")
                    if i2t is not None and t2i is not None:
                        r1_avg = (i2t + t2i) / 2

                row = {
                    "Model": model_name,
                    "Purity": _read_metric(retrieval_df, "coco/Purity"),
                    "Silhouette": _read_metric(retrieval_df, "coco/Silhouette"),
                    "Accuracy": accuracy,
                    "Accuracy_CIFAR10": cifar_acc,
                    "Accuracy_TinyImageNet": tiny_acc,
                    "Retrieval": r1_avg,
                    "Retrieval_I2T_R1": _read_metric(retrieval_df, "coco/I2T-R@1"),
                    "Retrieval_T2I_R1": _read_metric(retrieval_df, "coco/T2I-R@1"),
                }
                rows.append(row)
                continue
            for dim in args.dims:
                if method == "clip":
                    run_tag = f"clip_coco_{subset}_pca{dim}"
                elif method == "clip_structure":
                    run_tag = f"clip_structure_coco_{subset}_d{dim}"
                else:
                    run_tag = f"coco_{subset}_{method}_d{dim}"
                run_dir = _find_run_dir(results_dir, run_tag)
                if run_dir is None:
                    print(f"[warn] missing run dir for {run_tag}")
                    continue
                zero_shot_file = _find_result_file(run_dir, "zero_shot_results")
                retrieval_file = _find_result_file(run_dir, "retrieval_results")

                zero_shot_df = pd.read_csv(zero_shot_file) if zero_shot_file else pd.DataFrame()
                retrieval_df = pd.read_csv(retrieval_file) if retrieval_file else pd.DataFrame()

                cifar_acc = _read_metric(zero_shot_df, "cifar10/top1_acc_micro")
                tiny_acc = _read_metric(zero_shot_df, "tiny_imagenet/top1_acc_micro")
                acc_vals = [v for v in (cifar_acc, tiny_acc) if v is not None]
                accuracy = sum(acc_vals) / len(acc_vals) if acc_vals else None

                r1_avg = _read_metric(retrieval_df, "coco/R@1-avg")
                if r1_avg is None:
                    i2t = _read_metric(retrieval_df, "coco/I2T-R@1")
                    t2i = _read_metric(retrieval_df, "coco/T2I-R@1")
                    if i2t is not None and t2i is not None:
                        r1_avg = (i2t + t2i) / 2

                row = {
                    "Model": run_tag,
                    "Purity": _read_metric(retrieval_df, "coco/Purity"),
                    "Silhouette": _read_metric(retrieval_df, "coco/Silhouette"),
                    "Accuracy": accuracy,
                    "Accuracy_CIFAR10": cifar_acc,
                    "Accuracy_TinyImageNet": tiny_acc,
                    "Retrieval": r1_avg,
                    "Retrieval_I2T_R1": _read_metric(retrieval_df, "coco/I2T-R@1"),
                    "Retrieval_T2I_R1": _read_metric(retrieval_df, "coco/T2I-R@1"),
                }
                rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()

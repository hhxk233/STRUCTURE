from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.alignment.cca_class import NormalizedCCA
from src.evaluation.alignment_metrics import compute_cross_modal_purity_score
from src.trainers.alignment_trainer import AlignmentTrainer


def _load_features(path: Path):
    data = torch.load(path, weights_only=False)
    feats = data["features"]
    df = data.get("dataframe")
    return feats, df


def _get_label_list(df):
    if df is None:
        raise ValueError("Missing dataframe in saved features.")
    if "image_id" in df.columns:
        return df["image_id"].astype(str).tolist()
    if "image_path" in df.columns:
        return df["image_path"].astype(str).tolist()
    if "image_name" in df.columns:
        return df["image_name"].astype(str).tolist()
    raise ValueError("No label column found in dataframe.")


def _find_feature_file(save_dir: Path, model_name: str, dataset_tag: str, suffix: str):
    base = AlignmentTrainer.get_model_name(model_name)
    pattern = f"{base}-{dataset_tag}-{suffix}.npy"
    matches = list(save_dir.glob(pattern))
    if matches:
        return matches[0]
    # fallback: try partial match
    matches = list(save_dir.glob(f"{base}-{dataset_tag}-*.npy"))
    return matches[0] if matches else None


def _compute_cross_modal_purity(image_feats, text_feats, labels, batch_size=256):
    return compute_cross_modal_purity_score(
        image_feats, text_feats, labels, batch_size=batch_size
    )


def _load_layer_comb(retrieval_csv: Path):
    import pandas as pd

    df = pd.read_csv(retrieval_csv)
    if df.empty:
        raise ValueError(f"Empty retrieval csv: {retrieval_csv}")
    layer_comb = df.iloc[-1]["layer_comb"]
    if isinstance(layer_comb, str):
        # "(10, 10)" -> (10,10)
        layer_comb = tuple(int(x.strip()) for x in layer_comb.strip("()").split(","))
    return layer_comb, df


def _update_retrieval_csv(retrieval_csv: Path, new_purity: float):
    import pandas as pd

    df = pd.read_csv(retrieval_csv)
    if "coco/Purity" not in df.columns:
        df["coco/Purity"] = np.nan
    df.loc[df.index[-1], "coco/Purity"] = new_purity
    df.to_csv(retrieval_csv, index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="/home/yuheng/task/STRUCTURE/results")
    p.add_argument("--features_dir", type=str, default="/home/yuheng/task/STRUCTURE/results/features")
    p.add_argument("--llm_model", type=str, default="sentence-transformers/all-roberta-large-v1")
    p.add_argument("--lvm_model", type=str, default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    p.add_argument("--subsets", nargs="+", default=["2k", "5k", "10k"])
    p.add_argument("--methods", nargs="+", default=["csa", "mlp"])
    p.add_argument("--dims", nargs="+", type=int, default=[128, 256])
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    features_dir = Path(args.features_dir)

    subset_to_ntrain = {"2k": 2000, "5k": 5000, "10k": 10000}
    eval_tag = "CocoCaptionDataset-captions_val2014-val2014-n5000"
    train_tag_tpl = "CocoCaptionDataset-captions_train2014-train2014-n{ntrain}"

    for subset in args.subsets:
        ntrain = subset_to_ntrain[subset]
        train_tag = train_tag_tpl.format(ntrain=ntrain)

        img_train_path = _find_feature_file(
            features_dir, args.lvm_model, train_tag, "train-cls"
        )
        txt_train_path = _find_feature_file(
            features_dir, args.llm_model, train_tag, "train-avg"
        )
        img_eval_path = _find_feature_file(
            features_dir, args.lvm_model, eval_tag, "eval-cls"
        )
        txt_eval_path = _find_feature_file(
            features_dir, args.llm_model, eval_tag, "eval-avg"
        )

        if not all([img_train_path, txt_train_path, img_eval_path, txt_eval_path]):
            print(f"[warn] missing features for subset {subset}, skipping")
            continue

        img_train, _ = _load_features(img_train_path)
        txt_train, _ = _load_features(txt_train_path)
        img_eval, df_eval = _load_features(img_eval_path)
        txt_eval, _ = _load_features(txt_eval_path)
        labels = _get_label_list(df_eval)

        for method in args.methods:
            for dim in args.dims:
                run_tag = f"coco_{subset}_{method}_d{dim}"
                run_dir = None
                for pth in results_dir.iterdir():
                    if pth.is_dir() and pth.name.endswith(run_tag):
                        run_dir = pth
                        break
                if run_dir is None:
                    print(f"[warn] missing run dir for {run_tag}")
                    continue

                retrieval_csv = next(run_dir.glob("*retrieval_results*.csv"), None)
                if retrieval_csv is None:
                    print(f"[warn] missing retrieval csv for {run_tag}")
                    continue

                (img_layer, txt_layer), _ = _load_layer_comb(retrieval_csv)

                if method == "csa":
                    cca = NormalizedCCA(
                        sim_dim=dim,
                        equal_weights=False,
                        use_reg=True,
                        lambda_rs=10.0,
                        lambda_cca_coeff=1.0e-2,
                        L=1,
                        tau=0.2,
                        refine_epochs=10,
                        lr=5.0e-4,
                        batch_size=8,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )
                    cca.fit_transform_train_data(
                        img_train[:, img_layer, :].float().numpy(),
                        txt_train[:, txt_layer, :].float().numpy(),
                    )
                    aligned_img, aligned_txt = cca.transform_data(
                        img_eval[:, img_layer, :].float().numpy(),
                        txt_eval[:, txt_layer, :].float().numpy(),
                    )
                    aligned_img = torch.tensor(aligned_img)
                    aligned_txt = torch.tensor(aligned_txt)
                else:
                    ckpt = next(run_dir.glob("**/checkpoint_last.pt"), None)
                    if ckpt is None:
                        print(f"[warn] missing checkpoint for {run_tag}")
                        continue
                    ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
                    align_img = ckpt_data["alignment_image"].eval()
                    align_txt = ckpt_data["alignment_text"].eval()
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    align_img = align_img.to(device)
                    align_txt = align_txt.to(device)
                    with torch.no_grad():
                        aligned_img = align_img(
                            img_eval[:, img_layer, :].float().to(device)
                        ).cpu()
                        aligned_txt = align_txt(
                            txt_eval[:, txt_layer, :].float().to(device)
                        ).cpu()

                purity = _compute_cross_modal_purity(aligned_img, aligned_txt, labels)
                _update_retrieval_csv(retrieval_csv, purity)
                print(f"[ok] {run_tag}: Purity={purity:.6f}")


if __name__ == "__main__":
    main()

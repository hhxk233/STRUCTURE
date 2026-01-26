#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

from src.alignment.alignment_factory import AlignmentFactory
from src.core.src.datasets.image_text_dataset import ImageTextDataset
from src.loss.clip_loss import CLIPLoss
from src.trainers.alignment_trainer import AlignmentTrainer


class MetaImageTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_file"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, row["caption"]


def _remap_image_paths(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "image_file" not in df.columns:
        return df
    sample_path = Path(df.iloc[0]["image_file"])
    if sample_path.exists():
        return df
    replacements = [
        ("/home/yuheng/ICML/data/coco/raw", "/home/yuheng/task/dataset/coco"),
        ("/home/yuheng/ICML/data/coco", "/home/yuheng/task/dataset/coco"),
    ]
    for old, new in replacements:
        if old in str(sample_path):
            df = df.copy()
            df["image_file"] = df["image_file"].str.replace(old, new, regex=False)
            break
    return df


def _load_or_extract_features(
    df: pd.DataFrame,
    image_encoder: str,
    text_encoder: str,
    batch_size: int,
    device: str,
):
    img_path = df.attrs.get("img_feat_path")
    txt_path = df.attrs.get("txt_feat_path")
    if img_path and txt_path and Path(img_path).exists() and Path(txt_path).exists():
        return np.load(img_path), np.load(txt_path)

    dummy_config = {
        "paths": {"save_path": "/home/yuheng/task/STRUCTURE/results"},
        "features": {"pool_img": "cls", "pool_txt": "avg"},
        "random_state": 0,
        "training": {"batch_size": batch_size},
        "evaluation": {"batch_size": batch_size, "num_workers": 4},
    }
    dataset = MetaImageTextDataset(df)
    trainer = AlignmentTrainer(
        config=dummy_config,
        train_dataset=DataLoader(dataset, batch_size=batch_size),
        val_dataset=DataLoader(dataset, batch_size=batch_size),
        llm_model_name=text_encoder,
        lvm_model_name=image_encoder,
        wandb_logging=False,
    )
    _, image_transform = trainer.get_lvm(lvm_model_name=image_encoder)
    dataset.transform = image_transform
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=ImageTextDataset.collate_fn,
    )
    img_feats = trainer.get_image_features(
        loader=loader,
        lvm_model_name=image_encoder,
        suffix="meta-train-cls",
        dataset_name="meta_train",
    ).float().cpu().numpy()
    txt_feats = trainer.get_text_features(
        loader=loader,
        llm_model_name=text_encoder,
        suffix="meta-train-avg",
        dataset_name="meta_train",
    ).float().cpu().numpy()
    return img_feats, txt_feats


def _select_overlapping_labels(
    df: pd.DataFrame, rng: np.random.Generator, num_labels: int
) -> List[int]:
    class_pairs: Dict[int, Tuple[str, str]] = {}
    for class_id, row in df.drop_duplicates("class_id").iterrows():
        class_pairs[int(row["class_id"])] = (str(row["catA_id"]), str(row["catB_id"]))

    class_ids = list(class_pairs.keys())
    rng.shuffle(class_ids)

    for start in range(len(class_ids)):
        picked = []
        cats_seen = set()
        for cid in class_ids[start:]:
            pair = class_pairs[cid]
            if not picked:
                picked.append(cid)
                cats_seen.update(pair)
            else:
                if pair[0] in cats_seen or pair[1] in cats_seen:
                    picked.append(cid)
                    cats_seen.update(pair)
            if len(picked) >= num_labels:
                return picked
    return class_ids[:num_labels]


def train_mlp_alignment(
    image_feats: torch.Tensor,
    text_feats: torch.Tensor,
    dim_alignment: int,
    device: str,
    n_epochs: int,
    batch_size: int,
    structure_lambda: float,
):
    image_dim = image_feats.shape[-1]
    text_dim = text_feats.shape[-1]
    alignment_image = AlignmentFactory.create(
        "ResLowRankHead", input_dim=image_dim, dim_alignment=dim_alignment
    ).to(device)
    alignment_text = AlignmentFactory.create(
        "ResLowRankHead", input_dim=text_dim, dim_alignment=dim_alignment
    ).to(device)

    loss_fn = CLIPLoss(
        normalize_latents=True,
        learnable_temperature=True,
        structure_lambda=structure_lambda,
    ).to(device)

    params = list(alignment_image.parameters()) + list(alignment_text.parameters())
    params += [p for p in loss_fn.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

    num_samples = image_feats.shape[0]
    for epoch in range(n_epochs):
        perm = torch.randperm(num_samples)
        for i in range(0, num_samples, batch_size):
            idx = perm[i : i + batch_size]
            img = image_feats[idx].to(device)
            txt = text_feats[idx].to(device)

            loss_fn.step()
            optimizer.zero_grad()
            img_aligned = alignment_image(img)
            txt_aligned = alignment_text(txt)
            loss = loss_fn(
                image_embeddings_aligned=img_aligned,
                text_embeddings_aligned=txt_aligned,
                image_embeddings_original=img,
                text_embeddings_original=txt,
            )["overall_loss"]
            loss.backward()
            optimizer.step()

    return alignment_image.cpu(), alignment_text.cpu()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subset_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_labels", type=int, default=4)
    p.add_argument("--min_samples", type=int, default=5)
    p.add_argument("--max_samples", type=int, default=10)
    p.add_argument("--dim_alignment", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--structure_lambda", type=float, default=10.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--image_encoder",
        type=str,
        default="facebook/dinov3-vitb16-pretrain-lvd1689m",
    )
    p.add_argument(
        "--text_encoder",
        type=str,
        default="sentence-transformers/all-roberta-large-v1",
    )
    args = p.parse_args()

    subset_dir = Path(args.subset_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(subset_dir / "meta_train.csv")
    df = _remap_image_paths(df)
    rng = np.random.default_rng(args.seed)
    picked_labels = _select_overlapping_labels(df, rng, args.num_labels)

    selected_rows = []
    for cid in picked_labels:
        rows = df[df["class_id"] == cid].sample(
            n=min(3, len(df[df["class_id"] == cid])),
            random_state=args.seed,
        )
        selected_rows.append(rows)
    selected = pd.concat(selected_rows).reset_index(drop=True)
    if len(selected) > args.max_samples:
        selected = selected.sample(n=args.max_samples, random_state=args.seed)
    if len(selected) < args.min_samples:
        selected = df.sample(n=args.min_samples, random_state=args.seed)

    img_feats, txt_feats = _load_or_extract_features(
        df=df,
        image_encoder=args.image_encoder,
        text_encoder=args.text_encoder,
        batch_size=args.batch_size,
        device=args.device,
    )
    if img_feats.ndim == 3:
        img_feats = img_feats[:, -1, :]
    if txt_feats.ndim == 3:
        txt_feats = txt_feats[:, -1, :]
    img_feats = torch.from_numpy(img_feats).float()
    txt_feats = torch.from_numpy(txt_feats).float()

    alignment_image, alignment_text = train_mlp_alignment(
        image_feats=img_feats,
        text_feats=txt_feats,
        dim_alignment=args.dim_alignment,
        device=args.device,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        structure_lambda=args.structure_lambda,
    )

    sample_indices = selected["idx"].astype(int).tolist()
    sample_img = alignment_image(img_feats[sample_indices]).detach().cpu().numpy()
    sample_txt = alignment_text(txt_feats[sample_indices]).detach().cpu().numpy()

    pca = PCA(n_components=2, random_state=args.seed)
    pca.fit(np.vstack([sample_img, sample_txt]))
    img_2d = pca.transform(sample_img)
    txt_2d = pca.transform(sample_txt)

    label_ids = selected["class_id"].astype(int).to_numpy()
    label_names = (
        selected["catA"].astype(str) + "+" + selected["catB"].astype(str)
    ).to_numpy()
    unique_labels = sorted(set(label_ids))
    color_map = {cid: i for i, cid in enumerate(unique_labels)}
    colors = [color_map[cid] for cid in label_ids]

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(img_2d[:, 0], img_2d[:, 1], c=colors, cmap="tab10")
    for i, name in enumerate(label_names):
        plt.text(img_2d[i, 0], img_2d[i, 1], name, fontsize=6, alpha=0.8)
    handles = []
    for cid in unique_labels:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=str(cid),
                markerfacecolor=scatter.cmap(scatter.norm(color_map[cid])),
                markersize=6,
            )
        )
    plt.legend(
        handles=handles,
        title="class_id",
        loc="best",
        fontsize=7,
        title_fontsize=8,
    )
    plt.title("Image embeddings (R_S+MLP)")
    plt.tight_layout()
    plt.savefig(out_dir / "mlp_images_2d.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 6))
    scatter = plt.scatter(txt_2d[:, 0], txt_2d[:, 1], c=colors, cmap="tab10")
    for i, name in enumerate(label_names):
        plt.text(txt_2d[i, 0], txt_2d[i, 1], name, fontsize=6, alpha=0.8)
    handles = []
    for cid in unique_labels:
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=str(cid),
                markerfacecolor=scatter.cmap(scatter.norm(color_map[cid])),
                markersize=6,
            )
        )
    plt.legend(
        handles=handles,
        title="class_id",
        loc="best",
        fontsize=7,
        title_fontsize=8,
    )
    plt.title("Text embeddings (R_S+MLP)")
    plt.tight_layout()
    plt.savefig(out_dir / "mlp_texts_2d.png", dpi=200)
    plt.close()

    selected.to_csv(out_dir / "mlp_selected_samples.csv", index=False)
    np.save(out_dir / "mlp_image_embeds.npy", sample_img)
    np.save(out_dir / "mlp_text_embeds.npy", sample_txt)


if __name__ == "__main__":
    main()

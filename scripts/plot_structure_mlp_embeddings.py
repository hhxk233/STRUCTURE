#!/usr/bin/env python
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import font_manager, pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerBase
from matplotlib import patheffects as path_effects
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from torch.utils.data import DataLoader, Dataset

from src.core.src.datasets.image_text_dataset import ImageTextDataset
from src.trainers.alignment_trainer import AlignmentTrainer
from src.utils.utils import safe_normalize

PALETTE = [
    "#9E0142",
    "#D53E4F",
    "#F46D43",
    "#FDAE61",
    "#FEE08B",
    "#ABDDA4",
    "#66C2A5",
    "#3288BD",
]
CHAT_MARKER = "💬"


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
    dataset_name = df.attrs.get("dataset_name", f"selected_n{len(df)}")
    img_feats = trainer.get_image_features(
        loader=loader,
        lvm_model_name=image_encoder,
        suffix=f"{dataset_name}-cls",
        dataset_name=dataset_name,
    ).float().cpu().numpy()
    txt_feats = trainer.get_text_features(
        loader=loader,
        llm_model_name=text_encoder,
        suffix=f"{dataset_name}-avg",
        dataset_name=dataset_name,
    ).float().cpu().numpy()
    return img_feats, txt_feats


def _load_coco_json(coco_root: Path, filename: str) -> dict:
    path = coco_root / "annotations" / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing COCO annotation file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def _find_coco_cycle(
    edge_to_images: Dict[Tuple[int, int], List[int]],
    neighbors: Dict[int, set],
) -> List[int]:
    for a in neighbors:
        for b in neighbors[a]:
            for c in neighbors[b]:
                if c in (a, b):
                    continue
                for d in neighbors[c]:
                    if d in (a, b, c):
                        continue
                    if a in neighbors[d]:
                        return [a, b, c, d]
    return []


def _select_overlapping_coco_images(
    coco_root: Path,
    num_labels: int,
    num_samples: int,
    seed: int,
):
    rng = np.random.default_rng(seed)
    instances = _load_coco_json(coco_root, "instances_train2014.json")
    captions = _load_coco_json(coco_root, "captions_train2014.json")

    cat_id_to_name = {c["id"]: c["name"] for c in instances["categories"]}
    image_id_to_file = {img["id"]: img["file_name"] for img in instances["images"]}

    image_to_cats: Dict[int, set] = {}
    for ann in instances["annotations"]:
        image_to_cats.setdefault(ann["image_id"], set()).add(ann["category_id"])

    edge_to_images: Dict[Tuple[int, int], List[int]] = {}
    neighbors: Dict[int, set] = {}
    for image_id, cats in image_to_cats.items():
        if len(cats) < 2:
            continue
        cats_sorted = sorted(cats)
        for i in range(len(cats_sorted)):
            for j in range(i + 1, len(cats_sorted)):
                a, b = cats_sorted[i], cats_sorted[j]
                edge_to_images.setdefault((a, b), []).append(image_id)
                neighbors.setdefault(a, set()).add(b)
                neighbors.setdefault(b, set()).add(a)

    cycle = _find_coco_cycle(edge_to_images, neighbors)
    if not cycle:
        raise RuntimeError("Unable to find 4-category overlap cycle in COCO.")
    if len(cycle) > num_labels:
        cycle = cycle[:num_labels]

    edges = list(zip(cycle, cycle[1:] + cycle[:1]))
    edges = [(min(a, b), max(a, b)) for a, b in edges]

    per_edge = max(1, num_samples // len(edges))
    selected_rows = []
    used_images = set()
    for edge_idx, edge in enumerate(edges):
        candidates = edge_to_images.get(edge, [])
        rng.shuffle(candidates)
        for image_id in candidates:
            if image_id in used_images:
                continue
            selected_rows.append((image_id, edge_idx, edge))
            used_images.add(image_id)
            if len([r for r in selected_rows if r[1] == edge_idx]) >= per_edge:
                break

    remaining = num_samples - len(selected_rows)
    if remaining > 0:
        all_candidates = []
        for edge_idx, edge in enumerate(edges):
            for image_id in edge_to_images.get(edge, []):
                all_candidates.append((image_id, edge_idx, edge))
        rng.shuffle(all_candidates)
        for image_id, edge_idx, edge in all_candidates:
            if image_id in used_images:
                continue
            selected_rows.append((image_id, edge_idx, edge))
            used_images.add(image_id)
            remaining -= 1
            if remaining <= 0:
                break

    if len(selected_rows) < num_samples:
        raise RuntimeError("Not enough overlapping COCO images for selection.")

    cap_map: Dict[int, List[str]] = {}
    for ann in captions["annotations"]:
        cap_map.setdefault(ann["image_id"], []).append(ann["caption"])

    records = []
    meta_records = []
    for row_idx, (image_id, edge_idx, edge) in enumerate(selected_rows[:num_samples]):
        file_name = image_id_to_file[image_id]
        image_file = str(coco_root / "train2014" / file_name)
        cap_list = cap_map.get(image_id, [])[:5]
        caption_concat = " ".join(cap_list) if cap_list else ""
        cats = sorted(list(image_to_cats.get(image_id, set())))
        cat_names = [cat_id_to_name[cid] for cid in cats]
        cat_a, cat_b = edge
        records.append(
            {
                "idx": row_idx,
                "image_id": image_id,
                "image_file": image_file,
                "caption": caption_concat,
                "caption_concat": caption_concat,
                "catA_pair": cat_id_to_name[cat_a],
                "catB_pair": cat_id_to_name[cat_b],
                "class_id": edge_idx,
            }
        )
        meta_records.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "pair_cat_ids": f"{cat_a},{cat_b}",
                "pair_cat_names": f"{cat_id_to_name[cat_a]}+{cat_id_to_name[cat_b]}",
                "all_category_ids": ",".join(map(str, cats)),
                "all_category_names": ",".join(cat_names),
            }
        )

    df_selected = pd.DataFrame.from_records(records)
    df_meta = pd.DataFrame.from_records(meta_records)
    return df_selected, df_meta


def _build_caption_map(df: pd.DataFrame, max_caps: int = 5) -> Dict[str, str]:
    if "caption" not in df.columns:
        return {}
    cap_map = {}
    for img_path, group in df.groupby("image_file"):
        caps = [str(c) for c in group["caption"].dropna().tolist()]
        caps = caps[:max_caps]
        if caps:
            cap_map[str(img_path)] = " ".join(caps)
    return cap_map


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    parts = re.split(r"[.!?。！？]", text, maxsplit=1)
    return parts[0].strip()


def _truncate_words(text: str, max_words: int = 10) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def _wrap_caption(text: str) -> str:
    words = text.split()
    if len(words) <= 1:
        return text
    mid = max(1, len(words) // 2)
    return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])


def _ensure_ellipsis(text: str) -> str:
    stripped = text.rstrip()
    if stripped.endswith("..."):
        return stripped
    return stripped + "..."


def _select_overlapping_labels(
    df: pd.DataFrame, rng: np.random.Generator, num_labels: int
) -> List[int]:
    class_pairs: Dict[int, Tuple[str, str]] = {}
    for _, row in df.drop_duplicates("class_id").iterrows():
        class_id = int(row["class_id"])
        cat_a = str(row.get("catA_id", row.get("catA", "")))
        cat_b = str(row.get("catB_id", row.get("catB", "")))
        if cat_a == cat_b or not cat_a or not cat_b:
            continue
        class_pairs[class_id] = (cat_a, cat_b)

    class_ids = list(class_pairs.keys())
    if not class_ids:
        all_class_ids = df["class_id"].drop_duplicates().astype(int).tolist()
        rng.shuffle(all_class_ids)
        return all_class_ids[:num_labels]
    rng.shuffle(class_ids)

    for start in range(len(class_ids)):
        picked = []
        cats_seen = set()
        pair_seen = set()
        for cid in class_ids[start:]:
            pair = class_pairs[cid]
            pair_key = tuple(sorted(pair))
            if pair_key in pair_seen:
                continue
            if not picked:
                picked.append(cid)
                cats_seen.update(pair)
                pair_seen.add(pair_key)
            else:
                if pair[0] in cats_seen or pair[1] in cats_seen:
                    picked.append(cid)
                    cats_seen.update(pair)
                    pair_seen.add(pair_key)
            if len(picked) >= num_labels:
                return picked
    return class_ids[:num_labels]


def _select_images_per_class(
    df: pd.DataFrame,
    rng: np.random.Generator,
    class_ids: List[int],
    total_samples: int,
) -> pd.DataFrame:
    selected = []
    per_class = max(1, total_samples // len(class_ids))
    remaining = total_samples
    for cid in class_ids:
        class_rows = df[df["class_id"] == cid]
        if class_rows.empty:
            continue
        class_rows = class_rows.drop_duplicates(subset=["image_file"])
        take = min(per_class, len(class_rows))
        chosen = class_rows.sample(n=take, random_state=int(rng.integers(0, 1_000_000)))
        selected.append(chosen)
        remaining -= take
    if remaining > 0:
        leftovers = df[df["class_id"].isin(class_ids)]
        if selected:
            selected_df = pd.concat(selected)
            leftovers = leftovers.drop(selected_df.index, errors="ignore")
        leftovers = leftovers.drop_duplicates(subset=["image_file"])
        if not leftovers.empty:
            extra = leftovers.sample(
                n=min(remaining, len(leftovers)),
                random_state=int(rng.integers(0, 1_000_000)),
            )
            selected.append(extra)
    if not selected:
        return df.drop_duplicates(subset=["image_file"]).sample(
            n=min(total_samples, len(df)), random_state=0
        )
    selected_df = pd.concat(selected).reset_index(drop=True)
    if len(selected_df) > total_samples:
        selected_df = selected_df.sample(
            n=total_samples, random_state=int(rng.integers(0, 1_000_000))
        )
    return selected_df.reset_index(drop=True)


def _save_selected_images(
    df: pd.DataFrame, out_dir: Path
):
    out_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_dir.iterdir():
        if existing.is_file():
            existing.unlink()
    for idx, row in df.iterrows():
        img_path = Path(row["image_file"])
        if not img_path.exists():
            continue
        class_id = int(row["class_id"])
        cat_a = str(row.get("catA_pair", row.get("catA", "classA")))
        cat_b = str(row.get("catB_pair", row.get("catB", "classB")))
        class_name = f"{cat_a}+{cat_b}"
        dest_name = f"{idx:02d}_class{class_id}_{class_name}.jpg"
        dest_path = out_dir / dest_name
        image = Image.open(img_path).convert("RGB")
        image.save(dest_path)


def _save_figure_with_limit(fig, out_path: Path, max_bytes: int = 2_000_000):
    fig.savefig(out_path, dpi=220)
    if out_path.stat().st_size <= max_bytes:
        return
    jpg_path = out_path.with_suffix(".jpg")
    fig.savefig(jpg_path, dpi=220)
    image = Image.open(jpg_path).convert("RGB")
    image.save(jpg_path, quality=95, optimize=True, progressive=True)
    if jpg_path.stat().st_size <= max_bytes:
        out_path.unlink(missing_ok=True)
        return
    image.save(jpg_path, quality=90, optimize=True, progressive=True)
    out_path.unlink(missing_ok=True)


def _load_emoji_png(path_candidates: List[str]) -> Image.Image:
    for path in path_candidates:
        if not path:
            continue
        p = Path(path)
        if not p.exists():
            continue
        try:
            return Image.open(p).convert("RGBA")
        except Exception:
            continue
    raise ValueError("No usable emoji PNG found. Provide --emoji_png_path.")


def _tint_emoji_rgba(base: Image.Image, color: Tuple[float, float, float]) -> Image.Image:
    if isinstance(color, str):
        color = to_rgb(color)
    arr = np.array(base, dtype=np.uint8)
    alpha = arr[:, :, 3]
    rgb = np.zeros_like(arr[:, :, :3])
    rgb[:, :, 0] = int(color[0] * 255)
    rgb[:, :, 1] = int(color[1] * 255)
    rgb[:, :, 2] = int(color[2] * 255)
    out = np.dstack([rgb, alpha])
    return Image.fromarray(out, mode="RGBA")


def _outline_emoji_rgba(
    base: Image.Image, outline_px: int = 2, outline_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    alpha = np.array(base)[:, :, 3]
    expanded = Image.fromarray(alpha, mode="L").resize(
        (base.size[0] + outline_px * 2, base.size[1] + outline_px * 2),
        resample=Image.Resampling.NEAREST,
    )
    out = Image.new("RGBA", expanded.size, (0, 0, 0, 0))
    out_arr = np.array(out)
    out_arr[:, :, 3] = np.array(expanded)
    out_arr[:, :, 0:3] = np.array(outline_color, dtype=np.uint8)
    return Image.fromarray(out_arr, mode="RGBA")


def _parse_layer_combination_from_ckpt(ckpt_path: Path) -> Tuple[int, int] | None:
    match = re.search(r"\((\-?\d+),\s*(\-?\d+)\)", ckpt_path.parent.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _plot_aligned_samples(
    out_path: Path,
    img_points: np.ndarray,
    txt_points: np.ndarray,
    samples: pd.DataFrame,
    color_map: Dict[int, Tuple[float, float, float]],
    draw_links: bool = True,
    show_distances: bool = False,
    distance_fmt: str = "{:.2f}",
    thumb_size: int = 48,
    img_y_min: float = 0.55,
    img_y_max: float = 0.98,
    txt_y_min: float = 0.02,
    txt_y_max: float = 0.38,
    x_spread: float = 0.2,
    text_attract: float = 0.35,
    split_modalities: bool = False,
    emoji_png: Image.Image | None = None,
    emoji_px: int = 24,
    chat_marker_size: int = 12,
    image_alpha: float = 0.85,
    label_font_prop: font_manager.FontProperties | None = None,
    caption_font_prop: font_manager.FontProperties | None = None,
    caption_font_size: int = 16,
    legend_icon_zoom: float = 0.9,
    legend_title_size: int = 18,
):
    all_points = np.vstack([img_points, txt_points])
    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    img_points = (img_points - min_xy) / span
    txt_points = (txt_points - min_xy) / span
    if split_modalities:
        img_points[:, 1] = img_y_min + img_points[:, 1] * (img_y_max - img_y_min)
        txt_points[:, 1] = txt_y_min + txt_points[:, 1] * (txt_y_max - txt_y_min)
    if text_attract > 0:
        txt_points[:, 0] = (
            txt_points[:, 0] * (1.0 - text_attract)
            + img_points[:, 0] * text_attract
        )
    if x_spread > 0:
        order = np.argsort(img_points[:, 0])
        spread = np.linspace(-x_spread, x_spread, len(order))
        img_points[order, 0] = np.clip(img_points[order, 0] + spread, 0.02, 0.98)
        txt_points[order, 0] = np.clip(txt_points[order, 0] + spread, 0.02, 0.98)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(-0.1, 1.15)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("alignment dim 1", fontproperties=label_font_prop, fontsize=14)
    ax.set_ylabel("alignment dim 2", fontproperties=label_font_prop, fontsize=14)
    if label_font_prop is not None:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(label_font_prop)
            label.set_fontsize(14)
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)

    class HandlerImage(HandlerBase):
        def create_artists(
            self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
        ):
            image = orig_handle
            ab = AnnotationBbox(
                image,
                (xdescent + width / 2.0, ydescent + height / 2.0),
                frameon=False,
                box_alignment=(0.5, 0.5),
                xycoords=trans,
            )
            return [ab]

    handles = []
    labels = []

    for i, row in samples.iterrows():
        color = color_map[i]
        img_path = Path(row["image_file"])
        if img_path.exists():
            image = Image.open(img_path).convert("RGB")
            max_dim = max(image.size)
            zoom = thumb_size / max_dim
            im = OffsetImage(image, zoom=zoom)
            im.set_alpha(image_alpha)
            ab = AnnotationBbox(
                im,
                img_points[i],
                frameon=True,
                bboxprops=dict(edgecolor=color, linewidth=2, facecolor="none"),
                clip_on=True,
            )
            ax.add_artist(ab)

        if emoji_png is not None:
            outline = _outline_emoji_rgba(
                emoji_png, outline_px=18, outline_color=(255, 255, 255)
            )
            outline_zoom = (emoji_px * 1.15) / max(outline.size)
            outline_artist = AnnotationBbox(
                OffsetImage(outline, zoom=outline_zoom),
                txt_points[i],
                frameon=False,
                box_alignment=(0.5, 0.5),
                clip_on=True,
                zorder=5,
            )
            ax.add_artist(outline_artist)
            tinted = _tint_emoji_rgba(emoji_png, color)
            zoom = emoji_px / max(tinted.size)
            emoji_artist = AnnotationBbox(
                OffsetImage(tinted, zoom=zoom),
                txt_points[i],
                frameon=False,
                box_alignment=(0.5, 0.5),
                clip_on=True,
                zorder=6,
            )
            ax.add_artist(emoji_artist)
            legend_handle = OffsetImage(tinted, zoom=legend_icon_zoom)
        else:
            txt = ax.text(
                txt_points[i, 0],
                txt_points[i, 1],
                CHAT_MARKER,
                ha="center",
                va="center",
                color=color,
                fontsize=chat_marker_size,
                clip_on=True,
                zorder=6,
            )
            txt.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2.5, foreground="white"),
                    path_effects.Normal(),
                ]
            )
            legend_handle = plt.Line2D(
                [0],
                [0],
                marker=CHAT_MARKER,
                color="w",
                markerfacecolor=color,
                markeredgecolor="white",
                markersize=chat_marker_size,
                linewidth=0,
            )

        if draw_links:
            line = ax.plot(
                [img_points[i, 0], txt_points[i, 0]],
                [img_points[i, 1], txt_points[i, 1]],
                color=color,
                linewidth=2.0,
                alpha=0.75,
                zorder=10,
            )[0]
            line.set_path_effects(
                [
                    path_effects.Stroke(linewidth=4.0, foreground="white"),
                    path_effects.Normal(),
                ]
            )
            if show_distances:
                dx = img_points[i, 0] - txt_points[i, 0]
                dy = img_points[i, 1] - txt_points[i, 1]
                dist = float(np.hypot(dx, dy))
                ax.text(
                    (img_points[i, 0] + txt_points[i, 0]) / 2.0,
                    (img_points[i, 1] + txt_points[i, 1]) / 2.0,
                    distance_fmt.format(dist),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7),
                    zorder=11,
                )

        caption = str(row["caption_short"])
        handles.append(legend_handle)
        labels.append(caption)

    legend_prop = caption_font_prop.copy() if caption_font_prop is not None else None
    if legend_prop is not None:
        legend_prop.set_size(caption_font_size)
    title_font_prop = (
        label_font_prop.copy() if label_font_prop is not None else None
    )
    if title_font_prop is not None:
        title_font_prop.set_size(legend_title_size)
    ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=False,
        title="captions",
        prop=legend_prop,
        title_fontproperties=title_font_prop,
        handler_map={OffsetImage: HandlerImage()},
    )

    plt.subplots_adjust(bottom=0.08, left=0.08, right=0.62, top=0.98)
    _save_figure_with_limit(fig, out_path)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subset_dir", type=str, required=False)
    p.add_argument(
        "--coco_root",
        type=str,
        default="/home/yuheng/task/STRUCTURE/data/coco",
        help="Path to COCO root with annotations/ and train2014/.",
    )
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_labels", type=int, default=4)
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--dim_alignment", type=int, default=2)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--structure_lambda", type=float, default=10.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--tries", type=int, default=3)
    p.add_argument(
        "--comic_font_path",
        type=str,
        default=None,
        help="Path to Comic Sans MS .ttf (required if not installed system-wide).",
    )
    p.add_argument(
        "--emoji_png_path",
        type=str,
        default="/home/yuheng/task/STRUCTURE/noto-emoji/png/72/emoji_u1f4ac.png",
        help="Path to speech-balloon emoji PNG (U+1F4AC).",
    )
    p.add_argument("--thumb_size", type=int, default=84)
    p.add_argument("--img_y_min", type=float, default=0.55)
    p.add_argument("--img_y_max", type=float, default=0.98)
    p.add_argument("--txt_y_min", type=float, default=0.02)
    p.add_argument("--txt_y_max", type=float, default=0.38)
    p.add_argument("--x_spread", type=float, default=0.22)
    p.add_argument("--text_attract", type=float, default=0.35)
    p.add_argument(
        "--emoji_px",
        type=int,
        default=20,
        help="Pixel size for emoji icon overlay.",
    )
    p.add_argument(
        "--chat_marker_size",
        type=int,
        default=12,
        help="Font size for text chat marker.",
    )
    p.add_argument(
        "--image_alpha",
        type=float,
        default=0.85,
        help="Alpha transparency for image thumbnails.",
    )
    p.add_argument(
        "--caption_font_size",
        type=int,
        default=18,
        help="Font size for caption legend text.",
    )
    p.add_argument(
        "--legend_icon_zoom",
        type=float,
        default=0.75,
        help="Zoom factor for legend icons in caption list.",
    )
    p.add_argument(
        "--legend_title_size",
        type=int,
        default=20,
        help="Font size for the legend title.",
    )
    p.add_argument(
        "--label_font_path",
        type=str,
        default=None,
        help="Path to Calibri .ttf for axis labels/titles.",
    )
    p.add_argument(
        "--split_modalities",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, place image/text points into separate y bands.",
    )
    p.add_argument(
        "--draw_links",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw colored link between image and text points.",
    )
    p.add_argument(
        "--show_distances",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Annotate link with image-text distance.",
    )
    p.add_argument(
        "--distance_fmt",
        type=str,
        default="{:.2f}",
        help="Format string for distance labels.",
    )
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_font_prop = None
    caption_font_prop = None
    if args.comic_font_path:
        font_manager.fontManager.addfont(args.comic_font_path)
        caption_font_prop = font_manager.FontProperties(fname=args.comic_font_path)
    else:
        try:
            _ = font_manager.findfont("Comic Sans MS", fallback_to_default=False)
            caption_font_prop = font_manager.FontProperties(family="Comic Sans MS")
        except Exception:
            caption_font_prop = font_manager.FontProperties(family="DejaVu Sans")

    if args.label_font_path:
        font_manager.fontManager.addfont(args.label_font_path)
        label_font_prop = font_manager.FontProperties(fname=args.label_font_path)
    else:
        try:
            _ = font_manager.findfont("Calibri", fallback_to_default=False)
            label_font_prop = font_manager.FontProperties(family="Calibri")
        except Exception:
            label_font_prop = font_manager.FontProperties(family="DejaVu Sans")

    emoji_png = _load_emoji_png(
        [
            args.emoji_png_path,
            "/home/yuheng/task/STRUCTURE/noto-emoji/png/72/emoji_u1f4ac.png",
            "/home/yuheng/task/STRUCTURE/noto-emoji/png/128/emoji_u1f4ac.png",
        ]
    )

    coco_root = Path(args.coco_root)
    instances_path = coco_root / "annotations" / "instances_train2014.json"
    if instances_path.exists():
        selected, meta_df = _select_overlapping_coco_images(
            coco_root=coco_root,
            num_labels=args.num_labels,
            num_samples=args.num_samples,
            seed=args.seed,
        )
        selected["caption_short"] = selected["caption_concat"].apply(
            lambda x: _ensure_ellipsis(
                _wrap_caption(_truncate_words(_first_sentence(str(x)), max_words=10))
            )
        )
        meta_df.to_csv(out_dir / "selected_images_metadata.csv", index=False)
    else:
        if not args.subset_dir:
            raise ValueError("subset_dir is required when coco_root is missing.")
        subset_dir = Path(args.subset_dir)
        df = pd.read_csv(subset_dir / "meta_train.csv")
        df = _remap_image_paths(df)
        rng = np.random.default_rng(args.seed)
        picked_labels = _select_overlapping_labels(df, rng, args.num_labels)
        selected = _select_images_per_class(
            df=df,
            rng=rng,
            class_ids=picked_labels,
            total_samples=args.num_samples,
        )
        selected = selected.copy()
        cap_map = _build_caption_map(df, max_caps=5)
        selected["caption_concat"] = selected["image_file"].map(cap_map).fillna(
            selected.get("caption", "")
        )
        selected["caption"] = selected["caption_concat"]
        selected["caption_short"] = selected["caption_concat"].apply(
            lambda x: _ensure_ellipsis(
                _wrap_caption(_truncate_words(_first_sentence(str(x)), max_words=10))
            )
        )

    id_hash = hashlib.md5(
        ",".join(map(str, selected["image_id"].tolist())).encode("utf-8")
    ).hexdigest()[:8]
    selected.attrs["dataset_name"] = f"selected_n{len(selected)}_{id_hash}"
    img_feats, txt_feats = _load_or_extract_features(
        df=selected,
        image_encoder=args.image_encoder,
        text_encoder=args.text_encoder,
        batch_size=args.batch_size,
        device=args.device,
    )
    img_feats = torch.from_numpy(img_feats).float()
    txt_feats = torch.from_numpy(txt_feats).float()

    os.environ.setdefault("WANDB_MODE", "disabled")
    config = {
        "random_state": args.seed,
        "paths": {"save_path": str(out_dir)},
        "features": {"pool_img": "cls", "pool_txt": "avg"},
        "layer_selection": {
            "type": "random",
            "n_samples": len(selected),
            "metric": "mutual_knn",
            "metric_kwargs": {"topk": "rice", "normalize": True},
            "n_score_bins": 5,
            "best_only": True,
            "last_only": True,
            "n_last_layers": None,
        },
        "training": {
            "n_epochs": args.epochs,
            "batch_size": min(args.batch_size, len(selected)),
            "learning_rate": 1.0e-3,
            "use_lr_finder": False,
            "lr_finder": {"num_iter": 100, "start_lr": 1.0e-7, "end_lr": 10},
            "mixup_alpha": 0.0,
            "fixed_structure": False,
            "clip_grad": 1.0,
            "early_stopping": True,
            "early_stopping_patience": 50,
            "drop_duplicates": False,
            "n_dup_samples": 1,
            "wandb_watch": False,
            "embedding_visualization": 500,
            "visualize_original_embeddings": False,
            "log_structural_preservation": False,
            "structural_preservation_k": [100, 1000],
            "unimodal_data": {"use": False, "text": [], "image": []},
            "alignment_layer_name": "ResLowRankHead",
            "alignment_layer_kwargs": {
                "dim_alignment": args.dim_alignment,
                "rank": 64,
                "dropout_p": 0.1,
                "gate_init": 0.0,
            },
            "clip_loss_name": "CLIPLoss",
            "clip_loss": {
                "temperature": 0.05,
                "normalize_latents": True,
                "learnable_temperature": True,
                "warmup_steps": 1,
                "structure_lambda": args.structure_lambda,
                "structure_levels": 1,
                "structure_margin": 0.0,
                "structure_weighting": "none",
            },
            "optimizer_name": "AdamW",
            "optimizer_kwargs": {"betas": [0.9, 0.95], "weight_decay": 1.0e-4},
            "scheduler_name": "CosineAnnealingLR",
            "scheduler_kwargs": {},
            "scheduler_epoch_cycles": 1,
        },
        "evaluation": {
            "batch_size": min(args.batch_size, len(selected)),
            "num_workers": 4,
            "num_classes_per_batch": 2,
            "use_extended_prompts": False,
            "sample_by_sample_embedding": False,
            "plot_embedding_space": False,
            "log_structural_preservation": False,
            "structural_preservation_k": [100, 1000],
            "drop_duplicates": False,
            "zero_shot_datasets": [],
            "retrieval_datasets": [],
            "alignment_metrics": {"enabled": False},
        },
    }

    dataset = MetaImageTextDataset(selected)
    trainer = AlignmentTrainer(
        config=config,
        train_dataset=DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=ImageTextDataset.collate_fn,
        ),
        val_dataset=DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            drop_last=False,
            num_workers=0,
            collate_fn=ImageTextDataset.collate_fn,
        ),
        llm_model_name=args.text_encoder,
        lvm_model_name=args.image_encoder,
        wandb_logging=True,
        wandb_project_name="structure-mlp-rs-viz",
    )
    trainer.fit()

    ckpts = sorted(out_dir.glob("**/checkpoint_last.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise RuntimeError("No checkpoint saved by AlignmentTrainer.")
    ckpt_path = ckpts[-1]
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    alignment_image = ckpt["alignment_image"].cpu().eval()
    alignment_text = ckpt["alignment_text"].cpu().eval()

    layer_combo = _parse_layer_combination_from_ckpt(ckpt_path)
    image_layer_idx = layer_combo[0] if layer_combo else -1
    text_layer_idx = layer_combo[1] if layer_combo else -1
    img_layer = (
        img_feats[:, image_layer_idx, :] if img_feats.ndim == 3 else img_feats
    )
    txt_layer = (
        txt_feats[:, text_layer_idx, :] if txt_feats.ndim == 3 else txt_feats
    )
    with torch.no_grad():
        img_2d = alignment_image(img_layer).cpu()
        txt_2d = alignment_text(txt_layer).cpu()
        normalize_latents = (
            ckpt.get("config", {})
            .get("training", {})
            .get("clip_loss", {})
            .get("normalize_latents", False)
        )
        if normalize_latents:
            img_2d = safe_normalize(img_2d, p=2, dim=-1)
            txt_2d = safe_normalize(txt_2d, p=2, dim=-1)
        img_2d = img_2d.numpy()
        txt_2d = txt_2d.numpy()

    sample_colors = {i: PALETTE[i % len(PALETTE)] for i in range(len(selected))}

    _save_selected_images(selected, out_dir / "selected_images")
    _plot_aligned_samples(
        out_path=out_dir / "mlp_image_text_aligned.png",
        img_points=img_2d,
        txt_points=txt_2d,
        samples=selected,
        color_map=sample_colors,
        draw_links=args.draw_links,
        show_distances=args.show_distances,
        distance_fmt=args.distance_fmt,
        thumb_size=args.thumb_size,
        img_y_min=args.img_y_min,
        img_y_max=args.img_y_max,
        txt_y_min=args.txt_y_min,
        txt_y_max=args.txt_y_max,
        x_spread=args.x_spread,
        text_attract=args.text_attract,
        split_modalities=args.split_modalities,
        emoji_png=emoji_png,
        emoji_px=args.emoji_px,
        chat_marker_size=args.chat_marker_size,
        image_alpha=args.image_alpha,
        label_font_prop=label_font_prop,
        caption_font_prop=caption_font_prop,
        caption_font_size=args.caption_font_size,
        legend_icon_zoom=args.legend_icon_zoom,
        legend_title_size=args.legend_title_size,
    )

    selected.to_csv(out_dir / "mlp_selected_samples.csv", index=False)
    np.save(out_dir / "mlp_image_embeds.npy", img_2d)
    np.save(out_dir / "mlp_text_embeds.npy", txt_2d)


if __name__ == "__main__":
    main()

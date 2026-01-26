# Benchmarking Guide (STRUCTURE fork)

This document summarizes how we run the four benchmarks and how metrics are computed in this repo.
It also documents the COCO subset construction and the embedding-input option.

## 1) COCO subset construction (intersection pairs)
We build COCO subsets using `jointgwot/scripts/build_coco_intersection_subset.py`.
The script:
- Loads COCO train2017 instances and captions.
- Builds a category-pair list for each image (all pairs from the image’s object categories).
- Ranks category-pairs by how many images contain them, and selects the top `num_classes` pairs.
- Samples a fixed number of images per pair for train and eval.
- Concatenates all captions of the image as the text input.

Outputs (in the subset directory):
- `meta_train.csv`, `meta_eval.csv`
- `meta_train_pairs.csv`, `meta_eval_pairs.csv`
- `y_train.npy`, `y_eval.npy`, `class_pairs.json`

The `jointgwot/scripts/run_ours.sh` pipeline shows the typical parameters used (e.g., `num_classes=50`, `n_train=2000/5000/10000`).

We keep the pre-built subsets here:
```
/home/yuheng/jointgwot/dataset/coco/subsets/coco_pairs50_2k/coco_2k
/home/yuheng/jointgwot/dataset/coco/subsets/coco_pairs50_5k/coco_5k
/home/yuheng/jointgwot/dataset/coco/subsets/coco_pairs50_10k/coco_10k
```

## 2) Encoders and feature extraction
Default backbone encoders for STRUCTURE runs:
- Image: `facebook/dinov3-vitb16-pretrain-lvd1689m` (CLS token pooling)
- Text: `sentence-transformers/all-roberta-large-v1` (avg pooling)

Feature caching is enabled and keyed by dataset tag (annotation file, image dir, and sample count) to avoid collisions across subsets.

## 3) Benchmarks and scripts
We run **four benchmarks**: `csa_only`, `structure+csa`, `structure+mlp`, and `clip`.

### (A) CSA-only (ours)
Script: `scripts/run_coco_csa_only.sh`
- Runs CSA (no STRUCTURE training loop) on each subset and latent dim.

### (B) STRUCTURE + CSA / STRUCTURE + MLP
Script: `scripts/run_coco_benchmark.sh`
- Runs two methods across {2k,5k,10k} × {128,256}.
- CSA uses `NormalizedCCA` with structure regularization.
- MLP uses `ResLowRankHead` alignment.

### (C) CLIP (zero-shot, pretrained)
Script: `scripts/run_clip_benchmark.sh`
- Uses pretrained `openai/clip-vit-large-patch14-336` with PCA to latent dim 128/256.
- **No fine-tuning**.

Optional raw (no PCA) runs:
- `scripts/run_clip_benchmark_raw.sh`

### (D) CLIP loss in STRUCTURE (new)
Script: `scripts/run_coco_clip_structure.sh`
- Uses STRUCTURE training loop with **CLIP loss**:
  - L2-normalize image/text embeddings
  - learnable temperature `t` (logit scale)
  - symmetric cross-entropy loss

The CLIP loss implementation is in `src/loss/clip_loss.py`.

## 4) Metrics and evaluation
We compute:
- **Purity**: cross-modal nearest neighbors using `image_id` labels.
- **Silhouette**: cosine distance, computed on pooled embeddings.
- **Zero-shot accuracy**: CIFAR-10 and TinyImageNet top-1 (micro).
- **Retrieval**: COCO I2T/T2I R@1 (and R@1-avg in summary).

Purity and silhouette are computed using `src/evaluation/alignment_metrics.py`.
Zero-shot and retrieval are evaluated in `src/trainers/alignment_trainer.py` and `src/trainers/csa_trainer.py`.

## 5) Collect results into a single CSV
Use:
```
python /home/yuheng/task/STRUCTURE/scripts/collect_benchmark_results.py
```
This writes:
```
/home/yuheng/task/STRUCTURE/results/benchmark_summary.csv
```
The table includes:
- `Purity`, `Silhouette`
- `Accuracy_CIFAR10`, `Accuracy_TinyImageNet`
- `Retrieval_I2T_R1`, `Retrieval_T2I_R1`

## 6) Embedding-input mode + saving embeddings
STRUCTURE can use **precomputed embeddings** instead of raw images/text.
Set in config:
```
features:
  input_type: "embedding"
  embedding_paths:
    train_image: /path/X_img_train.npy
    train_text:  /path/X_text_train.npy
    val_image:   /path/X_img_eval.npy   # optional
    val_text:    /path/X_text_eval.npy  # optional
```

To save embeddings for the **best layers** and for zero-shot:
```
evaluation:
  save_embeddings:
    enabled: true
    top_k: 2
    datasets: ["coco", "cifar10", "tiny_imagenet"]
```
Output files go to:
```
results/<run_name>/saved_embeddings/
```

## 7) Visualization (R_S + MLP embeddings)
Script: `scripts/plot_structure_mlp_embeddings.py`
- Samples 5–10 images from 3–4 overlapping labels.
- Trains a small R_S+MLP alignment.
- Saves two plots (image points and text points) in 2D.

Example:
```
python /home/yuheng/task/STRUCTURE/scripts/plot_structure_mlp_embeddings.py \
  --subset_dir /home/yuheng/jointgwot/dataset/coco/subsets/coco_pairs50_2k \
  --out_dir /home/yuheng/task/STRUCTURE/results/plots_mlp \
  --num_labels 4 --max_samples 10 --dim_alignment 128 --epochs 50 --device cuda
```

Outputs:
- `mlp_images_2d.png`
- `mlp_texts_2d.png`
- `mlp_selected_samples.csv`
- `mlp_image_embeds.npy`
- `mlp_text_embeds.npy`

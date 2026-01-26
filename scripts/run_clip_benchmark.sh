#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yuheng/task/STRUCTURE"
DATA_ROOT="${PROJECT_ROOT}/data"
PYTHON_BIN="${PYTHON_BIN:-python}"

CLIP_MODEL="${CLIP_MODEL:-openai/clip-vit-large-patch14-336}"

SUBSET_KEYS=("2k" "5k" "10k")
SUBSET_PATHS=(
  "/home/yuheng/jointgwot/dataset/coco/subsets/coco_pairs50_2k/coco_2k"
  "/home/yuheng/jointgwot/dataset/coco/subsets/coco_pairs50_5k/coco_5k"
  "/home/yuheng/jointgwot/dataset/coco/subsets/coco_pairs50_10k/coco_10k"
)
PCA_DIMS=(128 256)

mkdir -p "${DATA_ROOT}" "${PROJECT_ROOT}/tmp_configs" "${PROJECT_ROOT}/results"

for i in "${!SUBSET_KEYS[@]}"; do
  subset_key="${SUBSET_KEYS[$i]}"
  subset_path="${SUBSET_PATHS[$i]}"

  if [[ ! -d "${subset_path}" ]]; then
    echo "Missing subset path: ${subset_path}" >&2
    exit 1
  fi

  ln -sfn "${subset_path}" "${DATA_ROOT}/COCO"

  for dim in "${PCA_DIMS[@]}"; do
    cfg_file="${PROJECT_ROOT}/tmp_configs/clip_${subset_key}_pca${dim}.yaml"

    cat > "${cfg_file}" <<YAML
defaults: !include ../configs/default.yaml

overrides:
  paths:
    data_path: "${DATA_ROOT}"
    save_path: "${PROJECT_ROOT}/results"
  alignment:
    llm_model_name: "${CLIP_MODEL}"
    lvm_model_name: "${CLIP_MODEL}"
  training:
    clip: true
  features:
    dataset: "coco"
    batch_size: 16
    num_workers: 8
  evaluation:
    batch_size: 8
    pca_dim: ${dim}
    zero_shot_datasets:
      - "cifar10"
      - "tiny_imagenet"
    retrieval_datasets:
      - "coco"
    retrieval_subset:
      size: 5000
      seed: 42
    alignment_metrics:
      enabled: true
      label_column: "image_id"
      silhouette_metric: "cosine"
      silhouette_sample_size: 10000
      batch_size: 256
YAML

    run_name="clip_coco_${subset_key}_pca${dim}"
    WANDB_NAME="${run_name}" "${PYTHON_BIN}" "${PROJECT_ROOT}/src/train_alignment.py" \
      --config_path "${cfg_file}" \
      --wandb_notes "clip benchmark subset=${subset_key} pca_dim=${dim}"
  done
done

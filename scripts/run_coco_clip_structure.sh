#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yuheng/task/STRUCTURE"
DATA_ROOT="${PROJECT_ROOT}/data"
PYTHON_BIN="${PYTHON_BIN:-python}"

SUBSET_KEYS=("2k" "5k" "10k")
SUBSET_PATHS=(
  "/home/yuheng/jointgwot/dataset/coco/subsets/coco_pairs50_2k/coco_2k"
  "/home/yuheng/jointgwot/dataset/coco/subsets/coco_pairs50_5k/coco_5k"
  "/home/yuheng/jointgwot/dataset/coco/subsets/coco_pairs50_10k/coco_10k"
)
LATENT_DIMS=(128 256)

mkdir -p "${DATA_ROOT}" "${PROJECT_ROOT}/tmp_configs" "${PROJECT_ROOT}/results"

for i in "${!SUBSET_KEYS[@]}"; do
  subset_key="${SUBSET_KEYS[$i]}"
  subset_path="${SUBSET_PATHS[$i]}"

  if [[ ! -d "${subset_path}" ]]; then
    echo "Missing subset path: ${subset_path}" >&2
    exit 1
  fi

  ln -sfn "${subset_path}" "${DATA_ROOT}/COCO"

  for dim in "${LATENT_DIMS[@]}"; do
    cfg_file="${PROJECT_ROOT}/tmp_configs/clip_structure_${subset_key}_d${dim}.yaml"

    cat > "${cfg_file}" <<YAML
defaults: !include ../configs/default.yaml

overrides:
  paths:
    data_path: "${DATA_ROOT}"
    save_path: "${PROJECT_ROOT}/results"
  alignment:
    llm_model_name: "sentence-transformers/all-roberta-large-v1"
    lvm_model_name: "facebook/dinov3-vitb16-pretrain-lvd1689m"
  training:
    alignment_layer_name: "LinearAlignmentLayer"
    alignment_layer_kwargs:
      dim_alignment: ${dim}
    clip_loss:
      normalize_latents: true
      learnable_temperature: true
      structure_lambda: 0.0
    n_epochs: 200
  features:
    dataset: "coco"
    batch_size: 16
    num_workers: 8
  evaluation:
    batch_size: 8
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

    run_name="clip_structure_coco_${subset_key}_d${dim}"
    WANDB_NAME="${run_name}" "${PYTHON_BIN}" "${PROJECT_ROOT}/src/train_alignment.py" \
      --config_path "${cfg_file}" \
      --wandb_notes "clip-loss structure subset=${subset_key} dim=${dim}"
  done
done

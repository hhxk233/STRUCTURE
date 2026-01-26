#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yuheng/task/STRUCTURE"
DATA_ROOT="${PROJECT_ROOT}/data"
PYTHON_BIN="${PYTHON_BIN:-python}"

DINOV3_MODEL="${DINOV3_MODEL:-facebook/dinov3-vitb16-pretrain-lvd1689m}"

SUBSET_KEYS=("2k" "5k" "10k")
SUBSET_PATHS=(
  "/home/yuheng/jointgwot/dataset/coco/subsets/coco_pairs50_2k/coco_2k"
  "/home/yuheng/jointgwot/dataset/coco/subsets/coco_pairs50_5k/coco_5k"
  "/home/yuheng/jointgwot/dataset/coco/subsets/coco_pairs50_10k/coco_10k"
)
SUBSET_TRAIN_SAMPLES=(2000 5000 10000)

DIMS=(128 256)

mkdir -p "${DATA_ROOT}" "${PROJECT_ROOT}/results" "${PROJECT_ROOT}/tmp_configs"

for i in "${!SUBSET_KEYS[@]}"; do
  subset_key="${SUBSET_KEYS[$i]}"
  subset_path="${SUBSET_PATHS[$i]}"
  train_samples="${SUBSET_TRAIN_SAMPLES[$i]}"

  if [[ ! -d "${subset_path}" ]]; then
    echo "Missing subset path: ${subset_path}" >&2
    exit 1
  fi

  ln -sfn "${subset_path}" "${DATA_ROOT}/COCO"

  for dim in "${DIMS[@]}"; do
    cfg_file="${PROJECT_ROOT}/tmp_configs/coco_${subset_key}_csa_d${dim}.yaml"

    cat > "${cfg_file}" <<YAML
defaults: !include ../configs/default.yaml

overrides:
  paths:
    data_path: "${DATA_ROOT}"
    save_path: "${PROJECT_ROOT}/results"
  alignment:
    lvm_model_name: "${DINOV3_MODEL}"
  features:
    dataset: "coco"
    batch_size: 16
    num_workers: 8
  training:
    batch_size: 8
    n_samples_train: ${train_samples}
    n_samples_val: 2000
    drop_duplicates: false
    cca: true
    cca_kwargs:
      sim_dim: ${dim}
      equal_weights: false
      use_reg: false
      lambda_rs: 0.0
      lambda_cca_coeff: 0.0
      L: 1
      tau: 0.2
      refine_epochs: 0
      lr: 5.0e-4
  layer_selection:
    best_only: true
    last_only: false
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

    run_name="coco_${subset_key}_csa_d${dim}"
    WANDB_NAME="${run_name}" "${PYTHON_BIN}" "${PROJECT_ROOT}/src/train_alignment.py" \
      --config_path "${cfg_file}" \
      --wandb_notes "benchmark subset=${subset_key} method=csa dim=${dim} use_reg=false"
  done
done

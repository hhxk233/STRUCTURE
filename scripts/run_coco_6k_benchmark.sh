#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yuheng/task/STRUCTURE"
DATA_ROOT="${PROJECT_ROOT}/data"
PYTHON_BIN="${PYTHON_BIN:-python}"

DINOV3_MODEL="facebook/dinov3-vitb16-pretrain-lvd1689m"
SUBSET_KEY="6k"
SUBSET_PATH="/home/yuheng/jointgwot/dataset/coco/subsets/coco_6k/coco"
TRAIN_SAMPLES=6000

DIMS=(128 256)

mkdir -p "${DATA_ROOT}" "${PROJECT_ROOT}/results" "${PROJECT_ROOT}/tmp_configs"

if [[ ! -d "${SUBSET_PATH}" ]]; then
  echo "Missing subset path: ${SUBSET_PATH}" >&2
  exit 1
fi

ln -sfn "${SUBSET_PATH}" "${DATA_ROOT}/COCO"

for dim in "${DIMS[@]}"; do
  # CSA only
  cfg_file="${PROJECT_ROOT}/tmp_configs/coco_${SUBSET_KEY}_csa_only_d${dim}.yaml"
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
    n_samples_train: ${TRAIN_SAMPLES}
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
  run_name="coco_${SUBSET_KEY}_csa_only_d${dim}"
  WANDB_NAME="${run_name}" "${PYTHON_BIN}" "${PROJECT_ROOT}/src/train_alignment.py" \
    --config_path "${cfg_file}" \
    --wandb_notes "benchmark subset=${SUBSET_KEY} method=csa_only dim=${dim}"

  # CSA + RS (STRUCTURE)
  cfg_file="${PROJECT_ROOT}/tmp_configs/coco_${SUBSET_KEY}_csa_rs_d${dim}.yaml"
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
    n_samples_train: ${TRAIN_SAMPLES}
    n_samples_val: 2000
    drop_duplicates: false
    cca: true
    cca_kwargs:
      sim_dim: ${dim}
      equal_weights: false
      use_reg: true
      lambda_rs: 10.0
      lambda_cca_coeff: 1.0e-2
      L: 1
      tau: 0.2
      refine_epochs: 10
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
  run_name="coco_${SUBSET_KEY}_csa_rs_d${dim}"
  WANDB_NAME="${run_name}" "${PYTHON_BIN}" "${PROJECT_ROOT}/src/train_alignment.py" \
    --config_path "${cfg_file}" \
    --wandb_notes "benchmark subset=${SUBSET_KEY} method=csa_rs dim=${dim}"

  # MLP + RS (STRUCTURE)
  cfg_file="${PROJECT_ROOT}/tmp_configs/coco_${SUBSET_KEY}_mlp_rs_d${dim}.yaml"
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
    n_samples_train: ${TRAIN_SAMPLES}
    n_samples_val: 2000
    drop_duplicates: false
    alignment_layer_name: "ResLowRankHead"
    alignment_layer_kwargs:
      dim_alignment: ${dim}
      rank: 64
      dropout_p: 0.1
      gate_init: 0.0
    clip_loss_name: "CLIPLoss"
    clip_loss:
      temperature: 0.05
      normalize_latents: true
      warmup_steps: 1000
      structure_lambda: 10.0
      structure_levels: 1
      structure_margin: 0.0
      structure_weighting: "none"
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
  run_name="coco_${SUBSET_KEY}_mlp_rs_d${dim}"
  WANDB_NAME="${run_name}" "${PYTHON_BIN}" "${PROJECT_ROOT}/src/train_alignment.py" \
    --config_path "${cfg_file}" \
    --wandb_notes "benchmark subset=${SUBSET_KEY} method=mlp_rs dim=${dim}"

  # CLIP loss in STRUCTURE
  cfg_file="${PROJECT_ROOT}/tmp_configs/clip_structure_${SUBSET_KEY}_d${dim}.yaml"
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
  run_name="clip_structure_coco_${SUBSET_KEY}_d${dim}"
  WANDB_NAME="${run_name}" "${PYTHON_BIN}" "${PROJECT_ROOT}/src/train_alignment.py" \
    --config_path "${cfg_file}" \
    --wandb_notes "clip-loss structure subset=${SUBSET_KEY} dim=${dim}"
done

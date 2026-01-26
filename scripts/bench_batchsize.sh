#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/yuheng/task/STRUCTURE"
DATA_ROOT="${PROJECT_ROOT}/data"
PYTHON_BIN="${PYTHON_BIN:-python}"

DINOV3_MODEL="${DINOV3_MODEL:-facebook/dinov3-vitb16-pretrain-lvd1689m}"
SUBSET_PATH="${SUBSET_PATH:-/home/yuheng/jointgwot/dataset/coco/subsets/coco_pairs50_2k/coco_2k}"

FEATURE_BS_LIST=(16 32)
EVAL_BS_LIST=(4 8)

N_SAMPLES_TRAIN="${N_SAMPLES_TRAIN:-512}"
N_SAMPLES_VAL="${N_SAMPLES_VAL:-256}"
REFINE_EPOCHS="${REFINE_EPOCHS:-5}"

mkdir -p "${DATA_ROOT}" "${PROJECT_ROOT}/tmp_configs" "${PROJECT_ROOT}/results"
ln -sfn "${SUBSET_PATH}" "${DATA_ROOT}/COCO"

run_mode () {
  local mode="$1"
  local feat_bs="$2"
  local eval_bs="$3"
  local cfg_file="${PROJECT_ROOT}/tmp_configs/bench_${mode}_f${feat_bs}_e${eval_bs}.yaml"

  local zero_shot_list="[]"
  local retrieval_list="[]"
  if [[ "${mode}" == "retrieval" ]]; then
    retrieval_list='["coco"]'
  elif [[ "${mode}" == "zeroshot" ]]; then
    zero_shot_list='["cifar10"]'
  fi

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
    batch_size: ${feat_bs}
    num_workers: 8
  training:
    batch_size: 8
    n_samples_train: ${N_SAMPLES_TRAIN}
    n_samples_val: ${N_SAMPLES_VAL}
    drop_duplicates: false
    cca: true
    cca_kwargs:
      sim_dim: 128
      equal_weights: false
      use_reg: true
      lambda_rs: 10.0
      lambda_cca_coeff: 1.0e-2
      L: 1
      tau: 0.2
      refine_epochs: ${REFINE_EPOCHS}
      lr: 5.0e-4
  layer_selection:
    best_only: true
    last_only: false
  evaluation:
    batch_size: ${eval_bs}
    zero_shot_datasets: ${zero_shot_list}
    retrieval_datasets: ${retrieval_list}
    retrieval_subset:
      size: 1000
      seed: 42
    alignment_metrics:
      enabled: true
      label_column: "image_path"
      silhouette_metric: "cosine"
      silhouette_sample_size: 5000
      batch_size: 256
YAML

  echo "[bench] mode=${mode} features.batch_size=${feat_bs} evaluation.batch_size=${eval_bs}"
  /usr/bin/time -f "[bench] mode=${mode} fbs=${feat_bs} ebs=${eval_bs} elapsed=%E" \
    "${PYTHON_BIN}" "${PROJECT_ROOT}/src/train_alignment.py" \
      --config_path "${cfg_file}" \
      --wandb_notes "bench-${mode}-f${feat_bs}-e${eval_bs}"
}

for feat_bs in "${FEATURE_BS_LIST[@]}"; do
  run_mode "features" "${feat_bs}" 4
done

for eval_bs in "${EVAL_BS_LIST[@]}"; do
  run_mode "retrieval" 16 "${eval_bs}"
done

for eval_bs in "${EVAL_BS_LIST[@]}"; do
  run_mode "zeroshot" 16 "${eval_bs}"
done

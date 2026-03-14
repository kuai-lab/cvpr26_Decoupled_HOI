#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"${PYTHON_BIN}" "${PROJECT_ROOT}/trainer_dechoi.py" \
--window=120 \
--batch_size=32 \
--learning_rate=0.0001 \
--n_head=4 \
--data_root_folder="${DATA_ROOT_FOLDER:-${PROJECT_ROOT}/data/processed_data}" \
--project="${PROJECT_DIR:-${PROJECT_ROOT}/dechoi_runs}" \
--exp_name="${EXP_NAME:-251117}" \
--wandb_pj_name="${WANDB_PROJECT:-dechoi_runs}" \
--entity="${WANDB_ENTITY:-}" \
--input_first_human_pose \
--use_random_frame_bps \
--add_language_condition \
--use_object_keypoints \
--loss_w_feet=0 \
--loss_w_fk=1.0 \
--loss_w_obj_pts=1 \
--loss_type="l1"

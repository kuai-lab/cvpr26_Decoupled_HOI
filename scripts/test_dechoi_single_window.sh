#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"${PYTHON_BIN}" "${PROJECT_ROOT}/trainer_dechoi.py" \
--window=120 \
--batch_size=32 \
--n_head=4 \
--data_root_folder="${DATA_ROOT_FOLDER:-${PROJECT_ROOT}/data/processed_data}" \
--pretrained_model="${PRETRAINED_MODEL:-${PROJECT_ROOT}/checkpoints/dechoi_best_final.pt}" \
--save_res_folder="${SAVE_RES_FOLDER:-${PROJECT_ROOT}/dechoi_results}" \
--input_first_human_pose \
--use_random_frame_bps \
--add_language_condition \
--use_object_keypoints \
--add_semantic_contact_labels \
--loss_w_feet=1 \
--loss_w_fk=0.5 \
--loss_w_obj_pts=1 \
--test_sample_res \
--use_guidance_in_denoising \
--for_quant_eval

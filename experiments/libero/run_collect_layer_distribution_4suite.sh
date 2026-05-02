#!/usr/bin/env bash
set -euo pipefail

# Hardcoded run config (no CLI args needed).
CKPT="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt"
VIDEO_DEVICE="cuda:2"
ACTION_DEVICE="cuda:3"

LIBERO_ROOT="/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/LIBERO"
if [[ ! -d "${LIBERO_ROOT}" ]]; then
  echo "[ERROR] LIBERO root not found: ${LIBERO_ROOT}"
  exit 1
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] ckpt file not found: ${CKPT}"
  exit 1
fi

export PYTHONPATH="${LIBERO_ROOT}:${PYTHONPATH:-}"
# Force clear any inherited CUDA mask to keep device IDs as global 6/7.
unset CUDA_VISIBLE_DEVICES

python experiments/libero/collect_libero_layer_distribution.py \
  --config-name sim_libero \
  task=libero_streaming_action_ft_2cam224_1e-4 \
  model._target_=fastwam.runtime.create_fastwam_streaming \
  ckpt="${CKPT}" \
  EVALUATION.async_video_device="${VIDEO_DEVICE}" \
  EVALUATION.async_action_device="${ACTION_DEVICE}" \
  EVALUATION.async_obs_stride_env_steps=3 \
  EVALUATION.async_action_trigger_every_n_obs=3 \
  EVALUATION.num_inference_steps=8 \
  EVALUATION.async_warmup_action_jobs=20 \
  EVALUATION.async_control_dt_ms=50

#!/usr/bin/env bash
set -euo pipefail

# Streaming training loop:
#   rollout async schedules -> train with real schedule pool -> repeat

TASK="${TASK:-libero_streaming_action_ft_2cam224_1e-4}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
TASKS_PER_SUITE="${TASKS_PER_SUITE:-10}"
TRIALS="${TRIALS:-2}"
REPLAY="${REPLAY:-4096}"

INITIAL_CKPT="${INITIAL_CKPT:-/inspire/qb-ilm/project/robot-reasoning/xiangyushun-p-xiangyushun/luye/FastWAM/runs/libero_async_offline_rl/2026-04-28_12-21-23/train/checkpoints/weights/step_000400.pt}"
RUN_ROOT="${RUN_ROOT:-./runs/libero_async_offline_rl/$(date +%Y-%m-%d_%H-%M-%S)}"
REPLAY_ROOT="${REPLAY_ROOT:-./data/trajectory_replay/libero_async_offline_rl}"

ROUNDS="${ROUNDS:-5}"
SAVE_EVERY="${SAVE_EVERY:-400}"
SEED="${SEED:-42}"
RENDER_GPU="${RENDER_GPU:-}"

PYTHON="${PYTHON:-python}"
TRAIN_ENTRY="${TRAIN_ENTRY:-scripts/train_streaming_action_ft.py}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-scripts/accelerate_configs/accelerate_zero2_ds.yaml}"

mkdir -p "${RUN_ROOT}" "${REPLAY_ROOT}"

if [[ -z "${GPU_PAIRS:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_PAIRS="$("${PYTHON}" - "${CUDA_VISIBLE_DEVICES}" <<'PY'
import sys
gpus = [v.strip() for v in sys.argv[1].split(",") if v.strip()]
if len(gpus) % 2 != 0:
    raise SystemExit("CUDA_VISIBLE_DEVICES must contain an even number of GPUs.")
print(" ".join(",".join(gpus[i:i + 2]) for i in range(0, len(gpus), 2)))
PY
)"
  else
    GPU_PAIRS="0,1 2,3"
  fi
fi

read -r -a GPU_PAIR_LIST <<< "${GPU_PAIRS}"
SHARDS="${#GPU_PAIR_LIST[@]}"
if [[ "${SHARDS}" -lt 1 ]]; then
  echo "ERROR: GPU_PAIRS must contain at least one video/action GPU pair." >&2
  exit 1
fi
COLLECT_SHARDS=1
COLLECT_GPU_PAIR_LIST=("${GPU_PAIR_LIST[@]:0:${COLLECT_SHARDS}}")
TRAIN_GPUS="$("${PYTHON}" - "${GPU_PAIR_LIST[@]}" <<'PY'
import sys
flat = []
for pair in sys.argv[1:]:
    gpus = [v.strip() for v in pair.split(",") if v.strip()]
    if len(gpus) != 2:
        raise SystemExit(f"Each GPU pair must contain exactly two GPUs, got: {pair}")
    flat.extend(gpus)
print(",".join(flat))
PY
)"
if [[ -z "${NPROC:-}" ]]; then
  NPROC="$("${PYTHON}" - "${TRAIN_GPUS}" <<'PY'
import sys
print(len([v for v in sys.argv[1].split(",") if v]))
PY
)"
fi
echo "[config] collect uses ${COLLECT_SHARDS}/${SHARDS} gpu pairs: ${COLLECT_GPU_PAIR_LIST[*]}"
if [[ -n "${RENDER_GPU}" ]]; then
  echo "[config] collect render gpu: ${RENDER_GPU}"
fi
echo "[config] train uses ${SHARDS} gpu pairs (${TRAIN_GPUS}), nproc=${NPROC}"

CURRENT_COLLECT_CKPT="${INITIAL_CKPT}"
CURRENT_TRAIN_RESUME="${INITIAL_CKPT}"

collect_shard() {
  local shard_id="$1"
  local num_shards="$2"
  local visible_devices="$3"
  local output_path="$4"
  local task_suite="$5"
  local task_id="$6"
  local collect_visible_devices="${visible_devices}"
  local render_device="${visible_devices%%,*}"
  if [[ -n "${RENDER_GPU}" ]]; then
    collect_visible_devices="${visible_devices},${RENDER_GPU}"
    render_device="${RENDER_GPU}"
  fi

  CUDA_VISIBLE_DEVICES="${collect_visible_devices}" \
  MUJOCO_EGL_DEVICE_ID="${render_device}" \
  "${PYTHON}" experiments/libero/collect_async_schedules.py \
    --config-name sim_libero \
    "task=${TASK}" \
    "ckpt=${CURRENT_COLLECT_CKPT}" \
    "gpu_id=${render_device}" \
    "seed=${SEED}" \
    "EVALUATION.task_suite_name=${task_suite}" \
    "EVALUATION.task_id=${task_id}" \
    "EVALUATION.num_trials=$((TRIALS * num_shards))" \
    "EVALUATION.device=cuda:0" \
    "EVALUATION.save_video=false" \
    "EVALUATION.schedule_output_path=${output_path}" \
    "EVALUATION.schedule_shard_id=${shard_id}" \
    "EVALUATION.schedule_num_shards=${num_shards}" \
    "EVALUATION.output_dir=${RUN_ROOT}/collect_round_${ROUND}/shard_${shard_id}"
}

merge_shards() {
  local output_path="$1"
  shift
  "${PYTHON}" - "$output_path" "$@" <<'PY'
import sys
from pathlib import Path
import torch

output = Path(sys.argv[1])
inputs = [Path(p) for p in sys.argv[2:]]
schedules = []
metas = []
for path in inputs:
    payload = torch.load(path, map_location="cpu")
    shard_schedules = list(payload.get("schedules", []))
    if len(shard_schedules) == 0:
        raise RuntimeError(f"Schedule shard is empty: {path}")
    schedules.extend(shard_schedules)
    metas.append(payload.get("meta", {"path": str(path), "schedules": len(shard_schedules)}))
output.parent.mkdir(parents=True, exist_ok=True)
torch.save({"meta": {"num_schedules": len(schedules), "shards": metas}, "schedules": schedules}, output)
print(f"[merge] wrote {len(schedules)} schedules -> {output}")
PY
}

choose_task() {
  "${PYTHON}" - "$SEED" "$ROUND" "$BATCH" "$TASKS_PER_SUITE" $SUITES <<'PY'
import random
import sys

seed = int(sys.argv[1])
round_idx = int(sys.argv[2])
batch_idx = int(sys.argv[3])
tasks_per_suite = int(sys.argv[4])
suites = sys.argv[5:]
rng = random.Random(seed + round_idx * 1000003 + batch_idx * 9176)
suite = rng.choice(suites)
task_id = rng.randrange(tasks_per_suite)
print(suite, task_id)
PY
}

schedule_records() {
  local path="$1"
  "${PYTHON}" - "$path" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu")
print(sum(len(s.get("steps", [])) for s in payload.get("schedules", [])))
PY
}

schedule_count() {
  local path="$1"
  "${PYTHON}" - "$path" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu")
print(len(payload.get("schedules", [])))
PY
}

for ROUND in $(seq 0 $((ROUNDS - 1))); do
  REPLAY_DIR="${REPLAY_ROOT}/round_${ROUND}"
  mkdir -p "${REPLAY_DIR}"

  MERGED_SCHEDULE="${REPLAY_DIR}/schedule.pt"

  BATCH=0
  while true; do
    if [[ -f "${MERGED_SCHEDULE}" ]]; then
      CURRENT_RECORDS="$(schedule_records "${MERGED_SCHEDULE}")"
    else
      CURRENT_RECORDS="0"
    fi
    if [[ "${CURRENT_RECORDS}" -ge "${REPLAY}" ]]; then
      break
    fi

    read -r TASK_SUITE TASK_ID < <(choose_task)
    BATCH_MERGED="${REPLAY_DIR}/schedule_batch${BATCH}.pt"

    echo "[round ${ROUND}] collect schedule batch=${BATCH} suite=${TASK_SUITE} task=${TASK_ID} records=${CURRENT_RECORDS}/${REPLAY}"
    SHARD_PATHS=()
    PIDS=()
    for SHARD_ID in "${!COLLECT_GPU_PAIR_LIST[@]}"; do
      SHARD_PATH="${REPLAY_DIR}/schedule_batch${BATCH}_shard${SHARD_ID}.pt"
      SHARD_PATHS+=("${SHARD_PATH}")
      collect_shard "${SHARD_ID}" "${COLLECT_SHARDS}" "${COLLECT_GPU_PAIR_LIST[${SHARD_ID}]}" "${SHARD_PATH}" "${TASK_SUITE}" "${TASK_ID}" &
      PIDS+=("$!")
    done
    for PID in "${PIDS[@]}"; do
      wait "${PID}"
    done

    if [[ -f "${MERGED_SCHEDULE}" ]]; then
      merge_shards "${BATCH_MERGED}" "${SHARD_PATHS[@]}"
      merge_shards "${MERGED_SCHEDULE}.tmp" "${MERGED_SCHEDULE}" "${BATCH_MERGED}"
      mv "${MERGED_SCHEDULE}.tmp" "${MERGED_SCHEDULE}"
    else
      merge_shards "${MERGED_SCHEDULE}" "${SHARD_PATHS[@]}"
    fi
    BATCH=$((BATCH + 1))
  done

  echo "[round ${ROUND}] collected $(schedule_count "${MERGED_SCHEDULE}") schedules, $(schedule_records "${MERGED_SCHEDULE}") schedule steps -> ${MERGED_SCHEDULE}"

  TARGET_STEP=$(((ROUND + 1) * SAVE_EVERY))
  echo "[round ${ROUND}] train to step ${TARGET_STEP}; training samples real schedules online"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    --num_processes "${NPROC}" \
    "${TRAIN_ENTRY}" \
    "task=${TASK}" \
    "seed=${SEED}" \
    "resume=${CURRENT_TRAIN_RESUME}" \
    "output_dir=${RUN_ROOT}/train" \
    "max_steps=${TARGET_STEP}" \
    "save_every=${SAVE_EVERY}" \
    "STREAMING.schedule_path=${MERGED_SCHEDULE}"

  NEXT_STATE="${RUN_ROOT}/train/checkpoints/state/step_$(printf "%06d" "${TARGET_STEP}")"
  NEXT_WEIGHTS="${RUN_ROOT}/train/checkpoints/weights/step_$(printf "%06d" "${TARGET_STEP}").pt"
  if [[ -f "${NEXT_WEIGHTS}" ]]; then
    CURRENT_COLLECT_CKPT="${NEXT_WEIGHTS}"
  else
    echo "[round ${ROUND}] ERROR: missing weights checkpoint ${NEXT_WEIGHTS}" >&2
    exit 1
  fi

  if [[ -d "${NEXT_STATE}" ]]; then
    CURRENT_TRAIN_RESUME="${NEXT_STATE}"
  elif [[ -f "${NEXT_WEIGHTS}" ]]; then
    CURRENT_TRAIN_RESUME="${NEXT_WEIGHTS}"
  else
    echo "[round ${ROUND}] ERROR: missing checkpoint ${NEXT_STATE} or ${NEXT_WEIGHTS}" >&2
    exit 1
  fi
  echo "[round ${ROUND}] next collect_ckpt=${CURRENT_COLLECT_CKPT}"
  echo "[round ${ROUND}] next train_resume=${CURRENT_TRAIN_RESUME}"
done

echo "[done] streaming schedule training loop finished. run_root=${RUN_ROOT}"

#!/usr/bin/env bash
# One-click launcher for the RoboTwin synchronous FastWAM chunk-replan policy.
#
# Usage:
#   bash experiments/robotwin/run_sync.sh [TASK_NAME] [TASK_CONFIG] [EVAL_NUM_EPISODES]
#
# Environment overrides:
#   CKPT                  path to the FastWAM .pt checkpoint
#   DATASET_STATS         path to dataset_stats.json
#   OUTPUT_DIR            default ./evaluate_results/robotwin_sync/real
#   PYTHON                python interpreter (default: .venv/bin/python if present, else python)
#   GPU_ID                optional CUDA_VISIBLE_DEVICES value
#   DEVICE                default cuda
#   RAND_DEVICE           default $DEVICE
#   REPLAN_STEPS          default 8
#   NUM_INFERENCE_STEPS   default 10
#   ACTION_HORIZON        default 10
#   INSTRUCTION_TYPE      default unseen
#   EXTRA_ARGS            extra flags appended verbatim to the python call

set -euo pipefail

TASK_NAME="${1:-click_alarmclock}"
TASK_CONFIG="${2:-demo_randomized}"
EVAL_NUM_EPISODES="${3:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CKPT="${CKPT:-checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt}"
DATASET_STATS="${DATASET_STATS:-checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json}"
OUTPUT_DIR="${OUTPUT_DIR:-./evaluate_results/robotwin_sync/real}"
DEVICE="${DEVICE:-cuda}"
RAND_DEVICE="${RAND_DEVICE:-$DEVICE}"
REPLAN_STEPS="${REPLAN_STEPS:-8}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON="$REPO_ROOT/.venv/bin/python"
    else
        PYTHON="python"
    fi
fi

if [[ ! -f "$CKPT" ]]; then
    echo "[run_sync] ERROR: checkpoint not found: $CKPT" >&2
    echo "  set CKPT=... to override." >&2
    exit 1
fi
if [[ ! -f "$DATASET_STATS" ]]; then
    echo "[run_sync] ERROR: dataset_stats.json not found: $DATASET_STATS" >&2
    echo "  set DATASET_STATS=... to override." >&2
    exit 1
fi
if [[ ! -d "$REPO_ROOT/third_party/RoboTwin/envs" ]]; then
    echo "[run_sync] ERROR: third_party/RoboTwin is missing envs/ — is this a real install?" >&2
    exit 1
fi

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

echo "[run_sync] repo       = $REPO_ROOT"
echo "[run_sync] python     = $PYTHON"
echo "[run_sync] task       = $TASK_NAME ($TASK_CONFIG)"
echo "[run_sync] episodes   = $EVAL_NUM_EPISODES"
echo "[run_sync] ckpt       = $CKPT"
echo "[run_sync] stats      = $DATASET_STATS"
echo "[run_sync] output     = $OUTPUT_DIR/$TASK_NAME/"

CMD=(
    "$PYTHON" experiments/robotwin/eval_robotwin_single_sync.py
    --task-name "$TASK_NAME"
    --task-config "$TASK_CONFIG"
    --ckpt-setting "$CKPT"
    --dataset-stats "$DATASET_STATS"
    --eval-num-episodes "$EVAL_NUM_EPISODES"
    --device "$DEVICE"
    --action-horizon "$ACTION_HORIZON"
    --replan-steps "$REPLAN_STEPS"
    --num-inference-steps "$NUM_INFERENCE_STEPS"
    --instruction-type "$INSTRUCTION_TYPE"
    --output-dir "$OUTPUT_DIR"
    --rand-device "$RAND_DEVICE"
)

if [[ -n "${GPU_ID:-}" ]]; then
    CMD+=(--gpu-id "$GPU_ID")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=($EXTRA_ARGS)
    CMD+=("${EXTRA_ARR[@]}")
fi

"${CMD[@]}"

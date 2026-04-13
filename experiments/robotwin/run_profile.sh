#!/usr/bin/env bash
# One-click launcher for the RoboTwin streaming profiling run on a real
# (fully-installed) RoboTwin machine.
#
# Usage:
#   bash experiments/robotwin/run_profile.sh [TASK_NAME] [TASK_CONFIG] [EVAL_NUM_EPISODES]
#
# Examples:
#   bash experiments/robotwin/run_profile.sh
#   bash experiments/robotwin/run_profile.sh click_alarmclock demo_randomized 1
#
# Environment overrides (export before calling, or prefix the command):
#   CKPT                  path to the streaming .pt checkpoint
#   DATASET_STATS         path to dataset_stats.json
#   ASYNC_VIDEO_DEVICE    default cuda:0
#   ASYNC_ACTION_DEVICE   default cuda:1
#   PROFILE_OUTPUT_DIR    default ./evaluate_results/robotwin_profile/real
#   PYTHON                python interpreter (default: .venv/bin/python if present, else python)
#   REPLAN_STEPS          default 24
#   NUM_INFERENCE_STEPS   default 8
#   INSTRUCTION_TYPE      default unseen
#   FASTWAM_TMPDIR        tmp dir for torch multiprocessing sharing (default /tmp/fw)
#   EXTRA_ARGS            extra flags appended verbatim to the python call

set -euo pipefail

TASK_NAME="${1:-click_alarmclock}"
TASK_CONFIG="${2:-demo_randomized}"
EVAL_NUM_EPISODES="${3:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CKPT="${CKPT:-checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt}"
DATASET_STATS="${DATASET_STATS:-checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json}"
ASYNC_VIDEO_DEVICE="${ASYNC_VIDEO_DEVICE:-cuda:0}"
ASYNC_ACTION_DEVICE="${ASYNC_ACTION_DEVICE:-cuda:1}"
PROFILE_OUTPUT_DIR="${PROFILE_OUTPUT_DIR:-./evaluate_results/robotwin_profile/real}"
REPLAN_STEPS="${REPLAN_STEPS:-24}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-8}"
INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
FASTWAM_TMPDIR="${FASTWAM_TMPDIR:-/tmp/fw}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON="$REPO_ROOT/.venv/bin/python"
    else
        PYTHON="python"
    fi
fi

if [[ ! -f "$CKPT" ]]; then
    echo "[run_profile] ERROR: checkpoint not found: $CKPT" >&2
    echo "  set CKPT=... to override." >&2
    exit 1
fi
if [[ ! -f "$DATASET_STATS" ]]; then
    echo "[run_profile] ERROR: dataset_stats.json not found: $DATASET_STATS" >&2
    echo "  set DATASET_STATS=... to override." >&2
    exit 1
fi
if [[ ! -d "$REPO_ROOT/third_party/RoboTwin/envs" ]]; then
    echo "[run_profile] ERROR: third_party/RoboTwin is missing envs/ — is this a real install?" >&2
    exit 1
fi

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export FASTWAM_TMPDIR

echo "[run_profile] repo       = $REPO_ROOT"
echo "[run_profile] python     = $PYTHON"
echo "[run_profile] task       = $TASK_NAME ($TASK_CONFIG)"
echo "[run_profile] episodes   = $EVAL_NUM_EPISODES"
echo "[run_profile] video gpu  = $ASYNC_VIDEO_DEVICE"
echo "[run_profile] action gpu = $ASYNC_ACTION_DEVICE"
echo "[run_profile] ckpt       = $CKPT"
echo "[run_profile] output     = $PROFILE_OUTPUT_DIR/$TASK_NAME/"
echo "[run_profile] tmpdir     = $FASTWAM_TMPDIR"

CMD=(
    "$PYTHON" experiments/robotwin/eval_robotwin_single_profiled.py
    --task-name "$TASK_NAME"
    --task-config "$TASK_CONFIG"
    --ckpt-setting "$CKPT"
    --dataset-stats "$DATASET_STATS"
    --eval-num-episodes "$EVAL_NUM_EPISODES"
    --async-video-device "$ASYNC_VIDEO_DEVICE"
    --async-action-device "$ASYNC_ACTION_DEVICE"
    --replan-steps "$REPLAN_STEPS"
    --num-inference-steps "$NUM_INFERENCE_STEPS"
    --instruction-type "$INSTRUCTION_TYPE"
    --profile-output-dir "$PROFILE_OUTPUT_DIR"
)

if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARR=($EXTRA_ARGS)
    CMD+=("${EXTRA_ARR[@]}")
fi

PY_PID=""
cleanup_pg() {
    if [[ -z "${PY_PID:-}" ]]; then
        return
    fi
    # Kill the whole process group in case native child processes were orphaned.
    kill -TERM "-$PY_PID" 2>/dev/null || true
    sleep 1
    kill -KILL "-$PY_PID" 2>/dev/null || true
}
trap cleanup_pg EXIT INT TERM

set +e
setsid "${CMD[@]}" &
PY_PID=$!
wait "$PY_PID"
RC=$?
set -e

if [[ $RC -ne 0 ]]; then
    echo "[run_profile] ERROR: eval process exited with code $RC" >&2
fi
exit "$RC"

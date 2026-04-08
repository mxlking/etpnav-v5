#!/usr/bin/env bash
set -euo pipefail

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"

stage="${1:-main}"
mode="${2:-train}"
MASTER_PORT="${3:-29541}"
EXP_NAME="${4:-efes_r2r_${stage}}"
CKPT_PATH="${5:-}"
PRED_FILE="${6:-preds.json}"
if [ "$mode" = "infer" ]; then
    shift $(( $# >= 6 ? 6 : $# )) || true
else
    shift $(( $# >= 5 ? 5 : $# )) || true
fi
EXTRA_OPTS=("$@")

EXP_CONFIG="run_r2r/efes/efes_main.yaml"
STAGE_OPTS=()
case "$stage" in
    main)
        ;;
    no_macro)
        STAGE_OPTS+=(EFES.use_macro False EFES.lambda_node 0.0)
        ;;
    no_grounding)
        STAGE_OPTS+=(EFES.use_grounding False)
        ;;
    no_confidence)
        STAGE_OPTS+=(EFES.use_confidence False EFES.lambda_cal 0.0)
        ;;
    no_recovery)
        STAGE_OPTS+=(EFES.use_recovery False)
        ;;
    *)
        echo "Unknown EFES stage: $stage"
        echo "Valid stages: main no_macro no_grounding no_confidence no_recovery"
        exit 1
        ;;
esac

PYTHON_BIN="${PYTHON_BIN:-python}"
PER_GPU_ENVS="${PER_GPU_ENVS:-3}"
TRAIN_ENVS_PER_RANK="${TRAIN_ENVS_PER_RANK:-$PER_GPU_ENVS}"
EVAL_ENVS_PER_RANK="${EVAL_ENVS_PER_RANK:-$PER_GPU_ENVS}"
MAX_EVAL_SCENES=11
IL_LOG_EVERY="${IL_LOG_EVERY:-200}"
STEP_LOG_EVERY="${STEP_LOG_EVERY:-1}"
USE_TQDM="${USE_TQDM:-True}"
WRITE_STEP_METRICS="${WRITE_STEP_METRICS:-True}"
LOG_FULL_MODEL_REPORT_TO_RUNTIME="${LOG_FULL_MODEL_REPORT_TO_RUNTIME:-False}"
BACK_ALGO="${BACK_ALGO:-}"

build_relative_gpu_list() {
    local count="$1"
    local gpu_list="["
    local i
    for ((i=0; i<count; i++)); do
        if [ "$i" -gt 0 ]; then
            gpu_list+=","
        fi
        gpu_list+="$i"
    done
    gpu_list+="]"
    printf '%s' "$gpu_list"
}

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra _devs <<< "$CUDA_VISIBLE_DEVICES"
    NPROC="${#_devs[@]}"
    if [ "$NPROC" -lt 1 ]; then
        NPROC=1
    fi
else
    NPROC=1
fi

GPU_LIST="$(build_relative_gpu_list "$NPROC")"
TRAIN_ENV_COUNT="$TRAIN_ENVS_PER_RANK"
EVAL_ENV_COUNT="$EVAL_ENVS_PER_RANK"
if [ "$EVAL_ENV_COUNT" -gt "$MAX_EVAL_SCENES" ]; then
    EVAL_ENV_COUNT="$MAX_EVAL_SCENES"
fi

export HABITAT_LOCAL_BASE="${HABITAT_LOCAL_BASE:-/home/D/liumeng/env/vln_env/habitat-lab/habitat-lab}"
export HABITAT_LOCAL_BASELINES="${HABITAT_LOCAL_BASELINES:-/home/D/liumeng/env/vln_env/habitat-lab/habitat-baselines}"
export PYTHONPATH="$HABITAT_LOCAL_BASELINES:$HABITAT_LOCAL_BASE:${PYTHONPATH:-}"

COMMON_GPU_ARGS=(
    SIMULATOR_GPU_IDS "$GPU_LIST"
    TORCH_GPU_IDS "$GPU_LIST"
    GPU_NUMBERS "$NPROC"
)

COMMON_LOGGING_ARGS=(
    IL.log_every "$IL_LOG_EVERY"
    EFES.LOGGING.step_log_every "$STEP_LOG_EVERY"
    EFES.LOGGING.use_tqdm "$USE_TQDM"
    EFES.LOGGING.write_step_metrics "$WRITE_STEP_METRICS"
    EFES.LOGGING.log_full_model_report_to_runtime "$LOG_FULL_MODEL_REPORT_TO_RUNTIME"
)

TRAIN_ARGS=(
    --exp_name "$EXP_NAME"
    --run-type train
    --exp-config "$EXP_CONFIG"
    "${COMMON_GPU_ARGS[@]}"
    "${COMMON_LOGGING_ARGS[@]}"
    "${STAGE_OPTS[@]}"
    NUM_ENVIRONMENTS "$TRAIN_ENV_COUNT"
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
)

if [ -n "$CKPT_PATH" ]; then
    TRAIN_ARGS+=(IL.load_from_ckpt True IL.ckpt_to_load "$CKPT_PATH")
fi
if [ "${#EXTRA_OPTS[@]}" -gt 0 ]; then
    TRAIN_ARGS+=("${EXTRA_OPTS[@]}")
fi

EVAL_ARGS=(
    --exp_name "$EXP_NAME"
    --run-type eval
    --exp-config "$EXP_CONFIG"
    "${COMMON_GPU_ARGS[@]}"
    "${COMMON_LOGGING_ARGS[@]}"
    "${STAGE_OPTS[@]}"
    NUM_ENVIRONMENTS "$EVAL_ENV_COUNT"
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
    EVAL.CKPT_PATH_DIR "$CKPT_PATH"
)
if [ -n "$BACK_ALGO" ]; then
    EVAL_ARGS+=(IL.back_algo "$BACK_ALGO")
fi
if [ "${#EXTRA_OPTS[@]}" -gt 0 ]; then
    EVAL_ARGS+=("${EXTRA_OPTS[@]}")
fi

INFER_ARGS=(
    --exp_name "$EXP_NAME"
    --run-type inference
    --exp-config "$EXP_CONFIG"
    "${COMMON_GPU_ARGS[@]}"
    "${COMMON_LOGGING_ARGS[@]}"
    "${STAGE_OPTS[@]}"
    NUM_ENVIRONMENTS "$EVAL_ENV_COUNT"
    TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True
    INFERENCE.CKPT_PATH "$CKPT_PATH"
    INFERENCE.PREDICTIONS_FILE "$PRED_FILE"
)
if [ -n "$BACK_ALGO" ]; then
    INFER_ARGS+=(IL.back_algo "$BACK_ALGO")
fi
if [ "${#EXTRA_OPTS[@]}" -gt 0 ]; then
    INFER_ARGS+=("${EXTRA_OPTS[@]}")
fi

run_single() {
    local -a args=("$@")
    "$PYTHON_BIN" run.py --local_rank 0 "${args[@]}"
}

run_ddp() {
    local -a args=("$@")
    "$PYTHON_BIN" -m torch.distributed.launch \
        --nproc_per_node="$NPROC" \
        --master_port="$MASTER_PORT" \
        run.py "${args[@]}"
}

case "$mode" in
    train)
        if [ "$NPROC" -eq 1 ]; then
            run_single "${TRAIN_ARGS[@]}"
        else
            run_ddp "${TRAIN_ARGS[@]}"
        fi
        ;;
    eval)
        if [ -z "$CKPT_PATH" ]; then
            echo "eval mode requires CKPT_PATH"
            exit 1
        fi
        if [ "$NPROC" -eq 1 ]; then
            run_single "${EVAL_ARGS[@]}"
        else
            run_ddp "${EVAL_ARGS[@]}"
        fi
        ;;
    infer)
        if [ -z "$CKPT_PATH" ]; then
            echo "infer mode requires CKPT_PATH"
            exit 1
        fi
        if [ "$NPROC" -eq 1 ]; then
            run_single "${INFER_ARGS[@]}"
        else
            run_ddp "${INFER_ARGS[@]}"
        fi
        ;;
    *)
        echo "Unknown mode: $mode"
        exit 1
        ;;
esac

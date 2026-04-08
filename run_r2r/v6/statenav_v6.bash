#!/usr/bin/env bash
set -euo pipefail

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${repo_root}/.." && pwd)"

stage="${1:-main}"
mode="${2:-train}"
MASTER_PORT="${3:-29531}"
EXP_NAME="${4:-statenav_v6_r2r_${stage}}"
CKPT_PATH="${5:-}"
PRED_FILE="${6:-preds.json}"
if [ "$mode" = "infer" ]; then
    shift $(( $# >= 6 ? 6 : $# )) || true
else
    shift $(( $# >= 5 ? 5 : $# )) || true
fi
EXTRA_OPTS=("$@")

EXP_CONFIG="run_r2r/v6/statenav_v6_main.yaml"
STAGE_OPTS=()
case "$stage" in
    main)
        ;;
    no_node_surprise)
        STAGE_OPTS+=(
            STATENAV.use_node_surprise False
            STATENAV.beta_node 0.0
            STATENAV.lambda_node_surprise 0.0
        )
        ;;
    no_topo_gate)
        STAGE_OPTS+=(
            STATENAV.use_topo_gate False
        )
        ;;
    no_level3)
        STAGE_OPTS+=(
            STATENAV.enable_level3_macro False
        )
        ;;
    latent_only)
        STAGE_OPTS+=(
            STATENAV.use_topo_bank False
            STATENAV.use_node_surprise False
            STATENAV.beta_node 0.0
            STATENAV.lambda_node_surprise 0.0
        )
        ;;
    topo_only)
        STAGE_OPTS+=(
            STATENAV.alpha_latent 0.0
            STATENAV.use_topo_bank True
            STATENAV.use_node_surprise True
        )
        ;;
    *)
        echo "Unknown V6 stage: $stage"
        echo "Valid stages: main no_node_surprise no_topo_gate no_level3 latent_only topo_only"
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
TOTAL_TRAIN_ENVS=$((TRAIN_ENV_COUNT * NPROC))
TOTAL_EVAL_ENVS=$((EVAL_ENV_COUNT * NPROC))

export HABITAT_LOCAL_BASE="${HABITAT_LOCAL_BASE:-${workspace_root}/habitat-lab/habitat-lab}"
export HABITAT_LOCAL_BASELINES="${HABITAT_LOCAL_BASELINES:-${workspace_root}/habitat-lab/habitat-baselines}"
export PYTHONPATH="$HABITAT_LOCAL_BASELINES:$HABITAT_LOCAL_BASE:${PYTHONPATH:-}"

COMMON_GPU_ARGS=(
    SIMULATOR_GPU_IDS "$GPU_LIST"
    TORCH_GPU_IDS "$GPU_LIST"
    GPU_NUMBERS "$NPROC"
)

COMMON_LOGGING_ARGS=(
    IL.log_every "$IL_LOG_EVERY"
    STATENAV.LOGGING.step_log_every "$STEP_LOG_EVERY"
    STATENAV.LOGGING.use_tqdm "$USE_TQDM"
    STATENAV.LOGGING.write_step_metrics "$WRITE_STEP_METRICS"
    STATENAV.LOGGING.log_full_model_report_to_runtime "$LOG_FULL_MODEL_REPORT_TO_RUNTIME"
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

echo "============================================================"
echo "STAGE                = $stage"
echo "MODE                 = $mode"
echo "MASTER_PORT          = $MASTER_PORT"
echo "EXP_NAME             = $EXP_NAME"
echo "EXP_CONFIG           = $EXP_CONFIG"
echo "CKPT_PATH            = ${CKPT_PATH:-<yaml default>}"
echo "PRED_FILE            = $PRED_FILE"
echo "PYTHON_BIN           = $PYTHON_BIN"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "NPROC                = $NPROC"
echo "GPU_LIST             = $GPU_LIST"
echo "TRAIN_ENVS_PER_RANK  = $TRAIN_ENV_COUNT"
echo "TOTAL_TRAIN_ENVS     = $TOTAL_TRAIN_ENVS"
echo "EVAL_ENVS_PER_RANK   = $EVAL_ENV_COUNT"
echo "TOTAL_EVAL_ENVS      = $TOTAL_EVAL_ENVS"
echo "IL_LOG_EVERY         = $IL_LOG_EVERY"
echo "STEP_LOG_EVERY       = $STEP_LOG_EVERY"
echo "USE_TQDM             = $USE_TQDM"
echo "WRITE_STEP_METRICS   = $WRITE_STEP_METRICS"
echo "BACK_ALGO            = ${BACK_ALGO:-<yaml default>}"
echo "STAGE_OPTS           = ${STAGE_OPTS[*]:-<none>}"
echo "EXTRA_OPTS           = ${EXTRA_OPTS[*]:-<none>}"
echo "============================================================"

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

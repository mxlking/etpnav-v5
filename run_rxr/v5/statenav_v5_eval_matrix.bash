#!/usr/bin/env bash
set -euo pipefail

stage="${1:-stage1}"
ckpt_path="${2:-}"
exp_prefix="${3:-diag_eval_matrix_rxr_${stage}}"
master_port_base="${4:-29631}"
if [ "$#" -ge 4 ]; then
    shift 4
elif [ "$#" -ge 3 ]; then
    shift 3
elif [ "$#" -ge 2 ]; then
    shift 2
else
    shift "$#" || true
fi
extra_opts=("$@")
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ -z "$ckpt_path" ]; then
    echo "Usage: $0 [stage] <ckpt_path> [exp_prefix] [master_port_base] [extra opts...]"
    echo "Example: CUDA_VISIBLE_DEVICES=5 EVAL_ENVS_PER_RANK=1 $0 stage1 data/logs/checkpoints/exp_rxr_v5_s1/best.pth"
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
run_script="${script_dir}/statenav_v5.bash"
repo_root="$(cd "${script_dir}/../.." && pwd)"
results_root="${repo_root}/data/logs/eval_results"
summary_script="${repo_root}/scripts/v5_eval_summary.py"
failures=0

run_case() {
    local suffix="$1"
    local master_port="$2"
    shift 2
    local -a case_opts=("$@")

    echo "============================================================"
    echo "RxR DIAGNOSTIC CASE   = ${suffix}"
    echo "STAGE                 = ${stage}"
    echo "CKPT_PATH             = ${ckpt_path}"
    echo "EXP_NAME              = ${exp_prefix}_${suffix}"
    echo "MASTER_PORT           = ${master_port}"
    echo "EXTRA_OPTS            = ${extra_opts[*]:-<none>}"
    echo "CASE_OPTS             = ${case_opts[*]:-<none>}"
    echo "============================================================"

    if bash "${run_script}" \
        "${stage}" \
        eval \
        "${master_port}" \
        "${exp_prefix}_${suffix}" \
        "${ckpt_path}" \
        "${extra_opts[@]}" \
        "${case_opts[@]}"; then
        return 0
    fi

    failures=$((failures + 1))
    echo "RxR diagnostic case failed: ${suffix}"
    return 0
}

run_case "statenav" "${master_port_base}" \
    STATENAV.eval_action_source statenav \
    STATENAV.enable_level3_macro True

run_case "etp" "$((master_port_base + 1))" \
    STATENAV.eval_action_source etp \
    STATENAV.enable_level3_macro True

run_case "no_level3" "$((master_port_base + 2))" \
    STATENAV.eval_action_source statenav \
    STATENAV.enable_level3_macro False

"${PYTHON_BIN}" "${summary_script}" \
    "${results_root}/${exp_prefix}_statenav" \
    "${results_root}/${exp_prefix}_etp" \
    "${results_root}/${exp_prefix}_no_level3" \
    --output-json "${results_root}/${exp_prefix}_matrix_summary.json"

echo "Completed diagnostic eval matrix for RxR."
echo "Expected runs: ${exp_prefix}_statenav, ${exp_prefix}_etp, ${exp_prefix}_no_level3"
if [ "${failures}" -gt 0 ]; then
    echo "One or more diagnostic cases failed. Combined summary still written if artifacts were available."
    exit 1
fi

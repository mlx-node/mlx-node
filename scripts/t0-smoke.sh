#!/usr/bin/env bash
# T=0 byte-equivalence smoke runner for the chat-engine refactor (S0).
#
# Usage:
#   scripts/t0-smoke.sh capture    — run all families, write digests to .t0-smoke/baseline/
#   scripts/t0-smoke.sh compare    — run all families, write digests to .t0-smoke/current/,
#                                    diff against baseline, print PASS/FAIL per family
#
# Model paths can be overridden via env vars:
#   SMOKE_QWEN3_PATH  SMOKE_QWEN35_PATH  SMOKE_GEMMA4_PATH  SMOKE_LFM2_PATH  SMOKE_QWEN35MOE_PATH
#
# GPU contention: families are run SERIALLY (never in parallel).
# Cold USB mmap: lfm2 prewarms the page cache during load; allow up to 15 minutes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Defaults (override via env)
# ---------------------------------------------------------------------------
SMOKE_QWEN3_PATH="${SMOKE_QWEN3_PATH:-/Volumes/P4510/models/qwen3-0.6b-mlx-bf16}"
SMOKE_QWEN35_PATH="${SMOKE_QWEN35_PATH:-/Volumes/P4510/models/qwen3.5-0.8b-mlx-bf16}"
SMOKE_GEMMA4_PATH="${SMOKE_GEMMA4_PATH:-/Volumes/P4510/models/gemma-4-e2b-it-mlx}"
SMOKE_LFM2_PATH="${SMOKE_LFM2_PATH:-/Volumes/P4510/models/lfm2.5-1.2b-thinking-mlx}"
SMOKE_QWEN35MOE_PATH="${SMOKE_QWEN35MOE_PATH:-/Volumes/P4510/models/Qwen3.6-35b-a3b-UD-Q4_K_XL-mlx}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
BASELINE_DIR="$REPO_ROOT/.t0-smoke/baseline"
CURRENT_DIR="$REPO_ROOT/.t0-smoke/current"

log() { echo "[t0-smoke] $*"; }
log_err() { echo "[t0-smoke] ERROR: $*" >&2; }

# Run a single family's cargo test, writing the digest to $out_dir/<family>.json.
# Returns 0 on success, 1 on SKIPPED (model path missing), 2 on failure.
run_family() {
    local family="$1"        # e.g. qwen3
    local test_name="$2"     # e.g. qwen3_smoke
    local model_path="$3"    # path to model dir
    local env_var="$4"       # e.g. MLX_SMOKE_QWEN3_MODEL_PATH
    local out_dir="$5"       # where to write <family>.json

    if [[ ! -d "$model_path" ]]; then
        log "SKIPPED $family: model path not found: $model_path"
        return 1
    fi

    log "Running $family (model=$model_path) ..."
    local cargo_env=(
        "env"
        "${env_var}=${model_path}"
        "MLX_SMOKE_OUT=${out_dir}"
        "PATH=/usr/bin:$PATH"
        "SDKROOT=$(xcrun --show-sdk-path)"
    )

    if "${cargo_env[@]}" cargo test -p mlx-core --test t0_smoke "$test_name" \
            -- --exact --nocapture --ignored 2>&1; then
        log "OK $family"
        return 0
    else
        log_err "FAILED $family — cargo test exited non-zero"
        return 2
    fi
}

# ---------------------------------------------------------------------------
# Mode: capture
# ---------------------------------------------------------------------------
do_capture() {
    mkdir -p "$BASELINE_DIR"
    log "Capturing baselines into $BASELINE_DIR"

    declare -A results=()

    # --- qwen3 ---
    run_family qwen3 qwen3_smoke "$SMOKE_QWEN3_PATH" MLX_SMOKE_QWEN3_MODEL_PATH "$BASELINE_DIR" \
        && results[qwen3]=CAPTURED || { [[ $? -eq 1 ]] && results[qwen3]=SKIPPED || results[qwen3]=FAILED; }

    # --- qwen3_5 ---
    run_family qwen3_5 qwen3_5_smoke "$SMOKE_QWEN35_PATH" MLX_SMOKE_QWEN35_MODEL_PATH "$BASELINE_DIR" \
        && results[qwen3_5]=CAPTURED || { [[ $? -eq 1 ]] && results[qwen3_5]=SKIPPED || results[qwen3_5]=FAILED; }

    # --- gemma4 ---
    run_family gemma4 gemma4_smoke "$SMOKE_GEMMA4_PATH" MLX_SMOKE_GEMMA4_MODEL_PATH "$BASELINE_DIR" \
        && results[gemma4]=CAPTURED || { [[ $? -eq 1 ]] && results[gemma4]=SKIPPED || results[gemma4]=FAILED; }

    # --- lfm2 (allow up to 15 minutes for cold USB mmap) ---
    run_family lfm2 lfm2_smoke "$SMOKE_LFM2_PATH" MLX_SMOKE_LFM2_MODEL_PATH "$BASELINE_DIR" \
        && results[lfm2]=CAPTURED || { [[ $? -eq 1 ]] && results[lfm2]=SKIPPED || results[lfm2]=FAILED; }

    # --- qwen3_5_moe (may be slow on cold USB) ---
    run_family qwen3_5_moe qwen3_5_moe_smoke "$SMOKE_QWEN35MOE_PATH" MLX_SMOKE_QWEN35MOE_MODEL_PATH "$BASELINE_DIR" \
        && results[qwen3_5_moe]=CAPTURED || { [[ $? -eq 1 ]] && results[qwen3_5_moe]=SKIPPED || results[qwen3_5_moe]=FAILED; }

    echo ""
    echo "=== capture summary ==="
    local any_failed=0
    for family in qwen3 qwen3_5 gemma4 lfm2 qwen3_5_moe; do
        echo "  ${results[$family]:-UNKNOWN}  $family"
        [[ "${results[$family]:-UNKNOWN}" == "FAILED" ]] && any_failed=1
    done
    echo ""
    echo "Baseline written to $BASELINE_DIR"
    [[ $any_failed -eq 0 ]] && return 0 || return 1
}

# ---------------------------------------------------------------------------
# Mode: compare
# ---------------------------------------------------------------------------
do_compare() {
    if [[ ! -d "$BASELINE_DIR" ]]; then
        log_err "No baseline found at $BASELINE_DIR — run 'capture' first"
        exit 1
    fi

    mkdir -p "$CURRENT_DIR"
    log "Comparing against baseline in $BASELINE_DIR"

    declare -A run_results=()

    # Run all families into current/
    run_family qwen3 qwen3_smoke "$SMOKE_QWEN3_PATH" MLX_SMOKE_QWEN3_MODEL_PATH "$CURRENT_DIR" \
        && run_results[qwen3]=RAN || { [[ $? -eq 1 ]] && run_results[qwen3]=SKIPPED || run_results[qwen3]=EXEC_FAILED; }

    run_family qwen3_5 qwen3_5_smoke "$SMOKE_QWEN35_PATH" MLX_SMOKE_QWEN35_MODEL_PATH "$CURRENT_DIR" \
        && run_results[qwen3_5]=RAN || { [[ $? -eq 1 ]] && run_results[qwen3_5]=SKIPPED || run_results[qwen3_5]=EXEC_FAILED; }

    run_family gemma4 gemma4_smoke "$SMOKE_GEMMA4_PATH" MLX_SMOKE_GEMMA4_MODEL_PATH "$CURRENT_DIR" \
        && run_results[gemma4]=RAN || { [[ $? -eq 1 ]] && run_results[gemma4]=SKIPPED || run_results[gemma4]=EXEC_FAILED; }

    run_family lfm2 lfm2_smoke "$SMOKE_LFM2_PATH" MLX_SMOKE_LFM2_MODEL_PATH "$CURRENT_DIR" \
        && run_results[lfm2]=RAN || { [[ $? -eq 1 ]] && run_results[lfm2]=SKIPPED || run_results[lfm2]=EXEC_FAILED; }

    run_family qwen3_5_moe qwen3_5_moe_smoke "$SMOKE_QWEN35MOE_PATH" MLX_SMOKE_QWEN35MOE_MODEL_PATH "$CURRENT_DIR" \
        && run_results[qwen3_5_moe]=RAN || { [[ $? -eq 1 ]] && run_results[qwen3_5_moe]=SKIPPED || run_results[qwen3_5_moe]=EXEC_FAILED; }

    echo ""
    echo "=== compare summary ==="
    local any_failed=0
    for family in qwen3 qwen3_5 gemma4 lfm2 qwen3_5_moe; do
        local run_status="${run_results[$family]:-UNKNOWN}"

        if [[ "$run_status" == "SKIPPED" ]]; then
            echo "  SKIPPED  $family (model not found)"
            continue
        fi

        if [[ "$run_status" == "EXEC_FAILED" ]]; then
            echo "  FAIL     $family (cargo test failed)"
            any_failed=1
            continue
        fi

        local baseline="$BASELINE_DIR/${family}.json"
        local current="$CURRENT_DIR/${family}.json"

        if [[ ! -f "$baseline" ]]; then
            echo "  SKIPPED  $family (no baseline digest — was this family skipped during capture?)"
            continue
        fi

        if [[ ! -f "$current" ]]; then
            echo "  FAIL     $family (no current digest produced)"
            any_failed=1
            continue
        fi

        if diff -q "$baseline" "$current" >/dev/null 2>&1; then
            echo "  PASS     $family"
        else
            echo "  FAIL     $family (digest mismatch)"
            diff --unified=5 "$baseline" "$current" || true
            any_failed=1
        fi
    done

    echo ""
    if [[ $any_failed -eq 0 ]]; then
        echo "All captured families PASSED."
        return 0
    else
        echo "One or more families FAILED — see diff above."
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
MODE="${1:-}"
case "$MODE" in
    capture) do_capture ;;
    compare) do_compare ;;
    *)
        echo "Usage: $0 {capture|compare}"
        echo ""
        echo "  capture  — run all families, write digests to .t0-smoke/baseline/"
        echo "  compare  — run all families, compare digests to baseline, exit nonzero on diff"
        exit 1
        ;;
esac

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
#
# S0 review fix (Finding 1): all 5 families are REQUIRED in both capture and
# compare. Missing model path, missing/empty digest, or missing baseline all
# produce loud errors and nonzero exit. Optional escape hatch:
#   MLX_SMOKE_ALLOW_SKIP="fam1 fam2"  — exempts listed families (prints WARNING).

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

# Return 0 if the given family name is listed in MLX_SMOKE_ALLOW_SKIP.
is_skip_allowed() {
    local family="$1"
    local skip_list="${MLX_SMOKE_ALLOW_SKIP:-}"
    for s in $skip_list; do
        [[ "$s" == "$family" ]] && return 0
    done
    return 1
}

# Assert that a digest file exists, is non-empty, and is valid JSON.
# Exits with error if any check fails.
assert_digest_valid() {
    local family="$1"
    local path="$2"

    if [[ ! -f "$path" ]]; then
        log_err "FATAL: expected digest file not produced for $family: $path"
        exit 1
    fi
    if [[ ! -s "$path" ]]; then
        log_err "FATAL: digest file for $family is empty: $path"
        exit 1
    fi
    if ! python3 -m json.tool "$path" > /dev/null 2>&1; then
        log_err "FATAL: digest file for $family is not valid JSON: $path"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Metallib canary guard — run before capture OR compare.
# Executes 4 fast behavioral tests that fail iff mlx.metallib is broken.
# A sha-pin would false-positive on legitimate rebuilds; behavioral tests don't.
# ---------------------------------------------------------------------------
check_metallib_canaries() {
    log "Running metallib canary tests ..."
    local canary_tests=(
        "models::lfm2::sparse_moe::tests::forward_64_index_gather_sort_branch"
        "models::qwen3::model::tests::test_chunked_prefill_uneven_tail_matches_single_shot"
        "nn::embedding::tests::packed_affine_as_linear_matches_dense_matmul"
        "array::banded_attention::tests::banded_attention_matches_reference_on_random_inputs"
    )

    if PATH=/usr/bin:$PATH SDKROOT="$(xcrun --show-sdk-path)" \
       cargo test -p mlx-core --lib -- --exact "${canary_tests[@]}" 2>&1; then
        log "Metallib canary: OK"
    else
        echo "" >&2
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
        echo "  METALLIB CANARY FAILED — aborting $MODE" >&2
        echo "" >&2
        echo "  One or more of the following metallib canary tests failed:" >&2
        for t in "${canary_tests[@]}"; do
            echo "    - $t" >&2
        done
        echo "" >&2
        echo "  Root cause: the locally-built target/debug/deps/mlx.metallib is likely" >&2
        echo "  broken. This is a known nondeterministic Metal build issue — the library" >&2
        echo "  compiles silently but drops kernel symbols, producing garbage output." >&2
        echo "" >&2
        echo "  Remediation (install known-good metallib and re-run):" >&2
        echo "    cp /Users/brooklyn/workspace/github/mlx-node/packages/core/mlx.metallib \\" >&2
        echo "       target/debug/deps/mlx.metallib" >&2
        echo "    scripts/t0-smoke.sh $MODE" >&2
        echo "" >&2
        echo "  Known-good sha256:" >&2
        echo "    84c24f93d8efd2d3930d362bc431753b3848e65a0527d8b88ec82f6bab5455b8" >&2
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
        echo "" >&2
        exit 1
    fi
}

# Run a single family's cargo test, writing the digest to $out_dir/<family>.json.
# S0 review fix (Finding 1):
#   - Missing model dir is an ERROR (not a soft skip) unless the family is in
#     MLX_SMOKE_ALLOW_SKIP.
#   - After a successful cargo run, assert the expected digest file exists,
#     is non-empty, and is valid JSON.
# Returns 0 on success. Exits with nonzero on any hard error.
# Returns 1 only when the family is in MLX_SMOKE_ALLOW_SKIP and the model is absent.
run_family() {
    local family="$1"        # e.g. qwen3
    local test_name="$2"     # e.g. qwen3_smoke
    local model_path="$3"    # path to model dir
    local env_var="$4"       # e.g. MLX_SMOKE_QWEN3_MODEL_PATH
    local out_dir="$5"       # where to write <family>.json

    if [[ ! -d "$model_path" ]]; then
        if is_skip_allowed "$family"; then
            echo "" >&2
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
            echo "  WARNING: $family skipped via MLX_SMOKE_ALLOW_SKIP" >&2
            echo "  Model path not found: $model_path" >&2
            echo "  This family will NOT be checked — gate is weakened." >&2
            echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
            echo "" >&2
            return 1
        fi
        log_err "FATAL: model path for $family not found: $model_path"
        local family_upper
        family_upper="$(echo "$family" | tr 'a-z' 'A-Z')"
        log_err "  Set SMOKE_${family_upper}_PATH or add $family to MLX_SMOKE_ALLOW_SKIP to skip."
        exit 1
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
        # Assert that the digest file was actually produced and is valid.
        assert_digest_valid "$family" "${out_dir}/${family}.json"
        log "OK $family"
        return 0
    else
        log_err "FAILED $family — cargo test exited non-zero"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Mode: capture
# ---------------------------------------------------------------------------
do_capture() {
    check_metallib_canaries
    mkdir -p "$BASELINE_DIR"
    log "Capturing baselines into $BASELINE_DIR"

    # Bash-3.2 compatible: per-family scalar variables instead of associative array.
    local res_qwen3=UNKNOWN res_qwen3_5=UNKNOWN res_gemma4=UNKNOWN res_lfm2=UNKNOWN res_qwen3_5_moe=UNKNOWN

    # --- qwen3 ---
    run_family qwen3 qwen3_smoke "$SMOKE_QWEN3_PATH" MLX_SMOKE_QWEN3_MODEL_PATH "$BASELINE_DIR" \
        && res_qwen3=CAPTURED || res_qwen3=SKIPPED

    # --- qwen3_5 ---
    run_family qwen3_5 qwen3_5_smoke "$SMOKE_QWEN35_PATH" MLX_SMOKE_QWEN35_MODEL_PATH "$BASELINE_DIR" \
        && res_qwen3_5=CAPTURED || res_qwen3_5=SKIPPED

    # --- gemma4 ---
    run_family gemma4 gemma4_smoke "$SMOKE_GEMMA4_PATH" MLX_SMOKE_GEMMA4_MODEL_PATH "$BASELINE_DIR" \
        && res_gemma4=CAPTURED || res_gemma4=SKIPPED

    # --- lfm2 (allow up to 15 minutes for cold USB mmap) ---
    run_family lfm2 lfm2_smoke "$SMOKE_LFM2_PATH" MLX_SMOKE_LFM2_MODEL_PATH "$BASELINE_DIR" \
        && res_lfm2=CAPTURED || res_lfm2=SKIPPED

    # --- qwen3_5_moe (may be slow on cold USB) ---
    run_family qwen3_5_moe qwen3_5_moe_smoke "$SMOKE_QWEN35MOE_PATH" MLX_SMOKE_QWEN35MOE_MODEL_PATH "$BASELINE_DIR" \
        && res_qwen3_5_moe=CAPTURED || res_qwen3_5_moe=SKIPPED

    echo ""
    echo "=== capture summary ==="
    local any_skipped=0
    for family in qwen3 qwen3_5 gemma4 lfm2 qwen3_5_moe; do
        local fam_status
        case "$family" in
            qwen3)       fam_status="$res_qwen3" ;;
            qwen3_5)     fam_status="$res_qwen3_5" ;;
            gemma4)      fam_status="$res_gemma4" ;;
            lfm2)        fam_status="$res_lfm2" ;;
            qwen3_5_moe) fam_status="$res_qwen3_5_moe" ;;
            *)           fam_status=UNKNOWN ;;
        esac
        echo "  $fam_status  $family"
        [[ "$fam_status" == "SKIPPED" ]] && any_skipped=1
    done
    echo ""
    echo "Baseline written to $BASELINE_DIR"
    if [[ $any_skipped -ne 0 ]]; then
        echo ""
        echo "WARNING: one or more families were SKIPPED (MLX_SMOKE_ALLOW_SKIP in effect)."
        echo "         The gate is weakened — missing families will not be checked in compare."
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Mode: compare
# ---------------------------------------------------------------------------
do_compare() {
    check_metallib_canaries
    if [[ ! -d "$BASELINE_DIR" ]]; then
        log_err "No baseline found at $BASELINE_DIR — run 'capture' first"
        exit 1
    fi

    # S0 review fix (Finding 1): wipe the current-run dir before every compare
    # so stale digests from a previous (possibly partial) run cannot satisfy checks.
    if [[ -d "$CURRENT_DIR" ]]; then
        log "Wiping stale current-run dir: $CURRENT_DIR"
        rm -rf "$CURRENT_DIR"
    fi
    mkdir -p "$CURRENT_DIR"
    log "Comparing against baseline in $BASELINE_DIR"

    # Bash-3.2 compatible: per-family scalar variables instead of associative array.
    local rr_qwen3=UNKNOWN rr_qwen3_5=UNKNOWN rr_gemma4=UNKNOWN rr_lfm2=UNKNOWN rr_qwen3_5_moe=UNKNOWN

    # Run all families into current/. run_family exits on hard errors;
    # return 1 only on MLX_SMOKE_ALLOW_SKIP soft-skip.
    run_family qwen3 qwen3_smoke "$SMOKE_QWEN3_PATH" MLX_SMOKE_QWEN3_MODEL_PATH "$CURRENT_DIR" \
        && rr_qwen3=RAN || rr_qwen3=SKIPPED

    run_family qwen3_5 qwen3_5_smoke "$SMOKE_QWEN35_PATH" MLX_SMOKE_QWEN35_MODEL_PATH "$CURRENT_DIR" \
        && rr_qwen3_5=RAN || rr_qwen3_5=SKIPPED

    run_family gemma4 gemma4_smoke "$SMOKE_GEMMA4_PATH" MLX_SMOKE_GEMMA4_MODEL_PATH "$CURRENT_DIR" \
        && rr_gemma4=RAN || rr_gemma4=SKIPPED

    run_family lfm2 lfm2_smoke "$SMOKE_LFM2_PATH" MLX_SMOKE_LFM2_MODEL_PATH "$CURRENT_DIR" \
        && rr_lfm2=RAN || rr_lfm2=SKIPPED

    run_family qwen3_5_moe qwen3_5_moe_smoke "$SMOKE_QWEN35MOE_PATH" MLX_SMOKE_QWEN35MOE_MODEL_PATH "$CURRENT_DIR" \
        && rr_qwen3_5_moe=RAN || rr_qwen3_5_moe=SKIPPED

    echo ""
    echo "=== compare summary ==="
    local any_failed=0
    for family in qwen3 qwen3_5 gemma4 lfm2 qwen3_5_moe; do
        local run_status
        case "$family" in
            qwen3)       run_status="$rr_qwen3" ;;
            qwen3_5)     run_status="$rr_qwen3_5" ;;
            gemma4)      run_status="$rr_gemma4" ;;
            lfm2)        run_status="$rr_lfm2" ;;
            qwen3_5_moe) run_status="$rr_qwen3_5_moe" ;;
            *)           run_status=UNKNOWN ;;
        esac

        if [[ "$run_status" == "SKIPPED" ]]; then
            # Skipped families must also be absent from baseline to be consistent.
            # A family that was captured but then skipped in compare is a gate hole.
            local baseline="$BASELINE_DIR/${family}.json"
            if [[ -f "$baseline" ]]; then
                log_err "FATAL: $family has a baseline digest but was SKIPPED in compare."
                log_err "  Either provide the model or remove the baseline and re-capture."
                any_failed=1
            else
                echo "  SKIPPED  $family (in MLX_SMOKE_ALLOW_SKIP, no baseline — consistent)"
            fi
            continue
        fi

        local baseline="$BASELINE_DIR/${family}.json"
        local current="$CURRENT_DIR/${family}.json"

        # S0 review fix (Finding 1): missing baseline is now a hard error.
        if [[ ! -f "$baseline" ]]; then
            log_err "FATAL: no baseline digest for $family at $baseline"
            log_err "  Run 'capture' first to record a baseline."
            any_failed=1
            continue
        fi

        # current digest must exist (assert_digest_valid already checked this
        # inside run_family on success, but guard again for clarity).
        if [[ ! -f "$current" ]]; then
            log_err "FAIL: no current digest for $family (cargo test ran but produced no file)"
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
        echo "All required families PASSED."
        return 0
    else
        echo "One or more families FAILED — see output above."
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

#!/usr/bin/env bash
# Autoresearch driver: rebuild native (if needed) → run prefill bench → emit
# METRIC lines.
#
# Reads experiment config from autoresearch.config.json. The config carries
# bench knobs (promptTokens, runs) and a free-form "notes" field the agent
# uses to record what kernel/algorithmic change was made for THIS run. The
# actual kernel edits live in tracked files; the JSON is a record + a knob
# set.
#
# Emits:
#   METRIC prefill_ms_median=... (primary, lower is better)
#   METRIC prefill_ms_min=...
#   METRIC max_abs_diff=...      (0.0 = correctness OK; 1.0 = fingerprint mismatch)
#   METRIC peak_mem_gb=...
#   METRIC build_seconds=...
#   METRIC prompt_token_count=...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CONFIG="${AUTORESEARCH_CONFIG:-autoresearch.config.json}"
BASELINE="${AUTORESEARCH_BASELINE:-.cache/autoresearch-flashqla-baseline.json}"
REPORT="/tmp/autoresearch-flashqla-bench.json"

[ -f "$CONFIG" ] || { echo "ERROR: $CONFIG not found"; exit 2; }
node -e "JSON.parse(require('fs').readFileSync('$CONFIG','utf8'))" \
  || { echo "ERROR: $CONFIG is not valid JSON"; exit 2; }

MODEL_DIR=$(node -e "console.log(JSON.parse(require('fs').readFileSync('$CONFIG','utf8')).modelPath)")
PROMPT_TOKENS=$(node -e "console.log(JSON.parse(require('fs').readFileSync('$CONFIG','utf8')).promptTokens)")
RUNS=$(node -e "console.log(JSON.parse(require('fs').readFileSync('$CONFIG','utf8')).runs)")
WARMUP=$(node -e "console.log(JSON.parse(require('fs').readFileSync('$CONFIG','utf8')).warmup)")
WRITE_BASELINE=$(node -e "console.log(JSON.parse(require('fs').readFileSync('$CONFIG','utf8')).writeBaseline ?? false)")
SKIP_BUILD=$(node -e "console.log(JSON.parse(require('fs').readFileSync('$CONFIG','utf8')).skipBuild ?? false)")

[ -d "$MODEL_DIR" ] || { echo "ERROR: model dir $MODEL_DIR missing"; exit 2; }

if [ "$SKIP_BUILD" != "true" ]; then
  echo ">>> building native"
  BUILD_START=$(date +%s)
  yarn build:native >/tmp/autoresearch-build.log 2>&1 || {
    echo "ERROR: yarn build:native failed; see /tmp/autoresearch-build.log"
    exit 4
  }
  BUILD_END=$(date +%s)
  BUILD_SECS=$((BUILD_END - BUILD_START))
  echo ">>> build done in ${BUILD_SECS}s"
else
  BUILD_SECS=0
  echo ">>> skipping build (skipBuild=true in config)"
fi

echo ">>> ensuring TS build is current"
yarn build:ts >/tmp/autoresearch-ts.log 2>&1 || {
  echo "ERROR: yarn build:ts failed; see /tmp/autoresearch-ts.log"
  exit 4
}

BASELINE_FLAG=()
if [ "$WRITE_BASELINE" = "true" ]; then
  BASELINE_FLAG=(--write-baseline)
  echo ">>> THIS RUN WRITES THE BASELINE (no correctness gate)"
fi

echo ">>> running prefill bench (prompt~${PROMPT_TOKENS} tokens, ${RUNS} runs, ${WARMUP} warmup)"
if [ -n "${MLX_NO_COMPILE:-}" ]; then echo ">>> MLX_NO_COMPILE=$MLX_NO_COMPILE (compiled forward DISABLED)"; fi
oxnode scripts/bench-gdn-prefill.ts \
  --model "$MODEL_DIR" \
  --prompt-tokens "$PROMPT_TOKENS" \
  --runs "$RUNS" \
  --warmup "$WARMUP" \
  --baseline "$BASELINE" \
  --out "$REPORT" \
  "${BASELINE_FLAG[@]}"

echo "METRIC build_seconds=${BUILD_SECS}"

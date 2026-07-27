#!/bin/bash
# Phase 0 A/B: identical MLX workload, bare Node vs Electron utilityProcess.
#
# The ONLY difference between the arms must be the host. Every MLX-relevant env
# var is exported here so neither arm can inherit a different default — the first
# attempt at this silently compared chunked prefill against unchunked, because
# spike/main.mjs defaults MLX_PAGED_PREFILL_CHUNK_SIZE and bare node does not.

set -u
cd "$(dirname "$0")/.."

export MLX_SPIKE_MODEL="$HOME/.mlx-node/models/qwen3.5-9B-bf16"
export MLX_SPIKE_DURATION_MS=240000
export MLX_SPIKE_MAX_NEW_TOKENS=256
export MLX_SPIKE_PROMPT_REPEATS=220
export MLX_PAGED_PREFILL_CHUNK_SIZE=2048

COOLDOWN=${COOLDOWN:-150}
OUT=spike/.spike-logs
mkdir -p "$OUT"

echo "===== ARM A: bare node ====="
node spike/workload.mjs > "$OUT/arm-a-node.jsonl" 2>&1
echo "arm A exit: $?"
grep -vE '"event":"token"' "$OUT/arm-a-node.jsonl" | tail -4

echo
echo "===== cooling ${COOLDOWN}s (M5 throttles; a hot second arm is not a fair test) ====="
sleep "$COOLDOWN"

echo "===== ARM B: electron utilityProcess (visible window, repainting) ====="
yarn exec electron spike/main.mjs > "$OUT/arm-b-electron.jsonl" 2>&1
echo "arm B exit: $?"
grep -vE '"event":"(token|child-stdout)"' "$OUT/arm-b-electron.jsonl" | tail -6

echo
echo "===== SUMMARY ====="
for arm in a-node b-electron; do
  f="$OUT/arm-$arm.jsonl"
  surv=$(grep -c '"event":"survived"' "$f" 2>/dev/null || echo 0)
  rounds=$(grep -o '"event":"round"' "$f" 2>/dev/null | wc -l | tr -d ' ')
  err=$(grep -c '"event":"error"' "$f" 2>/dev/null || echo 0)
  printf '  %-12s survived=%s rounds=%s errors=%s\n' "$arm" "$surv" "$rounds" "$err"
done
echo "  watchdog signature to look for: exit 11 / SIGABRT / kIOGPUCommandBufferCallbackErrorTimeout"
grep -il 'kIOGPU\|CommandBuffer.*[Tt]imeout' "$OUT"/*.jsonl 2>/dev/null | sed 's/^/  HIT: /' || echo "  no kIOGPU timeout strings found"

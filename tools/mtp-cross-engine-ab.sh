#!/usr/bin/env bash
# =============================================================================
# mtp-cross-engine-ab.sh
#
# THERMALLY-FAIR cross-engine A/B between OUR engine (mlx-node) and MTPLX on
# the IDENTICAL checkpoint, T=0 greedy, for {MTP depth-3, AR} x {count, essay}.
#
# WHY interleaved alternation + medians:
#   This M5 Max host is thermally UNSTABLE under sustained 27B load (AR CV
#   9-19%). A naive "all-ours then all-MTPLX" A/B confounds the engine
#   difference with monotonic thermal drift (whichever engine runs second eats
#   the accumulated heat). We instead INTERLEAVE the two engines block-by-block
#   and ALTERNATE which engine goes first on odd vs even reps. Over N reps the
#   warmer-slot bias lands on each engine equally often, so first-mover thermal
#   advantage cancels in the per-cell MEDIAN. Each 27B reload (~30-60s) is its
#   own cooldown; the explicit --cooldown sleep adds margin. We report the
#   MEDIAN (robust to the occasional throttle spike) plus min/max and the AR
#   coefficient-of-variation as an explicit thermal-stability witness.
#
# DELIBERATELY does NOT pin fans (no sudo allowed). If a fan-pin command is
# available, export FANPIN_CMD="..." and it runs before every block. MTPLX's
# own --max fan profile is also opt-in via MTPLX_MAX=1.
#
# Usage:
#   tools/mtp-cross-engine-ab.sh [--reps N] [--cooldown SEC] [--max-tokens N] \
#                                [--model NAME] [--csv PATH]
# Defaults: --reps 4 --cooldown 20 --max-tokens 120 --model mtplx-optimized-speed-mtp
#
# Env hooks:
#   FANPIN_CMD   shell command run once before each engine block (e.g. a fan-pin)
#   MTPLX_MAX=1  pass --max to MTPLX (its ThermalForge/TG Pro perf fan profile)
#   DRY_RUN=1    print the exact commands instead of running the 27B model
# =============================================================================
set -uo pipefail

# ----- repo / tool locations (absolute) --------------------------------------
REPO="/Users/brooklyn/workspace/github/mlx-node/.claude/worktrees/research+qwen-mtp"
MTPLX_BIN="/Users/brooklyn/workspace/github/MTPLX/.venv/bin/mtplx"
MTP_PROBE="examples/qwen35-mtp-cold-cache-probe.ts"   # MTP-only probe (our engine)
AR_PROBE="examples/_ar-probe-param.ts"                # AR (MTP-off) probe (our engine)

# Resolve a python3 robustly (the caller's PATH may not include Homebrew bin
# when this script is sourced from a non-interactive shell).
PY="$(command -v python3 || true)"
[[ -z "$PY" ]] && for c in /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
  [[ -x "$c" ]] && { PY="$c"; break; }
done
if [[ -z "$PY" ]]; then echo "FATAL: python3 not found (needed for JSON parse + stats)" >&2; exit 3; fi

# ----- defaults --------------------------------------------------------------
REPS=4
COOLDOWN=20
MAX_TOKENS=120
MODEL="mtplx-optimized-speed-mtp"          # our local merged checkpoint dir name
MTPLX_MODEL="Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
DEPTH=3
CSV=""

# Prompts (kept byte-identical across both engines)
PROMPT_COUNT="Count from 1 to 100, one number per line, nothing else."
PROMPT_ESSAY="Write a concise three-paragraph essay on why deterministic sampling at temperature 0 is useful for testing speculative decoding implementations."

# ----- arg parse -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reps)       REPS="$2"; shift 2 ;;
    --cooldown)   COOLDOWN="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --model)      MODEL="$2"; shift 2 ;;
    --mtplx-model) MTPLX_MODEL="$2"; shift 2 ;;
    --depth)      DEPTH="$2"; shift 2 ;;
    --csv)        CSV="$2"; shift 2 ;;
    -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

DRY_RUN="${DRY_RUN:-0}"
MTPLX_MAX="${MTPLX_MAX:-0}"
FANPIN_CMD="${FANPIN_CMD:-}"

MTPLX_MAX_FLAG=""
[[ "$MTPLX_MAX" == "1" ]] && MTPLX_MAX_FLAG="--max"

# ----- results store: key = "engine|mode|prompt" -> space-separated tok/s ----
declare -A SAMPLES         # decode tok/s samples
declare -A MTPLX_VERIFY    # verify_ms_per_call samples (MTPLX MTP only)
declare -A MTPLX_DRAFT     # draft_time_s samples (MTPLX MTP only)
declare -A MTPLX_ACCEPT    # last accepted_by_depth seen (MTPLX MTP only)

# strip our probes' noisy native log lines
GREP_FILTER='^\[|warning|Warning|Loading '

run_cmd() {
  # echoes the command, runs it unless DRY_RUN, returns its stdout
  local desc="$1"; shift
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY %s:\n  %s\n' "$desc" "$*" >&2
    return 0
  fi
  "$@"
}

# ----- one OUR-engine measurement (mode = mtp|ar) ----------------------------
measure_ours() {
  local mode="$1" prompt="$2"
  local out tps
  if [[ "$mode" == "mtp" ]]; then
    # chained-ON is the shipped default; the probe is MTP-only and forces it.
    out="$(run_cmd "ours/mtp" oxnode "$REPO/$MTP_PROBE" "$MODEL" "$DEPTH" "$MAX_TOKENS" "$prompt" 2>/dev/null \
            | grep -v -E "$GREP_FILTER")"
    # line: COLD MTP (...): cycles=.. mean_accepted=../cycle per_position=[..] decode=X tok/s
    tps="$(printf '%s\n' "$out" | sed -n 's/.*decode=\([0-9.]*\) tok\/s.*/\1/p' | tail -1)"
  else
    out="$(run_cmd "ours/ar" oxnode "$REPO/$AR_PROBE" "$MODEL" "$MAX_TOKENS" "$prompt" 2>/dev/null \
            | grep -v -E "$GREP_FILTER")"
    # line: AR <model>: decode=X tok/s | tokens=N
    tps="$(printf '%s\n' "$out" | sed -n 's/.*decode=\([0-9.]*\) tok\/s.*/\1/p' | tail -1)"
  fi
  printf '%s' "$tps"
}

# ----- one MTPLX measurement (mode = mtp|ar) ---------------------------------
measure_mtplx() {
  local mode="$1" prompt="$2"
  local mtp_flags json tps verify draft accept
  if [[ "$mode" == "mtp" ]]; then
    mtp_flags="--mtp --depth $DEPTH"
  else
    mtp_flags="--no-mtp"
  fi
  # shellcheck disable=SC2086
  json="$(run_cmd "mtplx/$mode" "$MTPLX_BIN" ask \
            --model "$MTPLX_MODEL" $mtp_flags $MTPLX_MAX_FLAG \
            --reasoning off --temperature 0 --top-p 1 --top-k 1 \
            --max-tokens "$MAX_TOKENS" --prompt "$prompt" \
            --stats --json --yes 2>/dev/null)"
  if [[ "$DRY_RUN" == "1" ]]; then printf '||'; return 0; fi
  # parse the nested stats object (json.dumps payload with a top-level "stats").
  # Emit a single TAB-free, '|'-delimited record: tps|verify_ms|draft_s|accept
  # (subshell var assignment can't escape command substitution, so we return
  #  every field on stdout and the caller splits it).
  tps="$(extract_json_num "$json" 'stats.decode_tok_s')"
  verify=""; draft=""; accept=""
  if [[ "$mode" == "mtp" ]]; then
    verify="$(extract_json_num "$json" 'stats.verify_ms_per_call')"
    draft="$(extract_json_num "$json" 'stats.draft_time_s')"
    accept="$(extract_json_arr "$json" 'stats.accepted_by_depth')"
  fi
  printf '%s|%s|%s|%s' "$tps" "$verify" "$draft" "$accept"
}

# ----- python helpers --------------------------------------------------------
# All args-only (no stdin) to avoid any heredoc/stdin collision. JSON and
# sample lists are passed as argv strings.

# extract_json_num <json> <dotted.path>  -> formatted float, or "" if null/missing
extract_json_num() {
  "$PY" -c '
import sys, json
try:
    d = json.loads(sys.argv[1])
    for k in sys.argv[2].split("."):
        d = d[k]
    print("" if d is None else f"{float(d):.4f}")
except Exception:
    print("")
' "$1" "$2"
}

# extract_json_arr <json> <dotted.path>  -> comma-joined ints, or "" if null/missing
extract_json_arr() {
  "$PY" -c '
import sys, json
try:
    d = json.loads(sys.argv[1])
    for k in sys.argv[2].split("."):
        d = d[k]
    print(",".join(str(int(x)) for x in (d or [])))
except Exception:
    print("")
' "$1" "$2"
}

# ----- stats helpers (median / min / max / cv over a sample list) ------------
# stats_line "<space-separated numbers>" -> "median min max n"
stats_line() {
  "$PY" -c '
import sys, statistics as st
vals=[float(x) for x in sys.argv[1].split() if x]
if not vals:
    print("NA NA NA 0")
else:
    m=st.median(vals)
    print(f"{m:.2f} {min(vals):.2f} {max(vals):.2f} {len(vals)}")
' "${1:-}"
}
# cv_pct "<space-separated numbers>" -> coefficient of variation (%) or "NA"
cv_pct() {
  "$PY" -c '
import sys, statistics as st
vals=[float(x) for x in sys.argv[1].split() if x]
if len(vals)<2:
    print("NA")
else:
    mean=st.mean(vals); sd=st.pstdev(vals)
    print(f"{(100.0*sd/mean):.1f}" if mean else "NA")
' "${1:-}"
}
# ratio <a> <b> -> a/b (3dp) or "NA"
ratio() {
  "$PY" -c '
import sys
try:
    a=float(sys.argv[1]); b=float(sys.argv[2])
    print(f"{a/b:.3f}" if b else "NA")
except Exception:
    print("NA")
' "$1" "$2"
}

append() { # append a sample to an assoc-array slot
  local -n arr="$1"; local key="$2" val="$3"
  [[ -z "$val" ]] && return 0
  arr["$key"]="${arr[$key]:-} $val"
}

cooldown() {
  [[ "$DRY_RUN" == "1" ]] && return 0
  echo ">> cooldown ${COOLDOWN}s" >&2
  # foreground sleep is blocked in some harnesses; tolerate failure.
  sleep "$COOLDOWN" 2>/dev/null || true
}

fanpin() {
  [[ -z "$FANPIN_CMD" ]] && return 0
  echo ">> FANPIN_CMD: $FANPIN_CMD" >&2
  [[ "$DRY_RUN" == "1" ]] && return 0
  bash -c "$FANPIN_CMD" || echo "   (fanpin command failed; continuing)" >&2
}

# Allow tests to source helpers without running the sweep.
[[ "${AB_SOURCE_ONLY:-0}" == "1" ]] && return 0 2>/dev/null

# ----- the interleaved sweep -------------------------------------------------
MODES=(mtp ar)
declare -A PROMPTS=( [count]="$PROMPT_COUNT" [essay]="$PROMPT_ESSAY" )
PROMPT_KEYS=(count essay)

echo "=============================================================="
echo " cross-engine A/B  model=$MODEL  mtplx=$MTPLX_MODEL"
echo " reps=$REPS cooldown=${COOLDOWN}s max_tokens=$MAX_TOKENS depth=$DEPTH"
echo " MTPLX_MAX=$MTPLX_MAX  FANPIN_CMD=${FANPIN_CMD:-<none>}  DRY_RUN=$DRY_RUN"
echo "=============================================================="

do_ours() {
  local mode="$1" pk="$2" prompt="$3"
  fanpin
  local t; t="$(measure_ours "$mode" "$prompt")"
  echo "   ours   $mode/$pk -> ${t:-NA} tok/s" >&2
  append SAMPLES "ours|$mode|$pk" "$t"
}
do_mtplx() {
  local mode="$1" pk="$2" prompt="$3"
  fanpin
  local rec; rec="$(measure_mtplx "$mode" "$prompt")"
  local t verify draft accept
  IFS='|' read -r t verify draft accept <<<"$rec"
  echo "   mtplx  $mode/$pk -> ${t:-NA} tok/s${accept:+ accept=[$accept] verify_ms=$verify draft_s=$draft}" >&2
  append SAMPLES "mtplx|$mode|$pk" "$t"
  if [[ "$mode" == "mtp" ]]; then
    append MTPLX_VERIFY "mtplx|$mode|$pk" "$verify"
    append MTPLX_DRAFT  "mtplx|$mode|$pk" "$draft"
    [[ -n "$accept" ]] && MTPLX_ACCEPT["mtplx|$mode|$pk"]="$accept"
  fi
}

for rep in $(seq 1 "$REPS"); do
  echo "----- rep $rep / $REPS -----" >&2
  for mode in "${MODES[@]}"; do
    for pk in "${PROMPT_KEYS[@]}"; do
      prompt="${PROMPTS[$pk]}"
      if (( rep % 2 == 1 )); then
        do_ours  "$mode" "$pk" "$prompt"; cooldown
        do_mtplx "$mode" "$pk" "$prompt"; cooldown
      else
        do_mtplx "$mode" "$pk" "$prompt"; cooldown
        do_ours  "$mode" "$pk" "$prompt"; cooldown
      fi
    done
  done
done

# ----- report ----------------------------------------------------------------
echo
echo "================================ RESULTS ================================"
printf '%-7s %-5s %-6s | %9s %9s %9s %3s\n' engine mode prompt median min max n
echo "------------------------------------------------------------------------"
for engine in ours mtplx; do
  for mode in "${MODES[@]}"; do
    for pk in "${PROMPT_KEYS[@]}"; do
      key="$engine|$mode|$pk"
      read -r med mn mx n < <(stats_line "${SAMPLES[$key]:-}")
      printf '%-7s %-5s %-6s | %9s %9s %9s %3s\n' "$engine" "$mode" "$pk" "$med" "$mn" "$mx" "$n"
    done
  done
done

echo
echo "------------------------- OURS / MTPLX ratio ---------------------------"
echo " (>1.0 = OURS faster; same checkpoint, same prompt/depth/T=0)"
printf '%-5s %-6s | %12s %12s %8s\n' mode prompt ours_median mtplx_median ratio
echo "------------------------------------------------------------------------"
for mode in "${MODES[@]}"; do
  for pk in "${PROMPT_KEYS[@]}"; do
    read -r o_med _ _ _ < <(stats_line "${SAMPLES[ours|$mode|$pk]:-}")
    read -r m_med _ _ _ < <(stats_line "${SAMPLES[mtplx|$mode|$pk]:-}")
    r="$(ratio "$o_med" "$m_med")"
    printf '%-5s %-6s | %12s %12s %8s\n' "$mode" "$pk" "$o_med" "$m_med" "$r"
  done
done

echo
echo "--------------- self-normalized MTP/AR speedup per engine --------------"
echo " (how much each engine's OWN MTP beats its OWN AR — isolates the"
echo "  speculative win from raw single-engine throughput)"
printf '%-7s %-6s | %9s %9s %10s\n' engine prompt mtp_median ar_median mtp/ar
echo "------------------------------------------------------------------------"
for engine in ours mtplx; do
  for pk in "${PROMPT_KEYS[@]}"; do
    read -r mtp_med _ _ _ < <(stats_line "${SAMPLES[$engine|mtp|$pk]:-}")
    read -r ar_med  _ _ _ < <(stats_line "${SAMPLES[$engine|ar|$pk]:-}")
    r="$(ratio "$mtp_med" "$ar_med")"
    printf '%-7s %-6s | %9s %9s %10s\n' "$engine" "$pk" "$mtp_med" "$ar_med" "$r"
  done
done

echo
echo "------------------- MTPLX MTP internals (medians) ----------------------"
printf '%-6s | %16s %14s %s\n' prompt verify_ms/call draft_time_s accepted_by_depth
echo "------------------------------------------------------------------------"
for pk in "${PROMPT_KEYS[@]}"; do
  read -r v_med _ _ _ < <(stats_line "${MTPLX_VERIFY[mtplx|mtp|$pk]:-}")
  read -r d_med _ _ _ < <(stats_line "${MTPLX_DRAFT[mtplx|mtp|$pk]:-}")
  printf '%-6s | %16s %14s %s\n' "$pk" "$v_med" "$d_med" "${MTPLX_ACCEPT[mtplx|mtp|$pk]:-NA}"
done

echo
echo "------------------- THERMAL-STABILITY witness (AR CV) ------------------"
echo " coefficient of variation of the AR tok/s samples per engine x prompt."
echo " High CV (>~8%) => this host is throttling; trust medians, not a"
echo " single-run delta. (AR is the clean witness: no acceptance variance.)"
printf '%-7s %-6s | %8s\n' engine prompt 'AR CV%'
echo "------------------------------------------------------------------------"
for engine in ours mtplx; do
  for pk in "${PROMPT_KEYS[@]}"; do
    cv="$(cv_pct "${SAMPLES[$engine|ar|$pk]:-}")"
    printf '%-7s %-6s | %8s\n' "$engine" "$pk" "$cv"
  done
done

# ----- optional CSV ----------------------------------------------------------
if [[ -n "$CSV" ]]; then
  {
    echo "engine,mode,prompt,samples,median,min,max,n,ar_cv_pct,mtplx_verify_ms,mtplx_draft_s,mtplx_accept"
    for engine in ours mtplx; do
      for mode in "${MODES[@]}"; do
        for pk in "${PROMPT_KEYS[@]}"; do
          key="$engine|$mode|$pk"
          read -r med mn mx n < <(stats_line "${SAMPLES[$key]:-}")
          cv="NA"; [[ "$mode" == "ar" ]] && cv="$(cv_pct "${SAMPLES[$key]:-}")"
          vm="NA"; dr="NA"; ac="NA"
          if [[ "$engine" == "mtplx" && "$mode" == "mtp" ]]; then
            read -r vm _ _ _ < <(stats_line "${MTPLX_VERIFY[$key]:-}")
            read -r dr _ _ _ < <(stats_line "${MTPLX_DRAFT[$key]:-}")
            ac="${MTPLX_ACCEPT[$key]:-NA}"
          fi
          printf '%s,%s,%s,"%s",%s,%s,%s,%s,%s,%s,%s,"%s"\n' \
            "$engine" "$mode" "$pk" "${SAMPLES[$key]:-}" "$med" "$mn" "$mx" "$n" "$cv" "$vm" "$dr" "$ac"
        done
      done
    done
  } > "$CSV"
  echo
  echo "CSV written: $CSV"
fi

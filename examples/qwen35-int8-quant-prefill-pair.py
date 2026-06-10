#!/usr/bin/env python3
"""
Qwen3.5 int8 W8A8 QUANT-prefill A/B orchestrator — thermally-fair paired verdict.

PROBE (Option C): measures the REALIZED prefill-TTFT delta of routing the
prefill matmuls of an ALREADY-QUANTIZED 8-bit-affine model through the
symmetric-int8 W8A8 kernel instead of the affine W8A16 `quantized_matmul`.
BOTH arms load the SAME quantized checkpoint; the only difference is the prefill
GEMM. Uses the established lfm2 paired-process methodology (median of per-pair
ratios + a control noise band) to clear the ~15% cross-run thermal drift on M5.

Launches the single-arm harness (examples/qwen35-int8-quant-prefill-ab.ts)
alternately as the TREATMENT (MLX_INT8_QUANT_PREFILL=1) and BASELINE
(MLX_INT8_QUANT_PREFILL unset) arms, BACK TO BACK (adjacent processes share a
thermal window). Each pair yields one unit-free ratio; the median of per-pair
ratios cancels the thermal drift.

NOTE on toggle polarity (this is a LOAD-TIME OPT-IN, default OFF):
  treatment = MLX_INT8_QUANT_PREFILL=1     (symmetric int8 W8A8 prefill ON)
  baseline  = MLX_INT8_QUANT_PREFILL unset (affine W8A16 path, unchanged default)
The int8 weight view is built at LOAD time (QuantizedLinear::new), so each arm
MUST be a fresh process with the env set/unset — which the per-arm fork here
guarantees. MLX_INT8_QUANT_PREFILL_DEBUG=1 is ALWAYS exported on the treatment
arm so the native side prints `[int8-quant-prefill] fired:` and the harness can
prove the W8A8 path actually executed (RESULT_JSON.int8FiredCount > 0). If the
treatment arm reports zero firings the run is FLAGGED — the model is probably
not 8-bit affine, or M never cleared MLX_INT8_QUANT_PREFILL_MIN_M (default 256).

A CONTROL set runs BOTH arms as BASELINE (identical code path) to measure the
ratio noise floor. An optimization counts as a REAL win only if its median
ratio improvement exceeds the control band AND the sign is consistent across
>=75% of pairs.

Ratio convention (>1.0 = treatment faster):
  prefill: medPrefillTps_treat / medPrefillTps_base  (higher tps better)
  ttft-ms: medTtftMs_base / medTtftMs_treat          (lower ms better)
  decode : medDecodeTps_treat / medDecodeTps_base    (~1.0 expected — decode
           always falls through to the affine path, M=1 < threshold)

Usage:
  PATH=/usr/bin:$PATH python3 examples/qwen35-int8-quant-prefill-pair.py \
    --model /Volumes/P4510/models/Qwen3.5-0.8B-UD-Q8_K_XL-mlx --mode ttft \
    --prompt-tokens 4096 --max-new 4 --reps 4 --warmup 1 \
    --pairs 6 --control-pairs 4

`--model` and `--prompt-tokens` are argv so the same harness sweeps multiple
models (0.8B / 4B / 9B 8-bit-affine) and the 512 / 4096 length variants.

Prints a human summary then one line `VERDICT_JSON:{...}`.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen35-int8-quant-prefill-ab.ts")
TOGGLE = "MLX_INT8_QUANT_PREFILL"
DEBUG_TOGGLE = "MLX_INT8_QUANT_PREFILL_DEBUG"
# Every int8-quant-prefill toggle the treatment arm may control. The baseline arm
# always UNSETS all of them so it runs the unchanged affine W8A16 default.
ALL_INT8_TOGGLES = (
    "MLX_INT8_QUANT_PREFILL",
    "MLX_INT8_QUANT_PREFILL_DEBUG",
    "MLX_INT8_QUANT_PREFILL_MIN_M",
)


def run_arm(args, treatment: bool) -> dict:
    """Run one harness arm. treatment=True -> sets MLX_INT8_QUANT_PREFILL=1 (+DEBUG,
    +optional MIN_M); baseline always unsets ALL int8 toggles (affine W8A16 path)."""
    env = dict(os.environ)
    # Baseline must never inherit a stray int8 toggle from the parent env.
    for k in ALL_INT8_TOGGLES:
        env.pop(k, None)
    if treatment:
        env[TOGGLE] = "1"
        # Always make firing visible so we can assert the W8A8 path actually ran.
        env[DEBUG_TOGGLE] = "1"
        if args.min_m is not None:
            env["MLX_INT8_QUANT_PREFILL_MIN_M"] = str(args.min_m)
    cmd = [
        "oxnode", HARNESS,
        "--model", args.model,
        "--mode", args.mode,
        "--prompt-tokens", str(args.prompt_tokens),
        "--max-new", str(args.max_new),
        "--reps", str(args.reps),
        "--warmup", str(args.warmup),
    ]
    # Large-model cold-start GPU watchdog timeout (pre-existing, int8-independent)
    # can transiently kill the FIRST command buffer of a freshly-loaded process.
    # It is non-deterministic and a fresh process retry clears it. Retry up to
    # args.arm_retries times; the timed numbers are unaffected (a retried arm is
    # a brand-new process producing its own measurements).
    last_stderr = ""
    for attempt in range(args.arm_retries + 1):
        p = subprocess.run(cmd, env=env, cwd=os.getcwd(), capture_output=True, text=True, timeout=args.timeout)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT_JSON:")), None)
        if line is not None:
            return json.loads(line[len("RESULT_JSON:"):])
        last_stderr = p.stderr[-2000:]
        sys.stderr.write(
            f"[arm treatment={treatment} attempt {attempt+1}/{args.arm_retries+1}] "
            f"no RESULT_JSON (retrying). stderr tail:\n{last_stderr}\n"
        )
    raise RuntimeError("harness produced no RESULT_JSON after retries")


def prefill_ratio(base: dict, treat: dict):
    b, o = base.get("medPrefillTps"), treat.get("medPrefillTps")
    if b and o and b > 0:
        return o / b
    return None


def decode_ratio(base: dict, treat: dict):
    b, o = base.get("medDecodeTps"), treat.get("medDecodeTps")
    if b and o and b > 0:
        return o / b
    return None


def ttft_ratio(base: dict, treat: dict):
    # lower ms better -> base/treat so >1.0 means treatment faster
    b, o = base.get("medTtftMs"), treat.get("medTtftMs")
    if b and o and o > 0:
        return b / o
    return None


def collect(args, pairs: int, control: bool, label: str):
    """Return (prefill_ratios, decode_ratios, ttft_ratios, treat_fired_total)."""
    pref_ratios, dec_ratios, ttft_ratios = [], [], []
    treat_fired_total = 0
    for i in range(pairs):
        # alternate order each pair to cancel order/thermal bias.
        # control -> "treat" arm is ALSO baseline (noise floor).
        treat_is_treatment = (not control)
        if i % 2 == 0:
            base = run_arm(args, treatment=False)
            treat = run_arm(args, treatment=treat_is_treatment)
        else:
            treat = run_arm(args, treatment=treat_is_treatment)
            base = run_arm(args, treatment=False)
        pr = prefill_ratio(base, treat)
        dr = decode_ratio(base, treat)
        tr = ttft_ratio(base, treat)
        if pr is not None:
            pref_ratios.append(pr)
        if dr is not None:
            dec_ratios.append(dr)
        if tr is not None:
            ttft_ratios.append(tr)
        fired = treat.get("int8FiredCount", 0) if treat_is_treatment else 0
        treat_fired_total += fired or 0
        bp, op = base.get("medPrefillTps"), treat.get("medPrefillTps")
        ptok = treat.get("promptTokensActual")
        print(
            f"  [{label} pair {i+1}/{pairs}] base_prefillTps={bp:.1f} treat_prefillTps={op:.1f}"
            f" prefillR={pr:.4f} decodeR={dr:.4f} ttftR={tr:.4f}"
            f" (promptTok={ptok} int8Fired={fired})",
            flush=True,
        )
    return pref_ratios, dec_ratios, ttft_ratios, treat_fired_total


def verdict_for(name, m_ratios, c_ratios):
    """Compute median-ratio verdict for one metric stream vs its control."""
    if not m_ratios:
        return None
    med = statistics.median(m_ratios)
    signal = med - 1.0
    control_devs = sorted(abs(r - 1.0) for r in c_ratios)
    # Robust band: median absolute deviation from 1.0, floored at 1.5%.
    control_band = max(statistics.median(control_devs) if control_devs else 0.0, 0.015)
    control_strict = max((abs(r - 1.0) for r in c_ratios), default=0.0)
    control_med = statistics.median(c_ratios) if c_ratios else 1.0
    same_side = sum(1 for r in m_ratios if (r > 1.0) == (med > 1.0))
    consistent = same_side >= (len(m_ratios) * 3 + 3) // 4  # >=75%
    real_win = (signal > control_band) and consistent and (med > 1.0)
    regression = (med < 1.0) and (abs(signal) > control_band) and consistent
    return {
        "metric": name,
        "median_ratio": round(med, 4),
        "pct_change": round(signal * 100, 2),
        "measure_ratios": [round(r, 4) for r in m_ratios],
        "control_ratios": [round(r, 4) for r in c_ratios],
        "control_band": round(control_band, 4),
        "control_strict": round(control_strict, 4),
        "control_median": round(control_med, 4),
        "same_side": same_side,
        "n_pairs": len(m_ratios),
        "consistent_sign": consistent,
        "real_win": bool(real_win),
        "regression": bool(regression),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="/Volumes/P4510/models/Qwen3.5-0.8B-UD-Q8_K_XL-mlx",
        help="Path to an 8-bit AFFINE quantized Qwen3.5 checkpoint (treatment is a "
        "no-op on 4-bit / mxfp* / nvfp4 / bf16).",
    )
    ap.add_argument("--mode", default="ttft", choices=["ttft", "decode"])
    ap.add_argument("--prompt-tokens", type=int, default=4096,
                    help="Approx prompt length. Use >=512 (and a 4096 variant) so "
                    "prefill M clears MLX_INT8_QUANT_PREFILL_MIN_M (default 256).")
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--reps", type=int, default=4, help="inner reps per process arm")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--control-pairs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument(
        "--min-m", type=int, default=None,
        help="Override MLX_INT8_QUANT_PREFILL_MIN_M on the treatment arm "
        "(default: unset -> native default 256).",
    )
    ap.add_argument(
        "--arm-retries", type=int, default=2,
        help="Retries for a single arm that produces no RESULT_JSON (transient "
        "cold-start GPU watchdog timeout on large models). Default 2.",
    )
    args = ap.parse_args()

    print(
        f"== MEASUREMENT (mode={args.mode}, treatment={TOGGLE}=1 vs unset, "
        f"prompt-tokens={args.prompt_tokens}, model={args.model}) ==",
        flush=True,
    )
    m_pref, m_dec, m_ttft, m_fired = collect(args, args.pairs, control=False, label="measure")
    print("== CONTROL (both arms baseline; ratio noise floor) ==", flush=True)
    c_pref, c_dec, c_ttft, _ = collect(args, args.control_pairs, control=True, label="control")

    prefill_v = verdict_for("prefill_tps", m_pref, c_pref)
    decode_v = verdict_for("decode_tps", m_dec, c_dec)
    ttft_v = verdict_for("ttft_ms", m_ttft, c_ttft)

    # decode "near 1.0" check: |median-1| within the decode control band.
    decode_near_one = None
    if decode_v is not None:
        decode_near_one = abs(decode_v["median_ratio"] - 1.0) <= decode_v["control_band"]

    # Sanity: the treatment arm must have actually fired the W8A8 path at least
    # once, else the A/B is meaningless (wrong dtype model / M below threshold).
    int8_fired = m_fired > 0

    verdict = {
        "mode": args.mode,
        "toggle": TOGGLE,
        "model": args.model,
        "prompt_tokens": args.prompt_tokens,
        "min_m": args.min_m,
        "int8_fired": bool(int8_fired),
        "int8_fired_total": m_fired,
        "prefill": prefill_v,
        "decode": decode_v,
        "ttft": ttft_v,
        "decode_near_one": bool(decode_near_one) if decode_near_one is not None else None,
        "gate_real_win": bool(int8_fired and prefill_v and prefill_v["real_win"]),
    }

    print("\n== SUMMARY ==")
    if not int8_fired:
        print(
            "  WARNING: treatment arm reported 0 int8-quant-prefill firings — the "
            "W8A8 path NEVER ran. Check the model is 8-bit affine and "
            f"prompt-tokens ({args.prompt_tokens}) clears MIN_M "
            f"({args.min_m or 256}). The ratios below are NOT a valid verdict."
        )
    else:
        print(f"  int8 W8A8 prefill FIRED (total firings across measure arms = {m_fired})")
    if prefill_v:
        print(
            f"  PREFILL-TPS median ratio = {prefill_v['median_ratio']:.4f} "
            f"({prefill_v['pct_change']:+.2f}%)  control band = ±{prefill_v['control_band']*100:.2f}%  "
            f"sign-consistent={prefill_v['consistent_sign']} ({prefill_v['same_side']}/{prefill_v['n_pairs']})  "
            f"REAL_WIN={prefill_v['real_win']}"
        )
    if ttft_v:
        print(
            f"  TTFT-MS  median ratio = {ttft_v['median_ratio']:.4f} "
            f"({ttft_v['pct_change']:+.2f}%)  control band = ±{ttft_v['control_band']*100:.2f}%  "
            f"REAL_WIN={ttft_v['real_win']}"
        )
    if decode_v:
        print(
            f"  DECODE-TPS median ratio = {decode_v['median_ratio']:.4f} "
            f"({decode_v['pct_change']:+.2f}%)  control band = ±{decode_v['control_band']*100:.2f}%  "
            f"near_1.0={decode_near_one}"
        )
    print(f"VERDICT_JSON:{json.dumps(verdict)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Qwen3.5 sym8 E2E A/B orchestrator — thermally-fair paired verdict (Phase 4).

Compares two DIFFERENT checkpoint directories with the SAME binary:
  treatment (sym8) : per-output-channel symmetric int8 checkpoint
                     (eager forward; int8 W8A8 GEMM prefill, fused int8 qmv decode)
  baseline  (base) : the 8-bit affine checkpoint of the same source model
                     (stock quantized_matmul; compiled C++ decode path)

Uses the established lfm2 paired-process methodology: arms run BACK TO BACK in
fresh processes (adjacent processes share a thermal window), alternating order
each pair; each pair yields one unit-free ratio; the median of per-pair ratios
cancels the ~15% M5 cross-run thermal drift. A CONTROL set runs BOTH arms on
the BASELINE checkpoint (identical code path) to measure the ratio noise floor.

MLX_SYM8_DEBUG is always UNSET in measured arms (it eprintlns per sym8 forward
and would tax the treatment arm only). Dispatch proof is structural (the sym8
checkpoint cannot take any other path) + a separate one-off debug run.

Ratio convention (>1.0 = sym8 faster):
  prefill: medPrefillTps_sym8 / medPrefillTps_base
  ttft-ms: medTtftMs_base / medTtftMs_sym8
  decode : medDecodeTps_sym8 / medDecodeTps_base

Usage:
  PATH=/usr/bin:$PATH python3 examples/qwen35-sym8-e2e-pair.py \
    --model-sym8 /tmp/qwen35-4b-sym8-mlx \
    --model-base /Volumes/P4510/models/Qwen3.5-4B-Q8affine-mlx \
    --mode ttft --prompt-tokens 1024 --max-new 4 --reps 3 --warmup 1 \
    --pairs 6 --control-pairs 4

Prints a human summary then one line `VERDICT_JSON:{...}`.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen35-sym8-e2e-ab.ts")


def run_arm(args, model_path: str, extra_env=None) -> dict:
    """Run one harness arm (a fresh process loading `model_path`).

    `extra_env`: list of "KEY=VAL" strings applied to this arm only —
    lets the SAME checkpoint run under different per-arm dispatch toggles
    (e.g. MLX_QWEN35_FORCE_EAGER=1 for the eager-vs-compiled control).
    """
    env = dict(os.environ)
    # Timed arms must never inherit the per-forward debug eprintln.
    env.pop("MLX_SYM8_DEBUG", None)
    for kv in extra_env or []:
        key, _, val = kv.partition("=")
        env[key] = val
    cmd = [
        "oxnode", HARNESS,
        "--model", model_path,
        "--mode", args.mode,
        "--prompt-tokens", str(args.prompt_tokens),
        "--max-new", str(args.max_new),
        "--reps", str(args.reps),
        "--warmup", str(args.warmup),
    ]
    last_stderr = ""
    for attempt in range(args.arm_retries + 1):
        p = subprocess.run(cmd, env=env, cwd=os.getcwd(), capture_output=True, text=True, timeout=args.timeout)
        line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT_JSON:")), None)
        if line is not None:
            return json.loads(line[len("RESULT_JSON:"):])
        last_stderr = p.stderr[-2000:]
        sys.stderr.write(
            f"[arm model={model_path} attempt {attempt+1}/{args.arm_retries+1}] "
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
    """Return (prefill_ratios, decode_ratios, ttft_ratios)."""
    pref_ratios, dec_ratios, ttft_ratios = [], [], []
    treat_model = args.model_base if control else args.model_sym8
    # CONTROL pairs run baseline-vs-baseline under the BASE env so the noise
    # floor measures the baseline code path, not the treatment toggle.
    treat_env = args.env_base if control else args.env_treat
    for i in range(pairs):
        # alternate order each pair to cancel order/thermal bias.
        if i % 2 == 0:
            base = run_arm(args, args.model_base, args.env_base)
            treat = run_arm(args, treat_model, treat_env)
        else:
            treat = run_arm(args, treat_model, treat_env)
            base = run_arm(args, args.model_base, args.env_base)
        pr = prefill_ratio(base, treat)
        dr = decode_ratio(base, treat)
        tr = ttft_ratio(base, treat)
        if pr is not None:
            pref_ratios.append(pr)
        if dr is not None:
            dec_ratios.append(dr)
        if tr is not None:
            ttft_ratios.append(tr)
        bp, op = base.get("medPrefillTps"), treat.get("medPrefillTps")
        bd, od = base.get("medDecodeTps"), treat.get("medDecodeTps")
        ptok = treat.get("promptTokensActual")
        print(
            f"  [{label} pair {i+1}/{pairs}] base_prefillTps={bp:.1f} treat_prefillTps={op:.1f}"
            f" base_decodeTps={bd:.1f} treat_decodeTps={od:.1f}"
            f" prefillR={pr:.4f} decodeR={dr:.4f} ttftR={tr:.4f} (promptTok={ptok})",
            flush=True,
        )
    return pref_ratios, dec_ratios, ttft_ratios


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
    ap.add_argument("--model-sym8", default="/tmp/qwen35-4b-sym8-mlx",
                    help="sym8 checkpoint dir (config quantization mode == 'sym8')")
    ap.add_argument("--model-base", default="/Volumes/P4510/models/Qwen3.5-4B-Q8affine-mlx",
                    help="8-bit affine baseline checkpoint dir of the same source model")
    ap.add_argument("--mode", default="ttft", choices=["ttft", "decode"])
    ap.add_argument("--prompt-tokens", type=int, default=1024)
    ap.add_argument("--max-new", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3, help="inner reps per process arm")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--control-pairs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--arm-retries", type=int, default=2,
                    help="Retries for an arm with no RESULT_JSON (transient cold-start "
                    "GPU watchdog on large models). Default 2.")
    ap.add_argument("--env-treat", action="append", default=[], metavar="KEY=VAL",
                    help="Env var applied only to the TREATMENT arm (repeatable), e.g. "
                    "MLX_QWEN35_FORCE_EAGER=1 for the eager-vs-compiled control.")
    ap.add_argument("--env-base", action="append", default=[], metavar="KEY=VAL",
                    help="Env var applied only to the BASELINE arm (repeatable). "
                    "Control pairs use this env on both arms.")
    args = ap.parse_args()

    print(
        f"== MEASUREMENT (mode={args.mode}, sym8={args.model_sym8} vs base={args.model_base}, "
        f"prompt-tokens={args.prompt_tokens}, max-new={args.max_new}, "
        f"env-treat={args.env_treat}, env-base={args.env_base}) ==",
        flush=True,
    )
    m_pref, m_dec, m_ttft = collect(args, args.pairs, control=False, label="measure")
    print("== CONTROL (both arms = baseline checkpoint; ratio noise floor) ==", flush=True)
    c_pref, c_dec, c_ttft = collect(args, args.control_pairs, control=True, label="control")

    prefill_v = verdict_for("prefill_tps", m_pref, c_pref)
    decode_v = verdict_for("decode_tps", m_dec, c_dec)
    ttft_v = verdict_for("ttft_ms", m_ttft, c_ttft)

    verdict = {
        "mode": args.mode,
        "model_sym8": args.model_sym8,
        "model_base": args.model_base,
        "prompt_tokens": args.prompt_tokens,
        "max_new": args.max_new,
        "env_treat": args.env_treat,
        "env_base": args.env_base,
        "prefill": prefill_v,
        "decode": decode_v,
        "ttft": ttft_v,
    }

    print("\n== SUMMARY ==")
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
            f"regression={decode_v['regression']}"
        )
    print(f"VERDICT_JSON:{json.dumps(verdict)}")


if __name__ == "__main__":
    main()

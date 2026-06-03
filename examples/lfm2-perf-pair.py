#!/usr/bin/env python3
"""
lfm2 perf A/B orchestrator — thermally-fair paired measurement + verdict.

Launches the single-arm harness (examples/lfm2-perf-ab.ts) alternately with
and without a MLX_LFM2_DISABLE_<OPT> toggle, BACK TO BACK (adjacent processes
share a thermal window). Each pair yields one unit-free ratio; the median of
per-pair ratios cancels the ~15% cross-run thermal drift seen on M5 Max.

A CONTROL set runs BOTH arms with the toggle SET (identical code path) to
measure the ratio noise floor. An optimization counts as a REAL win only if
its median ratio improvement exceeds the worst control deviation AND the sign
is consistent across pairs.

Ratio convention (>1.0 = optimized faster):
  decode -> medDecodeTps_opt / medDecodeTps_base   (higher tok/s better)
  ttft   -> medTtftMs_base   / medTtftMs_opt        (lower ms better)
           plus prefill -> medPrefillTps_opt / medPrefillTps_base

Usage:
  python3 examples/lfm2-perf-pair.py \
    --model lfm2.5-1.2b-thinking-mlx --mode ttft \
    --toggle MLX_LFM2_DISABLE_LAST_TOKEN_SLICE \
    --prompt-tokens 1500 --max-new 4 --reps 4 --warmup 1 \
    --pairs 5 --control-pairs 3

Prints a human summary then one line `VERDICT_JSON:{...}`.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm2-perf-ab.ts")


def run_arm(args, toggle_set: bool) -> dict:
    env = dict(os.environ)
    if args.toggle:
        if toggle_set:
            env[args.toggle] = "1"
        else:
            env.pop(args.toggle, None)
    cmd = [
        "oxnode", HARNESS,
        "--model", args.model,
        "--mode", args.mode,
        "--prompt-tokens", str(args.prompt_tokens),
        "--max-new", str(args.max_new),
        "--reps", str(args.reps),
        "--warmup", str(args.warmup),
    ]
    p = subprocess.run(cmd, env=env, cwd=os.getcwd(), capture_output=True, text=True, timeout=args.timeout)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT_JSON:")), None)
    if line is None:
        sys.stderr.write(f"[arm toggle={toggle_set}] no RESULT_JSON. stderr tail:\n{p.stderr[-1500:]}\n")
        raise RuntimeError("harness produced no RESULT_JSON")
    return json.loads(line[len("RESULT_JSON:"):])


def metric(args, r: dict) -> float:
    return r["medTtftMs"] if args.mode == "ttft" else r["medDecodeTps"]


def ratio(args, base: dict, opt: dict) -> float:
    if args.mode == "ttft":
        return metric(args, base) / metric(args, opt)  # lower ms better -> invert
    return metric(args, opt) / metric(args, base)        # higher tok/s better


def prefill_ratio(base: dict, opt: dict):
    b, o = base.get("medPrefillTps"), opt.get("medPrefillTps")
    if b and o and b > 0:
        return o / b
    return None


def collect(args, pairs: int, control: bool, label: str):
    ratios, pref_ratios = [], []
    for i in range(pairs):
        # alternate order each pair to cancel order/thermal bias
        if i % 2 == 0:
            base = run_arm(args, True)
            opt = run_arm(args, True if control else False)
        else:
            opt = run_arm(args, True if control else False)
            base = run_arm(args, True)
        rr = ratio(args, base, opt)
        ratios.append(rr)
        pr = prefill_ratio(base, opt)
        if pr is not None:
            pref_ratios.append(pr)
        bm, om = metric(args, base), metric(args, opt)
        extra = f" prefillR={pr:.3f}" if pr is not None else ""
        print(f"  [{label} pair {i+1}/{pairs}] base={bm:.2f} opt={om:.2f} ratio={rr:.4f}{extra}", flush=True)
    return ratios, pref_ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lfm2.5-1.2b-thinking-mlx")
    ap.add_argument("--mode", default="decode", choices=["ttft", "decode"])
    ap.add_argument("--toggle", required=True, help="MLX_LFM2_DISABLE_* env var")
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--reps", type=int, default=4, help="inner reps per process arm")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--pairs", type=int, default=5)
    ap.add_argument("--control-pairs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    print(f"== MEASUREMENT ({args.mode}, toggle={args.toggle}, model={args.model}) ==", flush=True)
    m_ratios, m_pref = collect(args, args.pairs, control=False, label="measure")
    print(f"== CONTROL (both arms baseline; ratio noise floor) ==", flush=True)
    c_ratios, _ = collect(args, args.control_pairs, control=True, label="control")

    med = statistics.median(m_ratios)
    signal = med - 1.0
    # Robust noise floor: median absolute deviation of control ratios from 1.0
    # (the max-dev is too outlier-sensitive at small N). Floor at 1.5% so we
    # never claim a win smaller than the harness's own resolution.
    control_devs = sorted(abs(r - 1.0) for r in c_ratios)
    control_band = max(statistics.median(control_devs) if control_devs else 0.0, 0.015)
    control_strict = max((abs(r - 1.0) for r in c_ratios), default=0.0)
    control_med = statistics.median(c_ratios) if c_ratios else 1.0
    same_side = sum(1 for r in m_ratios if (r > 1.0) == (med > 1.0))
    consistent = same_side >= (len(m_ratios) * 3 + 3) // 4  # >=75%
    real_win = (signal > control_band) and consistent and (med > 1.0)
    regression = (med < 1.0) and (abs(signal) > control_band) and consistent

    pref_med = statistics.median(m_pref) if m_pref else None

    verdict = {
        "mode": args.mode,
        "toggle": args.toggle,
        "model": args.model,
        "median_ratio": round(med, 4),
        "pct_change": round(signal * 100, 2),
        "measure_ratios": [round(r, 4) for r in m_ratios],
        "control_ratios": [round(r, 4) for r in c_ratios],
        "control_band": round(control_band, 4),
        "control_strict": round(control_strict, 4),
        "control_median": round(control_med, 4),
        "prefill_median_ratio": round(pref_med, 4) if pref_med is not None else None,
        "consistent_sign": consistent,
        "real_win": bool(real_win),
        "regression": bool(regression),
    }
    print("\n== SUMMARY ==")
    print(f"  median ratio = {med:.4f}  ({signal*100:+.2f}%)   control band = ±{control_band*100:.2f}%")
    if pref_med is not None:
        print(f"  prefill median ratio = {pref_med:.4f}  ({(pref_med-1)*100:+.2f}%)")
    print(f"  REAL WIN = {real_win}   (signal {signal*100:+.2f}% vs noise ±{control_band*100:.2f}%, consistent={consistent})")
    print(f"VERDICT_JSON:{json.dumps(verdict)}")


if __name__ == "__main__":
    main()

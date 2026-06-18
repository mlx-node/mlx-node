#!/usr/bin/env python3
"""
env-pair — generic same-binary thermally-fair A/B over engine-perf-probe.ts,
where the two arms differ only by ENVIRONMENT VARIABLES (not a binary swap).

ratio = cand_tps / base_tps.  signal = ratio - 1.
  ratio > 1  => cand FASTER than base.
Interleaving arm order per pair + median of per-pair ratios cancels the ~15%
cross-run M5 drift. CONTROL set runs BOTH arms = base (ratio noise floor).
With --emit-text, reports whether base and cand are T=0 byte-identical.

Env arms are JSON objects; a value of null UNSETS that key for the arm.

Usage:
  python3 examples/env-pair.py \
    --model-path /Volumes/P4510/models/lfm2.5-1.2b-thinking-mlx \
    --base-env  '{"MLX_LFM2_FORCE_EAGER":"1","MLX_LFM2_NATIVE_KV_WRITE":"0","MLX_LFM2_GRAPH_DECODE_GATHER":"0"}' \
    --cand-env  '{"MLX_LFM2_FORCE_EAGER":"1"}' \
    --label-base eager-sync --label-cand eager-graph \
    --mode decode --prompt-tokens 256 --max-new 256 --reps 4 --warmup 1 \
    --pairs 6 --control-pairs 3 --emit-text
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "engine-perf-probe.ts")


def make_env(overrides: dict) -> dict:
    env = dict(os.environ)
    for k, v in overrides.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = str(v)
    return env


def run_probe(args, overrides: dict, label: str) -> dict:
    cmd = [
        "oxnode", PROBE,
        "--model-path", args.model_path,
        "--mode", args.mode,
        "--prompt-tokens", str(args.prompt_tokens),
        "--max-new", str(args.max_new),
        "--reps", str(args.reps),
        "--warmup", str(args.warmup),
        "--label", label,
    ]
    if args.emit_text:
        cmd.append("--emit-text")
    p = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True,
                       timeout=args.timeout, env=make_env(overrides))
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT_JSON:")), None)
    if line is None:
        sys.stderr.write(f"[probe {label}] no RESULT_JSON. exit={p.returncode}\n"
                         f"stderr tail:\n{p.stderr[-2000:]}\n")
        raise RuntimeError("probe produced no RESULT_JSON")
    return json.loads(line[len("RESULT_JSON:"):])


def metric(args, r: dict) -> float:
    return r["medTtftMs"] if args.mode == "ttft" else r["medDecodeTps"]


def ratio(args, base: dict, cand: dict) -> float:
    if args.mode == "ttft":
        return metric(args, base) / metric(args, cand)  # lower ms better -> invert
    return metric(args, cand) / metric(args, base)        # higher tok/s better


def collect(args, pairs: int, control: bool, label: str):
    base_env = json.loads(args.base_env)
    cand_env = base_env if control else json.loads(args.cand_env)
    ratios, hashes = [], []
    for i in range(pairs):
        if i % 2 == 0:
            base = run_probe(args, base_env, args.label_base)
            cand = run_probe(args, cand_env, args.label_base if control else args.label_cand)
        else:
            cand = run_probe(args, cand_env, args.label_base if control else args.label_cand)
            base = run_probe(args, base_env, args.label_base)
        rr = ratio(args, base, cand)
        if not math.isfinite(rr):
            print(f"  [{label} {i+1}/{pairs}] non-finite, skipped", flush=True)
            continue
        ratios.append(rr)
        if args.emit_text:
            hashes.append((base.get("rawTextHash"), cand.get("rawTextHash")))
        print(f"  [{label} {i+1}/{pairs}] base={metric(args, base):.3f} "
              f"cand={metric(args, cand):.3f} ratio={rr:.4f}", flush=True)
    return ratios, hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--base-env", required=True, help="JSON object; null value unsets a key")
    ap.add_argument("--cand-env", required=True, help="JSON object; null value unsets a key")
    ap.add_argument("--label-base", default="base")
    ap.add_argument("--label-cand", default="cand")
    ap.add_argument("--mode", default="decode", choices=["ttft", "decode"])
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--control-pairs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--emit-text", action="store_true")
    args = ap.parse_args()

    print(f"== env-pair ({args.mode}, {os.path.basename(args.model_path)}) "
          f"{args.label_cand} vs {args.label_base} ==", flush=True)
    m_ratios, m_hashes = collect(args, args.pairs, False, "measure")
    print("== CONTROL (both arms = base; ratio noise floor) ==", flush=True)
    c_ratios, _ = collect(args, args.control_pairs, True, "control")

    if not m_ratios:
        sys.exit("no finite measurement ratios")

    med = statistics.median(m_ratios)
    signal = med - 1.0
    control_devs = sorted(abs(r - 1.0) for r in c_ratios)
    control_band = max(statistics.median(control_devs) if control_devs else 0.0, 0.015)
    same_side = sum(1 for r in m_ratios if (r > 1.0) == (med > 1.0))
    consistent = same_side >= (len(m_ratios) * 3 + 3) // 4
    hash_match = all(b == c and b is not None for b, c in m_hashes) if m_hashes else None

    verdict = {
        "model": os.path.basename(args.model_path),
        "cand": args.label_cand,
        "base": args.label_base,
        "cand_over_base_ratio": round(med, 4),
        "pct_change": round(signal * 100, 2),
        "measure_ratios": [round(r, 4) for r in m_ratios],
        "control_ratios": [round(r, 4) for r in c_ratios],
        "control_band": round(control_band, 4),
        "consistent_sign": consistent,
        "t0_byte_identical": hash_match,
    }
    print("\n== SUMMARY ==")
    print(f"  {args.label_cand}/{args.label_base} = {med:.4f}  ({signal*100:+.2f}%)   "
          f"control band = +-{control_band*100:.2f}%")
    print(f"  t0_byte_identical = {hash_match}")
    print(f"VERDICT_JSON:{json.dumps(verdict)}")


if __name__ == "__main__":
    main()

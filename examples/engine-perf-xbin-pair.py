#!/usr/bin/env python3
"""
engine-perf-xbin-pair — cross-binary perf-PARITY oracle (branch vs origin/main).

Thermally-fair paired A/B where the ARM is a NATIVE BINARY SWAP, not an env
toggle: this branch carries no MLX_ENGINE_LEGACY path, so "main" is the real
origin/main CI artifact (.node + mlx.metallib + paged_attn.metallib) and
"branch" is the current packages/core build.

Each arm = swap the trio into packages/core, launch engine-perf-probe.ts in a
fresh process (loads whatever .node is in place), parse RESULT_JSON. Adjacent
branch/baseline runs share a thermal window; the median of per-pair ratios
cancels the ~15% cross-run drift on M5 Max. A CONTROL set runs BOTH arms on the
SAME (baseline) binary to measure the ratio noise floor.

PARITY semantics (the hard constraint is "perf MUST == origin/main"):
  ratio > 1  = branch FASTER than main.  signal = median_ratio - 1.
  regression  = branch slower than main beyond the control band  -> FAIL.
  parity_pass = not regression (within noise, or faster).

Also reports rawText hash agreement (branch vs main) as an INFORMATIONAL
byte-identity signal (only meaningful when the branch contains all of main's
output-affecting commits; the hard correctness gate stays the on-branch
t0-smoke pre/post digest).

Usage:
  python3 examples/engine-perf-xbin-pair.py \
    --model-path /Volumes/P4510/models/qwen3.5-0.8b-mlx-bf16 \
    --mode decode --prompt-tokens 512 --max-new 256 --reps 4 --warmup 1 \
    --pairs 6 --control-pairs 3 [--emit-text]

Prints a human summary then one line VERDICT_JSON:{...}.
"""
import argparse
import atexit
import json
import math
import os
import shutil
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "engine-perf-probe.ts")
TRIO = ("mlx-core.darwin-arm64.node", "mlx.metallib", "paged_attn.metallib")
DEFAULT_BASELINE = "/Users/brooklyn/workspace/github/mlx-node-perf-baseline/main-fba240b8"
DEFAULT_BRANCH = "/Users/brooklyn/workspace/github/mlx-node-perf-baseline/branch-live"


def swap_in(src_dir: str, pkg_core: str) -> None:
    for f in TRIO:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(pkg_core, f))


def run_probe(args) -> dict:
    cmd = [
        "oxnode", PROBE,
        "--model-path", args.model_path,
        "--mode", args.mode,
        "--prompt-tokens", str(args.prompt_tokens),
        "--max-new", str(args.max_new),
        "--reps", str(args.reps),
        "--warmup", str(args.warmup),
    ]
    if args.emit_text:
        cmd.append("--emit-text")
    p = subprocess.run(cmd, cwd=os.getcwd(), capture_output=True, text=True, timeout=args.timeout)
    line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT_JSON:")), None)
    if line is None:
        sys.stderr.write(f"[probe] no RESULT_JSON. stderr tail:\n{p.stderr[-1800:]}\n")
        raise RuntimeError("probe produced no RESULT_JSON")
    return json.loads(line[len("RESULT_JSON:"):])


def run_arm(args, use_baseline: bool, state: dict) -> dict:
    src = args.baseline_dir if use_baseline else args.branch_dir
    if state.get("cur") != src:
        swap_in(src, args.pkg_core)
        state["cur"] = src
    return run_probe(args)


def metric(args, r: dict) -> float:
    return r["medTtftMs"] if args.mode == "ttft" else r["medDecodeTps"]


def ratio(args, base: dict, cand: dict) -> float:
    # >1 means cand is faster than base
    if args.mode == "ttft":
        return metric(args, base) / metric(args, cand)  # lower ms better -> invert
    return metric(args, cand) / metric(args, base)        # higher tok/s better


def collect(args, pairs: int, control: bool, label: str, state: dict):
    ratios, hashes = [], []
    for i in range(pairs):
        # alternate which arm runs first each pair to cancel order/thermal bias
        if i % 2 == 0:
            base = run_arm(args, True, state)
            cand = run_arm(args, True if control else False, state)
        else:
            cand = run_arm(args, True if control else False, state)
            base = run_arm(args, True, state)
        rr = ratio(args, base, cand)
        if not math.isfinite(rr):
            print(f"  [{label} {i+1}/{pairs}] non-finite ratio, skipped", flush=True)
            continue
        ratios.append(rr)
        if args.emit_text and not control:
            hashes.append((base.get("rawTextHash"), cand.get("rawTextHash")))
        print(
            f"  [{label} {i+1}/{pairs}] base={metric(args, base):.3f} "
            f"cand={metric(args, cand):.3f} ratio={rr:.4f}",
            flush=True,
        )
    return ratios, hashes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--mode", default="decode", choices=["ttft", "decode"])
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--control-pairs", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--emit-text", action="store_true")
    ap.add_argument("--baseline-dir", default=DEFAULT_BASELINE)
    ap.add_argument("--branch-dir", default=DEFAULT_BRANCH)
    ap.add_argument("--pkg-core", default=os.path.join(os.getcwd(), "packages", "core"))
    args = ap.parse_args()

    for f in TRIO:
        bp = os.path.join(args.baseline_dir, f)
        if not os.path.isfile(bp):
            sys.exit(f"missing baseline file: {bp}")

    # Snapshot the CURRENT (branch) build so the branch arm reflects THIS build,
    # and restore it on exit (pkg-core always ends on the branch binary).
    os.makedirs(args.branch_dir, exist_ok=True)
    swap_in(args.pkg_core, args.branch_dir)  # pkg-core -> branch snapshot
    state = {"cur": None}
    atexit.register(lambda: (swap_in(args.branch_dir, args.pkg_core), print("== restored branch trio into packages/core ==", flush=True)))

    print(f"== PARITY ({args.mode}, {os.path.basename(args.model_path)}) branch vs main ==", flush=True)
    m_ratios, m_hashes = collect(args, args.pairs, False, "measure", state)
    print("== CONTROL (both arms = main; ratio noise floor) ==", flush=True)
    c_ratios, _ = collect(args, args.control_pairs, True, "control", state)

    if not m_ratios:
        sys.exit("no finite measurement ratios")

    med = statistics.median(m_ratios)
    signal = med - 1.0
    control_devs = sorted(abs(r - 1.0) for r in c_ratios)
    control_band = max(statistics.median(control_devs) if control_devs else 0.0, 0.015)
    same_side = sum(1 for r in m_ratios if (r > 1.0) == (med > 1.0))
    consistent = same_side >= (len(m_ratios) * 3 + 3) // 4  # >=75%
    regression = (signal < -control_band) and consistent
    parity_pass = not regression
    faster = (signal > control_band) and consistent

    hash_match = None
    if m_hashes:
        hash_match = all(b == o and b is not None for b, o in m_hashes)

    verdict = {
        "model": os.path.basename(args.model_path),
        "mode": args.mode,
        "median_ratio": round(med, 4),
        "pct_change": round(signal * 100, 2),
        "measure_ratios": [round(r, 4) for r in m_ratios],
        "control_ratios": [round(r, 4) for r in c_ratios],
        "control_band": round(control_band, 4),
        "consistent_sign": consistent,
        "regression": bool(regression),
        "parity_pass": bool(parity_pass),
        "faster_than_main": bool(faster),
        "rawtext_hash_match": hash_match,
    }
    print("\n== SUMMARY ==")
    print(f"  median ratio = {med:.4f}  ({signal * 100:+.2f}%)   control band = ±{control_band * 100:.2f}%")
    print(f"  PARITY PASS = {parity_pass}   regression={regression}  faster={faster}  hash_match={hash_match}")
    print(f"VERDICT_JSON:{json.dumps(verdict)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Driver for the sym8 compiled-vs-eager byte-parity gate.

Runs examples/qwen35-sym8-compiled-parity.ts twice per model (fresh process
per arm — MLX_QWEN35_FORCE_EAGER is read once per process), then compares the
generated text byte-for-byte per prompt.

Usage:
  PATH=/usr/bin:$PATH python3 examples/qwen35-sym8-compiled-parity.py \
    --models /tmp/qwen35-0.8b-sym8-mlx /tmp/qwen35-4b-sym8-mlx --max-new 120
"""
import argparse
import json
import os
import subprocess
import sys

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen35-sym8-compiled-parity.ts")


def run_arm(model: str, max_new: int, force_eager: bool, timeout: int) -> dict:
    env = dict(os.environ)
    env.pop("MLX_SYM8_DEBUG", None)
    if force_eager:
        env["MLX_QWEN35_FORCE_EAGER"] = "1"
    else:
        env.pop("MLX_QWEN35_FORCE_EAGER", None)
    cmd = ["oxnode", HARNESS, "--model", model, "--max-new", str(max_new)]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    line = next((l for l in p.stdout.splitlines() if l.startswith("PARITY_JSON:")), None)
    if line is None:
        sys.stderr.write(p.stderr[-3000:] + "\n")
        raise RuntimeError(f"no PARITY_JSON from arm force_eager={force_eager} model={model}")
    return json.loads(line[len("PARITY_JSON:"):])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--max-new", type=int, default=120)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    all_ok = True
    verdict = {}
    for model in args.models:
        compiled = run_arm(model, args.max_new, force_eager=False, timeout=args.timeout)
        eager = run_arm(model, args.max_new, force_eager=True, timeout=args.timeout)
        per_prompt = []
        for c, e in zip(compiled["results"], eager["results"]):
            same = c["text"] == e["text"]
            all_ok = all_ok and same
            per_prompt.append({
                "prompt": c["prompt"][:48],
                "byte_identical": same,
                "compiled_sha": c["sha256"][:16],
                "eager_sha": e["sha256"][:16],
                "compiled_len": len(c["text"]),
                "eager_len": len(e["text"]),
            })
            if not same:
                # First divergence point for diagnosis.
                ct, et = c["text"], e["text"]
                i = next((k for k in range(min(len(ct), len(et))) if ct[k] != et[k]), min(len(ct), len(et)))
                per_prompt[-1]["first_divergence_at"] = i
                per_prompt[-1]["compiled_tail"] = ct[max(0, i - 40):i + 80]
                per_prompt[-1]["eager_tail"] = et[max(0, i - 40):i + 80]
        verdict[model] = per_prompt
        for row in per_prompt:
            print(f"{model} :: {'OK ' if row['byte_identical'] else 'DIFF'} :: {row['prompt']}")
    print("VERDICT_JSON:" + json.dumps({"ok": all_ok, "models": verdict}))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

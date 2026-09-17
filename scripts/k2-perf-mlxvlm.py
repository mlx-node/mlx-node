#!/usr/bin/env python3
"""K2-Horizon perf reference via vendored mlx-vlm (mxfp8 converted checkpoint).

Prints one JSON line per run: prompt tokens, prefill TPS, decode TPS, peak RSS.
Run: .venv/bin/python scripts/k2-perf-mlxvlm.py
"""

import json
import os
import sys

MODEL = os.environ.get("K2_MXFP8", "/tmp/k2-out/k2-horizon-7b-mxfp8")
MAX_TOKENS = int(os.environ.get("K2_MAX_TOKENS", "64"))

# Mirror the node-side bench prompts: one short, one ~500-token instruction.
SHORT = "What is 17 * 23? Give the final number."
LONG = (
    "You are a careful math tutor. Solve step by step, then give the final answer. "
    "Here is the problem context: " + " ".join(f"term{i}={i * 7 % 13}" for i in range(120))
    + " Compute the sum of all terms modulo 97, then multiply by 11."
)


def run(model, processor, prompt, label):
    from mlx_vlm import generate

    messages = [{"role": "user", "content": prompt}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    result = generate(
        model,
        processor,
        text,
        max_tokens=MAX_TOKENS,
        verbose=False,
    )
    print(
        json.dumps(
            {
                "label": label,
                "promptTokens": getattr(result, "prompt_tokens", None),
                "promptTps": round(getattr(result, "prompt_tps", 0.0), 2),
                "genTokens": getattr(result, "generation_tokens", None),
                "genTps": round(getattr(result, "generation_tps", 0.0), 2),
                "peakGB": round(getattr(result, "peak_memory", 0.0), 2),
                "text": (result.text or "")[:60],
            }
        ),
        flush=True,
    )


def main():
    from mlx_vlm import load

    print(json.dumps({"loading": MODEL}), flush=True)
    model, processor = load(MODEL)
    print(json.dumps({"loaded": True}), flush=True)

    # Warmup (JIT/compile amortization is part of first call in mlx too, so do
    # one throwaway short run before measuring).
    run(model, processor, "hi", "warmup")
    run(model, processor, SHORT, "short")
    run(model, processor, LONG, "long")
    run(model, processor, SHORT, "short2")


if __name__ == "__main__":
    sys.exit(main())

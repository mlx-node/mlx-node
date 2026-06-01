#!/usr/bin/env python3
"""Capture an INDEPENDENT golden greedy decode from mlx-lm (Apple's reference
MLX language-model library) for the LFM2 compiled-parity harness.

This script is the SOLE source of the frozen golden token-id / golden-text
constants embedded in `crates/mlx-core/tests/lfm2_compiled_e2e.rs`. It is a
genuinely independent oracle: mlx-lm is a SEPARATE codebase from this repo's
Rust/C++ compiled forward path, so a bug shared by every Rust path (e.g. wrong
rope base, wrong RMSNorm eps, wrong MoE routing order) shows up as a token-id
divergence against this golden — which the existing eager-flat / native-paged
references (both the SAME Rust+C++ math) cannot catch.

Greedy on BOTH sides:
  * mlx-lm: `generate_step(mx.array(ids, uint32), model)` with the DEFAULT
    sampler, which `generate.py:386` defines as `lambda x: mx.argmax(x, -1)`
    (pure greedy / temperature 0).
  * Rust compiled path (the side under test): `greedy_chat_config` =
    temperature 0 → per-step argmax, all penalties off.

Prompt parity is guaranteed by the SHARED chat template: this script renders the
prompt with `tokenizer.apply_chat_template(...)` from the SAME checkpoint the
Rust test loads, and prints the resulting prompt token-id list so the Rust test
can assert (cheap canary) that its re-templated prompt matches in length.

Run (independent of the Rust toolchain; resolves mlx-lm on the fly):

    uv run --python 3.12 --with mlx-lm python scripts/capture_lfm2_golden.py \
        .cache/models/lfm2.5-1.2b-thinking-mlx

Pin: captured against mlx-lm 0.31.3 (printed below). Refresh deliberately when
the checkpoint's chat_template or mlx-lm changes; paste the emitted
`GOLDEN_*` block into the Rust test.
"""

import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step

# The user string the Rust suite already uses (lfm2_compiled_e2e.rs). Re-templated
# identically on both sides via the shared chat template.
GOLDEN_PROMPT = "What is the capital of France? Answer in one short sentence."

# How many greedy tokens to capture. The harness asserts >= 64 generated tokens
# of agreement, so capture at least that many.
N_NEW = 80


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: capture_lfm2_golden.py <checkpoint-dir>", file=sys.stderr)
        return 2
    ckpt = sys.argv[1]

    try:
        import mlx_lm

        version = getattr(mlx_lm, "__version__", "unknown")
    except Exception:
        version = "unknown"

    model, tokenizer = load(ckpt)

    messages = [{"role": "user", "content": GOLDEN_PROMPT}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

    prompt = mx.array(prompt_ids, dtype=mx.uint32)

    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    # LFM2 thinking checkpoints emit <|im_end|> as the chat EOS.
    try:
        eos_ids.update(int(t) for t in tokenizer.eos_token_ids)
    except Exception:
        pass

    gen_ids = []
    for i, (tok, _logprobs) in enumerate(
        generate_step(prompt, model, max_tokens=N_NEW)
    ):
        tid = int(tok)
        gen_ids.append(tid)
        if tid in eos_ids:
            break
        if len(gen_ids) >= N_NEW:
            break

    # Decode the FULL generated span (including any <think> ... </think>) so the
    # golden text matches the Rust side's `raw_text` (verbatim, NOT the
    # think-stripped `.text`).
    golden_text = tokenizer.decode(gen_ids)

    print("=" * 72)
    print(f"// mlx-lm version: {version}")
    print(f"// checkpoint    : {ckpt}")
    print(f"// prompt        : {GOLDEN_PROMPT!r}")
    print(f"// prompt_ids ({len(prompt_ids)}): {prompt_ids}")
    print(f"// generated_ids ({len(gen_ids)}): {gen_ids}")
    print(f"// distinct gen ids: {len(set(gen_ids))}")
    print(f"// eos ids        : {sorted(eos_ids)}")
    print("=" * 72)
    print("// ---- paste GOLDEN_TEXT_* below (verbatim raw decode) ----")
    print(repr(golden_text))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

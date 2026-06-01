#!/usr/bin/env python3
"""Diagnostic probe: dump mlx-lm's per-step greedy token + top-k logit landscape
for the LFM2 golden prompt, so the "benign bf16 near-tie" claim at the compiled
MoE divergence step is REPRODUCIBLE from a committed artifact (not a comment).

This is NOT the frozen-golden source (that is capture_lfm2_golden.py). This is an
investigation tool: it prints, for every generated step, the argmax token and the
top-K (token_id, repr, logit, gap-to-next) so we can read the actual logit
geometry at the step where our compiled path picks a different token than mlx-lm.

mlx-lm's `generate_step` yields `(tok, logprobs)` where `logprobs` is the
log-softmax of that step's logits over the vocab. log-softmax is monotone and
shift-invariant in DIFFERENCES, so `logprob_a - logprob_b == logit_a - logit_b`:
the gaps between candidates are the true logit gaps, which is exactly what the
bf16-ULP argument needs.

Run:
    uv run --python 3.12 --with mlx-lm python scripts/probe_lfm2_divergence.py \
        .cache/models/lfm2.5-8b-a1b
"""

import sys

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step

GOLDEN_PROMPT = "What is the capital of France? Answer in one short sentence."
N_NEW = 80
TOPK = 8


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: probe_lfm2_divergence.py <checkpoint-dir>", file=sys.stderr)
        return 2
    ckpt = sys.argv[1]

    try:
        import mlx_lm

        version = getattr(mlx_lm, "__version__", "unknown")
    except Exception:
        version = "unknown"

    model, tokenizer = load(ckpt)

    messages = [{"role": "user", "content": GOLDEN_PROMPT}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt = mx.array(prompt_ids, dtype=mx.uint32)

    print("=" * 78)
    print(f"// mlx-lm version: {version}")
    print(f"// checkpoint    : {ckpt}")
    print(f"// prompt_ids ({len(prompt_ids)}): {prompt_ids}")
    print("=" * 78)
    print("step  pick_id  pick_repr            logp     gap2nd   | top-k (id:repr=logp)")

    gen_ids = []
    for i, (tok, logprobs) in enumerate(generate_step(prompt, model, max_tokens=N_NEW)):
        tid = int(tok)
        gen_ids.append(tid)

        lp = logprobs
        # Normalize to a 1-D vocab vector.
        if lp.ndim > 1:
            lp = lp.reshape(-1)
        # Top-K by logprob.
        order = mx.argsort(-lp)
        top_idx = [int(x) for x in order[:TOPK]]
        top_lp = [float(lp[j]) for j in top_idx]

        pick_repr = repr(tokenizer.decode([tid]))
        pick_lp = float(lp[tid])
        gap = top_lp[0] - top_lp[1] if len(top_lp) > 1 else float("nan")

        topk_str = "  ".join(
            f"{j}:{tokenizer.decode([j])!r}={v:.4f}" for j, v in zip(top_idx, top_lp)
        )
        print(
            f"{i:>4}  {tid:>7}  {pick_repr:<20} {pick_lp:8.4f} {gap:7.4f}  | {topk_str}"
        )

        if len(gen_ids) >= N_NEW:
            break

    print("=" * 78)
    print(f"// generated_ids ({len(gen_ids)}): {gen_ids}")
    print(f"// full decode: {tokenizer.decode(gen_ids)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

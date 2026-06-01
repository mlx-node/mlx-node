#!/usr/bin/env python3
"""Self-checking probe: dump mlx-lm's per-step greedy token + top-k logit
landscape for the LFM2 MoE golden prompt AND assert the committed benign-tie
invariants, so the "benign bf16 near-tie" claim at the compiled MoE divergence
step is a REPRODUCIBLE PASS/FAIL oracle (not just an inspector that prints + exits
0).

This is NOT the frozen-golden source (that is capture_lfm2_golden.py). It is the
reproducibility artifact behind the e2e harness's `GOLDEN_MOE_*` claims: it prints,
for every generated step, the argmax token and the top-K (token_id, repr, logit,
gap-to-next) so the actual logit geometry at the compiled-vs-mlx-lm divergence step
is readable, and then it VERIFIES a handful of committed invariants (mlx-lm version,
prompt ids, early generated ids, and the exact step-33 one-ULP tie). On ANY mismatch
it prints a clear message and `sys.exit(1)`; only when every invariant holds does it
print "PROBE SELF-CHECK PASSED" and exit 0. Regenerating the probe therefore
re-verifies the benign-tie claim.

mlx-lm's `generate_step` yields `(tok, logprobs)` where `logprobs` is the
log-softmax of that step's logits over the vocab. log-softmax is monotone and
shift-invariant in DIFFERENCES, so `logprob_a - logprob_b == logit_a - logit_b`:
the gaps between candidates are the true logit gaps, which is exactly what the
bf16-ULP argument needs.

Run (MoE 8B checkpoint):
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

# ---------------------------------------------------------------------------
# COMMITTED benign-tie invariants (the e2e harness's GOLDEN_MOE_* claims).
# Captured live on .cache/models/lfm2.5-8b-a1b (mlx-lm 0.31.3, M5 Max,
# 2026-06-01). These are the reproducible PASS/FAIL contract: regenerating this
# probe re-verifies them. They pin (1) the mlx-lm release the golden was frozen
# at, (2) the exact templated prompt ids, (3) the first 34 greedy generated ids,
# and (4) the step-33 one-bf16-ULP tie where our compiled MoE path picks ' So'
# (id 1672) while mlx-lm's continuous-prefill rounding picks ' One' (id 3231),
# with ' That' (id 3584) tied one ULP below ' So'.
# ---------------------------------------------------------------------------
EXPECT_MLX_LM_VERSION_PREFIX = "0.31"  # golden captured at 0.31.3

EXPECT_PROMPT_IDS = [
    124894, 124899, 5922, 207, 2992, 355, 278, 5205, 302, 3980, 39, 41774,
    296, 734, 2789, 12683, 22, 124900, 207, 124899, 63514, 207,
]

# First 34 greedy generated ids (0-indexed step 0..=33). Step 33 == id 3231 (' One').
EXPECT_GEN_IDS_34 = [
    124901, 207, 597, 4695, 20589, 34, 496, 2992, 355, 278, 5205, 302, 3980,
    39, 41774, 296, 734, 2789, 12683, 2426, 8, 2083, 1094, 310, 5141, 34, 440,
    5205, 302, 3980, 355, 4741, 22, 3231,
]

# Step-33 one-ULP tie token ids and the committed logprob-gap windows.
DIVERGENCE_STEP = 33
TOK_ONE = 3231  # ' One'  — mlx-lm's rank-#1 greedy pick at step 33
TOK_SO = 1672   # ' So'   — OUR compiled path's rank-#2 pick
TOK_THAT = 3584  # ' That' — tied one ULP below ' So'
# lp[' One'] - lp[' So'] ~= 0.25 == 1 bf16 ULP at this logit magnitude (~37).
EXPECT_ONE_MINUS_SO_RANGE = (0.20, 0.30)
# ' So' and ' That' are tied one ULP below ' One' (|gap| <= a sub-ULP slop).
EXPECT_SO_MINUS_THAT_ABS_MAX = 0.06


def _fail(msg: str) -> int:
    print(f"PROBE SELF-CHECK FAILED: {msg}", file=sys.stderr)
    return 1


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
    # Captured at the divergence step for the committed one-ULP-tie assertions.
    step33_lp = None
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

        if i == DIVERGENCE_STEP:
            # Materialize the step-33 log-softmax vector for the tie assertions.
            step33_lp = (
                float(lp[TOK_ONE]),
                float(lp[TOK_SO]),
                float(lp[TOK_THAT]),
                tid,
            )

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
    print("=" * 78)

    # ---- SELF-CHECK: the committed benign-tie invariants ------------------
    if not str(version).startswith(EXPECT_MLX_LM_VERSION_PREFIX):
        return _fail(
            f"mlx-lm version {version!r} does not start with "
            f"{EXPECT_MLX_LM_VERSION_PREFIX!r} (golden was captured at 0.31.3); the "
            f"committed invariants may not hold under a different release."
        )

    if list(prompt_ids) != EXPECT_PROMPT_IDS:
        return _fail(
            f"templated prompt_ids mismatch:\n  got    = {list(prompt_ids)}\n  "
            f"expect = {EXPECT_PROMPT_IDS}\nchat_template / tokenizer drift — the "
            f"frozen golden is stale."
        )

    if len(gen_ids) < len(EXPECT_GEN_IDS_34):
        return _fail(
            f"only {len(gen_ids)} ids generated; need >= {len(EXPECT_GEN_IDS_34)} "
            f"to verify the early-trajectory invariant."
        )
    if gen_ids[: len(EXPECT_GEN_IDS_34)] != EXPECT_GEN_IDS_34:
        return _fail(
            f"early generated ids mismatch:\n  got[:34]    = "
            f"{gen_ids[: len(EXPECT_GEN_IDS_34)]}\n  expect[:34] = {EXPECT_GEN_IDS_34}\n"
            f"the greedy trajectory diverged from the frozen golden."
        )

    if step33_lp is None:
        return _fail(
            f"did not capture step {DIVERGENCE_STEP} log-softmax (only "
            f"{len(gen_ids)} steps generated)."
        )
    lp_one, lp_so, lp_that, step33_pick = step33_lp

    if step33_pick != TOK_ONE:
        return _fail(
            f"step {DIVERGENCE_STEP} argmax id == {step33_pick} (expected "
            f"{TOK_ONE} == ' One'); mlx-lm's greedy pick at the divergence step changed."
        )

    one_minus_so = lp_one - lp_so
    lo, hi = EXPECT_ONE_MINUS_SO_RANGE
    if not (lo <= one_minus_so <= hi):
        return _fail(
            f"step {DIVERGENCE_STEP} lp[' One']-lp[' So'] = {one_minus_so:.4f} not in "
            f"[{lo:.2f}, {hi:.2f}] (the ~0.25 == 1 bf16 ULP gap; ' One' id {TOK_ONE}, "
            f"' So' id {TOK_SO}). The benign one-ULP tie no longer holds."
        )

    so_minus_that_abs = abs(lp_so - lp_that)
    if so_minus_that_abs > EXPECT_SO_MINUS_THAT_ABS_MAX:
        return _fail(
            f"step {DIVERGENCE_STEP} |lp[' So']-lp[' That']| = {so_minus_that_abs:.4f} > "
            f"{EXPECT_SO_MINUS_THAT_ABS_MAX:.2f} (' So' id {TOK_SO} and ' That' id "
            f"{TOK_THAT} were tied one ULP below ' One'). The tie structure changed."
        )

    print(
        f"// step {DIVERGENCE_STEP} tie: lp[' One']={lp_one:.4f} lp[' So']={lp_so:.4f} "
        f"lp[' That']={lp_that:.4f}  (One-So={one_minus_so:.4f}, "
        f"|So-That|={so_minus_that_abs:.4f})"
    )
    print("PROBE SELF-CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

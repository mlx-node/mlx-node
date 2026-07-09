# J-lens Phase 3 — GO/NO-GO verdict

**Verdict: 🟢 GO** (all four criteria met). Model: Qwen3.5-0.8B (24 layers, d=1024, tied embeddings).
Pack: `lens-pack-v1.safetensors` (F32, 96.5 MB, J.1..J.23, corpus-averaged over 100 windows × 128 tok).
Date: 2026-07-10. Decided from three deterministic, byte-identical-on-rerun artifacts (all in
`~/.cache/jlens/`, out of git): `eval-results-v1.json`, `band-report.json`, `qualitative-results.json`.

## The four GO criteria (plan §T3.3)

| # | criterion | result |
|---|-----------|:------:|
| 1 | J-lens pass@k AUC ≥ logit lens on ≥3/6 suites in-band | ✅ **6/6** |
| 2 | not materially worse than pilot on any suite in-band | ✅ **no material regression** |
| 3 | ≥2 qualitative slices: an interpretable intermediate at rank ≤~10 in J that logit does NOT show | ✅ **PASS** (clean; see below) |
| 4 | detectors show some band structure | ✅ **4/4** |

→ **GO**: ship a J-lens mode alongside the logit-lens lesson (Phase 2), with the honest scoping below.

## Criterion 1+2 — aggregate eval (the load-bearing metric)

`eval.mts` on the full pack, normalized log-k pass@k AUC (headline layers 1..23), J vs logit, 6 suites:

| suite | J-AUC | logit-AUC | Δ | J(pilot) | Δ vs pilot |
|-------|------:|----------:|--:|---------:|-----------:|
| multihop | 0.5296 | 0.2623 | +0.267 | 0.5433 | −0.014 |
| multilingual | 0.4838 | 0.2646 | +0.219 | 0.4936 | −0.010 |
| order-ops | 0.6376 | 0.2422 | +0.395 | 0.6094 | +0.028 |
| poetry | 0.0389 | 0.0306 | +0.008 | 0.0393 | −0.000 |
| typo | 0.7814 | 0.4323 | +0.349 | 0.7772 | +0.004 |
| association | 0.0534 | 0.0032 | +0.050 | 0.0360 | +0.017 |

J beats logit on all six; Δ is strongly positive on the four strong suites (+0.22..+0.40). logit-AUC matches
the pilot **exactly** on all six (the lens-independent sanity check — only J moves with the pack). vs pilot: 3
up, 3 within ±0.014 (corpus-averaging + the model's known decode bimodality) — no material regression.

## Criterion 4 — band structure (supporting proxies, NOT the paper's Fig-28 kurtosis)

`band-report.json`, 4 per-boundary detectors corpus-averaged over 551 eval prompts. **`bandStructurePresent =
true` (4/4)**. The one measured (not summary-stat) detector — the lens-legibility gap — rises cleanly from
0.11 (ℓ1) to 0.60 (ℓ17): a mid-stack interpretable band. All four are framed in-artifact as PROXIES /
supporting signal, never the GO criterion.

## Criterion 3 — qualitative pass (the deciding evidence for this phase)

`qualitative.mts` over a curated 14-item slice of the eval suites (paper story archetypes). A slice =
concept with J-rank ≤ 10 **and** logit-rank > 10 at the same fitted-J boundary (ℓ1..23), min over single-token
surface ids, apples-to-apples (only `useJacobian` differs). **56 J-only-legible slices across 5 prompts, 9
concepts** — but the *quality* is uneven, and honesty about that shapes what we feature:

**CLEAN — headline demonstration (feature this).** French `"La saison après l'été est l'"` (answer:
*l'automne*). The J-lens surfaces a coherent **season concept-cluster mid-stack** that logit never does:

```
ℓ17  J    : season · summer · winter · année · Season · phase …
     logit: most · busiest · <CJK junk> · easiest …
ℓ19  J    : season · summer · winter · year · month · autumn …
     logit: 最佳的 · busiest · opport · autumn(buried) …
```

`season` reaches J-rank 1 and `summer` J-rank 2 across an ℓ15..23 band while logit ranks them ≥14 / ≥32.
This is the paper's claim in miniature: the fitted Jacobian makes a mid-stack concept legible that the raw
unembedding does not.

**VALID but subtle.** Spanish `"Lo opuesto de \"grande\" es \""` (answer: *pequeño*). J keeps English
`small`/`tiny`/`micro` legible throughout; at ℓ20–22 logit *also* surfaces `small`, so the J-only slice is a
single late boundary (ℓ23, J-rank 2 vs logit 75) where logit has committed to Spanish surface forms
(`chico`/`pequeño`). Real, but not a clean mid-band story — feature only with that framing.

**WEAK / different phenomenon.** Arithmetic `"2 * (3 + 4) = "`. J's top-10 is a coherent **digit cluster**
(`2 6 3 5 8 4 7` … all ten digits by ℓ22) while logit is pure garbage — J "knows a number goes here" — but it
does not surface a crisp single intermediate concept (the operator `×` and results sit amid symbol soup).
Interpretable as *number-slot*, not as a named concept. "Try it" tier, not a headline.

**ABSENT in the curated sample.** multihop (4 items), association (3), poetry (2) produced **zero**
J-only-legible slices. On a 0.8B these tasks do not reliably show the phenomenon.

## Featured-preset recommendation (for a future J-lens widget/lesson ship)

- **Feature:** the French season prompt (clean mid-stack concept cluster) as the headline.
- **Feature with framing:** the Spanish opposite prompt (J holds the English concept while logit goes to the
  target language) — note it is a late-boundary effect.
- **Optional / "try it":** arithmetic (digit-cluster vs garbage) — honest as a *number-slot* demo, not a concept.
- **Do NOT hard-feature** multihop / association / poetry — frame "small models may not show this; try a
  bigger model." (Same honesty rule as the rest of the course: presets = only observed phenomena.)

## Scope of the GO

Ship the J-lens as an OPTIONAL mode next to the existing logit-lens lesson. Its honest pitch is:
1. **In aggregate** it makes intermediate concepts markedly more legible than the raw logit lens (6/6 suites,
   large Δ) — this is the strong, defensible claim.
2. **Qualitatively** it delivers at least one clean, screenshot-worthy mid-stack demonstration (French season)
   and a coherent supporting one (Spanish) — with an explicit caveat that on a 0.8B the effect is
   task-dependent and not universal.
The load-bearing claim is the aggregate metric; the qualitative presets are illustrations, curated to the ones
that actually reproduce.

## Honest caveats / non-claims

- 0.8B model: the qualitative effect is uneven (clean cross-lingual, digit-cluster arithmetic, absent on
  multihop/association/poetry in the sample). Do not claim universal per-prompt interpretability.
- The band-report detectors are PROXIES for the paper's residual excess-kurtosis onset (the read-only NAPI
  exposes only per-layer summary stats), not the literal 4th moment. Supporting signal only.
- Tuned-lens / causal-basis / swap results in the paper belong to Anthropic's models, not this run.

## Reproduce

```
env PATH="/opt/homebrew/bin:$PATH" JLENS_PACK=lens-pack-v1.safetensors oxnode packages/browser/scripts/jlens/eval.mts
env PATH="/opt/homebrew/bin:$PATH" JLENS_PACK=lens-pack-v1.safetensors oxnode packages/browser/scripts/jlens/band-report.mts
env PATH="/opt/homebrew/bin:$PATH" JLENS_PACK=lens-pack-v1.safetensors oxnode packages/browser/scripts/jlens/qualitative.mts
```

All three write deterministic, byte-identical-on-rerun JSON to `~/.cache/jlens/` (out of git).

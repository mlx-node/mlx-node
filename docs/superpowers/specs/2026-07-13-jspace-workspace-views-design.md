# J-Space "Workspace Views" — Design Spec

**Date:** 2026-07-13
**Source inspiration:** Anthropic, "Verbalizable Representations Form a Global Workspace in LMs" (transformer-circuits.pub/2026/workspace, Jul 2026) — Figs 3, 27, 28.
**App:** `packages/browser/demo/jspace/` (the `/jspace` Jacobian-lens playground on mlx.void.app).

## Goal

Add three model-free, client-only viewing features to `/jspace` that make the paper's "where in the stack does the model commit / route through the workspace" thesis visible on our existing readouts — **without any native or WASM change** (the branch `autoresearch/webgpu-production-20260406` forbids a rebuild) and without over-claiming metrics the shipped data cannot support.

## The three features

1. **Logit ↔ Jacobian divergence view** — a per-cell colormap over the same layers×positions grid showing how much the two lenses *disagree* at each cell. **Baked-only MVP** (both `frame.logit` and `frame.jacobian` already ship in every `BakedFile`, so this is zero worker calls and works cold). Live divergence is explicitly deferred (it needs two forward passes, a second held slice, the 46 MB pack gate, and a paired-state rewrite of the single-flight `useLensRun`).

2. **Workspace-metrics strip** — across-layer curves that locate the "workspace band," computed client-side from the shipped readout. **Honest-scoped** (see below).

3. **Motor-flip crossover marker** — per prompt position, the shallowest layer from which the readout's top guess already equals that position's final (ℓ-max) token and stays equal above it. The "motor" boundary from the paper's sensory/workspace/motor three-block picture. **LIVE-ONLY** (decided from data during the final review): commit depth is only well-defined on the contiguous 1..24 live layers. Baked frames sample only 11 layers ([6,8,10,12,14,16,17,18,20,22,24]), so the output token usually first wins in the ℓ22→ℓ24 gap and nearly every baked position reads "no lock" (verified: french-season is 0/9 locked on both lenses). `motorFlipLayer` returns `null` for that degenerate no-lock case, rendered as a neutral cell — never a false late commit. Coloured by physical layer depth so a layer reads identically across runs.

> **Scope change from the approved roadmap (item 5 → item 4), decided autonomously per the standing "decide yourself" directive:** The roadmap's item 5 was a "mental-arithmetic inner-sum" baked tile. Reading the gallery revealed `arith-inner-sum` (`'2 * (1 + 2) = '` → pins the unspoken inner sum `'3'`) is **already shipped** — plus `arith-precedence` and `arith-fewshot`. A `(3+4)*2=`→`7` tile would be a near-duplicate (YAGNI). It is replaced by the **motor-flip crossover marker** (roadmap item 4): genuinely net-new, pure client-side, **no offline GPU bake and no empirical dead-end risk**, and the natural companion to the metrics strip (both answer "where does the model commit").

## The data ceiling (verified in source — the honesty contract)

`lensReadout` ships, per `(layer, position)` cell (`LensCell`, `src/inspector-types.ts`):
`argmaxId`, `topKIds[K]`, `topKLogits[K]`, `topKProbs[K]` (TRUE full-vocab softmax), `topKTexts[K]`, with **`K = TOP_K = 10`** in the app (`JSpaceApp.tsx:48`).
Plus up to `LENS_MAX_PINNED = 8` pinned tokens, each with an **exact 1-based full-vocab `ranks` track** (capped 999) at every cell (`LensPinned`).

**No full `[vocab]` logit vector, no residual `h_ℓ` vector, and no activation tensor crosses the worker boundary.** Therefore:

| Paper Fig-28 metric | Needs | Verdict |
|---|---|---|
| top-k accuracy | rank of a target token | **FAITHFUL — for a *pinned* token only** |
| excess kurtosis | full ~151k logit row | **IMPOSSIBLE (omit)** |
| residual autocorrelation | full `h_ℓ` vectors | **IMPOSSIBLE** (readout-space proxy only) |
| activation participation-ratio | activation covariance | **IMPOSSIBLE** (readout-space proxy only) |

**Honesty invariants (binding on all UI copy):**
- The final **output token is never pinned** on a baked tile — the unspoken-word gate guarantees no pin equals the ℓ24 argmax. So exact full-vocab rank is available for **concepts**, not for the gold output token. The strip ships the **concept** rank trajectory, labeled as such — never "gold-token accuracy."
- Readout entropy / effective-dim / set-stability are computed over the visible top-10 and **MUST be labeled "readout-space proxy (top-10)"**, never presented as the paper's activation/residual quantity. `topKProbs` are true full-vocab probabilities, so entropy-over-visible is an honest *lower bound* (tail mass only sharpens it); `1/Σp²` over the visible set *overestimates* participation ratio — label as proxy, do not assert exactness.
- Excess kurtosis, residual autocorrelation, and activation participation-ratio are **OMITTED** with a one-line "needs full logits/residuals (not shipped in-browser)" note. Do **not** ship an approximate KL/JS full-distribution distance (only 10 probs/cell ship).

## What each feature computes

### 1. Divergence (per cell, from two slices of the SAME source)
- **`jaccardTopK(cellA, cellB)` → [0,1]** — `1 − |A∩B|/|A∪B|` over the two `topKIds` sets. **Default fill metric** (continuous). `divergence = 1 − Jaccard`.
- **`argmaxAgree(cellA, cellB)` → 0|1** — binary top-1 (dis)agreement. Optional overlay/mark.
- ~~`pinnedRankDelta`~~ — **DROPPED (codex T1).** A delta of two pinned ranks is dishonest: `rankAt` returns `RANK_CAP` (999) for both a genuine ≥999 rank and an out-of-range lookup, so the delta reports false agreement (both capped → 0) or a fabricated exact gap (one capped). `jaccardTopK` over the top-10 *set* has no such censoring and is the honest divergence signal (its only limit — depth 10 — is a stated scope bound, not a hidden lie).
- Never compare baked-vs-live (11 rows vs 24). Compare logit-vs-Jacobian **within one source only**.

### 2. Workspace-metrics strip (across layers, one slice)
- **(A) Concept rank trajectory** — `slice.rankAt(pinIdx, ℓ, lastPos)` across displayed layers. **FAITHFUL** full-vocab, exact **below** the cap; a value of `RANK_CAP` (999) is a censored off-scale floor ("at/beyond cap → not surfaced"), NOT an exact rank — the UI renders it off-scale (via `isCensoredRank`) and never labels it exact (codex T3). The real deliverable; directly carries the unspoken-word thesis (watch the concept dive from off-scale to ~1).
- **(B) Concept top-k accuracy** — `1[rank_concept ≤ k]` from the same exact track. **FAITHFUL**, labeled "concept in top-k" (not gold-token).
- **(C) Readout entropy / effective-dim** — `H = −Σ topKProbs·log topKProbs`, or `1/Σ topKProbs²`, over the visible 10. **Labeled "readout-space proxy (top-10)."**
- **(D) Top-K set stability** — adjacent-layer `Jaccard(topKIds_ℓ, topKIds_{ℓ+1})` at the last position. **Labeled "readout-space proxy"** (stand-in for autocorrelation, not the residual metric).
- Explicit one-line omit note for kurtosis / residual-autocorr / activation-PR.

### 3. Motor-flip crossover (per position)
- **`motorFlipLayer(slice, pos)` → layerIdx | null** — the **lowest** displayed layer index `ℓ` such that `cellAt(ℓ,pos).argmaxId === cellAt(topLayerIdx,pos).argmaxId` **and it stays equal for every layer above `ℓ` up to the top** (a stable lock). Computed by walking down from the top while the argmax holds; the lowest layer still equal is the flip layer. `null` if the top cell's argmax never appears below the top (degenerate). Interpretation: early flip = committed early (motor-like); late flip = deliberated (routed through the workspace).

## Architecture & constraints

- **Pure logic in `jlens-core/`** (`divergence.ts`, `workspace-metrics.ts`) as dependency-free functions over `LensSliceData` / `LensCell` — the TDD anchors. **UI in `jspace/`** reuses `RankHeatmapCanvas`/`RankChart`/canvas primitives.
- **Colormap dependency-free** — extend `colors.ts` with a blue–white–red diverging ramp beside `VIRIDIS_STOPS`; do NOT add d3/colormap.
- **Ephemeral by default** — divergence toggle, metrics strip, and motor-flip are NOT persisted in the permalink. This avoids re-opening the 5-touchpoint `JSpaceState` codec (recently hardened over 5 codex rounds). Divergence is a collapsed-by-default toggle; the strips render whenever a `slice` exists.
- **Mounting** — `JSpaceApp` is one long scroll of stacked `<section>`s (no tabs). New panels mount as siblings in the grid block, gated on `slice` (and `view.kind==='starter'` for baked divergence).
- **i18n** — `/jspace` has no `LocaleProvider`; copy is `JSPACE_COPY[locale]` (`{en, zh} as const`). **Every new UI string goes in BOTH blocks.** Glossary terms (Jacobian, logit lens, J-lens, token, rank, ℓ) stay English inside zh.
- **SSR-clean** — `matchMedia`-in-`useState` is the deliberate widget convention; do not "fix" it.

## Testing

`cd packages/browser && vp test --config vitest.config.ts [path]` (real headless Chromium, `fileParallelism:false`, `@`→`demo/`). Pure-fn suites (`jspace-divergence.test.ts`, `workspace-metrics.test.ts`) are the fast TDD anchors. UI tasks verified by their own suite + the existing jspace suite staying green.

## Out of scope (this plan)

Live divergence (two-pass paired state); any full-distribution metric; any native/WASM change; any new baked tile; any permalink field.

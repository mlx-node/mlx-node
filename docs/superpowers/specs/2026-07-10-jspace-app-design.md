# J-Space: a standalone Jacobian-lens explorer

**Date:** 2026-07-10
**Status:** design, pending user review
**Route:** `/jspace` (English only)
**Precedes:** implementation plan (`superpowers:writing-plans`)

---

## 1. What we are building

A full-page app at `/jspace` where you type your own prompt, press Enter, and see
the model's *internal* next-token guess at every layer and every position — the
same picture Anthropic's `slice_vis` viewer shows, but computed live in your
browser on WebGPU instead of baked offline in Python.

Today the J-lens only exists as a lesson widget locked to three fixed prompts.
The user cannot edit the prompt. This design unlocks that.

```
  today                                    /jspace
  ┌──────────────────────────┐             ┌──────────────────────────┐
  │ preset ▾  French season  │             │ ┌──────────────────────┐ │
  │                          │             │ │ type anything…    ⏎  │ │
  │  [ 11 layers × 9 pos ]   │    ──▶      │ └──────────────────────┘ │
  │                          │             │  [ 24 layers × ≤128 pos ]│
  │  LOGIT | JACOBIAN        │             │  + cross-sections        │
  └──────────────────────────┘             │  + rank charts, pins     │
   baked-first, SSG-safe                   │  + shareable link        │
                                           └──────────────────────────┘
```

---

## 2. Verified ground truth

Everything below was read from source, not inferred. Two claims from the research
pass were **wrong** and are corrected here.

| Claim | Verdict | Evidence |
|---|---|---|
| Pins over the cap silently truncate | **false** | `mlx-worker.ts:2732` and `model.rs:6250` both **error**. Only the comment at `mlx-worker.ts:2697` says "truncates" — the comment is wrong. |
| Overlapping runs can corrupt the model | **false** | `model.rs:754` — `lens_readout_sync` runs on a dedicated `Qwen35Cmd` command-loop thread. Concurrent calls **queue**; they cannot race. |
| Chunking layers into N calls multiplies the forward pass | true | `forward_capture_hidden` (`model.rs:9440`) loops all 24 layers with no early exit; `reset_caches_sync` (`model.rs:6452`) fires every call. |
| A position-tile of size ≥2 is bit-identical on WebGPU | true | `matmul.cpp:995` `(M==1 \|\| N==1) ? GEMV : GEMM`; the GEMM inner loop accumulates over `num_tiles = ceil(K/16)`, independent of `M`. A size-1 tile switches to a split-K tree reduction with a different summation order. |
| Query params are stripped ⇒ the permalink must use the hash | true | `__root.tsx:141` `validateSearch: searchSchema.parse` over a plain `z.object` — unknown keys are dropped on client navigation. |
| `eval_arrays` sits inside the layer loop | true | `model.rs:6422` — the big `[P,vocab]` transients are per-layer, not all-layers-live. |
| `bake.mts` and the widget derive pins differently | true | `bake.mts:189` id-equality against a learned standalone-space id, vs `JacobianLensLive.tsx:384` `text.trim() === ''`. |
| **There is a third pin loop** | **true, new** | `LogitLensLive.tsx:267-274` has **no whitespace guard at all**. Latent, not live: its concepts (`Paris`, `cold`, `blue`…) are alphabetic, so Qwen emits one `Ġ`-prefixed token. Change a concept to `"7"` and it pins a bare space. |

The last row is the whole argument for the architecture below. The Phase-4
whitespace-pin fix landed in two of three surfaces. Nobody noticed the third.

### From the reference implementations

`anthropics/jacobian-lens` is public, Apache-2.0. An agent fetched and read
`jlens/vis.py` and `jlens/data/slice_vis.html` verbatim.

- **A pin is a single token id.** `pinned = new Map()`, one rank series per id.
  There is no concept-family aggregation anywhere. Ours already matches.
- **The released data model stores ranks only — no probabilities.** The
  probability column visible in `assets/slice_vis.png` comes from an unreleased
  build. We compute honest full-vocab probabilities via `logsumexp`, so our
  tooltip is a strict superset.
- **The viewer runs no inference.** It is a static D3 renderer over precomputed
  gzip'd typed arrays. Neuronpedia's `/jlens`, by contrast, runs inference
  server-side and streams NDJSON per position. *Neither reference computes a lens
  in the browser.* Ours would be the first.
- **Deepest layer at the top.** `slice_vis.html:345` `layers[L-1-i]`. The repo's
  own README prose says the opposite and is stale; the code and screenshot agree.
- **Canvas, column-virtualized**, charts switch SVG→canvas past 200 x-values,
  by-position panel row-virtualized. Scaling is not speculation — it is what the
  reference does at 63 × 64.
- Their initial pin cap is 8 with a 10-colour palette. Ours hard-errors at 8.

---

## 3. Architecture

Three browser surfaces already exist (`LogitLensLive`, `JacobianLensLive`, and
now `/jspace`), plus one Node surface (`bake.mts`). Four copies of the same pin
logic is how we got here.

**Decision: share the pure core and the pin derivation. Keep the panels per-surface.**

```
                    ┌──────────────────────────────────────────┐
                    │  demo/jlens-core/     PURE. no React,     │
                    │                       no fs, no Worker.   │
                    │  types.ts      buildLensSlice  (moved)    │
                    │  colors.ts     RANK_CAP        (moved)    │
                    │  presets.ts                    (moved)    │
                    │  revive.ts                     (moved)    │
                    │  derive-pins.ts   ◀── NEW, one predicate  │
                    │  permalink.ts     ◀── NEW, pure codec     │
                    └──────────────────────────────────────────┘
                       ▲            ▲            ▲           ▲
        ┌──────────────┘            │            │           └──────────────┐
        │                           │            │                          │
  bake.mts (node)          LogitLensLive   JacobianLensLive           /jspace
  injects                  ┌─────────────────────────────┐      ┌──────────────────┐
  model.encodeTokens       │ lesson panels: DOM, ~99      │      │ app panels:      │
                           │ cells, SSG-prerendered,     │      │ canvas, ≤3072    │
                           │ baked-first                 │      │ cells, client    │
                           └─────────────────────────────┘      │ only             │
                                                                └──────────────────┘
```

**The import rule.** The core is pure, synchronous, and transport-free. It takes
the tokenizer as a **callback** — `bake.mts` injects `model.encodeTokens`, the
browsers inject the worker `tokenize`. The core never imports `node:fs`, React,
or `Worker`, and never touches `window`/`matchMedia` at module scope or in a
`useState` initializer. That last clause is what keeps the lesson chapters
prerendering with no model and no WebGPU.

Purity is already true of what moves: `colors.ts` has zero imports,
`jacobian-presets.ts` zero, `types.ts` one type-only import plus `RANK_CAP`,
`revive.ts` type-only.

**Why not share the panels.** `/jspace` needs a canvas grid at 24 × 128; the
lesson's `ArgmaxGrid` must stay DOM so its prerendered HTML is crawlable. Forcing
one component to serve both regresses one of them.

**Why not a standalone copy.** That is precisely the shape that produced the
whitespace-pin bug, and it would put a fourth divergent pin loop on the one
surface — arbitrary prompts — with no baked frame to catch it.

### `derivePins` — one predicate, three callers

```ts
// demo/jlens-core/derive-pins.ts
export type Encode = (text: string) => Promise<{ id: number; text: string }[]>;

export async function derivePins(concepts: string[], encode: Encode): Promise<{
  pinnedIds: number[];
  partialFlags: boolean[];
}>;
```

Pin each concept by the first token of `` ` ${concept}` ``. If that token's text
is whitespace-only, fall back to the space-less form. Skip empty encodings. This
adopts the widget's broader `text.trim() === ''` predicate (it catches `"\n"` and
`"  "`, which the bake's id-equality does not), and deletes all three call-site
loops. Cap at 8 — the backend errors above that.

`bake.mts` injects an adapter over `encodeTokens` (ids only) + `decodeTokens`;
the browsers inject the worker `tokenize`, which already returns `{id, text}[]`.

**`derivePins` maps concepts → pins. `/jspace` has no concepts.** Its pins are
single token ids the user clicks. `derivePins` is therefore used by the two lesson
widgets, by `bake.mts`, and by `/jspace` **only inside the self-test oracle**,
which replays the baked frame's four committed concepts. It must not appear
anywhere in the live user-prompt path.

---

## 4. The correctness oracle

`/jspace` runs arbitrary prompts. There is no baked frame to diff against. This
is the hardest problem in the design, and the reason is on the record:

> The f16 bug rendered garbage at every fitted layer while the worker truthfully
> reported `jacobianApplied: true`. A boolean flag cannot attest to numerics.

A cross-backend native-vs-WebGPU harness cannot settle it. Metal and WebGPU sum
`K` in different orders, and `argpartition` is exact — so any tolerance tight
enough to catch corruption also fires on benign GPU noise, and any tolerance
loose enough to pass the noise cannot catch corruption. It ships as a rubber
stamp or a permanent blocker.

**Instead: the app proves itself against a committed frame before it will assert
anything.** On the first Jacobian activation, `/jspace` silently recomputes
`french-season` — a *fixed* prompt whose native values are committed to the repo —
and compares. If it diverges, the Jacobian badge is refused and the user sees a
warning rather than a plausible-looking lie.

The self-test runs at the **baked frame's own settings**, not `/jspace`'s
defaults: `layers = JACOBIAN_LAYERS` (11 boundaries), the frame's `topK`, and
pins derived from the frame's four concepts. `promptLen` is 9, so it costs
roughly 160 ms — cheap enough to run silently every session. It also exercises
`derivePins` end-to-end against a committed reference, which no unit test can.

This works because the two distributions are orders of magnitude apart. We
measured both in Phase 4:

```
  benign cross-backend noise   │  the f16 storage bug
  ─────────────────────────────┼──────────────────────────────────────
  autumn: rank 634 vs 633      │  ℓ17 top-5 = ervo, ovski, ado,
  one top-2 swap on a          │             itten, ית
  0.096-logit tie              │  (zero token overlap with truth)
```

So the self-test is a **garbage detector, not a precision instrument** — and
garbage is exactly the failure mode it exists to catch. Thresholds: top-1 id
agrees on ≥ 90% of cells, and each pinned concept's best rank is within 3.

And because `ℓ24` is `J = I` by definition and never touches the pack, the two
boundaries together localize the fault:

```
                    ℓ24 (J = I)      a fitted ℓ (J from the pack)
  healthy           matches          matches
  pack broken       matches          garbage      ← the f16 bug, exactly
  readout broken    garbage          garbage
```

**What this does not catch:** a J that was fitted wrong but loaded correctly.
Native and browser would agree on the same wrong answer. That risk is carried by
the Phase-3 fit evaluation, not by this app. Say so in the UI copy.

---

## 5. Backend change: raise the cap, tile the unembed

`LENS_MAX_POSITIONS` is a **pure bound** — nothing is array-sized by it. Raising
it allocates nothing by itself. What it protects is the per-layer transient:

```
  per layer, live at the single eval_arrays (model.rs:6422):
    logits      [P, 248320] f32
    neg_logits  [P, 248320] f32
    partitioned [P, 248320] i32   (argpartition output)
    + one gt    [P, 248320]       per pinned id

  P =  48  →  ~143 MB and up      (today)
  P = 128  →  ~381 MB and up
```

**Raise to 128**, matching the context length the Jacobian was actually fitted at
(100 prompts × 128 tokens), and Anthropic's own `max_seq_len=128` fitting default.

**Tile the per-layer readout over positions** so peak memory stops tracking `P`.
Every operation inside the loop is last-axis / per-row — `final_norm`,
`argpartition(-1)`, `take_along_axis(-1)`, `argsort(-1)`, `logsumexp([-1])`, the
pinned `greater` + `sum([-1])`. Position-separable, verified line by line.
`forward_capture_hidden` stays full-`P` (attention mixes positions).

**Balanced tiling, C = 32.** Never `step_by`, which leaves a size-1 remainder:

```
  nTiles = ceil(P / 32)
  every tile is floor(P/nTiles) or that + 1
```

`P=33 → 17,16`  ·  `P=128 → 32,32,32,32`  ·  `P=9 → 9`  ·  `P=1 → 1`

**`eval_arrays` and `to_vec` must move inside the tile loop.** Slicing alone
changes nothing: MLX is lazy, so if the single `eval` stays at the layer level
the whole `[P, vocab]` graph still materializes at once and peak memory still
tracks `P`. Evaluating and reading back per tile is what makes the peak track the
*tile*, not the prompt. This is the acceptance criterion, not the slicing.

**`nTiles == 1` must take an un-sliced `h_l.clone()` branch**, not an identity
`slice_axis(1, 0, P)`. That explicit branch is what makes "lesson presets are
never tiled" an assertable invariant rather than a hope about graph equivalence.

**Push order is load-bearing.** `types.ts:39` indexes both `cells[]` and every
`pinned[pi].ranks[]` by `flat = layerIdx * promptLen + pos`. Cells must be pushed
at the **global** position `tile_start + local`, and pinned ranks appended with
`pi` outer / local position inner, per tile. Getting the pinned nesting wrong
compiles, passes every shape check, and silently scrambles the rank tracks.

A size-1 tile is the one thing that must never happen: `M == 1` routes to the
split-K GEMV, whose different `K`-summation order can flip an exact tie in the
downstream exact `argpartition`. Balanced tiling emits a size-1 tile **only** at
`P = 1`, which is a single untiled tile and therefore identical to today.

Lesson presets are `P ≤ 9` → one tile → **never tiled** → the committed baked
frames stay byte-identical. That is an assertable invariant, not a hope.

### The TS literals must move in the same commit

`mlx-worker.ts:2707` hard-rejects `> 48` before the wasm is ever called. Ship one
exported constant and reference it from both the guard and the client cap:

- `inspector.rs:1103` — the Rust const
- `mlx-worker.ts:2707`, `:2711`, `:2698` — guard, message, comment
- `inspector-types.ts:369` — JSDoc
- `mlx-worker.ts:2697` — **fix the lying "truncates pinnedIds" comment**

---

## 6. Execution model

**Trigger.** Enter (without Shift) or the Run button. Never debounce-on-keystroke.
This matches Neuronpedia's `jlens-completion.tsx` and is forced by the serial
command loop.

**Single-flight.** The Rust side serializes, so overlapping Runs are a *latency*
problem, not a corruption one. Never dispatch while one is in flight; disable
Run; keep only the newest pending prompt and fire it on resolve; a monotonic
generation ref drops stale results.

**No Stop button.** `AbortSignal` rejects the JS promise but the command is
already queued and `lens_readout_sync` has no cancellation token. A Stop button
that does not stop anything is a lie. Show elapsed time instead.

**No progressive rendering in v1.** Client-side layer chunking is not an
optimization — it is `N ×` the cost, because every `lensReadout` re-runs all 24
decoder layers. The only correct mechanism is a native per-layer callback (the
loop at `model.rs:6422` already materializes one boundary at a time, and the
forward runs once), which is a real wasm change. Defer it behind a measurement:

> Measure `T(L=1, P)` at `P ∈ {16, 48, 128}` and `T(L=2,P) − T(L=1,P)`.
> Nothing in our data separates the once-per-call forward from the per-layer
> cost — every sample we have is `L ≥ 8`. If the default prompt's p95 lands
> under ~800 ms, ship a blocking render with a spinner and drop progressive.

**Defaults.** Boundaries 1..24 in both modes (layer 0 has no `J` and hard-errors,
so excluding it keeps LOGIT and JACOBIAN directly comparable). `topK = 10`,
matching Anthropic. Pins capped at 8.

**Cost, projected** — these are extrapolations from a 4-point fit, all at `L ≥ 8`.
Label them as projections in the UI, not measurements:

```
  ms ≈ 58 + 1.4·L + 0.87·(L·P)        L = layers, P = positions

  a 16-token starter × 24 layers  ≈ 430 ms
  128 tokens × 24 layers          ≈ 2.8 s
```

---

## 7. Panels

| Panel | Status | Note |
|---|---|---|
| pos × layer argmax grid | generalize | `ArgmaxGrid` stays DOM for the lesson; `/jspace` gets a canvas twin, column-virtualized |
| top-k tooltip | reuse | `LensTooltip` + `TopKBars`. Keep the probability column — it is honest and it beats the reference |
| pin chips (display) | reuse | `PinChips` |
| pin add / remove | new | single token id, like Anthropic. Hard cap 8, Add disabled at 8 |
| rank heatmap over (pos × layer) | new | viridis on a **log**-rank domain, unranked cells grey |
| by-layer strip at fixed position | new | |
| by-position strip at fixed layer | new | row-virtualized |
| rank-vs-layer chart | new | log y, **rank 1 at the top** (bump-chart convention) |
| rank-vs-position chart | new | SVG → canvas past 200 x-values |
| keyboard scrub | new | WAI-ARIA grid, roving tabindex; `←→` position, `↑↓` layer, hold Shift to scrub |
| whitespace toggle | new | |
| click a character in the prompt | new | prompt renders as clickable token spans, `white-space: pre-wrap` |
| editable prompt + Run | new | one `tokenize` RPC → `promptIds`; client cap from the shared constant |
| LOGIT \| JACOBIAN toggle | reuse | `scaffolding/SegmentedToggle`, already used by both lesson widgets |
| starter chips | new | pre-baked offline so a cold visitor sees a real grid |
| permalink | new | |
| consent card | new | names the download sizes before anything is fetched |

**Existing `RankHeatmap` is not the pos × layer heatmap** — it collapses position
to the final one. Both ship; they answer different questions.

**Deepest layer at the top**, everywhere. Every panel reads cells through
`buildLensSlice`'s accessors, never by indexing `cells[]` directly. A panel that
gets this backwards turns "the concept surfaces at layer 18" into a confident
"layer 6".

**No layer selector.** `L` is fixed at 1..24 in v1, so `layers` is not permalink
state. Dropped.

---

## 8. Permalink and cold start

**Hash, not query** — the root route's `validateSearch` strips unknown query keys
on the first client navigation. The hash also keeps user prompts out of server
logs and referrers, and dodges CDN request-line limits.

```
  /jspace#p=La%20saison%20apr%C3%A8s…&mode=j&pins=1044-9871&sel=17,12
```

Readable, dependency-free, and the same shape as Anthropic's `?ctx&layer&pinned`.
No `lz-string`: a 128-token prompt is ~500 characters, far inside any limit.
The codec clamps `pins` to 8 on decode.

**A permalink restores state. It never runs.** Auto-running would auto-download
1.6 GB of weights for a stranger who clicked a link.

**Cold start.** Mounting `/jspace` creates no worker and downloads nothing.

```
  ┌─ /jspace, no prompt in the hash ──────────────────────────┐
  │  a pre-baked starter grid, real numbers, no model         │
  │  chips: French season · Spanish opposite · arithmetic     │
  │  [ consent card: ~1.6 GB weights, +46 MB Jacobian pack ]  │
  └───────────────────────────────────────────────────────────┘

  ┌─ /jspace#p=<a stranger's prompt> ─────────────────────────┐
  │  the prompt, shown as text. pins listed. grid skeleton.   │
  │  "Run to compute" + the same consent card.                │
  │  NOT a starter grid — that would be a different prompt's  │
  │  numbers under this prompt's heading.                     │
  └───────────────────────────────────────────────────────────┘
```

The 46 MB pack loads only when JACOBIAN is first selected, so LOGIT mode costs
one download, not two.

---

## 9. Rendering

`24 × 128 = 3072` cells. Anthropic's own viewer canvas-renders and
column-virtualizes at `64 × 63`. Follow it:

- main grid → canvas + pixel→cell hit-testing, columns virtualized, above
  `L·P > 1000`. Below that, the DOM grid the lesson already uses.
- cross-section strips (≤128 cells) → DOM.
- rank charts → SVG, switching to canvas past 200 x-values.
- `RANK_CAP` is 999 and is overloaded three ways (native ceiling, out-of-range
  sentinel, a genuine rank of 999). Anything `≥ 999` renders as a distinct
  off-scale mark with the axis labelled `≥999`. A log chart must not draw it as
  "moderately ranked".

---

## 10. Not building

- `/zh` mirror. English only.
- A Stop button (see §6).
- Client-side layer chunking (it is `N ×` slower).
- `lz-string` (unnecessary).
- Concept-family rank aggregation (the reference does not do it either).
- A layer selector.
- Raising `LENS_MAX_PINNED` above 8.

---

## 11. Verification

Value-based, not "review the code".

1. **Self-test frame.** `/jspace` recomputes `french-season` on first Jacobian
   activation and compares to the committed native frame. Refuses the badge on
   divergence. Assert **both** directions — a test that only ever passes proves
   nothing: the real pack passes, and a deliberately corrupted pack (e.g. one
   `J` tensor's bytes shifted by two, reproducing the f16 laning fault) fails.
2. **Tiling parity — on WebGPU, not Metal.** A `cargo test` runs the Metal
   backend, whose kernels are not the ones that ship; a Metal pass is ~no evidence
   about WebGPU. So tiling is toggled by a **diagnostic `noTile` request flag**
   threaded through `LensReadoutOptions` — *not* an env var, which does not reach
   the wasm worker — and the harness runs the *same binary* both ways **through
   the browser worker**: bit-identical top-k ids and pinned ranks at `P = 33`
   (proving the rebalance to 17+16 emits no size-1 tile) and `P = 1`.
   Plus a `P = 128` WebGPU smoke: the grid renders, the worker does not OOM.

   Note that the native `.node` addon is **not** rebuilt on this branch, so the
   offline bake keeps running the pre-tiling code. That is fine — lesson presets
   are `P ≤ 9` and never tile — but it means the re-bake gate exercises
   `derivePins`, not tiling. Only the browser harness tests the tiling.
3. **Lesson frames unchanged.** The three committed baked JSONs are byte-identical
   after the extraction *and* after the tiling change. Prerendered
   `/chapters/lm-head/jacobian-lens` still contains its baked token with zero
   model load.
4. **`derivePins` parity.** One golden test, two stub tokenizers (native-shaped
   and worker-shaped) → identical ids, texts, flags, including the `"7"` fallback.
   Then grep that **no pin loop remains** at any of the three call sites — a
   migration that adds the shared function and leaves the old loops in place
   passes every other test and still diverges.
5. **Pin cap.** 8 pins succeed; 9 are rejected by the UI before the worker errors.
6. **Cap raise.** `P = 128` passes the worker guard, `129` is rejected, and the
   same constant drives the guard and the client cap.
7. **Single-flight.** Three rapid Enters dispatch exactly one in-flight readout;
   only the newest pending prompt runs next; stale results are dropped.
8. **Permalink.** Round-trip identity; loading a hash restores state and does
   **not** kick off a download.
9. **Cold start.** Mount with providers null → starter grid renders, no `Worker`
   constructed, no weight request. Mount with a custom-prompt hash → skeleton,
   not a mismatched grid.
10. **Orientation.** A golden snapshot pinning deepest-at-top on every new
    cross-section panel, and a unit test that `rank ≥ 999` renders off-scale on an
    inverted axis.
11. **Progressive gate.** Record `T(L=1,P)` and the `P=128` tail after the wasm
    rebuild; decide per §6.

---

## 12. Unmeasured

Honest list. None of these block the design; all of them block a claim.

- `T(L=1, P)` — the forward pass has never been isolated from per-layer cost.
  Every timing sample we have is `L ≥ 8`. The cost model in §6 is a projection.
- WebGPU tile-≥2 bit-identity is read from `matmul.cpp`, not observed on a run.
  Verification #2 settles it.
- Native Metal tiling parity. Irrelevant if `C = 32` keeps lesson presets untiled,
  which it does, but assert it.
- Peak GPU memory: is the ceiling the resident tied embedding (~0.5 GB) or the
  readout transient? Tiling is prophylactic either way, and free.
- How often a ULP change actually flips a displayed cell on real prompts.

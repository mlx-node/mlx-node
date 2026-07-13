# /jspace Tier-1 Design Spec — Empirical Gallery, Concept Threading & Annotation Surfacing

**Repo:** `/Users/brooklyn/workspace/github/mlx-node-browser`
**Branch:** `autoresearch/webgpu-production-20260406` (WebGPU branch — native-rebuild forbidden; prebuilt wasm + existing `.node` addon only)
**Surface:** `packages/browser/demo/jspace/` — the standalone `/jspace` Jacobian/logit-lens playground on Qwen3.5-0.8B, deployed at `mlx.void.app/jspace`.

---

## 1. Goal

Turn `/jspace` from a bare lens viewer into an **honest, always-accessible empirical gallery** of vetted "unspoken word" examples, where a visitor can — with **no model download** — watch a concept the model never says surface mid-stack, see *which* concept is threading the argmax grid, and read where in the layer stack the effect lives.

Three decided upgrades (do **not** re-open):

| ID | Upgrade | One-line |
|----|---------|----------|
| **B** | Always-accessible gallery | Promote the cold-start starter selector to a persistent launcher of vetted examples (title + one-line hook cards). |
| **C** | Top-K concept threading | Tint an argmax-grid cell wherever a pinned concept is in that cell's top-K, opacity scaled by rank (argmax = strongest). |
| **D** | Annotation surfacing | Show the per-example blurb caption near the prompt, and a subtle **legibility band** on the layer axis (onset→peak). |

Plus **(A)** the empirical backbone: a committed, rubric-produced **honesty artifact** (`vetting.json`) that gates which examples ship and pins every caption number to measured data.

The gallery is **honest about the 0.8B scale gap**: the source paper (Anthropic, *Verbalizable Representations Form a Global Workspace in Language Models*, Sonnet 4.5) shows multi-hop chains, ValueError detection, injection-awareness, emoji→emotion mapping, and count-and-introspect. On Qwen3.5-0.8B these reproduce **only partially**. The gallery ships exactly what reproduces, labels weak tiles as weak, and discloses every failure.

---

## 2. Architecture / Approach

```
                         ┌─────────────────────────────────────────────┐
   headless vetting      │  packages/browser/scripts/jlens/             │
   (native .node addon)  │  vet-candidates.mts  ── runs 35 candidates ──┼─► vet-candidates-results.json (raw)
   Qwen35Model.lensReadout                                              │
                         │  bake.mts  ── bakes winners + resolves pins ─┼─► starters/<slug>.json  (model-free frames)
                         │            └─ writes rubric grades/bands ────┼─► starters/vetting.json (HONESTY ARTIFACT)
                         └─────────────────────────────────────────────┘
                                              │ (committed to git, reviewed like code)
                                              ▼
   ┌───────────────────────── /jspace runtime (browser, no model needed for gallery) ─────────────────────────┐
   │ starters/gallery.ts   GALLERY[] = single source of truth (slug, prompt, pinTokens, band, defaultMode, grade)│
   │ JSpaceApp.tsx         openStarter(slug) → loads baked frame → sets pins=pinIds, mode=defaultMode           │
   │ JSPACE_COPY (en/zh)   names · hooks · blurbs · bandNote  (COPY pattern, locale from readStoredLocale())     │
   │ ArgmaxGridCanvas.tsx  (C) top-K rank→alpha tint   ·   (D2) layer-axis band strip + accent onset/peak labels │
   └────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Key correction to the incoming brief (verified against source):**
- **BAND onset/peak render nowhere today.** `JacobianLensLive.tsx:523` is the *blurb*; only `BAND.boundaries` is consumed at `:60`. D2 is net-new UI.
- **`useLocale()` always returns `'en'` on `/jspace`** — the route `routes/jspace.tsx:6` is a bare `/jspace`, never wrapped in `<LocaleProvider>`. Locale MUST come from `readStoredLocale()` in an effect (SSG first paint stays EN-safe).
- **The vetted example set differs from the three legacy baked starters.** The incoming UI-design copy was written around `french-season / spanish-opposite / arithmetic-parens` (pre-vetting lesson examples). The Tier-1 gallery ships the **empirically vetted winners** (§4-A). Legacy `spanish-opposite` and `arithmetic-parens` are **superseded** and dropped from `/jspace`; `french-season` is **retained as the headline** (pilot-proven STRONG, memory `project_jlens_pilot_green`).
- **Concept threading is driven by the BAKED frame, not the `pins` state.** `colorByPinnedId` is built from `pinnedForView` = `view.pinned` = the baked frame's `slice.pinned` (and `PinManager`, the rank charts, and the labels all read `pinnedForView` too). So switching to a starter (`setStarterSlug`) is sufficient for (C) threading + chips + band ranks to light up — `openStarter` does **not** touch the `pins` state at all. (The `pins` state is only the permalink/live-run pin set; seeding it with the tile's concepts would merely dirty the cold URL and risk leaking gallery pins into a later custom live run.)
- **To avoid coupling the lesson, new example data lives in a `/jspace`-local registry** (`starters/gallery.ts`), NOT in the shared `jlens-core/jacobian-presets.ts` (which the lesson `JacobianLensLive.tsx` also imports — "core instrument unchanged").

---

## 3. Global Constraints

1. **0.8B honesty (non-negotiable).** No Sonnet-4.5 number or multi-hop chain that did not clear the rubric on *our* model. No mentalistic language ("thinks/plans/reasons/knows"); use "surfaces / appears in the readout / the lens ranks". Weak tiles say "faint but real" in the caption itself. Attribution once per gallery. Bridge-hop demos (Mars/France/Egypt) are **banned** — they do not reproduce here.
2. **Bilingual en/zh** via the COPY pattern. Glossary terms stay English (logit lens, Jacobian, token, rank, ℓ) per `feedback_zh_glossary_terms_english`. Every gallery name, hook, blurb, and `bandNote` has a zh string.
3. **No write path / no steering.** `lensReadout` is a pure, deterministic argmax forward pass. This spec adds **no** model mutation, no activation patching, no steering. Read-only.
4. **Core instrument unchanged.** No changes to `src/mlx-worker.ts handleLensReadout`, `packages/core/index.d.ts LensReadoutOptions`, `LensCell`, or `jlens-core/jacobian-presets.ts`. All required data (`LensCell.topKIds`, `BAND`, `readStoredLocale`, `STARTER_SLUGS`) already exists.
5. **No native rebuild on the WebGPU branch.** The existing `packages/core/mlx-core.darwin-arm64.node` (59 MB, Jul 9) runs the vetting headless as-is. `yarn build:native` is forbidden here.
6. **Prebuilt wasm only.** No wasm rebuild for this feature; the lens kernel already ships. The gallery is **model-free** at runtime — it renders baked JSON frames, so it works with zero wasm/model download.
7. **Position cap caveat.** The on-disk `.node` addon caps prompt positions at ~48 (the raised 128 cap is wasm-only). All vetted prompts are ≤16 tokens — comfortably under. Do not add gallery prompts >48 tokens.

---

## 4. Tier-1 Components

### A — Empirical gallery (vetted winners + honesty artifact)

Vetting was run **headless** (`canRunHeadless:true`) against the real native `lensReadout` (logit + fitted-Jacobian pack, 24 residual boundaries, topK=8, last-position next-prediction) over 35 paper-phenomenon candidates. The gallery ships **7 tiles (5 STRONG + 2 WEAK)** — within the 6–8 target, ≥4-STRONG floor satisfied. `giza-continent` is graded WEAK (the reproducible bake shows `Africa` at rank ~2 with a degenerate `'1'` output and no bridge hop, not a top-3 STRONG tile).

> **Post-review reconciliation (adversarial re-review, 8 findings).** `eiffel-capital` was **DROPPED**: `Paris` (id 11751) is the model's **ℓ24 greedy output** — a *spoken* answer, so it violates the gallery's "unspoken word" invariant. The bake now asserts no pinned concept equals the ℓ24 argmax, and no STRONG/WEAK caption may misstate a rank (all rank numbers are 1-based, matching the committed frames). Injection / emoji-face / count-and-introspect were dropped during vetting (total floor); eiffel joins them in `vetting.json`'s dropped roster. **The shipped copy + rank/band numbers in `demo/jspace/JSpaceApp.tsx` and `gallery.ts` are the source of truth** — the illustrative copy/table below is the design-time snapshot.

**Shipped gallery (STRONG first; `french-season` headline):**

| # | slug | prompt (verbatim) | pinTokens | concept (unspoken) | peak ℓ | band ℓ | grade | default lens |
|---|------|-------------------|-----------|--------------------|--------|--------|-------|--------------|
| 1 | `french-season` | *(existing baked prompt)* | *(existing)* | season → summer → *automne* | *BAND.legibilityPeak* | *BAND.onset..peak* | STRONG | jacobian |
| 2 | `arith-inner-sum` | `2 * (1 + 2) = ` | `["3"," 3"]` | **3** — inner sum (1+2) before the multiply | 18 | 15–18 | STRONG | jacobian |
| 3 | `arith-precedence` | `3 + 4 * 2 = ` | `["8"," 8"]` | **8** — product 4×2 held before adding 3 | 20 | 18–20 | STRONG | jacobian |
| 4 | `arith-fewshot` | `(1+2)*2=6. (2+3)*2= ` | `["5"," 5"]` | **5** — inner sum (2+3) in the analogy | 20 | 20–23 | STRONG | jacobian |
| 5 | `grammar-error` | `The plural of 'child' is childs.` | `[" Incorrect"]` | **Incorrect** — error flag on wrong plural | 17 | 16–17 | STRONG | jacobian |
| 6 | `giza-continent` | `Fact: The continent where the pyramids of Giza are located is ` | `[" Africa","Africa"]` | **Africa** — answer, before degenerate output | 17 | 17–18 | WEAK | jacobian |
| 7 | `int-cast-error` | `>>> int('hello')\n` | `[" Error"]` | **Error** — generic runtime-error concept | 18 | 18–18 | WEAK | jacobian |
| — | ~~`eiffel-capital`~~ | *dropped post-review — `Paris` IS the ℓ24 output (spoken)* | — | — | — | — | ~~WEAK~~ → **DROPPED** | — |

`band ℓ` = `[bandLayers[0], peakLayer]`, both **rubric-produced** and written to `vetting.json` (§5); the values above are from the completed vetting run and are re-confirmed (never hand-tuned) when the artifact regenerates. `french-season`'s band reuses the lesson's tuned `BAND.onsetBoundary / BAND.legibilityPeak` (no invented numbers).

**Dropped phenomena (recorded in `vetting.json` with `shipped:false` + reason):** prompt-injection recognition (only prompt-echo pins hit), ASCII/emoji faces (zero emotion-word hits), count-and-introspect (introspection floor; next-number is the model's *actual* output, failing Gate 3).

**Honest framing baked into captions:** arithmetic tiles 2–3 are the flagships (unspoken inner value held mid-stack by the J-lens while the logit lens shows it only at ℓ24 — *and* the model produces the correct final answer). Tiles 6/8 surface the **answer** before a degenerate `'1'` output — **not** a bridge hop (Egypt/France never surface); the caption says so. Tile 5's *correction* ('children') never surfaces — only the error flag; the caption says so.

**Task A is the build's first task** (headless): run the vetting runner, regenerate `vetting.json` + bake the 8 starter frames, before any UI work. Runner:

```bash
cd /Users/brooklyn/workspace/github/mlx-node-browser/packages/browser \
 && env PATH="/opt/homebrew/bin:$PATH" JLENS_PACK=lens-pack-v1.safetensors \
    oxnode scripts/jlens/vet-candidates.mts     # raw → vet-candidates-results.json
 && env PATH="/opt/homebrew/bin:$PATH" JLENS_PACK=lens-pack-v1.safetensors \
    oxnode scripts/jlens/bake.mts               # → starters/<slug>.json + starters/vetting.json
```

Native call shape (from feasibility, verbatim — used by `vet-candidates.mts` / `bake.mts`):

```js
const { Qwen35Model } = await import('@mlx-node/lm');
const model = await Qwen35Model.load(
  '/Users/brooklyn/workspace/github/mlx-node/.cache/models/qwen3.5-0.8b-mlx-bf16');
await model.loadLensPackFromFile(
  '/Users/brooklyn/workspace/github/mlx-node/.cache/jlens/lens-pack-v1.safetensors'); // 23 Jacobians J.1..J.23
const promptIds = await model.encodeTokens(prompt);
const out = await model.lensReadout(promptIds, {
  layers: Array.from({length:24}, (_,i)=>i+1),   // residual boundaries 1..24
  topK: 8,                                        // = UI K; never vet wider than we render
  pinnedIds,                                      // concept ids (Gate-1 rank probe)
  useJacobian: true,                              // false = logit lens (no pack needed)
});
// out.cells is LAYER-MAJOR flat: cells[li*promptLen + pos]; last-token = pos promptLen-1
// out.pinned[b].ranks: Int32Array length layers.length*promptLen → per-(layer,pos) rank
```

`bake.mts` additionally **resolves `pinTokens` → `pinIds`** (all vocab-present variants of each string, e.g. `"3"` and `" 3"`) and writes them into each `starters/<slug>.json` so `/jspace` never needs the model to pin.

---

### B — Always-accessible gallery UI — `JSpaceApp.tsx:658–677`

Today the selector is gated behind `view.kind === 'starter'` (`isColdPrompt(prompt)` = `prompt === ''`, `cold-prompt.ts:4`) — it vanishes the moment the user types. Promote it to a **persistent launcher**.

**New local registry** `demo/jspace/starters/gallery.ts` (single source of truth; `STARTER_SLUGS` derives from it):

```ts
export interface GalleryEntry {
  slug: string;
  prompt: string;
  pinTokens: string[];                 // human-readable; bake resolves → pinIds in <slug>.json
  band: { onset: number; peak: number };
  defaultMode: 'jacobian' | 'logit';
  grade: 'strong' | 'weak';
}
export const GALLERY: readonly GalleryEntry[] = [ /* the 8 rows of §4-A, in order */ ];
export const STARTER_SLUGS = GALLERY.map((g) => g.slug);
```

**Launcher handler** (add near `handleModeChange`). The view memo prefers `liveResult`, so we must `resetRun()` **and** blank the prompt to fall through to the starter branch. It sets the entry's default lens; it does **not** set `pins` (threading comes from the baked `view.pinned`, per the note in §2):

```ts
function openStarter(slug: string): void {
  const entry = GALLERY.find((g) => g.slug === slug);
  if (!entry) return;
  setStarterSlug(slug);
  setPrompt('');                         // cold → view.kind === 'starter'
  resetRun();                            // drop any live frame (memo prefers it otherwise)
  committedPromptIdsRef.current = null;
  setMode(entry.defaultMode);            // J-only tiles open in jacobian; band memo gates on this
  // NB: do NOT setPins here — threading/chips/charts read the baked view.pinned (§2).
  setActivePinIdx(null);
  setSelected(null); setHovered(null); setFocusCell(null);
  setTokenCount(null); setRunError(null);
  pendingSelRef.current = null;          // user divergence — cancel pending permalink restore
}
```

`pinsForSlug(slug)` reads the baked `starters/<slug>.json`'s `pinIds` (already in the bundle — model-free).

**Replace 658–677** — remove the `view.kind === 'starter'` gate, delete the old `SegmentedToggle` labeled with raw prompt strings (665–674), render **always** after the prompt/controls `</section>` (656) as localized title + one-line-hook cards:

```tsx
<section aria-labelledby="jspace-gallery" className="space-y-2">
  <span id="jspace-gallery" className="font-mono text-[10px] uppercase tracking-[0.18em] text-[color:var(--text-dim)]">
    {copy.galleryLabel}
  </span>
  <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
    {STARTER_SLUGS.map((slug) => {
      const active = view.kind === 'starter' && slug === starterSlug;   // launcher, not state mirror
      return (
        <button key={slug} type="button" aria-pressed={active} onClick={() => openStarter(slug)}
          className={['rounded-lg border p-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
            active ? 'border-primary/50 bg-primary/10'
                   : 'border-border/60 bg-card/30 hover:border-primary/40 hover:bg-primary/5'].join(' ')}>
          <span className="block font-display text-sm text-foreground">{copy.presetNames[slug] ?? slug}</span>
          <span className="mt-0.5 block text-[11px] leading-snug text-[color:var(--text-dim)]">{copy.galleryHooks[slug]}</span>
        </button>
      );
    })}
  </div>
</section>
```

- **Launcher, not state mirror:** `aria-pressed`/active styling only while a starter is displayed. Live/custom runs highlight nothing — it's a "load this" affordance.
- **Permalink:** `starterSlug` is not in the permalink schema (`permalink.ts` = prompt/mode/pins/sel). `openStarter` sets prompt `''` and mode, and leaves `pins` empty, so a fresh cold-gallery visit keeps a **clean URL** (the `isColdDefault` guard suppresses the write). Consequently a shared cold-gallery URL restores the *default* tile, not the specific one selected — an accepted Tier-1 limitation (deep-linking a specific tile would require adding `starterSlug` to the permalink schema, which is out of scope). Permalinks remain meaningful for custom prompts + user pins + selection.

---

### C — Top-K concept threading — `ArgmaxGridCanvas.tsx:205–211`

`LensCell.topKIds` (`src/inspector-types.ts:403`) is already the per-cell ranking in descending-logit order, so **the index into `topKIds` IS the rank-within-top-K** (0 = argmax; `topKIds[0] === cell.argmaxId`). No `rankAt`/pinnedIdx map needed; effective K = `cell.topKIds.length`. Baked starters carry `topKIds`, so threading is model-free.

**Replace 205–211** — argmax-only tint → top-K scan with an exact rank→alpha ramp:

```ts
// Concept threading: tint by the BEST-ranked pinned token in this cell's top-K.
// topKIds is descending-logit, so the first hit is the strongest concept here.
let tintColor: string | undefined;
let hitRank = 0;
for (let k = 0; k < cell.topKIds.length; k++) {
  const c = colorByPinnedId.get(cell.topKIds[k]!);
  if (c) { tintColor = c; hitRank = k; break; }        // strongest wins the cell
}
if (tintColor) {
  ctx.globalAlpha = tintAlphaForRank(hitRank, cell.topKIds.length);
  ctx.fillStyle = tintColor;
  ctx.fillRect(x, y, CELL_W, CELL_H);
  ctx.globalAlpha = 1;
}
```

**Extract the ramp as a pure, unit-testable function** (new export in `ArgmaxGridCanvas.tsx` or a sibling `argmax-tint.ts`):

```ts
export const ALPHA_MAX = 0.18;   // argmax hit — byte-identical to today's only tinted case
export const ALPHA_MIN = 0.045;  // faintest top-K-tail hit
export function tintAlphaForRank(hitRank: number, K: number): number {
  const frac = K > 1 ? hitRank / (K - 1) : 0;          // 0 at argmax … 1 at top-K tail
  return ALPHA_MAX - (ALPHA_MAX - ALPHA_MIN) * frac;
}
```

```
rank k:   0(argmax)  1     2     3   ...   K-1(tail)
alpha:    0.18 ──────────── linear ──────────── 0.045
          ^ unchanged: today's argmax-only case keeps exactly 0.18
```

- **Argmax preserved (byte-identical):** pin==argmax → `topKIds[0]` hits → `hitRank=0` → `frac=0` → `0.18`. Non-argmax top-K hits fade proportionally; no-hit cells stay untinted (as today).
- **Strongest-wins-cell** (single `fillRect`, `break` on first hit): avoids alpha-compositing multiple translucent fills into mud; keeps "argmax = strongest". Different pins still thread the grid **simultaneously in their own accent colors** — this is the global-workspace view (multiple concepts visible before any becomes argmax).
- **Perf:** ≤K `Map.get`, early-break (usually 0–1). Per redraw ≈ visibleCells·K ≈ 24×20×10 ≤ 4800 lookups — negligible vs. the existing per-cell text layout.
- **No new redraw trigger:** the draw-effect dep list (`:265`) already has `slice` + `colorByPinnedId`; `topKIds` lives inside `slice`. Nothing else to invalidate.

---

### D — Annotation surfacing

#### D1 — blurb caption near the prompt — `JSpaceApp.tsx` (starter section)

Render only when `view.kind === 'starter'`. Blurb text is bilingual, from `JSPACE_COPY`. Add after ~line 471:

```ts
const galleryEntry = GALLERY.find((g) => g.slug === starterSlug) ?? null;
```

Under the gallery:

```tsx
{view.kind === 'starter' && galleryEntry ? (
  <p className="max-w-[62ch] text-[12px] leading-relaxed text-[color:var(--text-dim)]">
    {copy.blurbs[galleryEntry.slug]}
  </p>
) : null}
```

#### D2 — legibility band on the LAYER axis — `ArgmaxGridCanvas.tsx`

The layer axis is the sticky gutter (`:251–264`). Mark the band **on the gutter only**; cells untouched. Add an optional prop `band?: { onset: number; peak: number } | null` to the destructure (`:72–89`) and the draw-effect deps (`:265`). Replace the gutter label loop (`:261–264`):

```ts
const ACCENT = CANVAS.selectionRing; // #ec4899, the app accent
for (let r = 0; r < rowOrder.length; r++) {
  const L = slice.layers[rowOrder[r]!]!;
  const inBand = band && L >= band.onset && L <= band.peak;
  if (inBand) {
    ctx.globalAlpha = 0.10;                         // faint accent strip, gutter inner edge
    ctx.fillStyle = ACCENT;
    ctx.fillRect(GUTTER_W - 3, r * CELL_H, 3, CELL_H);
    ctx.globalAlpha = 1;
  }
  const isMark = band && (L === band.onset || L === band.peak);
  ctx.fillStyle = isMark ? ACCENT : CANVAS.inkMuted;   // accent onset & peak labels only
  ctx.fillText(`ℓ${L}`, CELL_PAD, r * CELL_H + CELL_H / 2);
}
```

```
 ℓ24  |
 ℓ20 ▮|  ← peak label in accent
 ℓ19 ▮|
 ℓ18 ▮|   ▮ = 3px accent strip on band rows (α .10)
 ℓ17 ▮|  ← onset label in accent
 ℓ16  |
```

**Wire from `JSpaceApp.tsx`** — band is **per-entry** (each vetted tile has its own band; not the global `BAND`), memoized so a fresh literal doesn't retrigger redraw. Gate on **starter + jacobian** (live/custom + logit are visually unchanged):

```ts
const band = React.useMemo(
  () => (view.kind === 'starter' && mode === 'jacobian' && galleryEntry ? galleryEntry.band : null),
  [view.kind, mode, galleryEntry],
);
```
```tsx
<ArgmaxGridCanvas /* … */ band={band} />
```

Optional one-line caption above the grid header (~line 729): `copy.bandNote(galleryEntry.band.onset, galleryEntry.band.peak)`.

---

## 5. Honesty artifact — `demo/jspace/starters/vetting.json`

One committed JSON, regenerated by `bake.mts`, reviewed like code. **Every** paper phenomenon attempted gets a row — the `shipped:false` rows are the honesty.

```jsonc
{
  "model": "Qwen3.5-0.8B",
  "layers": 24,
  "topK": 8,
  "greedyContinuationTokens": 8,
  "intermediateBand": [6, 20],
  "vettedAt": "2026-07-13",
  "candidates": [
    { "slug": "arith-inner-sum", "prompt": "2 * (1 + 2) = ", "targetConcept": "3",
      "targetIds": [/* resolved */], "greedyOutput": "6…", "inPrompt": false, "inOutput": false,
      "logit":    { "pass": true,  "peakLayer": 24, "peakRank": 6, "bandLayers": [24] },
      "jacobian": { "pass": true,  "peakLayer": 18, "peakRank": 2, "bandLayers": [15,16,17,18] },
      "grade": "STRONG", "shipped": true, "note": "flagship: unspoken inner sum + correct final answer" },
    { "slug": "eiffel-capital", "targetConcept": "Paris",
      "logit":    { "pass": false },
      "jacobian": { "pass": true,  "peakLayer": 18, "peakRank": 5, "bandLayers": [18] },
      "grade": "WEAK", "shipped": true, "note": "J-lens only; answer before degenerate '1'; no France bridge hop" },
    { "slug": "empty-list-valueerror", "targetConcept": "ValueError",
      "logit": { "pass": false }, "jacobian": { "pass": false, "reason": "0.8B never forms the error concept" },
      "grade": "FAIL", "shipped": false, "note": "Paper shows this on Sonnet; does not reproduce at 0.8B." },
    { "slug": "prompt-injection", "grade": "FAIL", "shipped": false,
      "reason": "Total floor; only prompt-echo pins ('ignore','translate') hit — disqualified." }
    // … one row per attempted phenomenon, all 8 winners + every dropped candidate
  ]
}
```

**Rules:** `grade / peakRank / bandLayers` are produced by the rubric code, never hand-typed. The band fields feed §4-A's `band` values and the D2 annotations, so caption numbers and the artifact **cannot drift**. `greedyOutput` proves Gate 3 (concept ∉ output). Raw per-layer top-K stays at `.cache/jlens/vet-candidates-results.json`.

**Rubric (3 gates, produced by the runner):** Gate 1 APPEARS (target ∈ top-K at ℓ∈[6,20]); Gate 2 NOT-PROMPT (id ∉ prompt ids); Gate 3 NOT-OUTPUT (∉ 8-token greedy continuation). STRONG = top-3 at peak AND ≥3 consecutive band layers; WEAK = top-K but rank>3 or single layer; FAIL = any gate fails → not shipped. Vet at **K=8** (the UI K) — never wider than we render.

---

## 6. i18n plumbing (COPY pattern, en/zh)

`/jspace` has **no `LocaleProvider`** — resolve from the stored preference in an effect (EN-only prerender stays safe):

```ts
import { readStoredLocale, type Locale } from '../lib/i18n';
// top of state block (~line 77):
const [locale, setLocale] = React.useState<Locale>('en');            // SSG first paint = EN
React.useEffect(() => { const s = readStoredLocale(); if (s) setLocale(s); }, []);
const copy = JSPACE_COPY[locale];
```

A zh visitor arriving from the `/zh` landing already has `mlx:preferences:locale = 'zh'` in localStorage, so the gallery/captions flip to zh after mount.

**Module-scope `JSPACE_COPY`** (mirror `JacobianLensLive.tsx:95-160`), scoped to the new surfaces; page chrome (header "J-Space", consent copy, prompt label, Run button, status lines) stays English (out of scope; flag as follow-up):

```ts
const JSPACE_COPY = {
  en: {
    galleryLabel: 'Examples · no model needed',
    presetNames: {
      'french-season': 'French season', 'arith-inner-sum': 'Inner sum',
      'arith-precedence': 'Operator precedence', 'arith-fewshot': 'Few-shot sum',
      'grammar-error': 'Grammar error', 'giza-continent': 'Giza continent',
      'int-cast-error': 'Bad int() cast', 'eiffel-capital': 'Eiffel capital',
    } as Record<string, string>,
    galleryHooks: {
      'french-season': "Mid-stack the lens surfaces 'season' then 'summer' before the answer 'automne'.",
      'arith-inner-sum': "The unspoken inner sum '3' (1+2) surfaces at ℓ18; the model then answers 6.",
      'arith-precedence': "Precedence: the product '8' (4×2) is held mid-stack before adding 3.",
      'arith-fewshot': "The inner sum '5' (2+3) shows up across ℓ20–23 in the few-shot analogy.",
      'grammar-error': "After 'childs', the judgment 'incorrect' surfaces — the correction never does.",
      'giza-continent': "Faint: the answer 'Africa' surfaces mid-stack while the real output is a degenerate '1'; the Egypt bridge hop never appears.",
      'int-cast-error': "Faint: a generic 'error' concept for int('hello') — never 'ValueError'.",
      'eiffel-capital': "Faint: the answer 'Paris' surfaces; the France bridge hop never does.",
    } as Record<string, string>,
    bandNote: (onset: number, peak: number) =>
      `Legibility band: intermediate concepts start surfacing around ℓ${onset}, peaking near ℓ${peak}.`,
    blurbs: {
      'french-season': "…(keep existing english blurb text)…",
      'arith-inner-sum': "Jacobian lens surfaces the unspoken inner sum '3' (1+2) at rank 2 by ℓ18, then the model correctly outputs 6. The logit lens shows '3' only at the final layer (rank 6). '3' is in neither the prompt nor the output.",
      'arith-precedence': "Precedence in action: the Jacobian lens holds the unspoken product '8' (4×2) at rank 0 around ℓ20, then the model answers 11. The logit lens buries '8' at rank 7 in the last layer only.",
      'arith-fewshot': "The unspoken inner sum '5' (2+3) sits at rank 2 across ℓ20–23 in the Jacobian lens; the logit lens shows it only at the final layer. '5' appears in neither the prompt nor the output.",
      'grammar-error': "After the error 'childs', both lenses raise the unspoken judgment 'incorrect' to rank 1–2 from ℓ17 — genuine error detection. Honest caveat: the correction 'children' never surfaces, only the flag.",
      'giza-continent': "Faint but real: the Jacobian lens surfaces the unspoken answer 'Africa' (around rank 2, ℓ17–18) while the logit lens shows nothing and the model's actual greedy output is a degenerate '1'. Honest caveat: the Egypt bridge hop never appears on this 0.8B model.",
      'int-cast-error': "Faint but real: the Jacobian lens raises a generic 'error' concept to rank 1 at ℓ18 for the invalid int('hello') cast (the logit lens shows nothing). Never 'ValueError' or 'invalid' as the paper's larger models show.",
      'eiffel-capital': "Faint but real: the Jacobian lens surfaces the unspoken answer 'Paris' at rank 5–6 around ℓ18 while the model's real output is a degenerate '1'; the France bridge hop never appears. The logit lens shows nothing.",
    } as Record<string, string>,
  },
  zh: {
    galleryLabel: '示例 · 无需模型',
    presetNames: {
      'french-season': '法语·季节', 'arith-inner-sum': '内层求和',
      'arith-precedence': '运算优先级', 'arith-fewshot': '少样本求和',
      'grammar-error': '语法错误', 'giza-continent': '吉萨·大洲',
      'int-cast-error': 'int() 转换错误', 'eiffel-capital': '埃菲尔·首都',
    } as Record<string, string>,
    galleryHooks: {
      'french-season': "中间层 lens 先浮现 'season'，再是 'summer'，然后才是答案 'automne'。",
      'arith-inner-sum': "未说出口的内层和 '3'（1+2）在 ℓ18 浮现，随后模型答出 6。",
      'arith-precedence': "优先级：乘积 '8'（4×2）在中间层被暂存，之后才加上 3。",
      'arith-fewshot': "内层和 '5'（2+3）在 ℓ20–23 的少样本类比中出现。",
      'grammar-error': "在 'childs' 之后，判断词 'incorrect' 浮现——但正确写法从未出现。",
      'giza-continent': "微弱：答案 'Africa' 在中间层浮现，而真实输出是退化的 '1'；埃及这一跳桥概念从未出现。",
      'int-cast-error': "微弱：int('hello') 只浮现出泛化的 'error'——从不是 'ValueError'。",
      'eiffel-capital': "微弱：答案 'Paris' 浮现，但法国这一跳桥概念从未出现。",
    } as Record<string, string>,
    bandNote: (onset: number, peak: number) =>
      `可读性区间：中间概念大约从 ℓ${onset} 开始浮现，在 ℓ${peak} 附近达到峰值。`,
    blurbs: {
      'french-season': "…(保留现有中文 blurb)…",
      'arith-inner-sum': "J-lens 在 ℓ18 以 rank 2 浮现出未说出口的内层和 '3'（1+2），随后模型正确输出 6。logit lens 仅在最后一层（rank 6）显示 '3'。'3' 既不在 prompt 也不在输出中。",
      'arith-precedence': "优先级实况：J-lens 在 ℓ20 附近以 rank 0 暂存乘积 '8'（4×2），随后模型答出 11。logit lens 仅在最后一层把 '8' 埋在 rank 7。",
      'arith-fewshot': "未说出口的内层和 '5'（2+3）在 J-lens 中横跨 ℓ20–23 稳定处于 rank 2；logit lens 仅在最后一层显示。'5' 既不在 prompt 也不在输出中。",
      'grammar-error': "在错误的 'childs' 之后，两种 lens 都从 ℓ17 起把判断词 'incorrect' 抬到 rank 1–2——真实的错误检测。诚实提醒：正确写法 'children' 从未浮现，只有错误标记。",
      'giza-continent': "微弱但真实：J-lens 在 ℓ17–18 附近以 rank 2 浮现出未说出口的答案 'Africa'，而 logit lens 毫无显示，模型真实的贪心输出是退化的 '1'。诚实提醒：埃及这一跳桥概念在 0.8B 模型上从未出现。",
      'int-cast-error': "微弱但真实：J-lens 在 ℓ18 为非法的 int('hello') 转换把泛化的 'error' 概念抬到 rank 1（logit lens 毫无显示）。从不是论文中大模型显示的 'ValueError' 或 'invalid'。",
      'eiffel-capital': "微弱但真实：J-lens 在 ℓ18 附近以 rank 5–6 浮现出未说出口的答案 'Paris'，而模型真实输出是退化的 '1'；法国这一跳桥概念从未出现。logit lens 毫无显示。",
    } as Record<string, string>,
  },
} as const;
```

---

## 7. File Structure

| Action | File · anchor | Change |
|---|---|---|
| **create** | `demo/jspace/starters/gallery.ts` | `GalleryEntry` type + `GALLERY[]` (7 vetted entries) + `STARTER_SLUGS` derived |
| **create** | `demo/jspace/starters/<slug>.json` ×8 | baked model-free frames incl. resolved `pinIds` (from `bake.mts`) |
| **create** | `demo/jspace/starters/vetting.json` | honesty artifact (§5) |
| **create** | `demo/jspace/argmax-tint.ts` *(or export in-file)* | pure `tintAlphaForRank` + `ALPHA_MAX/MIN` |
| **create** | `demo/jspace/argmax-tint.test.ts` | golden opacity test (§8) |
| **modify** | `scripts/jlens/vet-candidates.mts` | run the **full** candidate roster (all attempted phenomena) through the 3-gate rubric → graded `vet-candidates-results.json` (raw; feeds bake). Does **not** write `vetting.json`. |
| **modify** | `scripts/jlens/bake.mts` | iterate `GALLERY`, resolve `pinTokens→pinIds`, emit `<slug>.json` (8 shipped frames) **and** `vetting.json` (one row per attempted candidate — shipped + dropped) from the graded raw results |
| **modify** | `demo/jspace/ArgmaxGridCanvas.tsx:205–211` | C: top-K scan + `tintAlphaForRank` |
| **modify** | `demo/jspace/ArgmaxGridCanvas.tsx:72–89, 261–265` | D2: `band` prop, accent onset/peak labels + 3px strip, add to deps |
| **modify** | `demo/jspace/JSpaceApp.tsx:77 (+import)` | `readStoredLocale` → `locale` state; module-scope `JSPACE_COPY` |
| **modify** | `demo/jspace/JSpaceApp.tsx:~471` | `galleryEntry`, memoized per-entry `band`, D1 blurb `<p>` |
| **modify** | `demo/jspace/JSpaceApp.tsx:658–677` | B: `openStarter()` (auto-pin + defaultMode); always-on title+hook cards; delete gated `SegmentedToggle` |
| **modify** | `demo/jspace/JSpaceApp.tsx:747–755` | pass `band={band}` to `<ArgmaxGridCanvas>` |
| **untouched** | `jlens-core/jacobian-presets.ts` | core instrument; `french-season` blurb/`BAND` reused as-is |
| **untouched** | `src/mlx-worker.ts`, `packages/core/index.d.ts` | no worker/type changes; `LensCell.topKIds` already exists |

---

## 8. Verification

**8.1 Golden opacity unit test (`argmax-tint.test.ts`)** — pure, no DOM/model:

| case | expect |
|---|---|
| `tintAlphaForRank(0, 10)` | `0.18` (argmax byte-identical to legacy) |
| `tintAlphaForRank(9, 10)` | `0.045` (tail) |
| `tintAlphaForRank(0, 1)` | `0.18` (K=1 guard, no div-by-zero) |
| `tintAlphaForRank(k, 10)` for k=1..8 | strictly decreasing, linear, `≈ 0.18 - 0.015·k` |

**8.2 Model-free gallery load** — with wasm/model **blocked** (network offline or model fetch stubbed), Chrome-load `/jspace`: all 7 tiles render, `openStarter` on each loads a baked frame, the grid draws with (C) concept tints and (D2) band strip, no model download, no console errors. Proves the gallery is genuinely model-free.

**8.3 Live Chrome verify, both locales** (use Chrome MCP per `reference_chrome_mcp_frontend`; subagent smoke reports are unreliable):
- **en:** fresh visit → gallery visible cold *and* after typing a custom prompt (persistence). Click `arith-inner-sum` → mode=jacobian, pin '3' threads the grid, band strip on ℓ15–18, ℓ18 label accented, blurb caption reads the en string.
- **zh:** set `localStorage['mlx:preferences:locale']='zh'`, reload → names/hooks/blurb/`bandNote` flip to zh; glossary terms (logit lens, Jacobian, ℓ, rank) stay English.
- **argmax regression:** a tile where the pin *is* the argmax at some cell shows that cell at the pre-change 0.18 tint (visually unchanged from today).
- **gate isolation:** switch a starter to logit mode → band strip disappears (D2 gated on jacobian); run a live custom prompt → no tile highlighted (launcher, not mirror), no band.

**8.4 Artifact ↔ caption non-drift** — re-run the Task-A runner; assert `vetting.json` `peakLayer`/`bandLayers` for each shipped slug equal the `GALLERY[].band` and the layer numbers referenced in each caption. Any mismatch fails the build.

**8.5 Honesty lint (review gate)** — grep every en/zh caption + hook + blurb for banned tokens ("thinks/plans/reasons/knows/proves"), for any Sonnet-only chain (Mars/France-hop/ValueError/injection) presented as reproduced, and confirm each WEAK tile's caption contains an explicit weakness statement ("faint"/"微弱"/"honest caveat"/"诚实提醒"). Attribution to the source paper appears exactly once in the gallery.

**8.6 Adversarial review** — after the feature lands, run `/codex:adversarial-review` in the background (per CLAUDE.md); independently verify each finding via subagents tracing the full control flow before applying fixes.

import * as React from 'react';

import { TopKBars, cleanupTokenText } from '../inspector/TopKBars';
import { SegmentedToggle } from '../scaffolding/SegmentedToggle';

/**
 * Chapter 12 (KV cache & hybrid attention) — the generation loop with KV
 * reuse, plus a cache-vs-no-cache cost contrast. Self-contained: NO worker,
 * NO model, NO WASM. A scripted prompt is prefilled in parallel, then a few
 * tokens are decoded one at a time; per step we show the next-token
 * distribution (reused TopKBars) and the per-step attention work.
 *
 * The cache toggle is the lesson:
 *   - ON  → each decode step (re)processes only the ONE new token (the prefix's
 *           K/V is read from cache, not recomputed). Positions processed per step
 *           stay flat at 1. (The new token still attends over the whole cached
 *           prefix, so the attention read itself still grows with context.)
 *   - OFF → each decode step re-processes the entire prefix plus everything
 *           generated so far. Positions processed GROW, so total work is quadratic.
 *
 * This is deliberately distinct from the in-chapter `DecodeAnimation` in the
 * Try-It panel (which draws bespoke SVG attention arcs and has no cache
 * toggle): here it's a token strip + per-step TopKBars + a per-step work bar.
 */

// Scripted sequence. Prompt is prefilled once; the rest is decoded one token
// at a time. (Mirrors the chapter's DECODE_* script so the two views agree.)
const PROMPT: ReadonlyArray<string> = ['The', ' cat', ' sat', ' on', ' the'];
const GENERATED: ReadonlyArray<string> = [' mat', '.', ' It'];

// Per-decode-step next-token candidates; index 0 is the sampled (greedy) token.
type Cand = { text: string; prob: number };
const DECODE_STEPS: ReadonlyArray<{ cands: Cand[] }> = [
  {
    cands: [
      { text: ' mat', prob: 0.42 },
      { text: ' floor', prob: 0.19 },
      { text: ' rug', prob: 0.12 },
      { text: ' couch', prob: 0.08 },
      { text: ' bed', prob: 0.05 },
    ],
  },
  {
    cands: [
      { text: '.', prob: 0.54 },
      { text: ',', prob: 0.18 },
      { text: ' and', prob: 0.1 },
      { text: ' beside', prob: 0.05 },
      { text: ' near', prob: 0.04 },
    ],
  },
  {
    cands: [
      { text: ' It', prob: 0.31 },
      { text: ' The', prob: 0.17 },
      { text: ' A', prob: 0.09 },
      { text: ' She', prob: 0.07 },
      { text: ' Then', prob: 0.05 },
    ],
  },
];

const PROMPT_LEN = PROMPT.length;
// Frame 0 = prefill, frames 1..GENERATED.length = decode steps.
const TOTAL_FRAMES = 1 + GENERATED.length;
const FRAME_MS = 1300;

const prefersReducedMotion = (): boolean =>
  typeof window !== 'undefined' ? window.matchMedia('(prefers-reduced-motion: reduce)').matches : false;

export function GenerationLoopTrace() {
  // Animation guard: don't auto-play under prefers-reduced-motion.
  const [playing, setPlaying] = React.useState<boolean>(() => !prefersReducedMotion());
  const [frame, setFrame] = React.useState(0);
  const [cacheOn, setCacheOn] = React.useState(true);

  React.useEffect(() => {
    if (!playing) return undefined;
    const id = window.setInterval(() => setFrame((f) => (f + 1) % TOTAL_FRAMES), FRAME_MS);
    return () => window.clearInterval(id);
  }, [playing]);

  const isPrefill = frame === 0;
  const decodeIdx = frame - 1; // valid only when !isPrefill

  // Tokens generated so far (newest flashes). During prefill none exist yet.
  const generatedCount = isPrefill ? 0 : decodeIdx + 1;
  const generated = GENERATED.slice(0, generatedCount);
  const flashIndex = isPrefill ? -1 : decodeIdx;

  // Sequence length already committed BEFORE this decode step's new token:
  // prompt + tokens generated in prior steps.
  const priorLen = PROMPT_LEN + (isPrefill ? 0 : decodeIdx);

  // Per-step attention work (number of token positions processed this step):
  //   cache ON  → just the 1 new token (prefix K/V reused).
  //   cache OFF → re-process the whole prefix + the new token.
  const workThisStep = isPrefill ? PROMPT_LEN : cacheOn ? 1 : priorLen + 1;
  // Worst-case work for the bar's full width: no-cache on the LAST decode step.
  const maxWork = PROMPT_LEN + GENERATED.length;
  const workPct = (workThisStep / maxWork) * 100;

  // Current decode step's distribution (prefill shows the first step's bars,
  // dimmed via sampledTokenId = -1 so nothing is "kept" yet).
  const stepForBars = isPrefill ? 0 : decodeIdx;
  const cands = DECODE_STEPS[stepForBars].cands;
  const ids = cands.map((_, i) => i);
  const probs = cands.map((c) => c.prob);
  const texts = cands.map((c) => c.text);
  const sampledTokenId = isPrefill ? -1 : 0;

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          The generation loop — KV reuse vs recompute
        </div>
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          aria-pressed={playing}
          className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
        >
          {playing ? '❚❚ Pause' : '▶ Play'}
        </button>
      </div>

      {/* Cache toggle. */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[11px] uppercase tracking-wider text-muted-foreground">KV cache</span>
        <SegmentedToggle
          value={cacheOn}
          onChange={setCacheOn}
          ariaLabel="KV cache toggle"
          options={[
            { value: true, label: 'on' },
            { value: false, label: 'off' },
          ]}
        />
        <span className="text-[11px] text-muted-foreground">
          {cacheOn
            ? 'prefix K/V reused — no re-projecting the prefix each step'
            : 'prefix re-processed every step — quadratic total'}
        </span>
      </div>

      {/* Token strip: wide/dim prompt (prefilled in parallel) + decode chips. */}
      <div className="flex flex-wrap items-center gap-1 rounded-md border border-border/60 bg-muted/30 p-2">
        <span className="mr-1 text-[10px] uppercase tracking-wider text-muted-foreground">prefill</span>
        {PROMPT.map((t, i) => (
          <span
            key={`p-${i}`}
            className={[
              'rounded px-2 py-0.5 font-mono text-[11px] tracking-wide transition-colors',
              isPrefill ? 'bg-primary/15 text-primary ring-1 ring-primary/40' : 'bg-background text-muted-foreground',
            ].join(' ')}
          >
            {cleanupTokenText(t)}
          </span>
        ))}
        <span className="mx-1 text-muted-foreground" aria-hidden>
          →
        </span>
        <span className="mr-1 text-[10px] uppercase tracking-wider text-muted-foreground">decode</span>
        {generated.map((t, i) => (
          <span
            key={`g-${i}`}
            className={[
              'rounded px-1.5 py-0.5 font-mono text-[11px] transition-colors',
              i === flashIndex
                ? 'bg-primary/20 text-primary ring-1 ring-primary/40'
                : 'bg-primary/10 text-foreground/90',
            ].join(' ')}
          >
            {cleanupTokenText(t)}
          </span>
        ))}
        <span className="ml-0.5 font-mono text-[11px] text-muted-foreground" aria-hidden>
          ▮
        </span>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        {/* Per-step work meter — the heart of the cache contrast. */}
        <div className="space-y-2">
          <div className="flex items-baseline justify-between text-[11px]">
            <span className="text-muted-foreground">
              {isPrefill ? 'Prefill (one parallel pass)' : `Decode step ${decodeIdx + 1} / ${GENERATED.length}`}
            </span>
            <span className="font-mono text-foreground/85">positions (re)processed: {workThisStep}</span>
          </div>
          <div
            className="relative h-6 w-full overflow-hidden rounded-sm bg-muted/60"
            role="img"
            aria-label={`Token positions (re)processed this step: ${workThisStep} of up to ${maxWork}`}
          >
            <div
              className="absolute inset-y-0 left-0 transition-all"
              style={{
                width: `${Math.max(2, Math.min(100, workPct)).toFixed(1)}%`,
                backgroundColor: isPrefill ? '#94a3b8' : cacheOn ? '#22c55e' : '#ef4444',
              }}
            />
          </div>
          <p className="text-[11px] leading-relaxed text-muted-foreground">
            {isPrefill
              ? `Prefill processes all ${PROMPT_LEN} prompt tokens together and writes their K/V into the cache.`
              : cacheOn
                ? `Cache ON: only the 1 new token's K/V is computed; the ${priorLen}-token prefix is read from cache, not recomputed — positions (re)processed stay flat at 1. (The new token still attends over all ${priorLen} cached keys, so that read still grows with context.)`
                : `Cache OFF: the model re-processes all ${priorLen} prior tokens plus the new one (${workThisStep} total). The bar grows every step — that is the quadratic blow-up.`}
          </p>
        </div>

        {/* Current decode step's next-token distribution. */}
        <div className="space-y-1">
          <div className="text-[11px] text-muted-foreground">
            {isPrefill ? 'Next-token scores (first decode step, not sampled yet)' : 'Next token → keep the top bar'}
          </div>
          <TopKBars ids={ids} probs={probs} texts={texts} sampledTokenId={sampledTokenId} runKey={frame + 1} />
        </div>
      </div>

      <p className="text-[12px] text-foreground/85">
        The KV cache is why decode stays affordable. With it (green), each step computes Q/K/V for just the single new
        token instead of re-projecting the whole prefix — so the positions it (re)processes stay flat at 1, and across{' '}
        <code>n</code> tokens that is linear rather than quadratic. The new token still attends over every cached key,
        so that attention read does grow with context — the cache removes the prefix re-projection, not the growing
        attention. Without it (red), every step re-runs the entire prefix from scratch, so the positions processed climb
        each step and the total work is quadratic in the sequence length.
      </p>

      <p className="text-[10px] text-muted-foreground">
        Scripted to show the loop&apos;s shape and the cache-vs-recompute cost — not live output from the model.
      </p>
    </div>
  );
}

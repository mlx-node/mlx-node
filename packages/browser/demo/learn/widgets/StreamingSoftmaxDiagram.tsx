import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * StreamingSoftmaxDiagram — chapter 4 deep-dive, the "online softmax" precursor
 * to FlashAttention.
 *
 * Self-contained: NO worker, NO model, NO WASM — pure SVG + React state, so it
 * server-renders for crawlers (createRoot replaces it on boot) and is robust
 * while the model downloads.
 *
 * It teaches the streaming / "online" softmax trick — the seed both flash
 * precursors share. A softmax over a row needs two things: the row max (for
 * numerical stability, so e^x can't overflow) and the sum of e^(score − max).
 * Naïvely that is TWO passes over the row — pass 1 finds the global max, pass 2
 * sums the exponentials. Streaming softmax does it in ONE pass by keeping a
 * running max m and a running sum ℓ: when a later chunk holds a bigger value,
 * it RESCALES the running sum (and a running weighted-value accumulator o) by
 * e^(m_old − m_new) before folding in the new chunk. m, ℓ, o stay exact after
 * every chunk, so you never revisit earlier scores.
 *
 * Illustrates arXiv 1805.02867 (Milakov & Gimelshein, online softmax, 2018) and
 * how arXiv 2112.05682 (Rabe & Staats, memory-efficient attention, 2021) walks
 * the row in chunks. m, ℓ, o are computed IN-RENDER from the scores array with
 * Math.exp, so the panel and the cells can never disagree.
 *
 * Animation model: a small set of discrete frames (one chunk consumed each),
 * advanced by a setInterval inside a guarded useEffect. Honors
 * prefers-reduced-motion (jumps straight to the final, fully-processed frame,
 * no autoplay).
 */

// ── Fixed illustrative row, split into 4 chunks of 3 ─────────────────────────
const SCORES = [2, 1, 3, 0, 5, 1, 4, 2, 1, 3, 6, 2];
const CHUNK_SIZE = 3;
const CHUNK_COUNT = SCORES.length / CHUNK_SIZE; // 4
// Per-position "value" v fed into the output accumulator o = Σ wᵢ·vᵢ. Using the
// position index keeps o meaningful (a weighted average of positions) but
// uncluttered.
const VALUES = SCORES.map((_, i) => i);

const FRAME_MS = 1900;

function chunkRange(k: number): [number, number] {
  return [k * CHUNK_SIZE, k * CHUNK_SIZE + CHUNK_SIZE];
}

// ── Running-state snapshot AFTER consuming chunks 0..k (k = -1 → initial) ─────
type State = {
  m: number; // running max
  l: number; // running sum of e^(score − m)
  o: number; // running Σ e^(score − m)·v
  /** Multiplier e^(m_old − m_new) applied this step; null on the initial frame. */
  mult: number | null;
  /** True when this chunk pushed the running max up — the rescale moment. */
  rescaled: boolean;
  /** Local max of the chunk just consumed; null on the initial frame. */
  localMax: number | null;
};

/** Compute the running state after consuming chunks 0..upTo (upTo = -1 → start). */
function stateAfter(upTo: number): State {
  let m = -Infinity;
  let l = 0;
  let o = 0;
  let mult: number | null = null;
  let rescaled = false;
  let localMax: number | null = null;
  for (let k = 0; k <= upTo; k++) {
    const [s, e] = chunkRange(k);
    localMax = Math.max(...SCORES.slice(s, e));
    const mNew = Math.max(m, localMax);
    const hadMax = Number.isFinite(m);
    rescaled = hadMax && mNew > m;
    mult = hadMax ? Math.exp(m - mNew) : null;
    // Rescale the running accumulators onto the new max before adding new terms.
    const scale = hadMax ? Math.exp(m - mNew) : 0;
    l *= scale;
    o *= scale;
    for (let i = s; i < e; i++) {
      const w = Math.exp(SCORES[i] - mNew);
      l += w;
      o += w * VALUES[i];
    }
    m = mNew;
  }
  return { m, l, o, mult, rescaled, localMax };
}

// Frames: initial (-1), one per chunk (0..3), then a "done" frame that repeats
// the final state. Total = 1 + 4 + 1 = 6.
const FRAME_CHUNKS: number[] = [-1, 0, 1, 2, 3, 3];
const DONE_FRAME = FRAME_CHUNKS.length - 1;
const TOTAL_FRAMES = FRAME_CHUNKS.length;

function fmt(x: number): string {
  if (!Number.isFinite(x)) return '—';
  return x.toFixed(2);
}

// ── Per-locale copy. COPY.en strings are the original English, verbatim. ─────
const COPY = {
  en: {
    title: 'Streaming softmax — the whole row in one pass',
    pause: '❚❚ Pause',
    play: '▶ Play',
    step: 'Step ›',
    naiveChip: (
      <>
        naïve softmax: <span className="font-mono text-foreground/80">2 passes</span> over the row
      </>
    ),
    streamChip: (
      <>
        streaming: <span className="font-mono">1 pass</span>
      </>
    ),
    captionInitial: (
      <>
        Start empty: running max <span className="font-mono">m = −∞</span>, running sum{' '}
        <span className="font-mono">ℓ = 0</span>. We will walk the row left to right, one chunk of {CHUNK_SIZE} at a time —
        <strong> a single pass</strong>.
      </>
    ),
    captionDone: (val: string) => (
      <>
        Done in <strong>one pass</strong>. The softmax-weighted output is{' '}
        <span className="font-mono">o / ℓ = {val}</span> — identical to the two-pass recipe, but we never
        revisited an earlier score.
      </>
    ),
    captionRescaled: (chunkNo: number, localMax: number | null, mult: string) => (
      <>
        Chunk {chunkNo} holds a bigger value (local max <span className="font-mono">{localMax}</span> &gt; old{' '}
        <span className="font-mono">m</span>). <strong>Rescale first:</strong> multiply ℓ and o by{' '}
        <span className="font-mono text-primary">
          e<sup>m_old − m_new</sup> = {mult}
        </span>
        , then fold in this chunk&apos;s e<sup>score − m</sup> terms.
      </>
    ),
    captionNoRescale: (chunkNo: number, localMax: number | null, mStr: string) => (
      <>
        Chunk {chunkNo}: its local max <span className="font-mono">{localMax}</span> does not beat{' '}
        <span className="font-mono">m = {mStr}</span>, so <strong>no rescale</strong> (multiplier = 1) — just add this
        chunk&apos;s e<sup>score − m</sup> terms to ℓ and o.
      </>
    ),
    ariaLabel:
      'Diagram of streaming softmax walking one row of scores in four chunks, keeping a running max m, running sum of exponentials ℓ, and a running weighted-value accumulator o, rescaling them by e to the power m_old minus m_new whenever a chunk raises the max — all in a single pass.',
    chunkLabel: (n: number) => `chunk ${n}`,
    rowLabel: 'one attention row · scores',
    runningMax: 'running max',
    runningSum: (
      <>
        running sum of e<tspan dy={-3} className="text-[7px]">score − m</tspan>
      </>
    ),
    outputAcc: (
      <>
        output accumulator  o = Σ e<tspan dy={-3} className="text-[7px]">score − m</tspan>
        <tspan dy={3} className="text-[10px]">·v</tspan>
      </>
    ),
    answerSoFar: 'answer so far  ',
    rescaleBadge: (mult: string) => `rescale ×${mult}`,
    chunkNotStarted: (
      <>
        chunk <span className="font-mono text-foreground/90">0 / {CHUNK_COUNT}</span> · not started
      </>
    ),
    chunkProgress: (cur: number) => (
      <>
        chunk <span className="font-mono text-foreground/90">{cur} / {CHUNK_COUNT}</span>
      </>
    ),
    runningFooter: (mStr: string, lStr: string) => (
      <>
        running <span className="font-mono text-primary">m = {mStr}</span>
        {'  ·  '}
        running <span className="font-mono text-primary">ℓ = {lStr}</span>
      </>
    ),
    footnote:
      "This single-pass rescale is the seed FlashAttention later runs entirely inside the GPU's tiny on-chip memory (SRAM) — so the N×N score matrix never has to be written out to slow main memory.",
  },
  zh: {
    title: '流式 softmax——一遍扫完整行',
    pause: '❚❚ 暂停',
    play: '▶ 播放',
    step: '单步 ›',
    naiveChip: (
      <>
        朴素 softmax：<span className="font-mono text-foreground/80">2 遍</span>扫过整行
      </>
    ),
    streamChip: (
      <>
        流式：<span className="font-mono">1 遍</span>
      </>
    ),
    captionInitial: (
      <>
        从空状态开始：滚动最大值 <span className="font-mono">m = −∞</span>，滚动和{' '}
        <span className="font-mono">ℓ = 0</span>。我们将从左到右扫过整行，每次一个 {CHUNK_SIZE} 元素的块——
        <strong>只扫一遍</strong>。
      </>
    ),
    captionDone: (val: string) => (
      <>
        <strong>一遍</strong>扫完。softmax 加权输出是 <span className="font-mono">o / ℓ = {val}</span>
        ——与两遍做法完全一致，但我们从未回头重读任何分数。
      </>
    ),
    captionRescaled: (chunkNo: number, localMax: number | null, mult: string) => (
      <>
        块 {chunkNo} 出现了更大的值（局部最大值 <span className="font-mono">{localMax}</span> &gt; 旧的{' '}
        <span className="font-mono">m</span>）。<strong>先重缩放：</strong>把 ℓ 和 o 乘以{' '}
        <span className="font-mono text-primary">
          e<sup>m_old − m_new</sup> = {mult}
        </span>
        ，再并入这一块的 e<sup>score − m</sup> 项。
      </>
    ),
    captionNoRescale: (chunkNo: number, localMax: number | null, mStr: string) => (
      <>
        块 {chunkNo}：它的局部最大值 <span className="font-mono">{localMax}</span> 没有超过{' '}
        <span className="font-mono">m = {mStr}</span>，所以<strong>不重缩放</strong>（乘数 = 1）——直接把这一块的 e
        <sup>score − m</sup> 项加进 ℓ 和 o。
      </>
    ),
    ariaLabel:
      '流式 softmax 示意图：按四个块扫过一行分数，维护滚动最大值 m、指数的滚动和 ℓ 与滚动加权值累加器 o，每当某块抬高最大值时按 e^(m_old − m_new) 重缩放——全程只扫一遍。',
    chunkLabel: (n: number) => `块 ${n}`,
    rowLabel: '一行注意力 · 分数',
    runningMax: '滚动最大值',
    runningSum: (
      <>
        e<tspan dy={-3} className="text-[7px]">score − m</tspan>
        <tspan dy={3} className="text-[10px]"> 的滚动和</tspan>
      </>
    ),
    outputAcc: (
      <>
        输出累加器  o = Σ e<tspan dy={-3} className="text-[7px]">score − m</tspan>
        <tspan dy={3} className="text-[10px]">·v</tspan>
      </>
    ),
    answerSoFar: '当前答案  ',
    rescaleBadge: (mult: string) => `重缩放 ×${mult}`,
    chunkNotStarted: (
      <>
        块 <span className="font-mono text-foreground/90">0 / {CHUNK_COUNT}</span> · 尚未开始
      </>
    ),
    chunkProgress: (cur: number) => (
      <>
        块 <span className="font-mono text-foreground/90">{cur} / {CHUNK_COUNT}</span>
      </>
    ),
    runningFooter: (mStr: string, lStr: string) => (
      <>
        滚动 <span className="font-mono text-primary">m = {mStr}</span>
        {'  ·  '}
        滚动 <span className="font-mono text-primary">ℓ = {lStr}</span>
      </>
    ),
    footnote:
      '这个单遍重缩放，正是后来 FlashAttention 完全在 GPU 微小片上内存（SRAM）里运行的种子——因此 N×N 分数矩阵永远不必写出到慢速主内存。',
  },
} as const;

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = React.useState<boolean>(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return false;
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  });
  React.useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return;
    const mql = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, []);
  return reduced;
}

// ── SVG geometry ────────────────────────────────────────────────────────────
const VB_W = 760;
const VB_H = 232;
const CELL_W = 46;
const CELL_H = 40;
const CELL_GAP = 5;
const CHUNK_GAP = 22;
const ROW_X = 24;
const ROW_Y = 40;

function cellX(i: number): number {
  const chunk = Math.floor(i / CHUNK_SIZE);
  const within = i % CHUNK_SIZE;
  return ROW_X + chunk * (CHUNK_SIZE * (CELL_W + CELL_GAP) + CHUNK_GAP) + within * (CELL_W + CELL_GAP);
}

// Running-state panel boxes.
const PANEL_Y = 130;
const PANEL_H = 74;
const PANEL = {
  m: { x: 24, w: 150 },
  l: { x: 196, w: 220 },
  o: { x: 438, w: 298 },
};

export function StreamingSoftmaxDiagram() {
  const copy = COPY[useLocale()];
  const reducedMotion = usePrefersReducedMotion();
  // Under reduced motion, start on the done frame and never auto-play.
  const [frame, setFrame] = React.useState(reducedMotion ? DONE_FRAME : 0);
  const [playing, setPlaying] = React.useState(!reducedMotion);

  React.useEffect(() => {
    if (!playing || reducedMotion) return;
    const t = window.setInterval(() => setFrame((f) => (f + 1) % TOTAL_FRAMES), FRAME_MS);
    return () => window.clearInterval(t);
  }, [playing, reducedMotion]);

  const chunkIdx = FRAME_CHUNKS[frame]; // -1 .. 3
  const isInitial = chunkIdx < 0;
  const isDone = frame === DONE_FRAME;
  const st = stateAfter(chunkIdx);
  // Flash the panel on the rescale moment (only on the live chunk frame, not the
  // repeated done frame, and never under reduced motion).
  const flashing = st.rescaled && !isDone && !reducedMotion;

  const step = () => {
    setPlaying(false);
    setFrame((x) => (x + 1) % TOTAL_FRAMES);
  };

  // Plain-language caption per frame.
  let caption: React.ReactNode;
  if (isInitial) {
    caption = copy.captionInitial;
  } else if (isDone) {
    caption = copy.captionDone(fmt(st.o / st.l));
  } else if (st.rescaled) {
    caption = copy.captionRescaled(chunkIdx + 1, st.localMax, fmt(st.mult ?? 0));
  } else {
    caption = copy.captionNoRescale(chunkIdx + 1, st.localMax, fmt(st.m));
  }

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title + controls */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>
        <div className="flex items-center gap-1">
          {!reducedMotion ? (
            <>
              <button
                type="button"
                onClick={() => setPlaying((p) => !p)}
                aria-pressed={playing}
                className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              >
                {playing ? copy.pause : copy.play}
              </button>
              <button
                type="button"
                onClick={step}
                className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              >
                {copy.step}
              </button>
            </>
          ) : null}
        </div>
      </div>

      {/* Pass-count contrast chips (static) */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1.5 rounded border border-border bg-muted px-2 py-0.5 text-[11px] text-muted-foreground">
          {copy.naiveChip}
        </span>
        <span className="inline-flex items-center gap-1.5 rounded border border-primary bg-primary/10 px-2 py-0.5 text-[11px] text-primary">
          {copy.streamChip}
        </span>
      </div>

      {/* The diagram */}
      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        className="w-full"
        role="img"
        aria-label={copy.ariaLabel}
      >
        {/* ── The row of scores, grouped into 4 chunks ── */}
        {SCORES.map((s, i) => {
          const chunk = Math.floor(i / CHUNK_SIZE);
          const isCurrent = !isInitial && chunk === chunkIdx;
          const isConsumed = !isInitial && chunk < chunkIdx;
          const x = cellX(i);
          return (
            <g key={i} style={{ opacity: isInitial ? 0.55 : isCurrent ? 1 : isConsumed ? 0.85 : 0.4 }}>
              <rect
                x={x}
                y={ROW_Y}
                width={CELL_W}
                height={CELL_H}
                rx={6}
                style={{
                  fill: isCurrent ? 'var(--primary)' : 'var(--card)',
                  fillOpacity: isCurrent ? 0.16 : 1,
                  stroke: isCurrent ? 'var(--primary)' : 'var(--border)',
                }}
                strokeWidth={isCurrent ? 2 : 1.5}
              />
              <text
                x={x + CELL_W / 2}
                y={ROW_Y + CELL_H / 2 + 5}
                textAnchor="middle"
                style={{ fill: isCurrent ? 'var(--primary)' : 'var(--foreground)' }}
                className="font-mono text-[14px] font-semibold"
              >
                {s}
              </text>
            </g>
          );
        })}
        {/* chunk labels under each group */}
        {Array.from({ length: CHUNK_COUNT }, (_, k) => {
          const [s] = chunkRange(k);
          const cx = cellX(s) + (CHUNK_SIZE * (CELL_W + CELL_GAP) - CELL_GAP) / 2;
          const isCurrent = !isInitial && k === chunkIdx;
          return (
            <text
              key={k}
              x={cx}
              y={ROW_Y + CELL_H + 16}
              textAnchor="middle"
              style={{ fill: isCurrent ? 'var(--primary)' : 'var(--muted-foreground)' }}
              className="text-[10px] font-medium"
            >
              {copy.chunkLabel(k + 1)}
            </text>
          );
        })}
        <text x={ROW_X} y={ROW_Y - 12} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
          {copy.rowLabel}
        </text>

        {/* ── Running-state panel: m, ℓ, o ── */}
        <g className={flashing ? 'animate-pulse' : undefined}>
          {/* m */}
          <StateBox x={PANEL.m.x} w={PANEL.m.w} flashing={flashing}>
            <text x={PANEL.m.x + 12} y={PANEL_Y + 20} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
              {copy.runningMax}
            </text>
            <text
              x={PANEL.m.x + 12}
              y={PANEL_Y + 46}
              style={{ fill: 'var(--foreground)' }}
              className="font-mono text-[18px] font-semibold"
            >
              m = {Number.isFinite(st.m) ? fmt(st.m) : '−∞'}
            </text>
          </StateBox>

          {/* ℓ */}
          <StateBox x={PANEL.l.x} w={PANEL.l.w} flashing={flashing}>
            <text x={PANEL.l.x + 12} y={PANEL_Y + 20} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
              {copy.runningSum}
            </text>
            <text
              x={PANEL.l.x + 12}
              y={PANEL_Y + 46}
              style={{ fill: 'var(--foreground)' }}
              className="font-mono text-[18px] font-semibold"
            >
              ℓ = {fmt(st.l)}
            </text>
          </StateBox>

          {/* o */}
          <StateBox x={PANEL.o.x} w={PANEL.o.w} flashing={flashing}>
            <text x={PANEL.o.x + 12} y={PANEL_Y + 20} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
              {copy.outputAcc}
            </text>
            <text
              x={PANEL.o.x + 12}
              y={PANEL_Y + 46}
              style={{ fill: 'var(--foreground)' }}
              className="font-mono text-[18px] font-semibold"
            >
              o = {fmt(st.o)}
            </text>
            <text x={PANEL.o.x + 12} y={PANEL_Y + 64} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
              {copy.answerSoFar}
              <tspan className="font-mono" style={{ fill: 'var(--foreground)' }}>
                o / ℓ = {st.l > 0 ? fmt(st.o / st.l) : '—'}
              </tspan>
            </text>
          </StateBox>

          {/* Rescale badge — only on a real rescale frame */}
          {flashing ? (
            <g>
              <rect
                x={PANEL.l.x + PANEL.l.w / 2 - 86}
                y={PANEL_Y - 26}
                width={172}
                height={20}
                rx={10}
                style={{ fill: 'var(--primary)', stroke: 'var(--primary)' }}
                strokeWidth={1}
              />
              <text
                x={PANEL.l.x + PANEL.l.w / 2}
                y={PANEL_Y - 12}
                textAnchor="middle"
                style={{ fill: 'var(--bg)' }}
                className="font-mono text-[10px] font-semibold"
              >
                {copy.rescaleBadge(fmt(st.mult ?? 0))}
              </text>
            </g>
          ) : null}
        </g>
      </svg>

      {/* Caption — what this chunk does */}
      <p className="min-h-[3.25rem] text-[13px] text-foreground/90">{caption}</p>

      {/* Live counters */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border/60 pt-3">
        <div className="text-[11px] text-muted-foreground">
          {isInitial ? copy.chunkNotStarted : copy.chunkProgress(Math.min(chunkIdx + 1, CHUNK_COUNT))}
        </div>
        <div className="text-[11px] text-muted-foreground">
          {copy.runningFooter(Number.isFinite(st.m) ? fmt(st.m) : '−∞', fmt(st.l))}
        </div>
      </div>

      <p className="text-[10px] text-muted-foreground/70">{copy.footnote}</p>
    </div>
  );
}

/** One running-state box (m / ℓ / o). Highlighted while the panel flashes. */
function StateBox({
  x,
  w,
  flashing,
  children,
}: {
  x: number;
  w: number;
  flashing: boolean;
  children: React.ReactNode;
}) {
  return (
    <g>
      <rect
        x={x}
        y={PANEL_Y}
        width={w}
        height={PANEL_H}
        rx={8}
        style={{
          fill: flashing ? 'var(--primary)' : 'var(--card)',
          fillOpacity: flashing ? 0.1 : 1,
          stroke: flashing ? 'var(--primary)' : 'var(--border)',
        }}
        strokeWidth={flashing ? 2 : 1.5}
      />
      {children}
    </g>
  );
}

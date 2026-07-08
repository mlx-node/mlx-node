import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * MemoryWallDiagram — chapter 4 deep-dive "The memory wall".
 *
 * Self-contained: NO worker, NO model, NO WASM — pure SVG + React state, so it
 * server-renders for crawlers (createRoot replaces it on boot) and is robust
 * while the model downloads.
 *
 * It teaches WHY attention is "memory-bound, not compute-bound" by animating
 * the physical bottleneck: the N×N score matrix is too big for the GPU's tiny
 * fast on-chip memory (SRAM), so the naïve attention recipe parks it in the big
 * but far-away main memory (HBM) and drags it back and forth across the slow
 * memory bus four times — once to write the scores, again to read them for
 * softmax, again to write the normalized scores, and a fourth time to read them
 * to weight V. A running "moved over the bus" counter and an N selector make
 * the quadratic (N²) blow-up tangible.
 *
 * Animation model: a small set of discrete frames (one bus crossing each). The
 * packet cluster slides between the two memory boxes via a CSS transform
 * transition, so no global keyframes are needed. Honors prefers-reduced-motion
 * (jumps straight to the final tally, no motion).
 */

// ── Sequence-length presets — matrix bytes grow with N². bf16 = 2 bytes/score,
// counted PER HEAD (the chapter prose uses ~268 MB for one 8-head layer at
// N=4,096, i.e. 33.6 MB/head × 8). ──────────────────────────────────────────
type SeqOption = { label: string; n: number };
const SEQ_OPTIONS: SeqOption[] = [
  { label: '1K', n: 1024 },
  { label: '4K', n: 4096 },
  { label: '16K', n: 16384 },
];
const HEADS = 8; // Qwen3.5-0.8B query heads (full-attention layers).
const BYTES_PER_SCORE = 2; // bf16

function matrixBytesPerHead(n: number): number {
  return n * n * BYTES_PER_SCORE;
}

function fmtBytes(b: number): string {
  if (b >= 1e9) return `${(b / 1e9).toFixed(2)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(1)} MB`;
  if (b >= 1e3) return `${(b / 1e3).toFixed(1)} KB`;
  return `${b} B`;
}

function fmtInt(n: number): string {
  return n.toLocaleString('en-US');
}

// ── Frame script. Each frame is one snapshot; frames 1–4 are the four bus
// crossings of the N×N matrix; frame 0 is the "just computed" start; frame 5 is
// the tally. ────────────────────────────────────────────────────────────────
type Frame = {
  /** Where the matrix packets sit AFTER this frame's movement. */
  at: 'hbm' | 'sram';
  /** Cumulative bus crossings of the N×N matrix up to and including this frame. */
  crossings: number;
  /** Pulse the SRAM compute core (a math step happens here). */
  compute: boolean;
  /** The tally frame shows the time-split bar instead of a live caption. */
  tally?: boolean;
};

// Captions live in the per-locale COPY record below, index-aligned with FRAMES.
const FRAMES: Frame[] = [
  { at: 'sram', crossings: 0, compute: true },
  { at: 'hbm', crossings: 1, compute: false },
  { at: 'sram', crossings: 2, compute: true },
  { at: 'hbm', crossings: 3, compute: false },
  { at: 'sram', crossings: 4, compute: true },
  { at: 'sram', crossings: 4, compute: false, tally: true },
];
const TOTAL_FRAMES = FRAMES.length;
const FRAME_MS = 1500;

// ── Per-locale copy. COPY.en strings are the original English, verbatim. ─────
const COPY = {
  en: {
    captions: [
      'Q·Kᵀ is computed in fast SRAM — but the full N×N score matrix is far too big to keep there.',
      '1 · Write the whole N×N score matrix out to slow HBM.',
      '2 · Read it all back into SRAM to run softmax.',
      '3 · Write the normalized N×N scores back out to HBM.',
      '4 · Read it a third time to weight V — and out comes the result.',
      'The N×N matrix crossed the slow bus ~4 times. Almost all the time went to moving it, not computing.',
    ],
    title: 'The memory wall — why attention is bandwidth-bound',
    pause: '❚❚ Pause',
    play: '▶ Play',
    step: 'Step ›',
    ariaLabel:
      'Diagram of the N by N attention score matrix shuttling between slow HBM main memory and fast on-chip SRAM across the memory bus four times.',
    memoryBus: 'memory bus',
    busHint: 'slow — the bottleneck',
    hbmTitle: 'HBM · GPU main memory',
    hbmHint: 'huge — but far away and slow',
    sramTitle: 'SRAM · on-chip',
    sramHint: 'tiny — but ~10× faster',
    math: 'math',
    matrixLabel: 'N×N score matrix',
    seqLen: 'Sequence length',
    tokensSuffix: 'tokens',
    matrixEquals: 'N×N matrix =',
    perHead: 'per head',
    movedSoFar: 'moved over the bus so far:',
    allHeads: (bytes: string, heads: number) => `≈ ${bytes} across all ${heads} heads in the layer`,
    tallyLine: (
      <>
        The N×N matrix crossed the slow bus <strong>~4 times</strong>. The actual arithmetic was a sliver — attention is{' '}
        <strong>memory-bandwidth-bound, not compute-bound</strong>.
      </>
    ),
    movingTheMatrix: 'moving the matrix',
    compute: 'compute',
    tallyNote: 'Illustrative split — attention time is dominated by data movement.',
  },
  zh: {
    captions: [
      'Q·Kᵀ 在快速 SRAM 中算出——但完整的 N×N 分数矩阵远远放不进去。',
      '1 · 把整个 N×N 分数矩阵写出到慢速 HBM。',
      '2 · 把它全部读回 SRAM 来跑 softmax。',
      '3 · 把归一化后的 N×N 分数再写回 HBM。',
      '4 · 第三次读出来给 V 加权——结果就此产生。',
      'N×N 矩阵在慢速总线上往返了约 4 趟。几乎所有时间都花在搬运它，而不是计算。',
    ],
    title: '显存带宽墙——注意力为何受带宽限制',
    pause: '❚❚ 暂停',
    play: '▶ 播放',
    step: '单步 ›',
    ariaLabel: 'N×N 注意力分数矩阵在慢速 HBM 主内存与快速片上 SRAM 之间跨内存总线往返四趟的示意图。',
    memoryBus: '内存总线',
    busHint: '慢——瓶颈所在',
    hbmTitle: 'HBM · GPU 主内存',
    hbmHint: '巨大——但遥远而慢',
    sramTitle: 'SRAM · 片上',
    sramHint: '极小——但快约 10×',
    math: '计算',
    matrixLabel: 'N×N 分数矩阵',
    seqLen: '序列长度',
    tokensSuffix: '个 token',
    matrixEquals: 'N×N 矩阵 =',
    perHead: '每头',
    movedSoFar: '至今搬过总线：',
    allHeads: (bytes: string, heads: number) => `≈ ${bytes}——该层全部 ${heads} 个头合计`,
    tallyLine: (
      <>
        N×N 矩阵在慢速总线上往返了<strong>约 4 趟</strong>。真正的算术只是一个零头——注意力
        <strong>受显存带宽限制，而非计算限制</strong>。
      </>
    ),
    movingTheMatrix: '搬运矩阵',
    compute: '计算',
    tallyNote: '示意性的划分——注意力耗时由数据搬运主导。',
  },
} as const;

// Illustrative time split on the tally frame (NOT a measured figure — labelled
// as such in the UI). Attention is bandwidth-bound, so data movement dominates.
const MOVE_PCT = 88;

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
// One matrix "block" docks in a clearly-marked empty slot inside each memory
// tier and slides between them along the bus — so nothing overlaps.
const VB_W = 720;
const VB_H = 270;
const LANE_Y = 162; // vertical centre the block + bus + docks share (low, below titles)
const HBM = { x: 24, y: 40, w: 272, h: 192 };
const SRAM = { x: 470, y: 70, w: 230, h: 150 };
const BUS = { h: 18, y: LANE_Y - 9, x1: HBM.x + HBM.w, x2: SRAM.x };
// The moving matrix block + its two docking slots.
const BLOCK_W = 96;
const BLOCK_H = 54;
const BLOCK_Y = LANE_Y - BLOCK_H / 2;
const HBM_DOCK_X = HBM.x + HBM.w - BLOCK_W - 26; // rests near the bus, inside HBM
const SRAM_DOCK_X = SRAM.x + 22; // rests on the bus side of SRAM
// The compute zone sits in the free right side of SRAM, clear of the dock.
const COMPUTE_CX = SRAM.x + SRAM.w - 42;

export function MemoryWallDiagram() {
  const copy = COPY[useLocale()];
  const reducedMotion = usePrefersReducedMotion();
  const [seqIdx, setSeqIdx] = React.useState(1); // default N = 4K (matches prose)
  // Under reduced motion, start on the tally frame and never auto-play.
  const [frame, setFrame] = React.useState(reducedMotion ? TOTAL_FRAMES - 1 : 0);
  const [playing, setPlaying] = React.useState(!reducedMotion);

  React.useEffect(() => {
    if (!playing || reducedMotion) return;
    const t = window.setInterval(() => setFrame((f) => (f + 1) % TOTAL_FRAMES), FRAME_MS);
    return () => window.clearInterval(t);
  }, [playing, reducedMotion]);

  const f = FRAMES[frame];
  const seq = SEQ_OPTIONS[seqIdx];
  const perHead = matrixBytesPerHead(seq.n);
  const moved = f.crossings * perHead;

  const dockX = f.at === 'hbm' ? HBM_DOCK_X : SRAM_DOCK_X;
  const matrixInHbm = f.at === 'hbm';

  const step = () => {
    setPlaying(false);
    setFrame((x) => (x + 1) % TOTAL_FRAMES);
  };

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
                className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
              >
                {playing ? copy.pause : copy.play}
              </button>
              <button
                type="button"
                onClick={step}
                className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
              >
                {copy.step}
              </button>
            </>
          ) : null}
        </div>
      </div>

      {/* The diagram */}
      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        className="w-full"
        role="img"
        aria-label={copy.ariaLabel}
      >
        {/* ── Memory bus channel (the bottleneck the block travels) ── */}
        <rect
          x={BUS.x1}
          y={BUS.y}
          width={BUS.x2 - BUS.x1}
          height={BUS.h}
          rx={4}
          style={{ fill: 'var(--muted)', stroke: 'var(--border)' }}
          strokeWidth={1}
        />
        <text
          x={(BUS.x1 + BUS.x2) / 2}
          y={BUS.y - 10}
          textAnchor="middle"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[11px]"
        >
          {copy.memoryBus}
        </text>
        <text
          x={(BUS.x1 + BUS.x2) / 2}
          y={BUS.y + BUS.h + 20}
          textAnchor="middle"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[10px]"
        >
          {copy.busHint}
        </text>

        {/* ── HBM box (big, slow, far) ── */}
        <rect
          x={HBM.x}
          y={HBM.y}
          width={HBM.w}
          height={HBM.h}
          rx={10}
          style={{ fill: 'var(--card)', stroke: 'var(--border)' }}
          strokeWidth={1.5}
        />
        <text x={HBM.x + 16} y={HBM.y + 27} style={{ fill: 'var(--foreground)' }} className="text-[14px] font-semibold">
          {copy.hbmTitle}
        </text>
        <text x={HBM.x + 16} y={HBM.y + 46} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
          {copy.hbmHint}
        </text>
        <DockSlot x={HBM_DOCK_X} occupied={matrixInHbm} />

        {/* ── SRAM box (tiny, fast, on-chip) ── */}
        <rect
          x={SRAM.x}
          y={SRAM.y}
          width={SRAM.w}
          height={SRAM.h}
          rx={10}
          style={{ fill: 'var(--card)', stroke: 'var(--border)' }}
          strokeWidth={1.5}
        />
        <text x={SRAM.x + 14} y={SRAM.y + 25} style={{ fill: 'var(--foreground)' }} className="text-[14px] font-semibold">
          {copy.sramTitle}
        </text>
        <text x={SRAM.x + 14} y={SRAM.y + 43} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
          {copy.sramHint}
        </text>
        <DockSlot x={SRAM_DOCK_X} occupied={!matrixInHbm} />
        {/* compute marker — lights up only on the math frames, clear of the dock */}
        <g className={f.compute ? 'animate-pulse' : undefined}>
          <circle
            cx={COMPUTE_CX}
            cy={LANE_Y}
            r={18}
            style={{ fill: f.compute ? 'var(--primary)' : 'var(--muted)', stroke: 'var(--border)' }}
            strokeWidth={1}
          />
          <text
            x={COMPUTE_CX}
            y={LANE_Y + 3}
            textAnchor="middle"
            style={{ fill: f.compute ? 'var(--bg)' : 'var(--muted-foreground)' }}
            className="text-[9px] font-semibold"
          >
            {copy.math}
          </text>
        </g>

        {/* ── The N×N matrix block, docking in one tier at a time ── */}
        <g
          style={{
            transform: `translateX(${dockX - HBM_DOCK_X}px)`,
            transition: reducedMotion ? undefined : 'transform 900ms cubic-bezier(0.4, 0, 0.2, 1)',
          }}
        >
          <MatrixBlock />
        </g>
      </svg>

      {/* Caption / tally */}
      {f.tally ? (
        <TimeSplitBar />
      ) : (
        <p className="min-h-[2.5rem] text-[13px] text-foreground/90">{copy.captions[frame]}</p>
      )}

      {/* Live counters + N selector */}
      <div className="flex flex-wrap items-end justify-between gap-3 border-t border-border/60 pt-3">
        <div className="space-y-1">
          <div className="text-[11px] text-muted-foreground">
            {copy.seqLen} <span className="font-mono text-foreground/80">N = {fmtInt(seq.n)}</span> {copy.tokensSuffix}
          </div>
          <div className="inline-flex overflow-hidden rounded border border-border">
            {SEQ_OPTIONS.map((o, i) => (
              <button
                key={o.label}
                type="button"
                onClick={() => setSeqIdx(i)}
                aria-pressed={i === seqIdx}
                className={[
                  'px-2.5 py-1 font-mono text-[11px] transition-colors',
                  i === seqIdx ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground',
                ].join(' ')}
              >
                {o.label}
              </button>
            ))}
          </div>
        </div>

        <div className="space-y-1 text-right">
          <div className="text-[11px] text-muted-foreground">
            {copy.matrixEquals}{' '}
            <span className="font-mono text-foreground/90">{fmtBytes(perHead)}</span>{' '}
            <span className="text-muted-foreground/80">{copy.perHead}</span>
          </div>
          <div className="text-[11px] text-muted-foreground">
            {copy.movedSoFar}{' '}
            <span className="font-mono text-primary">{fmtBytes(moved)}</span>{' '}
            <span className="text-muted-foreground/80">{copy.perHead}</span>
          </div>
          <div className="text-[10px] text-muted-foreground/70">{copy.allHeads(fmtBytes(moved * HEADS), HEADS)}</div>
        </div>
      </div>
    </div>
  );
}

/** A dashed docking slot — shows where the matrix block can rest in a tier.
 * Hidden while the block occupies it (the block draws on top). */
function DockSlot({ x, occupied }: { x: number; occupied: boolean }) {
  return (
    <rect
      x={x}
      y={BLOCK_Y}
      width={BLOCK_W}
      height={BLOCK_H}
      rx={8}
      fill="none"
      style={{ stroke: 'var(--border)', opacity: occupied ? 0 : 0.9 }}
      strokeWidth={1.5}
      strokeDasharray="5 5"
    />
  );
}

/** The N×N score matrix as ONE labeled block. Drawn at the HBM dock position;
 * the parent <g> slides it to whichever tier currently holds it. */
function MatrixBlock() {
  const copy = COPY[useLocale()];
  const cols = 5;
  const rows = 3;
  const pad = 12;
  const gx = 4;
  const cell = (BLOCK_W - pad * 2 - (cols - 1) * gx) / cols;
  // Centre the grid vertically inside the block (equal top/bottom padding)
  // instead of a hard-coded top offset.
  const vpad = (BLOCK_H - (rows * cell + (rows - 1) * gx)) / 2;
  return (
    <g>
      <rect
        x={HBM_DOCK_X}
        y={BLOCK_Y}
        width={BLOCK_W}
        height={BLOCK_H}
        rx={8}
        style={{ fill: 'var(--primary)', fillOpacity: 0.16, stroke: 'var(--primary)' }}
        strokeWidth={1.5}
      />
      {Array.from({ length: rows }, (_, r) =>
        Array.from({ length: cols }, (_, c) => (
          <rect
            key={`${r}-${c}`}
            x={HBM_DOCK_X + pad + c * (cell + gx)}
            y={BLOCK_Y + vpad + r * (cell + gx)}
            width={cell}
            height={cell}
            rx={1.5}
            style={{ fill: 'var(--primary)', fillOpacity: 0.85 }}
          />
        )),
      )}
      <text
        x={HBM_DOCK_X + BLOCK_W / 2}
        y={BLOCK_Y + BLOCK_H + 17}
        textAnchor="middle"
        style={{ fill: 'var(--primary)' }}
        className="text-[11px] font-medium"
      >
        {copy.matrixLabel}
      </text>
    </g>
  );
}

/** Illustrative "where the time goes" split shown on the tally frame. */
function TimeSplitBar() {
  const copy = COPY[useLocale()];
  return (
    <div className="space-y-1.5">
      <p className="text-[13px] text-foreground/90">{copy.tallyLine}</p>
      <div className="flex h-5 w-full overflow-hidden rounded border border-border" aria-hidden>
        <div
          className="flex items-center justify-center text-[10px] font-medium"
          style={{ width: `${MOVE_PCT}%`, background: 'var(--primary)', color: 'var(--bg)' }}
        >
          {copy.movingTheMatrix}
        </div>
        <div
          className="flex items-center justify-center text-[10px] text-muted-foreground"
          style={{ width: `${100 - MOVE_PCT}%`, background: 'var(--muted)' }}
        >
          {copy.compute}
        </div>
      </div>
      <p className="text-[10px] text-muted-foreground/70">{copy.tallyNote}</p>
    </div>
  );
}

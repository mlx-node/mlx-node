import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Flash2ParallelismDiagram — chapter 4 deep-dive "FlashAttention-2".
 *
 * Self-contained: NO worker, NO model, NO WASM — pure SVG + React state, so it
 * server-renders for crawlers (createRoot replaces it on boot) and is robust
 * while the model downloads.
 *
 * It teaches WHY FlashAttention-2 (arXiv 2307.08691, Tri Dao, 2023) runs ~2×
 * faster than FlashAttention-1 with the SAME math — purely better GPU mechanics.
 * A GPU has many cores (called SMs); work is launched as many "thread blocks";
 * inside a block, threads run in groups of 32 called "warps"; "occupancy" is how
 * much of the GPU is kept busy. FA2 changes two things:
 *
 *   (a) Sequence parallelism — FA1 launches only batch·heads thread blocks, so
 *       on a small model most cores sit idle. FA2 ALSO splits the query rows into
 *       tiles and gives each its own block → blocks = batch·heads × seq-tiles →
 *       the whole GPU fills up. (This is the same occupancy idea behind why this
 *       model routes single-token decode away from its fused kernel.)
 *   (b) Split-Q instead of split-K — FA1 makes the 4 warps in a block each chew
 *       a slice of the K dimension for ALL rows, so their partials must be merged
 *       through slow shared memory. FA2 gives each warp its own band of Q rows
 *       end-to-end, so warps never talk → no cross-warp shared-memory reduction.
 *
 * A FA1 ↔ FA2 toggle drives two coordinated panels (occupancy grid + warp split).
 * Honors prefers-reduced-motion (renders the FA2 end-state statically, no motion).
 */

type Variant = 'fa1' | 'fa2';

const VARIANTS: { id: Variant; label: string }[] = [
  { id: 'fa1', label: 'FlashAttention-1' },
  { id: 'fa2', label: 'FlashAttention-2' },
];

// ── Occupancy model (Panel A) ────────────────────────────────────────────────
// Illustrative small workload: batch·heads = 8 thread blocks. FA1 schedules only
// those 8 → 8 of 32 cores lit. FA2 also splits Q into 4 sequence tiles, so
// blocks = 8 × 4 = 32 → the grid saturates.
const GRID_COLS = 8;
const GRID_ROWS = 4;
const GRID_CELLS = GRID_COLS * GRID_ROWS; // 32 cores
const FA1_BLOCKS = 8; // batch·heads only
const SEQ_TILES = 4; // FA2 splits query rows into this many tiles
const FA2_BLOCKS = FA1_BLOCKS * SEQ_TILES; // 32 → fills the grid

const WARPS = 4; // threads run in groups of 32 = a "warp"; 4 warps share this block

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

// ── SVG geometry ─────────────────────────────────────────────────────────────
const VB_W = 760;
const VB_H = 320;

// Panel A — occupancy grid (left).
const A = { x: 20, y: 56, w: 320, h: 232 };
const CELL_GAP = 8;
const CELL_PAD = 16;
const CELL_W = (A.w - CELL_PAD * 2 - (GRID_COLS - 1) * CELL_GAP) / GRID_COLS;
// Reserve ~84px (48 top for title/subtitle + 36 bottom) so the grid never
// overlaps the blocks/occupancy readout that sits along the panel's bottom edge.
const CELL_H = (A.h - 84 - (GRID_ROWS - 1) * CELL_GAP) / GRID_ROWS;
const CELL_Y0 = A.y + 48;

// Panel B — one thread block, its score tile + 4 warps (right).
const B = { x: 392, y: 56, w: 348, h: 232 };
const TILE = { x: B.x + 24, y: B.y + 52, w: 120, h: 152 };
const WARP_X = TILE.x + TILE.w + 64;
const WARP_W = 132;
const WARP_H = 28;
const WARP_GAP = 10;
const WARP_Y0 = B.y + 52;

// Per-locale copy. English strings are moved verbatim from the original JSX.
const COPY = {
  en: {
    heading: 'FlashAttention-2 — same math, fuller GPU',
    pause: '❚❚ Pause',
    play: '▶ Play',
    step: 'Step ›',
    svgAria:
      'Two panels comparing FlashAttention-1 and FlashAttention-2. Left: a grid of 32 GPU cores; FlashAttention-1 lights only 8 because it launches batch times heads thread blocks, while FlashAttention-2 also splits the query sequence into tiles and lights all 32. Right: inside one thread block, FlashAttention-1 splits the key dimension across four warps that must combine partial results through shared memory, while FlashAttention-2 gives each warp its own band of query rows so the warps stay independent with no cross-warp synchronization.',
    panelATitle: 'GPU occupancy',
    panelASubtitle: `${GRID_CELLS} cores (SMs)`,
    blocksPrefix: 'blocks =',
    litSuffix: (lit: number) => `→ ${lit}/${GRID_CELLS} lit`,
    occSaturated: 'cores saturated',
    occIdle: 'most cores idle',
    panelBTitle: 'Inside one thread block',
    panelBSubtitleFa2: 'split-Q · warps own whole rows',
    panelBSubtitleFa1: 'split-K · warps slice the keys',
    warpRows: (w: number) => `Q rows ${w * 2 + 1}–${w * 2 + 2}`,
    allRows: 'all rows',
    captionFa2: (
      <>
        <strong>FlashAttention-2</strong> also splits the query rows into{' '}
        <span className="font-mono">{SEQ_TILES}</span> tiles, so it launches{' '}
        <span className="font-mono">
          {FA1_BLOCKS}×{SEQ_TILES} = {FA2_BLOCKS}
        </span>{' '}
        thread blocks and fills the whole GPU. Inside each block it uses{' '}
        <strong>split-Q</strong>: every warp owns its own band of rows end-to-end, so the warps never combine partials
        through shared memory.
      </>
    ),
    captionFa1: (
      <>
        <strong>FlashAttention-1</strong> launches only{' '}
        <span className="font-mono">batch·heads = {FA1_BLOCKS}</span> thread blocks, so on a small workload most of the{' '}
        {GRID_CELLS} cores sit idle. Inside each block it uses <strong>split-K</strong>: the 4 warps each chew a slice
        of the keys for every row, so their partial results must be{' '}
        <span style={{ color: 'var(--foreground)' }}>merged through slow shared memory</span> (a cross-warp reduction).
      </>
    ),
    grounding: (
      <>
        more thread blocks (sequence parallelism) + cleaner warp split (less shared-memory traffic){' '}
        <span aria-hidden>→</span> roughly{' '}
        <span className="font-mono text-primary">2× FA1</span>, same result.
      </>
    ),
    arxivNote: 'arXiv 2307.08691 · same online-softmax math, better GPU mechanics',
    scoreTile: 'score tile',
    splitByQ: 'split by Q rows →',
    splitByK: 'split by K columns →',
    sharedMemSync: 'shared-memory sync',
    noCrossWarpSync: 'no cross-warp sync',
  },
  zh: {
    heading: 'FlashAttention-2——同样的数学，更满的 GPU',
    pause: '❚❚ 暂停',
    play: '▶ 播放',
    step: '单步 ›',
    svgAria:
      '两幅面板对比 FlashAttention-1 与 FlashAttention-2。左：32 个 GPU 核心组成的网格；FlashAttention-1 只按 batch×heads 启动线程块，仅点亮 8 个核心，而 FlashAttention-2 还把 query 序列切成小块，点亮全部 32 个。右：在一个线程块内部，FlashAttention-1 把 key 维度切给四个 warp，它们必须经由共享内存合并部分结果，而 FlashAttention-2 让每个 warp 拥有自己的一段 query 行，warp 彼此独立，无需跨 warp 同步。',
    panelATitle: 'GPU 占用率',
    panelASubtitle: `${GRID_CELLS} 个核心（SM）`,
    blocksPrefix: '线程块 =',
    litSuffix: (lit: number) => `→ 已点亮 ${lit}/${GRID_CELLS}`,
    occSaturated: '核心全部占满',
    occIdle: '大部分核心闲置',
    panelBTitle: '一个线程块内部',
    panelBSubtitleFa2: 'split-Q · warp 拥有整行',
    panelBSubtitleFa1: 'split-K · warp 切分 key',
    warpRows: (w: number) => `Q 行 ${w * 2 + 1}–${w * 2 + 2}`,
    allRows: '所有行',
    captionFa2: (
      <>
        <strong>FlashAttention-2</strong> 还把 query 行也切成 <span className="font-mono">{SEQ_TILES}</span>{' '}
        个小块，于是它启动{' '}
        <span className="font-mono">
          {FA1_BLOCKS}×{SEQ_TILES} = {FA2_BLOCKS}
        </span>{' '}
        个线程块，铺满整块 GPU。在每个块内部它采用 <strong>split-Q</strong>：每个 warp
        从头到尾拥有自己的一段行，warp 之间从不经由共享内存合并部分结果。
      </>
    ),
    captionFa1: (
      <>
        <strong>FlashAttention-1</strong> 只启动 <span className="font-mono">batch·heads = {FA1_BLOCKS}</span>{' '}
        个线程块，所以在小负载上 {GRID_CELLS} 个核心大多闲置。在每个块内部它采用 <strong>split-K</strong>：4 个 warp
        各自为每一行啃一段 key，它们的部分结果必须
        <span style={{ color: 'var(--foreground)' }}>经由缓慢的共享内存合并</span>（一次跨 warp 归约）。
      </>
    ),
    grounding: (
      <>
        更多线程块（序列并行）+ 更干净的 warp 划分（更少共享内存流量）<span aria-hidden>→</span> 大约{' '}
        <span className="font-mono text-primary">2× FA1</span>，结果不变。
      </>
    ),
    arxivNote: 'arXiv 2307.08691 · 同样的在线 softmax 数学，更好的 GPU 机制',
    scoreTile: '分数块',
    splitByQ: '按 Q 行切分 →',
    splitByK: '按 K 列切分 →',
    sharedMemSync: '共享内存同步',
    noCrossWarpSync: '无跨 warp 同步',
  },
} as const;

export function Flash2ParallelismDiagram() {
  const reducedMotion = usePrefersReducedMotion();
  const copy = COPY[useLocale()];
  // FA2 is the default end-state the prose centers on (and the reduced-motion frame).
  const [variant, setVariant] = React.useState<Variant>('fa2');
  const [playing, setPlaying] = React.useState(!reducedMotion);
  // `lit` counts how many cells have lit up so far (animates the fill).
  const [lit, setLit] = React.useState(reducedMotion ? FA2_BLOCKS : 0);
  // `pulse` drives the FA1 cross-warp reduce arrows + FA2 check (a 0/1 toggle).
  const [pulse, setPulse] = React.useState(false);

  const target = variant === 'fa1' ? FA1_BLOCKS : FA2_BLOCKS;

  // Reset the fill whenever the variant changes (so the new target animates in).
  React.useEffect(() => {
    if (reducedMotion) {
      setLit(target);
      return;
    }
    setLit(0);
  }, [variant, reducedMotion, target]);

  // Animate cells lighting up toward the target, and pulse the warp panel.
  React.useEffect(() => {
    if (!playing || reducedMotion) return;
    const fill = window.setInterval(() => {
      setLit((n) => (n < target ? n + 1 : n));
    }, 90);
    const blink = window.setInterval(() => setPulse((p) => !p), 700);
    return () => {
      window.clearInterval(fill);
      window.clearInterval(blink);
    };
  }, [playing, reducedMotion, target]);

  const litCount = reducedMotion ? target : Math.min(lit, target);
  const isFa2 = variant === 'fa2';

  const step = () => {
    setPlaying(false);
    setLit((n) => (n < target ? Math.min(n + 1, target) : n));
    setPulse((p) => !p);
  };

  const occLabel = isFa2 ? copy.occSaturated : copy.occIdle;
  const showArrows = !isFa2 && (reducedMotion ? true : pulse);

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title + FA1/FA2 toggle (primary control) + play/step */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          {copy.heading}
        </div>
        <div className="flex items-center gap-2">
          <div className="inline-flex overflow-hidden rounded border border-border">
            {VARIANTS.map((v) => (
              <button
                key={v.id}
                type="button"
                onClick={() => setVariant(v.id)}
                aria-pressed={v.id === variant}
                className={[
                  'px-2.5 py-1 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                  v.id === variant ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground',
                ].join(' ')}
              >
                {v.label}
              </button>
            ))}
          </div>
          {!reducedMotion ? (
            <div className="flex items-center gap-1">
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
            </div>
          ) : null}
        </div>
      </div>

      {/* The diagram */}
      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        className="w-full"
        role="img"
        aria-label={copy.svgAria}
      >
        {/* ── Panel A frame: GPU occupancy ── */}
        <PanelFrame box={A} title={copy.panelATitle} subtitle={copy.panelASubtitle} />

        {/* The core grid — each cell is a streaming multiprocessor (a "core"). */}
        {Array.from({ length: GRID_ROWS }, (_, r) =>
          Array.from({ length: GRID_COLS }, (_, c) => {
            const idx = r * GRID_COLS + c;
            const on = idx < litCount;
            return (
              <rect
                key={`${r}-${c}`}
                x={A.x + CELL_PAD + c * (CELL_W + CELL_GAP)}
                y={CELL_Y0 + r * (CELL_H + CELL_GAP)}
                width={CELL_W}
                height={CELL_H}
                rx={3}
                style={{
                  fill: on ? 'var(--primary)' : 'var(--muted)',
                  fillOpacity: on ? 0.9 : 1,
                  stroke: 'var(--border)',
                  transition: reducedMotion ? undefined : 'fill 200ms linear, fill-opacity 200ms linear',
                }}
                strokeWidth={1}
              />
            );
          }),
        )}

        {/* Occupancy readout under the grid */}
        <text
          x={A.x + CELL_PAD}
          y={A.y + A.h - 10}
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[11px]"
        >
          {copy.blocksPrefix}{' '}
          <tspan className="font-mono" style={{ fill: 'var(--foreground)' }}>
            {isFa2 ? `${FA1_BLOCKS}×${SEQ_TILES} = ${FA2_BLOCKS}` : `${FA1_BLOCKS}`}
          </tspan>{' '}
          {copy.litSuffix(litCount)}
        </text>
        <text
          x={A.x + A.w - CELL_PAD}
          y={A.y + A.h - 10}
          textAnchor="end"
          style={{ fill: isFa2 ? 'var(--primary)' : 'var(--muted-foreground)' }}
          className="text-[11px] font-semibold"
        >
          {occLabel}
        </text>

        {/* ── Panel B frame: inside one thread block ── */}
        <PanelFrame
          box={B}
          title={copy.panelBTitle}
          subtitle={isFa2 ? copy.panelBSubtitleFa2 : copy.panelBSubtitleFa1}
        />

        {/* The score tile (Q rows × K columns) that the 4 warps must cover */}
        <ScoreTile isFa2={isFa2} />

        {/* The 4 warps — each a labelled row to the right of the tile */}
        {Array.from({ length: WARPS }, (_, w) => {
          const y = WARP_Y0 + w * (WARP_H + WARP_GAP);
          return (
            <g key={w}>
              <rect
                x={WARP_X}
                y={y}
                width={WARP_W}
                height={WARP_H}
                rx={6}
                style={{
                  fill: isFa2 ? 'var(--primary)' : 'var(--card)',
                  fillOpacity: isFa2 ? 0.12 : 1,
                  stroke: isFa2 ? 'var(--primary)' : 'var(--border)',
                }}
                strokeWidth={1.5}
              />
              <text
                x={WARP_X + 10}
                y={y + WARP_H / 2 + 4}
                style={{ fill: 'var(--foreground)' }}
                className="text-[11px] font-medium"
              >
                warp {w + 1}
              </text>
              <text
                x={WARP_X + WARP_W - 10}
                y={y + WARP_H / 2 + 4}
                textAnchor="end"
                style={{ fill: 'var(--muted-foreground)' }}
                className="text-[9px]"
              >
                {isFa2 ? copy.warpRows(w) : copy.allRows}
              </text>
            </g>
          );
        })}

        {/* FA1: red exchange/reduce arrows between adjacent warps + sync label.
            FA2: a green "no cross-warp sync" check. */}
        {showArrows ? <ReduceArrows animated={!reducedMotion && playing} /> : null}
        {isFa2 ? <NoSyncCheck /> : null}
      </svg>

      {/* Caption — what the current variant does */}
      <p className="min-h-[3.25rem] text-[13px] text-foreground/90">
        {isFa2 ? copy.captionFa2 : copy.captionFa1}
      </p>

      {/* Grounding line */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border/60 pt-3">
        <div className="text-[11px] text-muted-foreground">
          {copy.grounding}
        </div>
        <div className="text-[10px] text-muted-foreground/70">
          {copy.arxivNote}
        </div>
      </div>
    </div>
  );
}

/** A panel container with a title + subtitle along its top edge. */
function PanelFrame({
  box,
  title,
  subtitle,
}: {
  box: { x: number; y: number; w: number; h: number };
  title: string;
  subtitle: string;
}) {
  return (
    <g>
      <rect
        x={box.x}
        y={box.y}
        width={box.w}
        height={box.h}
        rx={10}
        style={{ fill: 'var(--card)', stroke: 'var(--border)' }}
        strokeWidth={1.5}
      />
      <text x={box.x + 16} y={box.y + 24} style={{ fill: 'var(--foreground)' }} className="text-[13px] font-semibold">
        {title}
      </text>
      <text x={box.x + box.w - 16} y={box.y + 24} textAnchor="end" style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
        {subtitle}
      </text>
    </g>
  );
}

/** The score tile inside the block — split horizontally (split-K, FA1) or into
 * row bands (split-Q, FA2) to show which axis each warp owns. */
function ScoreTile({ isFa2 }: { isFa2: boolean }) {
  const copy = COPY[useLocale()];
  return (
    <g>
      <rect
        x={TILE.x}
        y={TILE.y}
        width={TILE.w}
        height={TILE.h}
        rx={6}
        style={{ fill: 'var(--primary)', fillOpacity: 0.06, stroke: 'var(--primary)' }}
        strokeWidth={1.5}
      />
      {/* Partition guides: vertical (K slices) for FA1, horizontal (Q bands) for FA2 */}
      {isFa2
        ? Array.from({ length: WARPS - 1 }, (_, i) => {
            const y = TILE.y + (TILE.h / WARPS) * (i + 1);
            return (
              <line
                key={i}
                x1={TILE.x}
                y1={y}
                x2={TILE.x + TILE.w}
                y2={y}
                style={{ stroke: 'var(--primary)' }}
                strokeWidth={1}
              />
            );
          })
        : Array.from({ length: WARPS - 1 }, (_, i) => {
            const x = TILE.x + (TILE.w / WARPS) * (i + 1);
            return (
              <line
                key={i}
                x1={x}
                y1={TILE.y}
                x2={x}
                y2={TILE.y + TILE.h}
                style={{ stroke: 'var(--muted-foreground)' }}
                strokeWidth={1}
                strokeDasharray="3 3"
              />
            );
          })}
      <text
        x={TILE.x + TILE.w / 2}
        y={TILE.y - 8}
        textAnchor="middle"
        style={{ fill: 'var(--muted-foreground)' }}
        className="text-[10px]"
      >
        {copy.scoreTile}
      </text>
      <text
        x={TILE.x + TILE.w / 2}
        y={TILE.y + TILE.h + 16}
        textAnchor="middle"
        style={{ fill: 'var(--muted-foreground)' }}
        className="text-[9px]"
      >
        {isFa2 ? copy.splitByQ : copy.splitByK}
      </text>
    </g>
  );
}

/** FA1 cross-warp reduction: arrows between adjacent warps + a shared-memory
 * sync label, all in the "warning" foreground tone (no hex, tokens only). */
function ReduceArrows({ animated }: { animated: boolean }) {
  const copy = COPY[useLocale()];
  const cx = WARP_X - 18; // a channel just left of the warp rows
  const ys = Array.from({ length: WARPS }, (_, w) => WARP_Y0 + w * (WARP_H + WARP_GAP) + WARP_H / 2);
  return (
    <g className={animated ? 'animate-pulse' : undefined}>
      {/* vertical exchange spine */}
      <line x1={cx} y1={ys[0]} x2={cx} y2={ys[ys.length - 1]} style={{ stroke: 'var(--foreground)' }} strokeWidth={2} />
      {ys.map((y, i) => (
        <g key={i}>
          {/* stub from the spine into each warp */}
          <line x1={cx} y1={y} x2={WARP_X} y2={y} style={{ stroke: 'var(--foreground)' }} strokeWidth={2} />
          <polygon
            points={`${WARP_X},${y} ${WARP_X - 7},${y - 4} ${WARP_X - 7},${y + 4}`}
            style={{ fill: 'var(--foreground)' }}
          />
          {/* arrow back out to the spine (the exchange goes both ways) */}
          <polygon
            points={`${cx},${y} ${cx + 7},${y - 4} ${cx + 7},${y + 4}`}
            style={{ fill: 'var(--foreground)' }}
          />
        </g>
      ))}
      {/* Label below the warp stack (matching the FA2 badge position) so it never
          lands on warp 1 the way a top-anchored label did. */}
      <text
        x={WARP_X + WARP_W / 2}
        y={WARP_Y0 + WARPS * (WARP_H + WARP_GAP) + 16}
        textAnchor="middle"
        style={{ fill: 'var(--foreground)' }}
        className="text-[10px] font-semibold"
      >
        {copy.sharedMemSync}
      </text>
    </g>
  );
}

/** FA2 "no cross-warp sync" badge — a green check using the primary token. */
function NoSyncCheck() {
  const copy = COPY[useLocale()];
  const x = WARP_X - 30;
  const y = WARP_Y0 + (WARPS * (WARP_H + WARP_GAP)) / 2 - WARP_GAP / 2;
  return (
    <g>
      <circle cx={x} cy={y} r={11} style={{ fill: 'var(--primary)', fillOpacity: 0.15, stroke: 'var(--primary)' }} strokeWidth={1.5} />
      <path
        d={`M ${x - 5} ${y} L ${x - 1.5} ${y + 4} L ${x + 5.5} ${y - 4.5}`}
        fill="none"
        style={{ stroke: 'var(--primary)' }}
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Label sits BELOW the warp stack (centered under the warps), clear of the
          warp rows and the score tile — the tile↔warp gap is too narrow for it. */}
      <text
        x={WARP_X + WARP_W / 2}
        y={WARP_Y0 + WARPS * (WARP_H + WARP_GAP) + 16}
        textAnchor="middle"
        style={{ fill: 'var(--primary)' }}
        className="text-[10px] font-semibold"
      >
        {copy.noCrossWarpSync}
      </text>
    </g>
  );
}

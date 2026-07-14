import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * GpuHierarchyDiagram — chapter 4 deep-dive "Flash Attention", the hardware
 * primer that sits just before the FlashAttention-2 section.
 *
 * Self-contained: NO worker, NO model, NO WASM — pure SVG + React state, so it
 * server-renders for crawlers (createRoot replaces it on boot) and is robust
 * while the model downloads.
 *
 * It makes "thread block", "warp", and "occupancy" concrete BEFORE the FA2
 * section uses them. Three zoom levels, like stepping closer to the chip:
 *
 *   1. Chip   — a grid of 108 squares = the SMs (streaming multiprocessors) on
 *               an A100. Work arrives as "thread blocks"; an SM with no block
 *               assigned does nothing. An occupancy overlay previews the FA2
 *               punchline: v1's 8 blocks light 8/108 SMs (~7%); v2's
 *               8 heads × 32 query tiles = 256 blocks light all 108.
 *   2. One SM — inside one square: a few warp slots, Tensor Cores (the matrix
 *               math units), and shared memory ≈192 KB — the SRAM scratchpad
 *               the tiling diagram keeps score tiles in.
 *   3. One warp — 32 dots stepping through instructions in lockstep: the GPU's
 *               smallest unit of scheduling.
 *
 * This widget teaches the HARDWARE ladder only; the sibling
 * Flash2ParallelismDiagram teaches how FA2 re-splits the WORK across it.
 *
 * First render is deterministic (Chip level, v1 overlay, 8 SMs lit) — identical
 * server vs client. Animations live in useEffect guarded by playing &&
 * !reducedMotion. Honors prefers-reduced-motion (static representative state
 * per level, no motion).
 */

type Level = 'chip' | 'sm' | 'warp';
type Overlay = 'v1' | 'v2';

// Toggle ids — the per-locale button labels live in COPY below.
const LEVELS: Level[] = ['chip', 'sm', 'warp'];

const OVERLAYS: Overlay[] = ['v1', 'v2'];

// ── The numbers (A100) ───────────────────────────────────────────────────────
const SM_COUNT = 108; // streaming multiprocessors on an A100
const GRID_COLS = 12;
const GRID_ROWS = 9; // 12 × 9 = 108
const WARP_THREADS = 32; // threads per warp
const WARP_SLOTS = 4; // warp rows drawn inside the SM view
const V1_BLOCKS = 8; // batch 1 × 8 heads
const HEADS = 8;
const Q_TILES = 32; // 4096-token prompt / 128-row tiles
const V2_BLOCKS = HEADS * Q_TILES; // 256 — more blocks than SMs
const SHARED_KB = 192; // shared memory per SM (≈, usable up to ~164 KB/block)
const V1_PCT = Math.round((V1_BLOCKS / SM_COUNT) * 100); // ≈ 7

// The 8 SMs v1 lights up — fixed, scattered so the idleness reads at a glance.
const V1_SM_INDEXES = [7, 20, 34, 47, 61, 74, 88, 101];
const V1_ORDER = new Map(V1_SM_INDEXES.map((idx, order) => [idx, order]));

// One warp's instruction stream (warp level) — every thread runs the SAME one.
const INSTRS = ['load', 'multiply', 'add', 'store'];

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
const VB_H = 336;

const FRAME = { x: 20, y: 16, w: 720, h: 304 };

// Chip level — the 12×9 SM grid.
const G_PAD = 16;
const G_GAP_X = 6;
const G_GAP_Y = 5;
const G_X0 = FRAME.x + G_PAD;
const G_Y0 = FRAME.y + 48;
const CELL_W = (FRAME.w - G_PAD * 2 - (GRID_COLS - 1) * G_GAP_X) / GRID_COLS;
const CELL_H = (FRAME.h - 48 - 38 - (GRID_ROWS - 1) * G_GAP_Y) / GRID_ROWS;
const READOUT_Y = FRAME.y + FRAME.h - 14;

// SM level — warp slots (left), Tensor Cores (right), shared memory (bottom).
const SLOT = { x: 52, y0: 78, w: 250, h: 34, gap: 12 };
const TC = { x: 410, y: 78, w: 290, h: 118 };
const SMEM = { x: 52, y: 252, w: 648, h: 44 };

// Warp level — 32 dots sliding between 4 instruction lanes.
const LANE_LABEL_X = 60;
const LANE_X0 = 150;
const LANE_X1 = 700;
const LANE_Y0 = 104;
const LANE_GAP = 50;
const DOT_R = 5.5;
const DOT_X0 = 165;
const DOT_DX = (LANE_X1 - 20 - DOT_X0) / (WARP_THREADS - 1);

// Per-locale copy. English strings are moved verbatim from the original JSX.
// Hardware terms (SM, warp, Tensor Cores, A100, KB) and the mono identifiers
// stay English in both locales; only the prose around them translates.
const COPY = {
  en: {
    levels: { chip: 'Chip', sm: 'One SM', warp: 'One warp' } as Record<Level, string>,
    overlays: { v1: 'v1: 8 blocks', v2: 'v2: 8×32 blocks' } as Record<Overlay, string>,
    heading: 'Inside the GPU — chip → SM → warp',
    pause: '❚❚ Pause',
    play: '▶ Play',
    step: 'Step ›',
    ariaChipV2:
      'Grid of 108 squares, the streaming multiprocessors on an A100 GPU. With FlashAttention-2 launching 256 thread blocks, all 108 SMs light up and the extra blocks queue.',
    ariaChipV1:
      'Grid of 108 squares, the streaming multiprocessors on an A100 GPU. With FlashAttention-1 launching only 8 thread blocks, just 8 of the 108 SMs light up; the rest sit idle.',
    ariaSm:
      'One streaming multiprocessor, zoomed in: four warp slots taking turns on a Tensor Core unit that does the matrix math, above a shared-memory scratchpad of about 192 kilobytes.',
    ariaWarp:
      'One warp, zoomed in: a row of 32 dots stepping together through a list of instructions, always all on the same instruction at the same time.',
    captionChipV2: (
      <>
        Each square is an <strong>SM</strong> — a streaming multiprocessor, one of{' '}
        <span className="font-mono">108</span> independent little processors on an A100. Work is handed to SMs as{' '}
        <strong>thread blocks</strong>; an SM with no block assigned does nothing. FlashAttention-2 also splits the
        4096-token prompt into <span className="font-mono">32</span> query tiles (128 rows each), so{' '}
        <span className="font-mono">8 heads × 32 tiles = 256</span> blocks — more blocks than SMs. Every SM is
        busy, and the leftover blocks queue up.
      </>
    ),
    captionChipV1: (
      <>
        Each square is an <strong>SM</strong> — a streaming multiprocessor, one of{' '}
        <span className="font-mono">108</span> independent little processors on an A100. Work is handed to SMs as{' '}
        <strong>thread blocks</strong>; an SM with no block assigned does nothing. FlashAttention-1 launches{' '}
        <span className="font-mono">batch × heads = 1 × 8 = 8</span> blocks, so only 8 of the 108 SMs have
        anything to do.
      </>
    ),
    captionSm: (
      <>
        Zoomed into one square. A <strong>block runs on one SM</strong>; inside it, threads are bundled into{' '}
        <strong>warps</strong>. The warps take turns on the SM&apos;s compute units — the{' '}
        <strong>Tensor Cores</strong> do the matrix math — and the block keeps its tiles in{' '}
        <strong>shared memory</strong>: the SRAM scratchpad from the tiling diagram, a ~200 KB-class scratchpad
        (≈192 KB on an A100).
      </>
    ),
    captionWarp: (
      <>
        Zoomed all the way in: one <strong>warp</strong> —{' '}
        <span className="font-mono">{WARP_THREADS}</span> threads that execute the{' '}
        <strong>same instruction together</strong>, the GPU&apos;s smallest unit of scheduling. They move in
        lockstep; the hardware schedules warps, never individual threads.
      </>
    ),
    reducedNote: 'Reduced motion: showing a static representative state for this zoom level.',
    groundingLeft: (
      <>8 blocks can&apos;t feed 108 SMs — fixing exactly that is FlashAttention-2&apos;s whole move (next).</>
    ),
    groundingRight: (
      <>
        A100 · {SM_COUNT} SMs · warp = {WARP_THREADS} threads · shared memory ≈{SHARED_KB} KB/SM
      </>
    ),
    chipTitle: 'A100 chip',
    chipSubtitle: '108 SMs (streaming multiprocessors)',
    blocksPrefix: 'blocks =',
    blocksV2: `${HEADS} heads × ${Q_TILES} query tiles = ${V2_BLOCKS}`,
    blocksV1: `1 batch × ${HEADS} heads = ${V1_BLOCKS}`,
    busyV2: (lit: number) =>
      `${Math.min(lit, SM_COUNT)} / ${SM_COUNT} busy${lit >= SM_COUNT ? ', blocks queue up' : '…'}`,
    busyV1: (lit: number) =>
      `${Math.min(lit, V1_BLOCKS)} / ${SM_COUNT} SMs busy${lit >= V1_BLOCKS ? ` ≈ ${V1_PCT}%` : '…'}`,
    smTitle: 'One SM',
    smSubtitle: 'one thread block runs here',
    warpsBundled: `the block's threads, bundled into warps of ${WARP_THREADS}`,
    threadsLabel: `${WARP_THREADS} threads`,
    takeTurns: 'warps take turns',
    tensorCoresSub: 'the matrix math units',
    sharedMemLabel: `shared memory ≈${SHARED_KB} KB`,
    sharedMemNote: 'the SRAM scratchpad from the tiling diagram — score tiles live and die here',
    warpTitle: 'One warp',
    warpSubtitle: `${WARP_THREADS} threads, one instruction pointer`,
    lockstepNote: `all ${WARP_THREADS} dots sit on the same instruction line — always. That bundle is what the GPU schedules.`,
  },
  zh: {
    levels: { chip: '芯片', sm: '单个 SM', warp: '单个 warp' } as Record<Level, string>,
    overlays: { v1: 'v1：8 个块', v2: 'v2：8×32 个块' } as Record<Overlay, string>,
    heading: 'GPU 内部——芯片 → SM → warp',
    pause: '❚❚ 暂停',
    play: '▶ 播放',
    step: '单步 ›',
    ariaChipV2:
      '由 108 个方块组成的网格，即 A100 GPU 上的流式多处理器（SM）。FlashAttention-2 启动 256 个线程块时，108 个 SM 全部点亮，多出的块排队等待。',
    ariaChipV1:
      '由 108 个方块组成的网格，即 A100 GPU 上的流式多处理器（SM）。FlashAttention-1 只启动 8 个线程块时，108 个 SM 中只有 8 个点亮，其余闲置。',
    ariaSm:
      '放大到一个流式多处理器内部：四个 warp 槽轮流使用负责矩阵运算的 Tensor Core 单元，下方是约 192 KB 的共享内存暂存区。',
    ariaWarp: '放大到一个 warp 内部：一排 32 个点一起逐条执行指令，永远同时停在同一条指令上。',
    captionChipV2: (
      <>
        每个方块是一个 <strong>SM</strong>——流式多处理器，A100 上 <span className="font-mono">108</span>{' '}
        个彼此独立的小处理器之一。工作以<strong>线程块</strong>的形式交给 SM；没有分到块的 SM
        什么都不做。FlashAttention-2 还把 4096 token 的提示词切成 <span className="font-mono">32</span> 个 query
        块（每块 128 行），于是 <span className="font-mono">8 头 × 32 块 = 256</span> 个线程块——比 SM 还多。每个 SM
        都在忙，多出的块排队等候。
      </>
    ),
    captionChipV1: (
      <>
        每个方块是一个 <strong>SM</strong>——流式多处理器，A100 上 <span className="font-mono">108</span>{' '}
        个彼此独立的小处理器之一。工作以<strong>线程块</strong>的形式交给 SM；没有分到块的 SM
        什么都不做。FlashAttention-1 只启动 <span className="font-mono">batch × heads = 1 × 8 = 8</span> 个块，所以
        108 个 SM 里只有 8 个有活可干。
      </>
    ),
    captionSm: (
      <>
        放大到一个方块内部。<strong>一个块在一个 SM 上运行</strong>；块内的线程被捆成 <strong>warp</strong>。各个
        warp 轮流使用 SM 的计算单元——<strong>Tensor Core</strong> 负责矩阵乘法——块把自己的 tile 放在
        <strong>共享内存</strong>里：就是分块图中的那块 SRAM 暂存区，约 200 KB 级（A100 上 ≈192 KB）。
      </>
    ),
    captionWarp: (
      <>
        放大到底：一个 <strong>warp</strong>——<span className="font-mono">{WARP_THREADS}</span> 个
        <strong>一起执行同一条指令</strong>的线程，GPU 最小的调度单位。它们锁步前进；硬件调度的是
        warp，从不调度单个线程。
      </>
    ),
    reducedNote: '减少动态：此缩放级别仅显示一个静态代表状态。',
    groundingLeft: <>8 个块喂不饱 108 个 SM——FlashAttention-2 的全部招数（下一节）修的正是这件事。</>,
    groundingRight: (
      <>
        A100 · {SM_COUNT} 个 SM · warp = {WARP_THREADS} 线程 · 共享内存 ≈{SHARED_KB} KB/SM
      </>
    ),
    chipTitle: 'A100 芯片',
    chipSubtitle: '108 个 SM（流式多处理器）',
    blocksPrefix: '线程块 =',
    blocksV2: `${HEADS} 头 × ${Q_TILES} 个 query 块 = ${V2_BLOCKS}`,
    blocksV1: `1 batch × ${HEADS} 头 = ${V1_BLOCKS}`,
    busyV2: (lit: number) =>
      `${Math.min(lit, SM_COUNT)} / ${SM_COUNT} 忙碌${lit >= SM_COUNT ? '，后续块排队' : '……'}`,
    busyV1: (lit: number) =>
      `${Math.min(lit, V1_BLOCKS)} / ${SM_COUNT} 个 SM 忙碌${lit >= V1_BLOCKS ? ` ≈ ${V1_PCT}%` : '……'}`,
    smTitle: '单个 SM',
    smSubtitle: '一个线程块在这里运行',
    warpsBundled: `块内的线程，每 ${WARP_THREADS} 个捆成一个 warp`,
    threadsLabel: `${WARP_THREADS} 线程`,
    takeTurns: 'warp 轮流上阵',
    tensorCoresSub: '矩阵乘法单元',
    sharedMemLabel: `共享内存 ≈${SHARED_KB} KB`,
    sharedMemNote: '就是分块图中的那块 SRAM 暂存区——分数块在这里生灭',
    warpTitle: '单个 warp',
    warpSubtitle: `${WARP_THREADS} 个线程，一个指令指针`,
    lockstepNote: `所有 ${WARP_THREADS} 个点永远停在同一条指令线上。GPU 调度的正是这一整捆。`,
  },
} as const;

export function GpuHierarchyDiagram() {
  const reducedMotion = usePrefersReducedMotion();
  const copy = COPY[useLocale()];
  const [level, setLevel] = React.useState<Level>('chip');
  const [overlay, setOverlay] = React.useState<Overlay>('v1');
  const [playing, setPlaying] = React.useState(true);
  // Chip: how many SMs have lit up so far. Starts at the full v1 picture so the
  // server-rendered frame already shows "8 / 108 busy".
  const [lit, setLit] = React.useState(V1_BLOCKS);
  // SM: which warp slot currently owns the compute units.
  const [activeWarp, setActiveWarp] = React.useState(0);
  // Warp: which instruction all 32 threads are on.
  const [instr, setInstr] = React.useState(1);

  const target = overlay === 'v1' ? V1_BLOCKS : SM_COUNT;

  // Pin the representative static state under reduced motion.
  React.useEffect(() => {
    if (!reducedMotion) return;
    setPlaying(false);
    setActiveWarp(0);
    setInstr(1);
  }, [reducedMotion]);

  // Restart the chip fill whenever the overlay (or level) changes.
  React.useEffect(() => {
    if (level !== 'chip') return;
    if (reducedMotion) {
      setLit(target);
      return;
    }
    setLit(0);
  }, [level, overlay, reducedMotion, target]);

  // Drive the active level's animation.
  React.useEffect(() => {
    if (!playing || reducedMotion) return;
    if (level === 'chip') {
      const stride = overlay === 'v2' ? 6 : 1;
      const id = window.setInterval(
        () => setLit((n) => (n < target ? Math.min(n + stride, target) : n)),
        overlay === 'v2' ? 80 : 170,
      );
      return () => window.clearInterval(id);
    }
    if (level === 'sm') {
      const id = window.setInterval(() => setActiveWarp((w) => (w + 1) % WARP_SLOTS), 900);
      return () => window.clearInterval(id);
    }
    const id = window.setInterval(() => setInstr((i) => (i + 1) % INSTRS.length), 1100);
    return () => window.clearInterval(id);
  }, [playing, reducedMotion, level, overlay, target]);

  const litCount = reducedMotion ? target : Math.min(lit, target);
  const isV2 = overlay === 'v2';

  const step = () => {
    setPlaying(false);
    if (level === 'chip') setLit((n) => Math.min(n + (isV2 ? 6 : 1), target));
    else if (level === 'sm') setActiveWarp((w) => (w + 1) % WARP_SLOTS);
    else setInstr((i) => (i + 1) % INSTRS.length);
  };

  const ariaLabel =
    level === 'chip'
      ? isV2
        ? copy.ariaChipV2
        : copy.ariaChipV1
      : level === 'sm'
        ? copy.ariaSm
        : copy.ariaWarp;

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title + zoom-level toggle + (chip only) occupancy overlay + play/step */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          {copy.heading}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div className="inline-flex overflow-hidden rounded border border-border">
            {LEVELS.map((l) => (
              <button
                key={l}
                type="button"
                onClick={() => setLevel(l)}
                aria-pressed={l === level}
                className={[
                  'px-2.5 py-1 max-sm:px-3 max-sm:py-2 max-sm:min-h-[44px] inline-flex items-center text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                  l === level ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground',
                ].join(' ')}
              >
                {copy.levels[l]}
              </button>
            ))}
          </div>
          {level === 'chip' ? (
            <div className="inline-flex overflow-hidden rounded border border-border">
              {OVERLAYS.map((o) => (
                <button
                  key={o}
                  type="button"
                  onClick={() => setOverlay(o)}
                  aria-pressed={o === overlay}
                  className={[
                    'px-2.5 py-1 max-sm:px-3 max-sm:py-2 max-sm:min-h-[44px] inline-flex items-center text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                    o === overlay ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground',
                  ].join(' ')}
                >
                  {copy.overlays[o]}
                </button>
              ))}
            </div>
          ) : null}
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
      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={ariaLabel}>
        {level === 'chip' ? (
          <ChipView litCount={litCount} isV2={isV2} reducedMotion={reducedMotion} />
        ) : level === 'sm' ? (
          <SmView activeWarp={activeWarp} reducedMotion={reducedMotion} />
        ) : (
          <WarpView instr={instr} reducedMotion={reducedMotion} />
        )}
      </svg>

      {/* Caption — what the current zoom level shows */}
      <p className="min-h-[4rem] text-[13px] text-foreground/90">
        {level === 'chip'
          ? isV2
            ? copy.captionChipV2
            : copy.captionChipV1
          : level === 'sm'
            ? copy.captionSm
            : copy.captionWarp}
      </p>

      {reducedMotion ? (
        <p className="text-[10px] text-muted-foreground/70">
          {copy.reducedNote}
        </p>
      ) : null}

      {/* Grounding line — the bridge into the FA2 section */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border/60 pt-3">
        <div className="text-[11px] text-muted-foreground">
          {copy.groundingLeft}
        </div>
        <div className="text-[10px] text-muted-foreground/70">
          {copy.groundingRight}
        </div>
      </div>
    </div>
  );
}

/** A panel container with a title + subtitle along its top edge. */
function PanelFrame({ title, subtitle }: { title: string; subtitle: string }) {
  return (
    <g>
      <rect
        x={FRAME.x}
        y={FRAME.y}
        width={FRAME.w}
        height={FRAME.h}
        rx={10}
        style={{ fill: 'var(--card)', stroke: 'var(--border)' }}
        strokeWidth={1.5}
      />
      <text x={FRAME.x + 16} y={FRAME.y + 24} style={{ fill: 'var(--foreground)' }} className="text-[13px] font-semibold">
        {title}
      </text>
      <text
        x={FRAME.x + FRAME.w - 16}
        y={FRAME.y + 24}
        textAnchor="end"
        style={{ fill: 'var(--muted-foreground)' }}
        className="text-[10px]"
      >
        {subtitle}
      </text>
    </g>
  );
}

/** Zoom 1 — the whole chip: 108 SM squares + the v1/v2 occupancy readout. */
function ChipView({ litCount, isV2, reducedMotion }: { litCount: number; isV2: boolean; reducedMotion: boolean }) {
  const copy = COPY[useLocale()];
  return (
    <g>
      <PanelFrame title={copy.chipTitle} subtitle={copy.chipSubtitle} />
      {Array.from({ length: GRID_ROWS }, (_, r) =>
        Array.from({ length: GRID_COLS }, (_, c) => {
          const idx = r * GRID_COLS + c;
          // v2 fills in index order; v1 lights only its 8 scattered SMs.
          const on = isV2 ? idx < litCount : (V1_ORDER.get(idx) ?? Infinity) < litCount;
          return (
            <rect
              key={idx}
              x={G_X0 + c * (CELL_W + G_GAP_X)}
              y={G_Y0 + r * (CELL_H + G_GAP_Y)}
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
      <text x={G_X0} y={READOUT_Y} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px]">
        {copy.blocksPrefix}{' '}
        <tspan className="font-mono" style={{ fill: 'var(--foreground)' }}>
          {isV2 ? copy.blocksV2 : copy.blocksV1}
        </tspan>
      </text>
      <text
        x={FRAME.x + FRAME.w - G_PAD}
        y={READOUT_Y}
        textAnchor="end"
        style={{ fill: isV2 ? 'var(--primary)' : 'var(--muted-foreground)' }}
        className="text-[11px] font-semibold"
      >
        {isV2 ? copy.busyV2(litCount) : copy.busyV1(litCount)}
      </text>
    </g>
  );
}

/** Zoom 2 — inside one SM: warp slots, Tensor Cores, shared memory. */
function SmView({ activeWarp, reducedMotion }: { activeWarp: number; reducedMotion: boolean }) {
  const copy = COPY[useLocale()];
  const activeY = SLOT.y0 + activeWarp * (SLOT.h + SLOT.gap) + SLOT.h / 2;
  return (
    <g>
      <PanelFrame title={copy.smTitle} subtitle={copy.smSubtitle} />

      {/* Warp slots — the block's threads, bundled in 32s */}
      <text x={SLOT.x} y={SLOT.y0 - 10} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
        {copy.warpsBundled}
      </text>
      {Array.from({ length: WARP_SLOTS }, (_, w) => {
        const y = SLOT.y0 + w * (SLOT.h + SLOT.gap);
        const active = w === activeWarp;
        return (
          <g key={w}>
            <rect
              x={SLOT.x}
              y={y}
              width={SLOT.w}
              height={SLOT.h}
              rx={6}
              style={{
                fill: active ? 'var(--primary)' : 'var(--muted)',
                fillOpacity: active ? 0.15 : 1,
                stroke: active ? 'var(--primary)' : 'var(--border)',
                transition: reducedMotion ? undefined : 'fill 200ms linear, stroke 200ms linear',
              }}
              strokeWidth={active ? 1.5 : 1}
            />
            <text
              x={SLOT.x + 10}
              y={y + SLOT.h / 2 + 4}
              style={{ fill: 'var(--foreground)' }}
              className="text-[11px] font-medium"
            >
              warp {w + 1}
            </text>
            <text
              x={SLOT.x + SLOT.w - 10}
              y={y + SLOT.h / 2 + 4}
              textAnchor="end"
              style={{ fill: 'var(--muted-foreground)' }}
              className="text-[9px]"
            >
              {copy.threadsLabel}
            </text>
          </g>
        );
      })}

      {/* Active warp → Tensor Cores arrow */}
      <line
        x1={SLOT.x + SLOT.w + 8}
        y1={activeY}
        x2={TC.x - 10}
        y2={TC.y + TC.h / 2}
        style={{ stroke: 'var(--primary)', transition: reducedMotion ? undefined : 'all 300ms ease' }}
        strokeWidth={2}
      />
      <polygon
        points={`${TC.x - 2},${TC.y + TC.h / 2} ${TC.x - 11},${TC.y + TC.h / 2 - 5} ${TC.x - 11},${TC.y + TC.h / 2 + 5}`}
        style={{ fill: 'var(--primary)' }}
      />
      <text
        x={(SLOT.x + SLOT.w + TC.x) / 2}
        y={TC.y + TC.h / 2 - 12}
        textAnchor="middle"
        style={{ fill: 'var(--muted-foreground)' }}
        className="text-[9px]"
      >
        {copy.takeTurns}
      </text>

      {/* Tensor Cores — the matrix math units */}
      <rect
        x={TC.x}
        y={TC.y}
        width={TC.w}
        height={TC.h}
        rx={8}
        style={{ fill: 'var(--primary)', fillOpacity: 0.07, stroke: 'var(--primary)' }}
        strokeWidth={1.5}
      />
      <text x={TC.x + 14} y={TC.y + 24} style={{ fill: 'var(--foreground)' }} className="text-[12px] font-semibold">
        Tensor Cores
      </text>
      <text x={TC.x + 14} y={TC.y + 40} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
        {copy.tensorCoresSub}
      </text>
      {Array.from({ length: 3 }, (_, r) =>
        Array.from({ length: 8 }, (_, c) => (
          <rect
            key={`${r}-${c}`}
            x={TC.x + 14 + c * 33}
            y={TC.y + 52 + r * 20}
            width={26}
            height={14}
            rx={2}
            style={{ fill: 'var(--primary)', fillOpacity: 0.45 }}
          />
        )),
      )}

      {/* Shared memory — the SRAM scratchpad from the tiling diagram */}
      <rect
        x={SMEM.x}
        y={SMEM.y}
        width={SMEM.w}
        height={SMEM.h}
        rx={8}
        style={{ fill: 'var(--muted)', stroke: 'var(--border)' }}
        strokeWidth={1}
      />
      <text x={SMEM.x + 14} y={SMEM.y + SMEM.h / 2 + 4} style={{ fill: 'var(--foreground)' }} className="text-[11px] font-semibold">
        {copy.sharedMemLabel}
      </text>
      <text
        x={SMEM.x + SMEM.w - 14}
        y={SMEM.y + SMEM.h / 2 + 4}
        textAnchor="end"
        style={{ fill: 'var(--muted-foreground)' }}
        className="text-[10px]"
      >
        {copy.sharedMemNote}
      </text>
    </g>
  );
}

/** Zoom 3 — one warp: 32 dots stepping through instructions in lockstep. */
function WarpView({ instr, reducedMotion }: { instr: number; reducedMotion: boolean }) {
  const copy = COPY[useLocale()];
  const dotsY = LANE_Y0 + instr * LANE_GAP;
  return (
    <g>
      <PanelFrame title={copy.warpTitle} subtitle={copy.warpSubtitle} />

      {/* Instruction lanes */}
      {INSTRS.map((name, i) => {
        const y = LANE_Y0 + i * LANE_GAP;
        const current = i === instr;
        return (
          <g key={name}>
            <text
              x={LANE_LABEL_X}
              y={y + 4}
              style={{ fill: current ? 'var(--primary)' : 'var(--muted-foreground)' }}
              className={current ? 'font-mono text-[11px] font-semibold' : 'font-mono text-[11px]'}
            >
              {name}
            </text>
            <line
              x1={LANE_X0}
              y1={y}
              x2={LANE_X1}
              y2={y}
              style={{ stroke: current ? 'var(--primary)' : 'var(--border)' }}
              strokeWidth={current ? 1.5 : 1}
              strokeDasharray={current ? undefined : '4 4'}
            />
          </g>
        );
      })}

      {/* The 32 threads — one row of dots that moves as a single unit */}
      <g
        style={{
          transform: `translateY(${dotsY - LANE_Y0}px)`,
          transition: reducedMotion ? undefined : 'transform 500ms ease',
        }}
      >
        {Array.from({ length: WARP_THREADS }, (_, t) => (
          <circle
            key={t}
            cx={DOT_X0 + t * DOT_DX}
            cy={LANE_Y0}
            r={DOT_R}
            style={{ fill: 'var(--primary)', fillOpacity: 0.85, stroke: 'var(--bg)' }}
            strokeWidth={1}
          />
        ))}
      </g>

      {/* Lockstep note */}
      <text
        x={(LANE_X0 + LANE_X1) / 2}
        y={LANE_Y0 + (INSTRS.length - 1) * LANE_GAP + 46}
        textAnchor="middle"
        style={{ fill: 'var(--muted-foreground)' }}
        className="text-[10px]"
      >
        {copy.lockstepNote}
      </text>
    </g>
  );
}

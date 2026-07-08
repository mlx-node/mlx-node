import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * GranularityZoom — "one scale for how many weights?" The granularity ladder:
 * per-tensor → per-channel (one scale per output row) → per-group → per-block
 * (MX, fixed 32). Finer granularity fits each scale to a smaller set of values,
 * so its step size is tighter and rounding error drops — but you store more
 * scales, and the scales cost bits too, so the EFFECTIVE bits/param ticks up.
 *
 * Self-contained and STATIC: NO worker, NO model, NO WASM, NO network, NO
 * animation — pure React + Tailwind/SVG, so it server-renders for crawlers and
 * is deterministic server-vs-client. The only state is the selected granularity.
 *
 * EVERY NUMBER SOURCE-VERIFIED (base = int4 = 4 bits/weight; a shared scale is
 * 8–16 bits; effective bits/param = base + scale_bits / group_size):
 *   - per-tensor      : 1 scale over the whole matrix → ≈ 4.00 bits/param
 *     (one scale shared over millions of weights → negligible).
 *   - per-channel     : 1 scale per output row (a long row → tiny overhead) →
 *     ≈ 4.0+ bits/param. We label it ≈ 4.00.
 *   - per-group-128   : 16-bit scale shared over 128 weights →
 *     4 + 16/128 = 4.125 bits/param.
 *   - per-block-32 (MX): 8-bit E8M0 scale shared over 32 weights (the OCP
 *     Microscaling / MXFP standard fixes the block at 32) →
 *     4 + 8/32 = 4.25 bits/param.
 *   (per-group-64 with a 16-bit scale = 4 + 16/64 = 4.25 is mentioned in the
 *    rule line but not drawn — the drawn ladder is tensor → channel → 128 → 32.)
 *
 * THE RULE: finer granularity = lower rounding error, higher bit overhead.
 *
 * HONESTY: the course source gives NO perplexity/accuracy numbers, so this
 * widget renders NO quality bar — only the EXACT scale counts, group sizes, and
 * effective-bits arithmetic above, plus a qualitative "error" direction word.
 */

type Gran = 'tensor' | 'channel' | 'group128' | 'block32';
const GRANS: Gran[] = ['tensor', 'channel', 'group128', 'block32'];

// Tailwind palette hexes for SVG (SVG can't take TW classes).
const EMERALD = '#10b981'; // lower error / good direction

// Grid geometry: an 8-row × 16-col weight matrix = 128 cells. 128 is exactly
// one per-group-128 group, so the whole grid is one group in that mode — handy.
const ROWS = 8;
const COLS = 16;
const CELLS = ROWS * COLS; // 128

// ── Per-granularity ground truth (exact — see file JSDoc for the arithmetic). ──
type GranInfo = {
  /** number of scales for THIS drawn 8×16 = 128-cell matrix */
  scales: number;
  /** weights covered by one scale (the group size), or null for whole-tensor */
  groupSize: number | null;
  /** effective bits/param (base int4 + scale overhead), EXACT */
  effBits: number;
  /** bits in one scale, for the readout */
  scaleBits: number;
  /** maps a cell index (0..127) → which scale-group it belongs to (for tinting) */
  groupOf: (idx: number) => number;
};

const INFO: Record<Gran, GranInfo> = {
  tensor: {
    scales: 1,
    groupSize: null,
    effBits: 4.0,
    scaleBits: 16,
    groupOf: () => 0,
  },
  channel: {
    // one scale per output row (per-channel = per output channel = per row here)
    scales: ROWS, // 8
    groupSize: COLS, // 16 per row in this drawing
    effBits: 4.0, // a real row is long → overhead ≈ 0; labelled ≈ 4.00
    scaleBits: 16,
    groupOf: (i) => Math.floor(i / COLS), // 0..7 by row
  },
  group128: {
    scales: 1, // 128 cells / 128-per-group = exactly 1 group in this drawing
    groupSize: 128,
    effBits: 4.125, // 4 + 16/128
    scaleBits: 16,
    groupOf: () => 0,
  },
  block32: {
    // MX fixes the block at 32 → 128 / 32 = 4 blocks across this matrix
    scales: CELLS / 32, // 4
    groupSize: 32,
    effBits: 4.25, // 4 + 8/32
    scaleBits: 8, // E8M0
    groupOf: (i) => Math.floor(i / 32), // 0..3 in row-major order
  },
};

const BASE_BITS = 4; // int4 weight payload
const MAX_EFF = 4.25; // the top of the effective-bits readout (block32 / group64)

// Tint palette: a few opacities of var(--primary) so adjacent groups read as
// distinct "one shared scale" regions. Cycles for >4 groups (per-channel = 8).
const TINTS = [0.85, 0.55, 0.32, 0.68, 0.2, 0.45, 0.75, 0.38];
function tintFor(group: number): number {
  return TINTS[group % TINTS.length];
}

// ── Per-locale copy. en is the original; zh translates the prose but KEEPS the
// glossary terms in English (per-tensor, per-channel, per-group, per-block,
// block, scale, scale factor, int4, mxfp4, E8M0, MX, weight, bits/param,
// quantization, …) and uses full-width punctuation. The conceptual term
// granularity is translated to 粒度, matching the chapter prose. ───────────────
const COPY = {
  en: {
    heading: 'Granularity — one scale for how many weights?',
    granLabel: 'granularity',
    pickAria: (g: string) => `Show ${g} granularity`,
    names: {
      tensor: 'per-tensor',
      channel: 'per-channel',
      group128: 'per-group-128',
      block32: 'per-block-32 (MX)',
    } satisfies Record<Gran, string>,
    matrixHeading: 'weight matrix · one tint = one shared scale',
    matrixCaption: `${ROWS} rows × ${COLS} cols = ${CELLS} weights`,
    scalesHeading: 'scales stored',
    scalesValue: (n: number) => `${n}`,
    groupNote: {
      tensor: '1 scale over the whole tensor',
      channel: '1 scale per output channel (row)',
      group128: '1 scale per 128-weight group',
      block32: '1 scale per fixed 32-weight block',
    } satisfies Record<Gran, string>,
    effHeading: 'effective bits/param',
    effValue: (b: number) => `${b.toFixed(b % 1 === 0 ? 2 : 3)}`,
    effMath: {
      tensor: 'int4 + 16-bit ÷ tensor ≈ 4.00',
      channel: 'int4 + 16-bit ÷ row ≈ 4.00',
      group128: 'int4 + 16-bit ÷ 128 = 4.125',
      block32: 'int4 + 8-bit ÷ 32 = 4.25',
    } satisfies Record<Gran, string>,
    errorHeading: 'rounding error',
    errorWord: {
      tensor: 'coarsest — widest range per scale',
      channel: 'tighter — one range per row',
      group128: 'tight — 128 values per scale',
      block32: 'tightest — only 32 values per scale',
    } satisfies Record<Gran, string>,
    errorScale: 'ordinal · not measured',
    rule: (
      <>
        The rule: <strong>finer granularity = lower rounding error, higher bit overhead</strong>. Each scale is fit to a
        smaller set of values, so its step size is tighter and rounding error drops — but you store more scales, and the
        scales cost bits too. That is why per-group-64 (<span className="font-mono">4 + 16/64 = 4.25</span>) and the MX
        per-block-32 (<span className="font-mono">4 + 8/32 = 4.25</span>) land at the same effective width by different
        routes: a fatter scale over a bigger block, or a leaner 8-bit <span className="font-mono">E8M0</span> scale over
        a smaller one.
      </>
    ),
    mx: (
      <>
        The <strong>MX</strong> (Microscaling) standard takes this to a <em>fixed</em> 32-element block with a shared
        8-bit <span className="font-mono">E8M0</span> power-of-two scale — so <span className="font-mono">mxfp4</span>{' '}
        is int4-style 4-bit elements plus one tiny scale every 32 weights, a hardware-friendly fixed granularity.
      </>
    ),
    aria: (g: string, scales: number, eff: number) =>
      `Granularity diagram at ${g}: an ${ROWS}-by-${COLS} weight matrix tinted by scale-group, ${scales} scales stored, and an effective ${eff} bits per parameter readout. Finer granularity lowers rounding error but raises the bit overhead.`,
  },
  zh: {
    heading: 'granularity——一个 scale 覆盖多少个 weight？',
    granLabel: '粒度',
    pickAria: (g: string) => `显示 ${g} 粒度`,
    names: {
      tensor: 'per-tensor',
      channel: 'per-channel',
      group128: 'per-group-128',
      block32: 'per-block-32 (MX)',
    } satisfies Record<Gran, string>,
    matrixHeading: 'weight 矩阵 · 同一色调 = 同一个共享 scale',
    matrixCaption: `${ROWS} 行 × ${COLS} 列 = ${CELLS} 个 weight`,
    scalesHeading: 'scale 数量',
    scalesValue: (n: number) => `${n}`,
    groupNote: {
      tensor: '整个 tensor 共用 1 个 scale',
      channel: '每个输出 channel（行）1 个 scale',
      group128: '每 128 个 weight 1 个 scale',
      block32: '每 32 个 weight 的固定 block 1 个 scale',
    } satisfies Record<Gran, string>,
    effHeading: 'effective bits/param',
    effValue: (b: number) => `${b.toFixed(b % 1 === 0 ? 2 : 3)}`,
    effMath: {
      tensor: 'int4 + 16-bit ÷ tensor ≈ 4.00',
      channel: 'int4 + 16-bit ÷ row ≈ 4.00',
      group128: 'int4 + 16-bit ÷ 128 = 4.125',
      block32: 'int4 + 8-bit ÷ 32 = 4.25',
    } satisfies Record<Gran, string>,
    errorHeading: '舍入误差',
    errorWord: {
      tensor: '最粗——每个 scale 覆盖的范围最宽',
      channel: '更紧——每行一个范围',
      group128: '紧——每个 scale 覆盖 128 个值',
      block32: '最紧——每个 scale 只覆盖 32 个值',
    } satisfies Record<Gran, string>,
    errorScale: '序数 · 非测量',
    rule: (
      <>
        规律：<strong>粒度越细 = 舍入误差越低，bit 开销越高</strong>。每个 scale
        拟合到更小的一组值上，步长更紧、舍入误差下降——但你要存更多 scale，而 scale 本身也占 bit。这就是为什么
        per-group-64（<span className="font-mono">4 + 16/64 = 4.25</span>）和 MX 的 per-block-32（
        <span className="font-mono">4 + 8/32 = 4.25</span>）会以不同路径落到相同的有效位宽：要么用更胖的 scale
        覆盖更大的 block，要么用更精简的 8-bit <span className="font-mono">E8M0</span> scale 覆盖更小的 block。
      </>
    ),
    mx: (
      <>
        <strong>MX</strong>（Microscaling）标准把这个做法推到 <em>固定</em> 的 32 元素 block，配一个共享的 8-bit{' '}
        <span className="font-mono">E8M0</span> 二次幂 scale——所以 <span className="font-mono">mxfp4</span> 就是 int4
        风格的 4-bit 元素，外加每 32 个 weight 一个小 scale，是一种对硬件友好的固定粒度。
      </>
    ),
    aria: (g: string, scales: number, eff: number) =>
      `${g} 粒度下的 granularity 图：${ROWS}×${COLS} 的 weight 矩阵按 scale 分组着色，存储 ${scales} 个 scale，有效每参数 ${eff} bit。粒度越细舍入误差越低，但 bit 开销越高。`,
  },
} as const;

// ── SVG geometry. Sized for the EN labels (wider than zh). One grid on the
// left, two stacked readouts (scales / effective bits) on the right, with their
// section labels ABOVE the value — never wedged beside the grid. ──────────────
const VB_W = 560;
const VB_H = 252;
const PAD = 14;

// Grid block (left).
const GRID_LABEL_Y = 16; // matrix heading text baseline
const GRID_X = PAD;
const GRID_Y = 26;
const GRID_W = 300;
const CELL = GRID_W / COLS; // 18.75
const GRID_H = CELL * ROWS; // 150
const GRID_CAP_Y = GRID_Y + GRID_H + 16; // caption below grid

// Right column (readouts) — starts well right of the grid so labels never collide.
const RIGHT_X = GRID_X + GRID_W + 28; // 342
const RIGHT_W = VB_W - RIGHT_X - PAD; // 204

// scales readout.
const SCALES_LABEL_Y = 16;
const SCALES_VAL_Y = 58; // big number baseline
const SCALES_NOTE_Y = 78; // group note under the number

// effective-bits readout.
const EFF_LABEL_Y = 110;
const EFF_BAR_Y = 120;
const EFF_BAR_H = 22;
const EFF_MATH_Y = 168;

// rounding-error ordinal pips.
const ERR_LABEL_Y = 196;
const ERR_PIP_Y = 204;
const ERR_PIP_H = 18;
const ERR_PIPS = 4;

export function GranularityZoom() {
  const copy = COPY[useLocale()];
  const [gran, setGran] = React.useState<Gran>('tensor');
  const info = INFO[gran];

  // effective-bits fill: map [BASE_BITS .. MAX_EFF] → [0 .. RIGHT_W]. The bar
  // exaggerates the small overhead range so the tick-up is visible.
  const effFrac = (info.effBits - BASE_BITS) / (MAX_EFF - BASE_BITS); // 0..1
  const effW = Math.max(2, effFrac * RIGHT_W);

  // rounding-error ordinal: more groups → tighter range → "lower error". We map
  // the four drawn modes to an ordinal 1..4 of how tight the range is (NOT a
  // measurement — the course gives no perplexity numbers).
  const errOrdinal: Record<Gran, number> = { tensor: 1, channel: 2, group128: 3, block32: 4 };
  const filledPips = errOrdinal[gran];

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title + granularity ladder */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>
        <div className="flex items-center gap-1 text-[11px] text-muted-foreground">
          <span className="hidden sm:inline">{copy.granLabel}:</span>
          <div className="flex overflow-hidden rounded border border-border">
            {GRANS.map((g, i) => (
              <button
                key={g}
                type="button"
                onClick={() => setGran(g)}
                aria-pressed={gran === g}
                aria-label={copy.pickAria(copy.names[g])}
                className={[
                  'px-2 py-1 font-mono text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                  i > 0 ? 'border-l border-border' : '',
                ].join(' ')}
                style={{
                  background: gran === g ? 'var(--primary)' : 'transparent',
                  color: gran === g ? 'var(--background)' : 'var(--muted-foreground)',
                }}
              >
                {copy.names[g]}
              </button>
            ))}
          </div>
        </div>
      </div>

      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        className="w-full"
        role="img"
        aria-label={copy.aria(copy.names[gran], info.scales, info.effBits)}
      >
        {/* ── weight-matrix grid (left), tinted by scale-group ── */}
        <text
          x={GRID_X}
          y={GRID_LABEL_Y}
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[11px] font-medium"
        >
          {copy.matrixHeading}
        </text>
        {Array.from({ length: CELLS }).map((_, idx) => {
          const r = Math.floor(idx / COLS);
          const c = idx % COLS;
          const group = info.groupOf(idx);
          return (
            <rect
              key={idx}
              x={GRID_X + c * CELL}
              y={GRID_Y + r * CELL}
              width={CELL - 1.4}
              height={CELL - 1.4}
              rx={2}
              style={{ fill: 'var(--primary)', opacity: tintFor(group) }}
            />
          );
        })}
        {/* group boundary outlines so the "one shared scale" regions read clearly */}
        {gran === 'block32'
          ? // 4 horizontal blocks of 2 rows each (32 = 2 rows × 16 cols)
            Array.from({ length: ERR_PIPS }).map((_, b) => (
              <rect
                key={`b${b}`}
                x={GRID_X}
                y={GRID_Y + b * 2 * CELL}
                width={GRID_W}
                height={2 * CELL - 1.4}
                rx={3}
                fill="none"
                style={{ stroke: 'var(--foreground)', opacity: 0.45 }}
                strokeWidth={1.4}
              />
            ))
          : null}
        {gran === 'channel'
          ? // per-row outline
            Array.from({ length: ROWS }).map((_, r) => (
              <rect
                key={`r${r}`}
                x={GRID_X}
                y={GRID_Y + r * CELL}
                width={GRID_W}
                height={CELL - 1.4}
                rx={2}
                fill="none"
                style={{ stroke: 'var(--foreground)', opacity: 0.3 }}
                strokeWidth={1}
              />
            ))
          : null}
        {gran === 'tensor' || gran === 'group128' ? (
          // whole-matrix outline (one scale)
          <rect
            x={GRID_X}
            y={GRID_Y}
            width={GRID_W}
            height={GRID_H}
            rx={3}
            fill="none"
            style={{ stroke: 'var(--foreground)', opacity: 0.45 }}
            strokeWidth={1.4}
          />
        ) : null}
        <text x={GRID_X} y={GRID_CAP_Y} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
          {copy.matrixCaption}
        </text>

        {/* ── right column readouts ── */}
        {/* scales stored */}
        <text
          x={RIGHT_X}
          y={SCALES_LABEL_Y}
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[11px] font-medium"
        >
          {copy.scalesHeading}
        </text>
        <text
          x={RIGHT_X}
          y={SCALES_VAL_Y}
          style={{ fill: 'var(--primary)' }}
          className="font-mono text-[34px] font-semibold"
        >
          {copy.scalesValue(info.scales)}
        </text>
        <text x={RIGHT_X} y={SCALES_NOTE_Y} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
          {copy.groupNote[gran]}
        </text>

        {/* effective bits/param — value sits ABOVE the bar, math line below it */}
        <text
          x={RIGHT_X}
          y={EFF_LABEL_Y}
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[11px] font-medium"
        >
          {copy.effHeading}
        </text>
        <text
          x={RIGHT_X + RIGHT_W}
          y={EFF_LABEL_Y}
          textAnchor="end"
          style={{ fill: 'var(--foreground)' }}
          className="font-mono text-[11px] font-semibold"
        >
          {copy.effValue(info.effBits)}
        </text>
        {/* track ghost (the 4.25 ceiling) */}
        <rect
          x={RIGHT_X}
          y={EFF_BAR_Y}
          width={RIGHT_W}
          height={EFF_BAR_H}
          rx={4}
          fill="none"
          style={{ stroke: 'var(--border)' }}
          strokeWidth={1}
          strokeDasharray="3 3"
        />
        <rect
          x={RIGHT_X}
          y={EFF_BAR_Y}
          width={effW}
          height={EFF_BAR_H}
          rx={4}
          style={{ fill: 'var(--primary)', opacity: 0.55 }}
        />
        {/* base int4 (4.00) reference tick at the left edge of the overhead range */}
        <text x={RIGHT_X} y={EFF_MATH_Y} style={{ fill: 'var(--muted-foreground)' }} className="font-mono text-[10px]">
          {copy.effMath[gran]}
        </text>

        {/* ── rounding-error ordinal pips (NOT measured — no perplexity numbers) ── */}
        <text
          x={RIGHT_X}
          y={ERR_LABEL_Y}
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[11px] font-medium"
        >
          {copy.errorHeading}
        </text>
        <text
          x={RIGHT_X + RIGHT_W}
          y={ERR_LABEL_Y}
          textAnchor="end"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[10px]"
        >
          {copy.errorScale}
        </text>
        {Array.from({ length: ERR_PIPS }).map((_, i) => {
          const gap = 6;
          const pipW = (RIGHT_W - gap * (ERR_PIPS - 1)) / ERR_PIPS;
          const filled = i < filledPips;
          return (
            <rect
              key={i}
              x={RIGHT_X + i * (pipW + gap)}
              y={ERR_PIP_Y}
              width={pipW}
              height={ERR_PIP_H}
              rx={3}
              style={{
                fill: filled ? EMERALD : 'var(--muted)',
                opacity: filled ? 0.55 : 1,
                stroke: 'var(--border)',
              }}
              strokeWidth={1}
            />
          );
        })}
      </svg>

      {/* per-granularity error-direction line */}
      <p className="min-h-[2.5rem] text-[13px] text-foreground/90">
        <span className="font-mono text-foreground">{copy.names[gran]}</span> — {copy.errorWord[gran]}
      </p>

      {/* The rule + the MX standard */}
      <p className="border-t border-border/60 pt-3 text-[12px] text-muted-foreground">{copy.rule}</p>
      <p className="text-[12px] text-muted-foreground">{copy.mx}</p>
    </div>
  );
}

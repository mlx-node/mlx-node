import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * RangePrecisionScatter — every numeric format is a single point on a 2-D
 * tradeoff: x = dynamic range (how many decades of magnitude it can span) and
 * y = precision (how many decimal digits it pins down). The scatter makes the
 * Pareto frontier visible and shows WHY two fp8 / fp6 formats exist per
 * bit-width: at a fixed budget you spend bits on EITHER more exponent (range,
 * E5M2 / E3M2) OR more mantissa (precision, E4M3 / E2M3) — opposite corners.
 *
 * Self-contained and STATIC: NO worker, NO model, NO WASM, NO network, NO
 * animation — pure React + Tailwind/SVG, so it server-renders for crawlers and
 * is deterministic server-vs-client. The only state is which dot is selected
 * (a click readout), so it needs no prefers-reduced-motion handling.
 *
 * EXACT numbers (source-verified; every range is "approximate decades" and is
 * log10 of the exponent span, every precision is log10 of the mantissa levels):
 *   - fp32  range≈83  precision≈7.2   (8-bit exponent, 23-bit mantissa)
 *   - tf32  range≈83  precision≈3.3   (fp32 exponent range, fp16-class 10-bit mantissa)
 *   - bf16  range≈78  precision≈2.4   (8-bit exponent; log10(2^8)=2.41)
 *   - fp16  range≈12  precision≈3.3   (max 65504, min-subnormal 2^-24 ≈ 5.96e-8 → ~12 decades;
 *                                       like fp32/bf16 above, the span counts subnormals)
 *   - fp8 E4M3  range≈5.4  precision≈1.2   (~18 binades; log10(2^4)=1.2)
 *   - fp8 E5M2  range≈9.6  precision≈0.9   (~32 binades; log10(2^3)=0.9)
 *   - fp6 E3M2  range≈3.0  precision≈0.9
 *   - fp6 E2M3  range≈1.8  precision≈1.2
 *   - fp4 E2M1  range≈1.1  precision≈0.6
 *   - int8 / int4 — UNIFORM grids. An integer format has NO exponent, so it has
 *     NO inherent dynamic range at all: its range comes only from an external
 *     scale factor. They sit OFF this map, pinned to the y-axis edge.
 *
 * HONESTY: range/precision are exact and rendered precisely. There is no
 * quality/accuracy axis here — the course source gives no perplexity numbers,
 * so we never draw one. "Better" = up-and-to-the-right, but it costs bits.
 */

type Fmt = 'fp32' | 'tf32' | 'bf16' | 'fp16' | 'fp8 E4M3' | 'fp8 E5M2' | 'fp6 E3M2' | 'fp6 E2M3' | 'fp4 E2M1';

// Tailwind hexes allowed in SVG: EMERALD = the "dense / both-good" corner hint,
// RED-ish for integers (no real dynamic range). Floats use var(--primary).
const EMERALD = '#10b981';
const RED = '#ef4444';

type Pt = {
  fmt: Fmt;
  /** approximate dynamic range in decades (log10 of the exponent span) */
  range: number;
  /** precision in decimal digits (log10 of the mantissa levels) */
  precision: number;
  /** label offset from the dot (dx, dy) + text anchor — hand-staggered so no two labels collide */
  dx: number;
  dy: number;
  anchor: 'start' | 'middle' | 'end';
  /** the sibling at the same bit-width but the opposite corner (for the connector hint) */
  pair?: Fmt;
};

// Hand-tuned label offsets are verified collision-free at viewBox 640×412 (see
// the JSDoc geometry); the dense bottom-left cluster is spread by a log x-axis.
const PTS: Pt[] = [
  { fmt: 'fp32', range: 83, precision: 7.2, dx: 9, dy: 4, anchor: 'start' },
  { fmt: 'tf32', range: 83, precision: 3.3, dx: 9, dy: 4, anchor: 'start' },
  { fmt: 'bf16', range: 78, precision: 2.4, dx: 9, dy: 4, anchor: 'start' },
  { fmt: 'fp16', range: 12, precision: 3.3, dx: 0, dy: -12, anchor: 'middle' },
  { fmt: 'fp8 E4M3', range: 5.4, precision: 1.2, dx: -6, dy: -46, anchor: 'end', pair: 'fp8 E5M2' },
  { fmt: 'fp8 E5M2', range: 9.6, precision: 0.9, dx: 8, dy: -16, anchor: 'start', pair: 'fp8 E4M3' },
  { fmt: 'fp6 E3M2', range: 3.0, precision: 0.9, dx: 8, dy: -30, anchor: 'start', pair: 'fp6 E2M3' },
  { fmt: 'fp6 E2M3', range: 1.8, precision: 1.2, dx: -8, dy: -16, anchor: 'end', pair: 'fp6 E3M2' },
  { fmt: 'fp4 E2M1', range: 1.1, precision: 0.6, dx: 0, dy: -14, anchor: 'middle' },
];

// Integer formats live OFF the range axis (uniform grid). Stacked on the y-axis
// edge at nominal heights, clearly labelled "uniform · external scale".
const INTS: { fmt: 'int8' | 'int4'; nominalY: number }[] = [
  // nominal heights ONLY for vertical placement on the y-axis edge — these are
  // NOT precision claims (an integer's "precision" depends on its scale). They
  // sit clear of the float cluster so the labels never collide.
  { fmt: 'int8', nominalY: 3.2 },
  { fmt: 'int4', nominalY: 1.5 },
];

// ── Per-locale copy. en = original English. zh translates the prose only and
// keeps every glossary term English (fp32, tf32, bf16, fp16, fp8, fp6, fp4,
// E4M3, E5M2, E3M2, E2M3, E2M1, int8, int4, exponent, mantissa, sign, scale,
// scale factor, dynamic range, precision, decade, bits, Pareto). Full-width
// punctuation in the zh prose; numbers/units never translate. ─────────────────
const COPY = {
  en: {
    heading: 'Range vs precision — every format is one point',
    xAxis: 'dynamic range (approximate decades) →',
    yAxis: 'precision (decimal digits) →',
    floats: 'floating point',
    integers: 'integers (uniform)',
    intNote: 'uniform grid · range comes only from an external scale',
    pairHint: 'same bit-width, opposite corner',
    better: 'better',
    selectAria: (f: string) => `Show the range and precision readout for ${f}`,
    readoutRange: (r: number) => `range ≈ ${r} decades`,
    readoutPrec: (p: number) => `precision ≈ ${p} digits`,
    readoutInt: 'uniform grid — no inherent dynamic range',
    tapHint: 'tap a point',
    caption: (
      <>
        Up-and-to-the-right is better — but it costs bits. At a <strong>fixed bit-width</strong> you must pick a{' '}
        <strong>corner</strong>: more exponent buys <em>dynamic range</em> (<span className="font-mono">fp8 E5M2</span>,{' '}
        <span className="font-mono">fp6 E3M2</span>), more mantissa buys <em>precision</em> (
        <span className="font-mono">fp8 E4M3</span>, <span className="font-mono">fp6 E2M3</span>) — the same total bits,
        opposite trade. Integers sit off this map entirely: a <strong>uniform grid</strong> has no inherent dynamic
        range, so its reach depends wholly on an external <span className="font-mono">scale</span>. Range numbers are
        approximate decades.
      </>
    ),
    svgAria:
      'Scatter plot of numeric formats. The x-axis is dynamic range in approximate decades (log scale); the y-axis is precision in decimal digits. Floating-point formats spread across the plane — fp32 sits top-right with the most range and precision, while fp8, fp6, and fp4 formats cluster at the low-range, low-precision corner. Each bit-width offers two formats at opposite corners: one favoring exponent for range, one favoring mantissa for precision. Integer formats int8 and int4 are pinned to the y-axis edge because a uniform grid has no inherent dynamic range.',
  },
  zh: {
    heading: 'range 与 precision——每个格式都是一个点',
    xAxis: 'dynamic range（约多少个 decade）→',
    yAxis: 'precision（十进制位数）→',
    floats: 'floating point',
    integers: 'integers（uniform）',
    intNote: 'uniform 格点 · range 只来自外部的 scale',
    pairHint: '同样的 bit 宽度，相反的角',
    better: '更好',
    selectAria: (f: string) => `显示 ${f} 的 range 与 precision 读数`,
    readoutRange: (r: number) => `range ≈ ${r} 个 decade`,
    readoutPrec: (p: number) => `precision ≈ ${p} 位`,
    readoutInt: 'uniform 格点——没有内在的 dynamic range',
    tapHint: '点一个点',
    caption: (
      <>
        越靠右上越好——但这要用 bit 来换。在 <strong>固定的 bit 宽度</strong> 下，你必须挑一个 <strong>角</strong>：更多
        exponent 换来 <em>dynamic range</em>（<span className="font-mono">fp8 E5M2</span>、
        <span className="font-mono">fp6 E3M2</span>），更多 mantissa 换来 <em>precision</em>（
        <span className="font-mono">fp8 E4M3</span>、<span className="font-mono">fp6 E2M3</span>）——同样的总 bit 数，
        相反的取舍。integers 完全在这张图之外：<strong>uniform 格点</strong> 没有内在的 dynamic
        range，它能触及的范围全靠外部的 <span className="font-mono">scale</span>。range 的数字是约略的 decade。
      </>
    ),
    svgAria:
      '数值格式的散点图。x 轴是 dynamic range（约多少个 decade，对数刻度），y 轴是 precision（十进制位数）。floating-point 格式铺满整个平面——fp32 在右上角，range 与 precision 都最高，而 fp8、fp6、fp4 挤在低 range、低 precision 的角落。每个 bit 宽度都提供两个位于相反角的格式：一个偏重 exponent 换 range，一个偏重 mantissa 换 precision。integer 格式 int8 与 int4 被钉在 y 轴边缘，因为 uniform 格点没有内在的 dynamic range。',
  },
} as const;

// ── SVG geometry. Sized for the wider EN labels. The bottom-left cluster (all
// the low-range formats) is spread by a LOG x-axis (decades → linear pixels),
// which matches the "approximate decades" semantics and avoids label pileups. ──
const VB_W = 640;
const VB_H = 412;
const PLOT_X = 120; // left edge of plot (room for y-axis title + int markers)
const PLOT_Y = 44; // top edge
const PLOT_W = 486;
const PLOT_H = 292; // plot floor at PLOT_Y + PLOT_H = 336
const X_MAX_L = Math.log10(100); // log x-axis spans 1 → 100 decades

// scale helpers (x is log10; the int axis edge sits at the x=1 → log 0 origin).
const sx = (range: number) => PLOT_X + (Math.log10(range) / X_MAX_L) * PLOT_W;
const sy = (digits: number) => PLOT_Y + PLOT_H - (digits / 8) * PLOT_H; // y from 0..8 digits

const X_GRID = [1, 3, 10, 30, 100]; // labelled decades incl. 10/40-ish/80-ish
const Y_GRID = [0, 2, 4, 6, 8];

export function RangePrecisionScatter() {
  const copy = COPY[useLocale()];
  // default selection = bf16, the format the course demo actually ships
  const [sel, setSel] = React.useState<Fmt | 'int8' | 'int4'>('bf16');

  const selPt = PTS.find((p) => p.fmt === sel);
  const selInt = INTS.find((p) => p.fmt === sel);

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>
        <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
          <span className="inline-flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-full" style={{ background: 'var(--primary)' }} />
            {copy.floats}
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-sm" style={{ background: RED }} />
            {copy.integers}
          </span>
        </div>
      </div>

      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.svgAria}>
        <defs>
          {/* arrowhead for the "better →" guide (emerald = up-and-to-the-right ideal) */}
          <marker
            id="rps-better-arrow"
            viewBox="0 0 10 10"
            refX="8"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path d="M0 0 L10 5 L0 10 z" fill={EMERALD} opacity={0.75} />
          </marker>
        </defs>

        {/* ── axes frame ── */}
        <rect
          x={PLOT_X}
          y={PLOT_Y}
          width={PLOT_W}
          height={PLOT_H}
          fill="none"
          style={{ stroke: 'var(--border)' }}
          strokeWidth={1}
        />

        {/* x gridlines + decade tick labels */}
        {X_GRID.map((g) => {
          const x = sx(g);
          return (
            <g key={`x${g}`}>
              <line
                x1={x}
                y1={PLOT_Y}
                x2={x}
                y2={PLOT_Y + PLOT_H}
                style={{ stroke: 'var(--border)', opacity: 0.45 }}
                strokeWidth={1}
                strokeDasharray="2 3"
              />
              <text
                x={x}
                y={PLOT_Y + PLOT_H + 15}
                textAnchor="middle"
                style={{ fill: 'var(--muted-foreground)' }}
                className="font-mono text-[10px]"
              >
                {g}
              </text>
            </g>
          );
        })}

        {/* y gridlines + digit tick labels */}
        {Y_GRID.map((g) => {
          const y = sy(g);
          return (
            <g key={`y${g}`}>
              <line
                x1={PLOT_X}
                y1={y}
                x2={PLOT_X + PLOT_W}
                y2={y}
                style={{ stroke: 'var(--border)', opacity: 0.45 }}
                strokeWidth={1}
                strokeDasharray="2 3"
              />
              <text
                x={PLOT_X - 8}
                y={y + 3}
                textAnchor="end"
                style={{ fill: 'var(--muted-foreground)' }}
                className="font-mono text-[10px]"
              >
                {g}
              </text>
            </g>
          );
        })}

        {/* axis titles — placed clear of ticks; x below the plot, y rotated at far left */}
        <text
          x={PLOT_X + PLOT_W / 2}
          y={PLOT_Y + PLOT_H + 32}
          textAnchor="middle"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[11px] font-medium"
        >
          {copy.xAxis}
        </text>
        <text
          transform={`translate(22, ${PLOT_Y + PLOT_H / 2}) rotate(-90)`}
          textAnchor="middle"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[11px] font-medium"
        >
          {copy.yAxis}
        </text>

        {/* ── "better →" guide arrow toward the fp32 corner (emerald = the ideal) ── */}
        <line
          x1={430}
          y1={150}
          x2={540}
          y2={92}
          style={{ stroke: EMERALD, opacity: 0.55 }}
          strokeWidth={1.5}
          markerEnd="url(#rps-better-arrow)"
        />
        <text
          x={452}
          y={140}
          textAnchor="middle"
          style={{ fill: EMERALD, opacity: 0.85 }}
          className="text-[10px] font-medium italic"
        >
          {copy.better}
        </text>

        {/* ── "same bit-width, opposite corner" connector pairs (faint) ── */}
        {PTS.filter((p) => p.pair && p.fmt < (p.pair as string)).map((p) => {
          const a = p;
          const b = PTS.find((q) => q.fmt === p.pair)!;
          return (
            <line
              key={`pair-${p.fmt}`}
              x1={sx(a.range)}
              y1={sy(a.precision)}
              x2={sx(b.range)}
              y2={sy(b.precision)}
              style={{ stroke: 'var(--primary)', opacity: 0.35 }}
              strokeWidth={1.5}
              strokeDasharray="4 3"
            />
          );
        })}

        {/* ── integer markers on the y-axis edge (uniform · external scale) ── */}
        {INTS.map((it) => {
          const cy = sy(it.nominalY);
          const active = sel === it.fmt;
          return (
            <g
              key={it.fmt}
              role="button"
              tabIndex={0}
              aria-label={copy.selectAria(it.fmt)}
              aria-pressed={active}
              className="cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              onClick={() => setSel(it.fmt)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setSel(it.fmt);
                }
              }}
            >
              {/* small square on the axis line, distinct from the round float dots */}
              <rect
                x={PLOT_X - 5}
                y={cy - 5}
                width={10}
                height={10}
                rx={1.5}
                style={{ fill: RED, opacity: active ? 1 : 0.8, stroke: 'var(--background)' }}
                strokeWidth={1}
              />
              {/* label sits LEFT of the square; the int nominal heights are kept
                  OFF the y-axis gridlines (0/2/4/6/8) so these labels never share a
                  row with a tick number, and being left of the axis they never reach
                  the float dot labels either */}
              <text
                x={PLOT_X - 12}
                y={cy + 3}
                textAnchor="end"
                style={{ fill: active ? RED : 'var(--muted-foreground)' }}
                className="font-mono text-[11px] font-semibold"
              >
                {it.fmt}
              </text>
            </g>
          );
        })}
        {/* the uniform-grid note, parked in the empty mid-left region */}
        <text x={PLOT_X + 10} y={PLOT_Y + 64} style={{ fill: RED, opacity: 0.85 }} className="text-[10px] italic">
          {copy.intNote}
        </text>

        {/* ── float dots + staggered leader labels ── */}
        {PTS.map((p) => {
          const cx = sx(p.range);
          const cy = sy(p.precision);
          const lx = cx + p.dx;
          const ly = cy + p.dy;
          const active = sel === p.fmt;
          // leader line from dot to label when the label is offset far (cluster)
          const farLabel = Math.abs(p.dy) > 14 || Math.abs(p.dx) > 12;
          return (
            <g
              key={p.fmt}
              role="button"
              tabIndex={0}
              aria-label={copy.selectAria(p.fmt)}
              aria-pressed={active}
              className="cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
              onClick={() => setSel(p.fmt)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  setSel(p.fmt);
                }
              }}
            >
              {farLabel ? (
                <line
                  x1={cx}
                  y1={cy}
                  x2={lx}
                  y2={ly - 3}
                  style={{ stroke: 'var(--muted-foreground)', opacity: 0.4 }}
                  strokeWidth={1}
                />
              ) : null}
              {active ? <circle cx={cx} cy={cy} r={9} style={{ fill: 'var(--primary)', opacity: 0.18 }} /> : null}
              <circle
                cx={cx}
                cy={cy}
                r={active ? 5.5 : 4.5}
                style={{ fill: 'var(--primary)', opacity: active ? 1 : 0.85, stroke: 'var(--background)' }}
                strokeWidth={1}
              />
              <text
                x={lx}
                y={ly}
                textAnchor={p.anchor}
                style={{ fill: active ? 'var(--foreground)' : 'var(--muted-foreground)' }}
                className={['font-mono text-[11px]', active ? 'font-semibold' : ''].join(' ')}
              >
                {p.fmt}
              </text>
            </g>
          );
        })}

        {/* pair-hint label, riding the fp8 connector (its midpoint is in open space) */}
        {(() => {
          const a = PTS.find((q) => q.fmt === 'fp8 E4M3')!;
          const b = PTS.find((q) => q.fmt === 'fp8 E5M2')!;
          const mx = (sx(a.range) + sx(b.range)) / 2;
          const my = (sy(a.precision) + sy(b.precision)) / 2 + 14;
          return (
            <text
              x={mx}
              y={my}
              textAnchor="middle"
              style={{ fill: 'var(--primary)', opacity: 0.7 }}
              className="text-[9px] italic"
            >
              {copy.pairHint}
            </text>
          );
        })()}
      </svg>

      {/* readout for the selected point (deterministic; updates on click/Enter) */}
      <div className="min-h-[1.75rem] text-[13px] text-foreground/90">
        {selPt ? (
          <span>
            <span className="font-mono text-foreground">{selPt.fmt}</span> — {copy.readoutRange(selPt.range)} ·{' '}
            {copy.readoutPrec(selPt.precision)}
          </span>
        ) : selInt ? (
          <span>
            <span className="font-mono text-foreground">{selInt.fmt}</span> — {copy.readoutInt}
          </span>
        ) : (
          <span className="text-muted-foreground">{copy.tapHint}</span>
        )}
      </div>

      {/* caption */}
      <p className="border-t border-border/60 pt-3 text-[12px] text-muted-foreground">{copy.caption}</p>
    </div>
  );
}

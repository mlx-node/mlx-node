import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * QuantizeRoundTrip — the core mechanic of quantization on a number line.
 *
 * A continuous real value x is snapped to the NEAREST tick of a discrete grid;
 * the leftover is the quantization error. Symmetric signed grid, zero-point 0,
 * fixed real range [-1, 1]. The bit-width B picks the grid resolution: the
 * integer codes run q ∈ [−qmax, qmax] with qmax = 2^(B−1) − 1, so there are
 * 2^B − 1 ticks — one code is dropped to keep 0 exactly on a tick, exactly as
 * int8 uses −127..127. The snap x̂ = s·q therefore always lands on a drawn tick
 * and never leaves [-1, 1]. A finer grid (more bits) visibly shrinks the error
 * — at the cost of storage.
 *
 * Self-contained and STATIC: NO worker, NO model, NO WASM, NO network, NO
 * animation — pure React + Tailwind/SVG. First render is deterministic and
 * identical server vs client; the only state is the two UI selections (x slider
 * and bit-width B), so it needs no prefers-reduced-motion handling and SSRs
 * cleanly for crawlers (createRoot replaces it on boot).
 *
 * EVERY NUMBER IS DERIVED LIVE, NOT INVENTED. The grid, step, snap and error are
 * computed from x, B and the fixed range by the textbook symmetric formulas
 * (q = round(x / s), x̂ = s·q, error ≤ s/2; the error is ~uniform on ±s/2 with
 * variance s²/12). There are NO quality/accuracy numbers here — only the exact
 * arithmetic of mapping a real onto a uniform grid. The next diagram adds a
 * non-zero zero-point for non-centered (asymmetric) data.
 */

// Fixed real range for the number line: [-1, 1] (symmetric, zero-point 0).
const X_MIN = -1;
const X_MAX = 1;
const RANGE = X_MAX - X_MIN; // 2

// Selectable bit-widths. Symmetric signed quantizer: codes q ∈ [−qmax, qmax]
// with qmax = 2^(B−1) − 1, so 2^B − 1 ticks and scale s = 1 / qmax (absmax = 1).
const BITS: number[] = [2, 3, 4, 6, 8];
const DEFAULT_B = 3;
const DEFAULT_X = 0.42; // a deterministic seed value (not Math.random)

// The only literal hexes allowed: emerald = good/kept signal, red = the error.
const EMERALD = '#10b981';
const RED = '#ef4444';

function qmaxFor(b: number): number {
  return Math.pow(2, b - 1) - 1; // 2^(B-1) - 1  (int8 → 127)
}
function levelsFor(b: number): number {
  return 2 * qmaxFor(b) + 1; // 2^B - 1 grid points (one code dropped for exact 0)
}
function stepFor(b: number): number {
  return 1 / qmaxFor(b); // s = absmax / qmax, absmax = 1
}

// ── Per-locale copy. en is the original English. zh translates the prose but
// keeps glossary terms English (quantization, quantize, scale, step, symmetric,
// zero-point, bits, variance, weight) and uses full-width punctuation. Numbers
// and math symbols never translate. ──────────────────────────────────────────
const COPY = {
  en: {
    heading: 'Quantization round-trip — snap a real value to the nearest grid tick',
    xLabel: 'value x',
    xAria: 'Choose the continuous real value x on the range minus one to one',
    bitsLabel: 'bits B',
    bitsAria: (b: number) => `Choose the bit-width: ${b} bits per weight`,
    realTick: 'real x',
    snapTick: 'snapped x̂',
    errorTag: 'error',
    legendGrid: (n: number) => `${n} grid levels`,
    legendStep: 'step s',
    // readout rows
    rowStep: 'step  s = 1 / qmax,  qmax = 2^(B−1) − 1',
    rowQ: 'q = round(x / s)',
    rowXhat: 'x̂ = s · q',
    rowErr: 'error = |x − x̂|',
    rowBound: 'error ≤ s / 2',
    aside: 'The error is ~uniform on ±s/2, so its variance is s²/12.',
    caption: (
      <>
        Every quantized <strong>weight</strong> is the nearest grid point. The leftover —{' '}
        <span className="font-mono">|x − x̂|</span> — is the <em>quantization error</em>, and it can never exceed{' '}
        <span className="font-mono">s / 2</span> (half a step). So more <strong>bits</strong> means a finer grid and
        less error — but more storage per weight. Here the grid is <em>symmetric</em> with a zero-point of 0; the next
        diagram adds a zero-point for non-centered data.
      </>
    ),
    svgAria: (b: number, n: number, s: string, err: string) =>
      `Number line from minus one to one with ${n} evenly spaced grid ticks for ${b} bits (step ${s}). The continuous value x sits above the line and drops down to its nearest tick; the residual error is ${err}.`,
  },
  zh: {
    heading: 'quantization 往返——把实数吸附到最近的格点',
    xLabel: '数值 x',
    xAria: '在 −1 到 1 范围内选择连续实数 x',
    bitsLabel: 'bits B',
    bitsAria: (b: number) => `选择位宽：每个 weight 用 ${b} bits`,
    realTick: 'real x',
    snapTick: 'snapped x̂',
    errorTag: 'error',
    legendGrid: (n: number) => `${n} 个格点`,
    legendStep: 'step s',
    rowStep: 'step  s = 1 / qmax,  qmax = 2^(B−1) − 1',
    rowQ: 'q = round(x / s)',
    rowXhat: 'x̂ = s · q',
    rowErr: 'error = |x − x̂|',
    rowBound: 'error ≤ s / 2',
    aside: 'error 在 ±s/2 上近似均匀分布，所以它的 variance 是 s²/12。',
    caption: (
      <>
        每个量化后的 <strong>weight</strong> 都是离它最近的格点。剩下的那一点——
        <span className="font-mono">|x − x̂|</span>——就是 <em>quantization error</em>，它永远不会超过{' '}
        <span className="font-mono">s / 2</span>（半个 step）。所以 <strong>bits</strong> 越多，格点越密，error 越小——
        但每个 weight 的存储也越多。这里的格点是 <em>symmetric</em> 的，zero-point 为 0；下一张图会为非居中数据加上
        zero-point。
      </>
    ),
    svgAria: (b: number, n: number, s: string, err: string) =>
      `从 −1 到 1 的数轴，按 ${b} bits 等距画出 ${n} 个格点（step 为 ${s}）。连续值 x 位于数轴上方并下落到最近的格点；残差 error 为 ${err}。`,
  },
} as const;

// ── SVG geometry. Sized for the wider EN labels. A single number line with a
// drop from the real value to its snapped tick; an error bracket sits on the
// line. Wide labels (legend, tick captions) sit ABOVE the line, never wedged
// beside it, so nothing collides across locales. ─────────────────────────────
const VB_W = 560;
const VB_H = 200;
const PAD_L = 18;
const PAD_R = 18;
const LINE_X0 = PAD_L; // left end of the number line
const LINE_X1 = VB_W - PAD_R; // right end
const LINE_W = LINE_X1 - LINE_X0; // 524
const LINE_Y = 132; // y of the number line itself
const DOT_Y = 64; // y of the floating continuous value x
const LEGEND_Y = 18; // grid-level / step legend (top row, above everything)
const AXIS_LABEL_Y = LINE_Y + 30; // endpoint numeric labels (-1, 0, 1), below the line
const GRID_TICK_H = 9; // half-height of a grid tick mark
const MAX_TICKS_DRAWN = 257; // guard; B=8 → 2^8 − 1 = 255 ticks, all drawn

// Map a real value in [X_MIN, X_MAX] to an x pixel on the line.
function xToPx(v: number): number {
  return LINE_X0 + ((v - X_MIN) / RANGE) * LINE_W;
}

export function QuantizeRoundTrip() {
  const copy = COPY[useLocale()];
  const [b, setB] = React.useState<number>(DEFAULT_B);
  const [x, setX] = React.useState<number>(DEFAULT_X);

  const qmax = qmaxFor(b); // 2^(B-1) - 1
  const levels = levelsFor(b); // 2^B - 1
  const s = stepFor(b); // step = 1 / qmax
  const q = Math.max(-qmax, Math.min(qmax, Math.round(x / s))); // clamp to [−qmax, qmax]
  const xHat = s * q; // dequantized snap — always lands on a drawn tick
  const err = Math.abs(x - xHat); // residual
  const bound = s / 2; // theoretical max error

  const xPx = xToPx(x);
  const xHatPx = xToPx(xHat);

  // Number format helpers (3 decimals is plenty for [-1,1] grids).
  const f3 = (v: number) => v.toFixed(3);

  // Grid tick x positions: the codes q run from −qmax .. +qmax in steps of s.
  // Since the range is exactly [-1,1] and s = 1/qmax, the ticks are simply the
  // (levels = 2·qmax+1) evenly spaced points from -1 to 1, with 0 dead-center.
  const tickXs: number[] = [];
  for (let i = 0; i < levels && i < MAX_TICKS_DRAWN; i++) {
    tickXs.push(LINE_X0 + (i / (levels - 1)) * LINE_W);
  }

  // Error bracket lives on the number line between x's projection and the snap.
  const errLo = Math.min(xPx, xHatPx);
  const errHi = Math.max(xPx, xHatPx);
  const hasVisibleErr = errHi - errLo > 1.5; // hide the bracket when basically zero

  // Edge-aware placement for the two floating captions ("real x" above the dot,
  // "snapped x̂" below the line): a centered (textAnchor=middle) label clips the
  // viewBox when x is near ±1, so near an edge we flip the anchor and pin the
  // caption to the inner margin. CAP_EDGE is the px from an end within which we
  // flip. The top caption also stays clear of the top-right error readout.
  const CAP_EDGE = 70;
  function captionAnchor(px: number): { x: number; anchor: 'start' | 'middle' | 'end' } {
    if (px < LINE_X0 + CAP_EDGE) return { x: LINE_X0, anchor: 'start' };
    if (px > LINE_X1 - CAP_EDGE) return { x: LINE_X1 - 120, anchor: 'end' }; // keep clear of the error readout
    return { x: px, anchor: 'middle' };
  }
  const realCap = captionAnchor(xPx);
  const snapCap = captionAnchor(xHatPx);

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title */}
      <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>

      {/* Controls: x slider + bit-width buttons */}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 text-[11px] text-muted-foreground">
        <label className="flex items-center gap-2">
          <span className="font-mono">{copy.xLabel}</span>
          <input
            type="range"
            min={X_MIN}
            max={X_MAX}
            step={0.001}
            value={x}
            onChange={(e) => setX(Number(e.target.value))}
            aria-label={copy.xAria}
            className="h-1 w-40 cursor-pointer accent-[color:var(--primary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          />
          <span className="font-mono text-foreground">{f3(x)}</span>
        </label>
        <div className="flex items-center gap-2">
          <span className="font-mono">{copy.bitsLabel}</span>
          <div className="flex overflow-hidden rounded border border-border">
            {BITS.map((bb, i) => (
              <button
                key={bb}
                type="button"
                onClick={() => setB(bb)}
                aria-pressed={b === bb}
                aria-label={copy.bitsAria(bb)}
                className={[
                  'px-2 py-1 font-mono text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                  i > 0 ? 'border-l border-border' : '',
                ].join(' ')}
                style={{
                  background: b === bb ? 'var(--primary)' : 'transparent',
                  color: b === bb ? 'var(--background)' : 'var(--muted-foreground)',
                }}
              >
                {bb}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* The number line + snap diagram */}
      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        className="w-full"
        role="img"
        aria-label={copy.svgAria(b, levels, f3(s), f3(err))}
      >
        {/* Legend (top, above the line): grid level count + step value */}
        <text x={LINE_X0} y={LEGEND_Y} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px] font-medium">
          {copy.legendGrid(levels)}
        </text>
        <text
          x={LINE_X1}
          y={LEGEND_Y}
          textAnchor="end"
          style={{ fill: 'var(--foreground)' }}
          className="font-mono text-[11px] font-semibold"
        >
          {copy.legendStep} = {f3(s)}
        </text>

        {/* The number line */}
        <line
          x1={LINE_X0}
          y1={LINE_Y}
          x2={LINE_X1}
          y2={LINE_Y}
          style={{ stroke: 'var(--foreground)', opacity: 0.5 }}
          strokeWidth={1.5}
        />

        {/* Grid ticks (the discrete quantization grid) */}
        {tickXs.map((tx, i) => (
          <line
            key={i}
            x1={tx}
            y1={LINE_Y - GRID_TICK_H}
            x2={tx}
            y2={LINE_Y + GRID_TICK_H}
            style={{ stroke: 'var(--muted-foreground)', opacity: 0.55 }}
            strokeWidth={1}
          />
        ))}

        {/* Endpoint axis labels: -1, 0, 1 (below the line, evenly clear) */}
        <text
          x={LINE_X0}
          y={AXIS_LABEL_Y}
          textAnchor="start"
          style={{ fill: 'var(--muted-foreground)' }}
          className="font-mono text-[10px]"
        >
          −1
        </text>
        <text
          x={xToPx(0)}
          y={AXIS_LABEL_Y}
          textAnchor="middle"
          style={{ fill: 'var(--muted-foreground)' }}
          className="font-mono text-[10px]"
        >
          0
        </text>
        <text
          x={LINE_X1}
          y={AXIS_LABEL_Y}
          textAnchor="end"
          style={{ fill: 'var(--muted-foreground)' }}
          className="font-mono text-[10px]"
        >
          +1
        </text>

        {/* The snapped tick highlighted on the line (emerald = the kept value) */}
        <line
          x1={xHatPx}
          y1={LINE_Y - GRID_TICK_H - 2}
          x2={xHatPx}
          y2={LINE_Y + GRID_TICK_H + 2}
          style={{ stroke: EMERALD }}
          strokeWidth={2.5}
        />
        <circle cx={xHatPx} cy={LINE_Y} r={3.5} style={{ fill: EMERALD }} />

        {/* The continuous value x: a dot floating above the line */}
        <circle cx={xPx} cy={DOT_Y} r={4.5} style={{ fill: 'var(--primary)' }} />
        {/* drop line from x down to its snapped tick on the line */}
        <line
          x1={xPx}
          y1={DOT_Y + 5}
          x2={xHatPx}
          y2={LINE_Y - GRID_TICK_H - 2}
          style={{ stroke: 'var(--primary)', opacity: 0.7 }}
          strokeWidth={1.5}
          strokeDasharray="3 3"
        />

        {/* 'real x' caption above the floating dot (clears the dot, above the line) */}
        <text
          x={realCap.x}
          y={DOT_Y - 10}
          textAnchor={realCap.anchor}
          style={{ fill: 'var(--primary)' }}
          className="text-[10px] font-semibold"
        >
          {copy.realTick} = {f3(x)}
        </text>

        {/* 'snapped x̂' caption below the line, near the snapped tick */}
        <text
          x={snapCap.x}
          y={LINE_Y + 50}
          textAnchor={snapCap.anchor}
          style={{ fill: EMERALD }}
          className="text-[10px] font-semibold"
        >
          {copy.snapTick} = {f3(xHat)}
        </text>

        {/* Error bracket: a red segment on the line between x and x̂ + tag above it.
            The tag sits BELOW the line (well clear of the grid ticks above). */}
        {hasVisibleErr ? (
          <>
            <line
              x1={errLo}
              y1={LINE_Y + GRID_TICK_H + 6}
              x2={errHi}
              y2={LINE_Y + GRID_TICK_H + 6}
              style={{ stroke: RED }}
              strokeWidth={2.5}
            />
            <line
              x1={errLo}
              y1={LINE_Y + GRID_TICK_H + 3}
              x2={errLo}
              y2={LINE_Y + GRID_TICK_H + 9}
              style={{ stroke: RED }}
              strokeWidth={1.5}
            />
            <line
              x1={errHi}
              y1={LINE_Y + GRID_TICK_H + 3}
              x2={errHi}
              y2={LINE_Y + GRID_TICK_H + 9}
              style={{ stroke: RED }}
              strokeWidth={1.5}
            />
          </>
        ) : null}

        {/* error readout (top-right corner, clear of the dot and line) */}
        <text
          x={LINE_X1}
          y={DOT_Y - 26}
          textAnchor="end"
          style={{ fill: RED }}
          className="font-mono text-[11px] font-semibold"
        >
          {copy.errorTag} = {f3(err)}
        </text>
        <text
          x={LINE_X1}
          y={DOT_Y - 12}
          textAnchor="end"
          style={{ fill: 'var(--muted-foreground)' }}
          className="font-mono text-[10px]"
        >
          ≤ s/2 = {f3(bound)}
        </text>
      </svg>

      {/* Live formula readout: step, q, x̂, error, bound */}
      <div className="grid grid-cols-1 gap-x-6 gap-y-1 font-mono text-[12px] text-foreground/90 sm:grid-cols-2">
        <div>
          <span className="text-muted-foreground">{copy.rowStep}</span> = {f3(s)}
        </div>
        <div>
          {copy.rowQ} = {q}
        </div>
        <div>
          {copy.rowXhat} = {f3(xHat)}
        </div>
        <div>
          <span style={{ color: RED }}>
            {copy.rowErr} = {f3(err)}
          </span>{' '}
          <span className="text-muted-foreground">
            ({copy.rowBound} = {f3(bound)})
          </span>
        </div>
      </div>

      {/* optional aside: error distribution */}
      <p className="text-[11px] text-muted-foreground">{copy.aside}</p>

      {/* caption */}
      <p className="border-t border-border/60 pt-3 text-[12px] text-muted-foreground">{copy.caption}</p>
    </div>
  );
}

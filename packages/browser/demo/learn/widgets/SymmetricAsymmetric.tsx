import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * SymmetricAsymmetric — what a "zero-point" actually IS. Two stacked number
 * lines map a real-valued range onto an integer grid:
 *
 *   (a) SYMMETRIC (weights): weights are ≈ zero-centered, so the real range is
 *       symmetric [-m, +m] and lands on a symmetric int8 grid [-127, 127]
 *       (one code dropped vs the full -128). Real 0 ↔ code 0, so the
 *       zero-point is 0 — no shift, and no extra cross-term in the matmul.
 *   (b) ASYMMETRIC / affine (activations): activations are skewed/one-sided
 *       (e.g. post-ReLU/GELU), so the real range is lopsided. The grid is
 *       SHIFTED by an integer zero-point z so that real 0 still lands exactly
 *       on a code — which matters for padding and ReLU.
 *
 * Self-contained and STATIC: NO worker, NO model, NO WASM, NO network, NO
 * animation. Pure React + Tailwind/SVG, deterministic and identical server vs
 * client (the page is prerendered then hydrated). The only state is which
 * panel is highlighted — a local UI toggle, so no reduced-motion handling.
 *
 * EVERY NUMBER IS SOURCE-VERIFIED / EXACT — derived from the int8 grids, not
 * invented (quality is not shown at all here, so there is nothing qualitative
 * to hedge):
 *   SYMMETRIC int8 grid = [-127, 127]  (one code dropped vs -128, the standard
 *     symmetric convention). With example real max |m| = 1.0:
 *       s = m / 127 = 1 / 127 ≈ 0.00787 ,  z = 0 ,  real 0 ↔ code 0.
 *   ASYMMETRIC unsigned grid = [0, 255]. The general affine formulas are
 *       s = (r_max − r_min) / (q_max − q_min) ,
 *       z = round(q_min − r_min / s) .
 *     With an example skewed activation range [r_min, r_max] = [-1, 4]:
 *       s = (4 − (−1)) / (255 − 0) = 5 / 255 = 1/51 ≈ 0.01961 ,
 *       z = round(0 − (−1)/s) = round(1/s) = round(51) = 51 ,
 *     so real 0 ↔ code 51 EXACTLY (dequant s·(z − z) = 0), r_min ↔ 0 and
 *     r_max ↔ 255 both exact. z = 51 is a clean, clearly-nonzero zero-point.
 *
 * Formulas shown:
 *   symmetric   q = round(x / s),         x̂ = s · q,            z = 0
 *   asymmetric  q = round(x / s) + z,      x̂ = s · (q − z),
 *               z = round(q_min − r_min / s)
 *
 * LESSON: the zero-point is literally where the integer grid is anchored.
 * Zero-centered weights don't need one (symmetric — and it avoids an extra
 * cross-term in the matmul), so weight-only quantization is the easy first
 * step. Skewed activations do need one (asymmetric) so real 0 stays exact —
 * which is why activations (W8A8) are the hard part.
 */

type Mode = 'sym' | 'asym';

// The only literal hexes allowed: emerald (good/keep) and red (cost/shift).
const EMERALD = '#10b981'; // real 0 stays exact (the thing we protect)
const RED = '#ef4444'; // the zero-point shift / the cost it pays for

// ── Exact ground truth (do NOT edit a number without re-deriving from a grid). ─
// Symmetric weights: example |m| = 1.0 on grid [-127, 127].
const SYM = {
  rMin: -1,
  rMax: 1,
  qMin: -127,
  qMax: 127,
  s: 1 / 127, // ≈ 0.00787
  z: 0,
  zeroCode: 0, // real 0 ↔ code 0
} as const;

// Asymmetric activations: example skewed range [-1, 4] on grid [0, 255].
const ASYM = {
  rMin: -1,
  rMax: 4,
  qMin: 0,
  qMax: 255,
  s: 5 / 255, // = 1/51 ≈ 0.01961
  z: 51, // = round(q_min − r_min/s) = round(1/s) = 51
  zeroCode: 51, // real 0 ↔ code 51 (exact)
} as const;

// ── Per-locale copy. en is the original English. zh translates the prose but
// keeps glossary terms English (symmetric, asymmetric, zero-point, scale,
// quantize, dequantize, weight, activation, int8, per-tensor, ReLU, W8A8, …)
// and uses full-width punctuation. Numbers/units never translate. ────────────
const COPY = {
  en: {
    heading: 'Symmetric vs asymmetric — what the zero-point really is',
    symTab: 'symmetric · weights',
    asymTab: 'asymmetric · activations',
    pickAria: (m: string) => `Show the ${m} number line`,
    symHeading: 'symmetric · zero-centered weights · int8 grid [−127, 127]',
    asymHeading: 'asymmetric (affine) · skewed activations · uint8 grid [0, 255]',
    realLabel: 'real value',
    codeLabel: 'integer code',
    zeroReal: 'real 0',
    zeroCodeSym: 'code 0',
    zeroCodeAsym: (z: number) => `code ${z}  (zero-point z)`,
    zpZeroBadge: 'zero-point z = 0 — no shift',
    zpShiftBadge: (z: number) => `zero-point z = ${z} — grid shifted so real 0 is exact`,
    distSym: 'weights ≈ zero-centered',
    distAsym: 'activations skewed ≥ ~0 (post-ReLU/GELU)',
    formulaSymTitle: 'symmetric (z = 0)',
    formulaSym1: 'q = round(x / s)',
    formulaSym2: 'x̂ = s · q',
    formulaSymS: 's = m / 127 = 1 / 127 ≈ 0.00787',
    formulaAsymTitle: 'asymmetric (affine)',
    formulaAsym1: 'q = round(x / s) + z',
    formulaAsym2: 'x̂ = s · (q − z)',
    formulaAsymS: 's = (r_max − r_min) / (q_max − q_min) = 5 / 255 ≈ 0.01961',
    formulaAsymZ: 'z = round(q_min − r_min / s) = round(1 / s) = 51',
    legendZero: 'real 0 (kept exact)',
    legendZp: 'zero-point (grid anchor)',
    caption: (
      <>
        The <strong>zero-point</strong> is literally where the integer grid is anchored. Zero-centered{' '}
        <strong>weights</strong> map real 0 onto code 0 with no shift (<em>symmetric</em>, <span className="font-mono">z = 0</span>) —
        which also avoids an extra cross-term in the matmul, so weight-only quantization is the easy first step. Skewed{' '}
        <strong>activations</strong> (e.g. post-<span className="font-mono">ReLU</span>/<span className="font-mono">GELU</span>, mostly
        ≥ 0) need a nonzero zero-point (<em>asymmetric</em> / affine) so real 0 still lands exactly on a code — important for padding
        and <span className="font-mono">ReLU</span>. That asymmetry is why quantizing both weights <em>and</em> activations (
        <span className="font-mono">W8A8</span>) is the hard part.
      </>
    ),
    diagramAria: (m: string) =>
      m === 'sym'
        ? 'Symmetric int8 number line: a zero-centered real range maps onto the grid from minus 127 to 127, with real zero landing exactly on integer code 0, so the zero-point is 0.'
        : 'Asymmetric affine number line: a skewed activation range maps onto the unsigned grid from 0 to 255, with the grid shifted by zero-point 51 so real zero lands exactly on integer code 51.',
  },
  zh: {
    heading: 'symmetric 与 asymmetric——zero-point 到底是什么',
    symTab: 'symmetric · weights',
    asymTab: 'asymmetric · activations',
    pickAria: (m: string) => `显示 ${m} 数轴`,
    symHeading: 'symmetric · 零中心的 weights · int8 格点 [−127, 127]',
    asymHeading: 'asymmetric（affine）· 偏斜的 activations · uint8 格点 [0, 255]',
    realLabel: '真实值',
    codeLabel: '整数 code',
    zeroReal: 'real 0',
    zeroCodeSym: 'code 0',
    zeroCodeAsym: (z: number) => `code ${z}（zero-point z）`,
    zpZeroBadge: 'zero-point z = 0——无偏移',
    zpShiftBadge: (z: number) => `zero-point z = ${z}——格点偏移，使 real 0 精确`,
    distSym: 'weights ≈ 零中心',
    distAsym: 'activations 偏向 ≥ ~0（ReLU/GELU 之后）',
    formulaSymTitle: 'symmetric（z = 0）',
    formulaSym1: 'q = round(x / s)',
    formulaSym2: 'x̂ = s · q',
    formulaSymS: 's = m / 127 = 1 / 127 ≈ 0.00787',
    formulaAsymTitle: 'asymmetric（affine）',
    formulaAsym1: 'q = round(x / s) + z',
    formulaAsym2: 'x̂ = s · (q − z)',
    formulaAsymS: 's = (r_max − r_min) / (q_max − q_min) = 5 / 255 ≈ 0.01961',
    formulaAsymZ: 'z = round(q_min − r_min / s) = round(1 / s) = 51',
    legendZero: 'real 0（保持精确）',
    legendZp: 'zero-point（格点锚点）',
    caption: (
      <>
        <strong>zero-point</strong> 就是整数格点被锚定的位置。零中心的 <strong>weights</strong> 把 real 0 映到 code 0，不需要偏移（
        <em>symmetric</em>，<span className="font-mono">z = 0</span>）——这同时省掉了 matmul 里多出来的一个交叉项，所以 weight-only
        quantization 是最容易的第一步。偏斜的 <strong>activations</strong>（例如 <span className="font-mono">ReLU</span>/
        <span className="font-mono">GELU</span> 之后，多数 ≥ 0）需要一个非零的 zero-point（<em>asymmetric</em> / affine），让 real 0 仍然
        精确落在某个 code 上——这对 padding 和 <span className="font-mono">ReLU</span> 很重要。正是这种非对称，使得同时量化 weights{' '}
        <em>和</em> activations（<span className="font-mono">W8A8</span>）成为困难的部分。
      </>
    ),
    diagramAria: (m: string) =>
      m === 'sym'
        ? 'symmetric int8 数轴：零中心的真实范围映射到 −127 到 127 的格点上，real 0 精确落在整数 code 0 上，所以 zero-point 为 0。'
        : 'asymmetric affine 数轴：偏斜的 activation 范围映射到 0 到 255 的无符号格点上，格点被 zero-point 51 偏移，使 real 0 精确落在整数 code 51 上。',
  },
} as const;

// ── SVG geometry. EN labels are wider than zh, so size every box for EN. The
// single number line is full-width; the wide section heading sits ABOVE it, the
// "real value" axis sits above the bar and the "integer code" axis below, so no
// two <text> elements ever overlap and nothing is wedged beside the bar. ──────
const VB_W = 580;
const VB_H = 196;
const PAD_L = 16;
const PAD_R = 16;
const AXIS_X = PAD_L;
const AXIS_W = VB_W - PAD_L - PAD_R; // 548

const HEAD_Y = 16; // section heading
const REAL_LABEL_Y = 40; // "real value" + endpoint reals, above the bar
const BAR_Y = 52; // the number-line bar
const BAR_H = 16;
const CODE_LABEL_Y = 92; // "integer code" + endpoint codes, below the bar
const DIST_Y = 118; // faint distribution curve baseline
const DIST_H = 30;
const BADGE_Y = 170; // zero-point badge text + legend

/** Fraction [0..1] of where a real value sits within a panel's real range. */
function realFrac(x: number, rMin: number, rMax: number): number {
  return (x - rMin) / (rMax - rMin);
}

export function SymmetricAsymmetric() {
  const copy = COPY[useLocale()];
  const [mode, setMode] = React.useState<Mode>('sym');
  const cfg = mode === 'sym' ? SYM : ASYM;

  // x-pixel for a fraction of the axis.
  const px = (frac: number) => AXIS_X + frac * AXIS_W;
  // The real-0 marker position (the value we protect).
  const zeroFrac = realFrac(0, cfg.rMin, cfg.rMax);
  const zeroX = px(zeroFrac);

  // Distribution bump: a faint symmetric/one-sided hump under the line. We draw
  // it as a static smooth path sampled from a fixed shape — no randomness, so
  // the render is deterministic. Symmetric → centered hump; asymmetric → a
  // right-leaning hump biased toward the ≥0 side.
  const distPath = (() => {
    const n = 48;
    const pts: string[] = [];
    for (let i = 0; i <= n; i++) {
      const t = i / n; // 0..1 across the axis
      // height in [0..1]; symmetric is a Gaussian centered at the real-0 frac,
      // asymmetric is a skewed hump whose peak sits to the right of real 0.
      const peak = mode === 'sym' ? zeroFrac : zeroFrac + 0.34;
      const width = mode === 'sym' ? 0.18 : 0.22;
      const g = Math.exp(-((t - peak) ** 2) / (2 * width * width));
      const x = px(t);
      const y = DIST_Y - g * DIST_H;
      pts.push(`${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`);
    }
    return pts.join(' ');
  })();

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title + symmetric/asymmetric toggle */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>
        <div className="flex overflow-hidden rounded border border-border text-[11px]">
          {(['sym', 'asym'] as Mode[]).map((m, i) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              aria-pressed={mode === m}
              aria-label={copy.pickAria(m === 'sym' ? 'symmetric' : 'asymmetric')}
              className={[
                'px-2.5 py-1 font-mono font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                i > 0 ? 'border-l border-border' : '',
              ].join(' ')}
              style={{
                background: mode === m ? 'var(--primary)' : 'transparent',
                color: mode === m ? 'var(--background)' : 'var(--muted-foreground)',
              }}
            >
              {m === 'sym' ? copy.symTab : copy.asymTab}
            </button>
          ))}
        </div>
      </div>

      {/* The number-line diagram */}
      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.diagramAria(mode)}>
        {/* section heading ABOVE everything */}
        <text x={AXIS_X} y={HEAD_Y} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px] font-medium">
          {mode === 'sym' ? copy.symHeading : copy.asymHeading}
        </text>

        {/* "real value" axis label + endpoint reals, ABOVE the bar */}
        <text x={AXIS_X} y={REAL_LABEL_Y} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
          {copy.realLabel}
        </text>
        <text x={AXIS_X + 86} y={REAL_LABEL_Y} style={{ fill: 'var(--foreground)' }} className="font-mono text-[10px]">
          {cfg.rMin}
        </text>
        <text
          x={AXIS_X + AXIS_W}
          y={REAL_LABEL_Y}
          textAnchor="end"
          style={{ fill: 'var(--foreground)' }}
          className="font-mono text-[10px]"
        >
          {cfg.rMax}
        </text>

        {/* the number-line bar */}
        <rect
          x={AXIS_X}
          y={BAR_Y}
          width={AXIS_W}
          height={BAR_H}
          rx={4}
          style={{ fill: 'var(--muted)', stroke: 'var(--border)' }}
          strokeWidth={1}
        />

        {/* real-0 marker (kept exact) — a green tick through the bar */}
        <line
          x1={zeroX}
          y1={BAR_Y - 8}
          x2={zeroX}
          y2={BAR_Y + BAR_H + 8}
          style={{ stroke: EMERALD }}
          strokeWidth={2}
        />
        <circle cx={zeroX} cy={BAR_Y + BAR_H / 2} r={3.5} style={{ fill: EMERALD }} />
        <text
          x={zeroX}
          y={BAR_Y - 12}
          textAnchor="middle"
          style={{ fill: EMERALD }}
          className="font-mono text-[10px] font-semibold"
        >
          {copy.zeroReal}
        </text>

        {/* "integer code" axis label + endpoint codes, BELOW the bar */}
        <text x={AXIS_X} y={CODE_LABEL_Y} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
          {copy.codeLabel}
        </text>
        <text x={AXIS_X + 86} y={CODE_LABEL_Y} style={{ fill: 'var(--foreground)' }} className="font-mono text-[10px]">
          {cfg.qMin}
        </text>
        <text
          x={AXIS_X + AXIS_W}
          y={CODE_LABEL_Y}
          textAnchor="end"
          style={{ fill: 'var(--foreground)' }}
          className="font-mono text-[10px]"
        >
          {cfg.qMax}
        </text>

        {/* the code under real 0 — red callout below the green tick (asymmetric:
            the shifted zero-point; symmetric: just code 0). Centered on zeroX
            but clamped away from the edges so it never clips. */}
        <text
          x={Math.min(Math.max(zeroX, AXIS_X + 64), AXIS_X + AXIS_W - 64)}
          y={CODE_LABEL_Y + 16}
          textAnchor="middle"
          style={{ fill: mode === 'sym' ? 'var(--foreground)' : RED }}
          className="font-mono text-[10px] font-semibold"
        >
          {mode === 'sym' ? copy.zeroCodeSym : copy.zeroCodeAsym(cfg.zeroCode)}
        </text>

        {/* faint distribution hump (centered for weights, skewed for activations) */}
        <path d={distPath} fill="none" style={{ stroke: 'var(--primary)', opacity: 0.45 }} strokeWidth={1.5} />
        <text
          x={AXIS_X + AXIS_W}
          y={DIST_Y + 12}
          textAnchor="end"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[10px]"
        >
          {mode === 'sym' ? copy.distSym : copy.distAsym}
        </text>

        {/* zero-point badge: no-shift (z=0) vs shifted grid (z≠0) */}
        <text
          x={AXIS_X}
          y={BADGE_Y}
          style={{ fill: mode === 'sym' ? EMERALD : RED }}
          className="text-[11px] font-semibold"
        >
          {mode === 'sym' ? copy.zpZeroBadge : copy.zpShiftBadge(cfg.z)}
        </text>
      </svg>

      {/* legend: real-0 dot + zero-point swatch */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: EMERALD }} aria-hidden />
          {copy.legendZero}
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: RED }} aria-hidden />
          {copy.legendZp}
        </span>
      </div>

      {/* the two formula cards, side by side — symmetric vs asymmetric */}
      <div className="grid gap-2 sm:grid-cols-2">
        <div className="rounded border border-border bg-card p-2.5">
          <div className="mb-1 text-[11px] font-semibold text-foreground">{copy.formulaSymTitle}</div>
          <div className="space-y-0.5 font-mono text-[11px] text-foreground/90">
            <div>{copy.formulaSym1}</div>
            <div>{copy.formulaSym2}</div>
          </div>
          <div className="mt-1 font-mono text-[10px] text-muted-foreground">{copy.formulaSymS}</div>
        </div>
        <div className="rounded border border-border bg-card p-2.5">
          <div className="mb-1 text-[11px] font-semibold text-foreground">{copy.formulaAsymTitle}</div>
          <div className="space-y-0.5 font-mono text-[11px] text-foreground/90">
            <div>{copy.formulaAsym1}</div>
            <div>{copy.formulaAsym2}</div>
          </div>
          <div className="mt-1 space-y-0.5 font-mono text-[10px] text-muted-foreground">
            <div>{copy.formulaAsymS}</div>
            <div>{copy.formulaAsymZ}</div>
          </div>
        </div>
      </div>

      {/* lesson caption */}
      <p className="border-t border-border/60 pt-3 text-[12px] text-muted-foreground">{copy.caption}</p>
    </div>
  );
}

import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Fp8Fork — the same EIGHT bits, split two opposite ways (the fp8 sibling of
 * SixteenBitFork). Both E4M3 and E5M2 are 8-bit floats, but they make the
 * opposite trade with the bits left after the sign: E4M3 keeps an extra MANTISSA
 * bit (precision), E5M2 keeps an extra EXPONENT bit (range). It is the same
 * "exponent buys range, mantissa buys precision" rule one tier below the 16-bit
 * fork — and fp8 training actually uses BOTH halves: E4M3 on the forward pass
 * (weights / activations), E5M2 on the backward pass (the wide-range gradients).
 *
 * Self-contained and STATIC: NO worker, NO model, NO WASM, NO network, NO
 * animation — pure React + Tailwind/SVG, so it server-renders for crawlers
 * (createRoot replaces it on boot). The first render is deterministic and
 * identical server vs client; the only state is which strip is highlighted, so
 * it needs no prefers-reduced-motion handling.
 *
 * EXACT NUMBERS (source-verified against the OCP OFP8 spec / the FP8 paper
 * arXiv:2209.05433 — none invented; every value is derived from these exact
 * field widths):
 *   - E4M3: 1 sign / 4 exponent / 3 mantissa, exponent bias 7. OCP convention:
 *     NO infinities, a single NaN (S.1111.111), so the largest finite is
 *     S.1111.110 = 1.75 × 2^8 = 448 (a naive IEEE 1.4.3 minifloat would stop at
 *     240). Smallest normal 2^-6 ≈ 0.0156. 3 mantissa bits → finer precision.
 *   - E5M2: 1 sign / 5 exponent / 2 mantissa, exponent bias 15. IEEE-like: keeps
 *     infinities + NaN, so the largest finite is 1.75 × 2^15 = 57344. Smallest
 *     normal 2^-14 ≈ 6.1e-5. 5 exponent bits → wider range, coarser steps.
 *
 * THE LESSON: at 8 bits you cannot have both, so fp8 ships two. E4M3 (precision)
 * carries the forward pass; E5M2 (range) carries gradients. fp8 lands on Hopper
 * (H100) and Blackwell Tensor Cores; this browser demo still ships bf16, not fp8.
 */

type Fmt = 'e4m3' | 'e5m2';
const FORMATS: Fmt[] = ['e4m3', 'e5m2'];
const DEFAULT_FMT: Fmt = 'e4m3'; // the forward-pass (weights/activations) variant

const TOTAL_BITS = 8;

// Bit-layout segment kinds, drawn as a colored strip. Both formats are floats.
type SegKind = 'sign' | 'exp' | 'mant';
type Seg = { kind: SegKind; bits: number };

// ── Per-format ground truth (verified — do not edit numbers without a source). ─
type FormatInfo = {
  /** sign / exponent / mantissa bit layout */
  segs: Seg[];
  /** exponent bias */
  bias: number;
  /** which field this format "spends" its budget on */
  spends: 'range' | 'precision';
};

const INFO: Record<Fmt, FormatInfo> = {
  e4m3: {
    segs: [
      { kind: 'sign', bits: 1 },
      { kind: 'exp', bits: 4 },
      { kind: 'mant', bits: 3 },
    ],
    bias: 7,
    spends: 'precision',
  },
  e5m2: {
    segs: [
      { kind: 'sign', bits: 1 },
      { kind: 'exp', bits: 5 },
      { kind: 'mant', bits: 2 },
    ],
    bias: 15,
    spends: 'range',
  },
};

// Segment fills: sign = foreground, exponent = primary, mantissa = muted.
const SEG_FILL: Record<SegKind, string> = {
  sign: 'var(--foreground)',
  exp: 'var(--primary)',
  mant: 'var(--muted-foreground)',
};

// ── Per-locale copy. COPY.en strings are the original English. zh keeps the
// glossary terms English (fp8, E4M3, E5M2, sign, exponent, mantissa, bias,
// dynamic range, precision, bits, forward, backward, gradients, NaN, Hopper,
// bf16) and uses full-width punctuation; only the surrounding prose translates.
// Numbers and units never translate. ─────────────────────────────────────────
const COPY = {
  en: {
    heading: 'The fp8 fork — eight bits, two opposite trades',
    pickAria: (f: string) => `Highlight the ${f} bit layout`,
    roleSplit: 'fp8 ships both · E4M3 → forward (weights/activations) · E5M2 → backward (gradients)',
    sign: 'sign',
    exp: 'exponent',
    mant: 'mantissa',
    // per-strip sub-labels (drawn above each strip, left side)
    spendsPrecision: 'spends its bits on PRECISION (3-bit mantissa)',
    spendsRange: 'spends its bits on RANGE (5-bit exponent)',
    biasLabel: (b: number) => `exponent bias ${b}`,
    // two-item readouts under each strip
    e4m3Readout: ['max 448 (OCP) · smallest normal 2^-6 ≈ 0.0156', '3-bit mantissa → finer · no inf · 1 NaN'],
    e5m2Readout: ['max 57,344 · smallest normal 2^-14 ≈ 6.1e-5', '2-bit mantissa → coarse · IEEE-like (inf + NaN)'],
    caption: (
      <>
        Same 8 bits, opposite trade — and fp8 uses <strong>both</strong>. <strong>E4M3</strong> keeps a 3-bit{' '}
        <em>mantissa</em> (finer precision) and reaches <strong>448</strong>, so it carries the <em>forward</em> pass:
        weights and activations. <strong>E5M2</strong> trades a mantissa bit for a 5th <em>exponent</em> bit, stretching
        the range to <strong>57,344</strong> (and keeping IEEE infinities/NaN) to hold the wide{' '}
        <strong>dynamic range</strong> of gradients in the <em>backward</em> pass. E4M3&rsquo;s 448 max is the OCP/ML
        convention — a naive IEEE 8-bit float stops at 240; E4M3 reclaims the infinity and most NaN bit-patterns to
        extend it. fp8 lands on Hopper (H100) and Blackwell Tensor Cores; this browser demo still ships{' '}
        <strong>bf16</strong>.
      </>
    ),
    svgAria:
      'Two stacked 8-bit fp8 layouts. E4M3: 1 sign, 4 exponent, 3 mantissa, exponent bias 7, no infinities and one NaN, maximum finite 448, for the forward pass. E5M2: 1 sign, 5 exponent, 2 mantissa, exponent bias 15, IEEE-like with infinities and NaN, maximum 57344, for gradients. The exponent buys range; the mantissa buys precision.',
  },
  zh: {
    heading: 'fp8 的岔路口——8 个 bit，两种相反的取舍',
    pickAria: (f: string) => `高亮 ${f} 的 bit 布局`,
    roleSplit: 'fp8 两种都用 · E4M3 → forward（weights/activations） · E5M2 → backward（gradients）',
    sign: 'sign',
    exp: 'exponent',
    mant: 'mantissa',
    spendsPrecision: '把 bit 花在 PRECISION 上（3-bit mantissa）',
    spendsRange: '把 bit 花在 RANGE 上（5-bit exponent）',
    biasLabel: (b: number) => `exponent bias ${b}`,
    e4m3Readout: ['max 448（OCP） · 最小正规数 2^-6 ≈ 0.0156', '3-bit mantissa → 更细 · 没有 inf · 1 个 NaN'],
    e5m2Readout: ['max 57,344 · 最小正规数 2^-14 ≈ 6.1e-5', '2-bit mantissa → 更粗 · IEEE-like（有 inf + NaN）'],
    caption: (
      <>
        同样是 8 bit，取舍却相反——而 fp8 <strong>两种都用</strong>。<strong>E4M3</strong> 保留 3-bit{' '}
        <em>mantissa</em>（精度更细），最大到 <strong>448</strong>，所以它扛起<em>前向</em>：weights 和 activations。{' '}
        <strong>E5M2</strong> 用一个 mantissa bit 换来第 5 个 <em>exponent</em> bit，把范围拉到 <strong>57,344</strong>
        （还保留 IEEE 的 inf/NaN），好在<em>反向</em>里装下 gradients 那种很宽的 <strong>dynamic range</strong>。E4M3
        的 448 上限是 OCP/ML 约定——朴素的 IEEE 8-bit float 只到 240；E4M3 回收了 inf 和大部分 NaN 的位模式来扩展它。fp8
        落在 Hopper（H100）和 Blackwell 的 Tensor Core 上；而这个浏览器 demo 仍然用 <strong>bf16</strong>。
      </>
    ),
    svgAria:
      '两条堆叠的 8-bit fp8 布局。E4M3：1 sign、4 exponent、3 mantissa，exponent bias 7，没有 infinities、一个 NaN，最大有限值 448，用于前向。E5M2：1 sign、5 exponent、2 mantissa，exponent bias 15，IEEE-like 带 infinities 和 NaN，最大值 57344，用于 gradients。exponent 买的是 range，mantissa 买的是 precision。',
  },
} as const;

// Locale-agnostic copy shape, so the SVG helpers accept either en or zh without
// narrowing to the English literal strings.
type Copy = (typeof COPY)[keyof typeof COPY];

// ── SVG geometry. EN labels are wider than zh, so every box is sized for EN. ──
// Two stacked rows; each row = a wide sub-label ABOVE its strip (never beside),
// the 8-cell strip itself, then a two-line readout below it. Mirrors
// SixteenBitFork exactly so the two forks read as a pair across the chapter.
const VB_W = 560;
const VB_H = 300;
const PAD_L = 14;
const PAD_R = 14;
const TRACK_X = PAD_L;
const TRACK_W = VB_W - PAD_L - PAD_R; // 532
const CELL_W = TRACK_W / TOTAL_BITS; // width of one bit cell (66.5 for fp8)

// Row 1 (E4M3) anchors.
const R1_TITLE_Y = 16; // "E4M3" + bias, left/right of the row title line
const R1_SUB_Y = 32; // "spends its bits on PRECISION …"
const R1_STRIP_Y = 40;
const STRIP_H = 30;
const R1_READ1_Y = 88; // first readout line (below strip)
const R1_READ2_Y = 104; // second readout line
// Row 2 (E5M2) anchors — same offsets, shifted down by ROW_GAP.
const ROW_GAP = 132;
const R2_TITLE_Y = R1_TITLE_Y + ROW_GAP; // 148
const R2_SUB_Y = R1_SUB_Y + ROW_GAP; // 164
const R2_STRIP_Y = R1_STRIP_Y + ROW_GAP; // 172
const R2_READ1_Y = R1_READ1_Y + ROW_GAP; // 220
const R2_READ2_Y = R1_READ2_Y + ROW_GAP; // 236
// Bottom hint line (the universal rule).
const RULE_Y = 280;

function bitStrip(segs: Seg[], stripY: number, dim: boolean, copy: Copy) {
  let cellCursor = 0; // index of next bit cell (0..7)
  return segs.map((seg) => {
    const x = TRACK_X + cellCursor * CELL_W;
    const w = seg.bits * CELL_W;
    cellCursor += seg.bits;
    const labelText = seg.kind === 'sign' ? copy.sign : seg.kind === 'exp' ? copy.exp : copy.mant;
    const showLabel = w > 44; // every fp8 field is ≥1 cell (66.5px) wide, so all get a word
    return (
      <g key={seg.kind}>
        <rect
          x={x}
          y={stripY}
          width={Math.max(2, w - 1.5)}
          height={STRIP_H}
          rx={3}
          style={{
            fill: SEG_FILL[seg.kind],
            opacity: dim ? (seg.kind === 'mant' ? 0.18 : 0.35) : seg.kind === 'mant' ? 0.35 : 0.85,
          }}
        />
        {showLabel ? (
          <text
            x={x + w / 2}
            y={stripY + STRIP_H / 2 + 4}
            textAnchor="middle"
            style={{ fill: 'var(--background)', opacity: dim ? 0.7 : 1 }}
            className="text-[10px] font-medium"
          >
            {labelText} {seg.bits}
          </text>
        ) : (
          <text
            x={x + w / 2}
            y={stripY + STRIP_H / 2 + 4}
            textAnchor="middle"
            style={{ fill: 'var(--background)', opacity: dim ? 0.7 : 1 }}
            className="text-[9px] font-semibold"
          >
            {seg.bits}
          </text>
        )}
      </g>
    );
  });
}

function readoutLines(lines: readonly [string, string], y1: number, y2: number, dim: boolean) {
  const op = dim ? 0.55 : 1;
  return (
    <>
      <text x={TRACK_X} y={y1} style={{ fill: 'var(--foreground)', opacity: op }} className="font-mono text-[10px]">
        {lines[0]}
      </text>
      <text x={TRACK_X} y={y2} style={{ fill: 'var(--muted-foreground)', opacity: op }} className="text-[10px]">
        {lines[1]}
      </text>
    </>
  );
}

export function Fp8Fork() {
  const copy = COPY[useLocale()];
  const [fmt, setFmt] = React.useState<Fmt>(DEFAULT_FMT);

  const e4 = INFO.e4m3;
  const e5 = INFO.e5m2;
  const e4Active = fmt === 'e4m3';
  const e5Active = fmt === 'e5m2';

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title + format toggle */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>
        <div className="flex overflow-hidden rounded border border-border">
          {FORMATS.map((f, i) => (
            <button
              key={f}
              type="button"
              onClick={() => setFmt(f)}
              aria-pressed={fmt === f}
              aria-label={copy.pickAria(f)}
              className={[
                'px-2.5 py-1 font-mono text-[11px] font-medium uppercase transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                i > 0 ? 'border-l border-border' : '',
              ].join(' ')}
              style={{
                background: fmt === f ? 'var(--primary)' : 'transparent',
                color: fmt === f ? 'var(--background)' : 'var(--muted-foreground)',
              }}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Context badge: fp8 ships BOTH — the forward/backward role split. */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1.5 rounded border border-foreground/40 bg-foreground/5 px-2 py-0.5 text-[11px] font-medium text-foreground/80">
          🔬 {copy.roleSplit}
        </span>
      </div>

      {/* The two stacked bit strips */}
      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.svgAria}>
        {/* ── Row 1: E4M3 ── */}
        <text
          x={TRACK_X}
          y={R1_TITLE_Y}
          style={{ fill: 'var(--foreground)', opacity: e4Active ? 1 : 0.6 }}
          className="font-mono text-[12px] font-semibold"
        >
          E4M3
        </text>
        <text
          x={TRACK_X + TRACK_W}
          y={R1_TITLE_Y}
          textAnchor="end"
          style={{ fill: 'var(--muted-foreground)', opacity: e4Active ? 1 : 0.6 }}
          className="text-[10px]"
        >
          {copy.biasLabel(e4.bias)}
        </text>
        <text
          x={TRACK_X}
          y={R1_SUB_Y}
          style={{ fill: 'var(--muted-foreground)', opacity: e4Active ? 1 : 0.55 }}
          className="text-[10px] font-medium"
        >
          {copy.spendsPrecision}
        </text>
        {bitStrip(e4.segs, R1_STRIP_Y, !e4Active, copy)}
        {readoutLines(copy.e4m3Readout, R1_READ1_Y, R1_READ2_Y, !e4Active)}

        {/* divider between the two forks */}
        <line
          x1={TRACK_X}
          y1={R1_READ2_Y + 14}
          x2={TRACK_X + TRACK_W}
          y2={R1_READ2_Y + 14}
          style={{ stroke: 'var(--border)' }}
          strokeWidth={1}
          strokeDasharray="3 3"
        />

        {/* ── Row 2: E5M2 ── */}
        <text
          x={TRACK_X}
          y={R2_TITLE_Y}
          style={{ fill: 'var(--foreground)', opacity: e5Active ? 1 : 0.6 }}
          className="font-mono text-[12px] font-semibold"
        >
          E5M2
        </text>
        <text
          x={TRACK_X + TRACK_W}
          y={R2_TITLE_Y}
          textAnchor="end"
          style={{ fill: 'var(--muted-foreground)', opacity: e5Active ? 1 : 0.6 }}
          className="text-[10px]"
        >
          {copy.biasLabel(e5.bias)}
        </text>
        <text
          x={TRACK_X}
          y={R2_SUB_Y}
          style={{ fill: 'var(--primary)', opacity: e5Active ? 1 : 0.55 }}
          className="text-[10px] font-medium"
        >
          {copy.spendsRange}
        </text>
        {bitStrip(e5.segs, R2_STRIP_Y, !e5Active, copy)}
        {readoutLines(copy.e5m2Readout, R2_READ1_Y, R2_READ2_Y, !e5Active)}

        {/* ── bottom universal rule ── */}
        <text
          x={TRACK_X + TRACK_W / 2}
          y={RULE_Y}
          textAnchor="middle"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[10px] font-medium"
        >
          {copy.exp} → range · {copy.mant} → precision
        </text>
      </svg>

      {/* Caption: the E4M3-forward / E5M2-backward lesson */}
      <p className="border-t border-border/60 pt-3 text-[12px] text-muted-foreground">{copy.caption}</p>
    </div>
  );
}

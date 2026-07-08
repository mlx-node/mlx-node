import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * SixteenBitFork — the same 16 bits, split two opposite ways. Both bf16 and fp16
 * are 16-bit floats, but they make the opposite trade with the bits left over
 * after the sign: bf16 spends them on the EXPONENT (range), fp16 spends them on
 * the MANTISSA (precision). This is the canonical "exponent buys range, mantissa
 * buys precision" contrast, and it anchors "you are here = bf16" (the demo ships
 * bf16, which keeps fp32's full 8-bit exponent → fp32's full dynamic range).
 *
 * Self-contained and STATIC: NO worker, NO model, NO WASM, NO network, NO
 * animation — pure React + Tailwind/SVG, so it server-renders for crawlers
 * (createRoot replaces it on boot). The first render is deterministic and
 * identical server vs client; the only state is which strip is highlighted, so
 * it needs no prefers-reduced-motion handling.
 *
 * EXACT NUMBERS (source-verified against the IEEE / Brain-float layouts — none
 * invented; every value below is derived only from these exact field widths):
 *   - bf16 (bfloat16): 1 sign / 8 exponent / 7 mantissa, exponent bias 127.
 *     Same 8-bit exponent as fp32 → same enormous dynamic range (~1e38);
 *     max finite ≈ 3.38953e38; ~2–3 decimal digits of precision.
 *   - fp16 (IEEE 754 binary16): 1 sign / 5 exponent / 10 mantissa, bias 15.
 *     max finite = 65504; smallest normal 2^-14 ≈ 6.1e-5; ~3.31 decimal digits.
 *   - Both are 16 bits total. We do NOT cite a specific loss-scale magnitude —
 *     it is workload-dependent — only that fp16 training "needs loss scaling".
 *
 * THE LESSON: bf16 keeps fp32's full 8-bit exponent, so it has the same huge
 * dynamic range as fp32 (just coarser steps between values). That is why it won
 * for training — gradients rarely overflow or underflow, so it typically needs
 * no loss-scaling (it can still overflow past ~3.4e38 or underflow below its
 * smallest subnormal — just far less often than fp16). fp16 keeps more mantissa
 * (finer precision) but a tiny range, so fp16 training needs loss scaling to
 * keep small gradients from underflowing to zero.
 */

type Fmt = 'bf16' | 'fp16';
const FORMATS: Fmt[] = ['bf16', 'fp16'];
const DEMO_FMT: Fmt = 'bf16'; // the in-browser demo ships this — "you are here"

const TOTAL_BITS = 16;

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
  bf16: {
    segs: [
      { kind: 'sign', bits: 1 },
      { kind: 'exp', bits: 8 },
      { kind: 'mant', bits: 7 },
    ],
    bias: 127,
    spends: 'range',
  },
  fp16: {
    segs: [
      { kind: 'sign', bits: 1 },
      { kind: 'exp', bits: 5 },
      { kind: 'mant', bits: 10 },
    ],
    bias: 15,
    spends: 'precision',
  },
};

// Segment fills: sign = foreground, exponent = primary, mantissa = muted.
const SEG_FILL: Record<SegKind, string> = {
  sign: 'var(--foreground)',
  exp: 'var(--primary)',
  mant: 'var(--muted-foreground)',
};

// ── Per-locale copy. COPY.en strings are the original English. zh keeps the
// glossary terms English (bf16, fp16, fp32, sign, exponent, mantissa, bias,
// dynamic range, precision, bits, loss scaling, overflow/underflow) and uses
// full-width punctuation; only the surrounding prose translates. Numbers and
// units never translate. ──────────────────────────────────────────────────────
const COPY = {
  en: {
    heading: 'The 16-bit fork — same bits, opposite trade',
    pickAria: (f: string) => `Highlight the ${f} bit layout`,
    youAreHere: 'you are here · the demo ships bf16',
    sign: 'sign',
    exp: 'exponent',
    mant: 'mantissa',
    // per-strip sub-labels (drawn above each strip, left side)
    spendsRange: 'spends its bits on RANGE (8-bit exponent)',
    spendsPrecision: 'spends its bits on PRECISION (10-bit mantissa)',
    biasLabel: (b: number) => `exponent bias ${b}`,
    // two-item readouts under each strip
    bf16Readout: ['range ≈ 1e38 (max ≈ 3.39e38)', '~2–3 decimal digits · matches FP32 range'],
    fp16Readout: ['max 65504 (smallest normal 2^-14 ≈ 6.1e-5)', '~3.31 decimal digits · needs loss scaling'],
    caption: (
      <>
        Same 16 bits, opposite trade. <strong>bf16</strong> keeps fp32&rsquo;s full <em>8-bit exponent</em>, so it has the
        same enormous <strong>dynamic range</strong> as fp32 (~1e38) — just coarser steps between values. That is why it
        won for training: gradients <strong>rarely overflow or underflow</strong>, so it typically needs no loss scaling.{' '}
        <strong>fp16</strong> spends
        those bits on the <em>mantissa</em> instead (finer precision), but its <strong>5-bit exponent</strong> gives a tiny
        range — so training in fp16 <strong>needs loss scaling</strong> to keep small gradients from underflowing to zero.
      </>
    ),
    svgAria:
      'Two stacked 16-bit float layouts. bf16: 1 sign, 8 exponent, 7 mantissa, exponent bias 127, matching fp32 dynamic range up to about 3.39e38. fp16: 1 sign, 5 exponent, 10 mantissa, exponent bias 15, maximum value 65504, which needs loss scaling for training. The exponent buys range; the mantissa buys precision.',
  },
  zh: {
    heading: '16-bit 的岔路口——同样的 bit，相反的取舍',
    pickAria: (f: string) => `高亮 ${f} 的 bit 布局`,
    youAreHere: '你在这里 · demo 用的是 bf16',
    sign: 'sign',
    exp: 'exponent',
    mant: 'mantissa',
    spendsRange: '把 bit 花在 RANGE 上（8-bit exponent）',
    spendsPrecision: '把 bit 花在 PRECISION 上（10-bit mantissa）',
    biasLabel: (b: number) => `exponent bias ${b}`,
    bf16Readout: ['range ≈ 1e38（max ≈ 3.39e38）', '~2–3 位十进制有效数字 · 与 FP32 同范围'],
    fp16Readout: ['max 65504（最小正规数 2^-14 ≈ 6.1e-5）', '~3.31 位十进制有效数字 · 需要 loss scaling'],
    caption: (
      <>
        同样是 16 bit，取舍却相反。<strong>bf16</strong> 保留了 fp32 完整的 <em>8-bit exponent</em>，所以它拥有和 fp32 一样巨大的{' '}
        <strong>dynamic range</strong>（~1e38）——只是相邻值之间的步长更粗。这正是它在训练里胜出的原因：
        梯度<strong>很少 overflow 或 underflow</strong>，所以通常不需要 loss scaling。<strong>fp16</strong> 则把这些 bit 花在了{' '}
        <em>mantissa</em> 上（精度更细），但它的 <strong>5-bit exponent</strong> 只能给出很小的范围——所以用 fp16 训练{' '}
        <strong>需要 loss scaling</strong>，以免微小的梯度下溢到零。
      </>
    ),
    svgAria:
      '两条堆叠的 16-bit 浮点布局。bf16：1 sign、8 exponent、7 mantissa，exponent bias 127，与 fp32 同样的 dynamic range，最大约 3.39e38。fp16：1 sign、5 exponent、10 mantissa，exponent bias 15，最大值 65504，训练时需要 loss scaling。exponent 买的是 range，mantissa 买的是 precision。',
  },
} as const;

// Locale-agnostic copy shape, so the SVG helpers accept either en or zh without
// narrowing to the English literal strings.
type Copy = (typeof COPY)[keyof typeof COPY];

// ── SVG geometry. EN labels are wider than zh, so every box is sized for EN. ──
// Two stacked rows; each row = a wide sub-label ABOVE its strip (never beside),
// the 16-cell strip itself, then a two-line readout below it. Wide labels sit
// above their strips so nothing collides across locales.
const VB_W = 560;
const VB_H = 300;
const PAD_L = 14;
const PAD_R = 14;
const TRACK_X = PAD_L;
const TRACK_W = VB_W - PAD_L - PAD_R; // 532
const CELL_W = TRACK_W / TOTAL_BITS; // width of one bit cell

// Row 1 (bf16) anchors.
const R1_TITLE_Y = 16; // "bf16" + bias, left/right of the row title line
const R1_SUB_Y = 32; // "spends its bits on RANGE …"
const R1_STRIP_Y = 40;
const STRIP_H = 30;
const R1_READ1_Y = 88; // first readout line (below strip)
const R1_READ2_Y = 104; // second readout line
// Row 2 (fp16) anchors — same offsets, shifted down by ROW_GAP.
const ROW_GAP = 132;
const R2_TITLE_Y = R1_TITLE_Y + ROW_GAP; // 148
const R2_SUB_Y = R1_SUB_Y + ROW_GAP; // 164
const R2_STRIP_Y = R1_STRIP_Y + ROW_GAP; // 172
const R2_READ1_Y = R1_READ1_Y + ROW_GAP; // 220
const R2_READ2_Y = R1_READ2_Y + ROW_GAP; // 236
// Bottom hint line (the universal rule).
const RULE_Y = 280;

function bitStrip(segs: Seg[], stripY: number, dim: boolean, copy: Copy) {
  let cellCursor = 0; // index of next bit cell (0..15)
  return segs.map((seg) => {
    const x = TRACK_X + cellCursor * CELL_W;
    const w = seg.bits * CELL_W;
    cellCursor += seg.bits;
    const labelText = seg.kind === 'sign' ? copy.sign : seg.kind === 'exp' ? copy.exp : copy.mant;
    const showLabel = w > 44; // sign (1 bit) is too narrow for a word
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
      <text
        x={TRACK_X}
        y={y2}
        style={{ fill: 'var(--muted-foreground)', opacity: op }}
        className="text-[10px]"
      >
        {lines[1]}
      </text>
    </>
  );
}

export function SixteenBitFork() {
  const copy = COPY[useLocale()];
  const [fmt, setFmt] = React.useState<Fmt>(DEMO_FMT);

  const bf = INFO.bf16;
  const fp = INFO.fp16;
  const bfActive = fmt === 'bf16';
  const fpActive = fmt === 'fp16';

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
                'px-2.5 py-1 font-mono text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
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

      {/* Locked "you are here · bf16" badge (always shown — bf16 is the demo). */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1.5 rounded border border-foreground/40 bg-foreground/5 px-2 py-0.5 text-[11px] font-medium text-foreground/80">
          🔒 {copy.youAreHere}
        </span>
      </div>

      {/* The two stacked bit strips */}
      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.svgAria}>
        {/* ── Row 1: bf16 ── */}
        <text
          x={TRACK_X}
          y={R1_TITLE_Y}
          style={{ fill: 'var(--foreground)', opacity: bfActive ? 1 : 0.6 }}
          className="font-mono text-[12px] font-semibold"
        >
          bf16
        </text>
        <text
          x={TRACK_X + TRACK_W}
          y={R1_TITLE_Y}
          textAnchor="end"
          style={{ fill: 'var(--muted-foreground)', opacity: bfActive ? 1 : 0.6 }}
          className="text-[10px]"
        >
          {copy.biasLabel(bf.bias)}
        </text>
        <text
          x={TRACK_X}
          y={R1_SUB_Y}
          style={{ fill: 'var(--primary)', opacity: bfActive ? 1 : 0.55 }}
          className="text-[10px] font-medium"
        >
          {copy.spendsRange}
        </text>
        {bitStrip(bf.segs, R1_STRIP_Y, !bfActive, copy)}
        {readoutLines(copy.bf16Readout, R1_READ1_Y, R1_READ2_Y, !bfActive)}

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

        {/* ── Row 2: fp16 ── */}
        <text
          x={TRACK_X}
          y={R2_TITLE_Y}
          style={{ fill: 'var(--foreground)', opacity: fpActive ? 1 : 0.6 }}
          className="font-mono text-[12px] font-semibold"
        >
          fp16
        </text>
        <text
          x={TRACK_X + TRACK_W}
          y={R2_TITLE_Y}
          textAnchor="end"
          style={{ fill: 'var(--muted-foreground)', opacity: fpActive ? 1 : 0.6 }}
          className="text-[10px]"
        >
          {copy.biasLabel(fp.bias)}
        </text>
        <text
          x={TRACK_X}
          y={R2_SUB_Y}
          style={{ fill: 'var(--muted-foreground)', opacity: fpActive ? 1 : 0.55 }}
          className="text-[10px] font-medium"
        >
          {copy.spendsPrecision}
        </text>
        {bitStrip(fp.segs, R2_STRIP_Y, !fpActive, copy)}
        {readoutLines(copy.fp16Readout, R2_READ1_Y, R2_READ2_Y, !fpActive)}

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

      {/* Caption: the exponent-buys-range, mantissa-buys-precision lesson */}
      <p className="border-t border-border/60 pt-3 text-[12px] text-muted-foreground">{copy.caption}</p>
    </div>
  );
}

import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * NF4BellCurve — why NormalFloat-4 beats plain int4 for LLM weights.
 *
 * Pretrained weights are ≈ Normal(0,1)-shaped after normalization, so NF4 places
 * its 16 levels at the QUANTILES of a normal: dense near the peak, sparse in the
 * tails. int4 instead spaces its 16 levels UNIFORMLY across [-1, 1], spending as
 * many codes on the rare tails as on the common centre. This diagram draws a bell
 * curve, the 16 NF4 level ticks under it (clustered at 0), and a contrast row of
 * 16 evenly-spaced int4 ticks. NF4 is a fixed 16-entry LOOKUP table (not a
 * computed grid), includes an exact 0, and is asymmetric (7 negative + zero + 8
 * positive = 16). Magnitude is restored by a per-block scale (block size 64);
 * QLoRA additionally "double-quantizes" those scales.
 *
 * Self-contained and STATIC: NO worker / model / WASM / network / animation —
 * pure React + Tailwind/SVG. The first render is deterministic and identical
 * server vs client (the page is prerendered to static HTML then hydrated); the
 * only state is a local view toggle, so it needs no prefers-reduced-motion logic.
 *
 * SOURCE-VERIFIED NUMBERS (none invented):
 *   - The 16 NF4 levels are the canonical bitsandbytes NF4 codebook, range
 *     [-1, 1] (QLoRA, arXiv:2305.14314). Counts: 7 negative, exact 0, 8 positive.
 *     Smallest gap (around 0) ≈ 0.0796; largest gap ≈ 0.304 (−1 → −0.696);
 *     the positive-tail gap (0.723 → 1) ≈ 0.277, so the tails span ≈ 0.28–0.30.
 *   - int4 contrast row: 16 evenly-spaced levels, step 2/15 ≈ 0.1333.
 *   - Block size 64 and the double-quantization of scales are QLoRA's defaults.
 * QUALITY is NOT plotted: the course gives no perplexity numbers, and "optimal for
 * normal data" is stated as the information-theoretic property it is, not measured.
 */

// Tailwind palette hexes for SVG (SVG fills can't take TW classes). Only these
// two literal hexes are allowed by convention.
const EMERALD = '#10b981'; // good / where the data (and NF4 codes) cluster
const RED = '#ef4444'; // wasted / int4 codes stranded in the empty tails

// ── Exact codebooks (do not edit without a source). ──────────────────────────
// Canonical bitsandbytes NF4 codebook, range [-1, 1] (QLoRA, arXiv:2305.14314).
const NF4_LEVELS = [
  -1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.0910500, 0.0, 0.0795803,
  0.1609302, 0.2461123, 0.3379152, 0.4407098, 0.5626170, 0.7229568, 1.0,
] as const;
// int4: 16 evenly-spaced levels across [-1, 1], step 2/15.
const INT4_LEVELS = Array.from({ length: 16 }, (_, i) => -1 + (2 * i) / 15);

// "Bulk" of the curve we shade as the common centre: |x| ≤ this. Derived purely
// for the visual band — NF4 puts 12 of its 16 codes inside |x| ≤ 0.56, int4 only 8.
const BULK = 0.5626170; // an exact NF4 level, so the band edge lands on a tick
const nf4InBulk = NF4_LEVELS.filter((v) => Math.abs(v) <= BULK).length; // 12
const int4InBulk = INT4_LEVELS.filter((v) => Math.abs(v) <= BULK).length; // 8

type View = 'spacing' | 'bulk';

// ── Per-locale copy. zh translates the prose but keeps glossary terms English
// (NF4, int4, nf4, quantile, scale, per-block, block, lookup table, QLoRA, weight,
// sign, exponent, mantissa, quantization, bits). Full-width punctuation in zh. ──
const COPY = {
  en: {
    heading: 'NF4 — put the levels where the weights actually are',
    viewLabel: 'view',
    viewSpacing: 'level spacing',
    viewBulk: 'codes in the bulk',
    pickSpacing: 'Show NF4 vs int4 level spacing',
    pickBulk: 'Highlight how many codes land in the common centre',
    curveLabel: 'pretrained weights ≈ Normal(0, 1)',
    nf4Row: 'NF4 · 16 levels at normal quantiles',
    int4Row: 'int4 · 16 evenly-spaced levels',
    denseTag: 'dense near 0',
    sparseTag: 'sparse in the tails',
    evenTag: 'same step everywhere',
    zeroTag: 'exact 0',
    bulkBand: (n: number) => `common centre |x| ≤ 0.56 — ${n} codes`,
    bulkNf4: (n: number) => `NF4: ${n} / 16 codes in the bulk`,
    bulkInt4: (n: number) => `int4: ${n} / 16 — the rest stranded in empty tails`,
    gapSmall: 'smallest gap ≈ 0.08 (around 0)',
    gapLarge: 'largest gap ≈ 0.30 (near ±1)',
    caption: (
      <>
        int4 spreads its 16 levels <strong>uniformly</strong>, so it spends as many codes on the rare tails as on the
        common centre. NF4 instead places its 16 levels at the <strong>quantiles</strong> of a normal — dense near the
        peak, sparse in the tails — which is information-theoretically optimal for normal-shaped data. NF4 is a fixed
        16-entry <strong>lookup table</strong> (not a computed grid), includes an <em>exact 0</em>, and is asymmetric (7
        negative + 0 + 8 positive = 16).
      </>
    ),
    scaleNote: (
      <>
        These are unit-range <em>shapes</em>: real magnitude is restored by a <strong>per-block scale</strong> (block
        size 64). QLoRA additionally <strong>double-quantizes</strong> those scales. Used by{' '}
        <span className="font-mono">QLoRA</span> (arXiv:2305.14314).
      </>
    ),
    aria: (v: View) =>
      v === 'bulk'
        ? `A Normal(0,1) bell curve over the range minus one to one. Of NF4's 16 lookup levels, ${nf4InBulk} fall inside the common centre (absolute value at most 0.56), versus only ${int4InBulk} of int4's 16 evenly spaced levels — the rest of int4's codes are stranded in the near-empty tails.`
        : `A Normal(0,1) bell curve with NF4's 16 level ticks beneath it, clustered densely near zero and sparse toward plus or minus one (smallest gap about 0.08 near zero, largest about 0.30 near the ends), contrasted with a row of int4's 16 evenly spaced ticks. NF4 includes an exact zero and is asymmetric: 7 negative, zero, and 8 positive levels.`,
  },
  zh: {
    heading: 'NF4——把档位放在 weight 真正所在的地方',
    viewLabel: '视图',
    viewSpacing: '档位间距',
    viewBulk: '落在主体的码',
    pickSpacing: '展示 NF4 与 int4 的档位间距',
    pickBulk: '高亮有多少码落在常见的中心区',
    curveLabel: '预训练 weight ≈ Normal(0, 1)',
    nf4Row: 'NF4 · 16 个档位，按 normal 的 quantile 分布',
    int4Row: 'int4 · 16 个均匀分布的档位',
    denseTag: '靠近 0 处密集',
    sparseTag: '尾部稀疏',
    evenTag: '处处步长相同',
    zeroTag: '精确的 0',
    bulkBand: (n: number) => `常见中心区 |x| ≤ 0.56——${n} 个码`,
    bulkNf4: (n: number) => `NF4：16 个码里有 ${n} 个落在主体`,
    bulkInt4: (n: number) => `int4：16 个里只有 ${n} 个——其余都困在空旷的尾部`,
    gapSmall: '最小间距 ≈ 0.08（0 附近）',
    gapLarge: '最大间距 ≈ 0.30（±1 附近）',
    caption: (
      <>
        int4 把它的 16 个档位<strong>均匀</strong>铺开，于是在罕见的尾部花掉的码和在常见中心区一样多。NF4 则把 16 个档位放在
        normal 的 <strong>quantile</strong> 上——靠近峰值处密集、尾部稀疏——这对正态形状的数据在信息论意义上是最优的。NF4 是一张固定的 16
        项 <strong>lookup table</strong>（不是计算出来的格点），包含一个<em>精确的 0</em>，且是非对称的（7 个负 + 0 + 8 个正 = 16）。
      </>
    ),
    scaleNote: (
      <>
        这些只是单位区间内的<em>形状</em>：真实幅度由 <strong>per-block scale</strong>（block 大小 64）还原。QLoRA 还会对这些 scale 做{' '}
        <strong>double-quantize</strong>（双重量化）。由 <span className="font-mono">QLoRA</span> 使用（arXiv:2305.14314）。
      </>
    ),
    aria: (v: View) =>
      v === 'bulk'
        ? `一条 Normal(0,1) 钟形曲线，范围从负一到一。NF4 的 16 个 lookup 档位中有 ${nf4InBulk} 个落在常见中心区（绝对值不超过 0.56），而 int4 的 16 个均匀档位只有 ${int4InBulk} 个落在其中——int4 其余的码都困在几乎空旷的尾部。`
        : `一条 Normal(0,1) 钟形曲线，下方是 NF4 的 16 个档位刻度，靠近 0 处密集、靠近正负一处稀疏（0 附近最小间距约 0.08，两端最大间距约 0.30），并与一行 int4 的 16 个均匀刻度对比。NF4 含一个精确的 0，且非对称：7 个负、一个 0、8 个正。`,
  },
} as const;

// ── SVG geometry. Sized for the wider EN labels; section labels sit ABOVE their
// rows, never wedged beside them, so nothing collides across locales. ──────────
const VB_W = 580;
const VB_H = 330;
const PAD_L = 24;
const PAD_R = 24;
const TRACK_X = PAD_L;
const TRACK_W = VB_W - PAD_L - PAD_R; // 532

// x mapping: value v ∈ [-1, 1] → pixel on the track.
const xOf = (v: number) => TRACK_X + ((v + 1) / 2) * TRACK_W;

// Bell curve band.
const CURVE_LABEL_Y = 16;
const CURVE_BASE_Y = 118; // x-axis the curve sits on
const CURVE_PEAK_H = 78; // height of the peak above the base
const Z_SPREAD = 2.6; // map visible [-1,1] onto ≈ ±2.6σ so the tails read as low

// NF4 tick row.
const NF4_LABEL_Y = 150;
const NF4_TICK_TOP = 158;
const NF4_TICK_BOT = 188;
// int4 tick row.
const INT4_LABEL_Y = 222;
const INT4_TICK_TOP = 230;
const INT4_TICK_BOT = 260;

// Smooth bell path (sampled N(0,1) shape) as a filled area to the baseline.
const BELL_PATH = (() => {
  const N = 64;
  const pts: string[] = [];
  for (let i = 0; i <= N; i++) {
    const v = -1 + (2 * i) / N;
    const z = v * Z_SPREAD;
    const y = Math.exp(-0.5 * z * z);
    const px = xOf(v);
    const py = CURVE_BASE_Y - y * CURVE_PEAK_H;
    pts.push(`${i === 0 ? 'M' : 'L'}${px.toFixed(1)} ${py.toFixed(1)}`);
  }
  return pts.join(' ');
})();
const BELL_AREA = `${BELL_PATH} L${xOf(1).toFixed(1)} ${CURVE_BASE_Y} L${xOf(-1).toFixed(1)} ${CURVE_BASE_Y} Z`;

export function NF4BellCurve() {
  const copy = COPY[useLocale()];
  const [view, setView] = React.useState<View>('spacing');
  const showBulk = view === 'bulk';

  const bulkL = xOf(-BULK);
  const bulkR = xOf(BULK);

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title + view toggle */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>
        <div className="flex items-center gap-1 text-[11px] text-muted-foreground">
          <span className="hidden sm:inline">{copy.viewLabel}:</span>
          <div className="flex overflow-hidden rounded border border-border">
            {(
              [
                ['spacing', copy.viewSpacing, copy.pickSpacing],
                ['bulk', copy.viewBulk, copy.pickBulk],
              ] as const
            ).map(([v, label, aria], i) => (
              <button
                key={v}
                type="button"
                onClick={() => setView(v)}
                aria-pressed={view === v}
                aria-label={aria}
                className={[
                  'px-2 py-1 text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                  i > 0 ? 'border-l border-border' : '',
                ].join(' ')}
                style={{
                  background: view === v ? 'var(--primary)' : 'transparent',
                  color: view === v ? 'var(--background)' : 'var(--muted-foreground)',
                }}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.aria(view)}>
        {/* ── bell curve ── */}
        <text x={TRACK_X} y={CURVE_LABEL_Y} style={{ fill: 'var(--muted-foreground)' }} className="text-[11px] font-medium">
          {copy.curveLabel}
        </text>

        {/* bulk shading band (only meaningful in the "bulk" view) */}
        {showBulk ? (
          <rect
            x={bulkL}
            y={CURVE_BASE_Y - CURVE_PEAK_H - 2}
            width={bulkR - bulkL}
            height={INT4_TICK_BOT - (CURVE_BASE_Y - CURVE_PEAK_H - 2)}
            style={{ fill: EMERALD, opacity: 0.1 }}
          />
        ) : null}

        {/* filled bell area + outline */}
        <path d={BELL_AREA} style={{ fill: 'var(--primary)', opacity: 0.12 }} />
        <path d={BELL_PATH} fill="none" style={{ stroke: 'var(--primary)', opacity: 0.85 }} strokeWidth={1.75} />
        {/* baseline / x-axis */}
        <line
          x1={TRACK_X}
          y1={CURVE_BASE_Y}
          x2={TRACK_X + TRACK_W}
          y2={CURVE_BASE_Y}
          style={{ stroke: 'var(--border)' }}
          strokeWidth={1}
        />
        {/* centre (0) guide */}
        <line
          x1={xOf(0)}
          y1={CURVE_BASE_Y - CURVE_PEAK_H - 2}
          x2={xOf(0)}
          y2={INT4_TICK_BOT}
          style={{ stroke: 'var(--foreground)', opacity: 0.18 }}
          strokeWidth={1}
          strokeDasharray="2 3"
        />

        {/* bulk-view band caption sits ABOVE the band edge, right-aligned, clear of ticks */}
        {showBulk ? (
          <text
            x={bulkR}
            y={CURVE_BASE_Y - CURVE_PEAK_H - 6}
            textAnchor="end"
            style={{ fill: EMERALD }}
            className="text-[10px] font-medium"
          >
            {copy.bulkBand(nf4InBulk)}
          </text>
        ) : null}

        {/* ── NF4 tick row ── label ABOVE the ticks ── */}
        <text x={TRACK_X} y={NF4_LABEL_Y} style={{ fill: 'var(--foreground)' }} className="text-[11px] font-semibold">
          {copy.nf4Row}
        </text>
        {NF4_LEVELS.map((v, i) => {
          const x = xOf(v);
          const isZero = v === 0;
          const inBulk = Math.abs(v) <= BULK;
          // In bulk view, colour: in-bulk emerald, tail muted. In spacing view, primary.
          const stroke = showBulk ? (inBulk ? EMERALD : 'var(--muted-foreground)') : 'var(--primary)';
          return (
            <line
              key={`nf4-${i}`}
              x1={x}
              y1={isZero ? NF4_TICK_TOP - 4 : NF4_TICK_TOP}
              x2={x}
              y2={NF4_TICK_BOT}
              style={{ stroke, opacity: showBulk && !inBulk ? 0.55 : 0.95 }}
              strokeWidth={isZero ? 2.5 : 1.5}
            />
          );
        })}
        {/* exact-0 callout under the NF4 row, centred on 0 */}
        <text
          x={xOf(0)}
          y={NF4_TICK_BOT + 12}
          textAnchor="middle"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[10px]"
        >
          {copy.zeroTag}
        </text>
        {/* bulk readout for the NF4 row — at the row's label line, right-aligned,
            never over ticks. (Gap callouts live in the legend below the SVG to
            avoid colliding with the exact-0 tag and axis markers.) */}
        {showBulk ? (
          <text
            x={TRACK_X + TRACK_W}
            y={NF4_LABEL_Y}
            textAnchor="end"
            style={{ fill: EMERALD }}
            className="text-[10px] font-medium"
          >
            {copy.bulkNf4(nf4InBulk)}
          </text>
        ) : null}

        {/* ── int4 tick row ── label ABOVE the ticks ── */}
        <text x={TRACK_X} y={INT4_LABEL_Y} style={{ fill: 'var(--foreground)' }} className="text-[11px] font-semibold">
          {copy.int4Row}
        </text>
        {INT4_LEVELS.map((v, i) => {
          const x = xOf(v);
          const inBulk = Math.abs(v) <= BULK;
          // In bulk view, the tail int4 codes are the "wasted" ones → RED.
          const stroke = showBulk ? (inBulk ? EMERALD : RED) : 'var(--muted-foreground)';
          return (
            <line
              key={`int4-${i}`}
              x1={x}
              y1={INT4_TICK_TOP}
              x2={x}
              y2={INT4_TICK_BOT}
              style={{ stroke, opacity: showBulk && !inBulk ? 0.85 : 0.9 }}
              strokeWidth={1.5}
            />
          );
        })}
        {/* int4 row readout — right end, clear of ticks (last tick is at the far edge,
            so this label sits ABOVE the row on the label line, never over a tick) */}
        {showBulk ? (
          <text
            x={TRACK_X + TRACK_W}
            y={INT4_LABEL_Y}
            textAnchor="end"
            style={{ fill: RED }}
            className="text-[10px] font-medium"
          >
            {copy.bulkInt4(int4InBulk)}
          </text>
        ) : (
          <text
            x={TRACK_X + TRACK_W}
            y={INT4_LABEL_Y}
            textAnchor="end"
            style={{ fill: 'var(--muted-foreground)' }}
            className="text-[10px]"
          >
            {copy.evenTag}
          </text>
        )}
        {/* axis end markers (−1, 0, +1) under the int4 row */}
        {[
          [-1, '−1'],
          [0, '0'],
          [1, '+1'],
        ].map(([v, lab]) => (
          <text
            key={`ax-${lab}`}
            x={xOf(v as number)}
            y={INT4_TICK_BOT + 13}
            textAnchor={v === -1 ? 'start' : v === 1 ? 'end' : 'middle'}
            style={{ fill: 'var(--muted-foreground)' }}
            className="font-mono text-[10px]"
          >
            {lab as string}
          </text>
        ))}
      </svg>

      {/* dense / sparse legend tags below the SVG (text, never inside the plot) */}
      {!showBulk ? (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
          <span className="inline-flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: 'var(--primary)' }} />
            {copy.denseTag}
          </span>
          <span>{copy.sparseTag}</span>
          <span>{copy.gapSmall}</span>
          <span>{copy.gapLarge}</span>
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
          <span className="inline-flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: EMERALD }} />
            {copy.bulkNf4(nf4InBulk)}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: RED }} />
            {copy.bulkInt4(int4InBulk)}
          </span>
        </div>
      )}

      {/* caption: the lesson */}
      <p className="border-t border-border/60 pt-3 text-[12px] text-muted-foreground">{copy.caption}</p>
      {/* scale / QLoRA framing */}
      <p className="text-[12px] text-muted-foreground">{copy.scaleNote}</p>
    </div>
  );
}

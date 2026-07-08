import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * NumberLineGrid — the single most important diagram in the "how numbers are
 * stored in bits" chapter. Four stacked number lines, all drawn on the SAME
 * shared linear x-axis normalized to [0, 1] = value / that-format's-own-max.
 * Each representable POSITIVE value is a short vertical tick.
 *
 *   integers  → a UNIFORM grid: constant ABSOLUTE step everywhere, no exponent,
 *               so no dynamic range. Same gap from 0→1 as from (max−1)→max.
 *   floats    → a NON-uniform, LOG-spaced grid: constant RELATIVE step. Ticks
 *               pile up densely near 0 (fine precision where the values are) and
 *               thin out toward the max (coarse, but huge range). That is
 *               "exponent buys dynamic range, mantissa buys precision" made
 *               visible.
 *
 * Self-contained and STATIC: NO worker, NO model, NO WASM, NO network, NO
 * animation — pure React + Tailwind/SVG. First render is deterministic and
 * identical server vs client (the page is prerendered then hydrated); the only
 * state is which row is highlighted, so it needs no reduced-motion handling.
 *
 * EVERY NUMBER IS SOURCE-VERIFIED — none invented:
 *   - int8  : 8-bit two's-complement → 2^8 = 256 distinct codes/levels.
 *   - int4  : 4-bit → 2^4 = 16 distinct codes/levels.
 *   - fp8 E4M3 (OCP / NVIDIA Hopper): 1 sign · 4 exponent · 3 mantissa. Positive
 *     NORMAL values are value = 2^e · (1 + m/8), with the 3 mantissa bits giving
 *     2^3 = 8 steps per octave. Per the OCP MX / FP8 spec E4M3 has NO inf and a
 *     reduced range, so its max finite magnitude is 448 (not 2^8·1.875): exponent
 *     octaves run 2^-6 … 2^8 with the top octave's mantissa clamped so the
 *     largest value is 448. We render the normalized positions value/448.
 *   - fp16 (IEEE 754 binary16): 1 sign · 5 exponent · 10 mantissa, max finite
 *     magnitude 65504. Same octave·mantissa construction, normalized by 65504 —
 *     more octaves ⇒ even more extreme bunching near 0.
 * The tick RENDERING is schematic (we thin the float ticks so the strip stays
 * legible) but the SPACING PATTERN — uniform for ints, log for floats — is
 * faithful: each float tick sits at its true value/max position.
 */

type Fmt = 'int8' | 'int4' | 'fp8' | 'fp16';
const ROWS: Fmt[] = ['int8', 'int4', 'fp8', 'fp16'];
const DEFAULT_HIGHLIGHT: Fmt = 'fp8'; // fp8 E4M3 makes the bunching obvious

const EMERALD = '#10b981'; // floats: log-spaced (dense near 0)

type RowInfo = {
  kind: 'uniform' | 'log';
  /** exact level count for ints; undefined for floats */
  levels?: number;
  /** exact max finite magnitude for floats; undefined for ints */
  max?: number;
};

const INFO: Record<Fmt, RowInfo> = {
  int8: { kind: 'uniform', levels: 256 },
  int4: { kind: 'uniform', levels: 16 },
  fp8: { kind: 'log', max: 448 },
  fp16: { kind: 'log', max: 65504 },
};

// ── Tick generators. All return normalized x in [0, 1] = value / max. ─────────

/** Uniform ints: `count` evenly spaced ticks across [0, 1] (schematic for int8,
 *  exact for int4). Same absolute gap everywhere — that IS the lesson. */
function uniformTicks(count: number): number[] {
  if (count <= 1) return [0];
  return Array.from({ length: count }, (_, i) => i / (count - 1));
}

/** Float normals: value = 2^e · (1 + m/8) for e in [eLo, eHi], m in 0..7, then
 *  normalized by `max`. Bunches near 0, spreads toward 1. We thin to the higher
 *  octaves so the strip stays legible while keeping the log SPACING faithful.
 *  The exact max (x = 1) is always appended so the line visibly reaches its own
 *  maximum — the real mantissa has more steps than the 8 we render per octave. */
function floatTicks(eLo: number, eHi: number, max: number): number[] {
  const xs: number[] = [1]; // the format's exact max sits at value/max = 1
  for (let e = eLo; e <= eHi; e++) {
    for (let m = 0; m < 8; m++) {
      const v = Math.pow(2, e) * (1 + m / 8);
      if (v <= max) xs.push(v / max);
    }
  }
  // dedupe + sort for clean rendering
  return Array.from(new Set(xs.map((x) => Math.min(1, x)))).sort((a, b) => a - b);
}

// Per-row tick sets. Floats use only the higher octaves (the lowest ones crowd
// into a few pixels at x≈0 and would just be a smear); the chosen ranges still
// span ~0 → 1 and preserve the dense-near-0 / sparse-near-max signature.
const TICKS: Record<Fmt, number[]> = {
  int8: uniformTicks(32), // representative of 256 — reads as a uniform lattice
  int4: uniformTicks(16), // all 16 levels
  fp8: floatTicks(0, 8, 448), // 2^0 (=1/448) … 2^8 octaves of E4M3
  fp16: floatTicks(4, 15, 65504), // higher octaves of binary16 → tighter bunch near 0
};

// ── Per-locale copy. en = original English. zh translates the prose but keeps
// the glossary terms English (int8, int4, fp8, E4M3, fp16, exponent, mantissa,
// sign, bits) and uses full-width punctuation. Conceptual terms that read
// naturally in Chinese are translated (动态范围 / 精度), matching the chapter prose. ──
const COPY = {
  en: {
    heading: 'Number line — integers are a uniform grid, floats are log-spaced',
    highlightLabel: 'highlight',
    pickAria: (f: string) => `Highlight the ${f} number line`,
    axisLabel: 'value / max  (0 → that format’s own maximum)',
    uniform: 'uniform',
    logSpaced: 'log-spaced',
    rowAria: (f: string, tag: string) => `${f} number line — a ${tag} grid of representable positive values.`,
    svgAria:
      'Four stacked number lines on a shared 0-to-1 axis. The int8 and int4 rows show evenly spaced ticks (a uniform grid). The fp8 E4M3 and fp16 rows show ticks bunched near zero and spread out toward the maximum (a log-spaced grid).',
    note: {
      int8: '256 levels',
      int4: '16 levels',
      fp8: 'max 448',
      fp16: 'max 65504',
    } satisfies Record<Fmt, string>,
    caption: (
      <>
        Integers spread their levels <strong>evenly</strong> — the same step everywhere, and with no{' '}
        <em>exponent</em> there is no dynamic range. Floats spend bits on an <em>exponent</em>, so their levels{' '}
        <strong>cluster near zero</strong> (fine precision where the values actually are) and{' '}
        <strong>spread out toward the maximum</strong> (coarse, but a huge range). Same idea, opposite grid.
      </>
    ),
    facts: 'int8 = 256 levels · int4 = 16 levels · fp8 E4M3 max 448 · fp16 max 65504',
  },
  zh: {
    heading: '数轴——整数是均匀格点，浮点是 log 间距',
    highlightLabel: '高亮',
    pickAria: (f: string) => `高亮 ${f} 数轴`,
    axisLabel: 'value / max（0 → 该格式自己的最大值）',
    uniform: 'uniform',
    logSpaced: 'log-spaced',
    rowAria: (f: string, tag: string) => `${f} 数轴——可表示正值构成的 ${tag} 格点。`,
    svgAria:
      '四条共享 0 到 1 坐标轴的数轴。int8 与 int4 行的刻度均匀分布（uniform 格点）。fp8 E4M3 与 fp16 行的刻度在 0 附近密集、向最大值方向稀疏（log 间距格点）。',
    note: {
      int8: '256 个档位',
      int4: '16 个档位',
      fp8: 'max 448',
      fp16: 'max 65504',
    } satisfies Record<Fmt, string>,
    caption: (
      <>
        整数把档位 <strong>均匀</strong> 铺开——处处步长相同，而且没有 <em>exponent</em> 就没有动态范围。浮点把 bit 花在{' '}
        <em>exponent</em> 上，于是档位 <strong>在 0 附近聚拢</strong>（值真正所在之处精度更细），并{' '}
        <strong>向最大值方向散开</strong>（粗糙，但范围巨大）。同一个想法，相反的格点。
      </>
    ),
    facts: 'int8 = 256 个档位 · int4 = 16 个档位 · fp8 E4M3 max 448 · fp16 max 65504',
  },
} as const;

// ── SVG geometry. Sized for the wider EN labels. Each row: a left format label
// + "uniform/log-spaced" tag sit ABOVE the track (never wedged beside it), then
// a horizontal track with vertical ticks, then a right-side note. ─────────────
const VB_W = 600;
const PAD_L = 14;
const PAD_R = 14;
const TRACK_X = PAD_L;
const TRACK_W = VB_W - PAD_L - PAD_R; // 572

const ROW_TOP = 26; // y of first row's label baseline
// Wide enough that a row's below-track readout (at trackY + TRACK_H + 13 ≈
// labelY + 53) clears the NEXT row's name/tag label (at the next labelY), leaving
// ~25px between the two text lines so nothing overlaps across the inter-row gap.
const ROW_GAP = 78; // vertical distance between row label baselines
const LABEL_DY = 10; // track sits this far below the label baseline
const TRACK_H = 30; // height of the tick band
const TICK_H = 20; // tick mark height inside the band
const AXIS_Y = ROW_TOP + (ROWS.length - 1) * ROW_GAP + LABEL_DY + TRACK_H + 36;
const VB_H = AXIS_Y + 14;

function rowY(i: number): number {
  return ROW_TOP + i * ROW_GAP;
}

export function NumberLineGrid() {
  const copy = COPY[useLocale()];
  const [hi, setHi] = React.useState<Fmt>(DEFAULT_HIGHLIGHT);

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header: title + highlight toggle */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>
        <div className="flex items-center gap-1 text-[11px] text-muted-foreground">
          <span className="hidden sm:inline">{copy.highlightLabel}:</span>
          <div className="flex overflow-hidden rounded border border-border">
            {ROWS.map((f, i) => (
              <button
                key={f}
                type="button"
                onClick={() => setHi(f)}
                aria-pressed={hi === f}
                aria-label={copy.pickAria(f)}
                className={[
                  'px-2 py-1 font-mono text-[11px] font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                  i > 0 ? 'border-l border-border' : '',
                ].join(' ')}
                style={{
                  background: hi === f ? 'var(--primary)' : 'transparent',
                  color: hi === f ? 'var(--background)' : 'var(--muted-foreground)',
                }}
              >
                {f}
              </button>
            ))}
          </div>
        </div>
      </div>

      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.svgAria}>
        {ROWS.map((f, i) => {
          const info = INFO[f];
          const isFloat = info.kind === 'log';
          const tag = isFloat ? copy.logSpaced : copy.uniform;
          const selected = hi === f;
          const labelY = rowY(i);
          const trackY = labelY + LABEL_DY;
          const ticks = TICKS[f];
          // Highlighted row: primary color, thicker ticks. Floats otherwise
          // emerald (log-spaced family), ints muted-foreground (uniform family).
          const tickColor = selected ? 'var(--primary)' : isFloat ? EMERALD : 'var(--muted-foreground)';
          const tickW = selected ? 2 : 1;
          const tickOpacity = selected ? 0.95 : 0.55;
          const tickInsetY = trackY + (TRACK_H - TICK_H) / 2;

          return (
            <g key={f} aria-label={copy.rowAria(f, tag)}>
              {/* row label (format name) ABOVE the track, left-aligned */}
              <text
                x={TRACK_X}
                y={labelY}
                style={{ fill: selected ? 'var(--primary)' : 'var(--foreground)' }}
                className="font-mono text-[12px] font-semibold"
              >
                {f}
              </text>
              {/* uniform / log-spaced tag ABOVE-right of the track */}
              <text
                x={TRACK_X + TRACK_W}
                y={labelY}
                textAnchor="end"
                style={{ fill: isFloat ? EMERALD : 'var(--muted-foreground)' }}
                className="text-[11px] font-medium"
              >
                {tag}
              </text>

              {/* the track baseline */}
              <rect
                x={TRACK_X}
                y={trackY}
                width={TRACK_W}
                height={TRACK_H}
                rx={4}
                fill="none"
                style={{ stroke: 'var(--border)', opacity: selected ? 1 : 0.7 }}
                strokeWidth={1}
              />
              {/* representable positive value ticks */}
              {ticks.map((x, k) => {
                const px = TRACK_X + x * TRACK_W;
                return (
                  <line
                    key={k}
                    x1={px}
                    y1={tickInsetY}
                    x2={px}
                    y2={tickInsetY + TICK_H}
                    style={{ stroke: tickColor, opacity: tickOpacity }}
                    strokeWidth={tickW}
                    strokeLinecap="round"
                  />
                );
              })}
              {/* right-side mini readout (levels for ints, max for floats),
                  placed BELOW the track so it never sits over the ticks */}
              <text
                x={TRACK_X + TRACK_W}
                y={trackY + TRACK_H + 13}
                textAnchor="end"
                style={{ fill: 'var(--muted-foreground)' }}
                className="font-mono text-[10px]"
              >
                {copy.note[f]}
              </text>
              {/* "0" anchor under the left end of the track */}
              <text
                x={TRACK_X}
                y={trackY + TRACK_H + 13}
                style={{ fill: 'var(--muted-foreground)' }}
                className="font-mono text-[10px]"
              >
                0
              </text>
            </g>
          );
        })}

        {/* shared axis caption */}
        <line
          x1={TRACK_X}
          y1={AXIS_Y - 10}
          x2={TRACK_X + TRACK_W}
          y2={AXIS_Y - 10}
          style={{ stroke: 'var(--border)' }}
          strokeWidth={1}
        />
        <text
          x={TRACK_X + TRACK_W / 2}
          y={AXIS_Y}
          textAnchor="middle"
          style={{ fill: 'var(--muted-foreground)' }}
          className="text-[10px]"
        >
          {copy.axisLabel}
        </text>
      </svg>

      {/* caption + exact facts */}
      <p className="text-[13px] text-foreground/90">{copy.caption}</p>
      <p className="border-t border-border/60 pt-3 font-mono text-[11px] text-muted-foreground">{copy.facts}</p>
    </div>
  );
}

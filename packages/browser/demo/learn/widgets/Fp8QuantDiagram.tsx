import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Fp8QuantDiagram — chapter 4 deep-dive, the PRECISION side of FlashAttention-3's
 * FP8 story (Flash3AsyncDiagram covers the THROUGHPUT side).
 *
 * Self-contained: NO worker, NO model, NO WASM — pure SVG + React state, so it
 * server-renders for crawlers (createRoot replaces it on boot) and is robust
 * while the model downloads.
 *
 * It teaches the two tricks FA3 (arXiv 2407.08608) uses to keep FP8 matmuls
 * accurate. An E4M3 fp8 number has 256 bit patterns and a max of 448, so
 * everything hinges on the scale that maps your data onto that grid:
 *   1. block quantization — per-block scales instead of one tensor-wide scale,
 *      so a single outlier only coarsens its own block;
 *   2. incoherent processing — multiply by a fixed Hadamard-style rotation
 *      before quantizing, smearing the outlier's energy across all dimensions
 *      (the rotation cancels inside QKᵀ before rounding; after FP8 rounding it
 *      is an approximation whose job is to shrink the rounding error).
 *
 * All numbers shown are REAL round-trips through a faithful E4M3 quantizer
 * (sign + 4 exponent bits + 3 mantissa bits, subnormals, max finite 448),
 * computed at module scope from the fixed example below — the dots, ticks and
 * error percentages can never disagree. The rotated vector is the (hardcoded)
 * product of the 16 values with a normalized 16×16 Sylvester Hadamard matrix
 * H/4; nothing is random at render time.
 *
 * Animation model: dots ping-pong between "original value" and "where the fp8
 * grid puts it" (CSS transform transition), flipped by a setInterval inside a
 * guarded useEffect. Honors prefers-reduced-motion (static: dots rest at their
 * quantized positions, no autoplay).
 */

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

// ── Fixed example: 15 small values (two of them ≈0, like real activations) plus
// one outlier, split into 4 blocks of 4. Index 10 (block C) is the outlier. ───
const VALUES = [
  0.62, -0.41, 0.0001, 0.93, // block A
  -0.77, 0.05, 0.34, -0.59, // block B
  0.81, -0.00015, 52.0, 0.47, // block C ← holds the outlier
  -0.95, 0.26, -0.68, 0.73, // block D
];
const OUTLIER_INDEX = 10;
const OUTLIER = VALUES[OUTLIER_INDEX]; // 52

// VALUES multiplied by a FIXED normalized 16×16 Sylvester Hadamard matrix H/4
// (H·Hᵀ = 16·I, so H/4 is orthogonal). Precomputed offline and hardcoded —
// deterministic, nothing random at render time.
const ROTATED = [
  13.202487, 12.482563, -13.397562, -12.577488, 14.007488, 13.737562, -12.792562, -11.802487, -13.117437, -12.377513,
  12.802513, 12.682438, -12.952438, -13.742513, 12.667512, 13.657437,
];

// ── Faithful E4M3 round-to-nearest quantizer ─────────────────────────────────
// 1 sign + 4 exponent bits (bias 7) + 3 mantissa bits. Max finite 448
// (exponent 1111 + mantissa 111 is NaN), min normal 2^-6, subnormal step 2^-9.
const E4M3_MAX = 448;
const E4M3_MIN_NORMAL = 2 ** -6;
const E4M3_SUB_STEP = 2 ** -9;

function e4m3(v: number): number {
  if (v === 0 || !Number.isFinite(v)) return 0;
  const sign = Math.sign(v);
  const a = Math.abs(v);
  if (a >= E4M3_MAX) return sign * E4M3_MAX; // saturate
  if (a < E4M3_MIN_NORMAL) return sign * Math.round(a / E4M3_SUB_STEP) * E4M3_SUB_STEP;
  let e = Math.floor(Math.log2(a));
  if (2 ** e > a) e -= 1;
  if (2 ** (e + 1) <= a) e += 1;
  const step = 2 ** e / 8; // ULP within this exponent bin
  return sign * Math.min(Math.round(a / step) * step, E4M3_MAX);
}

/** Round-trip x through fp8 with the given scale: dequantize(quantize(x)). */
function roundTrip(x: number, scale: number): number {
  return e4m3(x * scale) / scale;
}

const amax = (arr: number[]): number => Math.max(...arr.map(Math.abs));

// All positive values an E4M3 number can take, ascending (subnormals first).
const E4M3_LEVELS: number[] = (() => {
  const out: number[] = [];
  for (let k = 1; k <= 7; k++) out.push(k * E4M3_SUB_STEP);
  for (let e = -6; e <= 8; e++) {
    for (let m = 8; m <= 15; m++) {
      const v = (m / 8) * 2 ** e;
      if (v <= E4M3_MAX) out.push(v);
    }
  }
  return out;
})();

// ── SVG geometry ─────────────────────────────────────────────────────────────
const VB_W = 760;
const PLOT_X = 80; // left edge of the number lines (room for block labels)
const PLOT_W = 590; // width of the number lines (right margin holds scale labels)
const LANE_Y0 = 34;
const LANE_H = 38;
const LANE_STRIDE = 47;
const AXIS_Y = LANE_Y0 + 4 * LANE_STRIDE + 6;
const VB_H = AXIS_Y + 18;

function xOf(v: number, axisMax: number): number {
  return PLOT_X + ((v + axisMax) / (2 * axisMax)) * PLOT_W;
}

/** One `<path>` of tick marks: every fp8-representable level (dequantized by
 *  `scale`) inside ±axisMax, dropped when closer than 2.4px to the last one. */
function ticksPath(scale: number, axisMax: number, laneTop: number): string {
  const y1 = (laneTop + 6).toFixed(1);
  const y2 = (laneTop + LANE_H - 6).toFixed(1);
  const xs: number[] = [];
  let lastPx = xOf(0, axisMax);
  for (const lvl of E4M3_LEVELS) {
    const v = lvl / scale;
    if (v > axisMax) break;
    const px = xOf(v, axisMax);
    if (px - lastPx >= 2.4) {
      xs.push(v);
      lastPx = px;
    }
  }
  let d = '';
  for (const v of xs) {
    d += `M${xOf(v, axisMax).toFixed(1)} ${y1}V${y2}M${xOf(-v, axisMax).toFixed(1)} ${y1}V${y2}`;
  }
  return d;
}

// ── Precompute the three modes entirely at module scope ─────────────────────
type ModeId = 'tensor' | 'block' | 'rotate';

type Dot = {
  v: number; // input value (original, or rotated in mode 3)
  q: number; // value after fp8 round-trip
  errPct: number;
  offScale: boolean; // the outlier in modes 1–2: outside the zoomed axis
};

type Lane = {
  letter: string;
  scale: number;
  coarse: boolean; // still pinned by the outlier
  ticks: string;
  dots: Dot[];
};

type Mode = {
  id: ModeId;
  avgPct: number;
  axisMax: number;
  lanes: Lane[];
};

const LETTERS = ['A', 'B', 'C', 'D'];

function buildMode(id: ModeId, source: number[], scales: number[], axisMax: number): Mode {
  let errSum = 0;
  let errCount = 0;
  const lanes: Lane[] = LETTERS.map((letter, b) => {
    const scale = scales[b];
    const laneTop = LANE_Y0 + b * LANE_STRIDE;
    const dots: Dot[] = [];
    for (let i = b * 4; i < b * 4 + 4; i++) {
      const v = source[i];
      const q = roundTrip(v, scale);
      const errPct = (Math.abs(q - v) / Math.abs(v)) * 100;
      const isOutlierSlot = id !== 'rotate' && i === OUTLIER_INDEX;
      if (!isOutlierSlot) {
        errSum += errPct;
        errCount += 1;
      }
      dots.push({ v, q, errPct, offScale: isOutlierSlot });
    }
    return { letter, scale, coarse: scale < 100, ticks: ticksPath(scale, axisMax, laneTop), dots };
  });
  return { id, avgPct: errSum / errCount, axisMax, lanes };
}

const TENSOR_SCALE = E4M3_MAX / amax(VALUES); // 448/52 ≈ 8.6 — the outlier pins it
const BLOCK_SCALES = LETTERS.map((_, b) => E4M3_MAX / amax(VALUES.slice(b * 4, b * 4 + 4)));
const ROT_SCALES = LETTERS.map((_, b) => E4M3_MAX / amax(ROTATED.slice(b * 4, b * 4 + 4)));
const ROT_MAX = amax(ROTATED); // ≈ 14.0 — the smeared outlier

const MODES: Mode[] = [
  buildMode('tensor', VALUES, [TENSOR_SCALE, TENSOR_SCALE, TENSOR_SCALE, TENSOR_SCALE], 1.05),
  buildMode('block', VALUES, BLOCK_SCALES, 1.05),
  buildMode('rotate', ROTATED, ROT_SCALES, 15),
];
// Sanity: these render as 11.5% → 5.0% → 1.2%, strictly decreasing.

const FLIP_MS = 1500;

function fmtVal(v: number): string {
  if (Math.abs(v) >= 10) return v.toFixed(1);
  if (Math.abs(v) < 0.005 && v !== 0) return v.toString();
  return v.toFixed(2);
}

// Per-locale copy. English strings are moved verbatim from the original JSX.
// Numbers (448, scales, error percentages) come from the same module-scope
// computation in both locales — only the prose around them changes.
const COPY = {
  en: {
    heading: 'FP8 rounding error — and the two tricks that tame it',
    pause: '❚❚ Pause',
    play: '▶ Play',
    modes: {
      tensor: { title: 'One scale for everything', avgLabel: 'avg error, 15 small values' },
      block: { title: 'Block quantization', avgLabel: 'avg error, 15 small values' },
      rotate: { title: '+ Incoherent processing', avgLabel: 'avg error, 16 stored values' },
    },
    svgAria: (title: string, avgLabel: string, pct: string) =>
      `Four number lines, one per block of four values. Tick marks show the fp8-representable levels for the current scale; each dot moves from its original value to the nearest level. Mode: ${title}, ${avgLabel} ${pct} percent.`,
    legendOriginal: 'original value',
    legendRoundTrip: 'after fp8 round-trip',
    legendLevels: 'fp8-representable levels',
    blockLabel: (letter: string) => `block ${letter}`,
    laneRotated: 'rotated',
    laneFourValues: '4 values',
    outlierLabel: `outlier ${OUTLIER} →`,
    dotTitle: (v: string, q: string, errPct: string) => `${v} → ${q} (${errPct}% off)`,
    axisRotate: 'rotated values — the outlier is gone',
    axisZoom: 'zoomed to ±1 — the outlier sits far off to the right',
    captionTensor: (
      <>
        <strong>The outlier wins.</strong> One scale must fit {OUTLIER}, so scale = 448/{OUTLIER} ≈{' '}
        <span className="font-mono">{TENSOR_SCALE.toFixed(1)}</span> — every row gets the same coarse grid (the
        ticks). The 15 small values snap onto a handful of levels, and the two near-zero values round almost
        entirely away (up to 100% error). Avg round-trip error of the 15 small values:{' '}
        <span className="font-mono text-primary">{MODES[0].avgPct.toFixed(1)}%</span>.
      </>
    ),
    captionBlock: (
      <>
        <strong>Four blocks, four scales</strong> — this is what FA3 does per tile. Blocks A, B and D pick fine
        grids (denser ticks) and their values barely move when they snap. But block C still holds the {OUTLIER}, so
        its scale stays coarse — its near-zero neighbour keeps a ~
        <span className="font-mono">{MODES[1].lanes[2].dots[1].errPct.toFixed(0)}%</span> error. Avg error of the
        15 small values: <span className="font-mono text-primary">{MODES[1].avgPct.toFixed(1)}%</span>.
      </>
    ),
    captionRotate: (
      <>
        <strong>Rotate first, then quantize.</strong> A fixed Hadamard rotation smears the {OUTLIER} across all 16
        slots: the biggest value drops {OUTLIER} → <span className="font-mono">{ROT_MAX.toFixed(1)}</span> (≈4×, that
        is √16), so every block&apos;s scale gets ≈4× finer. No outliers and no near-zeros remain — every stored
        value is mid-sized and lands within a few percent of itself. Avg error of the 16 stored values:{' '}
        <span className="font-mono text-primary">{MODES[2].avgPct.toFixed(1)}%</span>.
      </>
    ),
    footnote: (
      <>
        Before any rounding, the rotation is exactly free: rotate every q and k vector with the same orthogonal R and
        it cancels inside the matmul — (QRᵀ)(KRᵀ)ᵀ = Q(RᵀR)Kᵀ = QKᵀ, since RᵀR = I. After FP8 rounding nothing is
        exact anymore — the rotation&apos;s job is to make that rounding <em>cheaper</em>, which is exactly the error
        drop shown above. All percentages are real round-trips through an E4M3 quantizer (1 sign + 4 exponent + 3
        mantissa bits, max 448) on these 16 fixed values; in mode 3 the error is measured on the rotated values,
        because those are what actually get stored and multiplied.
      </>
    ),
  },
  zh: {
    heading: 'FP8 舍入误差——以及驯服它的两个技巧',
    pause: '❚❚ 暂停',
    play: '▶ 播放',
    modes: {
      tensor: { title: '整个张量一个缩放', avgLabel: '15 个小值的平均误差' },
      block: { title: '块量化', avgLabel: '15 个小值的平均误差' },
      rotate: { title: '+ 不相干处理', avgLabel: '16 个存储值的平均误差' },
    },
    svgAria: (title: string, avgLabel: string, pct: string) =>
      `四条数线，每条对应一组四个值。刻度线表示当前缩放下 fp8 可表示的级别；每个点从其原始值移动到最近的级别。模式：${title}，${avgLabel} ${pct}%。`,
    legendOriginal: '原始值',
    legendRoundTrip: '经 fp8 往返后',
    legendLevels: 'fp8 可表示的级别',
    blockLabel: (letter: string) => `块 ${letter}`,
    laneRotated: '旋转后',
    laneFourValues: '4 个值',
    outlierLabel: `离群值 ${OUTLIER} →`,
    dotTitle: (v: string, q: string, errPct: string) => `${v} → ${q}（误差 ${errPct}%）`,
    axisRotate: '旋转后的值——离群值消失了',
    axisZoom: '放大到 ±1——离群值远在右侧之外',
    captionTensor: (
      <>
        <strong>离群值赢了。</strong>一个缩放必须容下 {OUTLIER}，于是 scale = 448/{OUTLIER} ≈{' '}
        <span className="font-mono">{TENSOR_SCALE.toFixed(1)}</span>——每一行都拿到同一张粗网格（即那些刻度）。15
        个小值只能吸附到寥寥几级上，两个近零值几乎被整个舍掉（误差最高 100%）。15 个小值的平均往返误差：
        <span className="font-mono text-primary">{MODES[0].avgPct.toFixed(1)}%</span>。
      </>
    ),
    captionBlock: (
      <>
        <strong>四个块，四个缩放</strong>——FA3 在每个 tile 上做的正是这件事。块 A、B、D
        选到细网格（刻度更密），它们的值吸附时几乎不动。但块 C 仍然装着 {OUTLIER}
        ，它的缩放只能保持粗糙——它旁边的近零值仍有约{' '}
        <span className="font-mono">{MODES[1].lanes[2].dots[1].errPct.toFixed(0)}%</span> 的误差。15
        个小值的平均误差：<span className="font-mono text-primary">{MODES[1].avgPct.toFixed(1)}%</span>。
      </>
    ),
    captionRotate: (
      <>
        <strong>先旋转，再量化。</strong>一个固定的 Hadamard 旋转把 {OUTLIER} 抹匀到全部 16 个槽位：最大值从{' '}
        {OUTLIER} 降到 <span className="font-mono">{ROT_MAX.toFixed(1)}</span>（≈4×，即 √16），于是每个块的缩放都细了约
        4×。不再有离群值，也不再有近零值——每个存储值都是中等大小，吸附后离自己只差几个百分点。16
        个存储值的平均误差：<span className="font-mono text-primary">{MODES[2].avgPct.toFixed(1)}%</span>。
      </>
    ),
    footnote: (
      <>
        在任何舍入发生之前，旋转是完全免费的：用同一个正交矩阵 R 旋转每个 q 和 k
        向量，它会在矩阵乘法内部抵消——(QRᵀ)(KRᵀ)ᵀ = Q(RᵀR)Kᵀ = QKᵀ，因为 RᵀR = I。经过 FP8
        舍入之后，一切都不再精确——旋转的职责是让那次舍入变得更<em>便宜</em>
        ，也就是上面展示的误差下降。所有百分比都是这 16 个固定值经过 E4M3 量化器（1 个符号位 + 4 个指数位 + 3
        个尾数位，最大 448）的真实往返；模式 3 的误差是在旋转后的值上测量的，因为真正被存储和相乘的就是它们。
      </>
    ),
  },
} as const;

export function Fp8QuantDiagram() {
  const reducedMotion = usePrefersReducedMotion();
  const copy = COPY[useLocale()];
  const [modeId, setModeId] = React.useState<ModeId>('tensor');
  const [playing, setPlaying] = React.useState<boolean>(!reducedMotion);
  const [snapped, setSnapped] = React.useState<boolean>(false); // false = original positions, true = fp8 grid

  const mode = MODES.find((m) => m.id === modeId) ?? MODES[0];

  // Ping-pong the dots between "original" and "snapped to the fp8 grid".
  React.useEffect(() => {
    if (!playing || reducedMotion) return;
    const id = window.setInterval(() => setSnapped((s) => !s), FLIP_MS);
    return () => window.clearInterval(id);
  }, [playing, reducedMotion]);

  // Restart from the original positions when the mode changes, so each mode
  // opens with a fresh "value → grid" snap.
  React.useEffect(() => {
    setSnapped(false);
  }, [modeId]);

  const showQuantized = reducedMotion ? true : snapped;

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          {copy.heading}
        </div>
        {!reducedMotion ? (
          <button
            type="button"
            onClick={() => setPlaying((p) => !p)}
            aria-pressed={playing}
            className="rounded px-2.5 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-primary"
          >
            {playing ? copy.pause : copy.play}
          </button>
        ) : null}
      </div>

      {/* Mode toggle — each button carries its honest error readout */}
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
        {MODES.map((m, i) => (
          <button
            key={m.id}
            type="button"
            onClick={() => setModeId(m.id)}
            aria-pressed={m.id === modeId}
            className={[
              'rounded border px-2.5 py-1.5 text-left transition-colors focus-visible:ring-2 focus-visible:ring-primary',
              m.id === modeId ? 'border-primary bg-primary/10' : 'border-border hover:bg-muted',
            ].join(' ')}
          >
            <div className={['text-[11px] font-medium', m.id === modeId ? 'text-primary' : 'text-foreground'].join(' ')}>
              {i + 1}. {copy.modes[m.id].title}
            </div>
            <div className="text-[10px] text-muted-foreground">
              {copy.modes[m.id].avgLabel}: <span className="font-mono font-semibold">{m.avgPct.toFixed(1)}%</span>
            </div>
          </button>
        ))}
      </div>

      {/* Four number lines, one per block of 4 values */}
      <svg
        viewBox={`0 0 ${VB_W} ${VB_H}`}
        className="w-full"
        role="img"
        aria-label={copy.svgAria(copy.modes[mode.id].title, copy.modes[mode.id].avgLabel, mode.avgPct.toFixed(1))}
      >
        {/* Legend */}
        <circle cx={PLOT_X + 5} cy={16} r={4} fill="none" style={{ stroke: 'var(--muted-foreground)' }} strokeWidth={1.2} />
        <text x={PLOT_X + 14} y={20} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
          {copy.legendOriginal}
        </text>
        <circle cx={PLOT_X + 105} cy={16} r={4} style={{ fill: 'var(--primary)' }} />
        <text x={PLOT_X + 114} y={20} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
          {copy.legendRoundTrip}
        </text>
        <path d={`M${PLOT_X + 248} 11V21`} style={{ stroke: 'var(--border)' }} strokeWidth={1.5} />
        <text x={PLOT_X + 255} y={20} style={{ fill: 'var(--muted-foreground)' }} className="text-[10px]">
          {copy.legendLevels}
        </text>

        {mode.lanes.map((lane, b) => {
          const laneTop = LANE_Y0 + b * LANE_STRIDE;
          const cy = laneTop + LANE_H / 2;
          return (
            <g key={lane.letter}>
              {/* Lane label + per-block scale */}
              <text x={16} y={cy - 2} style={{ fill: 'var(--foreground)' }} className="text-[11px] font-semibold">
                {copy.blockLabel(lane.letter)}
              </text>
              <text x={16} y={cy + 11} style={{ fill: 'var(--muted-foreground)' }} className="text-[9px]">
                {modeId === 'rotate' ? copy.laneRotated : copy.laneFourValues}
              </text>
              <text
                x={VB_W - 8}
                y={cy + 3}
                textAnchor="end"
                style={{ fill: lane.coarse ? 'var(--primary)' : 'var(--muted-foreground)' }}
                className="font-mono text-[9px]"
              >
                ×{lane.scale.toFixed(1)}
              </text>

              {/* Lane background + fp8 grid ticks + zero line */}
              <rect
                x={PLOT_X}
                y={laneTop}
                width={PLOT_W}
                height={LANE_H}
                rx={6}
                style={{ fill: 'var(--muted)', stroke: 'var(--border)' }}
                strokeWidth={1}
              />
              <path d={lane.ticks} style={{ stroke: 'var(--border)' }} strokeWidth={1} />
              <line
                x1={xOf(0, mode.axisMax)}
                y1={laneTop + 3}
                x2={xOf(0, mode.axisMax)}
                y2={laneTop + LANE_H - 3}
                style={{ stroke: 'var(--muted-foreground)', opacity: 0.45 }}
                strokeWidth={1}
              />

              {/* Dots: hollow = original position, filled = animates onto the grid */}
              {lane.dots.map((d, i) => {
                if (d.offScale) {
                  // The 52 outlier — far outside the ±1 zoom. It maps to 448 exactly.
                  const ax = PLOT_X + PLOT_W - 6;
                  return (
                    <g key={i}>
                      <text x={ax - 12} y={cy + 3} textAnchor="end" style={{ fill: 'var(--primary)' }} className="text-[10px] font-semibold">
                        {copy.outlierLabel}
                      </text>
                      <path d={`M${ax - 9} ${cy - 5}L${ax} ${cy}L${ax - 9} ${cy + 5}Z`} style={{ fill: 'var(--primary)' }} />
                    </g>
                  );
                }
                const ix = xOf(d.v, mode.axisMax);
                const qx = xOf(d.q, mode.axisMax);
                return (
                  <g key={i}>
                    <circle cx={ix} cy={cy} r={4.5} fill="none" style={{ stroke: 'var(--muted-foreground)', opacity: 0.55 }} strokeWidth={1.2} />
                    <circle
                      cx={ix}
                      cy={cy}
                      r={4}
                      style={{
                        fill: 'var(--primary)',
                        transform: `translateX(${showQuantized ? qx - ix : 0}px)`,
                        transition: reducedMotion ? 'none' : 'transform 650ms ease-in-out',
                      }}
                    >
                      <title>{copy.dotTitle(fmtVal(d.v), fmtVal(d.q), d.errPct.toFixed(1))}</title>
                    </circle>
                  </g>
                );
              })}
            </g>
          );
        })}

        {/* Axis labels */}
        {[-1, 0, 1].map((t) => {
          const v = t * (modeId === 'rotate' ? 14 : 1);
          return (
            <text key={t} x={xOf(v, mode.axisMax)} y={AXIS_Y + 10} textAnchor="middle" style={{ fill: 'var(--muted-foreground)' }} className="font-mono text-[9px]">
              {v > 0 ? `+${v}` : v}
            </text>
          );
        })}
        <text x={PLOT_X + PLOT_W} y={AXIS_Y + 10} textAnchor="end" style={{ fill: 'var(--muted-foreground)' }} className="text-[9px]">
          {modeId === 'rotate' ? copy.axisRotate : copy.axisZoom}
        </text>
      </svg>

      {/* Caption — explains the active mode; all numbers computed, not typed */}
      <p className="min-h-[4rem] text-[13px] text-foreground/90">
        {modeId === 'tensor' ? copy.captionTensor : modeId === 'block' ? copy.captionBlock : copy.captionRotate}
      </p>

      <p className="text-[10px] text-muted-foreground/70">
        {copy.footnote}
      </p>
    </div>
  );
}

import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Sub-chapter 14.1 (Scaling laws) — a log-log scatter of training tokens vs.
 * parameters for real, published models, plus a diagonal reference line at
 * the ~20-tokens-per-parameter ratio implied by Chinchilla's own reported
 * numbers. Static SVG + React, no animation, no model, no WASM — SSR-safe.
 *
 * Data points (verified against the primary papers/sources; see the section
 * body for citations):
 *   - Gopher (2021):        280B params / 300B tokens   → ≈1.07 tok/param
 *   - Chinchilla (2022):     70B params / 1.4T tokens   → 20 tok/param (exact)
 *   - Llama 3 8B (2024):      8B params / ~15T tokens   → 1,875 tok/param
 *   - Llama 3 405B (2024): 405B params / 15.6T tokens   → ≈38.5 tok/param
 *   - This course's model (852,985,920 params) is NOT plotted with a real
 *     token count — no primary source discloses Qwen3.5-0.8B's pretraining
 *     corpus size. The fifth marker is a HOLLOW/DASHED point: what its token
 *     count would be IF it were trained exactly at the ~20:1 Chinchilla
 *     ratio (852,985,920 × 20 ≈ 17.06B tokens) — an illustrative
 *     extrapolation, not a claim about this checkpoint's real recipe.
 *
 * The reference line uses the 20:1 ratio derived from Chinchilla's own
 * 70B/1.4T headline numbers (1.4e12 / 70e9 = 20 exactly). That figure is not
 * a literal sentence in the paper — Hoffmann et al. 2022's actual finding is
 * a power-law fit (N and D both ∝ C^0.5), so the true compute-optimal ratio
 * drifts slowly with scale. 20:1 is the standard, widely-cited shorthand and
 * a fair reference across the range plotted here.
 */

const GOPHER_PARAMS = 280_000_000_000;
const GOPHER_TOKENS = 300_000_000_000;
const CHINCHILLA_PARAMS = 70_000_000_000;
const CHINCHILLA_TOKENS = 1_400_000_000_000;
const LLAMA8B_PARAMS = 8_000_000_000;
const LLAMA8B_TOKENS = 15_000_000_000_000;
const LLAMA405B_PARAMS = 405_000_000_000;
const LLAMA405B_TOKENS = 15_600_000_000_000;
const THIS_MODEL_PARAMS = 852_985_920;
const CHINCHILLA_OPTIMAL_RATIO = 20;
const THIS_MODEL_HYPOTHETICAL_TOKENS = THIS_MODEL_PARAMS * CHINCHILLA_OPTIMAL_RATIO; // ≈ 17.06B

const GOPHER_RATIO = (GOPHER_TOKENS / GOPHER_PARAMS).toFixed(2); // 1.07
const LLAMA8B_RATIO = LLAMA8B_TOKENS / LLAMA8B_PARAMS; // 1875 (exact)
const LLAMA8B_OVER_MULTIPLE = Math.round(LLAMA8B_RATIO / CHINCHILLA_OPTIMAL_RATIO); // ≈ 94
const LLAMA405B_RATIO = (LLAMA405B_TOKENS / LLAMA405B_PARAMS).toFixed(1); // 38.5

type PointKey = 'gopher' | 'chinchilla' | 'llama8b' | 'llama405b' | 'thisModel';

// Locale-independent geometry: real params/tokens plus a hand-tuned label
// offset (anchor + dx/dy for a two-line label) chosen so none of the five
// labels collide once placed on the fixed 8..12 / 9..14 decade log axes below.
const POINTS: {
  key: PointKey;
  params: number;
  tokens: number;
  hollow?: boolean;
  anchor: 'start' | 'end' | 'middle';
  dx: number;
  dy: number;
}[] = [
  { key: 'llama405b', params: LLAMA405B_PARAMS, tokens: LLAMA405B_TOKENS, anchor: 'end', dx: -10, dy: -16 },
  { key: 'llama8b', params: LLAMA8B_PARAMS, tokens: LLAMA8B_TOKENS, anchor: 'end', dx: -10, dy: -16 },
  { key: 'chinchilla', params: CHINCHILLA_PARAMS, tokens: CHINCHILLA_TOKENS, anchor: 'start', dx: 10, dy: -24 },
  { key: 'gopher', params: GOPHER_PARAMS, tokens: GOPHER_TOKENS, anchor: 'end', dx: -10, dy: 16 },
  {
    key: 'thisModel',
    params: THIS_MODEL_PARAMS,
    tokens: THIS_MODEL_HYPOTHETICAL_TOKENS,
    hollow: true,
    anchor: 'middle',
    dx: 0,
    dy: 16,
  },
];

const COPY = {
  en: {
    heading: 'Training tokens vs. parameters (log-log)',
    svgAria:
      'Log-log scatter of training tokens versus parameters for four published models — Gopher, Chinchilla, Llama 3 8B, Llama 3 405B — plus a diagonal reference line at 20 tokens per parameter, the Chinchilla-optimal ratio. A fifth, hollow marker shows what this course\'s 852,985,920-parameter model would need if trained at that same ratio — about 17.06 billion tokens — an illustrative, unverified extrapolation, not a real measurement.',
    xAxis: 'parameters (log scale)',
    yAxis: 'training tokens (log scale)',
    lineLabel: `≈${CHINCHILLA_OPTIMAL_RATIO} tokens/param — Chinchilla-optimal reference`,
    legendReal: 'published, measured training run',
    legendHollow: "this course's model, hypothetical — not a verified figure",
    labels: {
      gopher: { name: 'Gopher (2021)', note: `≈${GOPHER_RATIO} tok/param — under-trained` },
      chinchilla: { name: 'Chinchilla (2022)', note: `${CHINCHILLA_OPTIMAL_RATIO} tok/param — the reference itself` },
      llama8b: {
        name: 'Llama 3 8B (2024)',
        note: `≈${LLAMA8B_RATIO.toLocaleString('en-US')} tok/param — ≈${LLAMA8B_OVER_MULTIPLE}× over`,
      },
      llama405b: { name: 'Llama 3 405B (2024)', note: `≈${LLAMA405B_RATIO} tok/param — near its own optimum` },
      thisModel: { name: 'This model, IF Chinchilla-optimal', note: '≈17.06B tokens — hypothetical, unverified' },
    } satisfies Record<PointKey, { name: string; note: string }>,
    caption:
      'Chinchilla itself sits exactly on the reference line by construction. Gopher, matched to the same training compute but given far less data, falls well below it. Llama 3 8B sits far above it on purpose — see below. Llama 3 405B lands just above it, close to the ratio its own scaling-law fit actually predicted for that compute budget.',
  },
  zh: {
    heading: '训练 token 数 vs. 参数量（对数-对数）',
    svgAria:
      '训练 token 数与参数量的对数-对数散点图：四个已发布的模型——Gopher、Chinchilla、Llama 3 8B、Llama 3 405B——外加一条「每参数 20 个 token」（Chinchilla 最优比例）的对角参考线。第五个空心标记显示的是本课程这个 852,985,920 参数的模型若按同一比例训练需要多少 token——约 170.6 亿——这是一个示意性的、未经验证的外推,并非真实测量值。',
    xAxis: '参数量（对数刻度）',
    yAxis: '训练 token 数（对数刻度）',
    lineLabel: `≈每参数 ${CHINCHILLA_OPTIMAL_RATIO} 个 token —— Chinchilla 最优参考线`,
    legendReal: '已发布、实测的训练运行',
    legendHollow: '本课程模型，假设值——非已验证数字',
    labels: {
      gopher: { name: 'Gopher（2021）', note: `≈${GOPHER_RATIO} tok/param——训练不足` },
      chinchilla: { name: 'Chinchilla（2022）', note: `${CHINCHILLA_OPTIMAL_RATIO} tok/param——参考线本身` },
      llama8b: {
        name: 'Llama 3 8B（2024）',
        note: `≈${LLAMA8B_RATIO.toLocaleString('en-US')} tok/param——超出约 ${LLAMA8B_OVER_MULTIPLE}×`,
      },
      llama405b: { name: 'Llama 3 405B（2024）', note: `≈${LLAMA405B_RATIO} tok/param——接近自身最优点` },
      thisModel: { name: '这个模型，若按 Chinchilla 最优训练', note: '≈170.6 亿 token——假设值,未经验证' },
    } satisfies Record<PointKey, { name: string; note: string }>,
    caption:
      'Chinchilla 本身正好落在参考线上（因为线就是按它定义的）。Gopher 用了同样多的训练算力,却只拿到少得多的数据,明显落在线下方。Llama 3 8B 故意远远落在线上方——原因见下文。Llama 3 405B 刚好落在线上方一点,接近它自己的 scaling law 为那个算力预算实际预测出的比例。',
  },
} as const;

// ── geometry: a shared log-log axis, x in decades [8, 12] (1e8..1e12
// parameters), y in decades [9, 14] (1e9..1e14 tokens). ─────────────────────
const W = 600;
const H = 400;
const PAD_L = 52;
const PAD_R = 18;
const PAD_T = 18;
const PAD_B = 42;
const PLOT_W = W - PAD_L - PAD_R;
const PLOT_H = H - PAD_T - PAD_B;

const X_LOG_MIN = 8;
const X_LOG_MAX = 12;
const Y_LOG_MIN = 9;
const Y_LOG_MAX = 14;

function xFor(params: number): number {
  const l = Math.log10(Math.max(params, 1));
  return PAD_L + (PLOT_W * (l - X_LOG_MIN)) / (X_LOG_MAX - X_LOG_MIN);
}
function yFor(tokens: number): number {
  const l = Math.log10(Math.max(tokens, 1));
  return PAD_T + PLOT_H - (PLOT_H * (l - Y_LOG_MIN)) / (Y_LOG_MAX - Y_LOG_MIN);
}

// Reference line: tokens = 20 × params, drawn across the full x-domain.
const LINE_X1 = PAD_L;
const LINE_Y1 = yFor(Math.pow(10, X_LOG_MIN) * CHINCHILLA_OPTIMAL_RATIO);
const LINE_X2 = PAD_L + PLOT_W;
const LINE_Y2 = yFor(Math.pow(10, X_LOG_MAX) * CHINCHILLA_OPTIMAL_RATIO);

export function TokensPerParamScatter() {
  const copy = COPY[useLocale()];

  const xDecades: number[] = [];
  for (let d = X_LOG_MIN; d <= X_LOG_MAX; d++) xDecades.push(d);
  const yDecades: number[] = [];
  for (let d = Y_LOG_MIN; d <= Y_LOG_MAX; d++) yDecades.push(d);

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.heading}</div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={copy.svgAria}>
        {/* gridlines */}
        {xDecades.map((d) => {
          const x = PAD_L + (PLOT_W * (d - X_LOG_MIN)) / (X_LOG_MAX - X_LOG_MIN);
          return (
            <g key={`x-${d}`}>
              <line
                x1={x}
                y1={PAD_T}
                x2={x}
                y2={PAD_T + PLOT_H}
                stroke="var(--border)"
                strokeWidth={1}
                strokeDasharray="2 4"
              />
              <text x={x} y={H - PAD_B + 14} textAnchor="middle" className="font-mono" fontSize={9} fill="var(--muted-foreground)">
                10^{d}
              </text>
            </g>
          );
        })}
        {yDecades.map((d) => {
          const y = PAD_T + PLOT_H - (PLOT_H * (d - Y_LOG_MIN)) / (Y_LOG_MAX - Y_LOG_MIN);
          return (
            <g key={`y-${d}`}>
              <line
                x1={PAD_L}
                y1={y}
                x2={PAD_L + PLOT_W}
                y2={y}
                stroke="var(--border)"
                strokeWidth={1}
                strokeDasharray="2 4"
              />
              <text x={PAD_L - 6} y={y + 3} textAnchor="end" className="font-mono" fontSize={9} fill="var(--muted-foreground)">
                10^{d}
              </text>
            </g>
          );
        })}

        {/* axes baselines */}
        <line x1={PAD_L} y1={PAD_T + PLOT_H} x2={PAD_L + PLOT_W} y2={PAD_T + PLOT_H} stroke="var(--border)" strokeWidth={1.5} />
        <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + PLOT_H} stroke="var(--border)" strokeWidth={1.5} />
        <text x={PAD_L + PLOT_W / 2} y={H - 4} textAnchor="middle" fontSize={10} fill="var(--muted-foreground)">
          {copy.xAxis}
        </text>
        <text
          x={12}
          y={PAD_T + PLOT_H / 2}
          textAnchor="middle"
          fontSize={10}
          fill="var(--muted-foreground)"
          transform={`rotate(-90, 12, ${PAD_T + PLOT_H / 2})`}
        >
          {copy.yAxis}
        </text>

        {/* reference line: 20 tokens/param */}
        <line
          x1={LINE_X1}
          y1={LINE_Y1}
          x2={LINE_X2}
          y2={LINE_Y2}
          stroke="var(--primary)"
          strokeWidth={1.75}
          strokeDasharray="6 4"
          opacity={0.75}
        />
        <g>
          <rect x={252} y={162} width={148} height={16} rx={3} fill="var(--background)" opacity={0.85} />
          <text x={326} y={174} textAnchor="middle" fontSize={9.5} fill="var(--primary)" className="font-mono">
            {copy.lineLabel}
          </text>
        </g>

        {/* data points */}
        {POINTS.map((p) => {
          const cx = xFor(p.params);
          const cy = yFor(p.tokens);
          const label = copy.labels[p.key];
          const stroke = p.hollow ? 'var(--muted-foreground)' : 'var(--foreground)';
          const fill = p.hollow ? 'var(--card)' : 'var(--foreground)';
          return (
            <g key={p.key}>
              <circle
                cx={cx}
                cy={cy}
                r={5}
                fill={fill}
                stroke={stroke}
                strokeWidth={p.hollow ? 1.75 : 1}
                strokeDasharray={p.hollow ? '3 2' : undefined}
              />
              <text x={cx + p.dx} y={cy + p.dy} textAnchor={p.anchor} fontSize={10.5} fontWeight={600} fill="var(--foreground)">
                {label.name}
              </text>
              <text
                x={cx + p.dx}
                y={cy + p.dy + 12}
                textAnchor={p.anchor}
                fontSize={9.5}
                className="font-mono"
                fill="var(--muted-foreground)"
              >
                {label.note}
              </text>
            </g>
          );
        })}
      </svg>

      <div className="flex flex-wrap items-center gap-4 text-[11px] text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: 'var(--foreground)' }} />
          {copy.legendReal}
        </span>
        <span className="flex items-center gap-1.5">
          <span
            aria-hidden
            className="inline-block h-2.5 w-2.5 rounded-full border"
            style={{ borderColor: 'var(--muted-foreground)', borderStyle: 'dashed', background: 'var(--card)' }}
          />
          {copy.legendHollow}
        </span>
      </div>

      <p className="text-[12px] text-foreground/85">{copy.caption}</p>
    </div>
  );
}

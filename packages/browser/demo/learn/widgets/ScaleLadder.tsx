import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Chapter 14 (Scaling) opener — a 4-rung, log-scale parameter ladder that puts
 * the in-browser model on the same axis as a toy linear fit and two historical
 * GPTs. The whole point is honesty about size: our model is genuinely small
 * next to GPT-3. Each rung is its own evenly-spaced row whose bar LENGTH encodes
 * log10(params) on a shared 10^0–10^12 axis, so seven orders of magnitude fit in
 * one frame AND close-valued rungs (this model vs GPT-2 XL, ~¼ decade apart)
 * stay on separate, legible rows instead of overlapping.
 *
 * Rungs (exact counts; even rows, bar length = log10 params):
 *   • 2            — a straight line y = a·x + b (slope + intercept)
 *   • 852,985,920  — THIS model (matches the architecture chapter and ch1)
 *   • ~1.5B        — GPT-2 XL (2019)
 *   • 175B         — GPT-3 (2020), ≈ 205× this model
 *
 * Self-contained, static, SSR-safe (no window access at module top level),
 * locale-aware via the COPY pattern. NOT to be confused with ModelSizeLadder in
 * the architecture chapter — this component is intentionally named ScaleLadder.
 *
 * Honesty caveats baked into the copy:
 *   • The GPT-3 rung carries the hybrid caveat: a raw param-count comparison is
 *     fair, but GPT-3 is a DENSE transformer (every layer is softmax attention)
 *     while our model is hybrid — 6 full-attention + 18 GatedDeltaNet layers —
 *     so "more params" is NOT "more of the same attention".
 *   • The GPT-2 → GPT-3 jump is labelled as ONE historical fact (scale alone
 *     turned GPT-2's gibberish into GPT-3's coherence), never re-run here.
 */

type Rung = {
  /** exact parameter count, used for the log position */
  params: number;
  /** short display label for the count, e.g. "852,985,920" or "~1.5B" */
  count: string;
  /** name shown on the rung */
  name: string;
  /** one-line gloss under the name */
  note: string;
  /** highlight the "this model" rung */
  highlight?: boolean;
};

// Exact / canonical counts. 2 is the literal parameter count of a 1-D line
// (slope a + intercept b). 852,985,920 is this model. GPT-2 XL ≈ 1.5B, GPT-3
// ≈ 175B (≈ 205× this model: 175e9 / 852_985_920 ≈ 205).
const THIS_MODEL = 852_985_920;
const GPT3 = 175_000_000_000;
const GPT3_RATIO = Math.round(GPT3 / THIS_MODEL); // ≈ 205

const COPY = {
  en: {
    header: 'The parameter ladder (log scale)',
    svgAria:
      'Log-scale ladder of model sizes: a 2-parameter line, this 852,985,920-parameter model, GPT-2 XL near 1.5 billion, and GPT-3 near 175 billion parameters.',
    rungs: [
      { params: 2, count: '2', name: 'A straight line', note: 'y = a·x + b — one slope, one intercept' },
      {
        params: THIS_MODEL,
        count: '852,985,920',
        name: 'This model',
        note: 'the Qwen3.5-0.8B running in your browser',
        highlight: true,
      },
      { params: 1_500_000_000, count: '~1.5B', name: 'GPT-2 XL (2019)', note: 'the largest GPT-2' },
      {
        params: GPT3,
        count: '175B',
        name: 'GPT-3 (2020)',
        note: `dense transformer — every layer is softmax attention`,
      },
    ] as Rung[],
    axisLow: 'fewer parameters',
    axisHigh: 'more parameters',
    ratioBadge: `≈ ${GPT3_RATIO}× this model`,
    historical:
      'The jump from GPT-2 to GPT-3 was almost pure scale: the same recipe, ~100× the parameters, and the output went from broken-but-grammatical to startlingly coherent.',
    caveat: (
      <>
        Read this as <strong>raw parameter count only</strong>. GPT-3 is a <em>dense</em> transformer — every layer is
        softmax <ChapterLinkSlot slug="attention">attention</ChapterLinkSlot>. This model is{' '}
        <strong>hybrid</strong>: just 6 full-attention layers and 18 GatedDeltaNet (linear-recurrent) layers. So a
        bigger bar does not mean &ldquo;more of the same attention&rdquo; — it is a different shape of network, not a
        scaled-up copy.
      </>
    ),
    payoff: (
      <>
        Here is why this rung lives in <em>this</em> chapter: piling on parameters does <strong>not</strong>{' '}
        automatically overfit. Bigger models trained on enough text generalise <em>better</em>, not worse — which is
        exactly why the weight-decay and dropout knobs below are a live question, not an obvious win. You tune
        regularisation knowing the model is already huge and the data is already enormous.
      </>
    ),
  },
  zh: {
    header: '参数阶梯（对数刻度）',
    svgAria:
      '模型规模的对数刻度阶梯：一条 2 参数的直线、这个 852,985,920 参数的模型、约 15 亿参数的 GPT-2 XL，以及约 1750 亿参数的 GPT-3。',
    rungs: [
      { params: 2, count: '2', name: '一条直线', note: 'y = a·x + b——一个斜率，一个截距' },
      {
        params: THIS_MODEL,
        count: '852,985,920',
        name: '这个模型',
        note: '正在你浏览器里运行的 Qwen3.5-0.8B',
        highlight: true,
      },
      { params: 1_500_000_000, count: '~1.5B', name: 'GPT-2 XL（2019）', note: '最大的 GPT-2' },
      {
        params: GPT3,
        count: '175B',
        name: 'GPT-3（2020）',
        note: 'dense Transformer——每一层都是 softmax 注意力',
      },
    ] as Rung[],
    axisLow: '参数更少',
    axisHigh: '参数更多',
    ratioBadge: `≈ 这个模型的 ${GPT3_RATIO} 倍`,
    historical:
      '从 GPT-2 到 GPT-3 这一跳几乎是纯粹的扩展：同一套配方，参数约 100 倍，输出就从“语法通顺但说胡话”变成了出人意料的连贯。',
    caveat: (
      <>
        这里只能读作<strong>原始参数量的比较</strong>。GPT-3 是 <em>dense</em> Transformer——每一层都是 softmax{' '}
        <ChapterLinkSlot slug="attention">注意力</ChapterLinkSlot>。而这个模型是<strong>混合</strong>架构：只有 6
        层全注意力，外加 18 层 GatedDeltaNet（线性循环）层。所以条形更长并不意味着“同一种注意力更多”——它是一种不同形状的网络，而不是放大版的副本。
      </>
    ),
    payoff: (
      <>
        这根横档为什么放在<em>本章</em>：堆参数<strong>不会</strong>
        自动导致过拟合。在足够多文本上训练的更大模型，泛化得<em>更好</em>而非更差——这正是下面权重衰减和 dropout
        这些旋钮成为一个有讨论价值的问题（而不是显而易见的稳赢）的原因。你是在“模型本就巨大、数据本就海量”的前提下，去调正则化的。
      </>
    ),
  },
} as const;

// The chapter passes a real <ChapterLink> through via context so the widget
// stays self-contained (no scaffolding import) yet still links cross-chapter.
const LinkCtx = React.createContext<((slug: string, text: React.ReactNode) => React.ReactNode) | null>(null);

function ChapterLinkSlot({ slug, children }: { slug: string; children: React.ReactNode }) {
  const render = React.useContext(LinkCtx);
  if (render) return <>{render(slug, children)}</>;
  return <>{children}</>;
}

// --- geometry -------------------------------------------------------------
// Each rung is its own evenly-spaced ROW; the bar LENGTH (not its vertical
// position) encodes log10(params) on a shared 10^0…10^12 x-axis. Even rows are
// what keep close-valued rungs — this model (8.5e8) vs GPT-2 XL (1.5e9), only a
// quarter-decade apart — on separate, legible lines instead of colliding.
const W = 560;
const H = 300;
const PAD_L = 12;
const PAD_R = 16;
const PAD_T = 14;
const PAD_B = 30; // room for the x-axis decade labels
const PLOT_W = W - PAD_L - PAD_R;
const PLOT_H = H - PAD_T - PAD_B;

// Log axis bounds: 10^0 (a constant) up to 10^12 (a trillion params).
const LOG_MIN = 0;
const LOG_MAX = 12;

// x position for a given param count on the shared log axis.
function xFor(params: number): number {
  const l = Math.log10(Math.max(params, 1));
  return PAD_L + (PLOT_W * (l - LOG_MIN)) / (LOG_MAX - LOG_MIN);
}

function ScaleLadderSvg({ rungs, aria }: { rungs: Rung[]; aria: string }) {
  const decades: number[] = [];
  for (let d = LOG_MIN; d <= LOG_MAX; d += 2) decades.push(d);

  const n = rungs.length;
  const rowH = PLOT_H / n;
  const barH = Math.min(20, rowH * 0.34);
  // smallest rung at the bottom, largest at the top — a ladder you climb.
  const rowCenterY = (i: number) => PAD_T + PLOT_H - (i + 0.5) * rowH;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={aria}>
      {/* vertical decade gridlines + x-axis labels */}
      {decades.map((d) => {
        const x = PAD_L + (PLOT_W * d) / LOG_MAX;
        return (
          <g key={d}>
            <line x1={x} y1={PAD_T} x2={x} y2={PAD_T + PLOT_H} stroke="var(--border)" strokeWidth={1} strokeDasharray="2 4" />
            <text x={x} y={H - 8} textAnchor="middle" className="font-mono" fontSize={9} fill="var(--muted-foreground)">
              10^{d}
            </text>
          </g>
        );
      })}

      {/* x-axis baseline */}
      <line
        x1={PAD_L}
        y1={PAD_T + PLOT_H}
        x2={W - PAD_R}
        y2={PAD_T + PLOT_H}
        stroke="var(--border)"
        strokeWidth={1.5}
      />

      {rungs.map((r, i) => {
        const cy = rowCenterY(i);
        const w = Math.max(xFor(r.params) - PAD_L, 4);
        const fill = r.highlight ? 'var(--primary)' : 'var(--muted-foreground)';
        const op = r.highlight ? 0.85 : 0.3;
        return (
          <g key={i}>
            {/* label above the bar, left-aligned — never clips the frame, never
                collides with adjacent rows */}
            <text y={cy - barH / 2 - 5} fontSize={11.5}>
              <tspan x={PAD_L + 1} fontWeight={r.highlight ? 700 : 600} fill="var(--foreground)">
                {r.name}
              </tspan>
              <tspan dx={7} className="font-mono" fontSize={10} fill="var(--muted-foreground)">
                {r.count}
              </tspan>
            </text>
            {/* the bar — its length encodes log10(params) on the shared axis */}
            <rect
              x={PAD_L}
              y={cy - barH / 2}
              width={w}
              height={barH}
              rx={3}
              fill={fill}
              fillOpacity={op}
              stroke={r.highlight ? 'var(--primary)' : 'var(--border)'}
              strokeWidth={r.highlight ? 1.5 : 1}
            />
          </g>
        );
      })}
    </svg>
  );
}

export function ScaleLadder({
  renderLink,
}: {
  /** Optional cross-chapter link renderer supplied by the chapter (keeps the widget self-contained). */
  renderLink?: (slug: string, text: React.ReactNode) => React.ReactNode;
}) {
  const copy = COPY[useLocale()];
  return (
    <LinkCtx.Provider value={renderLink ?? null}>
      <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
          <span className="rounded-full border border-primary/40 bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
            {copy.ratioBadge}
          </span>
        </div>

        <ScaleLadderSvg rungs={copy.rungs} aria={copy.svgAria} />

        <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-muted-foreground">
          <span>{copy.axisLow}</span>
          <span>{copy.axisHigh}</span>
        </div>

        <p className="text-[12px] text-foreground/85">{copy.historical}</p>

        <div className="rounded-md border border-border bg-muted/20 p-2 text-[12px] text-foreground/85">
          {copy.caveat}
        </div>

        <p className="text-[12px] text-foreground/85">{copy.payoff}</p>
      </div>
    </LinkCtx.Provider>
  );
}

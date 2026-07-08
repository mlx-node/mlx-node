import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Chapter 14 (architecture) supplement — the model-size ladder.
 *
 * Static (no animation, no worker, no model): a compact horizontal log-scale
 * axis that situates the in-browser model's exact parameter count
 * (852,985,920) among the rough scales of language models in the wild — from
 * the tiniest on-device models up to frontier systems.
 *
 * Sibling to ParamBudgetDiagram: same surface, same COPY/useLocale pattern,
 * same design tokens. The only hard number is OUR model's; every other rung is
 * labelled as a GENERIC order of magnitude. In particular the top rung is
 * "frontier-scale, ~100–1000× larger" on purpose — printing a specific vendor
 * figure (e.g. a "175B") would be wrong to read as our peer, goes stale fast,
 * and the durable, honest framing is the ~100× gap itself. That gap is what
 * motivates the limitations arc that follows.
 */

type RungId = 'edge' | 'small' | 'ours' | 'frontier';

type Rung = {
  id: RungId;
  /** Log-scale anchor in parameters (used only to place the tick, not shown). */
  anchor: number;
  /** True for the one rung that is OUR exact model. */
  ours?: boolean;
};

// Anchors are rough order-of-magnitude placements on the log axis. Only the
// "ours" rung carries an exact, model-true count; the rest are generic decades.
const RUNGS: Rung[] = [
  { id: 'edge', anchor: 30_000_000 }, // ~10^7.5 — tiniest on-device models
  { id: 'small', anchor: 3_000_000_000 }, // ~10^9.5 — small open models, a few billion
  { id: 'ours', anchor: 852_985_920, ours: true }, // exact — the model this course runs
  { id: 'frontier', anchor: 200_000_000_000 }, // ~10^11.3 — frontier-scale (generic; no vendor figure)
];

const OURS = 852_985_920;

// Log axis bounds: a little padding below the smallest decade and above the
// largest, so no rung sits flush against the edge.
const LOG_MIN = Math.log10(8_000_000); // ~8M
const LOG_MAX = Math.log10(1_000_000_000_000); // 1T
const logPos = (n: number) => (Math.log10(n) - LOG_MIN) / (LOG_MAX - LOG_MIN);

const COPY = {
  en: {
    header: <>The size ladder — where this 0.85B model sits</>,
    readout: 'log scale · parameters',
    svgAria:
      "A horizontal log scale of language-model sizes in parameters. From left to right: tiny on-device models near ten million, this course's model at 852,985,920 parameters (~0.85 billion), small open models in the low billions, and frontier-scale systems roughly 100 to 1000 times larger than this one. Exact figures are given only for the model this course runs; the rest are generic orders of magnitude.",
    axisLo: '10M',
    axisMid: '1B',
    axisHi: '1T',
    rungs: {
      edge: { label: 'tiny on-device', sub: '~tens of millions' },
      small: { label: 'small open models', sub: 'a few billion' },
      ours: { label: 'this course', sub: '852,985,920 ≈ 0.85B' },
      frontier: { label: 'frontier-scale', sub: '~100–1000× larger' },
    } as Record<RungId, { label: string; sub: string }>,
    footnote: (
      <>
        Only one number here is exact: the {OURS.toLocaleString()} parameters of the model running in your browser. The
        other rungs are rough orders of magnitude — the log axis means each tick to the right is a roughly tenfold jump.
        The honest takeaway is the gap, not a headline figure: the largest systems are on the order of a hundred to a
        thousand times bigger than this one. That distance is worth keeping in mind as we turn, next, to what even a
        well-built model of this size genuinely cannot do.
      </>
    ),
  },
  zh: {
    header: <>规模阶梯——这个 0.85B 模型站在哪里</>,
    readout: 'log 刻度 · 参数量',
    svgAria:
      '一条按参数量排列的语言模型规模对数刻度。从左到右：约一千万参数的微型端上模型、本课程使用的 852,985,920 参数模型（约 8.5 亿）、几十亿参数级的小型开源模型，以及比本模型大约大 100 到 1000 倍的前沿级系统。只有本课程运行的模型给出了精确数字，其余都是大致的数量级。',
    axisLo: '10M',
    axisMid: '1B',
    axisHi: '1T',
    rungs: {
      edge: { label: '微型端上模型', sub: '约数千万' },
      small: { label: '小型开源模型', sub: '几十亿' },
      ours: { label: '本课程', sub: '852,985,920 ≈ 0.85B' },
      frontier: { label: '前沿级', sub: '约大 100–1000 倍' },
    } as Record<RungId, { label: string; sub: string }>,
    footnote: (
      <>
        这里只有一个数字是精确的：在你浏览器里运行的这个模型，共 {OURS.toLocaleString()} 个参数。其余各档都是大致的
        数量级——对数刻度意味着每往右一格，参数量大约翻十倍。诚实的结论是那段差距，而不是某个标题数字：最大的系统大约
        是本模型的一百到一千倍。带着这段距离感，我们接下来去看：即便是一个做得很好、这个规模的模型，到底有哪些事是它
        本质上做不到的。
      </>
    ),
  },
} as const;

// ── SVG geometry ─────────────────────────────────────────────────────────────
const VB_W = 760;
const AXIS_X = 12;
const AXIS_W = VB_W - 24;
const AXIS_Y = 78; // baseline of the axis line
const VB_H = 112;

export function ModelSizeLadder() {
  const copy = COPY[useLocale()];

  const placed = RUNGS.map((r) => ({ ...r, cx: AXIS_X + logPos(r.anchor) * AXIS_W }));
  // Decade gridlines at 10M, 100M, 1B, 10B, 100B, 1T for a quiet log backdrop.
  const decades = [1e7, 1e8, 1e9, 1e10, 1e11, 1e12];

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
        <div className="font-mono text-[11px] text-muted-foreground">{copy.readout}</div>
      </div>

      <svg viewBox={`0 0 ${VB_W} ${VB_H}`} className="w-full" role="img" aria-label={copy.svgAria}>
        {/* decade gridlines */}
        {decades.map((d) => {
          const x = AXIS_X + logPos(d) * AXIS_W;
          return (
            <line
              key={`grid-${d}`}
              x1={x}
              x2={x}
              y1={AXIS_Y - 40}
              y2={AXIS_Y}
              style={{ stroke: 'var(--border)' }}
              strokeWidth={1}
              strokeDasharray="2 4"
            />
          );
        })}

        {/* axis baseline */}
        <line
          x1={AXIS_X}
          x2={AXIS_X + AXIS_W}
          y1={AXIS_Y}
          y2={AXIS_Y}
          style={{ stroke: 'var(--muted-foreground)' }}
          strokeWidth={1.5}
        />

        {/* axis end + mid labels */}
        <text x={AXIS_X} y={AXIS_Y + 18} style={{ fill: 'var(--muted-foreground)' }} className="font-mono text-[10px]">
          {copy.axisLo}
        </text>
        <text
          x={AXIS_X + logPos(1e9) * AXIS_W}
          y={AXIS_Y + 18}
          textAnchor="middle"
          style={{ fill: 'var(--muted-foreground)' }}
          className="font-mono text-[10px]"
        >
          {copy.axisMid}
        </text>
        <text
          x={AXIS_X + AXIS_W}
          y={AXIS_Y + 18}
          textAnchor="end"
          style={{ fill: 'var(--muted-foreground)' }}
          className="font-mono text-[10px]"
        >
          {copy.axisHi}
        </text>

        {/* rungs: a tick on the axis + a stacked label above it */}
        {placed.map((r) => {
          const c = copy.rungs[r.id];
          // Keep the leftmost and rightmost labels inside the viewBox.
          const anchor = r.cx < 70 ? 'start' : r.cx > VB_W - 70 ? 'end' : 'middle';
          return (
            <g key={r.id}>
              <circle
                cx={r.cx}
                cy={AXIS_Y}
                r={r.ours ? 5 : 3.5}
                style={{
                  fill: r.ours ? 'var(--primary)' : 'var(--muted-foreground)',
                  stroke: 'var(--background)',
                }}
                strokeWidth={1.5}
              />
              <text
                x={r.cx}
                y={r.ours ? 22 : 34}
                textAnchor={anchor}
                style={{ fill: r.ours ? 'var(--foreground)' : 'var(--muted-foreground)' }}
                className={r.ours ? 'text-[12px] font-semibold' : 'text-[11px]'}
              >
                {c.label}
              </text>
              <text
                x={r.cx}
                y={r.ours ? 37 : 48}
                textAnchor={anchor}
                style={{ fill: 'var(--muted-foreground)' }}
                className="font-mono text-[10px]"
              >
                {c.sub}
              </text>
            </g>
          );
        })}
      </svg>

      <p className="text-[10px] text-muted-foreground/70">{copy.footnote}</p>
    </div>
  );
}

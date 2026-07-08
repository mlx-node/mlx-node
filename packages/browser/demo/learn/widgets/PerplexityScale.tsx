import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Chapter 16 "Evaluation" sub-chapter — perplexity is exp(cross-entropy loss),
 * so it lives on the same axis as every loss number the course has already
 * shown, just exponentiated. This is a log-scale ladder (same idiom as
 * ScaleLadder's parameter ladder) with one row per anchor point, ordered from
 * "total confusion" down to "confident, single-digit choice":
 *
 *   1. Uniform-random baseline — THIS course's own model, at initialization,
 *      over its own 248,320-token vocabulary. Perplexity of a uniform
 *      distribution over V outcomes is exactly V (cross-entropy = ln V, so
 *      PPL = exp(ln V) = V) — a mechanical consequence of the same ln(248,320)
 *      ≈ 12.4-nat number chapter 13 already establishes at init, not an
 *      independently published figure.
 *   2. GPT-3 175B, Penn Treebank, zero-shot: perplexity 20.50 (prior SOTA
 *      35.8) — Brown et al. 2020, "Language Models are Few-Shot Learners",
 *      arXiv:2005.14165, Table 3.2.
 *   3. GPT-2 (762M, second-largest size — 1.5B is the largest), WikiText-2 test
 *      perplexity 19.93 — Radford et al. 2019; reproduced by the HF
 *      Transformers docs as 19.44 (matching stride) / 16.44 (overlapping
 *      sliding window, stride 512).
 *   4. GPT-3 175B, LAMBADA, few-shot: perplexity 1.92 (prior SOTA 8.63) — same
 *      GPT-3 paper, Table 3.2.
 *
 * All four numbers are primary-sourced (see file header comments above); none
 * are invented. Static, SSR-safe: no worker, no model, no WASM, no animation.
 */

const VOCAB_SIZE = 248_320; // Qwen3.5-0.8B vocab_size — matches chapters 1/2/13
const UNIFORM_PPL = VOCAB_SIZE; // PPL of a uniform distribution over V outcomes = V

type Row = {
  ppl: number;
  highlight?: boolean;
};

// Exact / cited values — see file header for sourcing.
const ROWS: Row[] = [
  { ppl: UNIFORM_PPL, highlight: true },
  { ppl: 20.5 },
  { ppl: 19.93 },
  { ppl: 1.92 },
];

const COPY = {
  en: {
    header: 'Perplexity compresses as loss drops (log scale)',
    svgAria:
      'Log-scale ladder of perplexity values: a uniform-random baseline at 248,320 (this model at initialization), GPT-3 on Penn Treebank at 20.50, GPT-2 on WikiText-2 at 19.93, and GPT-3 on LAMBADA (few-shot) at 1.92.',
    rows: [
      {
        name: 'Uniform-random baseline',
        note: 'this model, at initialization — ln(248,320) ≈ 12.4 nats, exp(12.4) ≈ 248,320',
      },
      {
        name: 'GPT-3 175B · Penn Treebank (zero-shot)',
        note: 'prior SOTA was 35.8 — Brown et al. 2020',
      },
      {
        name: 'GPT-2 762M · WikiText-2 (test)',
        note: 'the second-largest GPT-2 checkpoint — Radford et al. 2019',
      },
      {
        name: 'GPT-3 175B · LAMBADA (few-shot)',
        note: 'prior SOTA was 8.63 — Brown et al. 2020',
      },
    ],
    axisLow: 'confident (low perplexity)',
    axisHigh: 'confused (high perplexity)',
    footnote:
      'Perplexity = exp(cross-entropy loss). A uniform guess over V outcomes always has perplexity exactly V — so this model’s own ln(248,320) ≈ 12.4-nat init loss from Chapter 13 IS this top rung, just exponentiated. Every nat the loss drops during training is this bar shrinking.',
  },
  zh: {
    header: '随着 loss 下降，perplexity 也在压缩（对数刻度）',
    svgAria:
      '对数刻度的 perplexity 阶梯：均匀随机基线在 248,320（这个模型在初始化时），GPT-3 在 Penn Treebank 上为 20.50，GPT-2 在 WikiText-2 上为 19.93，GPT-3 在 LAMBADA（few-shot）上为 1.92。',
    rows: [
      {
        name: '均匀随机基线',
        note: '这个模型，在初始化时——ln(248,320) ≈ 12.4 nats，exp(12.4) ≈ 248,320',
      },
      {
        name: 'GPT-3 175B · Penn Treebank（zero-shot）',
        note: '此前 SOTA 是 35.8——Brown 等人，2020',
      },
      {
        name: 'GPT-2 762M · WikiText-2（test）',
        note: '第二大的 GPT-2 checkpoint——Radford 等人，2019',
      },
      {
        name: 'GPT-3 175B · LAMBADA（few-shot）',
        note: '此前 SOTA 是 8.63——Brown 等人，2020',
      },
    ],
    axisLow: '自信（低 perplexity）',
    axisHigh: '困惑（高 perplexity）',
    footnote:
      'Perplexity = exp(cross-entropy loss)。对 V 个结果的均匀猜测，perplexity 恒等于 V——所以这个模型在第 13 章里那个 ln(248,320) ≈ 12.4 nat 的初始 loss，取个指数，就正是最上面这一档。训练中 loss 每降一个 nat，这根条就跟着缩短一截。',
  },
} as const;

// --- geometry: a 4-rung log-scale ladder, mirroring ScaleLadder's shape ----
const W = 560;
const H = 260;
const PAD_L = 12;
const PAD_R = 16;
const PAD_T = 14;
const PAD_B = 30;
const PLOT_W = W - PAD_L - PAD_R;
const PLOT_H = H - PAD_T - PAD_B;

// log10(1) = 0 .. log10(1,000,000) = 6, giving headroom above the 248,320 top rung.
const LOG_MIN = 0;
const LOG_MAX = 6;

function xFor(ppl: number): number {
  const l = Math.log10(Math.max(ppl, 1));
  return PAD_L + (PLOT_W * (l - LOG_MIN)) / (LOG_MAX - LOG_MIN);
}

function formatPpl(ppl: number): string {
  return ppl >= 1000 ? ppl.toLocaleString('en-US') : ppl.toFixed(2);
}

export function PerplexityScale() {
  const copy = COPY[useLocale()];
  const decades: number[] = [];
  for (let d = LOG_MIN; d <= LOG_MAX; d += 2) decades.push(d);

  const n = ROWS.length;
  const rowH = PLOT_H / n;
  const barH = Math.min(20, rowH * 0.34);
  // biggest (most confused) rung at the top, smallest (most confident) at the bottom.
  const rowCenterY = (i: number) => PAD_T + (i + 0.5) * rowH;

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>

      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={copy.svgAria}>
        {decades.map((d) => {
          const x = PAD_L + (PLOT_W * d) / LOG_MAX;
          return (
            <g key={d}>
              <line
                x1={x}
                y1={PAD_T}
                x2={x}
                y2={PAD_T + PLOT_H}
                stroke="var(--border)"
                strokeWidth={1}
                strokeDasharray="2 4"
              />
              <text x={x} y={H - 8} textAnchor="middle" className="font-mono" fontSize={9} fill="var(--muted-foreground)">
                10^{d}
              </text>
            </g>
          );
        })}

        <line
          x1={PAD_L}
          y1={PAD_T + PLOT_H}
          x2={W - PAD_R}
          y2={PAD_T + PLOT_H}
          stroke="var(--border)"
          strokeWidth={1.5}
        />

        {ROWS.map((r, i) => {
          const cy = rowCenterY(i);
          const w = Math.max(xFor(r.ppl) - PAD_L, 4);
          const rowCopy = copy.rows[i]!;
          const fill = r.highlight ? 'var(--primary)' : 'var(--muted-foreground)';
          const op = r.highlight ? 0.85 : 0.3;
          return (
            <g key={i}>
              <text y={cy - barH / 2 - 5} fontSize={11.5}>
                <tspan x={PAD_L + 1} fontWeight={r.highlight ? 700 : 600} fill="var(--foreground)">
                  {rowCopy.name}
                </tspan>
                <tspan dx={7} className="font-mono" fontSize={10} fill="var(--muted-foreground)">
                  PPL {formatPpl(r.ppl)}
                </tspan>
              </text>
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

      <div className="flex items-center justify-between text-[10px] uppercase tracking-wide text-muted-foreground">
        <span>{copy.axisLow}</span>
        <span>{copy.axisHigh}</span>
      </div>

      <p className="text-[12px] text-foreground/85">{copy.footnote}</p>
    </div>
  );
}

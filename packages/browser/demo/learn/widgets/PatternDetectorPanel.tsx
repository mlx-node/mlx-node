import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Three illustrative 6x6 attention heatmaps showing common archetypes of
 * what a head can learn to do. No model call — these are hand-built CSS
 * grids meant to convey the *shape* of each pattern rather than measured
 * data from any particular layer/head.
 */

const SIZE = 6;
const TOKEN_LABELS = ['t0', 't1', 't2', 't3', 't4', 't5'];

// Helper to construct a square zero matrix.
function zeroes(n: number): number[][] {
  return Array.from({ length: n }, () => Array(n).fill(0));
}

// Diagonal (positional) head — each row mostly attends to itself, with a
// small leak to immediate neighbours. Row 0 has no causal history so it
// fully self-attends.
function buildPositional(): number[][] {
  const m = zeroes(SIZE);
  for (let i = 0; i < SIZE; i++) {
    if (i === 0) {
      m[0]![0] = 1.0;
      continue;
    }
    m[i]![i] = 0.7;
    m[i]![i - 1] = 0.2;
    if (i - 2 >= 0) m[i]![i - 2] = 0.1;
  }
  return m;
}

// Recency head — heavy weight on the immediately preceding token. Row 0 has
// nothing to look back at, so it self-attends.
function buildRecency(): number[][] {
  const m = zeroes(SIZE);
  for (let i = 0; i < SIZE; i++) {
    if (i === 0) {
      m[0]![0] = 1.0;
      continue;
    }
    m[i]![i - 1] = 0.7;
    m[i]![i] = 0.2;
    if (i - 2 >= 0) m[i]![i - 2] = 0.1;
  }
  return m;
}

// Syntactic head — late tokens look back at the determiner at position 0.
function buildSyntactic(): number[][] {
  const m = zeroes(SIZE);
  for (let i = 0; i < SIZE; i++) {
    if (i === 0) {
      m[0]![0] = 1.0;
      continue;
    }
    m[i]![0] = 0.6;
    m[i]![i] = 0.25;
    if (i - 1 >= 0) m[i]![i - 1] = 0.15;
  }
  return m;
}

// weights[i][j] = attention from row i (query) to col j (key); 0..1. The
// per-pattern label/caption prose lives in COPY below, index-aligned.
const PATTERN_WEIGHTS: number[][][] = [buildPositional(), buildRecency(), buildSyntactic()];

// Per-locale copy: every user-visible string lives here so /zh localizes while
// the English output stays byte-identical.
const COPY = {
  en: {
    header: 'What heads learn · three archetypes',
    intro: (
      <>
        Different heads learn to detect different things. You can&apos;t tell what a head detects from its weights alone
        — you have to see the patterns it produces on real inputs. Here are three common archetypes a typical mid-size
        LLM contains, drawn as illustrative heatmaps.
      </>
    ),
    heatmapAria: '6 by 6 attention heatmap',
    patterns: [
      {
        label: 'Positional head',
        caption: 'Detector: position. Each token mostly attends to itself; minor leak to neighbours.',
      },
      {
        label: 'Recency head',
        caption: 'Detector: recency. Strong attention to the immediately-prior token.',
      },
      {
        label: 'Syntactic head',
        caption: 'Detector: syntactic head. Late tokens look back at the determiner ("t0").',
      },
    ],
    footnote:
      'These are hand-built diagrams, not measured from any specific model. Real heads are messier and often mix multiple of these archetypes.',
  },
  zh: {
    header: '头都学会了什么 · 三种原型',
    intro: (
      <>
        不同的头学会检测不同的东西。光看权重没法判断一个头在检测什么——必须看它在真实输入上产生的模式。下面是一个典型中等规模
        LLM 里常见的三种原型，用示意性热力图画出来。
      </>
    ),
    heatmapAria: '6 × 6 注意力热力图',
    patterns: [
      {
        label: '位置头',
        caption: '检测器：位置。每个 token 主要关注自己；少量泄漏到相邻 token。',
      },
      {
        label: '近因头',
        caption: '检测器：近因。对紧邻的前一个 token 投以强注意力。',
      },
      {
        label: '句法头',
        caption: '检测器：句法。靠后的 token 回头看位置 0 的限定词（"t0"）。',
      },
    ],
    footnote: '这些是手工构造的示意图，并非从某个具体模型测得。真实的头更杂乱，常常混合多种原型。',
  },
} as const;

function cellColor(v: number): string {
  // 0 → near-transparent gray; 1 → saturated blue.
  const clamped = Math.max(0, Math.min(1, v));
  const alpha = clamped * 0.85 + (clamped > 0 ? 0.05 : 0);
  return `rgba(56, 189, 248, ${alpha.toFixed(3)})`;
}

function HeatGrid({ weights, ariaLabel }: { weights: number[][]; ariaLabel: string }) {
  return (
    <div
      role="img"
      aria-label={ariaLabel}
      className="grid gap-0.5"
      style={{ gridTemplateColumns: `auto repeat(${SIZE}, minmax(0, 1fr))` }}
    >
      <div />
      {TOKEN_LABELS.map((c) => (
        <div key={`col-${c}`} className="text-center font-mono text-[9px] text-muted-foreground">
          {c}
        </div>
      ))}
      {weights.map((row, i) => (
        <React.Fragment key={`row-${i}`}>
          <div className="pr-1 text-right font-mono text-[9px] text-muted-foreground self-center">
            {TOKEN_LABELS[i]}
          </div>
          {row.map((v, j) => (
            <div
              key={`cell-${i}-${j}`}
              className="aspect-square rounded-[2px] border border-border/30"
              style={{ backgroundColor: cellColor(v) }}
              title={`(${i},${j}) = ${v.toFixed(2)}`}
            />
          ))}
        </React.Fragment>
      ))}
    </div>
  );
}

export function PatternDetectorPanel() {
  const copy = COPY[useLocale()];
  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
      </div>
      <p className="text-[12px] text-foreground/85">{copy.intro}</p>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        {PATTERN_WEIGHTS.map((weights, i) => (
          <div key={`pattern-${i}`} className="space-y-2 rounded-md border border-border/60 bg-muted/20 p-3">
            <div className="text-[12px] font-semibold text-foreground">{copy.patterns[i]!.label}</div>
            <HeatGrid weights={weights} ariaLabel={copy.heatmapAria} />
            <p className="text-[11px] text-muted-foreground">{copy.patterns[i]!.caption}</p>
          </div>
        ))}
      </div>
      <p className="text-[11px] text-muted-foreground">{copy.footnote}</p>
    </div>
  );
}

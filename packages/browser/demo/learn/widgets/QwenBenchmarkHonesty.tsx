import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Chapter 16 "Evaluation" sub-chapter — the honesty closer. Directly fetched
 * from the official Hugging Face model card (huggingface.co/Qwen/Qwen3.5-0.8B):
 * it publishes a rich non-thinking/thinking score table for MMLU-Pro,
 * MMLU-Redux, C-Eval, SuperGPQA, GPQA (thinking only), and IFEval — but it does
 * NOT publish GSM8K, HumanEval, or a classic (non-Pro/non-Redux) MMLU number
 * for this exact checkpoint. Targeted searches turned up no official number for
 * this checkpoint on those three benchmarks either.
 *
 * The published rows render as two bars (non-thinking / thinking, 0-100 scale —
 * every published metric here happens to be percentage-like); the unpublished
 * rows render as an empty, dashed, explicitly-labeled placeholder — the same
 * "shared/borrowed, not real" dashed idiom MtpModuleDiagram uses for borrowed
 * weights, repurposed here for "not measured, don't invent it".
 *
 * Static, SSR-safe: no worker, no model, no WASM, no animation.
 */

type PublishedRow = {
  kind: 'published';
  name: string;
  nonThinking: number | null; // null = not reported for this mode (e.g. GPQA)
  thinking: number;
};

type UnpublishedRow = {
  kind: 'unpublished';
  name: string;
};

type Row = PublishedRow | UnpublishedRow;

// Every number below is read directly off the official Qwen3.5-0.8B model
// card. Scale is 0-100 for all of them.
const ROWS: Row[] = [
  { kind: 'published', name: 'MMLU-Pro', nonThinking: 29.7, thinking: 42.3 },
  { kind: 'published', name: 'MMLU-Redux', nonThinking: 48.5, thinking: 59.5 },
  { kind: 'published', name: 'C-Eval', nonThinking: 46.4, thinking: 50.5 },
  { kind: 'published', name: 'SuperGPQA', nonThinking: 16.9, thinking: 21.3 },
  { kind: 'published', name: 'GPQA', nonThinking: null, thinking: 11.9 },
  { kind: 'published', name: 'IFEval', nonThinking: 52.1, thinking: 44.0 },
  { kind: 'unpublished', name: 'GSM8K' },
  { kind: 'unpublished', name: 'HumanEval' },
  { kind: 'unpublished', name: 'MMLU (classic)' },
];

const MAX_SCORE = 100;

const COPY = {
  en: {
    header: "What Qwen3.5-0.8B's own model card actually reports",
    sourceLine: 'Source: huggingface.co/Qwen/Qwen3.5-0.8B — read directly, nothing extrapolated.',
    legendNonThinking: 'non-thinking',
    legendThinking: 'thinking',
    notReported: 'not reported for this mode',
    notPublished: 'not published for this checkpoint',
    barAria: (name: string, mode: string, score: string) => `${name}, ${mode}: ${score}`,
    footnote: (
      <>
        Read the rows literally, including the odd one: on <strong>IFEval</strong> the card reports{' '}
        <strong>non-thinking scoring higher</strong> than thinking (52.1 vs 44.0) — thinking mode is not a strict
        upgrade on every metric, and the card does not smooth that over. And three benchmarks you might expect —{' '}
        <strong>GSM8K</strong>, <strong>HumanEval</strong>, and classic <strong>MMLU</strong> — simply are not in the
        table for this exact checkpoint. Not every model gets evaluated on every benchmark; that gap is honest data,
        not a hidden bad score.
      </>
    ),
  },
  zh: {
    header: 'Qwen3.5-0.8B 官方 model card 实际报告了什么',
    sourceLine: '来源：huggingface.co/Qwen/Qwen3.5-0.8B——直接读取，没有任何外推。',
    legendNonThinking: 'non-thinking',
    legendThinking: 'thinking',
    notReported: '该模式下未报告',
    notPublished: '这个 checkpoint 未公开该项',
    barAria: (name: string, mode: string, score: string) => `${name}，${mode}：${score}`,
    footnote: (
      <>
        把每一行都按字面读，包括那个反常的一项：在 <strong>IFEval</strong> 上，卡片报告的是{' '}
        <strong>non-thinking 分数更高</strong>，而不是 thinking（52.1 对 44.0）——thinking 模式并不是在每一项指标上都稳赢，卡片也没有把这一点抹平。而
        <strong>GSM8K</strong>、<strong>HumanEval</strong>，以及经典 <strong>MMLU</strong> 这三个你可能会预期出现的
        benchmark，在这个具体 checkpoint 的表里根本就没有。不是每个模型都会在每个 benchmark
        上被评测；这个空白是诚实的数据缺失，不是一个被藏起来的差分数。
      </>
    ),
  },
} as const;

function Bar({ pct, color, opacity }: { pct: number; color: string; opacity: number }) {
  return (
    <div className="relative h-3 w-full overflow-hidden rounded-sm bg-muted/60">
      <div
        className="absolute inset-y-0 left-0"
        style={{ width: `${pct.toFixed(1)}%`, backgroundColor: color, opacity }}
      />
    </div>
  );
}

export function QwenBenchmarkHonesty() {
  const copy = COPY[useLocale()];

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
        <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
          <span className="flex items-center gap-1">
            <span
              aria-hidden
              className="inline-block h-2.5 w-4 rounded-sm"
              style={{ backgroundColor: 'var(--muted-foreground)', opacity: 0.55 }}
            />
            {copy.legendNonThinking}
          </span>
          <span className="flex items-center gap-1">
            <span
              aria-hidden
              className="inline-block h-2.5 w-4 rounded-sm"
              style={{ backgroundColor: 'var(--primary)', opacity: 0.85 }}
            />
            {copy.legendThinking}
          </span>
        </div>
      </div>

      <div className="space-y-2.5">
        {ROWS.map((r, i) => {
          if (r.kind === 'unpublished') {
            return (
              <div
                key={i}
                className="flex items-center justify-between rounded-sm border border-dashed px-2 py-1.5 text-[12px]"
                style={{ borderColor: 'var(--muted-foreground)', background: 'var(--muted)' }}
              >
                <span className="font-mono text-foreground/70">{r.name}</span>
                <span className="text-muted-foreground">{copy.notPublished}</span>
              </div>
            );
          }
          return (
            <div key={i} className="space-y-1">
              <div className="flex items-baseline justify-between text-[12px]">
                <span className="font-mono text-foreground/90">{r.name}</span>
              </div>
              <div
                role="img"
                aria-label={copy.barAria(
                  r.name,
                  copy.legendNonThinking,
                  r.nonThinking === null ? copy.notReported : r.nonThinking.toFixed(1),
                )}
                className="flex items-center gap-2"
              >
                <div className="flex-1">
                  {r.nonThinking === null ? (
                    <div className="flex h-3 items-center text-[10px] italic text-muted-foreground/70">
                      {copy.notReported}
                    </div>
                  ) : (
                    <Bar pct={(r.nonThinking / MAX_SCORE) * 100} color="var(--muted-foreground)" opacity={0.55} />
                  )}
                </div>
                <span className="w-9 shrink-0 text-right font-mono text-[11px] text-muted-foreground">
                  {r.nonThinking === null ? '—' : r.nonThinking.toFixed(1)}
                </span>
              </div>
              <div
                role="img"
                aria-label={copy.barAria(r.name, copy.legendThinking, r.thinking.toFixed(1))}
                className="flex items-center gap-2"
              >
                <div className="flex-1">
                  <Bar pct={(r.thinking / MAX_SCORE) * 100} color="var(--primary)" opacity={0.85} />
                </div>
                <span className="w-9 shrink-0 text-right font-mono text-[11px] text-foreground/90">
                  {r.thinking.toFixed(1)}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      <p className="text-[12px] text-foreground/85">{copy.footnote}</p>
      <p className="text-[11px] text-muted-foreground">{copy.sourceLine}</p>
    </div>
  );
}

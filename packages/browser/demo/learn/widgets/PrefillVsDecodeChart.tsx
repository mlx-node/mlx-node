import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Two-bar comparison showing the time per phase for a hypothetical 1024
 * prompt-token + 256 decode-token generation. Visually contrasts prefill
 * (one big parallel matmul) with decode (many small sequential matmuls).
 */

const PROMPT_TOKENS = 1024;
const DECODE_TOKENS = 256;
const PREFILL_MS = 700;
const DECODE_PER_TOKEN_MS = 200;
const DECODE_TOTAL_MS = DECODE_PER_TOKEN_MS * DECODE_TOKENS;
// Per-token rates: ~0.7 ms/token prefill vs 200 ms/token decode ≈ 293×.
const PREFILL_PER_TOKEN_MS = PREFILL_MS / PROMPT_TOKENS;
const PER_TOKEN_RATIO = Math.round(DECODE_PER_TOKEN_MS / PREFILL_PER_TOKEN_MS);

// Per-locale copy. COPY.en is the original English, moved verbatim.
const COPY = {
  en: {
    title: (prompt: number, decode: number) => `Prefill vs decode · ${prompt} prompt + ${decode} decode tokens`,
    subtitle: 'Illustrative ballpark numbers',
    prefillLabel: 'Prefill',
    prefillValue: (ms: string) => `~${ms} (one big matmul)`,
    prefillAria: (ms: string, pct: string) => `prefill bar: ${ms} (${pct}% of total)`,
    decodeLabel: 'Decode',
    decodeAria: (ms: string, pct: string, n: number) =>
      `decode bar: ${ms} (${pct}% of total) split into ${n} per-token matmuls`,
    summary: (perTokenMs: string, decodeMs: string, ratio: number) =>
      `Prefill is one parallel matmul; decode is N small sequential matmuls. That's why your first token comes fast and the rest stream. Per token: ~${perTokenMs} ms in prefill vs ~${decodeMs} in decode — roughly ${ratio}× apart.`,
    timeToFirstToken: 'Time to first token',
    totalWallTime: 'Total wall time',
  },
  zh: {
    title: (prompt: number, decode: number) => `Prefill vs 解码 · ${prompt} 个提示词 token + ${decode} 个解码 token`,
    subtitle: '示意性的量级数字',
    prefillLabel: 'Prefill',
    prefillValue: (ms: string) => `~${ms}（一次大矩阵乘法）`,
    prefillAria: (ms: string, pct: string) => `prefill 柱：${ms}（占总时长 ${pct}%）`,
    decodeLabel: '解码',
    decodeAria: (ms: string, pct: string, n: number) =>
      `解码柱：${ms}（占总时长 ${pct}%），拆成 ${n} 次逐 token 的矩阵乘法`,
    summary: (perTokenMs: string, decodeMs: string, ratio: number) =>
      `Prefill 是一次并行矩阵乘法；解码是 N 次小而串行的矩阵乘法。这就是为什么第一个 token 来得快、其余的以流式吐出。按每 token 计：prefill 约 ${perTokenMs} ms，解码约 ${decodeMs}——相差约 ${ratio}×。`,
    timeToFirstToken: '首个 token 耗时',
    totalWallTime: '总墙钟时间',
  },
} as const;

function formatMs(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(1)} s`;
  return `${ms.toFixed(0)} ms`;
}

export function PrefillVsDecodeChart() {
  const copy = COPY[useLocale()];
  const total = PREFILL_MS + DECODE_TOTAL_MS;
  const prefillPct = (PREFILL_MS / total) * 100;
  const decodePct = (DECODE_TOTAL_MS / total) * 100;

  // Render the decode phase as a stack of equal-width segments to visually
  // sell "N small matmuls". Cap the visible segment count so a 256-token
  // run doesn't produce 256 unreadable hairlines.
  const visibleSegments = Math.min(DECODE_TOKENS, 32);

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          {copy.title(PROMPT_TOKENS, DECODE_TOKENS)}
        </div>
        <div className="text-[11px] text-muted-foreground">{copy.subtitle}</div>
      </div>

      <div className="space-y-2">
        <div className="flex items-baseline justify-between text-[12px]">
          <span className="font-mono text-foreground/90">{copy.prefillLabel}</span>
          <span className="font-mono text-foreground/80">{copy.prefillValue(formatMs(PREFILL_MS))}</span>
        </div>
        <div
          className="relative h-8 w-full overflow-hidden rounded-sm bg-muted/60"
          role="img"
          aria-label={copy.prefillAria(formatMs(PREFILL_MS), prefillPct.toFixed(1))}
        >
          <div className="absolute inset-y-0 left-0 bg-sky-500" style={{ width: `${prefillPct.toFixed(2)}%` }} />
          <div className="absolute inset-0 flex items-center justify-end pr-2 font-mono text-[10px] text-foreground/70">
            {prefillPct.toFixed(1)}%
          </div>
        </div>

        <div className="flex items-baseline justify-between pt-2 text-[12px]">
          <span className="font-mono text-foreground/90">{copy.decodeLabel}</span>
          <span className="font-mono text-foreground/80">
            ~{formatMs(DECODE_PER_TOKEN_MS)} × {DECODE_TOKENS} = {formatMs(DECODE_TOTAL_MS)}
          </span>
        </div>
        <div
          className="relative h-8 w-full overflow-hidden rounded-sm bg-muted/60"
          role="img"
          aria-label={copy.decodeAria(formatMs(DECODE_TOTAL_MS), decodePct.toFixed(1), DECODE_TOKENS)}
        >
          <div className="absolute inset-y-0 left-0 flex" style={{ width: `${decodePct.toFixed(2)}%` }}>
            {Array.from({ length: visibleSegments }).map((_, i) => (
              <div
                key={`seg-${i}`}
                className="h-full flex-1 border-r border-background/70 bg-amber-500 last:border-r-0"
              />
            ))}
          </div>
          <div className="absolute inset-0 flex items-center justify-end pr-2 font-mono text-[10px] text-foreground/70">
            {decodePct.toFixed(1)}%
          </div>
        </div>
      </div>

      <p className="text-[12px] text-foreground/85">
        {copy.summary(PREFILL_PER_TOKEN_MS.toFixed(1), formatMs(DECODE_PER_TOKEN_MS), PER_TOKEN_RATIO)}
      </p>

      <div className="grid grid-cols-2 gap-2 text-[11px]">
        <div className="rounded-md border border-border/60 bg-muted/20 p-2">
          <div className="text-muted-foreground">{copy.timeToFirstToken}</div>
          <div className="mt-1 font-mono text-foreground/90">~{formatMs(PREFILL_MS + DECODE_PER_TOKEN_MS)}</div>
        </div>
        <div className="rounded-md border border-border/60 bg-muted/20 p-2">
          <div className="text-muted-foreground">{copy.totalWallTime}</div>
          <div className="mt-1 font-mono text-foreground/90">~{formatMs(total)}</div>
        </div>
      </div>
    </div>
  );
}

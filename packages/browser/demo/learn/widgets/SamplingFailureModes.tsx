import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Three side-by-side mini panels demonstrating common sampling failure modes
 * on the same prompt. Pre-recorded, illustrative continuations — no live
 * model run — so the teaching contrast is stable visit-to-visit.
 */

const PROMPT = 'Once upon a time, in a forest far away';

type Panel = {
  label: string;
  config: string;
  continuation: string;
  takeaway: string;
};

const COPY = {
  en: {
    header: 'Sampling failure modes · same prompt, three regimes',
    subheader: 'Pre-recorded continuations · 10 tokens each',
    promptLabel: 'Prompt: ',
    panels: [
      {
        label: 'Greedy',
        config: 'T = 0, top-p = 1.0',
        continuation: ' there lived a small forest there lived a small forest',
        takeaway: 'Repetition trap. The model finds a high-confidence phrase and loops back into it.',
      },
      {
        label: 'Too hot',
        config: 'T = 2.0',
        continuation: ' a frgg moo whirr the of bicycle banana very',
        takeaway: 'Gibberish. The distribution is so flat the sampler picks rare tokens uniformly.',
      },
      {
        label: 'Sweet spot',
        config: 'T = 0.7, top-p = 0.9',
        continuation: ' a small village where everyone knew each other and shared their stories',
        takeaway: 'Coherent yet varied. Top-p trims the absurd tail, temperature keeps it from collapsing.',
      },
    ] as readonly Panel[],
    outro: (
      <>
        Production LLM serving typically lands somewhere around{' '}
        <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">T = 0.7-1.0</code> with{' '}
        <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">top-p = 0.9</code> (or a moderate top-K).
        The two knobs do different jobs: temperature reshapes the whole distribution, top-p trims the long tail.
        Together they avoid the greedy loop and the hot-gibberish failure modes you see above.
      </>
    ),
  },
  zh: {
    header: '采样失败模式 · 同一提示词，三种配置',
    subheader: '预先录制的续写 · 每段 10 个 token',
    promptLabel: '提示词：',
    panels: [
      {
        label: '贪心',
        config: 'T = 0, top-p = 1.0',
        continuation: ' there lived a small forest there lived a small forest',
        takeaway: '重复陷阱。模型找到一个高置信短语，便不断绕回去重复它。',
      },
      {
        label: '过热',
        config: 'T = 2.0',
        continuation: ' a frgg moo whirr the of bicycle banana very',
        takeaway: '胡言乱语。分布平得离谱，采样器几乎是均匀地挑中罕见 token。',
      },
      {
        label: '恰到好处',
        config: 'T = 0.7, top-p = 0.9',
        continuation: ' a small village where everyone knew each other and shared their stories',
        takeaway: '连贯又有变化。Top-p 修剪掉荒谬的尾部，temperature 防止分布坍缩。',
      },
    ] as readonly Panel[],
    outro: (
      <>
        生产环境的 LLM 服务通常落在{' '}
        <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">T = 0.7-1.0</code> 搭配{' '}
        <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">top-p = 0.9</code>（或一个适中的
        top-K）附近。两个旋钮分工不同：temperature 重塑整个分布，top-p 截断长尾。两者合力避开你在上面看到的贪心循环和高温胡话这两种失败模式。
      </>
    ),
  },
} as const;

export function SamplingFailureModes() {
  const copy = COPY[useLocale()];
  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
        <div className="text-[11px] text-muted-foreground">{copy.subheader}</div>
      </div>

      <div className="rounded-md bg-muted/40 px-3 py-2 font-mono text-[11px] text-muted-foreground">
        {copy.promptLabel}
        <span className="text-foreground/90">{JSON.stringify(PROMPT)}</span>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        {copy.panels.map((p, i) => (
          <div key={`panel-${i}`} className="space-y-2 rounded-md border border-border/60 bg-muted/20 p-3">
            <div className="flex items-baseline justify-between gap-2">
              <div className="font-mono text-[12px] font-semibold text-foreground">{p.label}</div>
              <div className="font-mono text-[10px] text-muted-foreground">{p.config}</div>
            </div>
            <div className="rounded-md border border-border/50 bg-background px-2 py-2 font-mono text-[12px] leading-relaxed">
              <span className="text-muted-foreground">{PROMPT}</span>
              <span className="text-primary">{p.continuation}</span>
            </div>
            <p className="text-[11px] text-muted-foreground">{p.takeaway}</p>
          </div>
        ))}
      </div>

      <p className="text-[12px] text-foreground/85">{copy.outro}</p>
    </div>
  );
}

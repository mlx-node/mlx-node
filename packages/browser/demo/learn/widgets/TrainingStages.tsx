import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Chapter 15 (post-training) — the three stages that turn a pile of weights
 * into a helpful assistant. Static, reading-focused: a left-to-right strip
 * (stacks on mobile) of pretraining → instruction tuning → preference tuning,
 * each card noting the data, what the stage teaches, and how the objective
 * relates to the next-token cross-entropy from the Training chapter.
 */

type Stage = {
  name: string;
  tag: string;
  data: string;
  learns: string;
  objective: string;
  bg: string;
};

// Stage card backgrounds, in display order (paint only — text lives in COPY).
const STAGE_BG = ['oklch(0.7 0.13 220 / 0.16)', 'oklch(0.72 0.15 150 / 0.16)', 'oklch(0.72 0.04 280 / 0.2)'];

const COPY = {
  en: {
    header: 'Base model → assistant, in three stages',
    dataLabel: 'Data: ',
    learnsLabel: 'Learns: ',
    stages: [
      {
        name: 'Pretraining',
        tag: 'base model',
        data: 'Trillions of tokens of raw web text, books, code',
        learns: 'Language and world knowledge',
        objective: 'Plain next-token cross-entropy. Almost all the GPU-hours live here.',
      },
      {
        name: 'Instruction tuning',
        tag: 'SFT',
        data: '~10K–1M curated (instruction, response) pairs',
        learns: 'To follow instructions in the assistant role',
        objective: 'Same next-token loss — only the dataset changes.',
      },
      {
        name: 'Preference tuning',
        tag: 'RLHF / DPO',
        data: 'Human comparisons of candidate responses',
        learns: 'To be helpful, harmless, and honest',
        objective: 'A new objective — optimize a preference/reward, not plain CE.',
      },
    ],
    body: (
      <>
        Only the first stage needs the internet-scale corpus. The two post-training stages are comparatively tiny —
        they don't teach the model new facts so much as <em>shape how it uses</em> what pretraining already gave it.
      </>
    ),
  },
  zh: {
    header: '基础模型 → 助手，三个阶段',
    dataLabel: '数据：',
    learnsLabel: '学到：',
    stages: [
      {
        name: '预训练',
        tag: '基础模型',
        data: '数万亿 token 的原始网页文本、书籍、代码',
        learns: '语言与世界知识',
        objective: '朴素的下一个 token 交叉熵。几乎所有 GPU 时数都花在这里。',
      },
      {
        name: '指令微调',
        tag: 'SFT',
        data: '约 10K–1M 条精心整理的（指令，回复）对',
        learns: '以助手的角色听从指令',
        objective: '同样的下一个 token 损失——只是数据集换了。',
      },
      {
        name: '偏好微调',
        tag: 'RLHF / DPO',
        data: '人类对候选回复的比较',
        learns: '变得有帮助、无害且诚实',
        objective: '一个新目标——优化偏好/奖励，而不是朴素的交叉熵。',
      },
    ],
    body: (
      <>
        只有第一个阶段需要互联网规模的语料。两个后训练阶段相对都很小——它们与其说在教模型新的事实，不如说在
        <em>塑造它如何使用</em>预训练已经给它的东西。
      </>
    ),
  },
} as const;

export function TrainingStages() {
  const copy = COPY[useLocale()];
  const stages: Stage[] = copy.stages.map((s, i) => ({ ...s, bg: STAGE_BG[i] }));
  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
      <div className="flex flex-col gap-2 md:flex-row md:items-stretch">
        {stages.map((s, i) => (
          <React.Fragment key={s.name}>
            <div
              className="flex min-w-0 flex-1 flex-col gap-1 rounded-md border border-border/60 p-2.5"
              style={{ backgroundColor: s.bg }}
            >
              <div className="flex items-baseline justify-between gap-2">
                <span className="text-[13px] font-medium text-foreground">{s.name}</span>
                <span className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">{s.tag}</span>
              </div>
              <div className="text-[11px] text-foreground/85">
                <span className="text-muted-foreground">{copy.dataLabel}</span>
                {s.data}
              </div>
              <div className="text-[11px] text-foreground/85">
                <span className="text-muted-foreground">{copy.learnsLabel}</span>
                {s.learns}
              </div>
              <div className="mt-auto pt-1 text-[11px] text-muted-foreground">{s.objective}</div>
            </div>
            {i < stages.length - 1 ? (
              <div className="flex items-center justify-center text-muted-foreground" aria-hidden>
                <span className="md:hidden">↓</span>
                <span className="hidden md:inline">→</span>
              </div>
            ) : null}
          </React.Fragment>
        ))}
      </div>
      <p className="text-[12px] text-foreground/85">{copy.body}</p>
    </div>
  );
}

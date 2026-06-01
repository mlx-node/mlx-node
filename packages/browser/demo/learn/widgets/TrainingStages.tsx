import * as React from 'react';

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

const STAGES: Stage[] = [
  {
    name: 'Pretraining',
    tag: 'base model',
    data: 'Trillions of tokens of raw web text, books, code',
    learns: 'Language and world knowledge',
    objective: 'Plain next-token cross-entropy. Almost all the GPU-hours live here.',
    bg: 'oklch(0.7 0.13 220 / 0.16)',
  },
  {
    name: 'Instruction tuning',
    tag: 'SFT',
    data: '~10K–1M curated (instruction, response) pairs',
    learns: 'To follow instructions in the assistant role',
    objective: 'Same next-token loss — only the dataset changes.',
    bg: 'oklch(0.72 0.15 150 / 0.16)',
  },
  {
    name: 'Preference tuning',
    tag: 'RLHF / DPO',
    data: 'Human comparisons of candidate responses',
    learns: 'To be helpful, harmless, and honest',
    objective: 'A new objective — optimize a preference/reward, not plain CE.',
    bg: 'oklch(0.72 0.04 280 / 0.2)',
  },
];

export function TrainingStages() {
  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="text-xs uppercase tracking-wider text-muted-foreground">
        Base model → assistant, in three stages
      </div>
      <div className="flex flex-col gap-2 md:flex-row md:items-stretch">
        {STAGES.map((s, i) => (
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
                <span className="text-muted-foreground">Data: </span>
                {s.data}
              </div>
              <div className="text-[11px] text-foreground/85">
                <span className="text-muted-foreground">Learns: </span>
                {s.learns}
              </div>
              <div className="mt-auto pt-1 text-[11px] text-muted-foreground">{s.objective}</div>
            </div>
            {i < STAGES.length - 1 ? (
              <div className="flex items-center justify-center text-muted-foreground" aria-hidden>
                <span className="md:hidden">↓</span>
                <span className="hidden md:inline">→</span>
              </div>
            ) : null}
          </React.Fragment>
        ))}
      </div>
      <p className="text-[12px] text-foreground/85">
        Only the first stage needs the internet-scale corpus. The two post-training stages are comparatively tiny — they
        don't teach the model new facts so much as <em>shape how it uses</em> what pretraining already gave it.
      </p>
    </div>
  );
}

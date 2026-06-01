import * as React from 'react';

import { TopKBars } from '../inspector/TopKBars';

/**
 * Chapter 16 (architecture) — why the model hallucinates, shown with the same
 * top-K bars used in the LM-head and Sampling chapters. Static synthetic
 * distributions (no worker): a grounded prompt where the most-probable token is
 * also true, versus an ungrounded one where no true answer exists but the model
 * still emits a fluent, confident guess. The contrast — peaked vs flat, but
 * "committed" either way — is the whole lesson.
 */

type Scenario = {
  key: 'grounded' | 'ungrounded';
  label: string;
  prompt: string;
  cands: { text: string; prob: number }[];
  note: string;
};

const SCENARIOS: Scenario[] = [
  {
    key: 'grounded',
    label: 'Well-known fact',
    prompt: 'The Eiffel Tower stands in the city of',
    cands: [
      { text: ' Paris', prob: 0.93 },
      { text: ' France', prob: 0.03 },
      { text: ' Lyon', prob: 0.012 },
      { text: ' Europe', prob: 0.008 },
      { text: ' Rome', prob: 0.005 },
    ],
    note: 'A fact that appears constantly in the training data: the most-probable token is also the true one. But the model never looked anything up — truth and plausibility just happen to coincide here.',
  },
  {
    key: 'ungrounded',
    label: 'No real answer',
    prompt: "Smith & Lee's 2017 paper on quantum bananas appeared in the journal",
    cands: [
      { text: ' Nature', prob: 0.16 },
      { text: ' Physical', prob: 0.12 },
      { text: ' the', prob: 0.1 },
      { text: ' Science', prob: 0.08 },
      { text: ' Journal', prob: 0.06 },
    ],
    note: "There is no such paper. The model has no built-in “I don't know” — it emits the most plausible-looking continuation anyway. The distribution is flatter (it is less sure), but it still commits to one fluent, confident, and false token. That is a hallucination.",
  },
];

function ToggleButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={[
        'rounded px-2.5 py-1 text-xs font-medium transition-colors',
        active ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground',
      ].join(' ')}
    >
      {children}
    </button>
  );
}

export function HallucinationDemo() {
  const [idx, setIdx] = React.useState(0);
  const s = SCENARIOS[idx];
  const ids = s.cands.map((_, i) => i);
  const probs = s.cands.map((c) => c.prob);
  const texts = s.cands.map((c) => c.text);

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">Plausible ≠ true</div>
        <div className="inline-flex rounded-md border border-border p-0.5">
          {SCENARIOS.map((sc, i) => (
            <ToggleButton key={sc.key} active={idx === i} onClick={() => setIdx(i)}>
              {sc.label}
            </ToggleButton>
          ))}
        </div>
      </div>

      <div className="rounded-md border border-border/60 bg-muted/30 p-2 font-mono text-[12px] text-foreground/90">
        {s.prompt}
        <span className="text-muted-foreground"> ▮</span>
      </div>

      <TopKBars ids={ids} probs={probs} texts={texts} sampledTokenId={0} runKey={idx} />

      <p className="text-[10px] text-muted-foreground">
        Illustrative distribution — hand-picked to show the shape, not live output from the model.
      </p>

      <p className="text-[12px] text-foreground/85">{s.note}</p>
    </div>
  );
}

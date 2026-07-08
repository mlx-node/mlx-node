import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';
import { TopKBars } from '../inspector/TopKBars';
import { SegmentedToggle } from '../scaffolding/SegmentedToggle';

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

// The prompts and candidate tokens are literal model I/O (they ARE the
// distributions being illustrated) and stay English in both locales — only
// the labels and notes around them localize. The zh toggle labels match the
// zh chapter's checkpoint text exactly ("广为人知的事实" / "没有真实答案").
const CANDS: Record<Scenario['key'], { text: string; prob: number }[]> = {
  grounded: [
    { text: ' Paris', prob: 0.93 },
    { text: ' France', prob: 0.03 },
    { text: ' Lyon', prob: 0.012 },
    { text: ' Europe', prob: 0.008 },
    { text: ' Rome', prob: 0.005 },
  ],
  ungrounded: [
    { text: ' Nature', prob: 0.16 },
    { text: ' Physical', prob: 0.12 },
    { text: ' the', prob: 0.1 },
    { text: ' Science', prob: 0.08 },
    { text: ' Journal', prob: 0.06 },
  ],
};

const COPY = {
  en: {
    header: 'Plausible ≠ true',
    scenarios: [
      {
        key: 'grounded',
        label: 'Well-known fact',
        prompt: 'The Eiffel Tower stands in the city of',
        cands: CANDS.grounded,
        note: 'A fact that appears constantly in the training data: the most-probable token is also the true one. But the model never looked anything up — truth and plausibility just happen to coincide here.',
      },
      {
        key: 'ungrounded',
        label: 'No real answer',
        prompt: "Smith & Lee's 2017 paper on quantum bananas appeared in the journal",
        cands: CANDS.ungrounded,
        note: "There is no such paper. The model has no built-in “I don't know” — it emits the most plausible-looking continuation anyway. The distribution is flatter (it is less sure), but it still commits to one fluent, confident, and false token. That is a hallucination.",
      },
    ] as Scenario[],
    footnote: 'Illustrative distribution — hand-picked to show the shape, not live output from the model.',
  },
  zh: {
    header: '“合理” ≠ “真实”',
    scenarios: [
      {
        key: 'grounded',
        label: '广为人知的事实',
        prompt: 'The Eiffel Tower stands in the city of',
        cands: CANDS.grounded,
        note: '一个在训练数据中反复出现的事实：最可能的 token 恰好也是真实的那个。但模型从未查过任何东西——只是“真实”与“合理”在这里碰巧重合。',
      },
      {
        key: 'ungrounded',
        label: '没有真实答案',
        prompt: "Smith & Lee's 2017 paper on quantum bananas appeared in the journal",
        cands: CANDS.ungrounded,
        note: '这篇论文根本不存在。模型没有内置的“我不知道”——它照样输出看起来最合理的续写。分布更平（它更没把握），但它仍然押注在一个流畅、自信而虚假的 token 上。这就是幻觉。',
      },
    ] as Scenario[],
    footnote: '示意性分布——为展示形状而手工挑选，并非模型的实时输出。',
  },
} as const;

export function HallucinationDemo() {
  const copy = COPY[useLocale()];
  const [idx, setIdx] = React.useState(0);
  const s = copy.scenarios[idx];
  const ids = s.cands.map((_, i) => i);
  const probs = s.cands.map((c) => c.prob);
  const texts = s.cands.map((c) => c.text);

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
        <SegmentedToggle
          value={idx}
          onChange={setIdx}
          options={copy.scenarios.map((sc, i) => ({ value: i, label: sc.label }))}
        />
      </div>

      <div className="rounded-md border border-border/60 bg-muted/30 p-2 font-mono text-[12px] text-foreground/90">
        {s.prompt}
        <span className="text-muted-foreground"> ▮</span>
      </div>

      <TopKBars ids={ids} probs={probs} texts={texts} sampledTokenId={0} runKey={idx} />

      <p className="text-[10px] text-muted-foreground">{copy.footnote}</p>

      <p className="text-[12px] text-foreground/85">{s.note}</p>
    </div>
  );
}

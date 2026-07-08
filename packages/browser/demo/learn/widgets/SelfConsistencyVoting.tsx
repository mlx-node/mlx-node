import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * SamplingReasoningSection widget (chapter 11 "Sampling" sub-chapter,
 * "Reasoning & test-time compute") — self-consistency, made concrete.
 *
 * Self-contained: NO worker, NO model, NO WASM — pure React state over a
 * fixed, scripted toy arithmetic problem, so it server-renders for crawlers
 * and needs no randomness (deterministic first paint, same server vs client).
 *
 * The lesson is about SAMPLING, not reasoning content: self-consistency
 * (Wang et al. 2022, arXiv:2203.11171) needs several INDEPENDENT completions
 * at temperature > 0 — this chapter's own knob — then majority-votes their
 * final answers. Toggling to "Greedy — T = 0" makes the point by contrast:
 * greedy decoding is deterministic, so re-running it produces the exact same
 * chain (here, a wrong one) every time — there is nothing to vote on. Only
 * once T > 0 gives five genuinely different chains does a majority vote have
 * anything to average over, and it can recover the right answer even though
 * two of the five individual samples are wrong (one of them making the exact
 * same slip as the greedy chain).
 *
 * All reasoning text and the answer values are illustrative/scripted, not
 * live model output — flagged in the footnote, matching this course's other
 * scripted-example widgets (TemperatureStories, SamplingFailureModes).
 */

type PathDatum = { answer: number; correct: boolean };

// Toy problem: 5 tennis balls already owned, plus 2 cans of 3 balls each — the
// correct total is 5 + 2*3 = 11. Locale-independent (only the answer values
// and correctness flags; the prose describing each path lives in COPY).
const CORRECT_ANSWER = 11;

const GREEDY_PATH: PathDatum = { answer: 10, correct: false };

const SAMPLED_PATHS: PathDatum[] = [
  { answer: 11, correct: true },
  { answer: 11, correct: true },
  { answer: 21, correct: false },
  { answer: 11, correct: true },
  { answer: 10, correct: false },
];

type Mode = 'greedy' | 'sampled';

const COPY = {
  en: {
    header: 'Self-consistency: many samples, one vote',
    modeGreedy: 'Greedy — T = 0',
    modeSampled: 'Self-consistency — T = 0.7, n = 5',
    problemLabel: 'Problem: ',
    problem: 'Roger has 5 tennis balls. He buys 2 more cans of tennis balls, 3 balls per can. How many does he have now?',
    correctAnswerNote: (n: number) => (
      <>
        Correct answer: <strong className="font-mono text-emerald-700 dark:text-emerald-300">{n}</strong>
      </>
    ),
    pathLabel: (i: number) => `Sample ${i + 1}`,
    greedyPathLabel: 'The one greedy chain',
    answerBadge: (n: number) => `→ ${n}`,
    correctTag: 'correct',
    wrongTag: 'wrong',
    greedyReasoning: '2 cans × 3 balls = 5 (slipped — should be 6). 5 already owned + 5 = 10.',
    sampledReasonings: [
      '5 already owned + 2×3 bought = 5 + 6 = 11.',
      '2 cans of 3 balls is 6. 6 + the 5 he already had = 11.',
      '5 + 2 = 7 groups of balls, 7 × 3 = 21.',
      '2×3 = 6 new balls. Plus the 5 he had: 11.',
      '2×3 = 5 (slipped). 5 + 5 = 10.',
    ],
    greedyNote:
      'Deterministic — rerun this a hundred times and you get this exact chain, this exact wrong answer, every time. There is nothing to vote on.',
    tallyHeader: 'Votes on the final answer',
    tallyRow: (answer: number, count: number, total: number) => `${answer} — ${count} of ${total} samples`,
    majorityTag: 'majority',
    tallyNote:
      'Majority: 3 of 5 paths land on 11, the correct answer — even though 2 of the 5 individual samples are wrong, and one of them makes the exact same slip as the greedy chain above. Voting over independent samples cancels out an error that any single chain, including the greedy one, would have committed to.',
    footnote:
      'Illustrative scripted example, not a live model run. The five sampled paths are independent draws at T = 0.7 from the exact same prompt — only where the sampler happens to land differs.',
  },
  zh: {
    header: '自洽性（self-consistency）：多份采样，一次投票',
    modeGreedy: 'Greedy — T = 0',
    modeSampled: '自洽性 — T = 0.7，n = 5',
    problemLabel: '题目：',
    problem: 'Roger 有 5 个网球。他又买了 2 罐网球，每罐 3 个。他现在一共有多少个网球？',
    correctAnswerNote: (n: number) => (
      <>
        正确答案：<strong className="font-mono text-emerald-700 dark:text-emerald-300">{n}</strong>
      </>
    ),
    pathLabel: (i: number) => `样本 ${i + 1}`,
    greedyPathLabel: '唯一的那条 greedy 链',
    answerBadge: (n: number) => `→ ${n}`,
    correctTag: '正确',
    wrongTag: '错误',
    greedyReasoning: '2 罐 × 3 个 = 5（算错了，应为 6）。原有 5 个 + 5 = 10。',
    sampledReasonings: [
      '原有 5 个 + 新买 2×3 = 5 + 6 = 11。',
      '2 罐、每罐 3 个，共 6 个。6 + 原有的 5 个 = 11。',
      '5 + 2 = 7 组网球，7 × 3 = 21。',
      '2×3 = 6 个新网球。加上原有的 5 个：11。',
      '2×3 = 5（算错了）。5 + 5 = 10。',
    ],
    greedyNote: '确定性的——重跑一百次也是这同一条链、同一个错误答案。没有什么可以投票的。',
    tallyHeader: '对最终答案的投票',
    tallyRow: (answer: number, count: number, total: number) => `${answer} —— ${total} 个样本中的 ${count} 个`,
    majorityTag: '多数票',
    tallyNote:
      '多数票：5 条路径里有 3 条落在 11，也就是正确答案——尽管 5 个独立样本里有 2 个是错的，其中一个还和上面那条 greedy 链犯了一模一样的错。对独立样本投票，抵消了任何单一链（包括 greedy 那条）本会一条道走到黑的错误。',
    footnote:
      '示意用的预先写定的例子，并非模型的实时输出。五条采样路径是在 T = 0.7、同一个提示词下各自独立抽取的——区别只在于采样器这次落在了哪里。',
  },
} as const;

function AnswerChip({ answer, correct, correctLabel, wrongLabel }: { answer: number; correct: boolean; correctLabel: string; wrongLabel: string }) {
  return (
    <span
      className={[
        'inline-flex items-center gap-1 rounded px-1.5 py-0.5 font-mono text-[11px] font-semibold',
        correct
          ? 'bg-emerald-500/15 text-emerald-700 dark:text-emerald-300'
          : 'bg-rose-500/15 text-rose-700 dark:text-rose-300',
      ].join(' ')}
      title={correct ? correctLabel : wrongLabel}
    >
      {correct ? '✓' : '✕'} {answer}
    </span>
  );
}

export function SelfConsistencyVoting() {
  const copy = COPY[useLocale()];
  const [mode, setMode] = React.useState<Mode>('sampled');

  const tally = React.useMemo(() => {
    const counts = new Map<number, number>();
    for (const p of SAMPLED_PATHS) counts.set(p.answer, (counts.get(p.answer) ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => b[1] - a[1]);
  }, []);
  const total = SAMPLED_PATHS.length;
  const majorityAnswer = tally[0]?.[0];
  const maxCount = tally[0]?.[1] ?? 1;

  return (
    <div className="not-prose my-5 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.header}</div>
        <div className="flex gap-1 rounded-full border border-border bg-muted/40 p-0.5 text-[11px]">
          {(['greedy', 'sampled'] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              className={[
                'rounded-full px-2.5 py-1 font-medium transition-colors',
                mode === m ? 'bg-primary text-primary-foreground' : 'text-muted-foreground hover:text-foreground',
              ].join(' ')}
              aria-pressed={mode === m}
            >
              {m === 'greedy' ? copy.modeGreedy : copy.modeSampled}
            </button>
          ))}
        </div>
      </div>

      <div className="rounded border border-border/60 bg-muted/20 p-2 text-[12px]">
        <span className="text-muted-foreground">{copy.problemLabel}</span>
        <span className="text-foreground/90">{copy.problem}</span>
        <span className="ml-2 whitespace-nowrap">{copy.correctAnswerNote(CORRECT_ANSWER)}</span>
      </div>

      {mode === 'greedy' ? (
        <div className="space-y-2">
          <div className="rounded-md border border-rose-500/30 bg-rose-500/5 p-2.5">
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
                {copy.greedyPathLabel}
              </span>
              <AnswerChip
                answer={GREEDY_PATH.answer}
                correct={GREEDY_PATH.correct}
                correctLabel={copy.correctTag}
                wrongLabel={copy.wrongTag}
              />
            </div>
            <p className="text-[12px] text-foreground/85">{copy.greedyReasoning}</p>
          </div>
          <p className="text-[11px] text-muted-foreground">{copy.greedyNote}</p>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-5">
            {SAMPLED_PATHS.map((p, i) => (
              <div key={i} className="rounded-md border border-border/70 bg-card p-2">
                <div className="mb-1 flex items-center justify-between gap-1">
                  <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                    {copy.pathLabel(i)}
                  </span>
                </div>
                <p className="mb-1.5 text-[11px] leading-snug text-foreground/85">{copy.sampledReasonings[i]}</p>
                <AnswerChip answer={p.answer} correct={p.correct} correctLabel={copy.correctTag} wrongLabel={copy.wrongTag} />
              </div>
            ))}
          </div>

          <div className="space-y-1.5 rounded-md border border-border/60 bg-muted/10 p-2.5">
            <div className="text-[11px] font-medium uppercase tracking-wider text-muted-foreground">
              {copy.tallyHeader}
            </div>
            {tally.map(([answer, count]) => {
              const isMajority = answer === majorityAnswer;
              const pct = (count / maxCount) * 100;
              return (
                <div key={answer} className="grid grid-cols-[9rem_minmax(0,1fr)] items-center gap-2 text-[12px]">
                  <span className="truncate font-mono text-foreground/80">
                    {copy.tallyRow(answer, count, total)}
                    {isMajority ? (
                      <span className="ml-1.5 rounded bg-primary/15 px-1 py-0.5 text-[10px] font-semibold text-primary">
                        {copy.majorityTag}
                      </span>
                    ) : null}
                  </span>
                  <div className="relative h-3 w-full overflow-hidden rounded bg-muted">
                    <div
                      className={['h-full origin-left', isMajority ? 'bg-primary' : 'bg-foreground/30'].join(' ')}
                      style={{ width: `${Math.max(4, Math.min(100, pct))}%` }}
                    />
                  </div>
                </div>
              );
            })}
            <p className="pt-1 text-[11px] text-muted-foreground">{copy.tallyNote}</p>
          </div>
        </div>
      )}

      <p className="text-[10px] text-muted-foreground">{copy.footnote}</p>
    </div>
  );
}

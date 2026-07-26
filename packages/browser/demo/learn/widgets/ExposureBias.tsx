import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';
import { DiagramFrame, useStepPlayer } from '../motion';

/**
 * Chapter 13 (Training) supplement — exposure bias, the reason inference drifts.
 *
 * TeacherForcingAnimation shows ONE training step: per-position distributions
 * and the cross-entropy bars. This widget shows the OTHER half of the story —
 * what happens ACROSS steps at inference time, and why it differs from training.
 *
 * Two lanes advance in lockstep, one token per step:
 *   Lane A "Teacher forcing (training)": the next input is always the GOLD
 *     token (chapter: "the input at position i+1 is the TRUE token from position
 *     i"). The lane can never leave the reference sentence — every step lands on
 *     green, regardless of what the model would have predicted.
 *   Lane B "Free-running (inference)": the model appends its OWN prediction. It
 *     tracks the reference until step 5, where it predicts " couch" instead of
 *     the gold " mat". From then on the context is off-reference, so the lane
 *     keeps drifting (" couch", " and", " knocked") — shown in a drift color.
 *
 * The point is exposure bias: training never lets the model consume its own
 * mistakes, so at inference small errors compound. Synthetic / scripted — no
 * model, no worker, no WASM.
 *
 * Animation model: `useStepPlayer` + `DiagramFrame` from `learn/motion`, in
 * place of this file's former hand-rolled `setInterval` + its own play button,
 * header, footnotes and divergence-note box. There is no `<svg>` here, so this
 * is a kit-only adoption: the two lane rows keep their existing markup.
 *
 * The divergence note IS this widget's caption — it is the one thing that
 * rewrites itself as the sweep advances — so it moves onto `DiagramFrame`'s
 * caption slot, where all three of its arms are stacked as invisible sizers and
 * the chapter body can no longer hop when the lanes split. What it loses in the
 * move is its own tinted box; the rose emphasis on the label at the moment of
 * divergence survives, because that emphasis is inline in the caption itself.
 */

// The shared gold reference both lanes are scored against.
const REFERENCE = ['The', ' cat', ' sat', ' on', ' the', ' mat'] as const;

// Free-running inference output. It matches REFERENCE up to DIVERGE_STEP, then
// the model picks its own (plausible but different) continuation and drifts.
const FREE_RUN = ['The', ' cat', ' sat', ' on', ' the', ' couch', ' and', ' knocked'] as const;

// Index of the first token where the free-running lane leaves the reference.
// At this step the model predicts " couch"; teacher forcing would force-feed " mat".
const DIVERGE_STEP = 5;

const TOTAL_STEPS = FREE_RUN.length;
const STEP_MS = 1500;

// The sweep runs 0..TOTAL_STEPS inclusive: frame 0 is the empty "(start)" state
// before any token is revealed, so there is one more frame than there are
// tokens. Same count the old `(s + 1) % (TOTAL_STEPS + 1)` interval walked.
const TOTAL_FRAMES = TOTAL_STEPS + 1;

function renderToken(t: string): string {
  return t.startsWith(' ') ? '·' + t.slice(1) : t;
}

type Cell = {
  text: string;
  /** 'gold' = on the reference, 'drift' = the model has left the reference. */
  kind: 'gold' | 'drift';
};

/** Build the free-running lane up to (and including) `count` revealed tokens. */
function freeRunCells(count: number): Cell[] {
  return FREE_RUN.slice(0, count).map((text, i) => ({
    text,
    kind: i >= DIVERGE_STEP ? 'drift' : 'gold',
  }));
}

// Per-locale copy — every user-visible English string moved here verbatim.
// Model tokens (' mat', ' couch') stay English in both locales.
//
// Play / Pause are NOT here any more: those verbs belong to the shared
// `StepControls`, which carries its own en/zh pair so every animated widget in
// the course says the same word. The step counter stays — it is content, not a
// control, and the shared controls do not render one.
const COPY = {
  en: {
    title: 'Exposure bias — why inference drifts',
    stepCounter: (s: number, total: number) => `step ${s}/${total}`,
    laneATitle: 'Teacher forcing (training)',
    laneANote: 'next input = true token',
    laneBTitle: 'Free-running (inference)',
    laneBNote: 'next input = own prediction',
    start: '(start)',
    divergenceLabel: 'Divergence:',
    divergence: (
      <>
        model predicted{' '}
        <span className="font-mono text-rose-700 dark:text-rose-300">{renderToken(FREE_RUN[DIVERGE_STEP]!)}</span> —
        training would have force-fed{' '}
        <span className="font-mono text-emerald-700 dark:text-emerald-300">
          {renderToken(REFERENCE[DIVERGE_STEP]!)}
        </span>
        . Now off-reference, the next tokens drift further.
      </>
    ),
    notYet: 'Both lanes track the reference so far — the model has not yet had to recover from one of its own choices.',
    scripted: 'Illustrative — the divergence is scripted, not live output from the model.',
    summary: (
      <>
        Teacher forcing means training <em>never</em> lets the model see its own mistakes: at every position the input
        is the true previous token. But at inference the model must consume its <em>own</em> outputs, so one off-gold
        choice shifts the context onto a path it was never trained on, and small errors compound into drift. That gap
        between the teacher-forced training distribution and the free-running inference distribution is{' '}
        <strong>exposure bias</strong>.
      </>
    ),
  },
  zh: {
    title: '曝光偏差（exposure bias）——推理为什么会跑偏',
    stepCounter: (s: number, total: number) => `第 ${s}/${total} 步`,
    laneATitle: 'Teacher forcing（训练）',
    laneANote: '下一个输入 = 真实 token',
    laneBTitle: '自由生成（推理）',
    laneBNote: '下一个输入 = 自己的预测',
    start: '（起点）',
    divergenceLabel: '分叉：',
    divergence: (
      <>
        模型预测了{' '}
        <span className="font-mono text-rose-700 dark:text-rose-300">{renderToken(FREE_RUN[DIVERGE_STEP]!)}</span>
        ——而训练时会强制喂入{' '}
        <span className="font-mono text-emerald-700 dark:text-emerald-300">
          {renderToken(REFERENCE[DIVERGE_STEP]!)}
        </span>
        。一旦偏离参考序列，后续 token 就越漂越远。
      </>
    ),
    notYet: '到目前为止，两条通道都贴着参考序列——模型还没有被迫从自己的某个选择中恢复。',
    scripted: '仅为示意——这处分叉是预先写好的脚本，不是模型的实时输出。',
    summary: (
      <>
        Teacher forcing 意味着训练<em>从不</em>让模型看到自己的错误：每个位置的输入都是真实的前一个
        token。但推理时模型必须消费自己的<em>输出</em>
        ，于是一次偏离真实答案的选择就把上下文带上一条它从未被训练过的路径，小错误不断累积成漂移。teacher forcing
        下的训练分布与自由生成的推理分布之间的这道鸿沟，就是<strong>exposure bias（曝光偏差）</strong>。
      </>
    ),
  },
} as const;

export function ExposureBias() {
  const locale = useLocale();
  const copy = COPY[locale];
  const player = useStepPlayer(TOTAL_FRAMES, { frameMs: STEP_MS });

  // `player.frame` runs 0..TOTAL_STEPS; the number of revealed tokens is the
  // frame index, so the final frame shows the whole sequence before looping.
  const revealed = player.frame;
  const goldCells = REFERENCE.slice(0, revealed);
  const freeCells = freeRunCells(revealed);
  const diverged = revealed > DIVERGE_STEP;
  const atDivergence = revealed === DIVERGE_STEP + 1;

  // The caption has exactly three arms, and this builder is two of them: the
  // label is highlighted on the single frame the lanes actually split, plain on
  // every frame after (the sentence itself is identical in both).
  const divergenceCaption = (highlight: boolean) => (
    <>
      <span className={highlight ? 'font-semibold text-rose-700 dark:text-rose-300' : 'text-foreground/90'}>
        {copy.divergenceLabel}
      </span>{' '}
      {copy.divergence}
    </>
  );

  const caption: React.ReactNode = diverged ? divergenceCaption(atDivergence) : copy.notYet;

  // Every caption this sweep can reach, so the slot reserves the tallest and the
  // chapter body cannot hop when the lanes split. Three entries for the three
  // arms above: not-yet-diverged, diverged-this-frame, diverged-earlier.
  const captions = [copy.notYet, divergenceCaption(true), divergenceCaption(false)];

  return (
    <>
      <DiagramFrame
        title={copy.title}
        player={player}
        locale={locale}
        caption={caption}
        captions={captions}
        note={copy.scripted}
      >
        <div className="space-y-3">
          {/* Not a control, so `StepControls` does not own it: it says which of the
              8 tokens the lanes are on. Right-aligned to sit under the controls,
              where it used to live. */}
          <div className="flex justify-end">
            <span className="font-mono text-[11px] text-muted-foreground">
              {copy.stepCounter(Math.min(revealed, TOTAL_STEPS), TOTAL_STEPS)}
            </span>
          </div>

          {/* Lane A — teacher forcing: always force-fed the gold token. */}
          <div className="space-y-1">
            <div className="flex items-baseline justify-between">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.laneATitle}</span>
              <span className="text-[10px] text-muted-foreground">{copy.laneANote}</span>
            </div>
            <div className="flex min-h-9 flex-wrap items-center gap-1 rounded-md border border-border/60 bg-muted/30 p-2">
              {goldCells.map((t, i) => (
                <span
                  key={`gold-${i}`}
                  className="rounded bg-emerald-500/70 px-1.5 py-0.5 font-mono text-[11px] text-white outline outline-1 outline-emerald-400"
                >
                  {renderToken(t)}
                </span>
              ))}
              {revealed === 0 ? (
                <span className="font-mono text-[11px] text-muted-foreground/60">{copy.start}</span>
              ) : null}
              <span className="ml-0.5 font-mono text-[11px] text-muted-foreground" aria-hidden>
                ▮
              </span>
            </div>
          </div>

          {/* Lane B — free-running inference: feeds its own predictions back in. */}
          <div className="space-y-1">
            <div className="flex items-baseline justify-between">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.laneBTitle}</span>
              <span className="text-[10px] text-muted-foreground">{copy.laneBNote}</span>
            </div>
            <div className="flex min-h-9 flex-wrap items-center gap-1 rounded-md border border-border/60 bg-muted/30 p-2">
              {freeCells.map((c, i) => (
                <span
                  key={`free-${i}`}
                  className={[
                    'rounded px-1.5 py-0.5 font-mono text-[11px] transition-colors',
                    c.kind === 'gold'
                      ? 'bg-emerald-500/70 text-white outline outline-1 outline-emerald-400'
                      : 'bg-rose-500/70 text-white outline outline-1 outline-rose-400',
                  ].join(' ')}
                >
                  {renderToken(c.text)}
                </span>
              ))}
              {revealed === 0 ? (
                <span className="font-mono text-[11px] text-muted-foreground/60">{copy.start}</span>
              ) : null}
              <span className="ml-0.5 font-mono text-[11px] text-muted-foreground" aria-hidden>
                ▮
              </span>
            </div>
          </div>
        </div>
      </DiagramFrame>

      {/* The chapter's takeaway, not a diagram footnote: body copy at body
          weight, so it stays outside the frame rather than being folded into
          `note` (which renders at footnote tier). */}
      <p className="text-[12px] text-foreground/85">{copy.summary}</p>
    </>
  );
}

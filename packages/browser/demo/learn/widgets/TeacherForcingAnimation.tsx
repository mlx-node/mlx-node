import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';
import { DiagramFrame, useStepPlayer } from '../motion';

/**
 * Chapter 12 (Training) supplement — animate teacher forcing.
 *
 * The whole rest of the course is inference: feed a prompt in, sample one
 * token, repeat. Training inverts the loop: feed the whole sequence in at
 * once, predict the next token at *every* position in parallel, take the
 * cross-entropy loss against the actual next token at each position.
 *
 * This widget plays one training step on a fixed 6-token sequence:
 *   "The cat sat on the mat"
 *
 * Step 1: feed all 6 tokens in (one row of cells).
 * Step 2: causal mask in place, model produces 6 predicted-next-token
 *         distributions in parallel (drawn as mini bar-stacks under each
 *         input position).
 * Step 3: the "true next token" at each position is highlighted in the
 *         predictions; the cross-entropy contribution -log(p_target) is
 *         shown as a stacked-bar.
 * Step 4: loss = mean of per-position contributions, drawn as a single bar.
 *
 * Numbers are illustrative — the point is the *shape* of the training step,
 * not real model probabilities.
 *
 * Animation model: `useStepPlayer` + `DiagramFrame` from `learn/motion`,
 * replacing this file's former hand-rolled `setInterval`, its private
 * play/pause button and its own `aria-live` caption box. Same four frames at
 * the same 2200ms pace — this is a refactor, not a retune. The frame now owns
 * the header row, the controls, the caption slot and the whole a11y contract
 * (one accessible copy of the caption at every instant; the live region only
 * fills once the reader presses something).
 *
 * There is no SVG here — the diagram is four rows of divs — so this widget
 * takes the kit only, not the `skin` tokens (RX/SW/DASH/DIM have nothing to
 * apply to).
 */

const SEQ = ['The', ' cat', ' sat', ' on', ' the', ' mat'];

// For each position, a synthetic predicted distribution over a small "alt
// vocab" — the actual next token at index TARGET_IDX, plus 3 confounders.
// We assign probability mass that gets ROUGHLY right (high p_target) for
// easier positions and uncertain (low p_target) for harder ones, so the
// cross-entropy bars have visible variance.
type Position = {
  /** Probabilities over [cat, sat, on, the, mat, dog] in that fixed order. */
  probs: number[];
  /** Which index in `probs` is the true next token. */
  targetIdx: number;
  /** Text labels for each prob bucket (display only). */
  labels: string[];
};

// Hand-picked so the loss visualization shows variety: position 0 ("The")
// is hard (low confidence in " cat"), position 4 (" the") is easy (high in
// " mat" since "on the mat" is a strong continuation).
const POSITIONS: Position[] = [
  // Position 0: input "The" → predict " cat" (low p, hard)
  { probs: [0.28, 0.14, 0.05, 0.18, 0.05, 0.3], targetIdx: 0, labels: ['cat', 'sat', 'on', 'the', 'mat', 'dog'] },
  // Position 1: input "The cat" → predict " sat" (medium)
  { probs: [0.05, 0.45, 0.05, 0.1, 0.05, 0.3], targetIdx: 1, labels: ['cat', 'sat', 'on', 'the', 'mat', 'dog'] },
  // Position 2: input "The cat sat" → predict " on" (medium-high)
  { probs: [0.02, 0.05, 0.6, 0.18, 0.05, 0.1], targetIdx: 2, labels: ['cat', 'sat', 'on', 'the', 'mat', 'dog'] },
  // Position 3: input "The cat sat on" → predict " the" (high)
  { probs: [0.02, 0.02, 0.05, 0.78, 0.08, 0.05], targetIdx: 3, labels: ['cat', 'sat', 'on', 'the', 'mat', 'dog'] },
  // Position 4: input "The cat sat on the" → predict " mat" (high)
  { probs: [0.05, 0.02, 0.02, 0.05, 0.78, 0.08], targetIdx: 4, labels: ['cat', 'sat', 'on', 'the', 'mat', 'dog'] },
  // Position 5: input "The cat sat on the mat" → predict <end> (no target — last position not trained)
  { probs: [0.15, 0.15, 0.15, 0.15, 0.15, 0.25], targetIdx: -1, labels: ['cat', 'sat', 'on', 'the', 'mat', 'dog'] },
];

function renderToken(t: string): string {
  return t.startsWith(' ') ? '·' + t.slice(1) : t;
}

// Per-locale copy — every user-visible English string moved here verbatim.
// Model tokens and the [cat, sat, on, the, mat, dog] bucket labels stay English.
//
// Play / Pause are NOT here any more: those verbs belong to the shared
// `StepControls`, which carries its own en/zh pair so every animated widget in
// the course says the same word. The old "step 3/4" counter is gone with them —
// every one of the four captions already opens with its own step number
// ("Step 3 — …" / "第 3 步——…"), so nothing a reader could see was lost.
const COPY = {
  en: {
    title: 'Teacher forcing — one training step on six tokens',
    stepLabels: [
      'Step 1 — Feed the whole sequence in (one parallel forward pass).',
      'Step 2 — At every position, predict a distribution over the next token. Causal mask keeps each prediction honest.',
      'Step 3 — Compare each prediction to the actual next token. Per-position loss = −log p(target).',
      'Step 4 — Mean across positions = the training loss. Backprop adjusts every parameter to push p(target) up.',
    ],
    inputRow: 'input sequence',
    predictionsRow: 'per-position predictions (after softmax)',
    lossRow: 'per-position cross-entropy = −log p(target)',
    noTarget: 'no target',
    meanRow: 'loss = mean(−log p(target_i)) over 5 trained positions',
    bridgeNote: (
      <>
        Read the loss bars against the probabilities above them — the bridge is just <code>−ln</code>:{' '}
        <span className="font-mono">−ln(0.78) = 0.25</span> (easy position, tiny bar) but{' '}
        <span className="font-mono">−ln(0.28) = 1.27</span> (hard position) — low probability blows the bar up.
      </>
    ),
    observations: (
      <>
        Two crucial observations. <strong>One</strong>: every position is trained simultaneously — the causal mask is
        the only thing that keeps it valid. <strong>Two</strong>: the input fed to position <em>i+1</em> is the{' '}
        <em>true</em> token from position <em>i</em>, not the model's prediction. That's "teacher forcing": during
        training, the model never has to recover from its own mistakes. (At inference, of course, it does — which is the
        small but real reason long-form generation sometimes drifts.)
      </>
    ),
    illustrative:
      'Illustrative — the per-position probabilities here are hand-picked to show the shape of the loss, not live output from the model.',
  },
  zh: {
    title: 'Teacher forcing——在六个 token 上的一次训练步',
    stepLabels: [
      '第 1 步——把整条序列一次喂入（一次并行前向传播）。',
      '第 2 步——在每个位置预测下一个 token 的分布。因果掩码让每个预测保持诚实。',
      '第 3 步——把每个预测与真实的下一个 token 对照。逐位置损失 = −log p(target)。',
      '第 4 步——对各位置取平均 = 训练损失。反向传播据此调整每个参数，把 p(target) 推高。',
    ],
    inputRow: '输入序列',
    predictionsRow: '逐位置预测（softmax 之后）',
    lossRow: '逐位置交叉熵 = −log p(target)',
    noTarget: '无目标',
    meanRow: 'loss = mean(−log p(target_i))，在 5 个受训位置上取平均',
    bridgeNote: (
      <>
        把损失条与上方的概率对照着读——桥梁只是 <code>−ln</code>：<span className="font-mono">−ln(0.78) = 0.25</span>
        （容易的位置，短条）；而 <span className="font-mono">−ln(0.28) = 1.27</span>
        （困难的位置）——概率越低，条被推得越高。
      </>
    ),
    observations: (
      <>
        两个关键观察。<strong>其一</strong>：所有位置同时被训练——唯一让这件事保持成立的是因果掩码。
        <strong>其二</strong>：喂给位置 <em>i+1</em> 的输入是位置 <em>i</em> 的<em>真实</em>{' '}
        token，而不是模型的预测。这就是「teacher
        forcing」：训练期间，模型从不需要从自己的错误中恢复。（推理时它当然必须如此——这正是长文本生成有时会跑偏的那个虽小但真实的原因。）
      </>
    ),
    illustrative: '仅为示意——这里的逐位置概率是手工挑选来展示损失形状的，不是模型的实时输出。',
  },
} as const;

/** One frame per step of the training step. Unchanged from the old modulo-4 sweep. */
const TOTAL_FRAMES = 4;
/** The old hand-rolled `setInterval` pace, kept exactly. */
const FRAME_MS = 2200;

export function TeacherForcingAnimation() {
  const locale = useLocale();
  const copy = COPY[locale];
  const player = useStepPlayer(TOTAL_FRAMES, { frameMs: FRAME_MS });
  const step = player.frame;

  // Every caption this sweep can reach, so DiagramFrame reserves the tallest
  // and chapter 12's body cannot hop when the beat changes.
  //
  // The visible caption has exactly ONE source — `copy.stepLabels[step]`, with
  // `step` the player's frame in [0, 4) — so the sizer list IS that array. No
  // branch, no builder, no variant toggle: four frames, four sentences, and
  // nothing else can appear in the slot.
  const captions = [...copy.stepLabels];

  // Per-position cross-entropy: -log p(target).
  const losses = POSITIONS.map((p) => (p.targetIdx < 0 ? null : -Math.log(Math.max(1e-9, p.probs[p.targetIdx]!))));
  const trainedLosses = losses.filter((l): l is number => l !== null);
  const meanLoss = trainedLosses.reduce((a, b) => a + b, 0) / Math.max(1, trainedLosses.length);

  return (
    <DiagramFrame
      title={copy.title}
      player={player}
      locale={locale}
      caption={copy.stepLabels[step]}
      captions={captions}
      note={
        // All three were footnote-tier before the conversion (11px / 11px /
        // 10px, all muted), so all three belong in `note`. It renders a div, so
        // these are real paragraphs.
        <>
          <p>{copy.bridgeNote}</p>
          <p>{copy.observations}</p>
          <p className="text-[10px]">{copy.illustrative}</p>
        </>
      }
    >
      <div className="space-y-4">
        {/* Row 1 — input tokens */}
        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.inputRow}</div>
          <div className="flex gap-1.5">
            {SEQ.map((t, i) => (
              <div
                key={`in-${i}`}
                className={[
                  'flex h-9 flex-1 items-center justify-center rounded border font-mono text-[12px] transition-all duration-300',
                  step >= 0
                    ? 'border-primary/40 bg-primary/10 text-foreground/95'
                    : 'border-border/40 bg-muted/40 text-muted-foreground/50',
                ].join(' ')}
              >
                {renderToken(t)}
              </div>
            ))}
          </div>
        </div>

        {/* Row 2 — per-position predicted distributions */}
        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.predictionsRow}</div>
          <div className="flex gap-1.5">
            {POSITIONS.map((pos, i) => {
              const visible = step >= 1;
              const targetVisible = step >= 2;
              return (
                <div
                  key={`pos-${i}`}
                  className={[
                    'flex flex-1 flex-col items-stretch gap-0.5 rounded border p-1 transition-all duration-500',
                    visible ? 'border-border/60 bg-background' : 'border-border/30 bg-muted/20 opacity-40',
                  ].join(' ')}
                >
                  {pos.probs.map((p, k) => {
                    const isTarget = k === pos.targetIdx;
                    return (
                      <div key={`b-${i}-${k}`} className="flex items-center gap-1">
                        <span
                          className={[
                            'w-7 shrink-0 truncate font-mono text-[9px]',
                            isTarget && targetVisible
                              ? 'text-emerald-600 dark:text-emerald-300'
                              : 'text-muted-foreground/70',
                          ].join(' ')}
                        >
                          {pos.labels[k]}
                        </span>
                        <div className="relative flex-1">
                          <div
                            className={[
                              'h-2.5 rounded-sm transition-all duration-500',
                              isTarget && targetVisible
                                ? 'bg-emerald-500/70 outline outline-1 outline-emerald-400'
                                : 'bg-primary/40',
                            ].join(' ')}
                            style={{ width: `${visible ? Math.max(2, p * 100) : 0}%` }}
                          />
                        </div>
                        <span className="w-7 shrink-0 text-right font-mono text-[9px] text-muted-foreground">
                          {visible ? p.toFixed(2) : ''}
                        </span>
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>
        </div>

        {/* Row 3 — per-position loss = -log p(target) */}
        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{copy.lossRow}</div>
          <div className="flex gap-1.5">
            {POSITIONS.map((pos, i) => {
              const visible = step >= 2;
              const l = losses[i];
              if (l === null) {
                return (
                  <div
                    key={`l-${i}`}
                    className={[
                      'flex h-12 flex-1 items-center justify-center rounded border text-[9px] text-muted-foreground/70',
                      visible ? 'border-border/40 bg-muted/30' : 'border-border/20 bg-muted/10 opacity-30',
                    ].join(' ')}
                  >
                    {copy.noTarget}
                  </div>
                );
              }
              const maxLossForScale = 3.5;
              const heightPct = Math.min(100, (l / maxLossForScale) * 100);
              return (
                <div
                  key={`l-${i}`}
                  className={[
                    'flex h-12 flex-1 flex-col items-stretch justify-end rounded border p-1 transition-all duration-500',
                    visible ? 'border-amber-500/40 bg-amber-500/5' : 'border-border/30 bg-muted/20 opacity-30',
                  ].join(' ')}
                >
                  <div
                    className="w-full rounded-sm bg-amber-500/70 transition-all duration-500"
                    style={{ height: `${visible ? heightPct : 0}%` }}
                  />
                  <span className="mt-0.5 text-center font-mono text-[9px] text-amber-700 dark:text-amber-400">
                    {visible ? l.toFixed(2) : ''}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Row 4 — mean = the training loss */}
        <div
          className={[
            'flex items-center justify-between rounded-md border px-3 py-2 transition-colors duration-500',
            step >= 3 ? 'border-amber-500/50 bg-amber-500/10' : 'border-border/40 bg-muted/30 opacity-50',
          ].join(' ')}
        >
          <div className="font-mono text-[12px] text-foreground/95">{copy.meanRow}</div>
          <div className="font-mono text-[14px] text-amber-700 dark:text-amber-400">
            = {step >= 3 ? meanLoss.toFixed(3) : '— —'}
          </div>
        </div>
      </div>
    </DiagramFrame>
  );
}

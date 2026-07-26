import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';
import { TopKBars, cleanupTokenText } from '../inspector/TopKBars';
import { DiagramFrame, useStepPlayer } from '../motion';

/**
 * Chapter 1 (overview) — the autoregressive generation loop, animated.
 *
 * Self-contained: NO worker, NO model, NO WASM. The whole point of this widget
 * is to be robust as the very first thing a learner sees, while the real model
 * may still be downloading. It replays a short SCRIPTED generation so the loop
 * — forward pass → sample one token → append → repeat — is visible end to end.
 *
 * A pseudocode block highlights the line matching the current phase; the
 * candidate bars (reused TopKBars) show "the model scored every token, we kept
 * one"; the token strip grows as each sampled token is appended.
 *
 * Animation model: `useStepPlayer` + `DiagramFrame` from `learn/motion`, in
 * place of this file's former hand-rolled `setInterval`, its own Play/Pause
 * button, and its own card shell. Frame count (12) and pace (1150ms) are
 * unchanged — this was a refactor, not a retune. Three things the swap fixes:
 * reduced-motion readers used to mount paused on frame 0 with NO Step button,
 * i.e. stranded on "forward pass, nothing sampled yet"; the local `useState`
 * initializer could not see the OS preference flipping mid-session; and the
 * caption's `aria-live` toggled between `off` and `polite` on the same commit
 * that changed its text, which is not reliably announced. `DiagramFrame` owns
 * all three now (a separate `sr-only` region, filled only on reader intent).
 *
 * There is no `<svg>` here — the diagram body is divs — so this widget takes
 * the motion kit only, and none of the skin's SVG tokens apply.
 */

const PROMPT = ['The', ' cat', ' sat', ' on', ' the'];

type Cand = { text: string; prob: number };
type Step = { cands: Cand[] };

// Each step's candidates are sorted; index 0 is the sampled (greedy) token.
const STEPS: Step[] = [
  {
    cands: [
      { text: ' floor', prob: 0.42 },
      { text: ' mat', prob: 0.19 },
      { text: ' rug', prob: 0.12 },
      { text: ' couch', prob: 0.08 },
      { text: ' bed', prob: 0.05 },
    ],
  },
  {
    cands: [
      { text: '.', prob: 0.54 },
      { text: ',', prob: 0.18 },
      { text: ' and', prob: 0.1 },
      { text: ' beside', prob: 0.05 },
      { text: ' near', prob: 0.04 },
    ],
  },
  {
    cands: [
      { text: ' It', prob: 0.31 },
      { text: ' The', prob: 0.17 },
      { text: ' A', prob: 0.09 },
      { text: ' She', prob: 0.07 },
      { text: ' Then', prob: 0.05 },
    ],
  },
  {
    cands: [
      { text: ' purred', prob: 0.27 },
      { text: ' slept', prob: 0.19 },
      { text: ' was', prob: 0.12 },
      { text: ' sat', prob: 0.08 },
      { text: ' yawned', prob: 0.06 },
    ],
  },
];

type Phase = 'forward' | 'sample' | 'append';
const PHASES: Phase[] = ['forward', 'sample', 'append'];
const TOTAL_FRAMES = STEPS.length * PHASES.length;
const FRAME_MS = 1150;

// Pseudocode stays English in both locales; only the trailing `//` comments
// localize (looked up per phase from COPY).
const LINES: { code: string; phase: Phase | null }[] = [
  { code: 'tokens = tokenize(prompt)', phase: null },
  { code: 'while (!done) {', phase: null },
  { code: '  logits = model(tokens)', phase: 'forward' },
  { code: '  next   = sample(logits)', phase: 'sample' },
  { code: '  tokens.push(next)', phase: 'append' },
  { code: '}', phase: null },
];

// Play / Pause are NOT here any more: those verbs now belong to the shared
// `StepControls` (rendered by `DiagramFrame`), which carries its own en/zh pair
// so every animated widget in the course says the same word.
const COPY = {
  en: {
    title: 'The generation loop',
    lineComments: {
      forward: 'one forward pass',
      sample: 'pick one token',
      append: 'append, then repeat',
    },
    phaseCaption: {
      forward: 'Forward pass → a score for every token',
      sample: 'Sample → keep one (greedy = the top bar)',
      append: 'Append the kept token, then loop',
    },
    footnote: "Illustrative numbers, scripted to show the loop's shape — not live output from the model.",
    outro: (
      <>
        An LLM is a function from a list of tokens to a score for <em>every</em> token in its vocabulary. To write more
        than one token it runs in a loop: forward pass → sample one token → append it → run again on the longer list.
        That single repeated step is all "generating text" is.
      </>
    ),
  },
  zh: {
    title: '生成循环',
    lineComments: {
      forward: '一次前向传播',
      sample: '挑出一个 token',
      append: '追加，然后重复',
    },
    phaseCaption: {
      forward: '前向传播 → 为每个 token 打一个分',
      sample: '采样 → 留下一个（贪心 = 最高的那根条）',
      append: '追加留下的 token，然后循环',
    },
    footnote: '数字为示意，按脚本演示循环的形状——不是模型的实时输出。',
    outro: (
      <>
        LLM 是一个函数：输入一串 token，为词表中的<em>每一个</em> token 输出一个分数。要写出不止一个
        token，它就放进循环里跑：前向传播 → 采样一个 token → 追加 →
        在变长了的列表上再跑一遍。“生成文本”的全部含义，就是这一个反复执行的步骤。
      </>
    ),
  },
} as const;

export function GenerationLoop() {
  const locale = useLocale();
  const copy = COPY[locale];
  const player = useStepPlayer(TOTAL_FRAMES, { frameMs: FRAME_MS });
  const { frame } = player;

  const step = Math.floor(frame / PHASES.length);
  const phase = PHASES[frame % PHASES.length];
  const cands = STEPS[step].cands;
  const pickedIndex = 0; // index 0 is always the sampled token in the script

  // Tokens appended so far: before this step's append we have `step` tokens;
  // during/after the append phase the new one is included (and flashes).
  const appendedCount = phase === 'append' ? step + 1 : step;
  const generated = STEPS.slice(0, appendedCount).map((s) => s.cands[0].text);
  const flashIndex = phase === 'append' ? step : -1;

  const ids = cands.map((_, i) => i);
  const probs = cands.map((c) => c.prob);
  const texts = cands.map((c) => c.text);
  // Highlight the kept token only once we've "sampled" it.
  const sampledTokenId = phase === 'forward' ? -1 : pickedIndex;

  // Every caption this loop can reach. The visible caption is
  // `copy.phaseCaption[phase]` and NOTHING else picks it — `phase` is
  // `PHASES[frame % PHASES.length]`, so the three entries of PHASES are the
  // complete set of arms. Stacking all three lets DiagramFrame reserve the
  // tallest, so the chapter body cannot hop as the loop cycles (the zh
  // "采样 → 留下一个（贪心 = 最高的那根条）" line wraps a row earlier than the
  // other two on a phone).
  const captions: React.ReactNode[] = PHASES.map((p) => copy.phaseCaption[p]);

  return (
    <>
      <DiagramFrame
        title={copy.title}
        player={player}
        locale={locale}
        caption={copy.phaseCaption[phase]}
        captions={captions}
        note={copy.footnote}
      >
        {/* Token strip — grows as tokens are appended. */}
        <div className="flex flex-wrap items-center gap-1 rounded-md border border-border/60 bg-muted/30 p-2">
          {PROMPT.map((t, i) => (
            <span
              key={`p-${i}`}
              className="rounded bg-background px-1.5 py-0.5 font-mono text-[11px] text-muted-foreground"
            >
              {cleanupTokenText(t)}
            </span>
          ))}
          {generated.map((t, i) => (
            <span
              key={`g-${i}`}
              className={[
                'rounded px-1.5 py-0.5 font-mono text-[11px] transition-colors',
                i === flashIndex
                  ? 'bg-primary/20 text-primary ring-1 ring-primary/40'
                  : 'bg-primary/10 text-foreground/90',
              ].join(' ')}
            >
              {cleanupTokenText(t)}
            </span>
          ))}
          <span className="ml-0.5 font-mono text-[11px] text-muted-foreground" aria-hidden>
            ▮
          </span>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          {/* Pseudocode with the active line highlighted. */}
          <pre className="overflow-x-auto rounded-md border border-border/60 bg-muted/30 p-2 font-mono text-[11px] leading-relaxed">
            {LINES.map((ln, i) => {
              const active = ln.phase !== null && ln.phase === phase;
              return (
                <div
                  key={i}
                  className={[
                    'rounded px-1 transition-colors',
                    active ? 'bg-primary/15 text-foreground' : 'text-foreground/70',
                  ].join(' ')}
                >
                  <span>{ln.code}</span>
                  {ln.phase ? (
                    <span className="text-muted-foreground"> {`// ${copy.lineComments[ln.phase]}`}</span>
                  ) : null}
                </div>
              );
            })}
          </pre>

          {/* Candidate distribution for this step. The phase caption that used
              to sit above these bars is now the frame's caption — one sentence,
              one accessible copy, in the slot the frame reserves for it. */}
          <TopKBars ids={ids} probs={probs} texts={texts} sampledTokenId={sampledTokenId} runKey={frame} />
        </div>
      </DiagramFrame>

      {/* The chapter's takeaway, not a diagram footnote: it stays body copy at
          body weight, outside the frame, rather than being folded into `note`
          (11px muted) alongside the illustrative-numbers disclaimer. */}
      <p className="text-[12px] text-foreground/85">{copy.outro}</p>
    </>
  );
}

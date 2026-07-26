import * as React from 'react';

import { Button } from '../../components/ui/button';
import { useLocale } from '../../lib/i18n-react';
import { DiagramFrame, useStepPlayer } from '../motion';

/**
 * BPE merge walkthrough — a stepped, illustrative animation that shows how
 * Byte-Pair Encoding builds a token list for a fixed input by repeatedly
 * merging adjacent fragments. The fixed input "unbelievable" keeps the
 * sequence deterministic so the lesson is contained.
 *
 * The merges below are *illustrative*, not the literal Qwen3 merge order —
 * the widget says so explicitly. The teaching goal is "BPE builds tokens by
 * merging adjacent pairs", not "memorise Qwen3's vocabulary".
 *
 * Animation model: `useStepPlayer` + `DiagramFrame` from `learn/motion`,
 * replacing this file's former hand-rolled `setInterval` + `autoplay` flag and
 * its own header row, play/next buttons, caption box and footnote. Same eight
 * beats, same 1200ms pace, still a NON-LOOPING sweep — the old effect called
 * `setAutoplay(false)` on the last step instead of wrapping, which is
 * `loop: false` — and still PARKED AT STEP 1 ON MOUNT, which is `initialPlaying:
 * false`. That last one is the pairing that matters: `loop: false` plus autoplay
 * means the sweep runs once and then sits on the final split forever, so the
 * byte-level start this walkthrough exists to explain is the one frame a reader
 * who arrives a few seconds late never gets to see. Pace, beat count, looping
 * and autoplay are therefore all exactly what they were before the conversion.
 * The one behaviour the kit genuinely ADDS is reduced-motion retargeting, and it
 * pins those readers on the LAST beat (`restFrame` defaults to `total - 1`);
 * Prev is how they walk back to the start, which is one more reason it stays.
 *
 * There is no `<svg>` here at all, so this is a KIT-ONLY adoption — none of the
 * skin's drawing tokens (RX / SW / DASH / PanelFrame …) have anything to apply
 * to, `PanBox` has nothing to pan (the fragment list is real wrapping markup,
 * so it reflows on a phone instead of scaling), and `CHIP_PALETTE` below is
 * deliberately left alone: it mirrors chapter 1's chips, so collapsing it onto
 * the skin's two hues would desync this widget from the rest of the chapter it
 * sits in.
 *
 * `StepControls` renders Play and Step (Step is the old "Next"); its Reset is
 * opt-in via `showReset`, which this widget does not pass, so Reset appears here
 * only under reduced motion, where the kit swaps it in for the hidden Play. That
 * leaves two affordances with no slot — Prev, and the "step N of 8" counter,
 * both here before the conversion — and they stay widget-owned in a row above
 * the fragment list. Prev goes through the raw `player.goTo` rather than
 * `DiagramFrame`'s wrapped copy, but it still announces: the frame's
 * `onClickCapture` treats a click on any `<button>` inside it as the reader
 * asking, and `Button` renders a real one. `goTo` also pauses, so the live
 * region is filled by the time the caption changes.
 */

// Chip palette mirrors chapter 1's CHIP_PALETTE so the visual language is
// consistent across the chapter.
const CHIP_PALETTE = [
  'bg-sky-100 dark:bg-sky-950/40 text-sky-900 dark:text-sky-100',
  'bg-amber-100 dark:bg-amber-950/40 text-amber-900 dark:text-amber-100',
  'bg-emerald-100 dark:bg-emerald-950/40 text-emerald-900 dark:text-emerald-100',
  'bg-rose-100 dark:bg-rose-950/40 text-rose-900 dark:text-rose-100',
  'bg-violet-100 dark:bg-violet-950/40 text-violet-900 dark:text-violet-100',
];

type Step = {
  /** Fragments after this merge step has been applied. */
  fragments: string[];
};

// Each step is the *result* after a merge. Step 0 is the byte-level start.
// The per-step prose (description + "why this pair" corpus-count note) lives
// in COPY below, aligned by index. The corpus counts are illustrative (like
// the merge order itself) but strictly decreasing, mirroring how greedy BPE
// always takes the current most-frequent adjacent pair.
const STEPS: Step[] = [
  { fragments: ['u', 'n', 'b', 'e', 'l', 'i', 'e', 'v', 'a', 'b', 'l', 'e'] },
  { fragments: ['u', 'n', 'b', 'el', 'i', 'e', 'v', 'a', 'b', 'l', 'e'] },
  { fragments: ['u', 'n', 'b', 'el', 'i', 'ev', 'a', 'b', 'l', 'e'] },
  { fragments: ['u', 'n', 'b', 'el', 'i', 'ev', 'a', 'b', 'le'] },
  { fragments: ['u', 'n', 'b', 'el', 'i', 'ev', 'a', 'ble'] },
  { fragments: ['un', 'b', 'el', 'i', 'ev', 'a', 'ble'] },
  { fragments: ['unb', 'el', 'i', 'ev', 'a', 'ble'] },
  { fragments: ['un', 'bel', 'iev', 'able'] },
];

type StepCopy = {
  /** A short sentence describing what just happened. */
  description: string;
  /**
   * The "why this pair" line: illustrative corpus counts that make the
   * greedy most-frequent rule visible. Null on steps without a merge.
   */
  countNote: string | null;
};

const COPY = {
  en: {
    headerTitle: 'BPE merge walkthrough · input',
    stepOf: (n: number, total: number) => `Step ${n} of ${total}`,
    fragListAria: 'Current fragment list',
    // Next / Play / Pause are NOT here any more: those verbs belong to the
    // shared `StepControls`, which carries its own en/zh pair so every animated
    // widget in the course says the same word. Replay is gone outright — the
    // kit's Play does that job, because `toggle` rewinds a non-looping sweep
    // that is parked on the last frame instead of starting a dead interval.
    // Prev has no slot there, so it stays.
    prev: 'Prev',
    fragmentCount: (n: number) => `${n} fragment${n === 1 ? '' : 's'}`,
    footnote:
      'Note: this merge order is illustrative — it teaches the pattern of merging adjacent pairs. The real Qwen3 vocabulary was learned from a massive training corpus and its merges differ in both order and final token boundaries.',
    steps: [
      {
        description: 'Start at the byte level. Every character is its own fragment — 12 in total.',
        countNote:
          'BPE training counts every adjacent pair across the whole corpus, then always merges the most frequent one.',
      },
      {
        description: 'Merge the adjacent pair e+l → el.',
        countNote:
          "Why this pair? Illustrative corpus counts: e+l ×512, e+v ×410, l+e ×395 — 'el' appears most often among adjacent pairs here, so the greedy rule merges it first.",
      },
      {
        description: 'Merge the adjacent pair e+v → ev.',
        countNote: 'e+v (×410) is now the most frequent remaining pair in the corpus counts — merge it.',
      },
      {
        description: 'Merge the adjacent pair l+e → le.',
        countNote: 'l+e (×395) tops the remaining counts — merge it.',
      },
      {
        description: 'Merge the adjacent pair b+le → ble.',
        countNote: 'b+le (×360) is the new winner — merges can chain on top of earlier merges.',
      },
      {
        description: 'Merge the adjacent pair u+n → un.',
        countNote: 'u+n (×340) tops the remaining counts — merge it.',
      },
      {
        description: 'Merge the adjacent pair un+b → unb.',
        countNote: 'un+b (×300) tops the remaining counts — merge it.',
      },
      {
        description: 'Final landing — a plausible vocabulary entry sequence. The actual Qwen3 split may differ.',
        countNote:
          'Many merge rounds later, the counts have carved the word into common sub-word chunks. The merge order IS the frequency order.',
      },
    ] as readonly StepCopy[],
  },
  zh: {
    headerTitle: 'BPE 合并演练 · 输入',
    stepOf: (n: number, total: number) => `第 ${n} / ${total} 步`,
    fragListAria: '当前片段列表',
    prev: '上一步',
    fragmentCount: (n: number) => `${n} 个片段`,
    footnote:
      '注意：这里的合并顺序仅为示意——它教的是“合并相邻对”这一模式。真实的 Qwen3 词表是从海量训练语料中学出来的，其合并在顺序和最终 token 边界上都会不同。',
    steps: [
      {
        description: '从字节级开始。每个字符都是独立的片段——共 12 个。',
        countNote: 'BPE 训练会统计整个语料库中每个相邻片段对的出现次数，然后永远合并最频繁的那一对。',
      },
      {
        description: '合并相邻的一对 e+l → el。',
        countNote:
          '为什么是这一对？示意性的语料计数：e+l ×512、e+v ×410、l+e ×395——在这里的相邻对中 “el” 出现最多，所以贪心规则先合并它。',
      },
      {
        description: '合并相邻的一对 e+v → ev。',
        countNote: 'e+v（×410）现在是语料计数中剩余最频繁的一对——合并它。',
      },
      {
        description: '合并相邻的一对 l+e → le。',
        countNote: 'l+e（×395）在剩余计数中居首——合并它。',
      },
      {
        description: '合并相邻的一对 b+le → ble。',
        countNote: 'b+le（×360）是新的第一名——合并可以叠在更早的合并之上继续链式进行。',
      },
      {
        description: '合并相邻的一对 u+n → un。',
        countNote: 'u+n（×340）在剩余计数中居首——合并它。',
      },
      {
        description: '合并相邻的一对 un+b → unb。',
        countNote: 'un+b（×300）在剩余计数中居首——合并它。',
      },
      {
        description: '最终落点——一个可信的词表条目序列。真实的 Qwen3 切分可能不同。',
        countNote: '再经过许多轮合并，计数已经把这个词雕成常见的子词块。合并顺序就是频率顺序。',
      },
    ] as readonly StepCopy[],
  },
} as const;

const FRAME_MS = 1200;

/**
 * One beat's caption: the sentence, with the smaller "why this pair" corpus
 * note under it. Both now live in the frame's caption slot rather than in their
 * own box, because that slot is the only place with a height reservation — it
 * stacks all eight beats in one grid cell and takes the tallest, so the chapter
 * body cannot hop as a longer sentence comes round. (The `countNote` lines are
 * what vary most — one is a single short clause, another runs to three.)
 *
 * `block` spans and not `<div>`s: the frame renders this inside a `<p>`.
 * `text-foreground/85` keeps the description a tier above the note, which is
 * exactly the relationship the two had in the box this replaces.
 */
function stepCaption(sc: StepCopy): React.ReactNode {
  return (
    <>
      <span className="block text-foreground/85">{sc.description}</span>
      {sc.countNote ? <span className="mt-1 block text-[11px]">{sc.countNote}</span> : null}
    </>
  );
}

export function BpeMergeWalkthrough() {
  const locale = useLocale();
  const copy = COPY[locale];
  // `initialPlaying: false` is load-bearing, not taste, and it is `loop: false`
  // that makes it so. A non-looping sweep that autoplays runs its eight beats
  // once and then parks on the FINAL split forever, so step 1 — "every
  // character is its own fragment", the premise the whole walkthrough builds
  // on — is the one beat a reader who scrolls down a few seconds late can
  // never get back to on their own. Parked at step 1 instead, the sweep waits
  // for Play and the reader starts where the lesson does. This is also the
  // pre-conversion behaviour: `autoplay` was `useState(false)`.
  //
  // NOT what this flag does: reduced-motion readers. `useStepPlayer` retargets
  // them to `restFrame`, which defaults to the LAST frame, so they still open
  // on the final split — they reach the start with Prev below, which is why
  // that button matters more here than the autoplay setting does.
  const player = useStepPlayer(STEPS.length, { frameMs: FRAME_MS, loop: false, initialPlaying: false });
  const { frame } = player;

  const step = STEPS[frame]!;

  // Every caption this sweep can reach, so DiagramFrame reserves the tallest.
  // The visible caption is `stepCaption(copy.steps[frame])` and `frame` is
  // clamped to [0, STEPS.length) by the player, so mapping the whole `steps`
  // array covers every arm by construction — there is no branch to miss, and
  // both locales get their own set because it is built from `copy`.
  const captions = copy.steps.map(stepCaption);
  // Indexing the same array the sizers come from is the invariant, not a
  // shortcut: the visible caption cannot be one the slot was not sized for.
  const caption = captions[frame]!;

  return (
    <DiagramFrame
      title={
        // The tokenised input is a mono chip, not prose — it is the literal
        // string this whole walkthrough merges, and at HEAD it rendered as
        // `<code>` inside a header div with exactly these classes. `title`
        // takes a node, so this is the pre-conversion markup unchanged.
        <>
          {copy.headerTitle} <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">unbelievable</code>
        </>
      }
      player={player}
      locale={locale}
      caption={caption}
      captions={captions}
      note={copy.footnote}
    >
      {/* The body keeps its own tighter rhythm; DiagramFrame's `space-y-4` is
          for the frame's own parts. */}
      <div className="space-y-3">
        {/* Prev and the step counter — neither is a control StepControls owns,
            so they stay with the widget, in a row directly above the body. */}
        <div className="flex flex-wrap items-center gap-2">
          {/* `size="sm"` is `h-8` — 32px, under the 44px touch floor this course
              holds every control to. The class is the floor, not decoration. */}
          <Button
            size="sm"
            variant="outline"
            className="max-sm:min-h-[44px]"
            onClick={() => player.goTo(frame - 1)}
            disabled={frame === 0}
          >
            {copy.prev}
          </Button>
          <span className="text-[11px] text-muted-foreground">{copy.stepOf(frame + 1, STEPS.length)}</span>
          <span className="ml-auto text-[11px] text-muted-foreground">
            {copy.fragmentCount(step.fragments.length)}
          </span>
        </div>

        <div
          role="list"
          aria-label={copy.fragListAria}
          className="flex flex-wrap gap-1.5 rounded-md border border-border/60 bg-muted/30 p-3 min-h-[3.5rem]"
        >
          {step.fragments.map((frag, i) => {
            const palette = CHIP_PALETTE[i % CHIP_PALETTE.length]!;
            return (
              <span
                key={`${frame}-${i}-${frag}`}
                role="listitem"
                className={[
                  'inline-flex items-center rounded px-1.5 py-1 text-[13px] font-mono leading-none border border-transparent',
                  palette,
                ].join(' ')}
              >
                {frag}
              </span>
            );
          })}
        </div>
      </div>
    </DiagramFrame>
  );
}

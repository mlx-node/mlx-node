import * as React from 'react';

import { Button } from '../../components/ui/button';
import { useLocale } from '../../lib/i18n-react';

/**
 * BPE merge walkthrough — a stepped, illustrative animation that shows how
 * Byte-Pair Encoding builds a token list for a fixed input by repeatedly
 * merging adjacent fragments. The fixed input "unbelievable" keeps the
 * sequence deterministic so the lesson is contained.
 *
 * The merges below are *illustrative*, not the literal Qwen3 merge order —
 * the widget says so explicitly. The teaching goal is "BPE builds tokens by
 * merging adjacent pairs", not "memorise Qwen3's vocabulary".
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
    prev: 'Prev',
    next: 'Next',
    pause: 'Pause',
    replay: 'Replay',
    play: 'Play',
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
    next: '下一步',
    pause: '暂停',
    replay: '重播',
    play: '播放',
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

const AUTOPLAY_MS = 1200;

export function BpeMergeWalkthrough() {
  const copy = COPY[useLocale()];
  const [stepIdx, setStepIdx] = React.useState(0);
  const [autoplay, setAutoplay] = React.useState(false);

  React.useEffect(() => {
    if (!autoplay) return undefined;
    const id = window.setInterval(() => {
      setStepIdx((s) => {
        if (s + 1 >= STEPS.length) {
          setAutoplay(false);
          return s;
        }
        return s + 1;
      });
    }, AUTOPLAY_MS);
    return () => window.clearInterval(id);
  }, [autoplay]);

  const safeIdx = Math.max(0, Math.min(stepIdx, STEPS.length - 1));
  const step = STEPS[safeIdx]!;
  const stepCopy = copy.steps[safeIdx]!;
  const atEnd = safeIdx >= STEPS.length - 1;

  function go(next: number) {
    setAutoplay(false);
    setStepIdx(Math.max(0, Math.min(STEPS.length - 1, next)));
  }

  function togglePlay() {
    if (atEnd) {
      setStepIdx(0);
      setAutoplay(true);
    } else {
      setAutoplay((p) => !p);
    }
  }

  return (
    <div className="space-y-3 rounded-md border border-border bg-background p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          {copy.headerTitle}{' '}
          <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">unbelievable</code>
        </div>
        <div className="text-[11px] text-muted-foreground">{copy.stepOf(safeIdx + 1, STEPS.length)}</div>
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
              key={`${safeIdx}-${i}-${frag}`}
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

      <div className="space-y-1 rounded-md bg-muted/40 px-3 py-2 text-xs text-foreground/85">
        <div>{stepCopy.description}</div>
        {stepCopy.countNote ? <div className="text-[11px] text-muted-foreground">{stepCopy.countNote}</div> : null}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" variant="outline" onClick={() => go(safeIdx - 1)} disabled={safeIdx === 0}>
          {copy.prev}
        </Button>
        <Button size="sm" onClick={togglePlay}>
          {autoplay ? copy.pause : atEnd ? copy.replay : copy.play}
        </Button>
        <Button size="sm" variant="outline" onClick={() => go(safeIdx + 1)} disabled={atEnd}>
          {copy.next}
        </Button>
        <span className="ml-2 text-[11px] text-muted-foreground">{copy.fragmentCount(step.fragments.length)}</span>
      </div>

      <p className="text-[11px] text-muted-foreground">{copy.footnote}</p>
    </div>
  );
}

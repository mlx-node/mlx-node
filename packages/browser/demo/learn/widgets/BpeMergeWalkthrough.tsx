import * as React from "react";

import { Button } from "../../components/ui/button";

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
  "bg-sky-100 dark:bg-sky-950/40 text-sky-900 dark:text-sky-100",
  "bg-amber-100 dark:bg-amber-950/40 text-amber-900 dark:text-amber-100",
  "bg-emerald-100 dark:bg-emerald-950/40 text-emerald-900 dark:text-emerald-100",
  "bg-rose-100 dark:bg-rose-950/40 text-rose-900 dark:text-rose-100",
  "bg-violet-100 dark:bg-violet-950/40 text-violet-900 dark:text-violet-100",
];

type Step = {
  /** Fragments after this merge step has been applied. */
  fragments: string[];
  /** A short sentence describing what just happened. */
  description: string;
};

// Each step is the *result* after a merge. Step 0 is the byte-level start.
const STEPS: Step[] = [
  {
    fragments: ["u", "n", "b", "e", "l", "i", "e", "v", "a", "b", "l", "e"],
    description:
      "Start at the byte level. Every character is its own fragment — 12 in total.",
  },
  {
    fragments: ["u", "n", "b", "el", "i", "e", "v", "a", "b", "l", "e"],
    description: "Merge the adjacent pair e+l → el.",
  },
  {
    fragments: ["u", "n", "b", "el", "i", "ev", "a", "b", "l", "e"],
    description: "Merge the adjacent pair e+v → ev.",
  },
  {
    fragments: ["u", "n", "b", "el", "i", "ev", "a", "b", "le"],
    description: "Merge the adjacent pair l+e → le.",
  },
  {
    fragments: ["u", "n", "b", "el", "i", "ev", "a", "ble"],
    description: "Merge the adjacent pair b+le → ble.",
  },
  {
    fragments: ["un", "b", "el", "i", "ev", "a", "ble"],
    description: "Merge the adjacent pair u+n → un.",
  },
  {
    fragments: ["unb", "el", "i", "ev", "a", "ble"],
    description: "Merge the adjacent pair un+b → unb.",
  },
  {
    fragments: ["un", "bel", "iev", "able"],
    description:
      "Final landing — a plausible vocabulary entry sequence. The actual Qwen3 split may differ.",
  },
];

const AUTOPLAY_MS = 1200;

export function BpeMergeWalkthrough() {
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
          BPE merge walkthrough · input{" "}
          <code className="rounded bg-muted px-1 py-0.5 font-mono text-[11px]">
            unbelievable
          </code>
        </div>
        <div className="text-[11px] text-muted-foreground">
          Step {safeIdx + 1} of {STEPS.length}
        </div>
      </div>

      <div
        role="list"
        aria-label="Current fragment list"
        className="flex flex-wrap gap-1.5 rounded-md border border-border/60 bg-muted/30 p-3 min-h-[3.5rem]"
      >
        {step.fragments.map((frag, i) => {
          const palette = CHIP_PALETTE[i % CHIP_PALETTE.length]!;
          return (
            <span
              key={`${safeIdx}-${i}-${frag}`}
              role="listitem"
              className={[
                "inline-flex items-center rounded px-1.5 py-1 text-[13px] font-mono leading-none border border-transparent",
                palette,
              ].join(" ")}
            >
              {frag}
            </span>
          );
        })}
      </div>

      <div className="rounded-md bg-muted/40 px-3 py-2 text-xs text-foreground/85">
        {step.description}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          variant="outline"
          onClick={() => go(safeIdx - 1)}
          disabled={safeIdx === 0}
        >
          Prev
        </Button>
        <Button size="sm" onClick={togglePlay}>
          {autoplay ? "Pause" : atEnd ? "Replay" : "Play"}
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={() => go(safeIdx + 1)}
          disabled={atEnd}
        >
          Next
        </Button>
        <span className="ml-2 text-[11px] text-muted-foreground">
          {step.fragments.length} fragment
          {step.fragments.length === 1 ? "" : "s"}
        </span>
      </div>

      <p className="text-[11px] text-muted-foreground">
        Note: this merge order is illustrative — it teaches the pattern of
        merging adjacent pairs. The real Qwen3 vocabulary was learned from a
        massive training corpus and its merges differ in both order and final
        token boundaries.
      </p>
    </div>
  );
}

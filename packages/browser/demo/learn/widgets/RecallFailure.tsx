import * as React from 'react';

import { TopKBars } from '../inspector/TopKBars';

/**
 * Chapter 12 (KV cache & hybrid attention) — the full-vs-linear recall
 * trade-off as a "needle in a haystack." Self-contained: NO worker, NO model,
 * NO WASM. Two synthetic next-token distributions over the same candidate set
 * (the buried "needle" token plus a few distractors), shown side by side:
 *
 *   - Full attention recalls the EXACT needle: a sharp spike on the right
 *     token, near-zero mass everywhere else.
 *   - GatedDeltaNet (linear) recalls APPROXIMATELY: its fixed-size recurrent
 *     state has mixed the whole history together, so the mass is smeared
 *     across the needle and several distractors — the needle is at best
 *     slightly highest, never dominant.
 *
 * This is exactly why Qwen3.5 keeps 6 full-attention layers for the
 * "find that one specific token" jobs. The distributions are hand-picked to
 * show the shape, not live output from the model.
 */

// The unique token buried in the long context. The query asks to recall it.
const NEEDLE = 'X7-Q';

// Candidate next tokens: the needle plus plausible distractors that the
// linear layer's blurred state confuses it with.
const CANDIDATES: ReadonlyArray<string> = [NEEDLE, 'X7-G', 'K3-Q', 'X1-Q', 'B7-Q'];
const NEEDLE_ID = 0;

type Panel = {
  key: 'full' | 'linear';
  label: string;
  // Probability per candidate, aligned to CANDIDATES.
  probs: number[];
  note: string;
};

const PANELS: Panel[] = [
  {
    key: 'full',
    label: 'Full attention',
    // Sharp spike on the exact needle — correct exact recall.
    probs: [0.91, 0.03, 0.025, 0.02, 0.015],
    note: 'Softmax attention scores the needle against every cached key directly, so it can put almost all the mass on the one exact token it was asked to recall.',
  },
  {
    key: 'linear',
    label: 'GatedDeltaNet (linear)',
    // Blurred / flat — the fixed-size state mixed the history together.
    probs: [0.3, 0.2, 0.18, 0.17, 0.15],
    note: 'A fixed-size recurrent state compresses the whole history into one constant-size tensor; here that mixing leaves the needle only barely ahead of look-alike distractors, rather than pinned down.',
  },
];

export function RecallFailure() {
  const ids = CANDIDATES.map((_, i) => i);
  const texts = CANDIDATES.map((t) => t);

  return (
    <div className="not-prose my-4 space-y-3 rounded-md border border-border bg-background p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-xs uppercase tracking-wider text-muted-foreground">
          Needle in a haystack — exact recall
        </div>
        <div className="text-[11px] text-muted-foreground">
          Recall token <span className="font-mono text-foreground/80">{NEEDLE}</span>
        </div>
      </div>

      {/* The scripted scenario: a unique needle buried among filler. */}
      <div className="rounded-md border border-border/60 bg-muted/30 p-2 font-mono text-[12px] leading-relaxed text-foreground/90">
        <span className="text-muted-foreground">… set the access code to </span>
        <span className="rounded bg-primary/15 px-1 text-primary">{NEEDLE}</span>
        <span className="text-muted-foreground">
          {' '}
          before continuing. <span className="opacity-60">[ 4,000 tokens of filler … ]</span> The access code was{' '}
        </span>
        <span className="text-foreground/80">▮</span>
      </div>

      {/* Two side-by-side distributions over the same candidate set. */}
      <div className="grid gap-3 sm:grid-cols-2">
        {PANELS.map((panel, i) => {
          const isFull = panel.key === 'full';
          return (
            <div key={panel.key} className="space-y-1">
              <div className="flex items-center justify-between">
                <span
                  className={[
                    'rounded px-2 py-0.5 text-[11px] font-medium',
                    isFull
                      ? 'bg-amber-500/15 text-amber-700 dark:text-amber-300'
                      : 'bg-violet-500/15 text-violet-700 dark:text-violet-300',
                  ].join(' ')}
                >
                  {panel.label}
                </span>
                <span className="text-[11px] text-muted-foreground">{isFull ? 'sharp recall' : 'blurred recall'}</span>
              </div>
              <TopKBars ids={ids} probs={panel.probs} texts={texts} sampledTokenId={NEEDLE_ID} runKey={i + 1} />
              <p className="text-[11px] leading-relaxed text-muted-foreground">{panel.note}</p>
            </div>
          );
        })}
      </div>

      <p className="text-[12px] text-foreground/85">
        Full-attention layers keep direct, per-token access to every cached key, so they are well suited to pulling out
        one exact token on demand. A fixed-state recurrent layer (GatedDeltaNet) compresses the whole history into a
        constant-size tensor, so under recall pressure it is <em>less reliable</em> at pinning down an arbitrary token —
        as in this illustrative case, where the needle barely leads its look-alikes. Neither is absolute (attention can
        still mis-attend, and a recurrent state isn&apos;t always blurred), but that reliability gap is why Qwen3.5
        keeps <strong>6 full-attention layers</strong> for the &quot;find that one specific token&quot; jobs its {18}
        -layer linear majority handles less precisely.
      </p>

      <p className="text-[10px] text-muted-foreground">
        Illustrative distributions — hand-picked to show the exact-vs-blurred shape, not live output from the model.
      </p>
    </div>
  );
}

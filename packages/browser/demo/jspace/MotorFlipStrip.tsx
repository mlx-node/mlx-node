// MotorFlipStrip — a per-position "commit depth" (motor-flip) strip.
//
// For each prompt position, show the layer at which the readout's top guess
// LOCKS ONTO that position's final (ℓ-max) token — i.e. the lowest displayed
// layer whose argmax already equals the top layer's argmax and never changes
// above it (Task-3's pure `motorFlipLayer`). Cells are coloured by EARLINESS:
// an early commit is hot/bright (viridis 1.0), a late commit is cold/dark. Pure
// presentational over `slice` — no model, no worker, no window/document. Works
// identically for baked starters and live runs.

import { cleanupTokenText } from '../learn/inspector/TopKBars';
import { readableInk, viridis } from '../jlens-core/colors';
import { motorFlipLayer } from '../jlens-core/workspace-metrics';
import type { LensSliceData } from '../jlens-core/types';

/** The `motorFlip` sub-block of `JSPACE_COPY[locale]` — all display strings. */
export type MotorFlipCopy = {
  eyebrow: string;
  hint: string;
};

export function MotorFlipStrip({ slice, copy }: { slice: LensSliceData; copy: MotorFlipCopy }) {
  // Top displayed layer index; the committed token is that row's top guess.
  const top = slice.layers.length - 1;

  return (
    <section aria-label={copy.eyebrow} className="jspace-panel space-y-3 p-4">
      <h3 className="jspace-eyebrow">{copy.eyebrow}</h3>
      <p className="text-[11px] leading-snug text-[color:var(--text-dim)]">{copy.hint}</p>
      <div className="overflow-x-auto rounded-lg bg-background/40 p-2">
        <div className="flex w-max gap-1">
          {Array.from({ length: slice.promptLen }, (_, p) => {
            const flip = motorFlipLayer(slice, p);
            // Defensive: an empty slice yields -1 (won't happen when `slice` is
            // set, but guard so we never index `layers[-1]`).
            if (flip < 0 || top < 0) {
              return (
                <div
                  key={p}
                  className="flex min-w-[3.5rem] shrink-0 flex-col items-center gap-0.5 rounded-md border border-border/50 bg-muted/30 px-2 py-1.5 font-mono text-[color:var(--text-dim)]"
                >
                  <span className="text-[11px]">ℓ—</span>
                  <span className="text-[8px] opacity-70">#{p + 1}</span>
                </div>
              );
            }
            const layer = slice.layers[flip]!;
            // Earliness: earlier flip → higher score → hotter/brighter. When the
            // whole slice is a single layer (top === 0) everything commits "now".
            const score = top > 0 ? 1 - flip / top : 1;
            const bg = viridis(score);
            const ink = readableInk(score);
            const token = cleanupTokenText(slice.cellAt(top, p).topKTexts[0] ?? '');
            const label = `position ${p + 1} commits at ℓ${layer}`;
            return (
              <div
                key={p}
                title={label}
                aria-label={label}
                style={{ backgroundColor: bg, color: ink }}
                className="flex min-w-[3.5rem] shrink-0 flex-col items-center gap-0.5 rounded-md px-2 py-1.5"
              >
                <span className="font-mono text-[11px] font-semibold leading-none">ℓ{layer}</span>
                <span className="max-w-[4.5rem] truncate font-mono text-[10px] leading-tight" title={token}>
                  {token}
                </span>
                <span className="font-mono text-[8px] leading-none opacity-70">#{p + 1}</span>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

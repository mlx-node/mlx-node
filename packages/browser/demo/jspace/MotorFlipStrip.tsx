// MotorFlipStrip — a per-position "commit depth" (motor-flip) strip.
//
// For each prompt position, show the shallowest DISPLAYED layer from which the
// readout's top guess already equals the position's final (ℓ-max) token and
// stays equal at every displayed layer above it (Task-3's pure `motorFlipLayer`).
// Cells are coloured by PHYSICAL layer depth (not array index) so the same layer
// reads identically in baked (sampled) vs live (1..24) slices: a shallow/early
// commit is hot/bright, a deep/late commit is cold/dark.
//
// Honesty: displayed layers are SAMPLED (baked frames skip e.g. ℓ19/21/23), so
// this is stability AMONG the displayed layers, not continuity between them.
// A position whose top guess never holds the output continuously up to the final
// layer has no stable lock (`motorFlipLayer` returns null — this includes an
// early flicker-then-relapse, NOT only a final-layer-only match) and is rendered
// as a neutral "no lock" cell, never as a late commit. Pure presentational over
// `slice` — no model, no worker, no window/document.

import { cleanupTokenText } from '../learn/inspector/TopKBars';
import { readableInk, viridis } from '../jlens-core/colors';
import { motorFlipLayer } from '../jlens-core/workspace-metrics';
import type { LensSliceData } from '../jlens-core/types';

/** The `motorFlip` sub-block of `JSPACE_COPY[locale]` — all display strings. */
export type MotorFlipCopy = {
  eyebrow: string;
  hint: string;
  /** Per-cell label for a real commit (used for both `title` and `aria-label`). */
  cellLabel: (position: number, layer: number) => string;
  /** Per-cell label for the degenerate "no stable lock" case. Takes the 1-based
   *  position so the accessible name names which token it describes. */
  noLock: (position: number) => string;
};

export function MotorFlipStrip({ slice, copy }: { slice: LensSliceData; copy: MotorFlipCopy }) {
  const top = slice.layers.length - 1;
  // A slice with fewer than two layers carries no below-top evidence, so there is
  // nothing meaningful to plot (and `cellAt(top=-1, …)` on a zero-layer slice
  // would throw). `motorFlipLayer` already returns null for every such position;
  // render nothing rather than a strip of guaranteed no-lock cells.
  if (top < 1) return null;
  // Physical depth of the output layer (24 for both baked and live), used to
  // normalise the earliness colour by real layer number rather than array index.
  const topLayer = slice.layers[top] ?? 1;

  return (
    <section aria-label={copy.eyebrow} className="jspace-panel space-y-3 p-4">
      <h3 className="jspace-eyebrow">{copy.eyebrow}</h3>
      <p className="text-[11px] leading-snug text-[color:var(--text-dim)]">{copy.hint}</p>
      <div className="overflow-x-auto rounded-lg bg-background/40 p-2">
        <div className="flex w-max gap-1">
          {Array.from({ length: slice.promptLen }, (_, p) => {
            const token = cleanupTokenText(slice.cellAt(top, p).topKTexts[0] ?? '');
            const flip = motorFlipLayer(slice, p);
            // No stable lock: the top guess never held the output continuously up to
            // the final layer (final-layer-only OR an earlier flicker-then-relapse).
            // Render neutral — NOT as a late (dark) commit. `role="img"` gives the
            // cell a named, non-generic role so the localized alt text is announced.
            if (flip == null) {
              const noLockLabel = copy.noLock(p + 1);
              return (
                <div
                  key={p}
                  role="img"
                  title={noLockLabel}
                  aria-label={noLockLabel}
                  className="flex min-w-[3.5rem] shrink-0 flex-col items-center gap-0.5 rounded-md border border-dashed border-border/60 bg-muted/30 px-2 py-1.5 font-mono text-[color:var(--text-dim)]"
                >
                  <span aria-hidden className="text-[11px] leading-none">
                    —
                  </span>
                  <span aria-hidden className="max-w-[4.5rem] truncate text-[10px] leading-tight" title={token}>
                    {token}
                  </span>
                  <span aria-hidden className="text-[8px] leading-none opacity-70">
                    #{p + 1}
                  </span>
                </div>
              );
            }
            const layer = slice.layers[flip]!;
            // Earliness by PHYSICAL depth: shallow (early) → bright, deep (late) → dark.
            const score = topLayer > 1 ? 1 - (layer - 1) / (topLayer - 1) : 1;
            const bg = viridis(score);
            const ink = readableInk(score);
            const label = copy.cellLabel(p + 1, layer);
            return (
              <div
                key={p}
                role="img"
                title={label}
                aria-label={label}
                style={{ backgroundColor: bg, color: ink }}
                className="flex min-w-[3.5rem] shrink-0 flex-col items-center gap-0.5 rounded-md px-2 py-1.5"
              >
                <span aria-hidden className="font-mono text-[11px] font-semibold leading-none">
                  ℓ{layer}
                </span>
                <span aria-hidden className="max-w-[4.5rem] truncate font-mono text-[10px] leading-tight" title={token}>
                  {token}
                </span>
                <span aria-hidden className="font-mono text-[8px] leading-none opacity-70">
                  #{p + 1}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}

import * as React from 'react';

import { renderTokenDisplay } from '../../inspector/TopKBars';
import type { LensSliceData } from '../../../jlens-core/types';

/**
 * ArgmaxGrid — the money shot. A CSS-grid table of the model's greedy top token
 * (`argmax`) read out of every residual boundary through the final unembedding.
 *
 * Layout:
 *   - Rows = layers, ordered DEEPEST / output-nearest at the TOP (we render
 *     `layers` reversed, so boundary ℓ24 is the top row and the embedding side
 *     is the bottom). Depth therefore increases UPWARD toward the output — the
 *     reader watches gibberish in the middle resolve into the answer as their
 *     eye travels up.
 *   - Columns = prompt positions. The first column is a sticky layer label
 *     ("ℓ24", "ℓ20", …) that stays put while the token columns scroll.
 *   - The FINAL position column (the next-token prediction, evolving by depth)
 *     is emphasized with a tint + accent border — that column is the whole point.
 *
 * A cell whose argmax equals one of the pinned answer tokens gets an inset
 * underline in that pin's colour, so you can literally see the answer light up
 * near the top. Hover or click any cell to drive the LensTooltip.
 *
 * a11y: the grid is a single `role="img"` with a descriptive `aria-label`
 * summarizing it (the course convention for dense SVG/grid views) rather than
 * exposing hundreds of cells to a screen reader; the tooltip carries the detail.
 */

export type CellRef = { layerIdx: number; pos: number };

export function ArgmaxGrid({
  slice,
  colorByPinnedId,
  selected,
  onHover,
  onSelect,
  ariaLabel,
}: {
  slice: LensSliceData;
  /** argmax id → pin accent, for the "answer surfaces" underline. */
  colorByPinnedId: Map<number, string>;
  selected: CellRef | null;
  onHover: (ref: CellRef | null) => void;
  onSelect: (ref: CellRef) => void;
  ariaLabel: string;
}) {
  const { promptLen, tokens } = slice;
  const finalPos = promptLen - 1;
  // Deepest boundary first (top row) → shallowest last (bottom row).
  const rowOrder = slice.layers.map((_, i) => i).reverse();
  const gridCols = `minmax(2.9rem, auto) repeat(${promptLen}, minmax(4.25rem, 1fr))`;

  const finalCol = (pos: number) => pos === finalPos;

  return (
    <div className="overflow-x-auto">
      <div
        role="img"
        aria-label={ariaLabel}
        className="grid min-w-max gap-px rounded-md border border-border bg-border text-[11px]"
        style={{ gridTemplateColumns: gridCols }}
      >
        {/* Corner */}
        <div className="sticky left-0 z-10 bg-background px-1.5 py-1 text-[9px] font-medium uppercase tracking-wider text-muted-foreground">
          ℓ \ pos
        </div>
        {/* Position headers (the prompt tokens) */}
        {tokens.map((t, pos) => (
          <div
            key={`h-${pos}`}
            className={[
              'px-1.5 py-1 text-center font-mono text-foreground/70',
              finalCol(pos) ? 'border-l border-primary/40 bg-primary/10 font-semibold text-primary' : 'bg-muted/40',
            ].join(' ')}
            title={t.text}
          >
            <span className="block truncate">{renderTokenDisplay(t.text)}</span>
          </div>
        ))}

        {/* One row per layer, deepest at the top */}
        {rowOrder.map((layerIdx) => (
          <React.Fragment key={`row-${layerIdx}`}>
            <div className="sticky left-0 z-10 bg-background px-1.5 py-1 font-mono text-muted-foreground">
              ℓ{slice.layers[layerIdx]}
            </div>
            {tokens.map((_t, pos) => {
              const cell = slice.cellAt(layerIdx, pos);
              const text = renderTokenDisplay(cell.topKTexts[0] ?? '');
              const accent = colorByPinnedId.get(cell.argmaxId);
              const isSelected = selected != null && selected.layerIdx === layerIdx && selected.pos === pos;
              return (
                <div
                  key={`c-${layerIdx}-${pos}`}
                  onMouseEnter={() => onHover({ layerIdx, pos })}
                  onMouseLeave={() => onHover(null)}
                  onClick={() => onSelect({ layerIdx, pos })}
                  title={`ℓ${slice.layers[layerIdx]} · pos ${pos + 1}: ${cell.topKTexts[0] ?? ''}`}
                  className={[
                    'cursor-pointer px-1.5 py-2 text-center font-mono transition-colors',
                    finalCol(pos) ? 'border-l border-primary/40 bg-primary/[0.06] text-foreground' : 'bg-card text-foreground/80',
                    isSelected ? 'outline outline-2 -outline-offset-2 outline-primary' : 'hover:bg-muted',
                  ].join(' ')}
                  style={accent ? { boxShadow: `inset 0 -2px 0 0 ${accent}` } : undefined}
                >
                  <span className="block truncate">{text}</span>
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

// ByLayerStrip — a fixed-position cross-section of the lens grid.
//
// At ONE prompt position, list every layer's top-K readout as its own row, so
// you can watch the model's guess evolve DOWN the residual stack. Rows follow
// `displayRowOrder(slice)` (the same orientation as the canvas grid: the
// DEEPEST layer is the FIRST row). This is ≤128 rows, so it stays plain DOM —
// no canvas, no virtualization. Pure presentational: it reads `slice` and
// reports clicks; it owns no worker, no `window`/`document`, no data fetch.

import { cleanupTokenText } from '../learn/inspector/TopKBars';
import type { LensSliceData } from '../jlens-core/types';
import { displayRowOrder, type CellRef } from './ArgmaxGridCanvas';

export function ByLayerStrip({
  slice,
  pos,
  selected,
  onSelect,
}: {
  slice: LensSliceData;
  /** The fixed prompt position (column) this strip cross-sections. */
  pos: number;
  selected: CellRef | null;
  onSelect: (ref: CellRef) => void;
}) {
  // Deepest layer first — the shared grid orientation.
  const rowOrder = displayRowOrder(slice);
  // Human 1-based position + the prompt token that sits there. Whitespace is
  // made visible (`cleanupTokenText`) so an empty/space token stays legible in
  // the header — this strip has no whitespace toggle of its own.
  const promptToken = cleanupTokenText(slice.tokens[pos]?.text ?? '');

  return (
    <section
      aria-label={`By layer at position ${pos + 1}`}
      className="jspace-panel flex flex-col space-y-2 p-4"
    >
      <h3 className="font-mono text-[10px] uppercase tracking-[0.16em] text-[color:var(--text-dim)]">
        By Layer (Pos={pos + 1} {promptToken})
      </h3>
      <div className="min-h-0 flex-1 space-y-0.5 rounded-lg bg-background/40 p-2">
        {rowOrder.map((layerIdx) => {
          const cell = slice.cellAt(layerIdx, pos);
          const isSelected = selected?.pos === pos && selected?.layerIdx === layerIdx;
          return (
            <button
              key={layerIdx}
              type="button"
              aria-pressed={isSelected}
              onClick={() => onSelect({ layerIdx, pos })}
              className={[
                'grid w-full grid-cols-[3rem_minmax(0,1fr)] items-center gap-2 rounded px-1.5 py-1 text-left text-[12px]',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                isSelected ? 'bg-primary/10' : 'hover:bg-muted',
              ].join(' ')}
            >
              <span className="font-mono text-muted-foreground">ℓ{slice.layers[layerIdx]}</span>
              <span className="flex flex-wrap gap-x-1.5 gap-y-0.5 font-mono">
                {cell.topKTexts.map((text, i) => (
                  <span
                    key={i}
                    className={i === 0 ? 'font-semibold text-foreground' : 'text-foreground/55'}
                  >
                    {cleanupTokenText(text)}
                  </span>
                ))}
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

// ByPosStrip — a fixed-layer cross-section of the lens grid.
//
// At ONE residual-stream boundary (layer), list every prompt position as its
// own row, each showing the prompt token beside that cell's top-K readout, so
// you can scan ACROSS the sequence at a single depth. Positions run `0..P-1`
// (P ≤ 128), all rendered — no virtualization. Pure presentational: it reads
// `slice` and reports clicks; it owns no worker, no `window`/`document`, no
// data fetch.

import { cleanupTokenText } from '../learn/inspector/TopKBars';
import type { LensSliceData } from '../jlens-core/types';
import type { CellRef } from './ArgmaxGridCanvas';

export function ByPosStrip({
  slice,
  layerIdx,
  selected,
  onSelect,
}: {
  slice: LensSliceData;
  /** Index into `slice.layers` — the fixed depth this strip cross-sections. */
  layerIdx: number;
  selected: CellRef | null;
  onSelect: (ref: CellRef) => void;
}) {
  return (
    <section aria-label={`By position at layer ${slice.layers[layerIdx]}`} className="space-y-1">
      <h3 className="font-mono text-[12px] font-semibold text-foreground/80">
        By Pos (Layer={slice.layers[layerIdx]})
      </h3>
      <div className="space-y-0.5 rounded-md border border-border bg-background p-2">
        {Array.from({ length: slice.promptLen }, (_, pos) => {
          const cell = slice.cellAt(layerIdx, pos);
          const isSelected = selected?.layerIdx === layerIdx && selected?.pos === pos;
          // Whitespace made visible so a space/empty prompt token stays legible;
          // this strip has no whitespace toggle of its own.
          const promptToken = cleanupTokenText(slice.tokens[pos]?.text ?? '');
          return (
            <button
              key={pos}
              type="button"
              aria-pressed={isSelected}
              onClick={() => onSelect({ layerIdx, pos })}
              className={[
                'grid w-full grid-cols-[2.5rem_6rem_minmax(0,1fr)] items-center gap-2 rounded px-1.5 py-1 text-left text-[12px]',
                'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                isSelected ? 'bg-primary/10' : 'hover:bg-muted',
              ].join(' ')}
            >
              <span className="font-mono text-muted-foreground">#{pos + 1}</span>
              <span className="truncate font-mono text-foreground/70" title={promptToken}>
                {promptToken}
              </span>
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

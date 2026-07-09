import * as React from 'react';

import { renderTokenDisplay } from '../../inspector/TopKBars';
import { RANK_CAP, SURFACE_RANK } from './colors';
import type { LensSliceData } from './types';

/**
 * PinChips — one chip per pinned answer token, colour-matched to its RankHeatmap
 * column. Each chip reports, over the FINAL-position rank track (the same series
 * the heatmap draws):
 *   - the token text (whitespace made visible),
 *   - its best (minimum) rank across all read depths, and
 *   - the shallowest boundary where it first breaks into the top-{SURFACE_RANK}
 *     ("surfaces at ℓ19") — or a note that it never does.
 * A token tokenized to more than one piece is flagged so the reader knows only
 * its first token is being tracked.
 */

export type PinCopy = {
  chipsAria: string;
  bestRank: (n: number) => string;
  surfaces: (layer: number) => string;
  neverSurfaces: (k: number) => string;
  partial: string;
};

export function PinChips({
  slice,
  finalPos,
  pinTexts,
  pinColors,
  partialFlags,
  copy,
}: {
  slice: LensSliceData;
  finalPos: number;
  pinTexts: string[];
  pinColors: string[];
  partialFlags: boolean[];
  copy: PinCopy;
}) {
  return (
    <div role="list" aria-label={copy.chipsAria} className="flex flex-wrap items-center gap-2">
      {slice.pinned.map((_p, i) => {
        // Scan depths shallow → deep (layers ascending) at the final position.
        let best = RANK_CAP;
        let surfaceLayer: number | null = null;
        for (let layerIdx = 0; layerIdx < slice.layers.length; layerIdx++) {
          const rank = slice.rankAt(i, layerIdx, finalPos);
          if (rank < best) best = rank;
          if (surfaceLayer === null && rank <= SURFACE_RANK) surfaceLayer = slice.layers[layerIdx]!;
        }
        const color = pinColors[i] ?? 'var(--muted-foreground)';
        return (
          <span
            key={`pin-${i}`}
            role="listitem"
            className="inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[12px]"
            style={{ borderColor: color, background: `${color}1a` }}
          >
            <span className="font-mono font-semibold" style={{ color }}>
              {renderTokenDisplay(pinTexts[i] ?? '')}
            </span>
            <span className="text-[10px] text-muted-foreground">{copy.bestRank(best)}</span>
            <span className="text-[10px] text-muted-foreground">
              {surfaceLayer !== null ? copy.surfaces(surfaceLayer) : copy.neverSurfaces(SURFACE_RANK)}
            </span>
            {partialFlags[i] ? (
              <span className="text-[10px] italic text-muted-foreground/80" title={copy.partial}>
                *
              </span>
            ) : null}
          </span>
        );
      })}
    </div>
  );
}

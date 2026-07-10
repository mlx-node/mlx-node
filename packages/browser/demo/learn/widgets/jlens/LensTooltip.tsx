import * as React from 'react';

import { TopKBars } from '../../inspector/TopKBars';
import type { LensSliceData } from '../../../jlens-core/types';

/**
 * LensTooltip — the per-cell detail panel. Given a selected (layerIdx, pos)
 * cell it shows that cell's top-K read: token texts with a bar per token whose
 * width ∝ the token's full-vocab softmax probability. These ARE honest
 * probabilities (`LensCell.topKProbs` is normalized over the whole vocab row,
 * not a top-K-only softmax), so the panel labels them "probability". The raw
 * `topKLogits` are deliberately NOT surfaced here — they are not probabilities.
 *
 * Purely presentational: the container owns the selection and passes the
 * already-localized `header` / `probLabel` strings. Reuses the shared TopKBars
 * so the bars match chapters 9/10 and SpeculativeVerifyLive exactly (including
 * its reduced-motion-aware grow-in, driven by `runKey`).
 */
export function LensTooltip({
  slice,
  layerIdx,
  pos,
  runKey,
  header,
  probLabel,
}: {
  slice: LensSliceData;
  layerIdx: number;
  pos: number;
  runKey: number;
  header: string;
  probLabel: string;
}) {
  const cell = slice.cellAt(layerIdx, pos);
  // Full-vocab softmax probabilities as a plain number[] for TopKBars.
  const probs = Array.from(cell.topKProbs);

  return (
    <div className="space-y-1.5 rounded-md border border-border bg-muted/20 p-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-2 gap-y-0.5">
        <span className="font-mono text-[12px] text-foreground/85">{header}</span>
        <span className="text-[10px] uppercase tracking-wider text-muted-foreground">{probLabel}</span>
      </div>
      <TopKBars
        ids={cell.topKIds}
        probs={probs}
        texts={cell.topKTexts}
        sampledTokenId={cell.argmaxId}
        runKey={runKey}
      />
    </div>
  );
}

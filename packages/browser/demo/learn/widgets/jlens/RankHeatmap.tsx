import * as React from 'react';

import { RANK_CAP, readableInk, rankToScore, viridis, VIRIDIS_LEGEND_STOPS } from './colors';
import type { LensSliceData } from './types';

/**
 * RankHeatmap — the pinned-answer view. For each pinned answer token we track
 * its 1-based full-vocab rank at the FINAL prompt position (the next-token
 * slot — where the answer actually competes) across depth.
 *
 * Framing (documented, per the two options in the brief): rows = layers (same
 * reversed order as ArgmaxGrid, deepest on top), columns = pinned tokens, fill
 * = that token's rank AT THE FINAL POSITION mapped through a viridis ramp
 * (rank 1 = bright/yellow, rank ≥ cap = dark/purple). Reading a column bottom→
 * top shows the answer climbing the ranks as depth increases — bright only near
 * the top. The column header carries the pin's accent colour so it lines up
 * with the matching PinChip.
 *
 * SVG `<rect>`s, never canvas: the codebase is 100% SVG/CSS, canvas needs DPR
 * handling and is not SSR-clean, and a ≤16×8 grid does not need it.
 */

const CELL_W = 46;
const CELL_H = 20;
const LEFT_GUTTER = 30; // layer labels
const TOP_GUTTER = 22; // column headers
const PAD = 4;

export function RankHeatmap({
  slice,
  finalPos,
  pinTexts,
  pinColors,
  ariaLabel,
  legendLabel,
  brightLabel,
  darkLabel,
}: {
  slice: LensSliceData;
  finalPos: number;
  /** Decoded text per pinned token (index-aligned with slice.pinned). */
  pinTexts: string[];
  /** Accent per pinned token (index-aligned) — matches the PinChip colours. */
  pinColors: string[];
  ariaLabel: string;
  legendLabel: string;
  brightLabel: string;
  darkLabel: string;
}) {
  const gradientId = React.useId();
  const nPins = slice.pinned.length;
  const rowOrder = slice.layers.map((_, i) => i).reverse();

  const gridW = LEFT_GUTTER + nPins * CELL_W;
  const gridH = TOP_GUTTER + rowOrder.length * CELL_H;
  const width = gridW + PAD * 2;
  const height = gridH + PAD * 2;

  const truncate = (s: string) => (s.length > 7 ? s.slice(0, 6) + '…' : s);

  return (
    <div className="space-y-2">
      <div role="img" aria-label={ariaLabel} className="overflow-x-auto">
        <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} className="max-w-full">
          {/* Column headers — pin text in the pin's accent colour */}
          {slice.pinned.map((_p, c) => (
            <text
              key={`hd-${c}`}
              x={PAD + LEFT_GUTTER + c * CELL_W + CELL_W / 2}
              y={PAD + TOP_GUTTER - 7}
              textAnchor="middle"
              fontSize={8.5}
              fontWeight={600}
              fill={pinColors[c] ?? 'var(--foreground)'}
              style={{ fontFamily: 'var(--font-mono, monospace)' }}
            >
              {truncate(pinTexts[c] ?? '')}
            </text>
          ))}

          {rowOrder.map((layerIdx, r) => {
            const yBase = PAD + TOP_GUTTER + r * CELL_H;
            return (
              <React.Fragment key={`row-${layerIdx}`}>
                {/* Layer label */}
                <text
                  x={PAD + LEFT_GUTTER - 4}
                  y={yBase + CELL_H / 2}
                  textAnchor="end"
                  dominantBaseline="central"
                  fontSize={8.5}
                  fill="var(--muted-foreground)"
                  style={{ fontFamily: 'var(--font-mono, monospace)' }}
                >
                  ℓ{slice.layers[layerIdx]}
                </text>
                {slice.pinned.map((_p, c) => {
                  const rank = slice.rankAt(c, layerIdx, finalPos);
                  const score = rankToScore(rank);
                  const x = PAD + LEFT_GUTTER + c * CELL_W;
                  return (
                    <g key={`cell-${layerIdx}-${c}`}>
                      <rect
                        x={x + 1}
                        y={yBase + 1}
                        width={CELL_W - 2}
                        height={CELL_H - 2}
                        rx={2}
                        fill={viridis(score)}
                      >
                        <title>{`${pinTexts[c] ?? ''} · ℓ${slice.layers[layerIdx]}: rank ${rank >= RANK_CAP ? `${RANK_CAP}+` : rank}`}</title>
                      </rect>
                      <text
                        x={x + CELL_W / 2}
                        y={yBase + CELL_H / 2}
                        textAnchor="middle"
                        dominantBaseline="central"
                        fontSize={8.5}
                        fill={readableInk(score)}
                        style={{ fontFamily: 'var(--font-mono, monospace)' }}
                      >
                        {rank >= RANK_CAP ? `${RANK_CAP}+` : rank}
                      </text>
                    </g>
                  );
                })}
              </React.Fragment>
            );
          })}
        </svg>
      </div>

      {/* Legend: dark(≥cap) → bright(rank 1) */}
      <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
        <span>{legendLabel}</span>
        <span className="font-mono">{darkLabel}</span>
        <svg viewBox="0 0 120 8" width={120} height={8} aria-hidden="true" className="rounded-sm">
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="0">
              {VIRIDIS_LEGEND_STOPS.map((s, i) => (
                <stop key={i} offset={s.offset} stopColor={s.color} />
              ))}
            </linearGradient>
          </defs>
          <rect x={0} y={0} width={120} height={8} rx={2} fill={`url(#${gradientId})`} />
        </svg>
        <span className="font-mono">{brightLabel}</span>
      </div>
    </div>
  );
}

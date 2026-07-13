// WorkspaceMetricsStrip — a compact "where does the readout commit?" panel of
// across-layer mini-charts at the last prompt position, computed CLIENT-SIDE from
// the pure `workspace-metrics` module over what lensReadout already ships. No
// model, no worker, no window/document — purely presentational over `slice`.
//
// THE HONESTY CONTRACT (mirrored from workspace-metrics.ts and enforced here):
//   - Concept rank trajectory  = FAITHFUL-with-cap. Full-vocab pinned rank per
//     layer. A censored value (≥ RANK_CAP) is drawn as an OFF-SCALE open marker on
//     the bottom floor, NEVER joined into the line as an exact value — and the
//     chart carries NO "exact" tag (its caption states "≥999 = off-scale").
//   - Concept in top-10 by layer = FAITHFUL, exact (the k=10 threshold sits far
//     below the 999 cap, so a censored 999 correctly reads "not in top-10"). Tagged
//     `faithfulTag` ("exact").
//   - Readout entropy + top-10 set stability = PROXY (readout-space, top-10). Each
//     carries the `proxyTag` chip — they are NOT the paper's activation/residual
//     quantities.
//   - The impossible full-distribution metrics (excess kurtosis, residual
//     autocorrelation, activation participation-ratio) are OMITTED with a one-line
//     note — they need full logits/residuals that no cell ships in-browser.

import * as React from 'react';

import {
  conceptRankTrajectory,
  conceptTopKAccuracy,
  isCensoredRank,
  readoutEntropy,
  topKSetStability,
} from '../jlens-core/workspace-metrics';
import { pinColor, RANK_CAP, SURFACE_RANK } from '../jlens-core/colors';
import type { LensSliceData } from '../jlens-core/types';
import { rankToY } from './RankChart';

/** The `metrics` sub-block of `JSPACE_COPY[locale]` — all display strings. */
export type WorkspaceMetricsCopy = {
  title: string;
  faithfulTag: string;
  proxyTag: string;
  /** Sub-caption: every series is read at the FINAL prompt position (1-based #n). */
  atPosition: (position: number) => string;
  /** Short per-chart suffix naming the same fixed position (for accessible names). */
  atPositionShort: (position: number) => string;
  rankTrajectory: string;
  topkAcc: string;
  entropy: string;
  /** Fixed-domain scale hint for the entropy chart, e.g. "0–ln 10 nats". */
  entropyScale: string;
  stability: string;
  /** Fixed-domain scale hint for the stability chart, e.g. "0–1 Jaccard". */
  stabilityScale: string;
  omitNote: string;
};

const MINI_W = 140;
const MINI_H = 48;
const PAD = { t: 5, b: 7, l: 5, r: 5 };
const INNER_W = MINI_W - PAD.l - PAD.r;
const INNER_H = MINI_H - PAD.t - PAD.b;

// FIXED honest domains for the PROXY charts (codex final): auto-scaling each
// series to its own min→max destroyed magnitude — a flat [0,0] and a flat [1,1]
// rendered identically, and [0.49,0.51] filled the same height as [0,1]. Plot on
// a constant axis so heights are comparable across lenses and runs.
//  - entropy over the visible top-10 is bounded by ln(SURFACE_RANK) (equal shares
//    of a mass ≤ 1 maximize it exactly at ln K); floor 0.
//  - top-10 set Jaccard stability is a similarity in [0, 1] by construction.
const ENTROPY_DOMAIN: readonly [number, number] = [0, Math.log(SURFACE_RANK)];
const STABILITY_DOMAIN: readonly [number, number] = [0, 1];

/** faithful/exact = primary-tinted chip; proxy = de-emphasised neutral chip so it
 *  reads as "an approximation, not the real quantity" at a glance. */
function Tag({ label, kind }: { label: string; kind: 'faithful' | 'proxy' }) {
  return (
    <span
      className={[
        'inline-flex shrink-0 items-center rounded-full border px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-[0.14em]',
        kind === 'faithful'
          ? 'border-primary/40 bg-primary/10 text-primary'
          : 'border-border/70 bg-muted/40 text-[color:var(--text-dim)]',
      ].join(' ')}
    >
      {label}
    </span>
  );
}

/** One mini-chart cell: caption row (+ optional tag) above a small chart. */
function MiniPanel({
  label,
  tag,
  children,
}: {
  label: string;
  tag?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div className="space-y-1.5 rounded-lg border border-border/50 bg-background/30 p-2.5">
      <div className="flex items-start justify-between gap-2">
        <span className="text-[11px] leading-snug text-[color:var(--text-dim)]">{label}</span>
        {tag}
      </div>
      {children}
    </div>
  );
}

const xAcross = (i: number, n: number) => PAD.l + (n > 1 ? i / (n - 1) : 0.5) * INNER_W;

/**
 * FAITHFUL-with-cap concept rank trajectory: log-y line (rank 1 at top, via the
 * shared {@link rankToY}). Censored points (≥ RANK_CAP) break the line and are
 * drawn as hollow off-scale markers on the bottom floor — never joined in as an
 * exact value (mirrors RankChart's off-scale convention).
 */
function RankTrajectoryMini({
  trajectory,
  color,
  ariaLabel,
}: {
  trajectory: number[];
  color: string;
  ariaLabel: string;
}) {
  const n = trajectory.length;
  const yFor = (rank: number) => PAD.t + rankToY(rank, INNER_H);
  const floorY = yFor(RANK_CAP);

  // Contiguous in-scale runs (rank < cap); a censored point breaks the run.
  const runs: Array<Array<{ x: number; y: number }>> = [];
  let run: Array<{ x: number; y: number }> = [];
  trajectory.forEach((rank, i) => {
    if (isCensoredRank(rank)) {
      if (run.length) runs.push(run);
      run = [];
    } else {
      run.push({ x: xAcross(i, n), y: yFor(rank) });
    }
  });
  if (run.length) runs.push(run);
  const offScale = trajectory
    .map((rank, i) => ({ rank, i }))
    .filter((p) => isCensoredRank(p.rank));

  return (
    <svg
      viewBox={`0 0 ${MINI_W} ${MINI_H}`}
      role="img"
      aria-label={ariaLabel}
      className="block h-auto w-full"
      style={{ maxWidth: '100%' }}
    >
      {/* rank-1 top guide + cap floor, faint. */}
      <line x1={PAD.l} x2={MINI_W - PAD.r} y1={yFor(1)} y2={yFor(1)} stroke="currentColor" strokeOpacity={0.08} />
      <line x1={PAD.l} x2={MINI_W - PAD.r} y1={floorY} y2={floorY} stroke="currentColor" strokeOpacity={0.08} />
      {runs.map((r, ri) =>
        r.length === 1 ? (
          <circle key={`d-${ri}`} cx={r[0]!.x} cy={r[0]!.y} r={1.5} fill={color} />
        ) : (
          <path
            key={`r-${ri}`}
            d={'M ' + r.map((p) => `${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(' L ')}
            fill="none"
            stroke={color}
            strokeWidth={1.4}
          />
        ),
      )}
      {/* Off-scale (censored ≥ cap) points: hollow markers on the floor. */}
      {offScale.map((p) => (
        <circle
          key={`cap-${p.i}`}
          cx={xAcross(p.i, n)}
          cy={floorY}
          r={2}
          fill="none"
          stroke={color}
          strokeOpacity={0.75}
          strokeWidth={1.1}
        />
      ))}
    </svg>
  );
}

/** FIXED-domain mini line for the PROXY series (entropy / stability). Plots on a
 *  constant [lo, hi] axis (NOT per-series auto-scale) so heights are comparable
 *  across lenses and runs; a flat series sits at its true level, not the middle.
 *  The `scaleHint` is drawn (faintly) and folded into the accessible name so the
 *  axis is never a hidden normalization. */
function LineMini({
  values,
  ariaLabel,
  domain,
  scaleHint,
}: {
  values: number[];
  ariaLabel: string;
  domain: readonly [number, number];
  scaleHint: string;
}) {
  const n = values.length;
  const [lo, hi] = domain;
  const span = hi - lo || 1;
  const clamp01 = (t: number) => (t < 0 ? 0 : t > 1 ? 1 : t);
  const yFor = (v: number) => PAD.t + (1 - clamp01((v - lo) / span)) * INNER_H;
  const pts = values.map((v, i) => ({ x: xAcross(i, n), y: yFor(Number.isFinite(v) ? v : lo) }));

  return (
    <svg
      viewBox={`0 0 ${MINI_W} ${MINI_H}`}
      role="img"
      aria-label={`${ariaLabel} · ${scaleHint}`}
      className="block h-auto w-full text-foreground/45"
      style={{ maxWidth: '100%' }}
    >
      {/* Fixed-domain guides: floor (lo) and ceiling (hi), so magnitude is legible. */}
      <line x1={PAD.l} x2={MINI_W - PAD.r} y1={PAD.t} y2={PAD.t} stroke="currentColor" strokeOpacity={0.08} />
      <line
        x1={PAD.l}
        x2={MINI_W - PAD.r}
        y1={PAD.t + INNER_H}
        y2={PAD.t + INNER_H}
        stroke="currentColor"
        strokeOpacity={0.12}
      />
      {pts.length === 1 ? (
        <circle cx={pts[0]!.x} cy={pts[0]!.y} r={1.5} fill="currentColor" />
      ) : (
        <path
          d={'M ' + pts.map((p) => `${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(' L ')}
          fill="none"
          stroke="currentColor"
          strokeWidth={1.4}
        />
      )}
      <text x={MINI_W - PAD.r} y={MINI_H - 1.5} textAnchor="end" fontSize={5.5} fill="currentColor" opacity={0.55}>
        {scaleHint}
      </text>
    </svg>
  );
}

/** FAITHFUL exact: one pip per layer, filled iff the concept sits within top-k. */
function PipRow({ bits, color, ariaLabel }: { bits: Array<0 | 1>; color: string; ariaLabel: string }) {
  return (
    <div role="img" aria-label={ariaLabel} className="flex min-h-[1.75rem] flex-wrap items-center gap-1">
      {bits.map((b, i) => (
        <span
          key={i}
          aria-hidden
          className="h-2.5 w-2.5 rounded-[2px] border"
          style={
            b
              ? { backgroundColor: color, borderColor: color }
              : { backgroundColor: 'transparent', borderColor: 'var(--border)' }
          }
        />
      ))}
    </div>
  );
}

export function WorkspaceMetricsStrip({
  slice,
  pinnedIdx,
  copy,
}: {
  slice: LensSliceData;
  pinnedIdx: number;
  copy: WorkspaceMetricsCopy;
}) {
  const pos = Math.max(0, slice.promptLen - 1);
  const hasPins = slice.pinned.length > 0;
  const pinIdx = hasPins ? Math.min(Math.max(pinnedIdx, 0), slice.pinned.length - 1) : 0;
  const accent = pinColor(pinIdx);

  const trajectory = React.useMemo(
    () => (hasPins ? conceptRankTrajectory(slice, pinIdx, pos) : []),
    [slice, pinIdx, pos, hasPins],
  );
  const topkAcc = React.useMemo(
    () => (hasPins ? conceptTopKAccuracy(slice, pinIdx, SURFACE_RANK, pos) : []),
    [slice, pinIdx, pos, hasPins],
  );
  const entropy = React.useMemo(
    () => slice.layers.map((_l, l) => readoutEntropy(slice, l, pos)),
    [slice, pos],
  );
  const stability = React.useMemo(() => topKSetStability(slice, pos), [slice, pos]);

  // Every series is read at the FINAL prompt position (1-based #promptLen). It is
  // fixed, NOT the currently-selected grid cell, so disclose it (codex final) —
  // both as a visible sub-caption and inside each chart's accessible name.
  const posN = slice.promptLen;
  const posSuffix = copy.atPositionShort(posN);

  return (
    <section aria-label={copy.title} className="jspace-panel space-y-3 p-4">
      <h3 className="jspace-eyebrow">{copy.title}</h3>
      <p className="text-[11px] leading-snug text-[color:var(--text-dim)]">{copy.atPosition(posN)}</p>
      <div className="grid gap-3 sm:grid-cols-2">
        {hasPins ? (
          <>
            {/* (A) FAITHFUL-with-cap — no "exact" tag; caption states ≥999 off-scale. */}
            <MiniPanel label={copy.rankTrajectory}>
              <RankTrajectoryMini
                trajectory={trajectory}
                color={accent}
                ariaLabel={`${copy.rankTrajectory} · ${posSuffix}`}
              />
            </MiniPanel>
            {/* (B) FAITHFUL, exact — k=10 sits far below the 999 cap. */}
            <MiniPanel label={copy.topkAcc} tag={<Tag label={copy.faithfulTag} kind="faithful" />}>
              <PipRow bits={topkAcc} color={accent} ariaLabel={`${copy.topkAcc} · ${posSuffix}`} />
            </MiniPanel>
          </>
        ) : null}
        {/* (C) PROXY — readout-space top-10 entropy, fixed [0, ln 10] axis. */}
        <MiniPanel label={copy.entropy} tag={<Tag label={copy.proxyTag} kind="proxy" />}>
          <LineMini
            values={entropy}
            ariaLabel={`${copy.entropy} · ${posSuffix}`}
            domain={ENTROPY_DOMAIN}
            scaleHint={copy.entropyScale}
          />
        </MiniPanel>
        {/* (D) PROXY — readout-space adjacent-layer top-10 set stability, fixed [0,1]. */}
        <MiniPanel label={copy.stability} tag={<Tag label={copy.proxyTag} kind="proxy" />}>
          <LineMini
            values={stability}
            ariaLabel={`${copy.stability} · ${posSuffix}`}
            domain={STABILITY_DOMAIN}
            scaleHint={copy.stabilityScale}
          />
        </MiniPanel>
      </div>
      {/* Honesty footer (mandatory): the impossible full-distribution metrics. */}
      <p className="max-w-[72ch] text-[11px] leading-relaxed text-[color:var(--text-dim)]">{copy.omitNote}</p>
    </section>
  );
}

import * as React from 'react';

import { pinColor, RANK_CAP } from '../jlens-core/colors';
import { CANVAS } from './canvas-theme';

/**
 * RankChart — the pinned tokens' rank trajectories as bump-chart lines.
 *
 * Rank 1 at the TOP, log scale (the bump-chart convention, and what Anthropic's
 * viewer does: `d3.scaleLog().domain([0.9, VMAX])`). A rank at or above
 * RANK_CAP is NOT plotted as "moderately ranked" — it is pinned to an off-scale
 * mark at the bottom band and drawn as an open marker, never joined into the
 * line. One line per pinned series, coloured by the caller's accent (which is
 * `pinColor(i)`), so a line and its PinChip read as the same thing.
 *
 * No charting library: the repo hand-rolls SVG paths (learn/widgets/KvGrowthCurve).
 * Past ~200 x-samples an SVG path per series gets heavy, so we mirror Anthropic's
 * threshold and switch to a single <canvas> render (same DPR discipline as the
 * canvas grid).
 */

/**
 * Rank 1 at the TOP, log scale (the bump-chart convention, and what Anthropic's
 * viewer does: `d3.scaleLog().domain([0.9, VMAX])`). A rank at or above
 * RANK_CAP is NOT plotted as "moderately ranked" — it is pinned to an off-scale
 * mark at the bottom band and drawn as an open marker.
 */
export function rankToY(rank: number, height: number): number {
  const clamped = Math.min(Math.max(rank, 1), RANK_CAP);
  return (Math.log10(clamped) / Math.log10(RANK_CAP)) * height;
}

const HEIGHT = 240;
const PAD_LEFT = 44;
const PAD_RIGHT = 14;
const PAD_TOP = 12;
const PAD_BOTTOM = 26;
const SVG_WIDTH = 560; // logical viewBox width for the SVG path renderer
const CANVAS_THRESHOLD = 200; // > this many x-samples ⇒ canvas (Anthropic's cutoff)
const FONT = '11px ui-monospace, SFMono-Regular, Menlo, monospace';

/** y-axis ticks: rank 1 at the top, then decades, with the cap band at the bottom. */
const Y_TICKS: Array<{ rank: number; label: string }> = [
  { rank: 1, label: '1' },
  { rank: 10, label: '10' },
  { rank: 100, label: '100' },
  { rank: RANK_CAP, label: `≥${RANK_CAP}` },
];

type Point = { x: number; rank: number };

/**
 * CSS-pixel height the chart renders at for a container of `containerWidth` px.
 *
 * BOTH renderers share this ONE responsive aspect ratio (`HEIGHT / SVG_WIDTH`):
 *   - the SVG path has `viewBox="0 0 560 240"` + `h-auto w-full`, so the browser
 *     lays it out at exactly `containerWidth * 240/560`.
 *   - the canvas path sets `style.height` to this value and scales ALL drawing
 *     coordinates by `k = containerWidth / SVG_WIDTH` (see RankChartCanvas).
 * That keeps the picture geometrically CONTINUOUS across the 200→201-sample
 * SVG→canvas switch, instead of jumping from ~171px (scaled SVG) to a fixed 240px.
 */
export function chartHeightFor(containerWidth: number): number {
  return containerWidth * (HEIGHT / SVG_WIDTH);
}

/** Drawing geometry for a given total drawing width `w` (SVG units or CSS px). */
function scales(w: number, xMin: number, xMax: number) {
  const innerW = Math.max(1, w - PAD_LEFT - PAD_RIGHT);
  const innerH = HEIGHT - PAD_TOP - PAD_BOTTOM;
  const span = xMax - xMin;
  const xFor = (x: number) => PAD_LEFT + (span > 0 ? (x - xMin) / span : 0.5) * innerW;
  const yFor = (rank: number) => PAD_TOP + rankToY(rank, innerH);
  return { innerW, innerH, xFor, yFor };
}

/** x-domain across every series (they share the same axis). */
function xDomain(points: Point[][]): { xMin: number; xMax: number } {
  let xMin = Infinity;
  let xMax = -Infinity;
  for (const series of points) {
    for (const p of series) {
      if (p.x < xMin) xMin = p.x;
      if (p.x > xMax) xMax = p.x;
    }
  }
  if (!Number.isFinite(xMin) || !Number.isFinite(xMax)) return { xMin: 0, xMax: 1 };
  return { xMin, xMax };
}

/**
 * Split a series into contiguous runs of IN-SCALE points (rank < RANK_CAP). A
 * capped point breaks the line — it is drawn as an open marker instead, never
 * joined in — so each returned run becomes its own path (or a lone dot).
 */
function inScaleRuns(series: Point[]): Point[][] {
  const runs: Point[][] = [];
  let run: Point[] = [];
  for (const p of series) {
    if (p.rank >= RANK_CAP) {
      if (run.length) runs.push(run);
      run = [];
    } else {
      run.push(p);
    }
  }
  if (run.length) runs.push(run);
  return runs;
}

export function RankChart({
  points,
  colors,
  xLabel,
  selectedX,
}: {
  points: Point[][];
  colors: string[];
  xLabel: string;
  selectedX: number | null;
}) {
  const { xMin, xMax } = React.useMemo(() => xDomain(points), [points]);
  const useCanvas = (points[0]?.length ?? 0) > CANVAS_THRESHOLD;
  const colorFor = (i: number) => colors[i] ?? pinColor(i);
  const selInRange = selectedX != null && selectedX >= xMin && selectedX <= xMax;

  if (useCanvas) {
    return (
      <RankChartCanvas
        points={points}
        colors={colors}
        xLabel={xLabel}
        selectedX={selectedX}
        xMin={xMin}
        xMax={xMax}
      />
    );
  }

  const { xFor, yFor } = scales(SVG_WIDTH, xMin, xMax);

  return (
    <div className="overflow-x-auto">
      <svg
        viewBox={`0 0 ${SVG_WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={xLabel}
        className="block h-auto w-full"
        style={{ maxWidth: '100%' }}
      >
        {/* Horizontal gridlines + y tick labels (rank 1 on top, cap band at bottom). */}
        {Y_TICKS.map((t) => {
          const y = yFor(t.rank);
          return (
            <g key={`y-${t.rank}`}>
              <line
                x1={PAD_LEFT}
                x2={SVG_WIDTH - PAD_RIGHT}
                y1={y}
                y2={y}
                stroke="currentColor"
                strokeOpacity={0.08}
              />
              <text
                x={PAD_LEFT - 6}
                y={y + 3}
                fontSize={9}
                textAnchor="end"
                fill="currentColor"
                fillOpacity={0.55}
                style={{ fontFamily: 'var(--font-mono, monospace)' }}
              >
                {t.label}
              </text>
            </g>
          );
        })}

        {/* Dashed vertical rule at the selected x-position. */}
        {selInRange && (
          <line
            x1={xFor(selectedX)}
            x2={xFor(selectedX)}
            y1={PAD_TOP}
            y2={PAD_TOP + (HEIGHT - PAD_TOP - PAD_BOTTOM)}
            stroke={CANVAS.selectionRing}
            strokeWidth={1}
            strokeDasharray="3 3"
          />
        )}

        {/* One line per pinned series, in the pin's accent. Capped points break it. */}
        {points.map((series, i) => {
          const color = colorFor(i);
          return (
            <g key={`series-${i}`}>
              {inScaleRuns(series).map((run, ri) => {
                if (run.length === 1) {
                  const p = run[0]!;
                  return (
                    <circle key={`dot-${ri}`} cx={xFor(p.x)} cy={yFor(p.rank)} r={1.8} fill={color} />
                  );
                }
                const d =
                  'M ' + run.map((p) => `${xFor(p.x).toFixed(2)} ${yFor(p.rank).toFixed(2)}`).join(' L ');
                return (
                  <path key={`run-${ri}`} d={d} fill="none" stroke={color} strokeWidth={1.5} />
                );
              })}
              {/* Off-scale (rank ≥ cap) points: hollow markers on the bottom band. */}
              {series
                .filter((p) => p.rank >= RANK_CAP)
                .map((p, mi) => (
                  <circle
                    key={`cap-${mi}`}
                    cx={xFor(p.x)}
                    cy={yFor(RANK_CAP)}
                    r={2.5}
                    fill="none"
                    stroke={color}
                    strokeWidth={1.25}
                  />
                ))}
            </g>
          );
        })}

        {/* x-axis label. */}
        <text
          x={PAD_LEFT + (SVG_WIDTH - PAD_LEFT - PAD_RIGHT) / 2}
          y={HEIGHT - 6}
          fontSize={9}
          textAnchor="middle"
          fill="currentColor"
          fillOpacity={0.55}
          style={{ fontFamily: 'var(--font-mono, monospace)' }}
        >
          {xLabel}
        </text>
      </svg>
    </div>
  );
}

/**
 * Canvas twin of the SVG path renderer, used once a series exceeds
 * CANVAS_THRESHOLD samples. Same DPR-aware backing-store discipline as
 * ArgmaxGridCanvas: a `resizeTick` bumped on devicePixelRatio change forces a
 * resample even when the CSS width is unchanged.
 */
function RankChartCanvas({
  points,
  colors,
  xLabel,
  selectedX,
  xMin,
  xMax,
}: {
  points: Point[][];
  colors: string[];
  xLabel: string;
  selectedX: number | null;
  xMin: number;
  xMax: number;
}) {
  const wrapRef = React.useRef<HTMLDivElement>(null);
  const canvasRef = React.useRef<HTMLCanvasElement>(null);
  const [width, setWidth] = React.useState(0);
  const [resizeTick, setResizeTick] = React.useState(0);
  const lastDprRef = React.useRef(0);
  const colorFor = (i: number) => colors[i] ?? pinColor(i);

  React.useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const measure = () => {
      setWidth(el.clientWidth);
      const dpr = window.devicePixelRatio || 1;
      if (dpr !== lastDprRef.current) {
        lastDprRef.current = dpr;
        setResizeTick((n) => n + 1);
      }
    };
    measure();
    let ro: ResizeObserver | undefined;
    if (typeof ResizeObserver !== 'undefined') {
      ro = new ResizeObserver(measure);
      ro.observe(el);
    }
    window.addEventListener('resize', measure);
    return () => {
      ro?.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, []);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = width > 0 ? width : SVG_WIDTH;
    const k = cssW / SVG_WIDTH; // uniform responsive scale — matches the SVG viewBox
    const cssH = chartHeightFor(cssW); // == k * HEIGHT: same aspect ratio as the SVG path
    canvas.width = Math.max(1, Math.round(cssW * dpr));
    canvas.height = Math.max(1, Math.round(cssH * dpr));
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    // Draw in the SAME logical 560×240 space as the SVG path; fold the responsive
    // scale `k` and the backing-store `dpr` into ONE transform so every coordinate
    // (and the padding) scales identically — the two renderers stay geometrically
    // continuous across the CANVAS_THRESHOLD sample-count switch.
    ctx.setTransform(k * dpr, 0, 0, k * dpr, 0, 0);
    ctx.clearRect(0, 0, SVG_WIDTH, HEIGHT);

    const { xFor, yFor, innerH } = scales(SVG_WIDTH, xMin, xMax);

    // Gridlines + y tick labels.
    ctx.font = FONT;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    for (const t of Y_TICKS) {
      const y = yFor(t.rank);
      ctx.strokeStyle = 'rgba(236, 228, 218, 0.07)'; // warm hairline — matches CANVAS.grid
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(PAD_LEFT, y);
      ctx.lineTo(SVG_WIDTH - PAD_RIGHT, y);
      ctx.stroke();
      ctx.fillStyle = CANVAS.inkMuted;
      ctx.fillText(t.label, PAD_LEFT - 6, y);
    }

    // Dashed vertical rule at the selected x-position.
    if (selectedX != null && selectedX >= xMin && selectedX <= xMax) {
      const sx = xFor(selectedX);
      ctx.strokeStyle = CANVAS.selectionRing;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(sx, PAD_TOP);
      ctx.lineTo(sx, PAD_TOP + innerH);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Lines per series (broken at caps) + hollow markers for off-scale points.
    points.forEach((series, i) => {
      const color = colorFor(i);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      for (const run of inScaleRuns(series)) {
        if (run.length === 1) {
          const p = run[0]!;
          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(xFor(p.x), yFor(p.rank), 1.8, 0, Math.PI * 2);
          ctx.fill();
          continue;
        }
        ctx.beginPath();
        run.forEach((p, k) => {
          const px = xFor(p.x);
          const py = yFor(p.rank);
          if (k === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
      }
      ctx.lineWidth = 1.25;
      for (const p of series) {
        if (p.rank < RANK_CAP) continue;
        ctx.beginPath();
        ctx.arc(xFor(p.x), yFor(RANK_CAP), 2.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    });

    // x-axis label — logical coords, matching the SVG path's placement exactly.
    ctx.fillStyle = CANVAS.inkMuted;
    ctx.textAlign = 'center';
    ctx.fillText(xLabel, PAD_LEFT + (SVG_WIDTH - PAD_LEFT - PAD_RIGHT) / 2, HEIGHT - 6);
  }, [points, colors, xLabel, selectedX, xMin, xMax, width, resizeTick]);

  return (
    <div ref={wrapRef} className="overflow-x-auto">
      <canvas ref={canvasRef} role="img" aria-label={xLabel} style={{ display: 'block' }} />
    </div>
  );
}

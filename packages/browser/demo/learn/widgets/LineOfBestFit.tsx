import * as React from 'react';

import { useLocale } from '../../lib/i18n-react';

/**
 * Chapter 1 (overview) — the tiniest possible "training is fitting" picture.
 *
 * Self-contained: NO worker, NO model, NO WASM, NO animation — a single static
 * SVG, SSR-safe (no window access). Scattered dots with one straight line
 * y = a·x + b drawn through them, to make the point concrete: training is
 * nudging a small set of knobs (here just two, the slope a and the intercept b)
 * until the line lands close to the examples. The chapter prose then scales that
 * intuition up to Qwen's 852,985,920 knobs.
 *
 * The dots are hand-placed (deterministic constants, not Math.random) so server
 * and client render identically and the line visibly "fits" them.
 */

// Hand-placed example points in an 0..10 × 0..10 data space, loosely along the
// true line y ≈ 0.62·x + 1.6, with scatter — so the fitted line is believable.
const POINTS: ReadonlyArray<readonly [number, number]> = [
  [0.6, 2.4],
  [1.5, 1.9],
  [2.4, 3.6],
  [3.1, 3.0],
  [3.9, 4.7],
  [4.8, 4.0],
  [5.6, 5.5],
  [6.4, 4.9],
  [7.2, 6.7],
  [8.1, 5.8],
  [8.9, 7.4],
  [9.6, 6.9],
];

// The fitted line, y = a·x + b — the two knobs.
const A = 0.62;
const B = 1.6;

// SVG plot geometry (viewBox units). Data space 0..10 maps into the inner box.
const W = 320;
const H = 180;
const PAD_L = 16;
const PAD_R = 16;
const PAD_T = 16;
const PAD_B = 16;
const PLOT_W = W - PAD_L - PAD_R;
const PLOT_H = H - PAD_T - PAD_B;

function dataToX(x: number): number {
  return PAD_L + (x / 10) * PLOT_W;
}
function dataToY(y: number): number {
  // Flip: data y grows upward, SVG y grows downward.
  return PAD_T + (1 - y / 10) * PLOT_H;
}

const COPY = {
  en: {
    title: 'Two knobs, fitted to examples',
    knobs: 'y = a·x + b — slope a and intercept b are the only two knobs',
    caption: 'Training nudges those two numbers until the line sits as close to the dots as it can.',
  },
  zh: {
    title: '两个旋钮，拟合到样例上',
    knobs: 'y = a·x + b——斜率 a 与截距 b 是仅有的两个旋钮',
    caption: '训练不断微调这两个数字，直到这条线尽可能贴近这些散点。',
  },
} as const;

export function LineOfBestFit() {
  const copy = COPY[useLocale()];

  const x0 = 0;
  const x1 = 10;
  const lineX0 = dataToX(x0);
  const lineY0 = dataToY(A * x0 + B);
  const lineX1 = dataToX(x1);
  const lineY1 = dataToY(A * x1 + B);

  return (
    <div className="space-y-2 rounded-md border border-border bg-background p-3">
      <div className="text-xs uppercase tracking-wider text-muted-foreground">{copy.title}</div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="mx-auto block w-full max-w-[360px]"
        role="img"
        aria-label={copy.knobs}
      >
        {/* Plot frame */}
        <rect
          x={PAD_L}
          y={PAD_T}
          width={PLOT_W}
          height={PLOT_H}
          fill="none"
          stroke="var(--border)"
          strokeWidth={1}
          rx={3}
        />
        {/* The fitted line — the two knobs at work. */}
        <line
          x1={lineX0}
          y1={lineY0}
          x2={lineX1}
          y2={lineY1}
          stroke="var(--primary)"
          strokeWidth={2.5}
          strokeLinecap="round"
        />
        {/* The example points. */}
        {POINTS.map(([px, py], i) => (
          <circle
            key={i}
            cx={dataToX(px)}
            cy={dataToY(py)}
            r={3.5}
            fill="var(--card)"
            stroke="var(--foreground)"
            strokeWidth={1.5}
          />
        ))}
      </svg>
      <p className="text-center font-mono text-[11px] text-foreground/80">{copy.knobs}</p>
      <p className="text-[10px] text-muted-foreground">{copy.caption}</p>
    </div>
  );
}

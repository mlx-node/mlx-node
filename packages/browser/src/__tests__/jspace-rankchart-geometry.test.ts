// Geometry continuity for RankChart across the 200→201-sample SVG→canvas switch
// (the adversarial finding). The SVG path draws in a responsive
// `viewBox="0 0 560 240"` (so it lays out at containerWidth × 240/560), while the
// canvas path used to render at a FIXED 240px height with unscaled padding — so
// at ~400px the chart height jumped ~171px → 240px and every coordinate shifted.
//
// Both paths now share ONE responsive aspect ratio (`chartHeightFor`) and the
// SAME logical 560×240 drawing space, so the picture is continuous. This mounts
// both branches in a NON-560px container and asserts their rendered heights match
// within ~1px.
import * as React from 'react';
import { flushSync } from 'react-dom';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { RankChart, chartHeightFor } from '../../demo/jspace/RankChart';

const CONTAINER_W = 400; // deliberately NOT 560 — exercises the responsive scale.

type Point = { x: number; rank: number };

/** One series of `n` in-scale points (ranks 1..50 so lines are actually drawn). */
function makePoints(n: number): Point[][] {
  return [Array.from({ length: n }, (_, i) => ({ x: i, rank: 1 + (i % 50) }))];
}

let root: Root | null = null;
let container: HTMLDivElement | null = null;

/** Let the ResizeObserver measure, the width state update, and the draw effect
 *  (which sets `canvas.style.height`) settle. */
async function settle(): Promise<void> {
  for (let i = 0; i < 8; i++) {
    await new Promise<void>((r) => requestAnimationFrame(() => r()));
  }
}

/** Mount RankChart with `n` points in a fixed-width container; return the rendered
 *  chart element (svg for ≤200 points, canvas for >200) after it settles. */
async function mountChart(n: number): Promise<SVGSVGElement | HTMLCanvasElement> {
  container = document.createElement('div');
  container.style.width = `${CONTAINER_W}px`;
  container.style.padding = '0';
  container.style.border = '0';
  container.style.boxSizing = 'border-box';
  document.body.appendChild(container);
  root = createRoot(container);
  flushSync(() => {
    root!.render(
      React.createElement(RankChart, {
        points: makePoints(n),
        colors: ['#3b82f6'],
        xLabel: 'layer',
        selectedX: null,
      }),
    );
  });
  await settle();
  return (container.querySelector('svg') ?? container.querySelector('canvas'))!;
}

afterEach(() => {
  if (root) flushSync(() => root!.unmount());
  container?.remove();
  root = null;
  container = null;
});

describe('RankChart geometry across the SVG↔canvas threshold', () => {
  it('renders the SVG path for 200 samples and the canvas path for 201', async () => {
    const svg = await mountChart(200);
    expect(svg.tagName.toLowerCase()).toBe('svg');
    afterEachCleanup();

    const canvas = await mountChart(201);
    expect(canvas.tagName.toLowerCase()).toBe('canvas');
  });

  it('keeps the rendered height continuous (no ~171→240px jump) across the switch', async () => {
    const expected = chartHeightFor(CONTAINER_W); // 400 * 240/560 ≈ 171.43

    const svg = await mountChart(200);
    const svgH = svg.getBoundingClientRect().height;
    afterEachCleanup();

    const canvas = await mountChart(201);
    const canvasH = canvas.getBoundingClientRect().height;

    // Each branch matches the shared responsive aspect ratio…
    expect(svgH).toBeGreaterThan(0);
    expect(Math.abs(svgH - expected)).toBeLessThan(1);
    expect(Math.abs(canvasH - expected)).toBeLessThan(1);
    // …and therefore each other: the picture does not jump at the threshold.
    expect(Math.abs(svgH - canvasH)).toBeLessThan(1);
    // Guard against a silent regression back to the old fixed 240px canvas height.
    expect(canvasH).toBeLessThan(200);
  });

  it('chartHeightFor is the single responsive aspect ratio for any width', () => {
    // Pure helper: HEIGHT/SVG_WIDTH = 240/560, applied linearly.
    expect(chartHeightFor(560)).toBeCloseTo(240, 5);
    expect(chartHeightFor(400)).toBeCloseTo((400 * 240) / 560, 5);
    expect(chartHeightFor(280)).toBeCloseTo(120, 5);
  });
});

/** Tear down the current mount between two mounts inside ONE test. */
function afterEachCleanup(): void {
  if (root) flushSync(() => root!.unmount());
  container?.remove();
  root = null;
  container = null;
}

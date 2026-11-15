/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import { MorphLoader } from '../ui/src/components/arc-x/morph-loader/morph-loader.js';
import {
  LOADER_SHAPES,
  mixShape,
  segmentPath,
  turnShape,
  type LoaderShape,
  type LoaderStatus,
  type Segment,
} from '../ui/src/components/arc-x/morph-loader/shapes.js';
import './render.js';

const STATUSES: readonly LoaderStatus[] = ['pending', 'loading', 'success', 'error'];
const EPSILON = 1e-6;

function mount(element: ReactElement): { container: HTMLElement; rerender(next: ReactElement): void } {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(element));
  return { container, rerender: (next) => act(() => root.render(next)) };
}

/** Twice the signed area of the triangle a, b, c: 0 when the three points are on one line. */
function cross(ax: number, ay: number, bx: number, by: number, cx: number, cy: number): number {
  return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

function isStraight(segment: Segment): boolean {
  const { x0, y0, x1, y1, x2, y2, x3, y3 } = segment;
  return Math.abs(cross(x0, y0, x3, y3, x1, y1)) < 1e-2 && Math.abs(cross(x0, y0, x3, y3, x2, y2)) < 1e-2;
}

function collinear(a: Segment, b: Segment): boolean {
  return (
    Math.abs(cross(a.x0, a.y0, a.x3, a.y3, b.x0, b.y0)) < EPSILON &&
    Math.abs(cross(a.x0, a.y0, a.x3, a.y3, b.x3, b.y3)) < EPSILON
  );
}

function sharesEnd(a: Segment, b: Segment): boolean {
  const ends = (s: Segment): Array<[number, number]> => [
    [s.x0, s.y0],
    [s.x3, s.y3],
  ];
  return ends(a).some(([x, y]) => ends(b).some(([u, v]) => Math.abs(x - u) < EPSILON && Math.abs(y - v) < EPSILON));
}

function close(a: LoaderShape, b: LoaderShape, tolerance = 1e-2): boolean {
  return a.every((segment, i) =>
    (Object.keys(segment) as Array<keyof Segment>).every((key) => Math.abs(segment[key] - b[i][key]) < tolerance),
  );
}

afterEach(() => {
  document.body.innerHTML = '';
});

describe('loader shapes', () => {
  it('every shape has exactly four segments', () => {
    for (const status of STATUSES) expect(LOADER_SHAPES[status], status).toHaveLength(4);
  });

  it('mixShape is the first shape at 0 and the second at 1', () => {
    for (const from of STATUSES) {
      for (const to of STATUSES) {
        expect(mixShape(LOADER_SHAPES[from], LOADER_SHAPES[to], 0)).toEqual(LOADER_SHAPES[from]);
        expect(close(mixShape(LOADER_SHAPES[from], LOADER_SHAPES[to], 1), LOADER_SHAPES[to], EPSILON)).toBe(true);
      }
    }
  });

  it('check and cross strokes are straight and each stroke is two collinear halves', () => {
    for (const status of ['success', 'error'] as const) {
      const shape = LOADER_SHAPES[status];
      for (const segment of shape) expect(isStraight(segment), status).toBe(true);
      const pairs = shape.flatMap((a, i) => shape.slice(i + 1).map((b) => [a, b] as const));
      const halves = pairs.filter(([a, b]) => sharesEnd(a, b) && collinear(a, b));
      expect(halves.length, status).toBeGreaterThanOrEqual(2);
    }
  });

  it('the pending ring and the loading arcs lie on the r 8.5 ring around the center', () => {
    for (const status of ['pending', 'loading'] as const) {
      for (const segment of LOADER_SHAPES[status]) {
        expect(Math.hypot(segment.x0 - 12, segment.y0 - 12)).toBeCloseTo(8.5, 2);
        expect(Math.hypot(segment.x3 - 12, segment.y3 - 12)).toBeCloseTo(8.5, 2);
      }
    }
  });

  it('turnShape leaves the symmetric ring unchanged and is the identity at a whole turn', () => {
    for (const quarters of [1, 2, 3]) {
      expect(close(turnShape(LOADER_SHAPES.loading, quarters), LOADER_SHAPES.loading)).toBe(true);
      expect(close(turnShape(LOADER_SHAPES.pending, quarters), LOADER_SHAPES.pending)).toBe(true);
    }
    expect(turnShape(LOADER_SHAPES.success, 4)).toBe(LOADER_SHAPES.success);
  });

  it('segmentPath writes one cubic', () => {
    expect(segmentPath({ x0: 1, y0: 2, x1: 3, y1: 4, x2: 5, y2: 6, x3: 7, y3: 8.25 })).toBe('M 1 2 C 3 4 5 6 7 8.25');
  });
});

describe('MorphLoader under the harness', () => {
  it('renders the path of the given status and follows the prop', () => {
    const view = mount(createElement(MorphLoader, { status: 'loading' }));
    const svg = (): SVGSVGElement => view.container.querySelector('svg') as SVGSVGElement;
    const paths = (): string[] => [...svg().querySelectorAll('path')].map((path) => path.getAttribute('d') ?? '');
    expect(svg().getAttribute('data-status')).toBe('loading');
    expect(paths()).toEqual(LOADER_SHAPES.loading.map(segmentPath));

    view.rerender(createElement(MorphLoader, { status: 'success' }));
    expect(svg().getAttribute('data-status')).toBe('success');
    expect(paths()).toEqual(LOADER_SHAPES.success.map(segmentPath));

    view.rerender(createElement(MorphLoader, { status: 'error' }));
    expect(paths()).toEqual(LOADER_SHAPES.error.map(segmentPath));
  });

  it('is an image with a name when labelled, hidden otherwise', () => {
    const view = mount(createElement(MorphLoader, { status: 'success', label: 'Installed' }));
    const svg = view.container.querySelector('svg');
    expect(svg?.getAttribute('role')).toBe('img');
    expect(svg?.getAttribute('aria-label')).toBe('Installed');
    view.rerender(createElement(MorphLoader, { status: 'success' }));
    expect(view.container.querySelector('svg')?.getAttribute('aria-hidden')).toBe('true');
  });
});

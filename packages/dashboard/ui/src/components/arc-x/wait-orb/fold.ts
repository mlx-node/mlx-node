/**
 * The compaction orb: five dotted lines of text squeeze into one dark bar, then the next page fades in over it. Our
 * own shape, not from thinking-orbs; drawn by the same painter as the package's shapes.
 */

import { ORB_SIZE, type Dot } from './engine';

/** Seconds per page. */
const PERIOD = 2.4;
/** Share of a page spent squeezing, then the share after which the next page fades in. */
const SQUEEZE_END = 0.6;
const FADE_FROM = 0.75;
const ROWS = 5;
const COLS = 6;
/** Each line's length on a full page: ragged, like text. */
const LENGTHS = [1, 0.75, 0.9, 0.6, 0.8];
const C = ORB_SIZE / 2;

function easeInOut(x: number): number {
  return x < 0.5 ? 4 * x * x * x : 1 - (-2 * x + 2) ** 3 / 2;
}

function page(squeeze: number, alpha: number, z: number, dots: Dot[]): void {
  const pitch = 3.4 * (1 - 0.78 * squeeze);
  const width = 14 * (1 - 0.4 * squeeze);
  const left = C - width / 2;
  const r = 0.75 + 0.55 * squeeze;
  for (let row = 0; row < ROWS; row += 1) {
    const y = C + (row - (ROWS - 1) / 2) * pitch;
    const length = width * (LENGTHS[row]! + (1 - LENGTHS[row]!) * squeeze);
    for (let k = 0; k < COLS; k += 1) {
      dots.push({ x: left + (k / (COLS - 1)) * length, y, z, r, white: 0.5 - 0.42 * squeeze, a: alpha });
    }
  }
}

/** The dots at `seconds`, the leaving page first. */
export function foldFrame(seconds: number): Dot[] {
  const u = (((seconds / PERIOD) % 1) + 1) % 1;
  const squeeze = easeInOut(Math.min(1, u / SQUEEZE_END));
  const next = u < FADE_FROM ? 0 : (u - FADE_FROM) / (1 - FADE_FROM);
  const dots: Dot[] = [];
  page(squeeze, Math.max(0.02, 1 - next), 0, dots);
  if (next > 0) page(0, Math.max(0.02, next), 1, dots);
  return dots;
}

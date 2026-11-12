/**
 * The closed set of lines `server.js` may write to stderr. The agent saves the server's stderr to a file in its home,
 * and Node would print an uncaught error with its message (a `JSON.parse` error quotes the body it failed on, which
 * is web text), so nothing but these fixed lines is ever written.
 */

import { writeSync } from 'node:fs';

export const STDERR_LINES = {
  fatal: 'mlx_web: fatal\n',
  badFrame: 'mlx_web: bad frame\n',
  frameTooLarge: 'mlx_web: frame too large\n',
} as const;

export type StderrLine = keyof typeof STDERR_LINES;

/** Writes one fixed line to stderr; a closed stderr is ignored. */
export function writeStderrLine(line: StderrLine): void {
  try {
    writeSync(2, STDERR_LINES[line]);
  } catch {
    // stderr is gone; there is nobody left to tell.
  }
}

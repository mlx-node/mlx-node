/** `--decode-share SHARE`: Splash `parse_decode_share`. */

import { parsePythonFloat } from './idle-release.js';

/** Splash `EngineConfig::decodeShare`. */
export const DEFAULT_DECODE_SHARE = 0.5;

export const DECODE_SHARE_ERROR = '--decode-share must be a nonnegative number such as 0.5';

/**
 * Decode time owed per unit of prefill time while other requests generate:
 * a finite number of at least 0 (Python `float()` spelling). Throws
 * {@link DECODE_SHARE_ERROR} on anything else.
 */
export function parseDecodeShare(value: string): number {
  const share = parsePythonFloat(value) ?? Number.NaN;
  if (!(Number.isFinite(share) && share >= 0)) throw new Error(DECODE_SHARE_ERROR);
  return share;
}

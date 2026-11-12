/**
 * The turn journal's names and fingerprint syntax, shared by MAIN (which
 * reads the journal after a crash) and the sidecar (which writes it before
 * each turn). A leaf with no imports, like `exit-codes.ts`: MAIN must not pull
 * in anything the sidecar value-imports.
 *
 * The journal is Splash's `RuntimeCall.started` across a process boundary:
 * MAIN never sees an HTTP turn, so the sidecar writes one file per turn in
 * flight into the directory MAIN names, and MAIN reads what is left there when
 * the sidecar dies.
 */

/** The directory this generation's sidecar journals its turns in. */
export const TURN_JOURNAL_ENV = 'MLX_TURN_JOURNAL';
/** Comma-separated fingerprints the sidecar must refuse (Splash `Quarantine.refuses`). */
export const QUARANTINED_ENV = 'MLX_QUARANTINED';

const FINGERPRINT_LENGTH = 64;

/** A SHA-256 as 64 lower-case hex digits, the only form the server writes. */
export function isFingerprint(value: string): boolean {
  if (value.length !== FINGERPRINT_LENGTH) return false;
  for (let i = 0; i < value.length; i += 1) {
    const c = value.charCodeAt(i);
    const digit = c >= 48 && c <= 57;
    const lower = c >= 97 && c <= 102;
    if (!digit && !lower) return false;
  }
  return true;
}

/** Split a comma-separated list, keeping only fingerprints. */
export function parseFingerprintList(value: string): Set<string> {
  const out = new Set<string>();
  let start = 0;
  for (let i = 0; i <= value.length; i += 1) {
    if (i < value.length && value.charCodeAt(i) !== 44) continue;
    const item = value.slice(start, i);
    if (isFingerprint(item)) out.add(item);
    start = i + 1;
  }
  return out;
}

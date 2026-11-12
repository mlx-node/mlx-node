/**
 * MAIN's read of a dead sidecar's turn journal: the fingerprints of the turns
 * it had started (Splash `EngineFailure.started`). See
 * `src/inference/turn-journal.ts`.
 */

import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import { isFingerprint } from '../../inference/turn-journal.js';

export function readStartedFingerprints(dir: string): Set<string> {
  const started = new Set<string>();
  let names: string[];
  try {
    names = readdirSync(dir);
  } catch {
    return started;
  }
  for (const name of names) {
    let content: string;
    try {
      content = readFileSync(join(dir, name), 'utf8');
    } catch {
      continue;
    }
    if (isFingerprint(content)) started.add(content);
  }
  return started;
}

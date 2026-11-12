/**
 * The desktop sidecar's half of Splash's request quarantine
 * (server/supervision.py `Quarantine`, server/backend.py `_admit`).
 *
 * MAIN supervises this process and never sees an HTTP turn, so the server
 * writes one file per turn it dispatches into the directory MAIN named
 * (Splash `RuntimeCall.started`), and refuses the fingerprints MAIN handed it
 * at fork time (`Quarantine.refuses`). See
 * `packages/desktop/src/inference/turn-journal.ts`.
 */

import { createHash } from 'node:crypto';
import { unlinkSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import type { ChatConfig, ChatMessage } from '@mlx-node/core';

/** Splash `_admit`'s refusal, with mlx-node's restart in place of the server's. */
export const QUARANTINE_MESSAGE =
  'this request was in flight at two inference crashes in a row; mlx refuses it until inference is started again';

/**
 * Config keys that differ between a request and its retry, or only report
 * (Splash `_UNFINGERPRINTED`: identity and reporting, not content).
 */
const UNFINGERPRINTED_CONFIG: ReadonlySet<string> = new Set([
  'cacheOwnerId',
  'cacheRootOwnerId',
  'modelAlias',
  'reportPerformance',
]);

function contentDigest(bytes: Uint8Array): string {
  return createHash('sha256').update(bytes).digest('hex');
}

/**
 * Canonical JSON: keys sorted, `undefined` and functions dropped, bytes by
 * their SHA-256 (Splash `ImageSpan` digests, never the pixels themselves).
 */
function canonical(value: unknown, skip: ReadonlySet<string> | null): string {
  if (value instanceof Uint8Array) return JSON.stringify({ sha256: contentDigest(value) });
  if (Array.isArray(value)) {
    return `[${value.map((item) => (item === undefined || typeof item === 'function' ? 'null' : canonical(item, null))).join(',')}]`;
  }
  if (typeof value === 'object' && value !== null) {
    const record = value as Record<string, unknown>;
    const parts: string[] = [];
    for (const key of Object.keys(record).sort()) {
      const item = record[key];
      if (item === undefined || typeof item === 'function' || skip?.has(key) === true) continue;
      parts.push(`${JSON.stringify(key)}:${canonical(item, null)}`);
    }
    return `{${parts.join(',')}}`;
  }
  return JSON.stringify(value) ?? 'null';
}

/**
 * Splash `request_fingerprint`: the SHA-256 of what a turn asks the model,
 * which its retries share. Over the mapped messages and config rather than
 * prompt tokens: the endpoint has none before the model load.
 * `previousResponseId` stands for a stored chain not yet read.
 */
export function requestFingerprint(
  model: string,
  messages: readonly ChatMessage[],
  config: ChatConfig,
  previousResponseId?: string,
): string {
  const previous = previousResponseId === undefined ? '' : `,"previous":${JSON.stringify(previousResponseId)}`;
  return createHash('sha256')
    .update(
      `{"config":${canonical(config, UNFINGERPRINTED_CONFIG)},"messages":${canonical(messages, null)},"model":${JSON.stringify(model)}${previous}}`,
    )
    .digest('hex');
}

/** Splash `request_fingerprint` for a route whose body is passed through raw. */
export function rawRequestFingerprint(route: string, raw: string): string {
  return createHash('sha256')
    .update(`${JSON.stringify(route)}:${raw}`)
    .digest('hex');
}

export interface TurnJournalOptions {
  /** The directory MAIN reads after a crash. */
  dir: string;
  /** Fingerprints refused until inference is started again. */
  quarantined: ReadonlySet<string>;
}

export class TurnJournal {
  #seq = 0;

  constructor(private readonly options: TurnJournalOptions) {}

  /** Splash `Quarantine.refuses`. */
  refuses(fingerprint: string): boolean {
    return this.options.quarantined.has(fingerprint);
  }

  /**
   * Journal a turn being dispatched; returns its end, which is idempotent.
   * A write that fails runs the turn unjournaled: it is never quarantined.
   */
  begin(fingerprint: string): () => void {
    this.#seq += 1;
    const path = join(this.options.dir, String(this.#seq));
    try {
      writeFileSync(path, fingerprint);
    } catch (error) {
      console.error(`[mlx] turn journal write failed (${path}); the turn runs unjournaled:`, error);
      return () => {};
    }
    let ended = false;
    return () => {
      if (ended) return;
      ended = true;
      try {
        unlinkSync(path);
      } catch {
        // Gone with its directory.
      }
    };
  }
}

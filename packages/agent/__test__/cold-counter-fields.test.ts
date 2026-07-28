/**
 * Cross-language guard: every native cold-tier counter reaches the JSONL.
 *
 * The same defect happened twice. `coldCacheStats()` has always returned eight
 * counters and the provider forwarded four, so `corruptions` — the user's own
 * stated acceptance bar — was unreadable from `mlx agent` and the end-to-end
 * run had to drop to the API to check it. `ColdSidecarStats` had no napi
 * binding at all, so a run that restored blocks but silently failed to INSTALL
 * its sidecar (a full O(prefix) recurrent replay that produces correct state,
 * and therefore identical text, `cachedTokens`, `coldHits` and `coldCorruptions`)
 * was invisible to every JS consumer. Both are field-set NARROWING, and neither
 * breaks a type, a build, or any test that asserts on the fields it does carry.
 *
 * So the chain is pinned link by link, the way
 * `cold-tier-families.test.ts` pins the family allowlist to
 * `coldRestoreFamilies()`:
 *
 *   native structs  ──[1]──  COLD_DELTA_FIELDS  ──[2]──  JSONL line on disk
 *
 * [1] derives the expected names from `coldCacheStats()` / `coldSidecarStats()`
 *     AT RUNTIME. No field list is written down here; a list copied into the
 *     test would rot in lockstep with the list it guards.
 * [2] drives a real turn through the real provider into a real `MetricsTrace`
 *     and reads the file back. It must read the FILE, not a spy on `record()`:
 *     `record()` rebuilds its output from an allowlist, so a counter dropped
 *     there still shows up in a spy's argument and reaches nobody.
 */

import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { Api, Context, Model } from '@earendil-works/pi-ai';
import type { ExtensionAPI, ProviderConfig } from '@earendil-works/pi-coding-agent';
import { coldCacheStats, coldSidecarStats, type ColdCacheStats, type ColdSidecarStats } from '@mlx-node/core';
import type { ChatSession, ChatStreamEvent } from '@mlx-node/lm';
import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { createMlxProviderExtension } from '../src/provider/index.js';
import {
  COLD_COUNTER_FIELDS,
  COLD_DELTA_FIELDS,
  COLD_SIDECAR_FIELDS,
  MetricsTrace,
} from '../src/provider/metrics-trace.js';
import type { MlxModelHost } from '../src/provider/model-host.js';
import type { MlxModelInfo } from '../src/provider/models.js';

/**
 * The one `ColdCacheStats` field that is numeric but is NOT a counter: it
 * describes the tier's configured size, so its per-turn delta is always zero.
 * `enabled` and `root` are excluded structurally instead — they are not
 * numbers, so no name has to be written down for them.
 *
 * KNOWN HOLE, stated rather than papered over: this one name is hand-written,
 * and appending a real counter to it makes this test green again. That is the
 * standard way a drift guard gets "fixed" into a lie, and no derivation
 * available to JS can close it — neither JS nor Rust can ask a struct which of
 * its numeric fields are counters. It is kept to ONE name, spelled inline, so
 * the edit that would defeat the guard is a conspicuous one sitting under this
 * comment rather than a plausible-looking addition to a list of three.
 */
const NON_COUNTER_NUMERIC_FIELD = 'quotaBytes';

function prefixed(prefix: string, key: string): string {
  return `${prefix}${key.charAt(0).toUpperCase()}${key.slice(1)}`;
}

/**
 * The record field names the two native structs demand, derived from live
 * snapshots. BOTH tests below start here rather than from
 * `COLD_DELTA_FIELDS` — if the end-to-end test iterated the TS list, deleting
 * an entry from that list would shrink the list AND the assertions over it in
 * lockstep, and the narrowing this whole file exists to catch would pass twice.
 */
function nativeDeltaFieldNames(): { counters: string[]; sidecar: string[] } {
  const cache = coldCacheStats() as unknown as Record<string, unknown>;
  const sidecar = coldSidecarStats() as unknown as Record<string, unknown>;
  return {
    counters: Object.keys(cache)
      .filter((key) => typeof cache[key] === 'number' && key !== NON_COUNTER_NUMERIC_FIELD)
      .map((key) => prefixed('cold', key)),
    sidecar: Object.keys(sidecar).map((key) => prefixed('coldSidecar', key)),
  };
}

describe('cold-tier counter field set', () => {
  /**
   * The napi struct is generated from the Rust definition, so `Object.keys` of
   * a live snapshot IS the native field set — a stronger source than any list
   * a test could hold. It works with the tier closed: `coldCacheStats()` returns
   * a fully-populated default (`enabled: false`) rather than a partial object,
   * and `coldSidecarStats()` never consults the tier at all.
   */
  it('carries every native counter, from both structs, under a collision-free name', () => {
    const cache = coldCacheStats() as unknown as Record<string, unknown>;
    const sidecar = coldSidecarStats() as unknown as Record<string, unknown>;

    // The excluded name cannot rot into one the struct no longer has.
    expect(Object.keys(cache)).toContain(NON_COUNTER_NUMERIC_FIELD);
    // Every sidecar field is a counter — nothing to exclude, nothing to rot.
    for (const value of Object.values(sidecar)) expect(typeof value).toBe('number');

    const { counters: expectedCounters, sidecar: expectedSidecar } = nativeDeltaFieldNames();

    // A struct that lost its counters entirely would make both lists empty and
    // every comparison below vacuously true.
    expect(expectedCounters.length).toBeGreaterThan(0);
    expect(expectedSidecar.length).toBeGreaterThan(0);

    expect([...COLD_COUNTER_FIELDS].sort()).toEqual(expectedCounters.sort());
    expect([...COLD_SIDECAR_FIELDS].sort()).toEqual(expectedSidecar.sort());

    // Both structs carry `enqueued` and `queueDrops` counting DIFFERENT objects
    // (paged K/V blocks vs sidecars). The two prefixes are what keep them from
    // landing on one column and being silently summed.
    expect(new Set(COLD_DELTA_FIELDS).size).toBe(COLD_DELTA_FIELDS.length);
  });
});

/** A `ColdCacheStats` whose every counter is a distinct value. */
function coldSnapshot(base: number): ColdCacheStats {
  return {
    enabled: true,
    root: '',
    quotaBytes: 0,
    hits: base + 1,
    misses: base + 2,
    enqueued: base + 3,
    queueDrops: base + 4,
    bytesWritten: base + 5,
    bytesRestored: base + 6,
    evictions: base + 7,
    corruptions: base + 8,
    writeErrors: base + 9,
    restoreDeclines: base + 10,
  };
}

/** A `ColdSidecarStats` whose every counter is a distinct value. */
function sidecarSnapshot(base: number): ColdSidecarStats {
  return {
    captureReached: base + 11,
    chainEmpty: base + 12,
    boundarySkips: base + 13,
    alreadyPersisted: base + 14,
    enqueued: base + 15,
    queueDrops: base + 16,
    installed: base + 17,
    restoreSuppressed: base + 18,
  };
}

/**
 * Every field's expected delta, distinct from every other field's. Cross-wiring
 * — the copy-paste that reads `installed` into `coldSidecarEnqueued` — is only
 * observable when no two fields share a value, which is why the snapshots above
 * step by one and the turn's growth steps by a different amount per field.
 */
const EXPECTED_DELTAS: Record<string, number> = {
  coldHits: 100,
  coldMisses: 101,
  coldEnqueued: 102,
  coldQueueDrops: 103,
  coldBytesWritten: 104,
  coldBytesRestored: 105,
  coldEvictions: 106,
  coldCorruptions: 107,
  coldWriteErrors: 115,
  coldRestoreDeclines: 116,
  coldSidecarCaptureReached: 108,
  coldSidecarChainEmpty: 109,
  coldSidecarBoundarySkips: 110,
  coldSidecarAlreadyPersisted: 111,
  coldSidecarEnqueued: 112,
  coldSidecarQueueDrops: 113,
  coldSidecarInstalled: 114,
  coldSidecarRestoreSuppressed: 117,
};

/**
 * Grow each counter by its own field's expected delta. Built by NAME from the
 * same table the assertions read, so the fixture and the expectation can never
 * disagree about which field is which.
 */
function grow(cold: ColdCacheStats, sidecar: ColdSidecarStats): { cold: ColdCacheStats; sidecar: ColdSidecarStats } {
  const nextCold = { ...cold };
  for (const key of Object.keys(cold)) {
    const delta = EXPECTED_DELTAS[prefixed('cold', key)];
    if (delta !== undefined) nextCold[key as 'hits'] += delta;
  }
  const nextSidecar = { ...sidecar };
  for (const key of Object.keys(sidecar)) {
    const delta = EXPECTED_DELTAS[prefixed('coldSidecar', key)];
    if (delta !== undefined) nextSidecar[key as 'installed'] += delta;
  }
  return { cold: nextCold, sidecar: nextSidecar };
}

describe('cold-tier counters end to end', () => {
  let dir: string;
  const savedEnv = process.env.MLX_AGENT_METRICS;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'mlx-cold-counters-'));
    // The suite runs with MLX_AGENT_METRICS=0 (vite.config.ts), which would
    // make `record()` a no-op and this test vacuously green.
    delete process.env.MLX_AGENT_METRICS;
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
    if (savedEnv === undefined) delete process.env.MLX_AGENT_METRICS;
    else process.env.MLX_AGENT_METRICS = savedEnv;
  });

  /**
   * Reads the JSONL back rather than spying on `MetricsTrace.record`. A spy
   * would see the provider's argument and pass even if `record()`'s allowlist
   * dropped every cold field on the floor — which is the exact bug that shipped.
   */
  it('writes all eighteen per-turn deltas to the JSONL line', async () => {
    let cold = coldSnapshot(1000);
    let sidecar = sidecarSnapshot(2000);

    const session = {
      inFlight: false,
      history: [] as unknown[],
      lastImagesKey: null,
      lastAudioKey: null,
      turnCount: 0,
      unresolvedOkToolCallCount: null,
      needsFullReplay: false,
      contextLimits: () => undefined,
      supportsImages: () => false,
      primeHistory: () => undefined,
      // eslint-disable-next-line @typescript-eslint/require-await
      async *startFromHistoryStream(): AsyncGenerator<ChatStreamEvent> {
        // Inside the turn: after the start snapshot, before the terminal one.
        ({ cold, sidecar } = grow(cold, sidecar));
        yield {
          text: '',
          done: true,
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          numTokens: 2,
          promptTokens: 4,
          reasoningTokens: 0,
          rawText: '',
          cachedTokens: 0,
        };
      },
    } as unknown as ChatSession;

    let chain: Promise<unknown> = Promise.resolve();
    const host = {
      modelInfo: () => ({ name: 'qwen', path: '/models/qwen', modelType: 'qwen3_5' }),
      runWithResident<T>(_modelId: string, fn: (resident: ChatSession) => Promise<T>): Promise<T> {
        const result = chain.then(() => fn(session));
        chain = result.then(
          () => undefined,
          () => undefined,
        );
        return result;
      },
      markResidentDirty: () => undefined,
      consumeResidentDirty: () => false,
      invalidateResident: () => undefined,
    } as unknown as MlxModelHost;

    const modelInfo: MlxModelInfo = {
      discovered: { name: 'qwen', path: '/models/qwen', modelType: 'qwen3_5' },
      piModel: {
        id: 'qwen',
        name: 'qwen',
        reasoning: true,
        input: ['text'],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 262144,
        maxTokens: 8192,
      },
    };
    const model: Model<Api> = { ...modelInfo.piModel, api: 'mlx', provider: 'mlx', baseUrl: 'mlx://local' };
    const context: Context = { systemPrompt: '', messages: [] };

    const trace = new MetricsTrace({ dir });
    expect(trace.enabled).toBe(true);

    const extension = createMlxProviderExtension([modelInfo], host, {
      coldStats: () => cold,
      sidecarStats: () => sidecar,
      metricsTrace: trace,
    });
    if (typeof extension === 'function') throw new Error('expected a named extension');
    let provider: ProviderConfig | undefined;
    const pi = {
      registerProvider(_name: string, config: ProviderConfig): void {
        provider = config;
      },
      on(): void {
        // no lifecycle handlers needed here
      },
    } as unknown as ExtensionAPI;
    void extension.factory(pi);
    const streamSimple = provider?.streamSimple;
    if (!streamSimple) throw new Error('provider did not register streamSimple');

    for await (const _event of streamSimple(model, context, { sessionId: 's' })) {
      // drain
    }

    const lines = readFileSync(trace.currentFile(), 'utf8')
      .split('\n')
      .filter((line) => line.length > 0);
    expect(lines).toHaveLength(1);
    const line = JSON.parse(lines[0]) as Record<string, unknown>;

    // Driven off the NATIVE field names, not `COLD_DELTA_FIELDS`: a counter
    // deleted from that list must fail here too, and it cannot if the loop and
    // the list are the same object.
    const { counters, sidecar: sidecarFields } = nativeDeltaFieldNames();
    const expected = [...counters, ...sidecarFields];
    expect(expected.length).toBeGreaterThan(0);
    for (const field of expected) {
      // A native counter with no fixture entry is itself the failure: the
      // fixture would leave it flat and the line would read 0, not undefined.
      expect(EXPECTED_DELTAS[field]).toBeTypeOf('number');
      expect({ [field]: line[field] }).toEqual({ [field]: EXPECTED_DELTAS[field] });
    }
  });
});

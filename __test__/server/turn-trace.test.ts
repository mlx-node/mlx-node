/**
 * Per-turn trace production on the serve path.
 *
 * `SessionRegistry` wraps its model in {@link TurnTraceModel} once, so every
 * `ChatSession` it allocates — `/v1/responses` and `/v1/messages`, streaming
 * or not — appends one JSONL record per completed turn. These tests drive
 * the wrapper and the registry seam with a stub `SessionCapableModel` and a
 * temp-dir `MetricsTrace`.
 */

import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { ChatResult, ColdCacheStats, ColdSidecarStats } from '@mlx-node/core';
import type { ChatStreamEvent, SessionCapableModel } from '@mlx-node/lm';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { MetricsTrace, type MetricsTraceRecord } from '../../packages/server/src/metrics-trace.js';
import { SessionRegistry } from '../../packages/server/src/session-registry.js';
import { TurnTraceModel } from '../../packages/server/src/turn-trace.js';

const FIXED_NOW = Date.UTC(2026, 6, 21, 12, 0, 0); // 2026-07-21

function coldStats(overrides: Partial<ColdCacheStats> = {}): ColdCacheStats {
  return {
    enabled: true,
    root: '/private/tmp/cache/mlx-paged-v1',
    quotaBytes: 1_000_000,
    hits: 10,
    misses: 4,
    enqueued: 7,
    queueDrops: 1,
    bytesWritten: 2048,
    bytesRestored: 4096,
    evictions: 2,
    corruptions: 0,
    writeErrors: 0,
    restoreDeclines: 3,
    ...overrides,
  };
}

function sidecarStats(overrides: Partial<ColdSidecarStats> = {}): ColdSidecarStats {
  return {
    captureReached: 5,
    chainEmpty: 1,
    boundarySkips: 2,
    alreadyPersisted: 1,
    enqueued: 3,
    queueDrops: 0,
    installed: 4,
    restoreSuppressed: 0,
    ...overrides,
  };
}

function makeResult(overrides: Partial<ChatResult> = {}): ChatResult {
  return {
    text: 'ok',
    toolCalls: [],
    thinking: undefined,
    thinkingEnabled: true,
    numTokens: 7,
    promptTokens: 100,
    reasoningTokens: 2,
    finishReason: 'stop',
    rawText: 'ok',
    cachedTokens: 60,
    performance: {
      ttftMs: 125,
      prefillTokensPerSecond: 800,
      decodeTokensPerSecond: 50,
      mtpCycles: 3,
      mtpMeanAcceptedTokensTotal: 1.6,
    },
    ...overrides,
  };
}

function makeFinalEvent(overrides: Record<string, unknown> = {}): ChatStreamEvent {
  return {
    text: 'ok',
    done: true as const,
    finishReason: 'stop',
    toolCalls: [],
    thinking: null,
    thinkingEnabled: true,
    numTokens: 7,
    promptTokens: 100,
    reasoningTokens: 2,
    rawText: 'ok',
    cachedTokens: 60,
    ...overrides,
  } as ChatStreamEvent;
}

/** A `SessionCapableModel` stub with one scripted result per entry point. */
function makeInner(overrides: Partial<Record<keyof SessionCapableModel, unknown>> = {}): SessionCapableModel {
  const result = makeResult();
  return {
    chatSessionStart: vi.fn(async () => result),
    chatSessionContinue: vi.fn(async () => result),
    chatSessionContinueTool: vi.fn(async () => result),
    chatStreamSessionStart: vi.fn(async function* () {
      yield { text: 'o', done: false as const };
      yield makeFinalEvent();
    }),
    chatStreamSessionContinue: vi.fn(async function* () {
      yield makeFinalEvent();
    }),
    chatStreamSessionContinueTool: vi.fn(async function* () {
      yield makeFinalEvent();
    }),
    resetCaches: vi.fn(async () => {}),
    ...overrides,
  } as unknown as SessionCapableModel;
}

function traceFile(dir: string): string {
  return join(dir, `2026-07-21-${process.pid}.jsonl`);
}

function readTraceLines(dir: string): MetricsTraceRecord[] {
  return readFileSync(traceFile(dir), 'utf8')
    .split('\n')
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as MetricsTraceRecord);
}

describe('TurnTraceModel', () => {
  let dir: string;
  const savedEnv = process.env.MLX_AGENT_METRICS;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'mlx-turn-trace-'));
    delete process.env.MLX_AGENT_METRICS;
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
    if (savedEnv === undefined) delete process.env.MLX_AGENT_METRICS;
    else process.env.MLX_AGENT_METRICS = savedEnv;
  });

  it('records one allowlisted line per completed non-streaming turn', async () => {
    const inner = makeInner();
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    let coldCalls = 0;
    const model = new TurnTraceModel(inner, {
      modelName: 'qwen-small',
      trace,
      now: () => FIXED_NOW,
      // First call: turn-start baseline. Second: end-of-turn snapshot.
      coldStats: () => coldStats({ hits: 10 + coldCalls++ * 2 }),
      sidecarStats: () => sidecarStats(),
    });

    const result = await model.chatSessionContinue([{ role: 'user', content: 'hi' }], {
      cacheOwnerId: 'owner-1',
    });
    expect(result.finishReason).toBe('stop');

    const lines = readTraceLines(dir);
    expect(lines).toHaveLength(1);
    expect(lines[0]).toMatchObject({
      v: 1,
      sessionId: 'owner-1',
      model: 'qwen-small',
      finishReason: 'stop',
      promptTokens: 100,
      cachedTokens: 60,
      outputTokens: 7,
      reasoningTokens: 2,
      ttftMs: 125,
      prefillTps: 800,
      decodeTps: 50,
      mtpCycles: 3,
      mtpMeanAccepted: 1.6,
      // Baseline hits=10, end hits=12 → delta 2.
      coldHits: 2,
      coldMisses: 0,
      coldCorruptionsTotal: 0,
      coldEnabled: true,
      coldRoot: '/private/tmp/cache/mlx-paged-v1',
      coldSidecarCaptureReached: 0,
      durationMs: 0,
    });
    // Free text from the result must never reach the file.
    const raw = readFileSync(traceFile(dir), 'utf8');
    expect(raw).not.toContain('"text"');
    expect(raw).not.toContain('rawText');
  });

  it('records after the terminal streaming event and streams every event through', async () => {
    const inner = makeInner();
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    const model = new TurnTraceModel(inner, {
      modelName: 'qwen-small',
      trace,
      now: () => FIXED_NOW,
      coldStats: () => coldStats(),
      sidecarStats: () => sidecarStats(),
    });

    const seen: ChatStreamEvent[] = [];
    for await (const event of model.chatStreamSessionStart([{ role: 'user', content: 'hi' }])) {
      seen.push(event);
    }
    expect(seen).toHaveLength(2);
    expect(seen[1]).toMatchObject({ done: true, finishReason: 'stop' });

    const lines = readTraceLines(dir);
    expect(lines).toHaveLength(1);
    expect(lines[0]).toMatchObject({
      finishReason: 'stop',
      outputTokens: 7,
      cachedTokens: 60,
      // coldStats() returns the same snapshot at start and end → zero deltas.
      coldHits: 0,
      coldEnabled: true,
    });
    expect(lines[0].traceId).toBeTruthy();
  });

  it('writes nothing when the turn fails or ends on finishReason: error', async () => {
    const inner = makeInner({
      chatSessionStart: vi.fn(async () => makeResult({ finishReason: 'error' })),
      chatSessionContinueTool: vi.fn(async () => {
        throw new Error('native exploded');
      }),
      chatStreamSessionStart: vi.fn(async function* () {
        yield makeFinalEvent({ finishReason: 'error' });
      }),
      chatStreamSessionContinue: vi.fn(async function* () {
        yield { text: 'partial', done: false as const };
        throw new Error('mid-stream failure');
      }),
    });
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    const model = new TurnTraceModel(inner, { modelName: 'qwen-small', trace, now: () => FIXED_NOW });

    const errorResult = await model.chatSessionStart([]);
    expect(errorResult.finishReason).toBe('error');
    await expect(model.chatSessionContinueTool([])).rejects.toThrow('native exploded');
    for await (const _ of model.chatStreamSessionStart([])) {
      // drains the error-finish stream
    }
    await expect(async () => {
      for await (const _ of model.chatStreamSessionContinue([])) {
        // drains until the mid-stream throw
      }
    }).rejects.toThrow('mid-stream failure');

    expect(() => readFileSync(traceFile(dir), 'utf8')).toThrow();
  });

  it('writes nothing at all when the sink is disabled', async () => {
    process.env.MLX_AGENT_METRICS = '0';
    const inner = makeInner();
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    const model = new TurnTraceModel(inner, { modelName: 'qwen-small', trace, now: () => FIXED_NOW });

    await model.chatSessionStart([]);
    for await (const _ of model.chatStreamSessionStart([])) {
      // drains
    }
    expect(() => readFileSync(traceFile(dir), 'utf8')).toThrow();
  });

  it('forwards optional capability members verbatim and omits absent ones', async () => {
    const inner = makeInner({
      hasBlockPagedCache: vi.fn(() => true),
      hasMtpWeights: vi.fn(() => true),
      mtpAutoEnabled: vi.fn(() => false),
      maxConcurrentSequences: vi.fn(() => 4),
    });
    const model = new TurnTraceModel(inner, {
      modelName: 'qwen-small',
      trace: new MetricsTrace({ dir, now: () => FIXED_NOW }),
    });

    expect(model.hasBlockPagedCache?.()).toBe(true);
    expect(model.hasMtpWeights?.()).toBe(true);
    // The opt-out must reach ChatSession.mtpAutoDefaultAllowed unaltered —
    // a wrapper that dropped it would silently re-enable MTP on nemotron_h.
    expect(model.mtpAutoEnabled?.()).toBe(false);
    expect(model.maxConcurrentSequences?.()).toBe(4);
    // Not defined on the stub → must stay absent on the wrapper.
    expect(model.applyChatTemplate).toBeUndefined();
    expect(model.supportsImages).toBeUndefined();
    expect(model.releaseCacheOwner).toBeUndefined();
  });
});

describe('SessionRegistry trace hook', () => {
  let dir: string;
  const savedEnv = process.env.MLX_AGENT_METRICS;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'mlx-reg-trace-'));
    delete process.env.MLX_AGENT_METRICS;
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
    if (savedEnv === undefined) delete process.env.MLX_AGENT_METRICS;
    else process.env.MLX_AGENT_METRICS = savedEnv;
  });

  it('allocates sessions that record through the injected sink and modelName', async () => {
    const inner = makeInner();
    const reg = new SessionRegistry({
      model: inner,
      modelName: 'gemma4-26b',
      traceDeps: {
        trace: new MetricsTrace({ dir, now: () => FIXED_NOW }),
        now: () => FIXED_NOW,
        coldStats: () => coldStats(),
        sidecarStats: () => sidecarStats(),
      },
    });

    // The registry's session dispatches a turn through the wrapped model.
    const { session } = reg.getOrCreate(null, null);
    await session.send('hello');

    const lines = readTraceLines(dir);
    expect(lines).toHaveLength(1);
    expect(lines[0]).toMatchObject({
      model: 'gemma4-26b',
      finishReason: 'stop',
      outputTokens: 7,
      promptTokens: 100,
    });
    // ChatSession assigned a cache owner for the turn — it is the recorded
    // session id.
    expect(typeof lines[0].sessionId).toBe('string');
    expect((lines[0].sessionId as string).length).toBeGreaterThan(0);
  });
});

/**
 * Per-turn trace production on the serve path.
 *
 * `SessionRegistry` wraps its model in {@link TurnTraceModel} once, so every
 * `ChatSession` it allocates — `/v1/responses` and `/v1/messages`, streaming
 * or not — appends one JSONL record per completed turn. These tests drive
 * the wrapper and the registry seam with a stub `SessionCapableModel` and a
 * temp-dir `MetricsTrace`.
 */

import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { ChatResult, ChatStreamProgress, ColdCacheStats, ColdSidecarStats } from '@mlx-node/core';
import { ChatSession, type ChatStreamEvent, type ChatStreamProgressRef, type SessionCapableModel } from '@mlx-node/lm';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import { MetricsTrace, type MetricsTraceRecord } from '../../packages/server/src/metrics-trace.js';
import { SessionRegistry } from '../../packages/server/src/session-registry.js';
import {
  measuredPrefillOf,
  TurnTraceModel,
  turnCounters,
  type TracedChatConfig,
  type TurnTraceDeps,
} from '../../packages/server/src/turn-trace.js';

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
      watermarks: { cold: {}, sidecar: {} },
    });

    const result = await model.chatSessionContinue([{ role: 'user', content: 'hi' }], {
      cacheOwnerId: 'owner-1',
    });
    expect(result.finishReason).toBe('stop');

    const lines = readTraceLines(dir);
    expect(lines).toHaveLength(1);
    expect(lines[0]).toMatchObject({
      v: 2,
      pid: process.pid,
      outcome: 'ok',
      endpoint: 'other',
      stream: false,
      dispatch: 'continue',
      sessionId: 'owner-1',
      model: 'qwen-small',
      finishReason: 'stop',
      promptTokens: 100,
      cachedTokens: 60,
      outputTokens: 7,
      reasoningTokens: 2,
      ttftMs: 125,
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
    // A non-streamed turn has no progress handle, so no prefill window.
    expect('prefillMs' in lines[0]!).toBe(false);
    expect('prefillTokens' in lines[0]!).toBe(false);
    expect('prefillTps' in lines[0]!).toBe(false);
    // Free text from the result must never reach the file.
    const raw = readFileSync(traceFile(dir), 'utf8');
    expect(raw).not.toContain('"text"');
    expect(raw).not.toContain('rawText');
  });

  it('attributes a turn to the requesting alias stamped on the config', async () => {
    // One SessionRegistry (and one wrapper) serves every alias of a model;
    // the request's modelAlias must win over the binding name.
    const model = new TurnTraceModel(makeInner(), {
      modelName: 'qwen-small',
      trace: new MetricsTrace({ dir, now: () => FIXED_NOW }),
      now: () => FIXED_NOW,
      coldStats: () => coldStats(),
      sidecarStats: () => sidecarStats(),
      watermarks: { cold: {}, sidecar: {} },
    });
    await model.chatSessionContinue([{ role: 'user', content: 'hi' }], {
      modelAlias: 'qwen-small-q4',
    } as Parameters<typeof model.chatSessionContinue>[1]);

    const lines = readTraceLines(dir);
    expect(lines).toHaveLength(1);
    expect(lines[0].model).toBe('qwen-small-q4');
  });

  it('resolves the fallback modelName per turn so a surviving alias wins', async () => {
    // Aliased models share one registry built under the FIRST name; when
    // that name is unregistered the resolver must surface a live alias,
    // not keep stamping the dead one.
    let current: string | undefined = 'alias-a';
    const model = new TurnTraceModel(makeInner(), {
      modelName: () => current,
      trace: new MetricsTrace({ dir, now: () => FIXED_NOW }),
      now: () => FIXED_NOW,
      coldStats: () => coldStats(),
      sidecarStats: () => sidecarStats(),
      watermarks: { cold: {}, sidecar: {} },
    });
    current = 'alias-b';
    await model.chatSessionContinue([{ role: 'user', content: 'hi' }], null);

    const lines = readTraceLines(dir);
    expect(lines).toHaveLength(1);
    expect(lines[0].model).toBe('alias-b');
  });

  it('sums overlapping turns without double-counting shared counter advances', async () => {
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    // One mutable counter both turns read. Timeline: slow begins (seeds at
    // 10) → slow bumps +2 → fast begins → fast bumps +1 → fast finishes
    // (reads 13) → slow bumps +3 → slow finishes (reads 16). Real advance
    // across both turns is 16 − 10 = 6; naive per-turn baselines would
    // record 6 + 3 = 9, counting the +2 window twice.
    let counter = 10;
    const bump = (n: number) => {
      counter += n;
    };
    const sharedMarks = { cold: {}, sidecar: {} };
    const makeModel = (chatSessionContinue: () => Promise<ChatResult>) =>
      new TurnTraceModel(makeInner({ chatSessionContinue }), {
        modelName: 'qwen-small',
        trace,
        now: () => FIXED_NOW,
        coldStats: () => coldStats({ hits: counter }),
        sidecarStats: () => sidecarStats(),
        // The two wrappers share one marks store — same as every registry
        // wrapper in a serve process sharing SHARED_WATERMARKS.
        watermarks: sharedMarks,
      });
    const slow = makeModel(async () => {
      bump(2);
      const result = makeResult();
      bump(3);
      return result;
    });
    const fast = makeModel(async () => {
      bump(1);
      return makeResult();
    });

    const slowTurn = slow.chatSessionContinue([]);
    // Let slow's synchronous begin + inner dispatch run before fast begins.
    await Promise.resolve();
    await fast.chatSessionContinue([]);
    await slowTurn;

    const lines = readTraceLines(dir);
    expect(lines).toHaveLength(2);
    const hits = lines.map((l) => l.coldHits ?? 0);
    // Attribution may skew toward one turn; the sum is exact and monotone.
    expect(hits.reduce((a, b) => a + b, 0)).toBe(6);
    for (const h of hits) expect(h).toBeGreaterThanOrEqual(0);
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
      watermarks: { cold: {}, sidecar: {} },
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

  function deps(overrides: Partial<TurnTraceDeps> = {}): TurnTraceDeps {
    return {
      modelName: 'qwen-small',
      trace: new MetricsTrace({ dir, now: () => FIXED_NOW }),
      now: () => FIXED_NOW,
      coldStats: () => coldStats(),
      sidecarStats: () => sidecarStats(),
      watermarks: { cold: {}, sidecar: {} },
      memory: () => ({ activeBytes: 1000, peakBytes: 2000 }),
      ...overrides,
    };
  }

  function progress(overrides: Partial<ChatStreamProgress> = {}): ChatStreamProgress {
    return { phase: 'decode', prefillTotal: 40, prefillDone: 40, cachedTokens: 60, generatedTokens: 5, ...overrides };
  }

  it('records an error turn when the dispatch throws', async () => {
    const model = new TurnTraceModel(
      makeInner({
        chatSessionContinueTool: vi.fn(async () => {
          throw new Error('native exploded with secret prompt text');
        }),
      }),
      deps(),
    );
    await expect(model.chatSessionContinueTool([])).rejects.toThrow('native exploded');

    const lines = readTraceLines(dir);
    expect(lines).toHaveLength(1);
    expect(lines[0]).toMatchObject({
      outcome: 'error',
      errorKind: 'exception',
      finishReason: 'error',
      dispatch: 'continue_tool',
      stream: false,
      memActiveBytes: 1000,
      memPeakBytes: 2000,
    });
    // A non-stream turn has no progress snapshot to place the failure.
    expect('failPhase' in lines[0]!).toBe(false);
    expect(readFileSync(traceFile(dir), 'utf8')).not.toContain('secret');
  });

  it('records a cancelled turn when the signal aborted', async () => {
    const controller = new AbortController();
    const model = new TurnTraceModel(
      makeInner({
        chatSessionStart: vi.fn(async () => {
          controller.abort();
          throw new Error('aborted');
        }),
      }),
      deps(),
    );
    await expect(model.chatSessionStart([], null, controller.signal)).rejects.toThrow('aborted');

    const lines = readTraceLines(dir);
    expect(lines).toHaveLength(1);
    expect(lines[0]).toMatchObject({ outcome: 'cancelled', finishReason: 'cancelled' });
    expect('errorKind' in lines[0]!).toBe(false);
  });

  it('records a cancelled stream that ended without a terminal event', async () => {
    const model = new TurnTraceModel(
      makeInner({
        chatStreamSessionStart: vi.fn(async function* (
          _messages: unknown,
          _config: unknown,
          _signal: unknown,
          ref?: ChatStreamProgressRef,
        ) {
          const snap = progress({ phase: 'decode', generatedTokens: 3, decodeElapsedMs: 10, prefillElapsedMs: 5 });
          if (ref) ref.read = () => snap;
          try {
            yield { text: 'a', done: false as const };
            yield { text: 'b', done: false as const };
          } finally {
            if (ref) ref.read = null;
          }
        }),
      }),
      deps(),
    );
    // The consumer stops early: `return()` on the generator.
    for await (const _ of model.chatStreamSessionStart([])) break;

    const lines = readTraceLines(dir);
    expect(lines).toHaveLength(1);
    expect(lines[0]).toMatchObject({
      outcome: 'cancelled',
      finishReason: 'cancelled',
      stream: true,
      failPhase: 'decode',
      outputTokens: 3,
      cachedTokens: 60,
      promptTokens: 100,
      prefillDoneTokens: 40,
    });
  });

  it('records finishReason error as outcome error with errorKind finish_error', async () => {
    const model = new TurnTraceModel(
      makeInner({ chatSessionStart: vi.fn(async () => makeResult({ finishReason: 'error', numTokens: 4 })) }),
      deps(),
    );
    const result = await model.chatSessionStart([]);
    expect(result.finishReason).toBe('error');

    const lines = readTraceLines(dir);
    expect(lines).toHaveLength(1);
    expect(lines[0]).toMatchObject({
      outcome: 'error',
      errorKind: 'finish_error',
      finishReason: 'error',
      outputTokens: 4,
      promptTokens: 100,
    });
  });

  it('does not record the media-held restart reroute', async () => {
    const before = turnCounters();
    const model = new TurnTraceModel(
      makeInner({
        chatSessionContinue: vi.fn(async () => {
          throw new Error('IMAGE_CHANGE_REQUIRES_SESSION_RESTART: images changed');
        }),
      }),
      deps(),
    );
    await expect(model.chatSessionContinue([])).rejects.toThrow('IMAGE_CHANGE_REQUIRES_SESSION_RESTART');

    expect(existsSync(traceFile(dir))).toBe(false);
    expect(model.turnStats().live).toBe(0);
    const after = turnCounters();
    expect(after.error).toBe(before.error);
    expect(after.cancelled).toBe(before.cancelled);
  });

  it('failed turns carry no cold deltas and still advance the marks', async () => {
    let counter = 10;
    const shared = deps({ coldStats: () => coldStats({ hits: counter }), watermarks: { cold: {}, sidecar: {} } });
    const failing = new TurnTraceModel(
      makeInner({
        chatSessionContinue: vi.fn(async () => {
          counter += 3;
          throw new Error('native exploded');
        }),
      }),
      shared,
    );
    const passing = new TurnTraceModel(
      makeInner({
        chatSessionContinue: vi.fn(async () => {
          counter += 1;
          return makeResult();
        }),
      }),
      shared,
    );
    await expect(failing.chatSessionContinue([])).rejects.toThrow('native exploded');
    await passing.chatSessionContinue([]);

    const lines = readTraceLines(dir);
    expect(lines.map((l) => l.outcome)).toEqual(['error', 'ok']);
    for (const key of Object.keys(lines[0]!)) expect(key.startsWith('cold')).toBe(false);
    expect(lines[1]!.coldHits).toBe(1);
  });

  it('captures the last progress snapshot when the stream bridge clears the reader', async () => {
    const model = new TurnTraceModel(
      makeInner({
        chatStreamSessionContinue: vi.fn(async function* (
          _messages: unknown,
          _config: unknown,
          _signal: unknown,
          ref?: ChatStreamProgressRef,
        ) {
          let snap = progress({ phase: 'prefill', prefillDone: 10, generatedTokens: undefined, prefillElapsedMs: 3 });
          if (ref) ref.read = () => snap;
          try {
            yield { text: '', done: false as const };
            snap = progress({ phase: 'done', prefillDone: 20, generatedTokens: undefined, prefillElapsedMs: 8 });
            throw new Error('native failure in prefill');
          } finally {
            if (ref) ref.read = null;
          }
        }),
      }),
      deps(),
    );
    await expect(async () => {
      for await (const _ of model.chatStreamSessionContinue([])) {
        // drains until the throw
      }
    }).rejects.toThrow('native failure');

    const [line] = readTraceLines(dir);
    expect(line).toMatchObject({
      outcome: 'error',
      errorKind: 'exception',
      failPhase: 'prefill',
      prefillDoneTokens: 20,
    });
  });

  it("forwards progress to the caller's ref", async () => {
    const seen: Array<ChatStreamProgress | null> = [];
    const outer: ChatStreamProgressRef = { read: null };
    const model = new TurnTraceModel(
      makeInner({
        chatStreamSessionStart: vi.fn(async function* (
          _messages: unknown,
          _config: unknown,
          _signal: unknown,
          ref?: ChatStreamProgressRef,
        ) {
          const snap = progress();
          if (ref) ref.read = () => snap;
          try {
            yield { text: 'a', done: false as const };
            yield makeFinalEvent();
          } finally {
            if (ref) ref.read = null;
          }
        }),
      }),
      deps(),
    );
    for await (const event of model.chatStreamSessionStart([], null, undefined, outer)) {
      if (!event.done) seen.push(outer.read?.() ?? null);
    }
    expect(seen).toEqual([progress()]);
    expect(outer.read).toBeNull();
  });

  /** A streamed turn whose reader returns `snap` through the terminal event, then `after` until cleared. */
  function streamWith(
    snap: () => ChatStreamProgress,
    final: ChatStreamEvent = makeFinalEvent(),
    after?: ChatStreamProgress,
  ) {
    return makeInner({
      chatStreamSessionStart: vi.fn(async function* (
        _messages: unknown,
        _config: unknown,
        _signal: unknown,
        ref?: ChatStreamProgressRef,
      ) {
        let read = snap;
        if (ref) ref.read = () => read();
        try {
          yield { text: 'a', done: false as const };
          yield final;
          if (after !== undefined) read = () => after;
        } finally {
          if (ref) ref.read = null;
        }
      }),
    });
  }

  async function drainStart(model: TurnTraceModel): Promise<void> {
    for await (const _ of model.chatStreamSessionStart([])) {
      // drains
    }
  }

  const FINISHED = { phase: 'done', prefillTotal: 400, prefillElapsedMs: 250, decodeElapsedMs: 900 } as const;

  it('records the measured prefill window of a finished stream', async () => {
    const model = new TurnTraceModel(
      streamWith(() => ({ ...FINISHED })),
      deps(),
    );
    await drainStart(model);
    const [line] = readTraceLines(dir);
    expect(line).toMatchObject({ outcome: 'ok', stream: true, prefillMs: 250, prefillTokens: 400 });
    expect('prefillTps' in line!).toBe(false);
  });

  it('writes no prefill window when native never began a prefill', async () => {
    const model = new TurnTraceModel(
      streamWith(() => ({ phase: 'done', prefillElapsedMs: 250, decodeElapsedMs: 900 })),
      deps(),
    );
    await drainStart(model);
    const [line] = readTraceLines(dir);
    expect(line!.outcome).toBe('ok');
    expect('prefillMs' in line!).toBe(false);
    expect('prefillTokens' in line!).toBe(false);
  });

  it('writes no prefill window while the window is still running', async () => {
    const model = new TurnTraceModel(
      streamWith(() => ({ phase: 'done', prefillTotal: 400, prefillElapsedMs: 250 })),
      deps(),
    );
    await drainStart(model);
    const [line] = readTraceLines(dir);
    expect('prefillMs' in line!).toBe(false);
    expect('prefillTokens' in line!).toBe(false);
  });

  it('reads the prefill window at the terminal event, before the reader is cleared', async () => {
    const model = new TurnTraceModel(
      streamWith(() => ({ ...FINISHED }), makeFinalEvent(), { phase: 'done', prefillElapsedMs: 9 }),
      deps(),
    );
    await drainStart(model);
    const [line] = readTraceLines(dir);
    expect(line).toMatchObject({ prefillMs: 250, prefillTokens: 400 });
  });

  it('still records the turn when the progress reader throws', async () => {
    const model = new TurnTraceModel(
      streamWith(() => {
        throw new Error('handle gone');
      }),
      deps(),
    );
    await drainStart(model);
    const [line] = readTraceLines(dir);
    expect(line).toMatchObject({ outcome: 'ok', finishReason: 'stop', outputTokens: 7 });
    expect('prefillMs' in line!).toBe(false);
  });

  it('a finish_error stream carries no prefill window', async () => {
    const model = new TurnTraceModel(
      streamWith(() => ({ ...FINISHED }), makeFinalEvent({ finishReason: 'error', numTokens: 0 })),
      deps(),
    );
    await drainStart(model);
    const [line] = readTraceLines(dir);
    expect(line!.outcome).toBe('error');
    expect('prefillMs' in line!).toBe(false);
    expect('prefillTokens' in line!).toBe(false);
  });

  it('a cancelled stream carries no prefill window', async () => {
    const controller = new AbortController();
    const model = new TurnTraceModel(
      streamWith(() => ({ ...FINISHED })),
      deps(),
    );
    for await (const event of model.chatStreamSessionStart([], null, controller.signal)) {
      if (!event.done) {
        controller.abort();
        break;
      }
    }
    const [line] = readTraceLines(dir);
    expect(line!.outcome).toBe('cancelled');
    expect('prefillMs' in line!).toBe(false);
    expect('prefillTokens' in line!).toBe(false);
  });

  it('measuredPrefillOf rejects a snapshot that is not a measured window', () => {
    expect(measuredPrefillOf(null)).toBeUndefined();
    expect(measuredPrefillOf(undefined)).toBeUndefined();
    expect(measuredPrefillOf({ ...FINISHED, prefillTotal: 0 })).toBeUndefined();
    expect(measuredPrefillOf({ ...FINISHED, prefillElapsedMs: 0 })).toBeUndefined();
    expect(measuredPrefillOf({ ...FINISHED, prefillElapsedMs: Number.NaN })).toBeUndefined();
    expect(measuredPrefillOf({ ...FINISHED, prefillTotal: 1.5 })).toBeUndefined();
    expect(measuredPrefillOf({ ...FINISHED })).toEqual({ prefillMs: 250, prefillTokens: 400 });
  });

  it('stamps queueMs, loadWaitMs, resolveMs, preDispatchMs and endpoint from turnMeta', async () => {
    const model = new TurnTraceModel(makeInner(), deps());
    const config: TracedChatConfig = {
      turnMeta: { endpoint: 'responses', queueMs: 3, loadWaitMs: 4, resolveMs: 5, preDispatchMs: 15 },
      tools: [
        { type: 'function', function: { name: 'a' } },
        { type: 'function', function: { name: 'b' } },
      ] as TracedChatConfig['tools'],
    };
    await model.chatSessionStart(
      [{ role: 'user', content: 'x', images: [new Uint8Array(1), new Uint8Array(1)] }],
      config,
    );

    const [line] = readTraceLines(dir);
    expect(line).toMatchObject({
      endpoint: 'responses',
      queueMs: 3,
      loadWaitMs: 4,
      resolveMs: 5,
      preDispatchMs: 15,
      toolCount: 2,
      imageCount: 2,
      dispatch: 'start',
      toolCallCount: 0,
    });
  });

  it('records a turn with no turnMeta as endpoint other', async () => {
    const model = new TurnTraceModel(makeInner(), deps());
    await model.chatSessionStart([]);
    const [line] = readTraceLines(dir);
    expect(line!.endpoint).toBe('other');
    expect('queueMs' in line!).toBe(false);
  });

  it('a stream that ends in an error after decode began records failPhase decode', async () => {
    const model = new TurnTraceModel(
      makeInner({
        chatStreamSessionStart: vi.fn(async function* (
          _messages: unknown,
          _config: unknown,
          _signal: unknown,
          ref?: ChatStreamProgressRef,
        ) {
          // Native flips the phase to `done` before it sends the error.
          const snap = progress({ phase: 'done', prefillElapsedMs: 5, decodeElapsedMs: 9, generatedTokens: 6 });
          if (ref) ref.read = () => snap;
          try {
            yield { text: 'a', done: false as const };
            throw new Error('decode failure');
          } finally {
            if (ref) ref.read = null;
          }
        }),
      }),
      deps(),
    );
    await expect(async () => {
      for await (const _ of model.chatStreamSessionStart([])) {
        // drains until the throw
      }
    }).rejects.toThrow('decode failure');
    const [line] = readTraceLines(dir);
    expect(line).toMatchObject({ outcome: 'error', failPhase: 'decode', outputTokens: 6 });
  });

  it('a finish_error turn reads the live progress, not the captured one', async () => {
    const model = new TurnTraceModel(
      makeInner({
        chatStreamSessionStart: vi.fn(async function* (
          _messages: unknown,
          _config: unknown,
          _signal: unknown,
          ref?: ChatStreamProgressRef,
        ) {
          const snap = progress({ phase: 'done', prefillElapsedMs: 5, decodeElapsedMs: undefined, prefillDone: 12 });
          if (ref) ref.read = () => snap;
          try {
            yield makeFinalEvent({ finishReason: 'error', numTokens: 0 });
          } finally {
            if (ref) ref.read = null;
          }
        }),
      }),
      deps(),
    );
    for await (const _ of model.chatStreamSessionStart([])) {
      // drains
    }
    const [line] = readTraceLines(dir);
    expect(line).toMatchObject({
      outcome: 'error',
      errorKind: 'finish_error',
      failPhase: 'prefill',
      prefillDoneTokens: 12,
    });
  });

  it('stamps modelId, modelType and ctxWindow from identity, preferring contextLimits', async () => {
    const identity = { modelId: 'Qwen3.6-27B', modelType: 'qwen3_5', contextWindow: 131072 };
    const withLimits = new TurnTraceModel(
      makeInner({
        contextLimits: vi.fn(() => ({
          trainedWindowTokens: 262144,
          effectiveWindowTokens: 65536,
          pagedBlockCapacity: 0,
          pagedBlockSize: 0,
        })),
      }),
      deps({ identity }),
    );
    const withoutLimits = new TurnTraceModel(makeInner(), deps({ identity }));
    await withLimits.chatSessionStart([], { modelAlias: 'claude-haiku-4-5' } as TracedChatConfig);
    await withoutLimits.chatSessionStart([]);

    const lines = readTraceLines(dir);
    expect(lines[0]).toMatchObject({
      model: 'claude-haiku-4-5',
      modelId: 'Qwen3.6-27B',
      modelType: 'qwen3_5',
      ctxWindow: 65536,
    });
    expect(lines[1]).toMatchObject({ model: 'qwen-small', modelId: 'Qwen3.6-27B', ctxWindow: 131072 });
  });

  it('reports concurrency and concurrencyPeak for overlapping turns', async () => {
    const gates: Array<() => void> = [];
    const inner = makeInner({
      chatSessionContinue: vi.fn(
        () =>
          new Promise<ChatResult>((resolve) => {
            gates.push(() => resolve(makeResult()));
          }),
      ),
    });
    const model = new TurnTraceModel(inner, deps());
    const first = model.chatSessionContinue([]);
    const second = model.chatSessionContinue([]);
    expect(model.turnStats()).toMatchObject({ live: 2, begun: 2 });
    gates[0]!();
    await first;
    const third = model.chatSessionContinue([]);
    gates[1]!();
    await second;
    gates[2]!();
    await third;

    const lines = readTraceLines(dir);
    expect(lines.map((l) => [l.concurrency, l.concurrencyPeak])).toEqual([
      [1, 2],
      [2, 2],
      [2, 2],
    ]);
    expect(model.turnStats()).toMatchObject({ live: 0, begun: 3 });
  });

  it('credits streamed tokens to the process counter exactly once', async () => {
    let snap = progress({ generatedTokens: 0 });
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    const model = new TurnTraceModel(
      makeInner({
        chatStreamSessionStart: vi.fn(async function* (
          _messages: unknown,
          _config: unknown,
          _signal: unknown,
          ref?: ChatStreamProgressRef,
        ) {
          if (ref) ref.read = () => snap;
          try {
            yield { text: 'a', done: false as const };
            await gate;
            yield makeFinalEvent({ numTokens: 10 });
          } finally {
            if (ref) ref.read = null;
          }
        }),
      }),
      deps(),
    );
    const before = turnCounters().outputTokens;
    const iterator = model.chatStreamSessionStart([])[Symbol.asyncIterator]();
    await iterator.next();
    snap = progress({ generatedTokens: 4 });
    const [live] = model.liveTurns();
    expect(live).toMatchObject({ model: 'qwen-small', stream: true, dispatch: 'start', progress: snap });
    expect(turnCounters().outputTokens - before).toBe(4);
    model.liveTurns();
    expect(turnCounters().outputTokens - before).toBe(4);
    snap = progress({ generatedTokens: 7 });
    model.liveTurns();
    expect(turnCounters().outputTokens - before).toBe(7);
    release();
    while (!(await iterator.next()).done) {
      // drains
    }
    expect(turnCounters().outputTokens - before).toBe(10);
    expect(model.liveTurns()).toEqual([]);
  });

  it("consumes a failed turn's counter increments instead of leaking them to the next record", async () => {
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    // Timeline: marks seed at 10 → failed turn bumps +3 and throws →
    // successful turn bumps +1 → records. Without discard-turn mark
    // advances the success would report 4 (the failed turn's +3 folded in);
    // with them it reports only its own +1.
    let counter = 10;
    const bump = (n: number) => {
      counter += n;
    };
    const sharedMarks = { cold: {}, sidecar: {} };
    const deps = {
      modelName: 'qwen-small',
      trace,
      now: () => FIXED_NOW,
      coldStats: () => coldStats({ hits: counter }),
      sidecarStats: () => sidecarStats(),
      watermarks: sharedMarks,
    };
    const failing = new TurnTraceModel(
      makeInner({
        chatSessionContinue: vi.fn(async () => {
          bump(3);
          throw new Error('native exploded');
        }),
      }),
      deps,
    );
    const passing = new TurnTraceModel(
      makeInner({
        chatSessionContinue: vi.fn(async () => {
          bump(1);
          return makeResult();
        }),
      }),
      deps,
    );

    await expect(failing.chatSessionContinue([])).rejects.toThrow('native exploded');
    await passing.chatSessionContinue([]);

    const lines = readTraceLines(dir).filter((l) => l.outcome === 'ok');
    expect(lines).toHaveLength(1);
    expect(lines[0]!.coldHits).toBe(1);
  });

  it('a failed turn mid-overlap consumes nothing still inside a live turn\u2019s window', async () => {
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    // Timeline: slow begins (seeds at 10) → fast begins → fast bumps +3 and
    // throws → slow bumps +3 and finishes (reads 16). The failed turn's
    // discard must floor its mark advance at slow's baseline (10): keying on
    // the current counter (13) would eat +3 of slow's own window and slow
    // would record 3 instead of the real 6.
    let counter = 10;
    const bump = (n: number) => {
      counter += n;
    };
    const sharedMarks = { cold: {}, sidecar: {} };
    const deps = {
      modelName: 'qwen-small',
      trace,
      now: () => FIXED_NOW,
      coldStats: () => coldStats({ hits: counter }),
      sidecarStats: () => sidecarStats(),
      watermarks: sharedMarks,
    };
    let releaseSlow!: () => void;
    const slowGate = new Promise<void>((res) => {
      releaseSlow = res;
    });
    const slow = new TurnTraceModel(
      makeInner({
        chatSessionContinue: vi.fn(async () => {
          await slowGate;
          bump(3);
          return makeResult();
        }),
      }),
      deps,
    );
    const fast = new TurnTraceModel(
      makeInner({
        chatSessionContinue: vi.fn(async () => {
          bump(3);
          throw new Error('native exploded');
        }),
      }),
      deps,
    );

    const slowTurn = slow.chatSessionContinue([]);
    // Let slow's begin snapshot land before fast dispatches.
    await Promise.resolve();
    await expect(fast.chatSessionContinue([])).rejects.toThrow('native exploded');
    releaseSlow();
    await slowTurn;

    const lines = readTraceLines(dir).filter((l) => l.outcome === 'ok');
    expect(lines).toHaveLength(1);
    expect(lines[0]!.coldHits).toBe(6);
  });

  it("consumes an error-finish turn's increments the same way", async () => {
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    let counter = 10;
    const bump = (n: number) => {
      counter += n;
    };
    const sharedMarks = { cold: {}, sidecar: {} };
    const deps = {
      modelName: 'qwen-small',
      trace,
      now: () => FIXED_NOW,
      coldStats: () => coldStats({ hits: counter }),
      sidecarStats: () => sidecarStats(),
      watermarks: sharedMarks,
    };
    const erroring = new TurnTraceModel(
      makeInner({
        chatSessionContinue: vi.fn(async () => {
          bump(3);
          return makeResult({ finishReason: 'error' });
        }),
      }),
      deps,
    );
    const passing = new TurnTraceModel(
      makeInner({
        chatSessionContinue: vi.fn(async () => {
          bump(1);
          return makeResult();
        }),
      }),
      deps,
    );

    const errorResult = await erroring.chatSessionContinue([]);
    expect(errorResult.finishReason).toBe('error');
    await passing.chatSessionContinue([]);

    const lines = readTraceLines(dir).filter((l) => l.outcome === 'ok');
    expect(lines).toHaveLength(1);
    expect(lines[0]!.coldHits).toBe(1);
  });

  it('writes nothing at all when the sink is disabled', async () => {
    process.env.MLX_AGENT_METRICS = '0';
    const inner = makeInner();
    const trace = new MetricsTrace({ dir, now: () => FIXED_NOW });
    const model = new TurnTraceModel(inner, {
      modelName: 'qwen-small',
      trace,
      now: () => FIXED_NOW,
      watermarks: { cold: {}, sidecar: {} },
    });

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
      watermarks: { cold: {}, sidecar: {} },
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

  it.each([
    { className: 'NemotronHModel', autoEnabled: false, enableMtp: undefined },
    { className: 'Qwen35Model', autoEnabled: true, enableMtp: true },
  ])(
    'preserves the inner proto-walk MTP verdict for a getter-less $className',
    async ({ className, autoEnabled, enableMtp }) => {
      // The wrapper's own `TurnTraceModel` proto matches nothing, so
      // `ChatSession`'s suppressed-class fallback must run on the INNER
      // chain — the synthesized `mtpAutoEnabled` forwarder is what carries it.
      const Ctor = { [className]: class {} }[className]!;
      const inner = Object.assign(Object.create(Ctor.prototype) as object, makeInner(), {
        hasMtpWeights: vi.fn(() => true),
      }) as SessionCapableModel;
      const model = new TurnTraceModel(inner, {
        modelName: 'm',
        trace: new MetricsTrace({ dir, now: () => FIXED_NOW }),
        watermarks: { cold: {}, sidecar: {} },
      });

      expect(model.mtpAutoEnabled?.()).toBe(autoEnabled);

      const session = new ChatSession(model);
      await session.send('Hello');
      const [, config] = (inner.chatSessionStart as ReturnType<typeof vi.fn>).mock.calls[0]!;
      expect(config?.enableMtp).toBe(enableMtp);
    },
  );
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
        watermarks: { cold: {}, sidecar: {} },
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

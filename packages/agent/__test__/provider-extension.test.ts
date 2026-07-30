import { mkdtempSync, realpathSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import type { Api, Context, Model, SimpleStreamOptions } from '@earendil-works/pi-ai';
import type {
  ExtensionAPI,
  ExtensionContext,
  InlineExtension,
  ProviderConfig,
  SessionStartEvent,
} from '@earendil-works/pi-coding-agent';
import type { ColdCacheStats } from '@mlx-node/core';
import type { ChatConfig, ChatSession, ChatStreamEvent } from '@mlx-node/lm';
import { describe, expect, it, vi } from 'vite-plus/test';

import { createMlxProviderExtension } from '../src/provider/index.js';
import { MetricsTrace, type MetricsTraceRecord } from '../src/provider/metrics-trace.js';
import type { MlxModelHost } from '../src/provider/model-host.js';
import type { MlxModelInfo } from '../src/provider/models.js';

/**
 * A REAL directory, so `canonicalCacheRoot` resolves it through realpath the
 * way the running tier's root would be (on macOS `/var/folders/...` →
 * `/private/var/folders/...`, which is exactly the mismatch a raw string
 * compare would hit).
 */
const coldRootDir = mkdtempSync(join(tmpdir(), 'mlx-cold-root-'));

/** A ColdCacheStats snapshot carrying only the counters the provider diffs. */
function coldSnapshot(over: Partial<ColdCacheStats>): ColdCacheStats {
  return {
    enabled: true,
    root: '',
    quotaBytes: 0,
    hits: 0,
    misses: 0,
    enqueued: 0,
    queueDrops: 0,
    bytesWritten: 0,
    bytesRestored: 0,
    evictions: 0,
    corruptions: 0,
    writeErrors: 0,
    restoreDeclines: 0,
    ...over,
  };
}

/**
 * Drive ONE successful turn through the real provider extension and return the
 * records it wrote. `duringTurn` runs inside the streaming generator, i.e.
 * between the turn-start cold snapshot and the terminal one, so it simulates
 * exactly the activity a turn's delta should capture.
 */
async function buildOneTurnHarness(
  readCold: () => ColdCacheStats,
  duringTurn: () => void,
): Promise<{ records: Array<Omit<MetricsTraceRecord, 'v'>>; run: () => Promise<void> }> {
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
      duringTurn();
      yield {
        text: '',
        done: true,
        finishReason: 'stop',
        toolCalls: [],
        thinking: null,
        thinkingEnabled: true,
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

  const trace = new MetricsTrace({ dir: join(tmpdir(), 'mlx-cold-counter-test') });
  const records: Array<Omit<MetricsTraceRecord, 'v'>> = [];
  vi.spyOn(trace, 'record').mockImplementation((rec) => {
    records.push(rec);
  });

  const extension = createMlxProviderExtension([modelInfo], host, { coldStats: readCold, metricsTrace: trace });
  if (typeof extension === 'function') throw new Error('expected a named extension');
  let provider: ProviderConfig | undefined;
  const pi = {
    registerProvider(_name: string, config: ProviderConfig): void {
      provider = config;
    },
    on(): void {
      // no lifecycle handlers needed for this test
    },
  } as unknown as ExtensionAPI;
  void extension.factory(pi);
  const streamSimple = provider?.streamSimple;
  if (!streamSimple) throw new Error('provider did not register streamSimple');

  return {
    records,
    run: async () => {
      for await (const _e of streamSimple(model, context, { sessionId: 's' })) {
        // drain
      }
    },
  };
}

function loadExtension(): {
  handlers: Map<string, (event: never, ctx: ExtensionContext) => void>;
  registerProvider: ReturnType<typeof vi.fn>;
} {
  const handlers = new Map<string, (event: never, ctx: ExtensionContext) => void>();
  const registerProvider = vi.fn();
  const pi = {
    registerProvider,
    on(event: string, handler: (event: never, ctx: ExtensionContext) => void): void {
      handlers.set(event, handler);
    },
  } as unknown as ExtensionAPI;
  const extension: InlineExtension = createMlxProviderExtension([], {} as MlxModelHost);
  if (typeof extension === 'function') throw new Error('expected a named extension');
  void extension.factory(pi);
  return { handlers, registerProvider };
}

describe('createMlxProviderExtension', () => {
  it('registers the provider and all performance-status lifecycle handlers', () => {
    const { handlers, registerProvider } = loadExtension();

    expect(registerProvider).toHaveBeenCalledOnce();
    expect([...handlers.keys()]).toEqual(['session_start', 'message_end', 'model_select', 'session_shutdown']);
  });

  it('retains the last completed sample through a tool-loop turn boundary', () => {
    const { handlers } = loadExtension();

    // Pi starts a new turn after a tool result. There must be no turn_start
    // clear handler: the in-flight response has no replacement metrics until
    // its terminal event, so the latest completed sample stays informative.
    expect(handlers.has('turn_start')).toBe(false);
  });

  it.each(['model_select', 'session_shutdown'])('clears stale TUI performance on %s', (event) => {
    const { handlers } = loadExtension();
    const setStatus = vi.fn();
    const ctx = { mode: 'tui', ui: { setStatus } } as unknown as ExtensionContext;

    handlers.get(event)!({} as never, ctx);

    expect(setStatus).toHaveBeenCalledWith('mlx-performance', undefined);
  });

  it('maps startup/new/resume roots separately from child stream ownership', async () => {
    const configs: ChatConfig[] = [];
    const session = {
      inFlight: false,
      history: [],
      lastImagesKey: null,
      lastAudioKey: null,
      turnCount: 0,
      unresolvedOkToolCallCount: null,
      needsFullReplay: false,
      contextLimits: () => undefined,
      primeHistory: () => undefined,
      startFromHistoryStream(config: ChatConfig): AsyncGenerator<ChatStreamEvent> {
        configs.push(config);
        return (async function* () {
          yield {
            text: '',
            done: true,
            finishReason: 'stop',
            toolCalls: [],
            thinking: null,
            thinkingEnabled: true,
            numTokens: 1,
            promptTokens: 1,
            reasoningTokens: 0,
            rawText: '',
          };
        })();
      },
    } as unknown as ChatSession;
    const host = {
      modelInfo: () => ({ name: 'qwen', path: '/models/qwen', modelType: 'qwen3_5' }),
      runWithResident: async <T>(_modelId: string, fn: (resident: ChatSession) => Promise<T>): Promise<T> =>
        await fn(session),
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
    const model: Model<Api> = {
      ...modelInfo.piModel,
      api: 'mlx',
      provider: 'mlx',
      baseUrl: 'mlx://local',
    };
    const context: Context = { systemPrompt: '', messages: [] };
    const extension = createMlxProviderExtension([modelInfo], host);
    if (typeof extension === 'function') throw new Error('expected a named extension');

    const bindRuntime = async (
      reason: Extract<SessionStartEvent['reason'], 'startup' | 'new' | 'resume'>,
      rootSessionId: string,
      activeSessionId: string,
    ): Promise<void> => {
      const handlers = new Map<string, (event: SessionStartEvent, ctx: ExtensionContext) => void>();
      let provider: ProviderConfig | undefined;
      const pi = {
        registerProvider(_name: string, config: ProviderConfig): void {
          provider = config;
        },
        on(event: string, handler: (event: SessionStartEvent, ctx: ExtensionContext) => void): void {
          handlers.set(event, handler);
        },
      } as unknown as ExtensionAPI;
      void extension.factory(pi);
      handlers.get('session_start')!({ type: 'session_start', reason }, {
        sessionManager: { getSessionId: () => rootSessionId },
      } as unknown as ExtensionContext);
      const streamSimple = provider?.streamSimple;
      if (!streamSimple) throw new Error('provider did not register streamSimple');
      const stream = streamSimple(model, context, { sessionId: activeSessionId } satisfies SimpleStreamOptions);
      for await (const _event of stream) {
        // Drain the provider stream so the recorded native config is final.
      }
    };

    await bindRuntime('startup', 'root-0', 'root-0');
    await bindRuntime('new', 'root-1', 'child-of-root-1');
    await bindRuntime('resume', 'root-2', 'root-2');

    expect(configs.map(({ cacheOwnerId, cacheRootOwnerId }) => ({ cacheOwnerId, cacheRootOwnerId }))).toEqual([
      { cacheOwnerId: 'root-0', cacheRootOwnerId: 'root-0' },
      { cacheOwnerId: 'child-of-root-1', cacheRootOwnerId: 'root-1' },
      { cacheOwnerId: 'root-2', cacheRootOwnerId: 'root-2' },
    ]);
  });

  // Finding 11a: an aborted turn's cold-tier activity must not be attributed to
  // the next successful turn. The recorded delta is computed against a snapshot
  // taken at THIS turn's native start, so a prior aborted turn's restores are
  // already baked into the baseline.
  it('records only the successful turn own cold-cache deltas after an aborted turn', async () => {
    const cold = { hits: 0, bytesRestored: 0 };
    const readCold = (): ColdCacheStats => coldSnapshot(cold);

    const controller = new AbortController();
    const scripts: Array<(config: ChatConfig) => AsyncGenerator<ChatStreamEvent>> = [
      // Turn 1: restores blocks (cold activity) then the user aborts — no record.
      // eslint-disable-next-line @typescript-eslint/require-await
      async function* () {
        cold.hits = 5;
        cold.bytesRestored = 500;
        yield { text: 'partial', done: false };
        controller.abort();
      },
      // Turn 2: its OWN restores, then a clean final — records here.
      // eslint-disable-next-line @typescript-eslint/require-await
      async function* () {
        cold.hits = 8;
        cold.bytesRestored = 800;
        yield {
          text: '',
          done: true,
          finishReason: 'stop',
          toolCalls: [],
          thinking: null,
          thinkingEnabled: true,
          numTokens: 2,
          promptTokens: 4,
          reasoningTokens: 0,
          rawText: '',
          cachedTokens: 0,
        };
      },
    ];
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
      startFromHistoryStream(config: ChatConfig): AsyncGenerator<ChatStreamEvent> {
        const script = scripts.shift();
        if (!script) throw new Error('no script left');
        return script(config);
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

    const trace = new MetricsTrace({ dir: '/tmp/mlx-11a-test' });
    const records: Array<Omit<MetricsTraceRecord, 'v' | 'rootSessionId' | 'rootSessionFile'>> = [];
    vi.spyOn(trace, 'record').mockImplementation((rec) => {
      records.push(rec);
    });

    const extension = createMlxProviderExtension([modelInfo], host, { coldStats: readCold, metricsTrace: trace });
    if (typeof extension === 'function') throw new Error('expected a named extension');
    let provider: ProviderConfig | undefined;
    const pi = {
      registerProvider(_name: string, config: ProviderConfig): void {
        provider = config;
      },
      on(): void {
        // no lifecycle handlers needed for this test
      },
    } as unknown as ExtensionAPI;
    void extension.factory(pi);
    const streamSimple = provider?.streamSimple;
    if (!streamSimple) throw new Error('provider did not register streamSimple');

    // Turn 1: aborts mid-stream (records nothing).
    for await (const _e of streamSimple(model, context, { sessionId: 's', signal: controller.signal })) {
      // drain
    }
    // Turn 2: succeeds and records its own deltas.
    for await (const _e of streamSimple(model, context, { sessionId: 's' })) {
      // drain
    }

    expect(records).toHaveLength(1);
    // Only turn 2's own activity: 8-5 hits, 800-500 restored — NOT 8 / 800.
    expect(records[0].coldHits).toBe(3);
    expect(records[0].coldBytesRestored).toBe(300);
  });

  /**
   * `coldCacheStats()` returns 11 fields; the recorder kept 4. `corruptions`,
   * `evictions`, `enqueued` and `queueDrops` were read into a local and dropped
   * when it went out of scope, which is why the stated acceptance bar
   * ("corruptions must be 0") could not be checked from anything shipped.
   *
   * `root` was dropped too, which is what made the Cache page unable to tell
   * ITS cache's hit rate from every cache the machine had ever used.
   */
  it('records every cold-tier counter as a per-turn delta, plus the cumulative latches', async () => {
    const cold = {
      hits: 1,
      misses: 1,
      bytesWritten: 10,
      bytesRestored: 10,
      enqueued: 100,
      queueDrops: 4,
      evictions: 7,
      corruptions: 2,
    };
    const readCold = (): ColdCacheStats => coldSnapshot({ ...cold, root: coldRootDir });

    const { records, run } = await buildOneTurnHarness(readCold, () => {
      // The turn's own activity, on top of a non-zero pre-turn baseline.
      cold.hits = 4;
      cold.misses = 3;
      cold.bytesWritten = 40;
      cold.bytesRestored = 90;
      cold.enqueued = 106;
      cold.queueDrops = 5;
      cold.evictions = 9;
      cold.corruptions = 3;
    });
    await run();

    expect(records).toHaveLength(1);
    const rec = records[0];
    // Deltas, never absolutes: a baseline of 100 enqueued must not be reported.
    expect(rec.coldEnqueued).toBe(6);
    expect(rec.coldQueueDrops).toBe(1);
    expect(rec.coldEvictions).toBe(2);
    expect(rec.coldCorruptions).toBe(1);
    // Cumulative latches ARE absolutes, deliberately: a turn that aborts never
    // reaches this recorder, so its corruption lands in no delta anywhere.
    expect(rec.coldCorruptionsTotal).toBe(3);
    expect(rec.coldQueueDropsTotal).toBe(5);
    // Identity, canonicalized by the writer so the reader never has to match
    // whatever spelling the native side constructed.
    expect(rec.coldEnabled).toBe(true);
    expect(rec.coldRoot).toBe(realpathSync(coldRootDir));
  });

  // The tier opens LAZILY, so on a process's first turn the START snapshot is
  // the all-zero default (enabled: false, root: ''). Reading identity from the
  // baseline instead of the end-of-turn snapshot would stamp every first turn
  // as unattributed.
  it('takes the cache identity from the end-of-turn snapshot, not the lazy-open baseline', async () => {
    const state = { enabled: false, root: '' };
    const readCold = (): ColdCacheStats => coldSnapshot({ enabled: state.enabled, root: state.root });

    const { records, run } = await buildOneTurnHarness(readCold, () => {
      // The tier opens during the turn.
      state.enabled = true;
      state.root = coldRootDir;
    });
    await run();

    expect(records[0].coldEnabled).toBe(true);
    expect(records[0].coldRoot).toBe(realpathSync(coldRootDir));
  });

  // A turn that ran with the tier off must be recorded as OFF (a known,
  // attributable fact) and carry no cache identity — never as an empty-string
  // root, which would land in a bucket no dashboard root can ever match.
  it('records a disabled tier without a cache identity', async () => {
    const readCold = (): ColdCacheStats => coldSnapshot({ enabled: false, root: '' });
    const { records, run } = await buildOneTurnHarness(readCold, () => undefined);
    await run();

    expect(records[0].coldEnabled).toBe(false);
    expect(records[0].coldRoot).toBeUndefined();
  });

  /**
   * The writer's emptiness test and `canonicalCacheRoot`'s must be the SAME
   * test. `canonicalCacheRoot` trims, so a whitespace-only native root
   * canonicalizes to `''`; gating on the RAW string passed `length > 0` and
   * assigned that empty string, `MetricsTrace.record` then dropped the field,
   * and the row landed as "tier ON, no root" — a shape the dashboard's scope
   * partition had no arm for, so its lookups were reported by nothing at all.
   */
  it('never records an empty cache identity for a whitespace-only native root', async () => {
    const readCold = (): ColdCacheStats => coldSnapshot({ enabled: true, root: '   ' });
    const { records, run } = await buildOneTurnHarness(readCold, () => undefined);
    await run();

    expect(records[0].coldEnabled).toBe(true);
    // Not `''`: an empty root must never be assigned in the first place.
    expect(records[0].coldRoot).toBeUndefined();
  });
});

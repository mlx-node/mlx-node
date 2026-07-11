import type {
  Api,
  AssistantMessage,
  AssistantMessageEvent,
  AssistantMessageEventStream,
  Context,
  Model,
  Tool,
  TSchema,
} from '@earendil-works/pi-ai';
import type { ChatConfig, ChatMessage, ChatSession, ChatStreamEvent, ChatStreamFinal } from '@mlx-node/lm';
import { describe, expect, it } from 'vite-plus/test';

import { makeMlxStreamSimple, type StreamSimpleHost } from '../src/provider/stream-adapter.js';
import type { DiscoveredModelLike } from '../src/types.js';

const DISCOVERED: DiscoveredModelLike = { name: 'qwen-small', path: '/models/qwen-small', modelType: 'qwen3_5' };

const MODEL: Model<Api> = {
  id: 'qwen-small',
  name: 'qwen-small',
  api: 'mlx',
  provider: 'mlx',
  baseUrl: 'mlx://local',
  reasoning: true,
  input: ['text'],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 262144,
  maxTokens: 81920,
};

const CONTEXT: Context = {
  systemPrompt: 'Be terse.',
  messages: [{ role: 'user', content: 'Hi', timestamp: 1 }],
};

function delta(text: string): ChatStreamEvent {
  return { text, done: false };
}

function finalEvent(overrides: Partial<ChatStreamFinal> = {}): ChatStreamFinal {
  return {
    text: '',
    done: true,
    finishReason: 'stop',
    toolCalls: [],
    thinking: null,
    numTokens: 4,
    promptTokens: 20,
    reasoningTokens: 0,
    rawText: '',
    cachedTokens: 8,
    ...overrides,
  };
}

type Script = (config: ChatConfig | undefined, signal: AbortSignal | undefined) => AsyncGenerator<ChatStreamEvent>;

/**
 * Scripted stand-in for the resident `ChatSession`. Seeds STALE JS state
 * (`turnCount`/`history`/…) so the warm-reuse wipe is observable: the
 * snapshot taken inside `primeHistory` can only read 0/0 if
 * `resetPreservingNativeCacheForWarmReuse` ran first.
 */
class FakeChatSession {
  inFlight = false;
  history: unknown[] = ['stale-entry'];
  lastImagesKey: string | null = 'stale-images-key';
  turnCount = 5;
  unresolvedOkToolCallCount: number | null = 2;

  primedWith: ChatMessage[] | null = null;
  stateAtPrime: { turnCount: number; historyLength: number } | null = null;
  configSeen: ChatConfig | undefined;
  signalSeen: AbortSignal | undefined;

  /** Full-reset observability: `reset()` increments this; warm-reuse does not. */
  resetCalls = 0;
  /** When set, `reset()` throws (post-error reset failure → resident invalidation). */
  resetShouldThrow = false;

  constructor(
    private readonly scripts: Script[],
    readonly log: string[] = [],
  ) {}

  /**
   * Full public reset — distinct from the JS-only warm-reuse wipe. The
   * adapter calls this ONLY on the post-error recovery turn; a counter +
   * log entry make "full reset vs warm reuse" observable.
   */
  // eslint-disable-next-line @typescript-eslint/require-await
  async reset(): Promise<void> {
    this.resetCalls += 1;
    this.log.push('reset');
    if (this.resetShouldThrow) {
      throw new Error('reset boom');
    }
    this.history = [];
    this.lastImagesKey = null;
    this.turnCount = 0;
    this.unresolvedOkToolCallCount = null;
  }

  primeHistory(messages: ChatMessage[]): void {
    this.stateAtPrime = { turnCount: this.turnCount, historyLength: this.history.length };
    this.primedWith = messages;
    this.history = [...messages];
    this.log.push('prime');
  }

  startFromHistoryStream(config?: ChatConfig, signal?: AbortSignal): AsyncGenerator<ChatStreamEvent> {
    this.configSeen = config;
    this.signalSeen = signal;
    this.log.push('stream-created');
    const script = this.scripts.shift();
    if (!script) throw new Error('FakeChatSession: no script left for startFromHistoryStream');
    const inner = script(config, signal);
    const log = this.log;
    return (async function* () {
      log.push('stream-first-pull');
      yield* inner;
      log.push('stream-done');
    })();
  }

  asChatSession(): ChatSession {
    return this as unknown as ChatSession;
  }
}

/** The three post-error dirty-tracking methods as inert stubs for hosts that never exercise them. */
function dirtyStubs(): Pick<StreamSimpleHost, 'markResidentDirty' | 'consumeResidentDirty' | 'invalidateResident'> {
  return {
    markResidentDirty: () => undefined,
    consumeResidentDirty: () => false,
    invalidateResident: () => undefined,
  };
}

/** Test view of {@link makeFakeHost} exposing the post-error dirty state it tracks. */
interface FakeHost extends StreamSimpleHost {
  /** Model ids passed to `invalidateResident`, in order. */
  invalidatedIds: string[];
  /** Whether `modelId` is currently flagged dirty (post-error). */
  isDirty(modelId: string): boolean;
}

/**
 * Fake host mirroring `MlxModelHost`'s serialized promise chain, sharing
 * `log` with the session so the full closure/stream interleaving is
 * assertable. Tracks the post-error `dirty` flag per model id (single-model
 * tests never swap) and records `invalidateResident` calls.
 */
function makeFakeHost(session: FakeChatSession, log: string[] = session.log): FakeHost {
  let chain: Promise<unknown> = Promise.resolve();
  let calls = 0;
  const dirty = new Set<string>();
  const invalidatedIds: string[] = [];
  return {
    modelInfo: (modelId) => (modelId === DISCOVERED.name ? DISCOVERED : undefined),
    runWithResident<T>(modelId: string, fn: (s: ChatSession) => Promise<T>): Promise<T> {
      const n = ++calls;
      const run = async (): Promise<T> => {
        log.push(`fn${n}-start:${modelId}`);
        try {
          return await fn(session.asChatSession());
        } finally {
          log.push(`fn${n}-end:${modelId}`);
        }
      };
      const result = chain.then(run);
      chain = result.then(
        () => undefined,
        () => undefined,
      );
      return result;
    },
    markResidentDirty: (modelId) => {
      dirty.add(modelId);
    },
    consumeResidentDirty: (modelId) => {
      const was = dirty.has(modelId);
      dirty.delete(modelId);
      return was;
    },
    invalidateResident: (modelId) => {
      invalidatedIds.push(modelId);
      dirty.delete(modelId);
    },
    invalidatedIds,
    isDirty: (modelId) => dirty.has(modelId),
  };
}

async function collect(stream: AssistantMessageEventStream): Promise<AssistantMessageEvent[]> {
  const events: AssistantMessageEvent[] = [];
  for await (const event of stream) events.push(event);
  return events;
}

function types(events: AssistantMessageEvent[]): string[] {
  return events.map((e) => e.type);
}

function finalMessage(events: AssistantMessageEvent[]): AssistantMessage {
  const last = events[events.length - 1]!;
  if (last.type === 'done') return last.message;
  if (last.type === 'error') return last.error;
  throw new Error(`stream did not terminate: last event ${last.type}`);
}

describe('makeMlxStreamSimple', () => {
  it('streams a happy text turn in order with stopReason stop', async () => {
    const session = new FakeChatSession([
      // eslint-disable-next-line @typescript-eslint/require-await
      async function* () {
        yield delta('Hel');
        yield delta('lo!');
        yield finalEvent({ text: 'Hello!', numTokens: 2, promptTokens: 10, cachedTokens: 4 });
      },
    ]);
    const streamSimple = makeMlxStreamSimple(makeFakeHost(session));

    const events = await collect(streamSimple(MODEL, CONTEXT, { maxTokens: 64 }));
    expect(types(events)).toEqual(['start', 'text_start', 'text_delta', 'text_delta', 'text_end', 'done']);

    const message = finalMessage(events);
    expect(message.stopReason).toBe('stop');
    expect(message.content).toEqual([{ type: 'text', text: 'Hello!' }]);
    expect(message.model).toBe('qwen-small');
    expect(message.usage.cacheRead).toBe(4);
    expect(message.usage.totalTokens).toBe(12);

    // Warm-reset ran BEFORE prime (stale 5/1 wiped to 0/0), prime before iteration.
    expect(session.stateAtPrime).toEqual({ turnCount: 0, historyLength: 0 });
    expect(session.log).toEqual([
      'fn1-start:qwen-small',
      'prime',
      'stream-created',
      'stream-first-pull',
      'stream-done',
      'fn1-end:qwen-small',
    ]);

    // pi context converted through contextToChatMessages.
    expect(session.primedWith).toEqual([
      { role: 'system', content: 'Be terse.' },
      { role: 'user', content: 'Hi' },
    ]);

    // Chat config assembled from the qwen3_5 launch preset + pi options overlay.
    expect(session.configSeen?.maxNewTokens).toBe(64);
    expect(session.configSeen?.reasoningEffort).toBe('none');
    expect(session.configSeen?.tools).toBeUndefined();
  });

  it('does not prime a prior error/aborted partial assistant turn into the reset session (WB-4)', async () => {
    // Integration guard for WB-4: pi's history can carry a turn that errored (or
    // was aborted) mid-decode with PARTIAL content. contextToChatMessages must
    // drop it, so what reaches primeHistory is the clean history without the
    // invalid partial turn. Mutation guard: reverting WB-4 (keep partial
    // error/aborted turns) primes 'half a broken answer' here → this fails.
    const session = new FakeChatSession([
      // eslint-disable-next-line @typescript-eslint/require-await
      async function* () {
        yield delta('ok');
        yield finalEvent({ text: 'ok' });
      },
    ]);
    const streamSimple = makeMlxStreamSimple(makeFakeHost(session));

    const brokenTurn: AssistantMessage = {
      role: 'assistant',
      content: [{ type: 'text', text: 'half a broken answer' }],
      api: 'mlx',
      provider: 'mlx',
      model: 'qwen-small',
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      stopReason: 'error',
      timestamp: 2,
    };
    const ctxWithBrokenTurn: Context = {
      systemPrompt: 'Be terse.',
      messages: [
        { role: 'user', content: 'Q1', timestamp: 1 },
        brokenTurn,
        { role: 'user', content: 'Q2', timestamp: 3 },
      ],
    };

    await collect(streamSimple(MODEL, ctxWithBrokenTurn));
    expect(session.primedWith).toEqual([
      { role: 'system', content: 'Be terse.' },
      { role: 'user', content: 'Q1' },
      { role: 'user', content: 'Q2' },
    ]);
    // No trace of the dropped partial turn survives into the primed history.
    expect(session.primedWith?.some((m) => m.content === 'half a broken answer')).toBe(false);
  });

  it('emits the toolcall trio and finishes with toolUse on a tool-call turn', async () => {
    const session = new FakeChatSession([
      // eslint-disable-next-line @typescript-eslint/require-await
      async function* () {
        yield delta('\n\n');
        yield delta('<tool_call>{"name":"get_weather","arguments":{"location":"Paris"}}</tool_call>');
        yield finalEvent({
          finishReason: 'tool_calls',
          toolCalls: [
            {
              id: 'call_1',
              name: 'get_weather',
              arguments: { location: 'Paris' },
              status: 'ok',
              rawContent: '{"name":"get_weather","arguments":{"location":"Paris"}}',
            },
          ],
        });
      },
    ]);
    const streamSimple = makeMlxStreamSimple(makeFakeHost(session));

    const weatherTool: Tool = {
      name: 'get_weather',
      description: 'Get current weather for a city',
      parameters: {
        type: 'object',
        properties: { location: { type: 'string', description: 'City name' } },
        required: ['location'],
      } as unknown as TSchema,
    };
    const events = await collect(streamSimple(MODEL, { ...CONTEXT, tools: [weatherTool] }));

    // The tag-buffered markup and leading whitespace never surface as text.
    expect(types(events)).toEqual(['start', 'toolcall_start', 'toolcall_delta', 'toolcall_end', 'done']);
    const message = finalMessage(events);
    expect(message.stopReason).toBe('toolUse');
    expect(message.content).toEqual([
      { type: 'toolCall', id: 'call_1', name: 'get_weather', arguments: { location: 'Paris' } },
    ]);

    // pi tools crossed into ChatConfig.tools via toolsToDefinitions.
    expect(session.configSeen?.tools).toHaveLength(1);
    expect(session.configSeen?.tools?.[0]?.function.name).toBe('get_weather');
  });

  it('synthesizes an aborted terminal when the signal fires and the generator ends without a final', async () => {
    const controller = new AbortController();
    const session = new FakeChatSession([
      // eslint-disable-next-line @typescript-eslint/require-await
      async function* () {
        yield delta('partial answ');
        controller.abort();
        // Aborted native streams end with NO final event.
      },
    ]);
    const streamSimple = makeMlxStreamSimple(makeFakeHost(session));

    const events = await collect(streamSimple(MODEL, CONTEXT, { signal: controller.signal }));
    expect(types(events)).toEqual(['start', 'text_start', 'text_delta', 'text_end', 'error']);
    const last = events[events.length - 1]!;
    expect(last.type === 'error' && last.reason).toBe('aborted');
    const message = finalMessage(events);
    expect(message.stopReason).toBe('aborted');
    expect(message.content).toEqual([{ type: 'text', text: 'partial answ' }]);

    // The signal was handed through to the native stream.
    expect(session.signalSeen).toBe(controller.signal);
    // Warm-reset and prime both happened BEFORE iteration started.
    expect(session.stateAtPrime).toEqual({ turnCount: 0, historyLength: 0 });
    expect(session.log.indexOf('prime')).toBeLessThan(session.log.indexOf('stream-first-pull'));
  });

  it('treats a final-less ending without an abort as an error terminal', async () => {
    const session = new FakeChatSession([
      // eslint-disable-next-line @typescript-eslint/require-await
      async function* () {
        yield delta('half');
      },
    ]);
    const streamSimple = makeMlxStreamSimple(makeFakeHost(session));

    const events = await collect(streamSimple(MODEL, CONTEXT));
    expect(types(events)).toEqual(['start', 'text_start', 'text_delta', 'text_end', 'error']);
    const message = finalMessage(events);
    expect(message.stopReason).toBe('error');
    expect(message.errorMessage).toBe('stream ended without final event');
  });

  it('turns a runWithResident rejection (load failure) into an error terminal without throwing', async () => {
    const host: StreamSimpleHost = {
      modelInfo: () => DISCOVERED,
      ...dirtyStubs(),
      runWithResident: () => Promise.reject(new Error('load failed: metallib exploded')),
    };
    const streamSimple = makeMlxStreamSimple(host);

    let stream: AssistantMessageEventStream;
    expect(() => {
      stream = streamSimple(MODEL, CONTEXT);
    }).not.toThrow();

    const events = await collect(stream!);
    expect(types(events)).toEqual(['start', 'error']);
    const last = events[events.length - 1]!;
    expect(last.type === 'error' && last.reason).toBe('error');
    const message = finalMessage(events);
    expect(message.stopReason).toBe('error');
    expect(message.errorMessage).toBe('load failed: metallib exploded');
    // pi's loop consumes result(): it must resolve to the terminal message.
    expect(await stream!.result()).toBe(message);
  });

  it('reports an unknown model through the stream instead of throwing', async () => {
    const session = new FakeChatSession([]);
    const streamSimple = makeMlxStreamSimple(makeFakeHost(session));

    const unknownModel: Model<Api> = { ...MODEL, id: 'not-discovered' };
    const events = await collect(streamSimple(unknownModel, CONTEXT));
    expect(types(events)).toEqual(['start', 'error']);
    expect(finalMessage(events).errorMessage).toMatch(/no discovery record for model "not-discovered"/);
    // The failure fired before any session work.
    expect(session.log).toEqual(['fn1-start:not-discovered', 'fn1-end:not-discovered']);
  });

  it('terminates promptly with aborted when the signal fires while QUEUED, then skips all session work', async () => {
    // Reviewer repro: a request parked behind stalled prior inference or
    // loading must not stay start-only forever when its signal aborts.
    const session = new FakeChatSession([]);
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    let closureRan = false;
    const host: StreamSimpleHost = {
      modelInfo: () => DISCOVERED,
      ...dirtyStubs(),
      async runWithResident<T>(_modelId: string, fn: (s: ChatSession) => Promise<T>): Promise<T> {
        await gate; // prior work that never yields until released
        closureRan = true;
        return fn(session.asChatSession());
      },
    };
    const controller = new AbortController();
    const streamSimple = makeMlxStreamSimple(host);

    const stream = streamSimple(MODEL, CONTEXT, { signal: controller.signal });
    controller.abort(); // abort while queued — the resident closure has not run

    // The stream must terminate WITHOUT the gate ever being released.
    const events = await collect(stream);
    expect(types(events)).toEqual(['start', 'error']);
    const last = events[events.length - 1]!;
    expect(last.type === 'error' && last.reason).toBe('aborted');
    const message = finalMessage(events);
    expect(message.stopReason).toBe('aborted');
    expect(await stream.result()).toBe(message);

    // Release the stalled work: the closure runs but must skip ALL session
    // work (no warm-reset side effects, no prime, no stream) and emit
    // nothing further.
    release();
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(closureRan).toBe(true);
    expect(session.log).toEqual([]);
    expect(session.primedWith).toBeNull();
    expect(session.stateAtPrime).toBeNull();
    expect(await collect(stream)).toEqual([]); // no events after the terminal
    expect(await stream.result()).toBe(message);
  });

  it('pre-checks the signal: an already-aborted call terminates without engaging the host', async () => {
    let hostCalls = 0;
    const host: StreamSimpleHost = {
      modelInfo: () => DISCOVERED,
      ...dirtyStubs(),
      runWithResident: () => {
        hostCalls += 1;
        return Promise.reject(new Error('must not run'));
      },
    };
    const controller = new AbortController();
    controller.abort();
    const streamSimple = makeMlxStreamSimple(host);

    const events = await collect(streamSimple(MODEL, CONTEXT, { signal: controller.signal }));
    expect(types(events)).toEqual(['start', 'error']);
    const last = events[events.length - 1]!;
    expect(last.type === 'error' && last.reason).toBe('aborted');
    expect(finalMessage(events).stopReason).toBe('aborted');
    expect(hostCalls).toBe(0);
  });

  it('contains a synchronous setup failure (poisoned Model getter) inside the stream', async () => {
    const session = new FakeChatSession([]);
    const streamSimple = makeMlxStreamSimple(makeFakeHost(session));
    const evilModel = Object.defineProperty({ ...MODEL }, 'api', {
      get(): string {
        throw new Error('poisoned model.api');
      },
    }) as Model<Api>;

    let stream!: AssistantMessageEventStream;
    expect(() => {
      stream = streamSimple(evilModel, CONTEXT);
    }).not.toThrow();

    // The TurnEmitter died before it could push 'start'; the failsafe
    // still delivers exactly one terminal so result() settles.
    const events = await collect(stream);
    expect(types(events)).toEqual(['error']);
    const message = finalMessage(events);
    expect(message.stopReason).toBe('error');
    expect(message.errorMessage).toBe('poisoned model.api');
    expect(message.api).toBe('unknown'); // hostile field reads fall back safely
    expect(await stream.result()).toBe(message);
    expect(session.log).toEqual([]); // the host was never engaged
  });

  it('falls back to a TurnEmitter-independent terminal when the error defeats coercion (null-prototype)', async () => {
    // String(err) throws for a null-prototype object, which blows up
    // TurnEmitter.onError itself — the failsafe must still terminalize.
    const evil: unknown = Object.create(null);
    const host: StreamSimpleHost = {
      modelInfo: () => DISCOVERED,
      ...dirtyStubs(),
      runWithResident: () => Promise.reject(evil),
    };
    const stream = makeMlxStreamSimple(host)(MODEL, CONTEXT);

    const events = await collect(stream);
    expect(types(events)).toEqual(['start', 'error']);
    const message = finalMessage(events);
    expect(message.stopReason).toBe('error');
    expect(message.errorMessage).toBe('unserializable error');
    expect(await stream.result()).toBe(message);
  });

  it('terminalizes a revoked-Proxy rejection (instanceof itself throws) with no unhandled rejection', async () => {
    // Reviewer repro: `err instanceof Error` throws on a revoked Proxy
    // (`TypeError: Cannot perform 'getPrototypeOf' on a proxy that has
    // been revoked`). With the check outside the guard, the coercion
    // helper itself threw → detached catch rejected → the stream stayed
    // start-only AND an unhandled rejection escaped.
    const { proxy, revoke } = Proxy.revocable({}, {});
    revoke();
    const host: StreamSimpleHost = {
      modelInfo: () => DISCOVERED,
      ...dirtyStubs(),
      runWithResident: () => Promise.reject(proxy),
    };

    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown): void => {
      unhandled.push(reason);
    };
    process.on('unhandledRejection', onUnhandled);
    try {
      const stream = makeMlxStreamSimple(host)(MODEL, CONTEXT);
      const events = await collect(stream);
      // Exactly one terminal: 'start' then a single 'error' event.
      expect(types(events)).toEqual(['start', 'error']);
      const last = events[events.length - 1]!;
      expect(last.type === 'error' && last.reason).toBe('error');
      const message = finalMessage(events);
      expect(message.stopReason).toBe('error');
      // String(revokedProxy) also throws → constant fallback.
      expect(message.errorMessage).toBe('unserializable error');
      // result() settles on the terminal message.
      expect(await stream.result()).toBe(message);
      // Give any detached rejection a chance to surface before asserting.
      await new Promise((resolve) => setTimeout(resolve, 20));
      expect(unhandled).toEqual([]);
    } finally {
      process.off('unhandledRejection', onUnhandled);
    }
  });

  it('falls back to the failsafe terminal for an Error whose message getter throws', async () => {
    class PoisonError extends Error {
      constructor() {
        super(); // no message argument → the prototype getter stays active
      }
      override get message(): string {
        throw new Error('message getter exploded');
      }
    }
    const host: StreamSimpleHost = {
      modelInfo: () => DISCOVERED,
      ...dirtyStubs(),
      runWithResident: () => Promise.reject(new PoisonError()),
    };
    const stream = makeMlxStreamSimple(host)(MODEL, CONTEXT);

    const events = await collect(stream);
    expect(types(events)).toEqual(['start', 'error']);
    const message = finalMessage(events);
    expect(message.stopReason).toBe('error');
    // Both err.message and String(err) throw → constant fallback.
    expect(message.errorMessage).toBe('unserializable error');
    expect(await stream.result()).toBe(message);
  });

  it('swallows a terminal-push failure instead of cascading into pi', async () => {
    const host: StreamSimpleHost = {
      modelInfo: () => DISCOVERED,
      ...dirtyStubs(),
      runWithResident: () => Promise.reject(new Error('load failed')),
    };
    const streamSimple = makeMlxStreamSimple(host);

    const unhandled: unknown[] = [];
    const onUnhandled = (reason: unknown): void => {
      unhandled.push(reason);
    };
    process.on('unhandledRejection', onUnhandled);
    try {
      const stream = streamSimple(MODEL, CONTEXT);
      // Sabotage the terminal surface AFTER setup pushed 'start' but
      // BEFORE the detached rejection path fires (microtask ordering).
      let pushAttempts = 0;
      stream.push = () => {
        pushAttempts += 1;
        throw new Error('stream.push exploded');
      };
      stream.end = () => {
        throw new Error('stream.end exploded');
      };
      await new Promise((resolve) => setTimeout(resolve, 20));
      expect(pushAttempts).toBeGreaterThan(0); // the terminal WAS attempted
      expect(unhandled).toEqual([]); // and its failure never escaped
    } finally {
      process.off('unhandledRejection', onUnhandled);
    }
  });

  it('runs two concurrent calls strictly sequentially, streaming INSIDE each closure', async () => {
    const session = new FakeChatSession([
      async function* () {
        yield delta('first');
        await new Promise((resolve) => setTimeout(resolve, 0));
        yield finalEvent({ text: 'first' });
      },
      // eslint-disable-next-line @typescript-eslint/require-await
      async function* () {
        yield delta('second');
        yield finalEvent({ text: 'second' });
      },
    ]);
    const streamSimple = makeMlxStreamSimple(makeFakeHost(session));

    // Fire both calls back-to-back WITHOUT awaiting the first.
    const streamA = streamSimple(MODEL, CONTEXT);
    const streamB = streamSimple(MODEL, CONTEXT);
    const [eventsA, eventsB] = await Promise.all([collect(streamA), collect(streamB)]);

    expect(finalMessage(eventsA).content).toEqual([{ type: 'text', text: 'first' }]);
    expect(finalMessage(eventsB).content).toEqual([{ type: 'text', text: 'second' }]);

    // Call 2's closure must not start until call 1's closure — INCLUDING its
    // full stream iteration — has finished. Streaming outside runWithResident
    // (the rejected ensureResident pattern) would interleave this log.
    expect(session.log).toEqual([
      'fn1-start:qwen-small',
      'prime',
      'stream-created',
      'stream-first-pull',
      'stream-done',
      'fn1-end:qwen-small',
      'fn2-start:qwen-small',
      'prime',
      'stream-created',
      'stream-first-pull',
      'stream-done',
      'fn2-end:qwen-small',
    ]);
  });

  describe('post-error KV recovery (full reset instead of warm reuse)', () => {
    it('full-resets on the turn AFTER a native ERROR terminal', async () => {
      const session = new FakeChatSession([
        // Turn 1: native error mid-decode (throws out of the stream loop).
        // eslint-disable-next-line @typescript-eslint/require-await
        async function* () {
          yield delta('par');
          throw new Error('native decode fault');
        },
        // Turn 2: normal completion.
        // eslint-disable-next-line @typescript-eslint/require-await
        async function* () {
          yield delta('ok');
          yield finalEvent({ text: 'ok' });
        },
      ]);
      const host = makeFakeHost(session);
      const streamSimple = makeMlxStreamSimple(host);

      const events1 = await collect(streamSimple(MODEL, CONTEXT));
      expect(finalMessage(events1).stopReason).toBe('error');
      // The error terminal flagged the resident dirty; turn 1 itself still
      // used the warm-reuse wipe (no full reset).
      expect(host.isDirty(MODEL.id)).toBe(true);
      expect(session.resetCalls).toBe(0);

      const events2 = await collect(streamSimple(MODEL, CONTEXT));
      expect(finalMessage(events2).stopReason).toBe('stop');
      // Turn 2 did a FULL reset (cold prefill) instead of warm reuse, and the
      // dirty flag was cleared.
      expect(session.resetCalls).toBe(1);
      expect(session.log).toContain('reset');
      expect(host.isDirty(MODEL.id)).toBe(false);
    });

    it('keeps warm reuse on the turn AFTER an ABORTED terminal (no over-fix)', async () => {
      const controller = new AbortController();
      const session = new FakeChatSession([
        // Turn 1: user cancel — the native stream ends with NO final event.
        // eslint-disable-next-line @typescript-eslint/require-await
        async function* () {
          yield delta('partial');
          controller.abort();
        },
        // Turn 2: normal completion.
        // eslint-disable-next-line @typescript-eslint/require-await
        async function* () {
          yield delta('next');
          yield finalEvent({ text: 'next' });
        },
      ]);
      const host = makeFakeHost(session);
      const streamSimple = makeMlxStreamSimple(host);

      const events1 = await collect(streamSimple(MODEL, CONTEXT, { signal: controller.signal }));
      expect(finalMessage(events1).stopReason).toBe('aborted');
      // Abort must NOT mark the resident dirty (a clean cancel realigns the
      // cache) — preserving warm reuse is the whole point.
      expect(host.isDirty(MODEL.id)).toBe(false);

      const events2 = await collect(streamSimple(MODEL, CONTEXT));
      expect(finalMessage(events2).stopReason).toBe('stop');
      expect(session.resetCalls).toBe(0);
      expect(session.log).not.toContain('reset');
    });

    it('keeps warm reuse on the turn AFTER a normal completion', async () => {
      const session = new FakeChatSession([
        // eslint-disable-next-line @typescript-eslint/require-await
        async function* () {
          yield delta('a');
          yield finalEvent({ text: 'a' });
        },
        // eslint-disable-next-line @typescript-eslint/require-await
        async function* () {
          yield delta('b');
          yield finalEvent({ text: 'b' });
        },
      ]);
      const host = makeFakeHost(session);
      const streamSimple = makeMlxStreamSimple(host);

      await collect(streamSimple(MODEL, CONTEXT));
      await collect(streamSimple(MODEL, CONTEXT));
      // Never a full reset on the happy path.
      expect(session.resetCalls).toBe(0);
      expect(session.log).not.toContain('reset');
    });

    it('invalidates the resident when the post-error full reset itself throws', async () => {
      const session = new FakeChatSession([
        // eslint-disable-next-line @typescript-eslint/require-await
        async function* () {
          yield delta('x');
          throw new Error('native fault');
        },
        // eslint-disable-next-line @typescript-eslint/require-await
        async function* () {
          yield delta('y');
          yield finalEvent({ text: 'y' });
        },
      ]);
      session.resetShouldThrow = true;
      const host = makeFakeHost(session);
      const streamSimple = makeMlxStreamSimple(host);

      const events1 = await collect(streamSimple(MODEL, CONTEXT));
      expect(finalMessage(events1).stopReason).toBe('error');
      expect(host.isDirty(MODEL.id)).toBe(true);

      const events2 = await collect(streamSimple(MODEL, CONTEXT));
      // Turn 2 attempted the full reset; it threw, so the turn errors AND the
      // resident was invalidated (next call reloads from scratch).
      expect(session.resetCalls).toBe(1);
      expect(finalMessage(events2).stopReason).toBe('error');
      expect(host.invalidatedIds).toContain(MODEL.id);
    });

    it('marks the resident dirty synchronously before the callback rejects (queued turn full-resets)', async () => {
      // Reviewer race: the dirty mark must land on the SAME synchronous stack as
      // the callback's rejection — before runSerialized releases the chain — or
      // a queued turn's consumeResidentDirty() sees false and warm-reuses onto
      // the KV advanced by the failed decode. This host captures the dirty state
      // at the exact instant fn1 rejects; the detached `.catch` (terminalize)
      // has not run yet, so ONLY a synchronous in-callback mark shows up here.
      const session = new FakeChatSession([
        // Turn 1: native decode fault throws out of the stream loop.
        // eslint-disable-next-line @typescript-eslint/require-await
        async function* () {
          yield delta('par');
          throw new Error('native decode fault');
        },
        // Turn 2: normal completion — the QUEUED turn.
        // eslint-disable-next-line @typescript-eslint/require-await
        async function* () {
          yield delta('ok');
          yield finalEvent({ text: 'ok' });
        },
      ]);
      const dirty = new Set<string>();
      let dirtyAtReject: boolean | undefined;
      let chain: Promise<unknown> = Promise.resolve();
      const host: StreamSimpleHost = {
        modelInfo: () => DISCOVERED,
        markResidentDirty: (id) => {
          dirty.add(id);
        },
        consumeResidentDirty: (id) => {
          const was = dirty.has(id);
          dirty.delete(id);
          return was;
        },
        invalidateResident: () => undefined,
        runWithResident<T>(id: string, fn: (s: ChatSession) => Promise<T>): Promise<T> {
          const run = async (): Promise<T> => {
            try {
              return await fn(session.asChatSession());
            } catch (err) {
              // At fn's rejection: with the fix, dirty is already set
              // (synchronous, in the callback). Reverting leaves this false.
              if (dirtyAtReject === undefined) dirtyAtReject = dirty.has(id);
              throw err;
            }
          };
          const result = chain.then(run);
          chain = result.then(
            () => undefined,
            () => undefined,
          );
          return result;
        },
      };
      const streamSimple = makeMlxStreamSimple(host);

      const events1 = await collect(streamSimple(MODEL, CONTEXT));
      expect(finalMessage(events1).stopReason).toBe('error');
      // The mark landed on the same synchronous stack as the callback rejection.
      expect(dirtyAtReject).toBe(true);

      const events2 = await collect(streamSimple(MODEL, CONTEXT));
      expect(finalMessage(events2).stopReason).toBe('stop');
      // The queued turn observed dirty=true → FULL reset (cold prefill).
      expect(session.resetCalls).toBe(1);
      expect(session.log).toContain('reset');
    });

    it('marks the resident dirty on an in-band finishReason=error terminal (next turn full-resets)', async () => {
      const session = new FakeChatSession([
        // Turn 1: an in-band error terminal — a `done` event with finishReason
        // 'error', NO throw and NO abort (chat-session yields this without
        // committing a final).
        // eslint-disable-next-line @typescript-eslint/require-await
        async function* () {
          yield delta('par');
          yield finalEvent({ finishReason: 'error' });
        },
        // Turn 2: normal completion.
        // eslint-disable-next-line @typescript-eslint/require-await
        async function* () {
          yield delta('ok');
          yield finalEvent({ text: 'ok' });
        },
      ]);
      const host = makeFakeHost(session);
      const streamSimple = makeMlxStreamSimple(host);

      const events1 = await collect(streamSimple(MODEL, CONTEXT));
      // Routed to onError (stopReason 'error'), NOT treated as a success final.
      expect(finalMessage(events1).stopReason).toBe('error');
      // The in-band error flagged the resident dirty (no throw, no abort).
      expect(host.isDirty(MODEL.id)).toBe(true);
      expect(session.resetCalls).toBe(0);

      const events2 = await collect(streamSimple(MODEL, CONTEXT));
      expect(finalMessage(events2).stopReason).toBe('stop');
      // Turn 2 saw dirty=true → FULL reset (cold prefill) instead of warm reuse.
      expect(session.resetCalls).toBe(1);
      expect(session.log).toContain('reset');
    });
  });
});

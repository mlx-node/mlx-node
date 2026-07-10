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

  constructor(
    private readonly scripts: Script[],
    readonly log: string[] = [],
  ) {}

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

/**
 * Fake host mirroring `MlxModelHost`'s serialized promise chain, sharing
 * `log` with the session so the full closure/stream interleaving is
 * assertable.
 */
function makeFakeHost(session: FakeChatSession, log: string[] = session.log): StreamSimpleHost {
  let chain: Promise<unknown> = Promise.resolve();
  let calls = 0;
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
});

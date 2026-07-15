import type { Api, Context, Model, SimpleStreamOptions } from '@earendil-works/pi-ai';
import type {
  ExtensionAPI,
  ExtensionContext,
  InlineExtension,
  ProviderConfig,
  SessionStartEvent,
} from '@earendil-works/pi-coding-agent';
import type { ChatConfig, ChatSession, ChatStreamEvent } from '@mlx-node/lm';
import { describe, expect, it, vi } from 'vite-plus/test';

import { createMlxProviderExtension } from '../src/provider/index.js';
import type { MlxModelHost } from '../src/provider/model-host.js';
import type { MlxModelInfo } from '../src/provider/models.js';

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
      handlers.get('session_start')!(
        { type: 'session_start', reason },
        {
          sessionManager: { getSessionId: () => rootSessionId },
        } as unknown as ExtensionContext,
      );
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
});

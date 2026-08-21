/**
 * `ChatSession` MTP auto-default policy. Hermetic — every case drives
 * `mergeConfig` through `send()` and reads the `ChatConfig` handed to native.
 *
 * Separate from `chat-session.test.ts` because the class-name fallback needs
 * mocks with a real prototype chain, which the shared `makeMockModel` object
 * literal cannot provide.
 */
import { NemotronHModel } from '@mlx-node/core';
import type { ChatConfig, ChatMessage, ChatResult } from '@mlx-node/core';
import { ChatSession, type SessionCapableModel } from '@mlx-node/lm';
import type { ChatStreamEvent, ChatStreamFinal } from '@mlx-node/lm';
import { describe, expect, it, vi } from 'vite-plus/test';

function makeChatResult(text: string): ChatResult {
  return {
    text,
    rawText: text,
    toolCalls: [],
    thinking: null,
    thinkingEnabled: true,
    numTokens: 1,
    promptTokens: 1,
    reasoningTokens: 0,
    finishReason: 'stop',
    performance: undefined,
  } as unknown as ChatResult;
}

function finalChunk(text: string): ChatStreamFinal {
  return {
    text,
    done: true,
    finishReason: 'stop',
    toolCalls: [],
    thinking: null,
    thinkingEnabled: true,
    numTokens: 2,
    promptTokens: 1,
    reasoningTokens: 0,
    rawText: text,
    cachedTokens: 0,
  } satisfies ChatStreamFinal;
}

/** The session surface every case needs, minus the MTP getters. */
function sessionStubs() {
  const chatSessionStart = vi.fn(
    async (_messages: ChatMessage[], _config?: ChatConfig | null): Promise<ChatResult> => makeChatResult('reply'),
  );
  const chatSessionContinue = vi.fn(
    async (_messages: ChatMessage[], _config?: ChatConfig | null): Promise<ChatResult> => makeChatResult('reply'),
  );
  const chatSessionContinueTool = vi.fn(
    async (_messages: ChatMessage[], _config?: ChatConfig | null): Promise<ChatResult> => makeChatResult('reply'),
  );
  const chatStreamSessionStart = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
    yield finalChunk('reply');
  });
  const chatStreamSessionContinue = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
    yield finalChunk('reply');
  });
  const chatStreamSessionContinueTool = vi.fn(async function* (): AsyncGenerator<ChatStreamEvent> {
    yield finalChunk('reply');
  });
  return {
    chatSessionStart,
    chatSessionContinue,
    chatSessionContinueTool,
    chatStreamSessionStart,
    chatStreamSessionContinue,
    chatStreamSessionContinueTool,
    resetCaches: vi.fn(() => Promise.resolve(undefined)),
    releaseCacheOwner: vi.fn((_ownerId: string) => Promise.resolve(undefined)),
  };
}

type MtpGetters = Pick<SessionCapableModel, 'hasMtpWeights' | 'mtpAutoEnabled'>;

/**
 * Build a mock whose runtime class name is `className`, so the prototype chain
 * the fallback walks looks like the real wrapper's — `@mlx-node/lm` hands
 * `ChatSession` a `makeStreamingModel(...)` subclass named for the family.
 */
function makeNamedModel(className: string, mtp: MtpGetters) {
  const stubs = sessionStubs();
  const Ctor = { [className]: class {} }[className]!;
  const model = Object.assign(Object.create(Ctor.prototype) as object, stubs, mtp) as SessionCapableModel;
  return { model, chatSessionStart: stubs.chatSessionStart };
}

/** The `ChatConfig` the session passed to native on its first turn. */
async function enableMtpForFirstTurn(className: string, mtp: MtpGetters): Promise<boolean | undefined> {
  const { model, chatSessionStart } = makeNamedModel(className, mtp);
  const session = new ChatSession(model);
  await session.send('Hello');
  const [, config] = chatSessionStart.mock.calls[0]!;
  return config?.enableMtp;
}

describe('ChatSession MTP auto-default', () => {
  it('does not auto-default enableMtp when the model reports mtpAutoEnabled()==false', async () => {
    const enableMtp = await enableMtpForFirstTurn('SomeUnprofitableModel', {
      hasMtpWeights: () => true,
      mtpAutoEnabled: () => false,
    });
    expect(enableMtp).toBeUndefined();
  });

  it('auto-defaults enableMtp when the model reports mtpAutoEnabled()==true', async () => {
    const enableMtp = await enableMtpForFirstTurn('SomeProfitableModel', {
      hasMtpWeights: () => true,
      mtpAutoEnabled: () => true,
    });
    expect(enableMtp).toBe(true);
  });

  it('never auto-defaults when the model has no MTP head, whatever mtpAutoEnabled() says', async () => {
    // A preference, not a capability: it can never turn speculation on for a
    // checkpoint with no head to draft with.
    const enableMtp = await enableMtpForFirstTurn('SomeHeadlessModel', {
      hasMtpWeights: () => false,
      mtpAutoEnabled: () => true,
    });
    expect(enableMtp).toBeUndefined();
  });

  it('lets the native signal override the suppressed-class fallback', async () => {
    // Pins the priority order: checking the class list first would break this.
    const enableMtp = await enableMtpForFirstTurn('NemotronHModel', {
      hasMtpWeights: () => true,
      mtpAutoEnabled: () => true,
    });
    expect(enableMtp).toBe(true);
  });

  it('does not auto-default enableMtp on NemotronH without the native getter', async () => {
    // NemotronH always ships a complete MTP head, so `hasMtpWeights()` is
    // unconditionally true; auto-enabling would force every session into the
    // exclusive scheduler lane and lose continuous batching.
    const enableMtp = await enableMtpForFirstTurn('NemotronHModel', { hasMtpWeights: () => true });
    expect(enableMtp).toBeUndefined();
  });

  it('still honours an explicit enableMtp=true on NemotronH', async () => {
    // The opt-out is a DEFAULT, not a ban.
    const { model, chatSessionStart } = makeNamedModel('NemotronHModel', { hasMtpWeights: () => true });
    const session = new ChatSession(model);

    await session.send('Hello', { config: { enableMtp: true } });

    const [, config] = chatSessionStart.mock.calls[0]!;
    expect(config?.enableMtp).toBe(true);
  });

  it('matches a subclass of a suppressed wrapper, not just the exact class', async () => {
    const stubs = sessionStubs();
    class NemotronHModel {}
    class TunedNemotronHModel extends NemotronHModel {}
    const model = Object.assign(Object.create(TunedNemotronHModel.prototype) as object, stubs, {
      hasMtpWeights: () => true,
    }) as SessionCapableModel;
    const session = new ChatSession(model);

    await session.send('Hello');

    const [, config] = stubs.chatSessionStart.mock.calls[0]!;
    expect(config?.enableMtp).toBeUndefined();
  });

  it.each(['Qwen35Model', 'Qwen35MoeModel', 'Gemma4Model', 'MuseGlimmerModel'])(
    'still auto-defaults enableMtp on %s (no getter, not suppressed)',
    async (className) => {
      // The families the auto-default exists for. None expose
      // `mtpAutoEnabled()`, so they must take the historical arm.
      const enableMtp = await enableMtpForFirstTurn(className, { hasMtpWeights: () => true });
      expect(enableMtp).toBe(true);
    },
  );

  it('still auto-defaults enableMtp for a plain object model with no class identity', async () => {
    // Third-party implementations are frequently object literals, so the walk
    // must terminate cleanly on `Object.prototype` and take the historical arm.
    const stubs = sessionStubs();
    const model: SessionCapableModel = { ...stubs, hasMtpWeights: () => true };
    const session = new ChatSession(model);

    await session.send('Hello');

    const [, config] = stubs.chatSessionStart.mock.calls[0]!;
    expect(config?.enableMtp).toBe(true);
  });

  it('still auto-defaults enableMtp for a null-prototype model', async () => {
    // The walk must terminate on a `null` prototype without throwing.
    const stubs = sessionStubs();
    const model = Object.assign(Object.create(null) as object, stubs, {
      hasMtpWeights: () => true,
    }) as SessionCapableModel;
    const session = new ChatSession(model);

    await session.send('Hello');

    const [, config] = stubs.chatSessionStart.mock.calls[0]!;
    expect(config?.enableMtp).toBe(true);
  });
});

/**
 * CROSS-MODULE SEAM: the TS fallback is a STRING, the native class is Rust, and
 * every case above fabricates its own class so none of them can see a
 * divergence. Renaming the napi class or dropping the getter breaks the policy
 * SILENTLY — `mtpAutoEnabled?()` is OPTIONAL so a missing getter type-checks,
 * and a renamed class still satisfies every interface while the string fallback
 * stops matching and MTP quietly turns back on.
 */
describe('MTP auto-default: native surface agrees with the TS fallback', () => {
  it('exports a class literally named NemotronHModel', () => {
    // The exact string `MTP_AUTO_DEFAULT_SUPPRESSED_MODELS` holds.
    expect(NemotronHModel.name).toBe('NemotronHModel');
  });

  it('declares mtpAutoEnabled on the native prototype', () => {
    // The getter the TS gate PREFERS over the class-name fallback; without it
    // the fallback silently becomes the only policy and cannot see
    // `MLX_NEMOTRON_MTP_DEFAULT=1`.
    expect(typeof NemotronHModel.prototype.mtpAutoEnabled).toBe('function');
  });

  it('still exposes hasMtpWeights as a separate capability query', () => {
    // Collapsing the two into one would make the head unreachable rather than
    // merely off-by-default.
    expect(typeof NemotronHModel.prototype.hasMtpWeights).toBe('function');
    expect(NemotronHModel.prototype.mtpAutoEnabled).not.toBe(NemotronHModel.prototype.hasMtpWeights);
  });
});

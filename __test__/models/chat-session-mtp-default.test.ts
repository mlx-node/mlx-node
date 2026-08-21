/**
 * `ChatSession` MTP auto-default policy.
 *
 * Hermetic — no weights are loaded. Every case drives `mergeConfig`
 * through `send()` and reads the `ChatConfig` the session handed to the
 * native entry point.
 *
 * The rule under test: shipping an MTP head and PROFITING from it are two
 * different facts. `hasMtpWeights()` answers the first and stays a pure
 * capability query; `mtpAutoEnabled()` answers the second and gates only
 * the DEFAULT. When a model answers neither, a class-name fallback keeps
 * the policy correct on bindings that predate `mtpAutoEnabled()`.
 *
 * Two things must hold at once and are asserted here side by side:
 *   1. NemotronH — whose shipped checkpoint ALWAYS carries a complete MTP
 *      head, and whose MTP turns are forced out of continuous batching by
 *      the scheduler barrier — must NOT be auto-enabled.
 *   2. Every other MTP family (Qwen3.5 dense/MoE native heads, Gemma4
 *      external DSpark/assistant drafts) must be byte-identical to the
 *      historical behavior: auto-enabled whenever `hasMtpWeights()` is
 *      `true`.
 *
 * Lives in its own file rather than in `chat-session.test.ts` because it
 * needs mocks with a real prototype chain (the fallback matches class
 * names), which the shared `makeMockModel` object literal cannot provide.
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
 * Build a mock whose runtime class name is `className`, so the prototype
 * chain the fallback walks looks like the real wrapper's.
 *
 * `@mlx-node/lm` hands `ChatSession` a `makeStreamingModel(...)` subclass
 * whose own name is the family name (`NemotronHModel`), which is why the
 * production check reads class names and not a bespoke marker property.
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
  // -------------------------------------------------------------------
  // The native signal, when the binding provides it.
  // -------------------------------------------------------------------

  it('does not auto-default enableMtp when the model reports mtpAutoEnabled()==false', async () => {
    // Mutation caught: dropping the `mtpAutoDefaultAllowed()` clause from
    // `mergeConfig` — `enableMtp` would come back `true`.
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
    // `mtpAutoEnabled()` is a preference, not a capability: it can never
    // turn speculation on for a checkpoint that has no head to draft with.
    const enableMtp = await enableMtpForFirstTurn('SomeHeadlessModel', {
      hasMtpWeights: () => false,
      mtpAutoEnabled: () => true,
    });
    expect(enableMtp).toBeUndefined();
  });

  it('lets the native signal override the suppressed-class fallback', async () => {
    // The static list is only a fallback. A NemotronH binding that DOES
    // answer the getter — e.g. built with the fleet-wide opt-in — wins.
    // Mutation caught: checking the class list before the getter.
    const enableMtp = await enableMtpForFirstTurn('NemotronHModel', {
      hasMtpWeights: () => true,
      mtpAutoEnabled: () => true,
    });
    expect(enableMtp).toBe(true);
  });

  // -------------------------------------------------------------------
  // The fallback, for bindings that predate `mtpAutoEnabled()`.
  // -------------------------------------------------------------------

  it('does not auto-default enableMtp on NemotronH without the native getter', async () => {
    // The shipped `nemotron-3.5-lightning-30b-a3b` checkpoint ALWAYS
    // carries a complete MTP head, so `hasMtpWeights()` is unconditionally
    // true and the historical rule fired on every session. That forces the
    // turn into the exclusive scheduler lane, losing continuous batching
    // outright. (The throughput half of the old rationale is retracted: MTP
    // is now a wash at 0.994-1.04x. The barrier is the whole reason.) A
    // caller who configures nothing must land on the batching path.
    //
    // Mutation caught: emptying MTP_AUTO_DEFAULT_SUPPRESSED_MODELS, or
    // narrowing the prototype-chain walk so the wrapper subclass misses.
    const enableMtp = await enableMtpForFirstTurn('NemotronHModel', { hasMtpWeights: () => true });
    expect(enableMtp).toBeUndefined();
  });

  it('still honours an explicit enableMtp=true on NemotronH', async () => {
    // The opt-out is a DEFAULT, not a ban — operators must be able to A/B
    // MTP against AR on the same binary with no env var and no rebuild.
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

  // -------------------------------------------------------------------
  // Non-regression: every other family is untouched.
  // -------------------------------------------------------------------

  it.each(['Qwen35Model', 'Qwen35MoeModel', 'Gemma4Model', 'MuseGlimmerModel'])(
    'still auto-defaults enableMtp on %s (no getter, not suppressed)',
    async (className) => {
      // These are the families the auto-default exists for: Qwen3.5
      // dense/MoE in-checkpoint heads and Gemma4 external DSpark/assistant
      // drafts (which report through `hasMtpWeights()` too, and carry
      // their own break-even guard on the native side). None of them
      // expose `mtpAutoEnabled()`, so they must take the historical arm.
      //
      // Mutation caught: inverting the fallback to a suppress-by-default
      // allowlist, or making `mtpAutoDefaultAllowed()` return false when
      // the getter is absent.
      const enableMtp = await enableMtpForFirstTurn(className, { hasMtpWeights: () => true });
      expect(enableMtp).toBe(true);
    },
  );

  it('still auto-defaults enableMtp for a plain object model with no class identity', async () => {
    // Third-party `SessionCapableModel` implementations are frequently
    // object literals (`constructor.name === 'Object'`). The prototype
    // walk must terminate cleanly on `Object.prototype` / `null` and take
    // the historical arm.
    const stubs = sessionStubs();
    const model: SessionCapableModel = { ...stubs, hasMtpWeights: () => true };
    const session = new ChatSession(model);

    await session.send('Hello');

    const [, config] = stubs.chatSessionStart.mock.calls[0]!;
    expect(config?.enableMtp).toBe(true);
  });

  it('still auto-defaults enableMtp for a null-prototype model', async () => {
    // The walk must terminate on a `null` prototype without throwing.
    // Mutation caught: a `do { ... } while` walk, or reading
    // `proto.constructor.name` without the optional chain.
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
 * CROSS-MODULE SEAM: the TS fallback is a STRING, the native class is Rust.
 *
 * Nothing above can catch a divergence between them — every case there
 * fabricates its own class. The fallback in `chat-session.ts` matches
 * `'NemotronHModel'` against the runtime constructor name, and the native
 * getter it defers to is `#[napi] pub fn mtp_auto_enabled` on the Rust
 * `NemotronHModel`. Renaming the napi class, or dropping the getter from the
 * addon, breaks the policy SILENTLY in both directions:
 *
 *   * `mtpAutoEnabled?()` is declared OPTIONAL on `SessionCapableModel`, so a
 *     missing getter is a type-check pass, not a failure;
 *   * a renamed class still satisfies every interface, so the string fallback
 *     just stops matching and MTP quietly turns back on.
 *
 * These assertions read the real addon's exported surface. No model is
 * loaded — only the constructor and its prototype are inspected.
 */
describe('MTP auto-default: native surface agrees with the TS fallback', () => {
  it('exports a class literally named NemotronHModel', () => {
    // The exact string `MTP_AUTO_DEFAULT_SUPPRESSED_MODELS` holds.
    expect(NemotronHModel.name).toBe('NemotronHModel');
  });

  it('declares mtpAutoEnabled on the native prototype', () => {
    // The getter the TS gate PREFERS over the class-name fallback. If the
    // addon is rebuilt without it, the fallback silently becomes the only
    // policy — and that fallback cannot see `MLX_NEMOTRON_MTP_DEFAULT=1`.
    expect(typeof NemotronHModel.prototype.mtpAutoEnabled).toBe('function');
  });

  it('still exposes hasMtpWeights as a separate capability query', () => {
    // The two must stay DISTINCT methods: `hasMtpWeights()` is "a head
    // shipped", `mtpAutoEnabled()` is "turning it on by default pays". A
    // refactor that collapsed one into the other would make the head
    // unreachable rather than merely off-by-default.
    expect(typeof NemotronHModel.prototype.hasMtpWeights).toBe('function');
    expect(NemotronHModel.prototype.mtpAutoEnabled).not.toBe(NemotronHModel.prototype.hasMtpWeights);
  });
});

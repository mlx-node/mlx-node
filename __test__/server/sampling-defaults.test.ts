/**
 * Plumbing coverage for per-model sampling defaults:
 *
 *   register(name, model, { samplingDefaults })
 *      → SessionRegistry.samplingDefaults
 *      → new ChatSession(model, { defaultConfig: samplingDefaults })
 *      → chatSessionStart(messages, mergedConfig) receives the defaults
 *
 * `ChatSession.defaultConfig` is a private field, so rather than
 * reflecting into the instance we observe what the first
 * `chatSessionStart` dispatch receives: it is the result of
 * `ChatSession.mergeConfig({})`, which equals
 * `{ ...defaultConfig, reuseCache: true }`. That lets us
 * byte-for-byte assert every key we care about from the spy side.
 */

import type { ChatConfig, ChatMessage, ChatResult, ToolCallResult } from '@mlx-node/core';
import type { SessionCapableModel } from '@mlx-node/lm';
import { ModelRegistry, QWEN_SAMPLING_DEFAULTS } from '@mlx-node/server';
import { describe, expect, it, vi } from 'vite-plus/test';

// ---------------------------------------------------------------------------
// Mock model
// ---------------------------------------------------------------------------

interface CapturingModel extends SessionCapableModel {
  lastStartConfig: ChatConfig | null | undefined;
  startCallCount: number;
}

/**
 * Build a session-capable mock whose `chatSessionStart` captures the
 * `config` argument so tests can assert on the merged ChatConfig the
 * session passed down.
 */
function createCapturingModel(): CapturingModel {
  const emptyResult: ChatResult = {
    text: '',
    toolCalls: [] as ToolCallResult[],
    thinking: null,
    numTokens: 0,
    promptTokens: 0,
    reasoningTokens: 0,
    finishReason: 'stop',
    rawText: '',
  } as unknown as ChatResult;

  // eslint-disable-next-line @typescript-eslint/require-await
  async function* emptyStream(): AsyncGenerator<Record<string, unknown>> {
    yield { done: true, text: '', finishReason: 'stop', toolCalls: [], numTokens: 0, promptTokens: 0 };
  }

  const model = {
    lastStartConfig: undefined as ChatConfig | null | undefined,
    startCallCount: 0,
    chatSessionStart: vi.fn(async (_messages: ChatMessage[], config?: ChatConfig | null) => {
      model.lastStartConfig = config;
      model.startCallCount += 1;
      return emptyResult;
    }),
    chatSessionContinue: vi.fn().mockResolvedValue(emptyResult),
    chatSessionContinueTool: vi.fn().mockResolvedValue(emptyResult),
    chatStreamSessionStart: vi.fn(() => emptyStream()),
    chatStreamSessionContinue: vi.fn(() => emptyStream()),
    chatStreamSessionContinueTool: vi.fn(() => emptyStream()),
    resetCaches: vi.fn(),
  };
  return model as unknown as CapturingModel;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('QWEN_SAMPLING_DEFAULTS', () => {
  it('exports Unsloth coding-mode values verbatim', () => {
    // Spec-check against the values pulled from Unsloth's Qwen3.6
    // recommended-settings guide — if any of these drift the test
    // must fail loudly (callers opt in by name, not by shape).
    expect(QWEN_SAMPLING_DEFAULTS.thinkingCoding.temperature).toBe(0.6);
    expect(QWEN_SAMPLING_DEFAULTS.thinkingCoding.topP).toBe(0.95);
    expect(QWEN_SAMPLING_DEFAULTS.thinkingCoding.topK).toBe(20);
    expect(QWEN_SAMPLING_DEFAULTS.thinkingCoding.minP).toBe(0);
    expect(QWEN_SAMPLING_DEFAULTS.thinkingCoding.presencePenalty).toBe(0);
    expect(QWEN_SAMPLING_DEFAULTS.thinkingCoding.repetitionPenalty).toBe(1);
  });

  it('covers every documented mode', () => {
    expect(Object.keys(QWEN_SAMPLING_DEFAULTS).sort()).toEqual([
      'instructGeneral',
      'instructReasoning',
      'thinkingCoding',
      'thinkingGeneral',
    ]);
  });
});

describe('ModelRegistry.register samplingDefaults plumbing', () => {
  it('leaves ChatSession defaultConfig undefined when samplingDefaults is omitted (existing behaviour)', async () => {
    const registry = new ModelRegistry();
    const model = createCapturingModel();
    registry.register('plain', model);

    const sessionReg = registry.getSessionRegistry('plain');
    expect(sessionReg).toBeDefined();
    expect(sessionReg!.defaultSamplingConfig).toBeUndefined();

    // Drive the session path. With no defaults, `mergeConfig({})`
    // yields `{ reuseCache: true }` — no temperature / topK / etc.
    const { session } = sessionReg!.getOrCreate(null, null);
    await session.send('hi');
    expect(model.startCallCount).toBe(1);
    expect(model.lastStartConfig).toEqual({ reuseCache: true });
  });

  it('forwards samplingDefaults into the ChatSession so send() dispatches them to chatSessionStart', async () => {
    const registry = new ModelRegistry();
    const model = createCapturingModel();

    const defaults = QWEN_SAMPLING_DEFAULTS.thinkingCoding;
    registry.register('qwen', model, { samplingDefaults: defaults });

    const sessionReg = registry.getSessionRegistry('qwen');
    expect(sessionReg).toBeDefined();
    expect(sessionReg!.defaultSamplingConfig).toEqual(defaults);

    const { session } = sessionReg!.getOrCreate(null, null);
    await session.send('hi');

    // ChatSession.mergeConfig({}) yields `{ ...defaults, reuseCache: true }`.
    expect(model.lastStartConfig).toEqual({ ...defaults, reuseCache: true });
  });

  it('honours per-call ChatConfig overlay on top of defaults (client wins on shared keys)', async () => {
    const registry = new ModelRegistry();
    const model = createCapturingModel();
    registry.register('qwen', model, { samplingDefaults: QWEN_SAMPLING_DEFAULTS.thinkingCoding });

    const sessionReg = registry.getSessionRegistry('qwen')!;
    const { session } = sessionReg.getOrCreate(null, null);

    // Simulate the request path: client sends temperature + topP,
    // leaves topK/minP/penalties to the server defaults.
    await session.send('hi', { config: { temperature: 0.9, topP: 0.8 } });

    expect(model.lastStartConfig).toEqual({
      ...QWEN_SAMPLING_DEFAULTS.thinkingCoding,
      temperature: 0.9,
      topP: 0.8,
      reuseCache: true,
    });
  });

  it('overwrites samplingDefaults in place on re-register with the same model', async () => {
    const registry = new ModelRegistry();
    const model = createCapturingModel();

    registry.register('qwen', model, { samplingDefaults: QWEN_SAMPLING_DEFAULTS.thinkingCoding });
    registry.register('qwen', model, { samplingDefaults: QWEN_SAMPLING_DEFAULTS.instructReasoning });

    const sessionReg = registry.getSessionRegistry('qwen')!;
    expect(sessionReg.defaultSamplingConfig).toEqual(QWEN_SAMPLING_DEFAULTS.instructReasoning);

    const { session } = sessionReg.getOrCreate(null, null);
    await session.send('hi');
    expect(model.lastStartConfig).toEqual({
      ...QWEN_SAMPLING_DEFAULTS.instructReasoning,
      reuseCache: true,
    });
  });

  it('overwrites samplingDefaults on re-register with a different model', async () => {
    const registry = new ModelRegistry();
    const model1 = createCapturingModel();
    const model2 = createCapturingModel();

    registry.register('qwen', model1, { samplingDefaults: QWEN_SAMPLING_DEFAULTS.thinkingCoding });
    registry.register('qwen', model2, { samplingDefaults: QWEN_SAMPLING_DEFAULTS.instructGeneral });

    const sessionReg = registry.getSessionRegistry('qwen')!;
    expect(sessionReg.defaultSamplingConfig).toEqual(QWEN_SAMPLING_DEFAULTS.instructGeneral);

    const { session } = sessionReg.getOrCreate(null, null);
    await session.send('hi');
    expect(model2.lastStartConfig).toEqual({
      ...QWEN_SAMPLING_DEFAULTS.instructGeneral,
      reuseCache: true,
    });
    // The old model's chatSessionStart was never driven through the
    // new binding.
    expect(model1.startCallCount).toBe(0);
  });

  it('preserves samplingDefaults across a plain (no-opts) re-register of the same model', async () => {
    const registry = new ModelRegistry();
    const model = createCapturingModel();

    registry.register('qwen', model, { samplingDefaults: QWEN_SAMPLING_DEFAULTS.thinkingCoding });
    // Re-register with NO opts: the operator is just refreshing
    // createdAt. The previously-set defaults must survive so the next
    // cache-miss still allocates a ChatSession with the configured
    // knobs.
    registry.register('qwen', model);

    const sessionReg = registry.getSessionRegistry('qwen')!;
    expect(sessionReg.defaultSamplingConfig).toEqual(QWEN_SAMPLING_DEFAULTS.thinkingCoding);
  });

  it('clears samplingDefaults when re-register passes an explicit undefined', async () => {
    const registry = new ModelRegistry();
    const model = createCapturingModel();

    registry.register('qwen', model, { samplingDefaults: QWEN_SAMPLING_DEFAULTS.thinkingCoding });
    registry.register('qwen', model, { samplingDefaults: undefined });

    const sessionReg = registry.getSessionRegistry('qwen')!;
    expect(sessionReg.defaultSamplingConfig).toBeUndefined();
  });
});

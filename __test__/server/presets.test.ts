/**
 * Coverage for the launch-preset anti-repetition cutoff values.
 *
 * A live Claude Code session on Ornith (qwen3_5_moe) truncated an
 * ASCII box mid-diagram: a run of repeated `─` tripped the native
 * sampler's consecutive-token cutoff (`max_consecutive_tokens`
 * default 16) → finish_reason `"repetition"` → `stop_reason:"end_turn"`.
 * The Qwen launch presets loosen the cutoff so legit repeated content
 * (box borders, separators, tables) survives while a true runaway loop
 * still stops.
 *
 * The merge-survival tests drive the REAL `ChatSession.mergeConfig`
 * through `ModelRegistry.register({ samplingDefaults }) → getOrCreate →
 * session.send`, observing the merged ChatConfig the session forwards
 * to `chatSessionStart` (a session-capable mock — no native model).
 */

import type { ChatConfig, ChatMessage, ChatResult, ToolCallResult } from '@mlx-node/core';
import type { SessionCapableModel } from '@mlx-node/lm';
import {
  GEMMA4_SAMPLING_DEFAULTS,
  LAUNCH_PRESETS,
  LFM2_SAMPLING_DEFAULTS,
  ModelRegistry,
  QWEN_SAMPLING_DEFAULTS,
} from '@mlx-node/server';
import { describe, expect, it, vi } from 'vite-plus/test';

// ---------------------------------------------------------------------------
// Mock model (mirrors sampling-defaults.test.ts): captures the merged
// ChatConfig that ChatSession passes down to chatSessionStart.
// ---------------------------------------------------------------------------

interface CapturingModel extends SessionCapableModel {
  lastStartConfig: ChatConfig | null | undefined;
  startCallCount: number;
}

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
// Static preset-value assertions
// ---------------------------------------------------------------------------

const LOOSENED_CUTOFF = {
  maxConsecutiveTokens: 256,
  maxNgramRepeats: 8,
  ngramSize: 64,
} as const;

describe('QWEN launch-preset anti-repetition cutoff', () => {
  it('loosens the cutoff on every Qwen launch entry (qwen3 / qwen3_5 / qwen3_5_moe)', () => {
    for (const key of ['qwen3', 'qwen3_5', 'qwen3_5_moe'] as const) {
      const sampling = LAUNCH_PRESETS[key]!.sampling;
      expect(sampling.maxConsecutiveTokens).toBe(256);
      expect(sampling.maxNgramRepeats).toBe(8);
      expect(sampling.ngramSize).toBe(64);
    }
  });

  it('sets the loosened cutoff on all four QWEN_SAMPLING_DEFAULTS variants', () => {
    for (const key of ['thinkingCoding', 'thinkingGeneral', 'instructGeneral', 'instructReasoning'] as const) {
      const sampling = QWEN_SAMPLING_DEFAULTS[key];
      expect(sampling.maxConsecutiveTokens).toBe(256);
      expect(sampling.maxNgramRepeats).toBe(8);
      expect(sampling.ngramSize).toBe(64);
    }
  });

  it('does not disable the cutoff (runaway-loop protection must remain)', () => {
    for (const key of ['thinkingCoding', 'thinkingGeneral', 'instructGeneral', 'instructReasoning'] as const) {
      const sampling = QWEN_SAMPLING_DEFAULTS[key];
      expect(sampling.maxConsecutiveTokens).not.toBe(0);
      expect(sampling.maxNgramRepeats).not.toBe(0);
    }
  });

  it('leaves the existing sampling knobs intact (thinkingCoding spec)', () => {
    const s = QWEN_SAMPLING_DEFAULTS.thinkingCoding;
    expect(s.temperature).toBe(0.6);
    expect(s.topP).toBe(0.95);
    expect(s.topK).toBe(20);
    expect(s.minP).toBe(0);
    expect(s.presencePenalty).toBe(0);
    expect(s.repetitionPenalty).toBe(1);
  });
});

describe('scope lock: non-Qwen presets unchanged', () => {
  it('does not set the cutoff on the Gemma4 preset', () => {
    expect(GEMMA4_SAMPLING_DEFAULTS.maxConsecutiveTokens).toBeUndefined();
    expect(GEMMA4_SAMPLING_DEFAULTS.maxNgramRepeats).toBeUndefined();
    expect(GEMMA4_SAMPLING_DEFAULTS.ngramSize).toBeUndefined();
    expect(LAUNCH_PRESETS.gemma4!.sampling.maxConsecutiveTokens).toBeUndefined();
  });

  it('does not set the cutoff on the LFM2 preset', () => {
    expect(LFM2_SAMPLING_DEFAULTS.maxConsecutiveTokens).toBeUndefined();
    expect(LFM2_SAMPLING_DEFAULTS.maxNgramRepeats).toBeUndefined();
    expect(LFM2_SAMPLING_DEFAULTS.ngramSize).toBeUndefined();
    expect(LAUNCH_PRESETS.lfm2!.sampling.maxConsecutiveTokens).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Merge survival (real ChatSession.mergeConfig)
// ---------------------------------------------------------------------------

describe('cutoff survives ChatSession.mergeConfig', () => {
  it('carries the loosened cutoff through when the per-request overlay omits it', async () => {
    const registry = new ModelRegistry();
    const model = createCapturingModel();
    registry.register('qwen35moe', model, { samplingDefaults: LAUNCH_PRESETS.qwen3_5_moe!.sampling });

    const sessionReg = registry.getSessionRegistry('qwen35moe')!;
    const { session } = sessionReg.getOrCreate(null, null);

    // Claude Code never sends these fields — overlay omits them.
    await session.send('hi');

    expect(model.lastStartConfig).toMatchObject(LOOSENED_CUTOFF);
    // Full shape: defaults + reuseCache, nothing else injected.
    expect(model.lastStartConfig).toEqual({
      ...LAUNCH_PRESETS.qwen3_5_moe!.sampling,
      reuseCache: true,
    });
  });

  it('lets a per-request override win on maxConsecutiveTokens while the other cutoff fields survive', async () => {
    const registry = new ModelRegistry();
    const model = createCapturingModel();
    registry.register('qwen35moe', model, { samplingDefaults: LAUNCH_PRESETS.qwen3_5_moe!.sampling });

    const sessionReg = registry.getSessionRegistry('qwen35moe')!;
    const { session } = sessionReg.getOrCreate(null, null);

    await session.send('hi', { config: { maxConsecutiveTokens: 32 } });

    expect(model.lastStartConfig!.maxConsecutiveTokens).toBe(32);
    // Default n-gram fields still carried (overlay didn't touch them).
    expect(model.lastStartConfig!.maxNgramRepeats).toBe(8);
    expect(model.lastStartConfig!.ngramSize).toBe(64);
  });
});

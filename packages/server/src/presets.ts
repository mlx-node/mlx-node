/**
 * Sampling presets recommended by third-party model authors, exposed
 * here as `ChatConfig`-shaped objects so an operator can pin them at
 * `ModelRegistry.register(name, model, { samplingDefaults: ... })`
 * time with a single import.
 *
 * Per-request client values (OpenAI `temperature`/`top_p`, Anthropic
 * equivalents) still override these defaults where the client sends
 * them — `ChatSession.mergeConfig` treats per-call config as an
 * overlay on top of `defaultConfig`. These presets only fill in the
 * parameters clients never send (`top_k`, `min_p`, penalties).
 */
import type { ChatConfig } from '@mlx-node/core';

/**
 * Sampling defaults from Unsloth's Qwen3.6 guide:
 * https://unsloth.ai/docs/models/qwen3.6#recommended-settings
 *
 * All modes pin `top_k = 20` and `min_p = 0.0`; they differ in
 * `temperature`, `top_p`, and `presence_penalty`.
 */
export const QWEN_SAMPLING_DEFAULTS = {
  /** Thinking mode for precise coding tasks: temp=0.6, top_p=0.95, pp=0.0 */
  thinkingCoding: {
    temperature: 0.6,
    topP: 0.95,
    topK: 20,
    minP: 0.0,
    presencePenalty: 0.0,
    repetitionPenalty: 1.0,
  } satisfies ChatConfig,

  /** Thinking mode for general tasks: temp=1.0, top_p=0.95, pp=1.5 */
  thinkingGeneral: {
    temperature: 1.0,
    topP: 0.95,
    topK: 20,
    minP: 0.0,
    presencePenalty: 1.5,
    repetitionPenalty: 1.0,
  } satisfies ChatConfig,

  /** Instruct (non-thinking) for general tasks: temp=0.7, top_p=0.8, pp=1.5 */
  instructGeneral: {
    temperature: 0.7,
    topP: 0.8,
    topK: 20,
    minP: 0.0,
    presencePenalty: 1.5,
    repetitionPenalty: 1.0,
  } satisfies ChatConfig,

  /** Instruct (non-thinking) for reasoning tasks: temp=1.0, top_p=0.95, pp=1.5 */
  instructReasoning: {
    temperature: 1.0,
    topP: 0.95,
    topK: 20,
    minP: 0.0,
    presencePenalty: 1.5,
    repetitionPenalty: 1.0,
  } satisfies ChatConfig,
} as const;

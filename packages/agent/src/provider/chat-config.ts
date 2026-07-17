/**
 * Per-call `ChatConfig` assembly for the provider bridge.
 *
 * Base sampling + output budget come from `@mlx-node/server`'s
 * `LAUNCH_PRESETS` (the ONLY allowed server import in this package —
 * presets/preset types, nothing else) extended by the agent-local
 * {@link AGENT_LAUNCH_PRESETS}, then pi's per-call `SimpleStreamOptions`
 * overlay on top.
 */

import type { SimpleStreamOptions, ThinkingLevel } from '@earendil-works/pi-ai';
import type { ChatConfig, ModelType, ToolDefinition } from '@mlx-node/lm';
import { LAUNCH_PRESETS, type LaunchPreset } from '@mlx-node/server';

/**
 * Agent-local launch presets for model types `LAUNCH_PRESETS` does not
 * cover (kept here — this package must not fork `packages/server`).
 *
 * `lfm2_moe` (LFM2.5-8B-A1B): LiquidAI's HF model card for the MoE
 * checkpoint recommends temperature 0.2 / top_k 80 — deliberately NOT
 * the dense `lfm2` preset (LFM2.5-1.2B guidance: temperature 0.05 /
 * top_k 50). repetitionPenalty 1.05 and the 8192-token output budget
 * match the dense family entry.
 */
const AGENT_LAUNCH_PRESETS: Partial<Record<ModelType, LaunchPreset>> = {
  lfm2_moe: {
    sampling: {
      temperature: 0.2,
      topP: 1.0,
      topK: 80,
      minP: 0.0,
      presencePenalty: 0.0,
      repetitionPenalty: 1.05,
    },
    maxOutputTokens: 8192,
  },
};

/**
 * Preset lookup — agent-local entries win over `LAUNCH_PRESETS` (they
 * exist precisely because the server table has no correct entry for the
 * type). This is the ONE preset resolution shared by discovery
 * (`models.ts`) and per-call config assembly, so a model can never be
 * discovered without also being streamable (and vice versa).
 */
export function launchPresetFor(modelType: ModelType): LaunchPreset | undefined {
  return AGENT_LAUNCH_PRESETS[modelType] ?? LAUNCH_PRESETS[modelType];
}

/**
 * pi thinking level → native `reasoningEffort`. pi never delivers 'off'
 * here (the agent loop converts it to `undefined` before the provider
 * sees it), so `undefined` is the "thinking disabled" signal → 'none'.
 */
const THINKING_LEVEL_TO_EFFORT: Record<ThinkingLevel, 'low' | 'medium' | 'high'> = {
  minimal: 'low',
  low: 'low',
  medium: 'medium',
  high: 'high',
  xhigh: 'high',
  max: 'high',
};

export interface ResolvedReasoningMode {
  reasoningEffort: 'none' | 'low' | 'medium' | 'high';
  /** The `enable_thinking` value implied by `reasoningEffort` for templates. */
  thinkingEnabled: boolean;
}

/**
 * Resolve Pi's thinking level once for both native config and persisted replay
 * provenance. Keeping these values together prevents a low/minimal turn from
 * being replayed later as an enabled-thinking turn merely because the Pi
 * option was present.
 */
export function resolveReasoningMode(reasoning: ThinkingLevel | undefined): ResolvedReasoningMode {
  const reasoningEffort = reasoning === undefined ? 'none' : THINKING_LEVEL_TO_EFFORT[reasoning];
  return {
    reasoningEffort,
    thinkingEnabled: reasoningEffort === 'medium' || reasoningEffort === 'high',
  };
}

export function buildChatConfig(
  modelType: ModelType,
  options: SimpleStreamOptions | undefined,
  tools: ToolDefinition[] | undefined,
  rootCacheOwnerId?: string,
  resolvedReasoning = resolveReasoningMode(options?.reasoning),
): ChatConfig {
  const preset = launchPresetFor(modelType);
  if (!preset) {
    const known = [...new Set([...Object.keys(LAUNCH_PRESETS), ...Object.keys(AGENT_LAUNCH_PRESETS)])].join(', ');
    throw new Error(`buildChatConfig: no launch preset for model type "${modelType}" (known types: ${known})`);
  }

  const config: ChatConfig = {
    ...preset.sampling,
    maxNewTokens: preset.maxOutputTokens,
    reasoningEffort: resolvedReasoning.reasoningEffort,
    // The terminal native chunk carries TTFT/prefill/decode telemetry when
    // requested. The provider keeps it transient and only renders it in TUI.
    reportPerformance: true,
  };
  // Pi assigns one stable id to the root AgentSession and a distinct id to
  // every in-memory subagent session. Native Qwen3.5 uses this only to retain
  // GDN sidecars per logical branch; PagedAttention KV blocks remain shared by
  // their existing exact content hashes.
  if (options?.sessionId !== undefined) config.cacheOwnerId = options.sessionId;
  // The active owner above can be a child AgentSession. Keep the current
  // top-level session identity separate so a /new or /resume rotation updates
  // which branch the bounded GDN sidecar store protects from child eviction.
  if (rootCacheOwnerId !== undefined) config.cacheRootOwnerId = rootCacheOwnerId;
  if (options?.maxTokens !== undefined) config.maxNewTokens = options.maxTokens;
  if (options?.temperature !== undefined) config.temperature = options.temperature;
  if (tools && tools.length > 0) config.tools = tools;
  // `reuseCache` is deliberately NOT set: ChatSession.mergeConfig forces it on.
  return config;
}

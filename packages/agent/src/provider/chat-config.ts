/**
 * Per-call `ChatConfig` assembly for the provider bridge.
 *
 * Base sampling + output budget come from the family-data launch presets
 * (`@mlx-node/lm` — the agent overlay wins over the server row), then pi's
 * per-call `SimpleStreamOptions` overlay on top.
 */

import type { SimpleStreamOptions, ThinkingLevel } from '@earendil-works/pi-ai';
import {
  agentLaunchPresetFor,
  MODEL_FAMILY_DATA,
  type ChatConfig,
  type ModelFamilyData,
  type ModelType,
  type ToolDefinition,
} from '@mlx-node/lm';

/**
 * Model types the no-preset error names, reproducing byte-for-byte the key
 * order the message has always printed (the historical server
 * `LAUNCH_PRESETS` literal, then the agent-only overlay): trainable rows,
 * server-preset loadable rows, then agent-only rows, each group in registry
 * order. Pinned by `packages/agent/__test__/chat-config.test.ts`.
 */
const KNOWN_PRESET_MODEL_TYPES: readonly string[] = (() => {
  const rows: readonly ModelFamilyData[] = MODEL_FAMILY_DATA;
  const chatRows = rows.filter((row) => row.kind === 'trainable' || row.kind === 'loadable');
  const serverRows = chatRows.filter((row) => row.launchPreset !== undefined);
  return [
    ...serverRows.filter((row) => row.kind === 'trainable'),
    ...serverRows.filter((row) => row.kind === 'loadable'),
    ...chatRows.filter((row) => row.launchPreset === undefined),
  ].map((row) => row.id);
})();

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

function isPositiveSafeInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0;
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
  modelMaxTokens?: unknown,
): ChatConfig {
  const preset = agentLaunchPresetFor(modelType);
  if (!preset) {
    const known = KNOWN_PRESET_MODEL_TYPES.join(', ');
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
  const explicitMaxTokens = options?.maxTokens;
  if (isPositiveSafeInteger(explicitMaxTokens)) {
    // A valid per-call provider option is the topmost layer.
    config.maxNewTokens = explicitMaxTokens;
  } else if (isPositiveSafeInteger(modelMaxTokens)) {
    // Normal Pi agent turns omit SimpleStreamOptions.maxTokens. Honor the
    // composed Model metadata (including models.json modelOverrides) without
    // allowing malformed/hostile metadata to replace the family preset.
    config.maxNewTokens = modelMaxTokens;
  }
  if (options?.temperature !== undefined) config.temperature = options.temperature;
  if (tools && tools.length > 0) config.tools = tools;
  // `reuseCache` is deliberately NOT set: ChatSession.mergeConfig forces it on.
  return config;
}

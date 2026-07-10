/**
 * Per-call `ChatConfig` assembly for the provider bridge.
 *
 * Base sampling + output budget come from `@mlx-node/server`'s
 * `LAUNCH_PRESETS` (the ONLY allowed server import in this package —
 * presets/preset types, nothing else), then pi's per-call
 * `SimpleStreamOptions` overlay on top.
 */

import type { SimpleStreamOptions, ThinkingLevel } from '@earendil-works/pi-ai';
import type { ChatConfig, ModelType, ToolDefinition } from '@mlx-node/lm';
import { LAUNCH_PRESETS, type LaunchPreset } from '@mlx-node/server';

/**
 * Family variants that share the base family's `LAUNCH_PRESETS` entry.
 * `lfm2_moe` checkpoints load through the same LFM2.5 family as `lfm2`
 * (`Lfm2Model.load` serves both model-loader registry rows) and LiquidAI
 * publishes one set of sampling recommendations for the family, so the
 * MoE variant maps onto the `lfm2` preset here instead of forking
 * `packages/server`.
 */
const PRESET_FAMILY_ALIASES: Partial<Record<ModelType, string>> = {
  lfm2_moe: 'lfm2',
};

/**
 * `LAUNCH_PRESETS` lookup with {@link PRESET_FAMILY_ALIASES} applied —
 * the ONE preset resolution shared by discovery (`models.ts`) and
 * per-call config assembly, so a model can never be discovered without
 * also being streamable (and vice versa).
 */
export function launchPresetFor(modelType: ModelType): LaunchPreset | undefined {
  const direct = LAUNCH_PRESETS[modelType];
  if (direct) return direct;
  const alias = PRESET_FAMILY_ALIASES[modelType];
  return alias === undefined ? undefined : LAUNCH_PRESETS[alias];
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

export function buildChatConfig(
  modelType: ModelType,
  options: SimpleStreamOptions | undefined,
  tools: ToolDefinition[] | undefined,
): ChatConfig {
  const preset = launchPresetFor(modelType);
  if (!preset) {
    const known = [...Object.keys(LAUNCH_PRESETS), ...Object.keys(PRESET_FAMILY_ALIASES)].join(', ');
    throw new Error(`buildChatConfig: no launch preset for model type "${modelType}" (known types: ${known})`);
  }

  const config: ChatConfig = {
    ...preset.sampling,
    maxNewTokens: preset.maxOutputTokens,
    reasoningEffort: options?.reasoning === undefined ? 'none' : THINKING_LEVEL_TO_EFFORT[options.reasoning],
  };
  if (options?.maxTokens !== undefined) config.maxNewTokens = options.maxTokens;
  if (options?.temperature !== undefined) config.temperature = options.temperature;
  if (tools && tools.length > 0) config.tools = tools;
  // `reuseCache` is deliberately NOT set: ChatSession.mergeConfig forces it on.
  return config;
}

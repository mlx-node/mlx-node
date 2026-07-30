/**
 * Local model discovery for the mlx pi provider.
 *
 * Ports the discovery walk from `@mlx-node/server/host`
 * (`packages/server/src/host/discover.ts`; that copy stays untouched) and
 * pairs every discovered checkpoint with a pi `ProviderModelConfig` entry
 * ready for `pi.registerProvider('mlx', { models })`.
 *
 * `contextWindow` starts as the checkpoint's trained window, read from the model dir's
 * `config.json` `max_position_embeddings` (root first, then the
 * `text_config` nesting used by qwen3_5 / qwen3_5_moe / gemma4 unified
 * checkpoints). Once a Qwen model loads, the provider narrows this shared
 * model metadata to the physical paged-cache window so pi's later
 * auto-compaction thresholds match reality. When both config fields are
 * absent the documented per-family fallback below applies.
 */

import type { Dirent } from 'node:fs';
import { readdir, readFile } from 'node:fs/promises';
import { basename, join } from 'node:path';

import type { ProviderModelConfig } from '@earendil-works/pi-coding-agent';
import { detectModelType, type ModelType } from '@mlx-node/lm';

import type { DiscoveredModelLike } from '../types.js';
import { launchPresetFor } from './chat-config.js';

/** A discovered local checkpoint paired with its pi provider model entry. */
export interface MlxModelInfo {
  discovered: DiscoveredModelLike;
  piModel: ProviderModelConfig;
}

// Non-generative detection results that cannot back a chat endpoint
// (mirrors the cli discover walk).
const NON_GENERATIVE: ReadonlySet<ModelType> = new Set<ModelType>(['harrier', 'qianfan-ocr', 'internvl_chat']);

interface FamilyTraits {
  /**
   * Whether the family emits `<think>` reasoning (drives pi's thinking
   * levels): true for qwen3 / qwen3_5 / qwen3_5_moe / gemma4 / lfm2 /
   * lfm2_moe. Gemma4 routes its `<|channel>thought` protocol through its
   * family stream parser rather than the generic `<think>` tracker, but the
   * user-facing level still controls the prompt's `<|think|>` capability.
   */
  reasoning: boolean;
  /**
   * Optional family-specific projection of Pi's thinking controls. Gemma4's
   * prompt protocol has two modes rather than four distinct effort levels:
   * minimal disables `<|think|>` and high enables it.
   */
  thinkingLevelMap?: ProviderModelConfig['thinkingLevelMap'];
  /**
   * Context-window fallback when `config.json` carries no
   * `max_position_embeddings` at either nesting level. Values are the
   * trained windows of the reference checkpoints: Qwen3 40960,
   * Qwen3.5 (+MoE) 262144, Gemma4 131072, LFM2.5 (dense + MoE) 128000
   * (`LFM2_CONFIGS[*].maxPositionEmbeddings` in `packages/lm`).
   */
  fallbackContextWindow: number;
}

interface DiscoveryMetadata {
  contextWindow: number;
  supportsImages: boolean;
}

/**
 * Keyed by `ModelType`: a chat-capable family must have BOTH an entry
 * here and a launch preset via `launchPresetFor` (which serves `lfm2_moe`
 * from the agent-local MoE preset) to be served — missing either side is
 * skipped, never guessed.
 */
const FAMILY_TRAITS: Record<string, FamilyTraits> = {
  qwen3: { reasoning: true, fallbackContextWindow: 40960 },
  qwen3_5: { reasoning: true, fallbackContextWindow: 262144 },
  qwen3_5_moe: { reasoning: true, fallbackContextWindow: 262144 },
  gemma4: {
    reasoning: true,
    thinkingLevelMap: {
      minimal: 'minimal',
      low: null,
      medium: null,
      high: 'high',
    },
    fallbackContextWindow: 131072,
  },
  lfm2: { reasoning: true, fallbackContextWindow: 128000 },
  lfm2_moe: { reasoning: true, fallbackContextWindow: 128000 },
};

function positiveInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.floor(value) : undefined;
}

function nonEmptyRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value) && Object.keys(value).length > 0;
}

/**
 * Read cheap discovery metadata from `<modelPath>/config.json`.
 *
 * The trained context window comes from:
 * root `max_position_embeddings` first (qwen3, lfm2), then
 * `text_config.max_position_embeddings` (qwen3_5, qwen3_5_moe, gemma4
 * unified), else the family fallback.
 *
 * Image support is advertised only when a family with a native multimodal
 * implementation carries its valid, non-empty vision marker: `vision_config`
 * for Qwen, and either `vision_config` or `unified_vision_config` for Gemma.
 * This lets Pi's model picker and `--list-models` expose checkpoint capability
 * without loading weights. The first resident load remains authoritative and
 * reconciles this optimistic config-level advertisement via
 * `session.supportsImages()` (for example, when conversion stripped an
 * incompatible vision tower).
 *
 * `detectModelType` already parsed this file, so a read/parse failure here
 * (e.g. a racing rewrite) lands on the context fallback and text-only input
 * instead of dropping the model or guessing a positive capability.
 */
async function readDiscoveryMetadata(
  modelPath: string,
  modelType: ModelType,
  fallbackContextWindow: number,
): Promise<DiscoveryMetadata> {
  try {
    const raw = await readFile(join(modelPath, 'config.json'), 'utf-8');
    const config = JSON.parse(raw) as Record<string, unknown>;
    const root = positiveInteger(config.max_position_embeddings);
    const textConfig = config.text_config;
    const nested = nonEmptyRecord(textConfig) ? positiveInteger(textConfig.max_position_embeddings) : undefined;
    const hasVisionConfig = nonEmptyRecord(config.vision_config);
    const supportsImages =
      modelType === 'gemma4'
        ? hasVisionConfig || nonEmptyRecord(config.unified_vision_config)
        : (modelType === 'qwen3_5' || modelType === 'qwen3_5_moe') && hasVisionConfig;

    return {
      contextWindow: root ?? nested ?? fallbackContextWindow,
      supportsImages,
    };
  } catch {
    return { contextWindow: fallbackContextWindow, supportsImages: false };
  }
}

/**
 * Scan `modelsDir` for chat-capable model subdirectories and build their
 * pi provider entries. Same tolerance as the cli discover walk: an
 * unreadable dir yields `[]`; entries with an undetectable config, a
 * non-generative type, or no launch preset are skipped silently
 * (warnings only when `MLX_DEBUG` is set). Cheap by contract — no
 * weights are loaded here. Results are sorted by directory name, which
 * becomes both the pi model `id` and display `name`.
 */
export async function discoverMlxModels(modelsDir: string): Promise<MlxModelInfo[]> {
  const debug = Boolean(process.env.MLX_DEBUG);

  let entries: Dirent[];
  try {
    entries = await readdir(modelsDir, { withFileTypes: true });
  } catch {
    return [];
  }

  const out: MlxModelInfo[] = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const full = join(modelsDir, entry.name);

    let modelType: ModelType;
    try {
      modelType = await detectModelType(full);
    } catch (err) {
      if (debug) console.warn(`[mlx] skip ${full}: ${(err as Error).message}`);
      continue;
    }

    if (NON_GENERATIVE.has(modelType)) continue;

    const preset = launchPresetFor(modelType);
    if (!preset) {
      if (debug) console.warn(`[mlx] skip ${full}: no launch preset for ${modelType}`);
      continue;
    }
    const traits = FAMILY_TRAITS[modelType];
    if (!traits) {
      if (debug) console.warn(`[mlx] skip ${full}: no FAMILY_TRAITS entry for ${modelType}`);
      continue;
    }

    const name = basename(full);
    const { contextWindow, supportsImages } = await readDiscoveryMetadata(full, modelType, traits.fallbackContextWindow);
    out.push({
      discovered: { name, path: full, modelType },
      piModel: {
        id: name,
        name,
        reasoning: traits.reasoning,
        thinkingLevelMap: traits.thinkingLevelMap,
        input: supportsImages ? ['text', 'image'] : ['text'],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow,
        maxTokens: preset.maxOutputTokens,
      },
    });
  }

  out.sort((a, b) => (a.discovered.name < b.discovered.name ? -1 : a.discovered.name > b.discovered.name ? 1 : 0));
  return out;
}

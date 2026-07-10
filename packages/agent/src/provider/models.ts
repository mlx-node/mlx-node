/**
 * Local model discovery for the mlx pi provider.
 *
 * Ports the discovery walk from
 * `packages/cli/src/commands/launch-claude/discover.ts` (which is bin-only
 * and must not be imported from here; the cli copy stays untouched) and
 * pairs every discovered checkpoint with a pi `ProviderModelConfig` entry
 * ready for `pi.registerProvider('mlx', { models })`.
 *
 * `contextWindow` is the checkpoint's REAL trained window — it drives
 * pi's auto-compaction thresholds — read from the model dir's
 * `config.json` `max_position_embeddings` (root first, then the
 * `text_config` nesting used by qwen3_5 / qwen3_5_moe / gemma4 unified
 * checkpoints). When both are absent the documented per-family fallback
 * below applies.
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
   * levels): true for qwen3 / qwen3_5 / qwen3_5_moe / lfm2 / lfm2_moe,
   * false for the instruct-only gemma4.
   */
  reasoning: boolean;
  /**
   * Context-window fallback when `config.json` carries no
   * `max_position_embeddings` at either nesting level. Values are the
   * trained windows of the reference checkpoints: Qwen3 40960,
   * Qwen3.5 (+MoE) 262144, Gemma4 131072, LFM2.5 (dense + MoE) 128000
   * (`LFM2_CONFIGS[*].maxPositionEmbeddings` in `packages/lm`).
   */
  fallbackContextWindow: number;
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
  gemma4: { reasoning: false, fallbackContextWindow: 131072 },
  lfm2: { reasoning: true, fallbackContextWindow: 128000 },
  lfm2_moe: { reasoning: true, fallbackContextWindow: 128000 },
};

function positiveInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.floor(value) : undefined;
}

/**
 * Read the trained context window from `<modelPath>/config.json`:
 * root `max_position_embeddings` first (qwen3, lfm2), then
 * `text_config.max_position_embeddings` (qwen3_5, qwen3_5_moe, gemma4
 * unified), else the family fallback. `detectModelType` already parsed
 * this file, so a read/parse failure here (e.g. a racing rewrite) lands
 * on the fallback instead of dropping the model.
 */
async function readContextWindow(modelPath: string, fallback: number): Promise<number> {
  try {
    const raw = await readFile(join(modelPath, 'config.json'), 'utf-8');
    const config = JSON.parse(raw) as Record<string, unknown>;
    const root = positiveInteger(config.max_position_embeddings);
    if (root !== undefined) return root;
    const textConfig = config.text_config;
    if (typeof textConfig === 'object' && textConfig !== null && !Array.isArray(textConfig)) {
      const nested = positiveInteger((textConfig as Record<string, unknown>).max_position_embeddings);
      if (nested !== undefined) return nested;
    }
  } catch {
    // fall through to the family fallback
  }
  return fallback;
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
    const contextWindow = await readContextWindow(full, traits.fallbackContextWindow);
    out.push({
      discovered: { name, path: full, modelType },
      piModel: {
        id: name,
        name,
        reasoning: traits.reasoning,
        input: ['text'],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow,
        maxTokens: preset.maxOutputTokens,
      },
    });
  }

  out.sort((a, b) => (a.discovered.name < b.discovered.name ? -1 : a.discovered.name > b.discovered.name ? 1 : 0));
  return out;
}

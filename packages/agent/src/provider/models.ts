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
 * checkpoints). Once a Qwen or Muse-Glimmer model loads, the provider narrows
 * this shared model metadata to the physical paged-cache window so pi's later
 * auto-compaction thresholds match reality. When both config fields are absent
 * the per-family fallback documented on `FamilyTraits` (`@mlx-node/lm`
 * family-data) applies.
 */

import type { Dirent } from 'node:fs';
import { readdir, readFile } from 'node:fs/promises';
import { basename, join } from 'node:path';

import type { ProviderModelConfig } from '@earendil-works/pi-coding-agent';
import { detectModelType, familyTraitsFor, NON_GENERATIVE_FAMILY_IDS, type ModelType } from '@mlx-node/lm';

import type { DiscoveredModelLike } from '../types.js';
import { launchPresetFor } from './chat-config.js';

/** A discovered local checkpoint paired with its pi provider model entry. */
export interface MlxModelInfo {
  discovered: DiscoveredModelLike;
  piModel: ProviderModelConfig;
}

interface DiscoveryMetadata {
  contextWindow: number;
  supportsImages: boolean;
  draftOnly: boolean;
}

/**
 * Native direct-GGUF loading currently exists for dense Qwen3.5/Qwen3.8.
 * Match the Unsloth Dynamic XL target names users download, while excluding
 * ordinary Q4_K_M files and companion artifacts such as imatrix/mmproj/draft.
 */
const QWEN35_XL_GGUF = /(?:^|[-_.])Q\d+_K_XL\.gguf$/i;
const GGUF_COMPANION_NAME = /(?:^|[-_.])(?:imatrix|mmproj|dflash|draft)(?:[-_.]|$)/i;

function isQwen35XlGguf(name: string): boolean {
  return QWEN35_XL_GGUF.test(name) && !GGUF_COMPANION_NAME.test(name);
}

function ggufModelName(name: string): string {
  return name.slice(0, -'.gguf'.length);
}

/**
 * Agent-local convention for pairing a dense Qwen3.5/Qwen3.8 target with an
 * external DFlash2 checkpoint. mlx-vlm accepts an explicit `--draft-model`
 * path and does not define a combined directory layout; the agent needs a
 * deterministic relationship it can discover without another CLI flag.
 */
async function embeddedDFlash2Path(modelDir: string): Promise<string | undefined> {
  const draftPath = join(modelDir, 'draft');
  try {
    const raw = await readFile(join(draftPath, 'config.json'), 'utf-8');
    const config = JSON.parse(raw) as Record<string, unknown>;
    return Array.isArray(config.architectures) && config.architectures.includes('DFlash2DraftModel')
      ? draftPath
      : undefined;
  } catch {
    return undefined;
  }
}

interface ModelFileInventory {
  xlGgufs: string[];
  hasGguf: boolean;
  hasSafetensors: boolean;
}

async function modelFileInventory(modelDir: string): Promise<ModelFileInventory> {
  try {
    const files = (await readdir(modelDir, { withFileTypes: true }))
      .filter((entry) => entry.isFile())
      .map((entry) => entry.name);
    return {
      xlGgufs: files.filter(isQwen35XlGguf).sort(),
      hasGguf: files.some((name) => name.toLowerCase().endsWith('.gguf')),
      hasSafetensors: files.some((name) => name.toLowerCase().endsWith('.safetensors')),
    };
  } catch {
    return { xlGgufs: [], hasGguf: false, hasSafetensors: false };
  }
}

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
    const draftOnly = Array.isArray(config.architectures) && config.architectures.includes('DFlash2DraftModel');

    return {
      contextWindow: root ?? nested ?? fallbackContextWindow,
      supportsImages,
      draftOnly,
    };
  } catch {
    return { contextWindow: fallbackContextWindow, supportsImages: false, draftOnly: false };
  }
}

/**
 * Scan `modelsDir` for chat-capable model subdirectories and native dense
 * Qwen3.5/Qwen3.8 `Q<number>_K_XL.gguf` files, then build their pi provider
 * entries. XL files may live directly under `modelsDir` or one level inside a
 * downloaded GGUF repository. Each is registered by filename stem so multiple
 * quant variants in one repository remain independently selectable.
 *
 * Same tolerance as the cli discover walk: an unreadable dir yields `[]`;
 * entries with an undetectable config, a non-generative type, or no launch
 * preset are skipped silently (warnings only when `MLX_DEBUG` is set). Cheap
 * by contract — no weights are loaded here. Results are sorted by model name.
 */
export async function discoverMlxModels(modelsDir: string): Promise<MlxModelInfo[]> {
  const debug = Boolean(process.env.MLX_DEBUG);

  let entries: Dirent[];
  try {
    entries = await readdir(modelsDir, { withFileTypes: true });
  } catch {
    return [];
  }
  // Collision resolution below gives the first occurrence the bare filename
  // stem. Directory enumeration order is unspecified, so sort before assigning
  // IDs to keep persisted `mlx/<id>` selections stable across filesystems.
  entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));

  const out: MlxModelInfo[] = [];
  const usedNames = new Set<string>();

  const append = async (
    preferredName: string,
    path: string,
    metadataRoot: string,
    modelType: ModelType,
    scopeName: string,
    draftModelPath?: string,
  ): Promise<void> => {
    if (NON_GENERATIVE_FAMILY_IDS.has(modelType)) return;

    // Fail-closed guards: dead-by-construction for chat families (the
    // family-data row type requires traits + a preset), live for any foreign
    // string that slips through detection.
    const preset = launchPresetFor(modelType);
    if (!preset) {
      if (debug) console.warn(`[mlx] skip ${path}: no launch preset for ${modelType}`);
      return;
    }
    const traits = familyTraitsFor(modelType);
    if (!traits) {
      if (debug) console.warn(`[mlx] skip ${path}: no FAMILY_TRAITS entry for ${modelType}`);
      return;
    }

    const metadata = await readDiscoveryMetadata(metadataRoot, modelType, traits.fallbackContextWindow);
    if (metadata.draftOnly) {
      if (debug) console.warn(`[mlx] skip ${path}: companion draft checkpoint is not a chat model`);
      return;
    }

    let name = preferredName;
    if (usedNames.has(name)) {
      name = `${scopeName}-${preferredName}`;
      let suffix = 2;
      while (usedNames.has(name)) name = `${scopeName}-${preferredName}-${suffix++}`;
    }
    usedNames.add(name);
    const discovered: DiscoveredModelLike = {
      name,
      path,
      modelType,
      ...(draftModelPath === undefined ? {} : { draftModelPath }),
    };
    out.push({
      discovered,
      piModel: {
        id: name,
        name,
        reasoning: traits.reasoning,
        // The structural FamilyThinkingLevelMap must stay assignable to the
        // pi type (pi types are agent-only, so family-data cannot name it).
        thinkingLevelMap: traits.thinkingLevelMap satisfies ProviderModelConfig['thinkingLevelMap'],
        input: metadata.supportsImages ? ['text', 'image'] : ['text'],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: metadata.contextWindow,
        maxTokens: preset.maxOutputTokens,
      },
    });
  };

  for (const entry of entries) {
    if (entry.isFile() && isQwen35XlGguf(entry.name)) {
      const full = join(modelsDir, entry.name);
      try {
        const modelType = await detectModelType(full);
        if (modelType === 'qwen3_5') {
          await append(ggufModelName(entry.name), full, modelsDir, modelType, basename(modelsDir));
        } else if (debug) {
          console.warn(`[mlx] skip ${full}: direct XL GGUF loading is not supported for ${modelType}`);
        }
      } catch (err) {
        if (debug) console.warn(`[mlx] skip ${full}: ${(err as Error).message}`);
      }
      continue;
    }
    if (!entry.isDirectory()) continue;
    const full = join(modelsDir, entry.name);

    let modelType: ModelType;
    try {
      modelType = await detectModelType(full);
    } catch (err) {
      if (debug) console.warn(`[mlx] skip ${full}: ${(err as Error).message}`);
      continue;
    }

    const inventory = await modelFileInventory(full);
    const { xlGgufs } = inventory;
    if (xlGgufs.length > 0) {
      if (modelType !== 'qwen3_5') {
        if (debug) {
          console.warn(`[mlx] skip ${full}: direct XL GGUF loading is not supported for ${modelType}`);
        }
        continue;
      }
      const draftModelPath = await embeddedDFlash2Path(full);
      for (const gguf of xlGgufs) {
        await append(ggufModelName(gguf), join(full, gguf), full, modelType, entry.name, draftModelPath);
      }
      continue;
    }

    // A raw GGUF repository is not itself a loadable model path. Only the
    // selected native Qwen3.5 XL files above may be handed directly to the
    // loader. Keep converted model directories discoverable when they retain
    // an imatrix/source GGUF beside their actual SafeTensors weights.
    if (inventory.hasGguf && !inventory.hasSafetensors) {
      if (debug) console.warn(`[mlx] skip ${full}: no supported direct GGUF target`);
      continue;
    }

    const draftModelPath = modelType === 'qwen3_5' ? await embeddedDFlash2Path(full) : undefined;
    await append(basename(full), full, full, modelType, entry.name, draftModelPath);
  }

  out.sort((a, b) => (a.discovered.name < b.discovered.name ? -1 : a.discovered.name > b.discovered.name ? 1 : 0));
  return out;
}

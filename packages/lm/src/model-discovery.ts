/** Shared, native-free discovery of local chat checkpoints. */

import type { Dirent } from 'node:fs';
import { readdir, stat } from 'node:fs/promises';
import { basename, join } from 'node:path';

import {
  launchPresetFor,
  familyTraitsFor,
  NON_GENERATIVE_FAMILY_IDS,
  MODEL_FAMILY_DATA,
  type FamilyTraits,
  type LaunchPreset,
  type ModelType,
} from './family-data.js';
import { detectModelType, readGgufArchitecture, readModelConfig } from './model-detection.js';

/** Native-free inventory shared by the agent, setup UI, and inference host. */
export interface LocalChatModel {
  name: string;
  path: string;
  modelType: ModelType;
  preset: LaunchPreset;
  traits: FamilyTraits;
  contextWindow: number;
  supportsImages: boolean;
}

interface DiscoveryMetadata {
  contextWindow: number;
  supportsImages: boolean;
  draftOnly: boolean;
}

/**
 * The Qwen3.5/Qwen3.8 discovery filter retains its XL policy. Gemma4 and Muse
 * accept all supported tensor formats, including Q4_0 QAT checkpoints.
 * Match the Unsloth Dynamic XL target names users download, while excluding
 * ordinary Q4_K_M files and companion artifacts such as imatrix/mmproj/draft.
 */
const QWEN35_XL_GGUF = /(?:^|[-_.])Q\d+_K_XL\.gguf$/i;
const GGUF_COMPANION_NAME = /(?:^|[-_.])(?:imatrix|mmproj|dflash|draft)(?:[-_.]|$)/i;
// Match the native loaders' primary files/shards. A draft or projector
// SafeTensors file beside a GGUF is not a converted target checkpoint.
const PRIMARY_SAFETENSORS = /^(?:model|weights)\.safetensors$|^model(?:-|\.safetensors-).+-of-.+\.safetensors$/;

function isQwen35XlGguf(name: string): boolean {
  return QWEN35_XL_GGUF.test(name) && !GGUF_COMPANION_NAME.test(name);
}

function ggufModelName(name: string): string {
  return name.slice(0, -'.gguf'.length);
}

function requiresGgufAssets(modelType: ModelType): boolean {
  return modelType === 'gemma4' || modelType === 'muse_glimmer';
}

function matchesGgufFamily(path: string, modelType: ModelType): boolean {
  const architecture = readGgufArchitecture(path);
  return MODEL_FAMILY_DATA.some(
    (family) =>
      family.id === modelType &&
      'ggufArchitectures' in family &&
      family.ggufArchitectures.some((supported) => supported === architecture),
  );
}

async function hasGgufAssets(modelDir: string): Promise<boolean> {
  try {
    const assets = await Promise.all(['config.json', 'tokenizer.json'].map((name) => stat(join(modelDir, name))));
    return assets.every((asset) => asset.isFile());
  } catch {
    return false;
  }
}

interface ModelFileInventory {
  xlGgufs: string[];
  targetGgufs: string[];
  hasGguf: boolean;
  hasSafetensors: boolean;
  hasPrimarySafetensors: boolean;
}

async function modelFileInventory(modelDir: string): Promise<ModelFileInventory> {
  try {
    const files = (await readdir(modelDir, { withFileTypes: true }))
      .filter((entry) => entry.isFile())
      .map((entry) => entry.name);
    return {
      xlGgufs: files.filter(isQwen35XlGguf).sort(),
      targetGgufs: files
        .filter((name) => name.toLowerCase().endsWith('.gguf') && !GGUF_COMPANION_NAME.test(name))
        .sort(),
      hasGguf: files.some((name) => name.toLowerCase().endsWith('.gguf')),
      hasSafetensors: files.some((name) => name.toLowerCase().endsWith('.safetensors')),
      hasPrimarySafetensors: files.some((name) => PRIMARY_SAFETENSORS.test(name)),
    };
  } catch {
    return { xlGgufs: [], targetGgufs: [], hasGguf: false, hasSafetensors: false, hasPrimarySafetensors: false };
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
    const config = (await readModelConfig(modelPath)) as Record<string, unknown>;
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
 * Scan `modelsDir` for chat-capable model subdirectories, Gemma4/Muse GGUFs, and
 * dense Qwen3.5/Qwen3.8 `Q<number>_K_XL.gguf` files. GGUF files may live directly
 * under `modelsDir` or one level inside a downloaded GGUF repository. Each is
 * registered by filename stem so quant variants remain independently selectable.
 *
 * An unreadable dir yields `[]`. Entries with an undetectable config, a
 * non-generative type, or no launch preset are skipped silently (warnings only
 * when `MLX_DEBUG` is set). No weights are loaded. Results are sorted by name.
 */
export async function discoverLocalChatModels(modelsDir: string): Promise<LocalChatModel[]> {
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

  const out: LocalChatModel[] = [];
  const usedNames = new Set<string>();

  const append = async (
    preferredName: string,
    path: string,
    metadataRoot: string,
    modelType: ModelType,
    scopeName: string,
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
    // Automatic companions can be installed or removed after this startup
    // scan. The host resolves them on each load; draftModelPath is reserved
    // for caller-supplied paths that the loader must treat as authoritative.
    out.push({
      name,
      path,
      modelType,
      preset,
      traits,
      contextWindow: metadata.contextWindow,
      supportsImages: metadata.supportsImages,
    });
  };

  for (const entry of entries) {
    if (entry.isFile() && entry.name.toLowerCase().endsWith('.gguf') && !GGUF_COMPANION_NAME.test(entry.name)) {
      const full = join(modelsDir, entry.name);
      try {
        const modelType = await detectModelType(full);
        // A shared sibling config can describe another target or a projector.
        // Never advertise a file under a loader that disagrees with its header.
        if (!matchesGgufFamily(full, modelType)) continue;
        if (requiresGgufAssets(modelType) && !(await hasGgufAssets(modelsDir))) {
          if (debug)
            console.warn(
              `[mlx] skip ${full}: native ${modelType} GGUF requires sibling config.json and tokenizer.json`,
            );
          continue;
        }
        if (
          modelType === 'gemma4' ||
          modelType === 'muse_glimmer' ||
          (modelType === 'qwen3_5' && isQwen35XlGguf(entry.name))
        ) {
          await append(ggufModelName(entry.name), full, modelsDir, modelType, basename(modelsDir));
        } else if (debug) {
          console.warn(`[mlx] skip ${full}: no supported direct GGUF target for ${modelType}`);
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
    const hasModelWeights = requiresGgufAssets(modelType) ? inventory.hasPrimarySafetensors : inventory.hasSafetensors;
    if (requiresGgufAssets(modelType) && !hasModelWeights && inventory.targetGgufs.length > 0) {
      if (!(await hasGgufAssets(full))) {
        if (debug)
          console.warn(`[mlx] skip ${full}: native ${modelType} GGUF requires sibling config.json and tokenizer.json`);
        continue;
      }
      for (const gguf of inventory.targetGgufs) {
        const path = join(full, gguf);
        try {
          if (!matchesGgufFamily(path, modelType)) continue;
          await append(ggufModelName(gguf), path, full, modelType, entry.name);
        } catch (err) {
          if (debug) console.warn(`[mlx] skip ${path}: ${(err as Error).message}`);
        }
      }
      continue;
    }
    const { xlGgufs } = inventory;
    if (xlGgufs.length > 0 && !inventory.hasPrimarySafetensors) {
      if (modelType !== 'qwen3_5') {
        if (debug) {
          console.warn(`[mlx] skip ${full}: direct XL GGUF loading is not supported for ${modelType}`);
        }
        continue;
      }
      for (const gguf of xlGgufs) {
        const path = join(full, gguf);
        try {
          if (matchesGgufFamily(path, modelType)) await append(ggufModelName(gguf), path, full, modelType, entry.name);
        } catch (err) {
          if (debug) console.warn(`[mlx] skip ${path}: ${(err as Error).message}`);
        }
      }
      continue;
    }

    // Present each supported GGUF variant separately in the picker. Keep
    // converted model directories discoverable when they retain
    // an imatrix/source GGUF beside their actual SafeTensors weights.
    if (inventory.hasGguf && !hasModelWeights) {
      if (debug) console.warn(`[mlx] skip ${full}: no supported direct GGUF target`);
      continue;
    }

    await append(basename(full), full, full, modelType, entry.name);
  }

  out.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  return out;
}

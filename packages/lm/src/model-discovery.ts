/** Shared, native-free discovery of local chat checkpoints. */

import type { Dirent } from 'node:fs';
import { readdir, stat } from 'node:fs/promises';
import { basename, join } from 'node:path';

import {
  launchPresetFor,
  familyTraitsFor,
  NON_GENERATIVE_FAMILY_IDS,
  familyDataFor,
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
const BONSAI_PQ2_GGUF = /^ternary-bonsai-2-27b-pq2_0\.gguf$/i;
// `mtp` joins the rule because the catalog ships MTP weights BESIDE a target
// (Qwen3.8's `MTP/mtp-*.gguf`, Gemma's `mtp-*.gguf`) and nothing in the runtime
// pairs a standalone GGUF MTP file: counting one as weights certifies a
// directory the loader cannot open, and discovery would enumerate the sidecar
// as a model.
const GGUF_COMPANION_NAME = /(?:^|[-_.])(?:imatrix|mmproj|dflash|draft|mtp)(?:[-_.]|$)/i;

/**
 * True when a `.gguf` filename is a companion artifact (projector, calibration,
 * draft, MTP) rather than a loadable model payload. Discovery, the dashboard's
 * publish gate, and the CLI's weight classification all key off this one rule —
 * a file any of them certifies as weights is a file the others must accept.
 */
export function isGgufCompanionName(fileName: string): boolean {
  return GGUF_COMPANION_NAME.test(fileName);
}
// Match the native loaders' primary files/shards. A draft or projector
// SafeTensors file beside a GGUF is not a converted target checkpoint.
const PRIMARY_SAFETENSORS = /^(?:model|weights)\.safetensors$|^model(?:-|\.safetensors-).+-of-.+\.safetensors$/;

function isQwen35XlGguf(name: string): boolean {
  return QWEN35_XL_GGUF.test(name) && !GGUF_COMPANION_NAME.test(name);
}

function ggufModelName(name: string): string {
  return name.slice(0, -'.gguf'.length);
}

function supportedGgufName(name: string, modelType: ModelType): boolean {
  const policy = familyDataFor(modelType)?.ggufDiscovery;
  if (!policy || GGUF_COMPANION_NAME.test(name)) return false;
  const split = /-(\d{5})-of-(\d{5})\.gguf$/i.exec(name);
  if (split) {
    const total = Number(split[2]);
    if (Number(split[1]) !== 1 || total < 1 || total > 1024 || (total > 1 && !policy.split)) return false;
  }
  if (modelType === 'qwen3_5' && BONSAI_PQ2_GGUF.test(name)) return true;
  return policy.variants === 'all' || isQwen35XlGguf(name);
}

async function hasGgufAssets(modelDir: string, onIoFailure?: (error: unknown) => void): Promise<boolean> {
  try {
    const assets = await Promise.all(['config.json', 'tokenizer.json'].map((name) => stat(join(modelDir, name))));
    return assets.every((asset) => asset.isFile());
  } catch (error) {
    onIoFailure?.(error);
    return false;
  }
}

interface ModelFileInventory {
  targetGgufs: string[];
  hasGguf: boolean;
  hasPrimarySafetensors: boolean;
}

async function modelFileInventory(modelDir: string): Promise<ModelFileInventory> {
  const files = (await readdir(modelDir, { withFileTypes: true }))
    .filter((entry) => entry.isFile())
    .map((entry) => entry.name);
  return {
    targetGgufs: files.filter((name) => name.toLowerCase().endsWith('.gguf') && !GGUF_COMPANION_NAME.test(name)).sort(),
    hasGguf: files.some((name) => name.toLowerCase().endsWith('.gguf')),
    hasPrimarySafetensors: files.some((name) => PRIMARY_SAFETENSORS.test(name)),
  };
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
    const supportsImages =
      familyDataFor(modelType)?.visionConfigKeys?.some((key) => nonEmptyRecord(config[key])) ?? false;
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
 * Walk the cause chain for an errno. An "absent" code (ENOENT/ENOTDIR) is a
 * definitive answer — the config or asset does not exist — not a failure, and
 * a code-free error (bad GGUF header, corrupt JSON, unsupported family) is a
 * definitive "not this model". Only a real I/O errno means the entry could
 * not be evaluated at all and MIGHT be a model missing from the result.
 */
function isEvaluationFailure(error: unknown): boolean {
  let current: unknown = error;
  while (current instanceof Error) {
    const code = (current as NodeJS.ErrnoException).code;
    if (code !== undefined) return code !== 'ENOENT' && code !== 'ENOTDIR';
    current = current.cause;
  }
  return false;
}

/** Options for {@link discoverLocalChatModels}. */
export interface DiscoveryScanOptions {
  /**
   * Called when a directory entry or GGUF file could not be evaluated and may
   * have been omitted from the result — the scan was INCOMPLETE. Per-entry
   * failures are still swallowed so one corrupt model never kills discovery
   * for the server host, agent provider, and dashboard that share this code;
   * the callback only exposes that the result is not evidence of emptiness.
   * Lets callers separate "scan completed, zero models" from "couldn't look" —
   * a permanent versus retryable empty result.
   */
  onEntryFailure?: (error: unknown, entryPath: string) => void;
}

/**
 * Scan `modelsDir` for chat-capable model subdirectories, Gemma4/Muse/Qwen4 GGUFs, and
 * dense Qwen3.5/Qwen3.8 `Q<number>_K_XL.gguf` files. GGUF files may live directly
 * under `modelsDir` or one level inside a downloaded GGUF repository. Each is
 * registered by filename stem so quant variants remain independently selectable.
 *
 * A MISSING dir yields `[]` — nothing is installed. A dir that exists but
 * cannot be read THROWS: "scan failed" is not "confirmed empty", and callers
 * decide those differently (the desktop supervisor treats confirmed-empty as
 * permanent and stops retrying; an I/O error may clear on the next attempt).
 * Entries with an undetectable config, a non-generative type, or no launch
 * preset are skipped silently (warnings only when `MLX_DEBUG` is set) — but an
 * entry the scan could not even evaluate is reported through
 * `opts.onEntryFailure`, so an empty result carries evidence of whether it
 * means "nothing installed" or "couldn't check". No weights are loaded.
 * Results are sorted by name.
 */
export async function discoverLocalChatModels(
  modelsDir: string,
  opts?: DiscoveryScanOptions,
): Promise<LocalChatModel[]> {
  const debug = Boolean(process.env.MLX_DEBUG);
  const reportFailure = (error: unknown, entryPath: string): void => {
    if (isEvaluationFailure(error)) opts?.onEntryFailure?.(error, entryPath);
  };

  let entries: Dirent[];
  try {
    entries = await readdir(modelsDir, { withFileTypes: true });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return [];
    throw error;
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

  const appendGguf = async (name: string, metadataRoot: string, scopeName: string): Promise<void> => {
    const path = join(metadataRoot, name);
    try {
      const modelType = await detectModelType(path);
      const family = familyDataFor(modelType);
      if (!supportedGgufName(name, modelType)) return;
      // A sibling config must agree with the target header, never a projector.
      if (!family?.ggufArchitectures?.includes(await readGgufArchitecture(path))) return;
      if (
        family.ggufDiscovery?.requiresAssets &&
        !(await hasGgufAssets(metadataRoot, (error) => reportFailure(error, path)))
      ) {
        if (debug)
          console.warn(`[mlx] skip ${path}: native ${modelType} GGUF requires sibling config.json and tokenizer.json`);
        return;
      }
      await append(ggufModelName(name), path, metadataRoot, modelType, scopeName);
    } catch (err) {
      reportFailure(err, path);
      if (debug) console.warn(`[mlx] skip ${path}: ${(err as Error).message}`);
    }
  };

  for (const entry of entries) {
    if (entry.isFile() && entry.name.toLowerCase().endsWith('.gguf') && !GGUF_COMPANION_NAME.test(entry.name)) {
      await appendGguf(entry.name, modelsDir, basename(modelsDir));
      continue;
    }
    // Native Qwen4 GGUF loads publish tokenizer/config sidecars here, including
    // temporary directories during publication. They contain no model weights.
    if (!entry.isDirectory() || entry.name.startsWith('.mlx-qwen4-assets-')) continue;
    const full = join(modelsDir, entry.name);
    try {
      const inventory = await modelFileInventory(full);
      // Prefer converted targets over retained source files or companion weights.
      // Each native GGUF variant is otherwise a separate entry. Header detection
      // also handles Qwen4 split directories with no config/tokenizer sidecars.
      if (inventory.hasGguf && !inventory.hasPrimarySafetensors) {
        for (const name of inventory.targetGgufs) await appendGguf(name, full, entry.name);
        continue;
      }
      const modelType = await detectModelType(full);
      await append(entry.name, full, full, modelType, entry.name);
    } catch (err) {
      reportFailure(err, full);
      if (debug) console.warn(`[mlx] skip ${full}: ${(err as Error).message}`);
    }
  }

  out.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  return out;
}

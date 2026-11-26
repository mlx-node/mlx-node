/** Shared, native-free discovery of local chat checkpoints. */

import type { Dirent } from 'node:fs';
import { readdir, realpath, stat } from 'node:fs/promises';
import { basename, dirname, join } from 'node:path';

import {
  launchPresetFor,
  familyTraitsFor,
  NON_GENERATIVE_FAMILY_IDS,
  familyDataFor,
  reasoningEffortsFor,
  type FamilyTraits,
  type LaunchPreset,
  type ModelType,
} from './family-data.js';
import { isBonsaiPq2GgufName, isGgufCompanionName, isQwen35XlGgufName, parseGgufFileName } from './gguf-name.js';
import {
  detectModelType,
  ggufArchitectureModelType,
  readDirectoryChatTemplate,
  readGgufDiscoveryMetadata,
  readModelConfig,
  type GgufDiscoveryMetadata,
} from './model-detection.js';
import {
  folderVariant,
  ggufNames,
  ggufVariant,
  settleVariants,
  type ModelFormat,
  type ModelVariantName,
} from './model-variant.js';

export { ggufSplitPartNames, isGgufCompanionName, parseGgufFileName, type GgufFileName } from './gguf-name.js';
export type { ModelFormat } from './model-variant.js';

/** Native-free inventory shared by the agent, setup UI, and inference host. */
export interface LocalChatModel {
  name: string;
  path: string;
  modelType: ModelType;
  preset: LaunchPreset;
  traits: FamilyTraits;
  contextWindow: number;
  supportsImages: boolean;
  /** The model without its packaging: `Qwen3.8-Flash-Next`. Equal strings mean one model. */
  base: string;
  /** The build of that model: `UD-Q4_K_XL`, `MXFP4 Unsloth`, `BF16`. Unique inside one base. Never empty. */
  variant: string;
  format: ModelFormat;
}

interface DiscoveryMetadata {
  contextWindow: number;
  supportsImages: boolean;
  draftOnly: boolean;
  /** The parsed `config.json`; undefined when it could not be read. */
  config: unknown;
}

// Match the native loaders' primary files/shards. A draft or projector
// SafeTensors file beside a GGUF is not a converted target checkpoint.
const PRIMARY_SAFETENSORS = /^(?:model|weights)\.safetensors$|^model(?:-|\.safetensors-).+-of-.+\.safetensors$/;

/**
 * The parts of a split GGUF name `STEM-0000K-of-0000N.gguf` (any `.gguf`
 * case; K and N five digits, 1 <= K <= N <= 1024), or null.
 */
export function ggufSplitPart(name: string): { stem: string; part: number; total: number } | null {
  if (name.length < 5 || name.slice(-5).toLowerCase() !== '.gguf') return null;
  const base = name.slice(0, -5);
  const digits = (text: string): number | null => {
    if (text.length !== 5) return null;
    for (const ch of text) if (ch < '0' || ch > '9') return null;
    return Number(text);
  };
  const of = base.lastIndexOf('-of-');
  if (of < 0) return null;
  const total = digits(base.slice(of + 4));
  const dash = base.lastIndexOf('-', of - 1);
  if (total === null || dash < 0) return null;
  const part = digits(base.slice(dash + 1, of));
  if (part === null || part < 1 || total > 1024 || part > total) return null;
  return { stem: base.slice(0, dash), part, total };
}

function supportedGgufName(name: string, modelType: ModelType): boolean {
  const policy = familyDataFor(modelType)?.ggufDiscovery;
  if (!policy || isGgufCompanionName(name)) return false;
  const split = parseGgufFileName(name);
  if (split !== undefined && (split.ordinal !== 1 || split.shards !== 1)) {
    const { ordinal, shards } = split;
    if (ordinal !== 1 || shards < 1 || shards > 1024 || (shards > 1 && !policy.split)) return false;
  }
  const whole = split === undefined ? name : `${split.stem}.gguf`;
  if (modelType === 'qwen3_5' && isBonsaiPq2GgufName(whole)) return true;
  return policy.variants === 'all' || isQwen35XlGgufName(whole);
}

/**
 * The family discovery lists the GGUF file `name` under when no `config.json` sits beside it and its header states
 * `architecture`: the family of that architecture, when its GGUF discovery needs no asset files and admits the name.
 * `undefined` when discovery skips such a file.
 */
export function bareGgufFamily(name: string, architecture: string): ModelType | undefined {
  const modelType = ggufArchitectureModelType(architecture);
  if (modelType === undefined || familyDataFor(modelType)?.ggufDiscovery?.requiresAssets) return undefined;
  return supportedGgufName(name, modelType) ? modelType : undefined;
}

async function hasRegularFile(path: string): Promise<boolean> {
  try {
    return (await stat(path)).isFile();
  } catch {
    return false;
  }
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
  /** A link named like weights (`.gguf`, `.safetensors`) whose target cannot be reached: a disk that is away. */
  hasDeadWeightLink: boolean;
}

/**
 * Resolve a `readdir(withFileTypes)` entry to the kind of object it POINTS AT.
 *
 * `Dirent.isFile()` / `isDirectory()` reflect the directory entry itself, so
 * both return `false` for a symlink — symlinked models and GGUFs were
 * silently invisible to discovery. For links we `stat()` (follows the
 * target) instead of `lstat()`: discovery must classify the object the
 * loader will actually open. A dangling or unreadable link resolves to
 * `'other'` (skipped, same as sockets/fifos — a broken link admits
 * nothing) and non-symlink entries keep their `Dirent` answer unchanged.
 */
export async function resolveDirentKind(dir: string, entry: Dirent): Promise<'file' | 'dir' | 'other'> {
  if (!entry.isSymbolicLink()) {
    if (entry.isFile()) return 'file';
    if (entry.isDirectory()) return 'dir';
    return 'other';
  }
  try {
    const resolved = await stat(join(dir, entry.name));
    if (resolved.isFile()) return 'file';
    if (resolved.isDirectory()) return 'dir';
    return 'other';
  } catch {
    return 'other';
  }
}

async function modelFileInventory(modelDir: string): Promise<ModelFileInventory> {
  const entries = await readdir(modelDir, { withFileTypes: true });
  const kinds = await Promise.all(entries.map((entry) => resolveDirentKind(modelDir, entry)));
  const files = entries.filter((_, index) => kinds[index] === 'file').map((entry) => entry.name);
  const isWeightName = (name: string): boolean => {
    const lower = name.toLowerCase();
    return lower.endsWith('.gguf') || lower.endsWith('.safetensors');
  };
  const deadWeightLinks = entries
    .filter((entry, index) => entry.isSymbolicLink() && kinds[index] === 'other' && isWeightName(entry.name))
    .map((entry) => entry.name);
  // A split set or a safetensors checkpoint with one part away cannot load: none of its live parts is a target.
  const deadSplitPart = (name: string): boolean => {
    const split = ggufSplitPart(name);
    return (
      split !== null &&
      deadWeightLinks.some((dead) => {
        const part = ggufSplitPart(dead);
        return part !== null && part.stem === split.stem && part.total === split.total;
      })
    );
  };
  const deadPrimarySafetensors = deadWeightLinks.some((name) => PRIMARY_SAFETENSORS.test(name));
  return {
    targetGgufs: files
      .filter((name) => name.toLowerCase().endsWith('.gguf') && !isGgufCompanionName(name) && !deadSplitPart(name))
      .sort(),
    hasGguf: files.some((name) => name.toLowerCase().endsWith('.gguf')),
    hasPrimarySafetensors: !deadPrimarySafetensors && files.some((name) => PRIMARY_SAFETENSORS.test(name)),
    hasDeadWeightLink: deadWeightLinks.length > 0,
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
 * This lets the agent's model picker and `--list-models` expose checkpoint capability
 * without loading weights. The first resident load remains authoritative and
 * reconciles this optimistic config-level advertisement via
 * `session.supportsImages()` (for example, when conversion stripped an
 * incompatible vision tower).
 *
 * `detectModelType` already parsed this file, so a read/parse failure here
 * (e.g. a racing rewrite) lands on the context fallback and text-only input
 * instead of dropping the model or guessing a positive capability.
 */
async function hasSiblingMmprojGguf(dir: string): Promise<boolean> {
  try {
    const entries = await readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      const lower = entry.name.toLowerCase();
      if (!lower.startsWith('mmproj-') || !lower.endsWith('.gguf')) continue;
      if ((await resolveDirentKind(dir, entry)) === 'file') return true;
    }
    return false;
  } catch {
    return false;
  }
}

async function readDiscoveryMetadata(
  modelPath: string,
  modelType: ModelType,
  fallbackContextWindow: number,
): Promise<DiscoveryMetadata> {
  try {
    const config = (await readModelConfig(modelPath)) as Record<string, unknown>;
    const root = positiveInteger(config.max_position_embeddings) ?? positiveInteger(config.model_max_length);
    const textConfig = config.text_config;
    const nested = nonEmptyRecord(textConfig)
      ? (positiveInteger(textConfig.max_position_embeddings) ?? positiveInteger(textConfig.model_max_length))
      : undefined;
    const supportsImages =
      familyDataFor(modelType)?.visionConfigKeys?.some((key) => nonEmptyRecord(config[key])) ?? false;
    const draftOnly = Array.isArray(config.architectures) && config.architectures.includes('DFlash2DraftModel');

    return {
      contextWindow: root ?? nested ?? fallbackContextWindow,
      supportsImages,
      draftOnly,
      config,
    };
  } catch {
    return { contextWindow: fallbackContextWindow, supportsImages: false, draftOnly: false, config: undefined };
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
  /** Server-only opt-in: decision models must stay out of chat clients. */
  includeDecisions?: boolean;
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
 * dense Qwen3.5/Qwen3.8 `Q<number>_K_XL.gguf` / `Q<number>_K_M.gguf` / `UD-IQ*.gguf` files. GGUF files may live directly
 * under `modelsDir`, one level inside a downloaded GGUF repository, or in one of its
 * variant folders (`<repo>/UD-Q4_K_XL/*.gguf`) when the repository root holds no
 * target GGUF and no primary safetensors. Each is
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

  /** A listed model before its id: ids are given after the scan, in scan order. */
  interface Found {
    model: Omit<LocalChatModel, 'name' | 'base' | 'variant'>;
    preferredName: string;
    scopeName: string;
    /** A split set's file name without `.gguf`: its id when every GGUF was named by its file. */
    splitFileId: string | undefined;
    naming: (id: string) => ModelVariantName;
  }
  const found: Found[] = [];

  const append = async (
    preferredName: string,
    path: string,
    metadataRoot: string,
    modelType: ModelType,
    scopeName: string,
    format: ModelFormat,
    splitFileId: string | undefined,
    naming: (config: unknown, id: string) => ModelVariantName,
    ggufMetadata?: Pick<GgufDiscoveryMetadata, 'chatTemplate' | 'hasEmbeddedGpt2Tokenizer'>,
  ): Promise<void> => {
    const decision = familyDataFor(modelType)?.kind === 'decision' && opts?.includeDecisions === true;
    if (NON_GENERATIVE_FAMILY_IDS.has(modelType) && !decision) return;

    // Fail-closed guards: dead-by-construction for chat families (the
    // family-data row type requires traits + a preset), live for any foreign
    // string that slips through detection.
    const decisionContextCap = modelType === 'clairic' ? 65536 : 16384;
    const preset = decision ? { sampling: {}, maxOutputTokens: 0 } : launchPresetFor(modelType);
    if (!preset) {
      if (debug) console.warn(`[mlx] skip ${path}: no launch preset for ${modelType}`);
      return;
    }
    const baseTraits = decision
      ? { reasoning: false, fallbackContextWindow: decisionContextCap }
      : familyTraitsFor(modelType);
    if (!baseTraits) {
      if (debug) console.warn(`[mlx] skip ${path}: no FAMILY_TRAITS entry for ${modelType}`);
      return;
    }

    // Qwen3.5/3.6 and Qwen3.8 share `qwen3_5` (and potentially `qwen3_5_moe`).
    // Mirror native GGUF preparation (`gguf_plan.rs` + `write_embedded_gpt2_tokenizer`):
    // native only materializes the GGUF's embedded `tokenizer.chat_template` when
    // `tokenizer.json` is absent and embedded GPT-2 tokenizer metadata is present
    // (overwriting any sidecar template). When `tokenizer.json` exists or embedded
    // reconstruction cannot run, native never extracts the GGUF template and only
    // reads directory sidecars.
    let traits = baseTraits;
    if (!decision && (modelType === 'qwen3_5' || modelType === 'qwen3_5_moe')) {
      const ggufMaterializesTemplate =
        format === 'gguf' &&
        ggufMetadata?.hasEmbeddedGpt2Tokenizer === true &&
        ggufMetadata.chatTemplate !== undefined &&
        !(await hasRegularFile(join(metadataRoot, 'tokenizer.json')));
      const chatTemplate = ggufMaterializesTemplate
        ? ggufMetadata.chatTemplate
        : await readDirectoryChatTemplate(metadataRoot);
      const resolvedEfforts = reasoningEffortsFor(modelType, { chatTemplate });
      if (resolvedEfforts !== baseTraits.reasoningEfforts) {
        traits = { ...baseTraits, reasoningEfforts: resolvedEfforts };
      }
    }

    const metadata = await readDiscoveryMetadata(metadataRoot, modelType, traits.fallbackContextWindow);
    if (metadata.draftOnly) {
      if (debug) console.warn(`[mlx] skip ${path}: companion draft checkpoint is not a chat model`);
      return;
    }

    // A Qwen3.5 or Gemma4 GGUF load takes its vision tower only from a sibling mmproj, and a Qwen3.5 MoE GGUF load
    // never takes one (native_gguf_preflight). Qwen3.5 ignores config.json; Gemma4 looks for the projector only when
    // config.json declares a vision tower (gemma4_native_preflight).
    let supportsImages: boolean;
    if (modelType === 'clef') supportsImages = false;
    else if (format === 'gguf' && modelType === 'qwen3_5_moe') supportsImages = false;
    else if (format === 'gguf' && modelType === 'qwen3_5') supportsImages = await hasSiblingMmprojGguf(metadataRoot);
    else if (format === 'gguf' && modelType === 'gemma4')
      supportsImages = metadata.supportsImages && (await hasSiblingMmprojGguf(metadataRoot));
    else
      supportsImages =
        metadata.supportsImages ||
        (format === 'gguf' && modelType === 'clairic' && (await hasSiblingMmprojGguf(metadataRoot)));

    // Automatic companions can be installed or removed after this startup
    // scan. The host resolves them on each load; draftModelPath is reserved
    // for caller-supplied paths that the loader must treat as authoritative.
    found.push({
      model: {
        path,
        modelType,
        preset,
        traits,
        contextWindow: decision ? Math.min(decisionContextCap, metadata.contextWindow) : metadata.contextWindow,
        supportsImages,
        format,
      },
      preferredName,
      scopeName,
      splitFileId,
      naming: (id) => naming(metadata.config, id),
    });
  };

  /** `repoFolder` is undefined for a loose file in the models root; `variantFolder` for a file in the repo folder. */
  const appendGguf = async (
    name: string,
    metadataRoot: string,
    scopeName: string,
    repoFolder: string | undefined,
    variantFolder: string | undefined,
  ): Promise<void> => {
    const path = join(metadataRoot, name);
    try {
      // `resolveDirentKind` admits a symlinked `.gguf`, but native preflight
      // canonicalizes it before preparing the separate small asset dir, so
      // every check that looks BESIDE the file — family detection's sibling
      // config.json, the `requiresAssets` gate, discovery metadata — must run
      // against the resolved target's directory, not the link's. `realpath`
      // also resolves a symlinked PARENT directory, covering a linked-in
      // model dir wholesale. The recorded path stays the lexical link path:
      // loading it reaches the same canonical file either way.
      const canonical = await realpath(path);
      const assetsRoot = dirname(canonical);
      const modelType = await detectModelType(canonical);
      // Clairic GGUFs wrap a Qwen3.5 backbone (`general.architecture = "qwen35"`)
      // promoted by sibling `clair_head.{safetensors,config.json}` assets.
      // Validate filename and header against the Qwen3.5 backbone policy, then
      // register under the promoted `clairic` family.
      const ggufPolicyModelType: ModelType = modelType === 'clairic' ? 'qwen3_5' : modelType;
      const family = familyDataFor(ggufPolicyModelType);
      if (!supportedGgufName(name, ggufPolicyModelType)) return;
      // A sibling config must agree with the target header, never a projector.
      const ggufMetadata = await readGgufDiscoveryMetadata(canonical);
      if (!family?.ggufArchitectures?.includes(ggufMetadata.architecture)) return;
      if (
        family.ggufDiscovery?.requiresAssets &&
        !(await hasGgufAssets(assetsRoot, (error) => reportFailure(error, path)))
      ) {
        if (debug)
          console.warn(`[mlx] skip ${path}: native ${modelType} GGUF requires sibling config.json and tokenizer.json`);
        return;
      }
      const names = ggufNames(name, repoFolder, variantFolder);
      const splitFileId = names.split ? name.slice(0, -'.gguf'.length) : undefined;
      await append(
        names.id,
        path,
        assetsRoot,
        modelType,
        scopeName,
        'gguf',
        splitFileId,
        (_config, id) => ggufVariant(names.stem, repoFolder, variantFolder, id),
        ggufMetadata,
      );
    } catch (err) {
      reportFailure(err, path);
      if (debug) console.warn(`[mlx] skip ${path}: ${(err as Error).message}`);
    }
  };

  /**
   * Discover target GGUFs in the variant folders of `repoDir` (one level, no
   * dot-folders, no deeper). `claimed` when any variant holds a target file
   * name: `repoDir` is then a repository, not a model. `deadLink` when a
   * variant folder holds a weight link whose target cannot be reached.
   */
  const appendVariantGguf = async (
    repoDir: string,
    scopeName: string,
  ): Promise<{ claimed: boolean; deadLink: boolean }> => {
    const children = await readdir(repoDir, { withFileTypes: true });
    children.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    let claimed = false;
    let deadLink = false;
    for (const child of children) {
      if (child.name.startsWith('.') || (await resolveDirentKind(repoDir, child)) !== 'dir') continue;
      const variantDir = join(repoDir, child.name);
      try {
        const inventory = await modelFileInventory(variantDir);
        if (inventory.hasDeadWeightLink) deadLink = true;
        if (inventory.hasPrimarySafetensors) continue;
        for (const name of inventory.targetGgufs) {
          claimed = true;
          await appendGguf(name, variantDir, scopeName, scopeName, child.name);
        }
      } catch (err) {
        reportFailure(err, variantDir);
        if (debug) console.warn(`[mlx] skip ${variantDir}: ${(err as Error).message}`);
      }
    }
    return { claimed, deadLink };
  };

  for (const entry of entries) {
    // Dirent kind reflects the link itself, never the target — resolve so a
    // symlinked `.gguf` or model directory classifies by what it points at.
    const kind = await resolveDirentKind(modelsDir, entry);
    if (kind === 'file' && entry.name.toLowerCase().endsWith('.gguf') && !isGgufCompanionName(entry.name)) {
      await appendGguf(entry.name, modelsDir, basename(modelsDir), undefined, undefined);
      continue;
    }
    // Native Qwen4 GGUF loads publish tokenizer/config sidecars here, including
    // temporary directories during publication. They contain no model weights.
    if (kind !== 'dir' || entry.name.startsWith('.mlx-qwen4-assets-')) continue;
    const full = join(modelsDir, entry.name);
    try {
      const inventory = await modelFileInventory(full);
      // Prefer converted targets over retained source files or companion weights.
      // Each native GGUF variant is otherwise a separate entry. Header detection
      // also handles Qwen4 split directories with no config/tokenizer sidecars.
      if (!inventory.hasPrimarySafetensors) {
        if (inventory.targetGgufs.length > 0) {
          for (const name of inventory.targetGgufs) await appendGguf(name, full, entry.name, entry.name, undefined);
          continue;
        }
        // `mlx download` keeps repo paths, so a variant folder such as
        // `<repo>/UD-Q4_K_XL/*.gguf` sits one level down. A repo with such a
        // variant is never itself a model, even with a root config.json.
        // Dot-folders are never repos: `.staging/` holds the dashboard's
        // unpublished download jobs and publish backups.
        const variants = entry.name.startsWith('.')
          ? { claimed: false, deadLink: false }
          : await appendVariantGguf(full, entry.name);
        if (variants.claimed) continue;
        if (inventory.hasGguf) continue;
        // Weights on a disk that is away: the root config.json alone must not list the folder as a model.
        if (inventory.hasDeadWeightLink || variants.deadLink) continue;
      }
      const modelType = await detectModelType(full);
      await append(entry.name, full, full, modelType, entry.name, 'mlx', undefined, (config) =>
        folderVariant(entry.name, config),
      );
    } catch (err) {
      reportFailure(err, full);
      if (debug) console.warn(`[mlx] skip ${full}: ${(err as Error).message}`);
    }
  }

  // A saved id (a conversation, `mlx serve --model`, the agent's default) must keep opening the same model: every
  // model that is not a split set keeps the id it had when every GGUF was named by its file (split sets too, in scan
  // order), and only then does a split set take its shorter id, where it is free.
  const take = (used: Set<string>, preferredName: string, scopeName: string): string => {
    let name = preferredName;
    if (used.has(name)) {
      name = `${scopeName}-${preferredName}`;
      let suffix = 2;
      while (used.has(name)) name = `${scopeName}-${preferredName}-${suffix++}`;
    }
    used.add(name);
    return name;
  };
  const fileIds = new Set<string>();
  const ids = found.map((f) => take(fileIds, f.splitFileId ?? f.preferredName, f.scopeName));
  const usedNames = new Set(ids.filter((_id, index) => found[index]!.splitFileId === undefined));
  const out: LocalChatModel[] = found.map(({ model, preferredName, scopeName, splitFileId, naming }, index) => {
    const name = splitFileId === undefined ? ids[index]! : take(usedNames, preferredName, scopeName);
    const { format, ...rest } = model;
    return { name, ...rest, ...naming(name), format };
  });
  settleVariants(out);
  out.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  return out;
}

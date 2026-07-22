/**
 * Local model store for the dashboard: pure `fs` + JSON, never the native
 * addon (the dashboard is a viewer process that must start without Metal).
 *
 * Model metadata (family label, quantization, context window) is parsed from
 * each checkpoint's `config.json` with a small standalone parser — deliberately
 * duplicated from `packages/lm` to avoid pulling in `@mlx-node/core`, mirroring
 * the same trade-off already made in `packages/agent/src/provider/models.ts`.
 */

import { type Dirent, existsSync, lstatSync, readdirSync, readFileSync, rmSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';

export interface LocalModel {
  /** Checkpoint directory name under `modelsDir` (the model's local id). */
  name: string;
  /** Absolute path to the checkpoint directory. */
  path: string;
  /** Family label from `config.json` (`qwen3`, `qwen3_5`, `gemma4`, …). */
  modelType: string;
  /** Quantization label (`mxfp4`, `affine-4bit`, …), or `null` for full-precision. */
  quant: string | null;
  /** Trained context window from `max_position_embeddings`, or `null` if absent. */
  contextWindow: number | null;
  /** Total size on disk of the checkpoint directory, recursively. */
  sizeBytes: number;
  /** Number of files in the checkpoint directory, recursively. */
  fileCount: number;
}

/**
 * Raw config aliases → canonical family label. Mirrors the alias rows of
 * `MODEL_FAMILY_REGISTRY` in `packages/lm/src/models/model-loader.ts`; only the
 * label mapping is duplicated (no loaders, no native classes).
 */
const RAW_MODEL_TYPE_TO_LABEL: Record<string, string> = {
  gemma4: 'gemma4',
  gemma4_text: 'gemma4',
  gemma4_unified: 'gemma4',
  qwen3: 'qwen3',
  qwen3_5: 'qwen3_5',
  qwen3_5_moe: 'qwen3_5_moe',
  lfm2: 'lfm2',
  lfm2_moe: 'lfm2_moe',
  harrier: 'harrier',
  internvl_chat: 'internvl_chat',
  'qianfan-ocr': 'qianfan-ocr',
};

/**
 * Filename of the atomic-publish completion marker the download runner writes as
 * the LAST step of a download (inside the staging dir, before it is renamed onto
 * the final path). Its presence — not bare directory existence — is what marks a
 * checkpoint "installed".
 */
export const DOWNLOAD_COMPLETE_MARKER = '.mlx-download-complete.json';

/** Contents of {@link DOWNLOAD_COMPLETE_MARKER}: the pinned snapshot a dir holds. */
export interface DownloadCompletion {
  /** HuggingFace repo the checkpoint came from. */
  repo: string;
  /** The exact commit sha the whole download was pinned to (one snapshot). */
  revision: string;
  /** Repo-relative paths of every file published into the final dir. */
  files: string[];
  /** ISO timestamp of the atomic publish. */
  completedAt: string;
}

/**
 * A file that carries a model's weights: safetensors, GGUF, or PaddlePaddle
 * params. The single source of truth for "is this a weight payload" — shared
 * with the download runner (`download.ts` imports it) so the publish-time
 * payload gate and the install check agree on the same extension set.
 */
export function isWeightFile(path: string): boolean {
  return path.endsWith('.safetensors') || path.endsWith('.gguf') || path.endsWith('.pdiparams');
}

/**
 * A checkpoint counts as installed only when its atomic-publish completion
 * marker is present, parses, lists BOTH a `config.json` and at least one weight
 * file, and every file it lists still exists on disk — never bare directory
 * existence (a half-written or aborted download with a partial `config.json` +
 * missing weights) nor a one-sided manifest (config-only or weights-only), both
 * of which `loadModel` would reject yet a laxer check would falsely report as
 * installed, hiding the retry.
 */
export function isModelInstalled(modelDir: string): boolean {
  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(join(modelDir, DOWNLOAD_COMPLETE_MARKER), 'utf-8')) as unknown;
  } catch {
    return false;
  }
  const marker = asObject(parsed);
  if (marker === undefined || !Array.isArray(marker.files)) return false;
  const files = marker.files;
  // A loadable checkpoint needs a config AND a weight; a marker missing either
  // describes a hollow/one-sided publish that must NOT read as installed.
  const hasConfig = files.some((file) => file === 'config.json');
  const hasWeight = files.some((file) => typeof file === 'string' && isWeightFile(file));
  if (!hasConfig || !hasWeight) return false;
  return files.every((file) => typeof file === 'string' && existsSync(join(modelDir, file)));
}

/**
 * Whether a directory was written by the dashboard downloader — i.e. it carries
 * a parseable completion marker of our shape. Distinct from {@link isModelInstalled}:
 * ownership does NOT require every listed file to still be present (a partial,
 * mid-upgrade owned dir is still ours). The downloader uses this to decide
 * whether it may overwrite a final dir: a dir WITHOUT our marker was placed by a
 * human or by `mlx download` and must never be destroyed.
 */
export function isDownloaderOwned(modelDir: string): boolean {
  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(join(modelDir, DOWNLOAD_COMPLETE_MARKER), 'utf-8')) as unknown;
  } catch {
    return false;
  }
  const marker = asObject(parsed);
  return marker !== undefined && Array.isArray(marker.files);
}

function asObject(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

/**
 * Whether `p` is a safe repo-relative path — one that, joined onto a base dir,
 * can only ever resolve INSIDE that base. Rejects anything a `join(base, p)`
 * could use to escape: an empty string; a backslash or NUL; a Windows drive
 * prefix (`C:`); an absolute (leading-slash) path; and any `/`-separated segment
 * that is empty (a leading/trailing/double slash), `.`, or `..`. This is the
 * shared component-rejection rule behind both {@link deleteLocalModel}'s stricter
 * single-child check and the download runner's gate on untrusted `listFiles`
 * paths before they become `join(stagingDir, path)` write targets.
 */
export function isSafeRelPath(p: string): boolean {
  if (p.length === 0) return false;
  if (p.includes('\\') || p.includes('\0')) return false;
  if (p.startsWith('/')) return false; // absolute (posix)
  if (/^[a-zA-Z]:/.test(p)) return false; // Windows drive prefix (e.g. C:)
  // Every path segment must be a real name — never empty (leading/trailing/double
  // slash) and never a `.`/`..` traversal component.
  for (const segment of p.split('/')) {
    if (segment === '' || segment === '.' || segment === '..') return false;
  }
  return true;
}

function collectArchitectures(config: Record<string, unknown>): string[] {
  const raw = config.architectures;
  if (Array.isArray(raw)) return raw.filter((entry): entry is string => typeof entry === 'string');
  if (typeof raw === 'string') return [raw];
  return [];
}

/**
 * Family label from `config.json`. Uses `model_type` (default `qwen3` when
 * missing, mirroring `packages/lm`'s nullish default) and refines by
 * `architectures` only where the native registry treats architecture as
 * authoritative — the gemma4 unified checkpoint, whose `model_type` may not
 * name the family.
 */
function detectModelTypeLabel(config: Record<string, unknown>): string {
  if (collectArchitectures(config).includes('Gemma4UnifiedForConditionalGeneration')) return 'gemma4';
  const raw = typeof config.model_type === 'string' ? config.model_type : undefined;
  if (raw !== undefined) return RAW_MODEL_TYPE_TO_LABEL[raw] ?? raw;
  return 'qwen3';
}

/**
 * Quantization label from `quantization` / `quantization_config`. Micro-scaling
 * and NVFP modes surface by name (`mxfp4`, `mxfp8`, `nvfp4`, `sym8`); affine
 * folds bits into the label (`affine-4bit`); a config with no quant block is
 * full-precision (`null`).
 */
function detectQuantLabel(config: Record<string, unknown>): string | null {
  const quant = asObject(config.quantization) ?? asObject(config.quantization_config);
  if (quant === undefined) return null;
  const mode =
    typeof quant.mode === 'string'
      ? quant.mode
      : typeof quant.quant_method === 'string'
        ? quant.quant_method
        : undefined;
  const bits = typeof quant.bits === 'number' && Number.isFinite(quant.bits) ? quant.bits : undefined;
  if (mode !== undefined && mode !== 'affine') return mode;
  if (bits !== undefined) return `affine-${bits}bit`;
  if (mode === 'affine') return 'affine';
  return null;
}

function positiveInteger(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.floor(value) : undefined;
}

/** Root `max_position_embeddings` first, then the `text_config` nesting, else `null`. */
function detectContextWindow(config: Record<string, unknown>): number | null {
  const root = positiveInteger(config.max_position_embeddings);
  if (root !== undefined) return root;
  const textConfig = asObject(config.text_config);
  if (textConfig !== undefined) {
    const nested = positiveInteger(textConfig.max_position_embeddings);
    if (nested !== undefined) return nested;
  }
  return null;
}

/** Recursive size + file count of a directory. Unreadable entries are skipped. */
function walkDirStats(dir: string): { sizeBytes: number; fileCount: number } {
  let sizeBytes = 0;
  let fileCount = 0;
  let entries: Dirent[];
  try {
    entries = readdirSync(dir, { withFileTypes: true });
  } catch {
    return { sizeBytes, fileCount };
  }
  for (const entry of entries) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      const nested = walkDirStats(full);
      sizeBytes += nested.sizeBytes;
      fileCount += nested.fileCount;
      continue;
    }
    try {
      // `statSync` follows symlinks so HF-cache-symlinked blobs count at
      // their real byte size rather than the ~100-byte link.
      const stat = statSync(full);
      if (stat.isDirectory()) {
        const nested = walkDirStats(full);
        sizeBytes += nested.sizeBytes;
        fileCount += nested.fileCount;
      } else {
        sizeBytes += stat.size;
        fileCount += 1;
      }
    } catch {
      // Broken symlink / unreadable file: skip.
    }
  }
  return { sizeBytes, fileCount };
}

/**
 * Scan `modelsDir` for checkpoint subdirectories. A subdirectory is a model
 * only when it carries a readable `config.json`; anything else is reported as
 * a warning and skipped (never guessed). A missing `modelsDir` yields empty
 * results with no warning.
 */
export function discoverLocalModels(modelsDir: string): { models: LocalModel[]; warnings: string[] } {
  const models: LocalModel[] = [];
  const warnings: string[] = [];

  let entries: Dirent[];
  try {
    entries = readdirSync(modelsDir, { withFileTypes: true });
  } catch {
    return { models, warnings };
  }

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    // Hidden dirs are runner scratch (`.staging/<slug>`), never checkpoints:
    // skip them silently so an in-flight download raises no spurious warning.
    if (entry.name.startsWith('.')) continue;
    const full = join(modelsDir, entry.name);
    const configPath = join(full, 'config.json');

    let config: Record<string, unknown>;
    try {
      const parsed = JSON.parse(readFileSync(configPath, 'utf-8')) as unknown;
      const object = asObject(parsed);
      if (object === undefined) {
        warnings.push(`${entry.name}: config.json root is not a JSON object; skipped`);
        continue;
      }
      config = object;
    } catch {
      warnings.push(`${entry.name}: no readable config.json; skipped`);
      continue;
    }

    const { sizeBytes, fileCount } = walkDirStats(full);
    models.push({
      name: entry.name,
      path: full,
      modelType: detectModelTypeLabel(config),
      quant: detectQuantLabel(config),
      contextWindow: detectContextWindow(config),
      sizeBytes,
      fileCount,
    });
  }

  models.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
  return { models, warnings };
}

/**
 * Delete a checkpoint directory by its exact name. `name` must be a single
 * direct child of `modelsDir`: any path separator, `.`/`..`, or NUL is rejected,
 * so a decoded route param like `link%2Fvictim` (→ `link/victim`) or `../../etc`
 * can never traverse out of the store. The resolved child is `lstat`-ed without
 * following the final component, and a symlinked child is refused outright —
 * otherwise `rmSync(..., { recursive })` would follow the link and recursively
 * delete whatever lives outside the store that it points at.
 */
export function deleteLocalModel(modelsDir: string, name: string): void {
  // A deletable model is a SINGLE direct child: it must be a safe relative path
  // (no `.`/`..`/backslash/NUL/drive prefix — shared with the download gate) AND
  // carry no separator, so a decoded route param like `link%2Fvictim` or
  // `../../etc` can never traverse out of the store.
  if (!isSafeRelPath(name) || name.includes('/')) {
    throw new Error(`Refusing to delete "${name}": not a direct child of the models directory`);
  }
  // A legitimate checkpoint dir never starts with `.` — discovery skips dotdirs.
  // Rejecting them keeps the runner's reserved scratch (`.staging`, and any leaked
  // `<slug>.backup-*` / `<slug>@<sha>...` under it) out of reach of a delete route.
  if (name.startsWith('.')) {
    throw new Error(`Refusing to delete "${name}": reserved dashboard directory`);
  }

  const root = resolve(modelsDir);
  const target = join(root, name);
  // Belt-and-braces: the resolved child must sit exactly one level under root.
  if (dirname(target) !== root) {
    throw new Error(`Refusing to delete "${name}": resolves outside the models directory`);
  }

  let stat;
  try {
    stat = lstatSync(target);
  } catch {
    throw new Error(`Model "${name}" not found`);
  }
  if (stat.isSymbolicLink()) {
    throw new Error(`Refusing to delete "${name}": target is a symlink, not a model directory`);
  }
  rmSync(target, { recursive: true, force: true });
}

/**
 * Default models directory, mirroring `resolveModelsDir` in
 * `packages/cli/src/config.ts`: `MLX_MODELS_DIR` env → `modelsDir` field of
 * `~/.mlx-node/config.json` → `~/.mlx-node/models`. Unlike the CLI helper this
 * never creates the directory — a viewer only reads.
 */
export function defaultModelsDir(): string {
  const envDir = process.env.MLX_MODELS_DIR;
  if (envDir !== undefined && envDir.length > 0) return resolve(envDir);

  const home = join(homedir(), '.mlx-node');
  const fromConfig = readModelsDirFromConfig(join(home, 'config.json'));
  if (fromConfig !== undefined) return resolve(fromConfig);

  return join(home, 'models');
}

function readModelsDirFromConfig(configPath: string): string | undefined {
  let raw: string;
  try {
    raw = readFileSync(configPath, 'utf-8');
  } catch {
    return undefined;
  }
  try {
    const parsed = JSON.parse(raw) as { modelsDir?: unknown };
    if (typeof parsed.modelsDir === 'string' && parsed.modelsDir.length > 0) return parsed.modelsDir;
  } catch {
    // Malformed config.json: fall through to the default.
  }
  return undefined;
}

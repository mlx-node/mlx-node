/**
 * Local model store for the dashboard: pure `fs` + JSON, never the native
 * addon (the dashboard is a viewer process that must start without Metal).
 *
 * Model metadata (family label, quantization, context window) is parsed from
 * each checkpoint's `config.json` with a small standalone parser — deliberately
 * duplicated from `packages/lm` to avoid pulling in `@mlx-node/core`, mirroring
 * the same trade-off already made in `packages/agent/src/provider/models.ts`.
 */

import { type Dirent, lstatSync, readdirSync, readFileSync, rmSync, statSync } from 'node:fs';
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

function asObject(value: unknown): Record<string, unknown> | undefined {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
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
  if (
    name.length === 0 ||
    name === '.' ||
    name === '..' ||
    name.includes('/') ||
    name.includes('\\') ||
    name.includes('\0')
  ) {
    throw new Error(`Refusing to delete "${name}": not a direct child of the models directory`);
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

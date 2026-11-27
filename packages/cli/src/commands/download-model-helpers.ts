import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';

import type { ListFileEntry } from '@huggingface/hub';
import { MODEL_FAMILY_DATA, type ModelFamilyData } from '@mlx-node/lm/family-data';
import { readGgufArchitecture } from '@mlx-node/lm/model-detection';
import { isGgufCompanionName, parseGgufFileName } from '@mlx-node/lm/model-discovery';
import { ttsComponents } from '@mlx-node/tts/catalog';

/** Attempts per file before a download gives up. */
const MAX_FETCH_ATTEMPTS = 4;

/** Base backoff; attempt N waits `BASE << (N - 1)` ms (1s, 2s, 4s). */
const RETRY_BASE_MS = 1_000;

/** Node errno values no retry can fix: the LOCAL filesystem said no. */
const PERMANENT_FS_CODES = new Set([
  'ENOSPC', // disk full
  'EDQUOT', // over quota
  'EACCES', // not permitted
  'EPERM',
  'EROFS', // read-only filesystem
  'EISDIR',
  'ENOTDIR',
  'ENAMETOOLONG',
  'EFBIG',
  'EXDEV',
]);

const PERMANENT_PROTOCOL_MESSAGES = ['Unsupported chunk version', 'Unsupported compression scheme'];

const PERMANENT_DOWNLOAD_MESSAGES = [
  'Repository config disappeared',
  'Composite models require an immutable revision',
  'Required model component',
];

/** The human-readable text of a failure, whatever shape it arrived in. */
function failureText(error: unknown): string {
  if (typeof error === 'string') return error;
  if (error instanceof Error) return error.message;
  return String(error);
}

const FLATTENED_STATUS = /^Api error with status (\d{3})\b/;

function isRetriableStatus(status: number): boolean {
  return status >= 500 || status === 429 || status === 408;
}

/** Whether a failed download step is worth repeating. */
export function isRetriableFetchError(error: unknown): boolean {
  if (typeof error !== 'string' && !(error instanceof Error)) return false;

  const text = failureText(error);
  if (PERMANENT_PROTOCOL_MESSAGES.some((m) => text.includes(m))) return false;
  if (PERMANENT_DOWNLOAD_MESSAGES.some((m) => text.includes(m))) return false;

  if (typeof error === 'string') {
    const status = FLATTENED_STATUS.exec(error);
    return status ? isRetriableStatus(Number(status[1])) : true;
  }

  const status = (error as { statusCode?: unknown }).statusCode;
  if (typeof status === 'number') return isRetriableStatus(status);

  const code = (error as NodeJS.ErrnoException).code;
  if (typeof code === 'string' && PERMANENT_FS_CODES.has(code)) return false;

  return true;
}

/** Run `attempt`, repeating it while the failure looks transient. */
export async function withRetries<T>(what: string, attempt: () => Promise<T>): Promise<T> {
  for (let n = 1; ; n++) {
    try {
      return await attempt();
    } catch (error) {
      if (n >= MAX_FETCH_ATTEMPTS || !isRetriableFetchError(error)) throw error;
      const waitMs = RETRY_BASE_MS << (n - 1);
      const status = (error as { statusCode?: unknown }).statusCode;
      const reason = typeof status === 'number' ? `HTTP ${status}` : failureText(error);
      console.warn(`    ${what} failed (${reason}); retry ${n}/${MAX_FETCH_ATTEMPTS - 1} in ${waitMs}ms`);
      await new Promise((resolve) => setTimeout(resolve, waitMs));
    }
  }
}

export const CORE_FILES = [
  'LICENSE',
  'LICENSE.md',
  'README.md',
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'special_tokens_map.json',
  'vocab.json',
  'merges.txt',
];

export const ASSET_SIDECAR_CANDIDATES = [
  'config.json',
  'tokenizer.json',
  'tokenizer_config.json',
  'chat_template.jinja',
  'generation_config.json',
  'preprocessor_config.json',
  'video_preprocessor_config.json',
  'processor_config.json',
];

/** Convert a simple glob pattern (with * wildcards) to a RegExp */
export function globToRegex(pattern: string): RegExp {
  const escaped = pattern.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*');
  return new RegExp(`^${escaped}$`, 'i');
}

/** Check if a filename matches any of the glob patterns */
export function matchesAnyGlob(filename: string, patterns: RegExp[]): boolean {
  return patterns.some((re) => re.test(filename));
}

/** Root-default selection plus nested files already claimed by a prior marker. */
export function isDefaultModelDownloadPath(path: string, previouslyTracked: ReadonlySet<string>): boolean {
  if (previouslyTracked.has(path)) return true;
  if (path.includes('/')) return false;
  return (
    CORE_FILES.includes(path) ||
    path.endsWith('.safetensors') ||
    path.endsWith('.json') ||
    path.endsWith('.pdiparams') ||
    path.endsWith('.yml') ||
    path.endsWith('.gguf') ||
    path.endsWith('.jinja')
  );
}

/** Pre-flight check: does `outputDir` already hold a complete model download? */
export function isModelAlreadyDownloaded(
  outputDir: string,
  files: string[],
  excludedPaths: ReadonlySet<string> = new Set(),
): boolean {
  const fileSet = new Set(files.filter((file) => !excludedPaths.has(file)));
  const hasConfig = fileSet.has('config.json');
  if (!hasConfig) return false;
  try {
    const config = JSON.parse(readFileSync(join(outputDir, 'config.json'), 'utf8')) as { model_type?: unknown };
    for (const component of ttsComponents(config)) {
      const path = join(outputDir, component);
      const prefix = `${component}/`;
      const excluded = new Set(
        [...excludedPaths].filter((file) => file.startsWith(prefix)).map((file) => file.slice(prefix.length)),
      );
      if (!isModelAlreadyDownloaded(path, readdirSync(path), excluded)) return false;
    }
  } catch {
    return false;
  }

  const hasSingleModel = fileSet.has('model.safetensors') || fileSet.has('weights.safetensors');
  const hasPaddleModel = fileSet.has('inference.pdiparams');
  if (hasSingleModel || hasPaddleModel) return true;

  const hasShardedModel = fileSet.has('model.safetensors.index.json');
  if (!hasShardedModel) return false;

  let parsed: unknown;
  try {
    const raw = readFileSync(join(outputDir, 'model.safetensors.index.json'), 'utf8');
    parsed = JSON.parse(raw);
  } catch {
    return false;
  }
  const weightMap =
    parsed && typeof parsed === 'object' && 'weight_map' in parsed
      ? (parsed as { weight_map?: unknown }).weight_map
      : undefined;
  if (!weightMap || typeof weightMap !== 'object') return false;

  const shardFilenames = new Set<string>();
  for (const value of Object.values(weightMap as Record<string, unknown>)) {
    if (typeof value === 'string') shardFilenames.add(value);
  }
  if (shardFilenames.size === 0) return false;

  for (const shard of shardFilenames) {
    if (excludedPaths.has(shard) || !existsSync(join(outputDir, shard))) return false;
  }
  return true;
}

/** Pre-flight check: do any files in `outputDir` actually match the user's glob patterns? */
export function isGlobVariantPresent(files: string[], globPatterns: string[]): boolean {
  if (globPatterns.length === 0) return false;
  const globs = globPatterns.map(globToRegex);
  return files.some((f) => matchesAnyGlob(f, globs));
}

/** Pre-flight check: is the user's glob-filtered download complete? */
export function isGlobMatchedSetComplete(localFiles: string[], remoteFiles: string[], globPatterns: string[]): boolean {
  if (globPatterns.length === 0) return false;
  if (remoteFiles.length === 0) return false;
  const globs = globPatterns.map(globToRegex);
  const localSet = new Set(localFiles);
  let matched = 0;
  for (const remote of remoteFiles) {
    const base = remote.split('/').pop() ?? remote;
    if (!matchesAnyGlob(base, globs) && !matchesAnyGlob(remote, globs)) continue;
    matched++;
    if (!localSet.has(base)) return false;
  }
  if (matched === 0) return false;
  return true;
}

/** Pre-flight check: is a no-glob GGUF download complete? */
export function isGgufRepoComplete(localFiles: string[], remoteFiles: string[]): boolean {
  if (remoteFiles.length === 0) return false;
  const remoteGguf = remoteFiles.filter((f) => f.endsWith('.gguf'));
  if (remoteGguf.length === 0) return false;
  const localSet = new Set(localFiles);
  for (const remote of remoteGguf) {
    const base = remote.split('/').pop() ?? remote;
    if (!localSet.has(base)) return false;
  }
  return true;
}

/** Per-file check inside the download loop: is `destPath` already a complete copy of the remote `file`? */
export function isLocalCopyComplete(destPath: string, expectedSize: number): boolean {
  if (!existsSync(destPath)) return false;
  if (expectedSize <= 0) return true;
  try {
    return statSync(destPath).size === expectedSize;
  } catch {
    return false;
  }
}

/** Root-level sidecar entries found in an assetsRepo listing. */
export function pickAssetSidecars(allFiles: ListFileEntry[]): ListFileEntry[] {
  const byName = new Map(allFiles.filter((file) => !file.path.includes('/')).map((file) => [file.path, file]));
  return ASSET_SIDECAR_CANDIDATES.flatMap((name) => {
    const file = byName.get(name);
    return file === undefined ? [] : [file];
  });
}

/** Does `file` name a weight artifact a loader can open (companions/projectors do not count)? */
export function isLoadableWeightFile(file: string): boolean {
  return (
    file.endsWith('.safetensors') ||
    file.endsWith('.pdiparams') ||
    (file.endsWith('.gguf') && !isGgufCompanionName(basename(file)))
  );
}

export interface GgufLoadEntry {
  /** Repo-relative path a loader opens: the single file, or shard 00001 of a split set. */
  readonly path: string;
  /** Shard count declared by the file name; 1 for a single file. */
  readonly shards: number;
  /** Distinct shards of the set present in the selection; equals `shards` when the set is complete. */
  readonly present: number;
  /** Path without the `.gguf` and `-NNNNN-of-NNNNN` suffixes. */
  readonly stem: string;
}

/** Collapse every split set to one entry: loaders and converters open one file, never each shard. */
export function ggufLoadEntries(weightFiles: readonly string[]): GgufLoadEntry[] {
  const order: string[] = [];
  const sets = new Map<string, { stem: string; shards: number; ordinals: Map<number, string> }>();
  for (const path of weightFiles) {
    const name = parseGgufFileName(path);
    if (name === undefined) continue;
    const { stem, shards } = name;
    const key = `${stem.toLowerCase()}\0${shards}`;
    let set = sets.get(key);
    if (set === undefined) {
      set = { stem, shards, ordinals: new Map() };
      sets.set(key, set);
      order.push(key);
    }
    set.ordinals.set(name.ordinal, path);
  }
  return order.map((key) => {
    const { stem, shards, ordinals } = sets.get(key)!;
    const first = ordinals.get(1) ?? `${stem}-00001-of-${String(shards).padStart(5, '0')}.gguf`;
    const present = [...ordinals.keys()].filter((ordinal) => ordinal >= 1 && ordinal <= shards).length;
    return { path: first, shards, present, stem };
  });
}

/** The family whose native loader opens this GGUF directly, if any. */
function directGgufFamily(architecture: string | undefined, shards: number): ModelFamilyData | undefined {
  if (architecture === undefined) return undefined;
  const rows: readonly ModelFamilyData[] = MODEL_FAMILY_DATA;
  const family = rows.find((row) => row.ggufArchitectures?.includes(architecture));
  if (family === undefined || (shards > 1 && !family.ggufDiscovery?.split)) return undefined;
  return family;
}

/**
 * Post-download guidance for a GGUF selection. Families listed with
 * `ggufArchitectures` load the file directly. `mlx convert` reads one file, so
 * converting a split set shard by shard yields a model with one shard's tensors.
 */
export async function ggufNextSteps(outputDir: string, weightFiles: readonly string[]): Promise<string[]> {
  const entries = ggufLoadEntries(weightFiles);
  const mlxOutput = (entry: GgufLoadEntry): string =>
    entries.length === 1 ? `${outputDir}-mlx` : `${outputDir}-mlx-${entry.stem.replaceAll('/', '-').toLowerCase()}`;
  const incomplete: string[] = [];
  const direct: string[] = [];
  const merge: string[] = [];
  const convert: string[] = [];
  for (const entry of entries) {
    const fullPath = join(outputDir, entry.path);
    if (entry.present < entry.shards) {
      incomplete.push(`  ${join(outputDir, entry.stem)}: ${entry.present} of ${entry.shards} shards selected`);
      continue;
    }
    const architecture = await readGgufArchitecture(fullPath).catch(() => undefined);
    const family = directGgufFamily(architecture, entry.shards);
    if (family !== undefined) {
      direct.push(
        `  ${fullPath}${entry.shards > 1 ? ` (reads the other ${entry.shards - 1} shard(s) beside it)` : ''}`,
      );
      const folder = dirname(fullPath);
      if (
        family.ggufDiscovery?.requiresAssets &&
        !(existsSync(join(folder, 'config.json')) && existsSync(join(folder, 'tokenizer.json')))
      ) {
        direct.push(
          folder === resolve(outputDir)
            ? `    needs config.json and tokenizer.json in ${folder}: run again with --assets-repo <base-model repo>`
            : `    needs config.json and tokenizer.json in ${folder}: copy them there from the base-model repo`,
        );
      }
    } else if (entry.shards > 1) {
      const merged = join(outputDir, `${entry.stem}.gguf`);
      merge.push(`  llama-gguf-split --merge ${fullPath} ${merged}`);
      merge.push(`  mlx convert -i ${merged} -o ${mlxOutput(entry)}`);
    } else {
      convert.push(`  mlx convert -i ${fullPath} -o ${mlxOutput(entry)}`);
    }
  }
  const lines: string[] = [];
  if (incomplete.length > 0)
    lines.push('Incomplete split GGUF (select every shard to load or convert it):', ...incomplete);
  if (direct.length > 0) lines.push('Load directly (no conversion needed):', ...direct);
  if (merge.length > 0) lines.push('Split GGUF with no split-aware loader: merge the shards, then convert:', ...merge);
  if (convert.length > 0) lines.push('To convert GGUF to MLX SafeTensors format:', ...convert);
  return lines;
}

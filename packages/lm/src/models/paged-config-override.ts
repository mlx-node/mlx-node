/**
 * Process-local model-directory overrides for forcing block-paged KV caches.
 *
 * Native model loaders accept a directory path and read paging policy from
 * `config.json`. This manager creates an isolated temporary clone containing a
 * patched config plus symlinks to the source files, so callers can opt models
 * into paging without mutating downloaded checkpoints. Loader-known Qwen3.5
 * MTP sidecars are preserved explicitly; unrelated directories (including a
 * Gemma4 `draft/`) remain hidden unless a caller explicitly preserves the
 * Gemma draft for flat speculative decoding.
 */

import { mkdir, mkdtemp, readFile, readdir, rm, stat, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';

/** Every chat-capable family currently discovered by `@mlx-node/agent`. */
export const AGENT_PAGED_MODEL_TYPES = ['qwen3', 'qwen3_5', 'qwen3_5_moe', 'gemma4', 'lfm2', 'lfm2_moe'] as const;

/** Families historically forced paged by `mlx launch claude`. */
export const QWEN35_PAGED_MODEL_TYPES = ['qwen3_5', 'qwen3_5_moe'] as const;

const QWEN35_CACHE_FLOOR_MODEL_TYPES = new Set<string>(QWEN35_PAGED_MODEL_TYPES);
const DEFAULT_QWEN35_PAGED_CACHE_MB = 16_384;
const QWEN35_MTP_DRAFTER_DIR = 'mtp-drafter';
const QWEN35_DENSE_NESTED_MTP_SIDECAR = 'mtp/weights.safetensors';

export interface PagedConfigOverrideManagerOptions {
  /** Model types to force onto the paged path. Defaults to all agent chat families. */
  modelTypes?: readonly string[];
  /** Temporary-directory prefix. Primarily useful for diagnostics/tests. */
  tempDirPrefix?: string;
  /**
   * Pass Gemma4 checkpoints with an embedded `draft/` through unchanged.
   * This keeps native draft auto-discovery, which currently requires flat KV
   * caches. Defaults to false so a paged override remains a paged contract.
   */
  preserveEmbeddedGemmaDraft?: boolean;
}

/**
 * Owns one isolated set of temporary paged-config overrides.
 *
 * A manager is intentionally single-lifecycle: repeated resolution of one
 * source returns the same override, and `cleanup()` permanently disposes the
 * manager. Separate managers never share directories, so one launch cannot
 * remove another launch's live override.
 */
export class PagedConfigOverrideManager {
  private readonly modelTypes: ReadonlySet<string>;
  private readonly tempDirPrefix: string;
  private readonly preserveEmbeddedGemmaDraft: boolean;
  private readonly overrides = new Map<string, Promise<string>>();
  private readonly activeResolves = new Set<Promise<string>>();
  private rootPromise: Promise<string> | undefined;
  private cleanupPromise: Promise<void> | undefined;
  private disposed = false;

  constructor(options: PagedConfigOverrideManagerOptions = {}) {
    this.modelTypes = new Set(options.modelTypes ?? AGENT_PAGED_MODEL_TYPES);
    this.tempDirPrefix = options.tempDirPrefix ?? 'mlx-paged-overrides-';
    this.preserveEmbeddedGemmaDraft = options.preserveEmbeddedGemmaDraft ?? false;
  }

  /**
   * Resolve `modelPath` to a paged-aware clone when its model type is managed.
   * A caller-supplied canonical family takes precedence over the raw config
   * type (for example, `gemma4` for a `gemma4_unified` checkpoint).
   * Unmanaged, unreadable, or malformed checkpoints pass through unchanged.
   *
   * `persistPagedCache` is a tri-state cold-tier directive. `undefined` leaves
   * the field untouched (families with no cold-tier opt-in). A boolean is
   * AUTHORITATIVE: it writes `persist_paged_cache: <value>` into the cloned
   * config, overriding whatever the source config.json carries (either alias),
   * and forces a clone whenever the source's value disagrees so the directive
   * actually reaches the loader. Callers gate the boolean to the families whose
   * paged cold restore is sound — the allowlist in
   * `packages/agent/src/cold-tier.ts`, mirrored by `COLD_RESTORE_FAMILIES` in
   * `crates/mlx-core/src/cold_tier.rs`. It is a SET, not one family, and it is
   * deliberately not repeated here: `MlxModelHost` calls this for every
   * allowlisted family, and a second copy of the list would drift.
   */
  async resolve(modelPath: string, canonicalModelType?: string, persistPagedCache?: boolean): Promise<string> {
    if (this.disposed) {
      throw new Error('PagedConfigOverrideManager: resolve() called after cleanup()');
    }

    const operation = this.resolveInternal(modelPath, canonicalModelType, persistPagedCache);
    this.activeResolves.add(operation);
    try {
      return await operation;
    } finally {
      this.activeResolves.delete(operation);
    }
  }

  private async resolveInternal(
    modelPath: string,
    canonicalModelType?: string,
    persistPagedCache?: boolean,
  ): Promise<string> {
    const sourcePath = isAbsolute(modelPath) ? modelPath : resolve(modelPath);
    let config: Record<string, unknown>;
    try {
      config = JSON.parse(await readFile(join(sourcePath, 'config.json'), 'utf-8')) as Record<string, unknown>;
    } catch {
      return modelPath;
    }

    const rawModelType = typeof config.model_type === 'string' ? config.model_type : null;
    const modelType = canonicalModelType ?? rawModelType;
    if (modelType === null || !this.modelTypes.has(modelType)) {
      return modelPath;
    }

    // A boolean persist directive is authoritative: force a clone whenever the
    // source config's EFFECTIVE value disagrees, so the directive actually reaches
    // the loader (both `--no-persist-cache` off AND default-on). Mirror the native
    // parser's snake-first precedence — `persist_paged_cache` wins when present,
    // else the camelCase alias — so a config with snake=false AND camel=true does
    // not read as `true` here (an OR) while native reads snake=false, silently
    // dropping an authoritative `true` request.
    const sourcePersist =
      typeof config.persist_paged_cache === 'boolean' ? config.persist_paged_cache : config.persistPagedCache === true;
    const persistOverrideNeeded = persistPagedCache !== undefined && persistPagedCache !== sourcePersist;

    // Gemma4's DSpark / assistant speculative executor currently owns flat KV
    // caches. Preserve native draft discovery only for explicit callers; the
    // default paged clone intentionally omits subdirectories, hiding `draft/`
    // while keeping the downloaded checkpoint unchanged.
    const hasEmbeddedGemmaDraft = modelType === 'gemma4' && (await isDirectory(join(sourcePath, 'draft')));
    if (this.preserveEmbeddedGemmaDraft && hasEmbeddedGemmaDraft) {
      return modelPath;
    }

    const cacheFloorMb = QWEN35_CACHE_FLOOR_MODEL_TYPES.has(modelType) ? resolveQwen35CacheFloorMb() : undefined;
    const pagedEnabled = config.use_block_paged_cache === true;
    const configuredMemoryMb = positiveNumber(config.paged_cache_memory_mb);
    const memorySatisfied = cacheFloorMb === undefined || (configuredMemoryMb ?? 0) >= cacheFloorMb;
    // Even an already-paged Gemma config must be cloned when `draft/` exists:
    // returning the source would expose the draft to native auto-discovery and
    // trigger the flat-speculation/paged-cache conflict. An authoritative
    // persist directive that disagrees with the source likewise blocks the
    // pass-through so the resolved value reaches the loader.
    if (pagedEnabled && memorySatisfied && !hasEmbeddedGemmaDraft && !persistOverrideNeeded) {
      return modelPath;
    }

    // Memoize per (source, resolved family, persist directive): the same
    // checkpoint resolved with a different persist tri-state must yield a distinct
    // clone, not the first one cached under the bare path. `cleanup()` iterates
    // every value, so multiple entries per path are all still disposed.
    const persistKey = persistPagedCache === undefined ? 'u' : persistPagedCache ? 't' : 'f';
    const cacheKey = `${sourcePath}\0${modelType}\0${persistKey}`;
    const existing = this.overrides.get(cacheKey);
    if (existing !== undefined) return existing;

    const pending = this.createOverride(sourcePath, config, cacheFloorMb, modelType, persistPagedCache);
    this.overrides.set(cacheKey, pending);
    try {
      return await pending;
    } catch (error) {
      this.overrides.delete(cacheKey);
      throw error;
    }
  }

  /** Remove this manager's temporary root without affecting other managers. */
  cleanup(): Promise<void> {
    if (this.cleanupPromise !== undefined) return this.cleanupPromise;
    this.disposed = true;
    this.cleanupPromise = this.performCleanup();
    return this.cleanupPromise;
  }

  private async performCleanup(): Promise<void> {
    await Promise.allSettled(this.activeResolves);
    this.activeResolves.clear();
    await Promise.allSettled(this.overrides.values());
    this.overrides.clear();
    if (this.rootPromise === undefined) return;

    const root = await this.rootPromise.catch(() => undefined);
    if (root !== undefined) {
      await rm(root, { recursive: true, force: true }).catch(() => undefined);
    }
  }

  private async createOverride(
    sourcePath: string,
    sourceConfig: Record<string, unknown>,
    cacheFloorMb: number | undefined,
    modelType: string,
    persistPagedCache: boolean | undefined,
  ): Promise<string> {
    const root = await this.getRoot();
    const overrideDir = await mkdtemp(join(root, 'model-'));

    const config: Record<string, unknown> = {
      ...sourceConfig,
      use_block_paged_cache: true,
    };
    if (persistPagedCache !== undefined) {
      // Authoritative: the loader reads snake_case, so that spelling is the one
      // that decides persistence. Reconcile a stray camelCase alias spread from
      // the source config so it can never contradict the authoritative value.
      config.persist_paged_cache = persistPagedCache;
      if ('persistPagedCache' in config) config.persistPagedCache = persistPagedCache;
    }
    if (cacheFloorMb !== undefined) {
      config.paged_cache_memory_mb = Math.max(positiveNumber(sourceConfig.paged_cache_memory_mb) ?? 0, cacheFloorMb);
    }
    await writeFile(join(overrideDir, 'config.json'), JSON.stringify(config, null, 2), 'utf-8');

    const sourceEntries = await readdir(sourcePath);
    for (const name of sourceEntries) {
      if (name === 'config.json') continue;
      const source = join(sourcePath, name);
      const destination = join(overrideDir, name);

      let isFile: boolean;
      try {
        isFile = (await stat(source)).isFile();
      } catch {
        continue;
      }
      if (!isFile) continue;

      try {
        await symlink(source, destination);
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
      }
    }

    if (QWEN35_CACHE_FLOOR_MODEL_TYPES.has(modelType)) {
      await preserveQwen35MtpSidecars(sourcePath, overrideDir, sourceConfig, modelType);
    }

    return overrideDir;
  }

  private getRoot(): Promise<string> {
    this.rootPromise ??= mkdtemp(join(tmpdir(), this.tempDirPrefix));
    return this.rootPromise;
  }
}

/**
 * Keep only the nested paths that the native Qwen3.5 loaders inspect.
 *
 * Both dense and MoE loaders accept an mlx-vlm `mtp-drafter/` directory.
 * Dense additionally accepts a config-declared relative sidecar and the
 * conventional `mtp/weights.safetensors` fallback. Top-level sidecar files
 * are already handled by the normal source-file loop above.
 */
async function preserveQwen35MtpSidecars(
  sourcePath: string,
  overrideDir: string,
  sourceConfig: Record<string, unknown>,
  modelType: string,
): Promise<void> {
  await symlinkDirectoryIfPresent(join(sourcePath, QWEN35_MTP_DRAFTER_DIR), join(overrideDir, QWEN35_MTP_DRAFTER_DIR));

  // The MoE loader supports the split drafter directory, but deliberately has
  // no `mtp.safetensors`-style sidecar discovery path.
  if (modelType !== 'qwen3_5') return;

  const relativeSidecars = new Set<string>([QWEN35_DENSE_NESTED_MTP_SIDECAR]);
  const configuredSidecar = configuredQwen35MtpFile(sourceConfig);
  if (configuredSidecar !== undefined && !isAbsolute(configuredSidecar)) {
    relativeSidecars.add(configuredSidecar);
  }

  for (const sidecar of relativeSidecars) {
    await symlinkContainedFileIfPresent(sourcePath, overrideDir, sidecar);
  }
}

function configuredQwen35MtpFile(config: Record<string, unknown>): string | undefined {
  const extra = config.mlx_lm_extra_tensors;
  if (extra === null || typeof extra !== 'object' || Array.isArray(extra)) return undefined;
  const mtpFile = (extra as Record<string, unknown>).mtp_file;
  return typeof mtpFile === 'string' && mtpFile.trim() !== '' ? mtpFile : undefined;
}

async function symlinkContainedFileIfPresent(
  sourceRoot: string,
  destinationRoot: string,
  relativePath: string,
): Promise<void> {
  const source = resolve(sourceRoot, relativePath);
  const normalizedRelative = relative(sourceRoot, source);
  if (
    normalizedRelative === '' ||
    normalizedRelative === '..' ||
    normalizedRelative.startsWith(`..${sep}`) ||
    isAbsolute(normalizedRelative)
  ) {
    return;
  }

  try {
    if (!(await stat(source)).isFile()) return;
  } catch {
    return;
  }

  const destination = join(destinationRoot, normalizedRelative);
  await mkdir(dirname(destination), { recursive: true });
  await symlinkIfMissing(source, destination);
}

async function symlinkDirectoryIfPresent(source: string, destination: string): Promise<void> {
  if (!(await isDirectory(source))) return;
  await symlinkIfMissing(source, destination);
}

async function symlinkIfMissing(source: string, destination: string): Promise<void> {
  try {
    await symlink(source, destination);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
  }
}

async function isDirectory(path: string): Promise<boolean> {
  try {
    return (await stat(path)).isDirectory();
  } catch {
    return false;
  }
}

function positiveNumber(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : undefined;
}

function resolveQwen35CacheFloorMb(): number {
  const raw = process.env.MLX_PAGED_CACHE_MEMORY_MB;
  if (raw == null || raw === '') return DEFAULT_QWEN35_PAGED_CACHE_MB;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_QWEN35_PAGED_CACHE_MB;
}

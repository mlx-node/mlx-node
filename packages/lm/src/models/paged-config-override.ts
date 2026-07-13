/**
 * Process-local model-directory overrides for forcing block-paged KV caches.
 *
 * Native model loaders accept a directory path and read paging policy from
 * `config.json`. This manager creates an isolated temporary clone containing a
 * patched config plus symlinks to the source files, so callers can opt models
 * into paging without mutating downloaded checkpoints.
 */

import { mkdir, mkdtemp, readFile, readdir, rm, stat, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { isAbsolute, join, resolve } from 'node:path';

/** Every chat-capable family currently discovered by `@mlx-node/agent`. */
export const AGENT_PAGED_MODEL_TYPES = ['qwen3', 'qwen3_5', 'qwen3_5_moe', 'gemma4', 'lfm2', 'lfm2_moe'] as const;

/** Families historically forced paged by `mlx launch claude`. */
export const QWEN35_PAGED_MODEL_TYPES = ['qwen3_5', 'qwen3_5_moe'] as const;

const QWEN35_CACHE_FLOOR_MODEL_TYPES = new Set<string>(QWEN35_PAGED_MODEL_TYPES);
const DEFAULT_QWEN35_PAGED_CACHE_MB = 16_384;

export interface PagedConfigOverrideManagerOptions {
  /** Model types to force onto the paged path. Defaults to all agent chat families. */
  modelTypes?: readonly string[];
  /** Temporary-directory prefix. Primarily useful for diagnostics/tests. */
  tempDirPrefix?: string;
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
  private readonly overrides = new Map<string, Promise<string>>();
  private rootPromise: Promise<string> | undefined;
  private disposed = false;

  constructor(options: PagedConfigOverrideManagerOptions = {}) {
    this.modelTypes = new Set(options.modelTypes ?? AGENT_PAGED_MODEL_TYPES);
    this.tempDirPrefix = options.tempDirPrefix ?? 'mlx-paged-overrides-';
  }

  /**
   * Resolve `modelPath` to a paged-aware clone when its model type is managed.
   * A caller-supplied canonical family takes precedence over the raw config
   * type (for example, `gemma4` for a `gemma4_unified` checkpoint).
   * Unmanaged, unreadable, or malformed checkpoints pass through unchanged.
   */
  async resolve(modelPath: string, canonicalModelType?: string): Promise<string> {
    if (this.disposed) {
      throw new Error('PagedConfigOverrideManager: resolve() called after cleanup()');
    }

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

    const cacheFloorMb = QWEN35_CACHE_FLOOR_MODEL_TYPES.has(modelType) ? resolveQwen35CacheFloorMb() : undefined;
    const pagedEnabled = config.use_block_paged_cache === true;
    const configuredMemoryMb = positiveNumber(config.paged_cache_memory_mb);
    const memorySatisfied = cacheFloorMb === undefined || (configuredMemoryMb ?? 0) >= cacheFloorMb;
    if (pagedEnabled && memorySatisfied) {
      return modelPath;
    }

    const existing = this.overrides.get(sourcePath);
    if (existing !== undefined) return existing;

    const pending = this.createOverride(sourcePath, config, cacheFloorMb);
    this.overrides.set(sourcePath, pending);
    try {
      return await pending;
    } catch (error) {
      this.overrides.delete(sourcePath);
      throw error;
    }
  }

  /** Remove this manager's temporary root without affecting other managers. */
  async cleanup(): Promise<void> {
    if (this.disposed) return;
    this.disposed = true;

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
  ): Promise<string> {
    const root = await this.getRoot();
    const overrideDir = join(root, hashPath(sourcePath));
    await mkdir(overrideDir, { recursive: true });

    const config: Record<string, unknown> = {
      ...sourceConfig,
      use_block_paged_cache: true,
    };
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

    return overrideDir;
  }

  private getRoot(): Promise<string> {
    this.rootPromise ??= mkdtemp(join(tmpdir(), this.tempDirPrefix));
    return this.rootPromise;
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

function hashPath(path: string): string {
  let hash = 0x811c9dc5;
  for (let index = 0; index < path.length; index++) {
    hash ^= path.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

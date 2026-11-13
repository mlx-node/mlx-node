/**
 * `--max-cache-disk SIZE` and `--cache-dir DIRECTORY` of the Qwen3.5 SSD state tier:
 * Splash `parse_max_cache_disk`, `parse_cache_dir` and the cache checks of
 * `check_serve_arguments`.
 */

import { homedir } from 'node:os';
import { join, resolve } from 'node:path';

import { parseMaxMemory } from './max-memory.js';

export const MAX_CACHE_DISK_ERROR = '--max-cache-disk must be auto, 0 to disable, or a size such as 16G';

export const CACHE_DIR_ERROR = '--cache-dir must name a directory';

/**
 * The native load option's text: `'auto'`, `'0'` (no SSD cache), or a
 * decimal byte count. Throws {@link MAX_CACHE_DISK_ERROR} on anything else.
 */
export function parseMaxCacheDisk(value: string): string {
  if (value.trim() === '0') return '0';
  let bytes: number | undefined;
  try {
    bytes = parseMaxMemory(value);
  } catch {
    throw new Error(MAX_CACHE_DISK_ERROR);
  }
  return bytes === undefined ? 'auto' : String(bytes);
}

/** An absolute directory, `~` expanded (Python `expanduser().absolute()`). */
export function parseCacheDir(value: string): string {
  if (value.trim().length === 0) throw new Error(CACHE_DIR_ERROR);
  if (value === '~') return homedir();
  if (value.startsWith('~/')) return join(homedir(), value.slice(2));
  return resolve(value);
}

/** The SSD cache flags of one `mlx serve` as given, with the resolved paged KV disk cache switch. */
export interface CacheDiskArgs {
  cacheDisk: string;
  /** The resolved `persistPagedCache` (`persist-cache.ts`). */
  persistPagedCache: boolean;
  /** Whether `--no-persist-cache` (rather than `MLX_PERSIST_PAGED_CACHE`) turned it off. */
  offByFlag?: boolean;
  cacheDir?: string;
}

/** The SSD cache options a qwen3_5 / qwen3_5_moe load receives. */
export interface ResolvedCacheDisk {
  cacheDisk: string;
  cacheDir?: string;
}

/**
 * Splash `check_serve_arguments` over the shared switch: `--cache-dir` needs a
 * persistent tier. A persistent switch with `--max-cache-disk 0` is no error:
 * the switch also serves the other families, and the Qwen3.5 tier stays off.
 */
export function resolveCacheDisk(args: CacheDiskArgs): ResolvedCacheDisk {
  if (args.cacheDir !== undefined && args.cacheDisk === '0') {
    throw new Error('--cache-dir needs the SSD cache --max-cache-disk 0 turns off');
  }
  if (args.cacheDir !== undefined && !args.persistPagedCache) {
    throw new Error(
      args.offByFlag === true
        ? '--cache-dir needs the persistent cache --no-persist-cache turns off'
        : '--cache-dir needs the persistent cache MLX_PERSIST_PAGED_CACHE turns off',
    );
  }
  return {
    cacheDisk: args.cacheDisk,
    ...(args.cacheDir !== undefined ? { cacheDir: args.cacheDir } : {}),
  };
}

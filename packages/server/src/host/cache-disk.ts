/**
 * `--max-cache-disk SIZE`, `--[no-]persistent-cache` and `--cache-dir DIRECTORY`:
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

/** The SSD cache flags of one `mlx serve` as given; `persistentCache` unset when neither form was passed. */
export interface CacheDiskArgs {
  cacheDisk: string;
  persistentCache?: boolean;
  cacheDir?: string;
}

/** The SSD cache options a load receives. */
export interface ResolvedCacheDisk {
  cacheDisk: string;
  persistentCache: boolean;
  cacheDir?: string;
}

/**
 * The cache kept across restarts unless `--no-persistent-cache` or
 * `--max-cache-disk 0` turns it off. Splash defaults to a temporary cache;
 * mlx-node keeps it because restart persistence is the point of the tier.
 */
export function defaultPersistentCache(cacheDisk: string | undefined): boolean {
  return cacheDisk !== '0';
}

/** Splash `check_serve_arguments` over the persistent default. */
export function resolveCacheDisk(args: CacheDiskArgs): ResolvedCacheDisk {
  if (args.persistentCache === true && args.cacheDisk === '0') {
    throw new Error('--persistent-cache needs the SSD cache --max-cache-disk 0 turns off');
  }
  const persistentCache = args.persistentCache ?? defaultPersistentCache(args.cacheDisk);
  if (args.cacheDir !== undefined && !persistentCache) {
    throw new Error(
      args.persistentCache === false
        ? '--cache-dir needs the persistent cache --no-persistent-cache turns off'
        : '--cache-dir needs the SSD cache --max-cache-disk 0 turns off',
    );
  }
  return {
    cacheDisk: args.cacheDisk,
    persistentCache,
    ...(args.cacheDir !== undefined ? { cacheDir: args.cacheDir } : {}),
  };
}

/** Both `--persistent-cache` and `--no-persistent-cache` given. */
export const PERSISTENT_CACHE_CONFLICT_ERROR = '--persistent-cache and --no-persistent-cache cannot be used together';

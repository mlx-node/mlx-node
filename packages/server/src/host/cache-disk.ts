/**
 * `--max-cache-disk SIZE`, `--persistent-cache` and `--cache-dir DIRECTORY`:
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

/** The SSD cache options of one `mlx serve`. */
export interface CacheDiskArgs {
  cacheDisk: string;
  persistentCache: boolean;
  cacheDir?: string;
}

/** Splash `check_serve_arguments`: the cache flags that need one another. */
export function checkCacheDiskArgs(args: CacheDiskArgs): void {
  if (args.persistentCache && args.cacheDisk === '0') {
    throw new Error('--persistent-cache needs the SSD cache --max-cache-disk 0 turns off');
  }
  if (args.cacheDir !== undefined && !args.persistentCache) {
    throw new Error('--cache-dir needs --persistent-cache');
  }
}

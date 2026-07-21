/**
 * Pure-fs view and management of the SSD cold tier that Phase A's
 * `ColdCacheManager` (crates/mlx-paged-attn/src/cold_cache.rs) writes to. The
 * dashboard reads the block directory directly and never links the native
 * addon.
 *
 * The default root mirrors the manager's own resolution: with no override the
 * manager's `open_default` operates directly in `~/.mlx-node/cache/paged/v1`;
 * with `MLX_COLD_CACHE_DIR` set the tier lives in that directory's
 * `mlx-paged-v1` managed child (crates/mlx-core/src/cold_tier.rs).
 *
 * Only canonical `<64-hex-sha256>.safetensors` block files are ever counted or
 * removed. The hardened tier also leaves writer temp files
 * (`.{64-hex}.{pid}.{tick}.tmp`) and quarantined obstructions
 * (`.blocked.<name>.{pid}.{tick}`) beside the blocks; those, and any other
 * foreign file, are never touched.
 */

import { type Stats, lstatSync, readdirSync, statSync, statfsSync, unlinkSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

/** Canonical persisted block filename (`ColdCacheKey::to_hex` + suffix). */
const BLOCK_FILE = /^[0-9a-f]{64}\.safetensors$/;

const GIB = 1024 ** 3;
/** `MAX_DEFAULT_QUOTA` in cold_cache.rs. */
const MAX_DEFAULT_QUOTA = 100 * GIB;
const DAY_MS = 24 * 60 * 60 * 1000;

/** Child of `MLX_COLD_CACHE_DIR` the tier operates in (`MANAGED_SUBDIR`). */
const MANAGED_SUBDIR = 'mlx-paged-v1';

export interface ColdCacheDiskInfo {
  /** The scanned directory (absolute or as supplied). */
  root: string;
  /** Whether `root` exists and is a directory. */
  exists: boolean;
  /** Number of canonical `<64-hex>.safetensors` blocks. */
  entryCount: number;
  /** Sum of block sizes in bytes. */
  totalBytes: number;
  /** `min(0.10 * fsCapacity, 100 GiB)` from `statfsSync(root)`; 0 when missing. */
  quotaBytes: number;
  /** Oldest block mtime (ms since epoch), or `null` when there are no blocks. */
  oldestMtime: number | null;
  /** Newest block mtime (ms since epoch), or `null` when there are no blocks. */
  newestMtime: number | null;
  /** Fixed-order age buckets over block mtime. */
  ageHistogram: Array<{ label: string; count: number; bytes: number }>;
}

const BUCKET_LABELS = ['<1d', '1-7d', '7-30d', '>30d'] as const;

/**
 * The on-disk block directory the running tier uses, mirroring
 * `global_cold_cache` (cold_tier.rs): `MLX_COLD_CACHE_DIR`'s `mlx-paged-v1`
 * child when the env var is set and non-empty, else `~/.mlx-node/cache/paged/v1`.
 */
function defaultColdCacheDir(): string {
  const override = process.env.MLX_COLD_CACHE_DIR;
  if (override !== undefined && override.length > 0) return join(override, MANAGED_SUBDIR);
  return join(homedir(), '.mlx-node', 'cache', 'paged', 'v1');
}

interface BlockEntry {
  path: string;
  size: number;
  mtimeMs: number;
}

/**
 * Canonical block files under `root`. The root itself is typed with `lstatSync`
 * first: a symlinked or non-directory root is refused (empty list), mirroring
 * the native manager's opener (cold_cache.rs `symlink_metadata` root check),
 * so clear/evict can never unlink through a symlinked root into foreign files.
 * Entries are likewise typed with `lstatSync`, so a symlink at a block name is
 * never followed or counted — mirroring the manager's no-follow `stat_file`,
 * which indexes regular files only. A missing or unreadable directory yields an
 * empty list.
 */
function listBlocks(root: string): BlockEntry[] {
  let rootStat: Stats;
  try {
    rootStat = lstatSync(root);
  } catch {
    return [];
  }
  if (rootStat.isSymbolicLink() || !rootStat.isDirectory()) return [];
  let names: string[];
  try {
    names = readdirSync(root);
  } catch {
    return [];
  }
  const blocks: BlockEntry[] = [];
  for (const name of names) {
    if (!BLOCK_FILE.test(name)) continue;
    const path = join(root, name);
    let stat: Stats;
    try {
      stat = lstatSync(path);
    } catch {
      continue;
    }
    if (!stat.isFile()) continue;
    blocks.push({ path, size: stat.size, mtimeMs: stat.mtimeMs });
  }
  return blocks;
}

function directoryExists(root: string): boolean {
  try {
    return statSync(root).isDirectory();
  } catch {
    return false;
  }
}

/** `min(floor(capacity / 10), 100 GiB)`; 0 when the root cannot be stat'd. */
function quotaBytesFor(root: string): number {
  try {
    const fs = statfsSync(root);
    const capacity = fs.bsize * fs.blocks;
    return Math.min(Math.floor(capacity / 10), MAX_DEFAULT_QUOTA);
  } catch {
    return 0;
  }
}

function bucketIndex(ageMs: number): number {
  if (ageMs < DAY_MS) return 0;
  if (ageMs < 7 * DAY_MS) return 1;
  if (ageMs < 30 * DAY_MS) return 2;
  return 3;
}

function emptyHistogram(): ColdCacheDiskInfo['ageHistogram'] {
  return BUCKET_LABELS.map((label) => ({ label, count: 0, bytes: 0 }));
}

/** Read-only disk view of the cold tier. Defaults to the running-tier root. */
export function scanColdCache(root: string = defaultColdCacheDir()): ColdCacheDiskInfo {
  const histogram = emptyHistogram();
  if (!directoryExists(root)) {
    return {
      root,
      exists: false,
      entryCount: 0,
      totalBytes: 0,
      quotaBytes: 0,
      oldestMtime: null,
      newestMtime: null,
      ageHistogram: histogram,
    };
  }

  const blocks = listBlocks(root);
  const now = Date.now();
  let totalBytes = 0;
  let oldestMtime: number | null = null;
  let newestMtime: number | null = null;
  for (const block of blocks) {
    totalBytes += block.size;
    if (oldestMtime === null || block.mtimeMs < oldestMtime) oldestMtime = block.mtimeMs;
    if (newestMtime === null || block.mtimeMs > newestMtime) newestMtime = block.mtimeMs;
    const bucket = histogram[bucketIndex(now - block.mtimeMs)];
    bucket.count += 1;
    bucket.bytes += block.size;
  }

  return {
    root,
    exists: true,
    entryCount: blocks.length,
    totalBytes,
    quotaBytes: quotaBytesFor(root),
    oldestMtime,
    newestMtime,
    ageHistogram: histogram,
  };
}

/**
 * Remove every canonical block under `root`, leaving quarantine, writer-temp,
 * and any foreign file untouched. Only files actually unlinked are counted, so
 * a concurrent removal never inflates the totals. A missing root is a no-op.
 */
export function clearColdCache(root: string = defaultColdCacheDir()): { removed: number; freedBytes: number } {
  let removed = 0;
  let freedBytes = 0;
  for (const block of listBlocks(root)) {
    try {
      unlinkSync(block.path);
      removed += 1;
      freedBytes += block.size;
    } catch {
      // Vanished or unremovable: leave accounting to the files we truly cleared.
    }
  }
  return { removed, freedBytes };
}

/**
 * Remove canonical blocks whose mtime is strictly older than `days` ago,
 * leaving newer blocks and every foreign file in place. A missing root is a
 * no-op.
 */
export function evictOlderThan(
  days: number,
  root: string = defaultColdCacheDir(),
): { removed: number; freedBytes: number } {
  const cutoff = Date.now() - days * DAY_MS;
  let removed = 0;
  let freedBytes = 0;
  for (const block of listBlocks(root)) {
    if (block.mtimeMs >= cutoff) continue;
    try {
      unlinkSync(block.path);
      removed += 1;
      freedBytes += block.size;
    } catch {
      // Vanished or unremovable: skip.
    }
  }
  return { removed, freedBytes };
}

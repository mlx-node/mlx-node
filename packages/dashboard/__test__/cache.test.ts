import { existsSync, mkdtempSync, readdirSync, rmSync, utimesSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { clearColdCache, evictOlderThan, scanColdCache } from '../src/cache.js';

const DAY_MS = 24 * 60 * 60 * 1000;

/** 64-char lowercase-hex block stem (valid canonical filename component). */
function hexName(index: number): string {
  return `${index.toString(16).padStart(64, '0')}.safetensors`;
}

interface StagedBlock {
  name: string;
  bytes: number;
  ageDays: number;
}

const STAGED: StagedBlock[] = [
  { name: hexName(1), bytes: 10, ageDays: 0.5 }, // <1d
  { name: hexName(2), bytes: 20, ageDays: 3 }, //   1-7d
  { name: hexName(3), bytes: 30, ageDays: 10 }, //  7-30d
  { name: hexName(4), bytes: 40, ageDays: 40 }, //  >30d
];

let root: string;

/** (Re)write the four canonical blocks at staged mtimes plus foreign strangers. */
function stage(now: number): void {
  for (const block of STAGED) {
    const path = join(root, block.name);
    writeFileSync(path, Buffer.alloc(block.bytes));
    const when = new Date(now - block.ageDays * DAY_MS);
    utimesSync(path, when, when);
  }
  // Strangers the scan/clear/evict must never touch.
  writeFileSync(join(root, 'stats.json'), '{"n":1}');
  writeFileSync(join(root, `.blocked.${hexName(9)}.12345.67890`), 'quarantine');
  writeFileSync(join(root, `.${'a'.repeat(64)}.12345.67890.tmp`), 'writer-temp');
  writeFileSync(join(root, 'deadbeef.safetensors'), 'too-short-hex'); // 8 hex, not 64
  writeFileSync(join(root, `${'A'.repeat(64)}.safetensors`), 'uppercase-hex'); // not [0-9a-f]
}

const STRANGERS = [
  'stats.json',
  `.blocked.${hexName(9)}.12345.67890`,
  `.${'a'.repeat(64)}.12345.67890.tmp`,
  'deadbeef.safetensors',
  `${'A'.repeat(64)}.safetensors`,
];

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'dash-cold-'));
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

describe('scanColdCache', () => {
  it('counts only canonical 64-hex blocks with correct age-histogram buckets', () => {
    const now = Date.now();
    stage(now);

    const info = scanColdCache(root);
    expect(info.root).toBe(root);
    expect(info.exists).toBe(true);
    expect(info.entryCount).toBe(4);
    expect(info.totalBytes).toBe(100);
    expect(info.quotaBytes).toBeGreaterThan(0);

    expect(info.ageHistogram.map((b) => b.label)).toEqual(['<1d', '1-7d', '7-30d', '>30d']);
    expect(info.ageHistogram).toEqual([
      { label: '<1d', count: 1, bytes: 10 },
      { label: '1-7d', count: 1, bytes: 20 },
      { label: '7-30d', count: 1, bytes: 30 },
      { label: '>30d', count: 1, bytes: 40 },
    ]);

    expect(info.oldestMtime).not.toBeNull();
    expect(info.newestMtime).not.toBeNull();
    expect(Math.abs((info.oldestMtime as number) - (now - 40 * DAY_MS))).toBeLessThan(2000);
    expect(Math.abs((info.newestMtime as number) - (now - 0.5 * DAY_MS))).toBeLessThan(2000);
  });

  it('reports exists:false and zero counts for a missing root', () => {
    const missing = join(root, 'does', 'not', 'exist');
    const info = scanColdCache(missing);
    expect(info.exists).toBe(false);
    expect(info.entryCount).toBe(0);
    expect(info.totalBytes).toBe(0);
    expect(info.quotaBytes).toBe(0);
    expect(info.oldestMtime).toBeNull();
    expect(info.newestMtime).toBeNull();
    expect(info.ageHistogram).toEqual([
      { label: '<1d', count: 0, bytes: 0 },
      { label: '1-7d', count: 0, bytes: 0 },
      { label: '7-30d', count: 0, bytes: 0 },
      { label: '>30d', count: 0, bytes: 0 },
    ]);
  });

  it('defaults to ~/.mlx-node/cache/paged/v1, honoring MLX_COLD_CACHE_DIR', () => {
    const prev = process.env.MLX_COLD_CACHE_DIR;
    try {
      delete process.env.MLX_COLD_CACHE_DIR;
      expect(scanColdCache().root).toBe(join(homedir(), '.mlx-node', 'cache', 'paged', 'v1'));

      process.env.MLX_COLD_CACHE_DIR = root;
      expect(scanColdCache().root).toBe(join(root, 'mlx-paged-v1'));
    } finally {
      if (prev === undefined) delete process.env.MLX_COLD_CACHE_DIR;
      else process.env.MLX_COLD_CACHE_DIR = prev;
    }
  });
});

describe('clearColdCache', () => {
  it('removes only canonical blocks and leaves foreign files intact', () => {
    stage(Date.now());

    const result = clearColdCache(root);
    expect(result.removed).toBe(4);
    expect(result.freedBytes).toBe(100);

    for (const block of STAGED) expect(existsSync(join(root, block.name))).toBe(false);
    for (const stranger of STRANGERS) expect(existsSync(join(root, stranger))).toBe(true);

    // A rescan now sees no canonical blocks but the directory still exists.
    const after = scanColdCache(root);
    expect(after.exists).toBe(true);
    expect(after.entryCount).toBe(0);
    expect(after.totalBytes).toBe(0);
  });

  it('is a no-op for a missing root', () => {
    const result = clearColdCache(join(root, 'nope'));
    expect(result).toEqual({ removed: 0, freedBytes: 0 });
  });
});

describe('evictOlderThan', () => {
  it('removes only blocks older than the cutoff, leaving newer blocks and strangers', () => {
    stage(Date.now());

    const result = evictOlderThan(7, root);
    expect(result.removed).toBe(2); // 10d + 40d
    expect(result.freedBytes).toBe(70); // 30 + 40

    expect(existsSync(join(root, hexName(1)))).toBe(true); // 0.5d kept
    expect(existsSync(join(root, hexName(2)))).toBe(true); // 3d kept
    expect(existsSync(join(root, hexName(3)))).toBe(false); // 10d evicted
    expect(existsSync(join(root, hexName(4)))).toBe(false); // 40d evicted
    for (const stranger of STRANGERS) expect(existsSync(join(root, stranger))).toBe(true);

    // Remaining canonical blocks are exactly the two kept ones.
    const names = readdirSync(root).filter((n) => /^[0-9a-f]{64}\.safetensors$/.test(n));
    expect(names.sort()).toEqual([hexName(1), hexName(2)].sort());
  });

  it('is a no-op for a missing root', () => {
    const result = evictOlderThan(7, join(root, 'nope'));
    expect(result).toEqual({ removed: 0, freedBytes: 0 });
  });
});

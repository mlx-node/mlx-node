import { existsSync, mkdtempSync, readdirSync, rmSync, symlinkSync, utimesSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { MessageChannel } from 'node:worker_threads';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { clearColdCache, evictOlderThan, scanColdCache } from '../src/cache.js';
import { serveRuntimeOverPort } from '../src/rpc/host.js';
import { bindEventTargetPort } from '../src/rpc/port.js';
import type { ApiCall } from '../src/runtime.js';
import { connectDashboardApi, disconnectDashboardApi, mutate } from '../ui/src/lib/api.js';
import { coldObjectCounts, evictPreview, histogramObjectTotal } from '../ui/src/lib/cold-tier.js';

const DAY_MS = 24 * 60 * 60 * 1000;

/** 64-char lowercase-hex block stem (valid canonical filename component). */
function hexName(index: number): string {
  return `${index.toString(16).padStart(64, '0')}.safetensors`;
}

/** Canonical sidecar filename: same key, `ColdGroup` label as an infix. */
function sidecarName(index: number, group: 'gdn_state' | 'sliding_window'): string {
  return `${index.toString(16).padStart(64, '0')}.${group}.safetensors`;
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

  it('accounts sidecars separately but inside the quota total', () => {
    const now = Date.now();
    stage(now);
    // Two sidecars, aged into distinct buckets, plus near-miss names that must
    // be ignored: an unknown group label and the `kv` label (a KV object is
    // only ever the bare 64-hex form).
    const gdn = join(root, sidecarName(5, 'gdn_state'));
    writeFileSync(gdn, Buffer.alloc(7));
    const gdnWhen = new Date(now - 0.5 * DAY_MS);
    utimesSync(gdn, gdnWhen, gdnWhen);

    const win = join(root, sidecarName(6, 'sliding_window'));
    writeFileSync(win, Buffer.alloc(3));
    const winWhen = new Date(now - 10 * DAY_MS);
    utimesSync(win, winWhen, winWhen);

    writeFileSync(join(root, `${'0'.repeat(63)}7.mamba_state.safetensors`), 'unknown-group');
    writeFileSync(join(root, `${'0'.repeat(63)}8.kv.safetensors`), 'kv-is-never-an-infix');

    const info = scanColdCache(root);
    expect(info.entryCount).toBe(4); // blocks only
    expect(info.sidecarCount).toBe(2);
    expect(info.sidecarBytes).toBe(10);
    expect(info.totalBytes).toBe(110); // 100 block bytes + 10 sidecar bytes
    // Sidecars are evictable by mtime, so the preview histogram counts them.
    expect(info.ageHistogram).toEqual([
      { label: '<1d', count: 2, bytes: 17 },
      { label: '1-7d', count: 1, bytes: 20 },
      { label: '7-30d', count: 2, bytes: 33 },
      { label: '>30d', count: 1, bytes: 40 },
    ]);
  });

  it('clear and evict remove sidecars too, so none can outlive the quota view', () => {
    const now = Date.now();
    stage(now);
    const fresh = join(root, sidecarName(5, 'gdn_state'));
    writeFileSync(fresh, Buffer.alloc(7));
    const freshWhen = new Date(now - 0.5 * DAY_MS);
    utimesSync(fresh, freshWhen, freshWhen);

    const stale = join(root, sidecarName(6, 'sliding_window'));
    writeFileSync(stale, Buffer.alloc(3));
    const staleWhen = new Date(now - 40 * DAY_MS);
    utimesSync(stale, staleWhen, staleWhen);

    expect(evictOlderThan(7, root)).toEqual({ removed: 3, freedBytes: 73 }); // 30 + 40 + 3
    expect(existsSync(stale)).toBe(false);
    expect(existsSync(fresh)).toBe(true);

    expect(clearColdCache(root)).toEqual({ removed: 3, freedBytes: 37 }); // 10 + 20 + 7
    expect(existsSync(fresh)).toBe(false);
    for (const stranger of STRANGERS) expect(existsSync(join(root, stranger))).toBe(true);

    const after = scanColdCache(root);
    expect(after.entryCount).toBe(0);
    expect(after.sidecarCount).toBe(0);
    expect(after.totalBytes).toBe(0);
    expect(after.sidecarBytes).toBe(0);
  });

  it('reports exists:false and zero counts for a missing root', () => {
    const missing = join(root, 'does', 'not', 'exist');
    const info = scanColdCache(missing);
    expect(info.exists).toBe(false);
    expect(info.entryCount).toBe(0);
    expect(info.sidecarCount).toBe(0);
    expect(info.totalBytes).toBe(0);
    expect(info.sidecarBytes).toBe(0);
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

// Finding 3: a symlinked cache root must be refused, matching the native tier's
// opener (cold_cache.rs `symlink_metadata` root check). clear/evict must never
// unlink through the symlink into foreign files.
describe('symlinked cold-cache root (Finding 3)', () => {
  it('is not traversed and its target block is never deleted by clear/evict', () => {
    const victimBase = mkdtempSync(join(tmpdir(), 'dash-cold-victim-'));
    try {
      // A foreign canonical-looking block behind a symlinked root, aged well past
      // any eviction cutoff so a followed symlink would definitely delete it.
      const block = join(victimBase, hexName(1));
      writeFileSync(block, Buffer.alloc(10));
      const old = new Date(Date.now() - 40 * DAY_MS);
      utimesSync(block, old, old);

      const linkRoot = join(root, 'link-to-victim');
      symlinkSync(victimBase, linkRoot);

      // Not traversed: nothing listed, nothing removed.
      expect(scanColdCache(linkRoot).entryCount).toBe(0);
      expect(clearColdCache(linkRoot)).toEqual({ removed: 0, freedBytes: 0 });
      expect(evictOlderThan(7, linkRoot)).toEqual({ removed: 0, freedBytes: 0 });

      // The foreign target survives untouched.
      expect(existsSync(block)).toBe(true);
    } finally {
      rmSync(victimBase, { recursive: true, force: true });
    }
  });
});

// Finding 11: the Cache page's "Clear all" builds `{ all: true }` (not
// `undefined`). The `mutate` client omits the request body entirely when it is
// `undefined`, so the old clear-all body reached the server as an empty payload
// that `handleCacheDelete` rejects with 400 (its explicit `{all:true}` guard).
// This pins the client contract for both the clear and evict request bodies.
describe('cache DELETE request body (Cache page → mutate)', () => {
  it('sends {all:true} for clear and {olderThanDays} for evict, and no body for undefined', async () => {
    // Driven over a real port pair, so what is asserted is the `ApiCall` that
    // actually reaches the runtime after a structured clone — not an intermediate
    // the client happens to build.
    const calls: ApiCall[] = [];
    const { port1, port2 } = new MessageChannel();
    port1.unref();
    port2.unref();
    const dispose = serveRuntimeOverPort(
      {
        call: async (call: ApiCall) => {
          calls.push(call);
          return { ok: true, status: 200, body: { removed: 0, freedBytes: 0 } };
        },
        subscribe: () => () => {},
      },
      bindEventTargetPort(port2),
    );
    connectDashboardApi(bindEventTargetPort(port1), { onUnresponsive: () => port2.close() });

    try {
      // The bodies cache.tsx constructs per action kind.
      await mutate('DELETE', '/cache', { all: true });
      await mutate('DELETE', '/cache', { olderThanDays: 7 });
      // The OLD clear-all body: an omitted payload the server rejects.
      await mutate('DELETE', '/cache', undefined);
    } finally {
      disconnectDashboardApi();
      dispose();
    }

    expect(calls).toHaveLength(3);
    expect(calls[0]).toEqual({ method: 'DELETE', path: '/api/cache', body: { all: true } });
    expect(calls[1]).toEqual({ method: 'DELETE', path: '/api/cache', body: { olderThanDays: 7 } });
    // An omitted body must not become `{}` or `null` on the way across: the
    // handler's ambiguous-body 400 is what distinguishes it from a real clear-all.
    expect(calls[2]).toEqual({ method: 'DELETE', path: '/api/cache' });
    expect('body' in calls[2]).toBe(false);
  });
});

/**
 * F2 — the Blocks tile and the Blocks-by-age chart disagreed by exactly
 * `sidecarCount` (136 vs 138). Both numbers were correct; both were LABELLED
 * "blocks". `entryCount` is KV blocks alone, while the histogram (and
 * `totalBytes`, and both mtimes) covers blocks AND sidecars because clear/evict
 * remove both.
 *
 * These run the real filesystem scan and then the real UI counting helpers on
 * its output, so the invariant is asserted end-to-end rather than against a
 * hand-written literal that could agree with itself.
 */
describe('cold-tier object counting (Cache page ↔ scan)', () => {
  it('reports the histogram total as OBJECTS, matching blocks + sidecars', () => {
    const now = Date.now();
    stage(now);
    // Two sidecars beside the four blocks — the 138-vs-136 shape, in miniature.
    for (const [index, group, ageDays] of [
      [5, 'gdn_state', 0.5],
      [6, 'sliding_window', 10],
    ] as const) {
      const path = join(root, sidecarName(index, group));
      writeFileSync(path, Buffer.alloc(5));
      const when = new Date(now - ageDays * DAY_MS);
      utimesSync(path, when, when);
    }

    const info = scanColdCache(root);
    const counts = coldObjectCounts(info);

    // The tile's own number, unchanged and still blocks-only.
    expect(counts.blocks).toBe(info.entryCount);
    expect(counts.sidecars).toBe(info.sidecarCount);
    // THE INVARIANT: whatever the chart totals is what `objects` reports. A
    // tile driven by `blocks` disagrees with this chart by `sidecars`.
    expect(counts.objects).toBe(histogramObjectTotal(info.ageHistogram));
    expect(counts.objects).toBe(6);
    expect(counts.blocks).toBe(4);
    expect(counts.sidecars).toBe(2);
    // And the two really are different numbers here, so the assertion above is
    // not passing by coincidence on an all-blocks fixture.
    expect(counts.objects).not.toBe(counts.blocks);
  });

  it('counts sidecars in the eviction preview, matching what evictOlderThan unlinks', () => {
    const now = Date.now();
    stage(now);
    // One sidecar older than 7 days: it WILL be unlinked, so the preview the
    // confirm dialog shows has to include it.
    const old = join(root, sidecarName(6, 'sliding_window'));
    writeFileSync(old, Buffer.alloc(5));
    const when = new Date(now - 10 * DAY_MS);
    utimesSync(old, when, when);

    const preview = evictPreview(scanColdCache(root).ageHistogram, 7);
    const actual = evictOlderThan(7, root);
    expect(preview.count).toBe(actual.removed);
    expect(preview.bytes).toBe(actual.freedBytes);
    // blocks #3 (10d) + #4 (40d) + the sidecar.
    expect(actual.removed).toBe(3);
  });

  /**
   * `startIndex` has THREE arms and only the middle one was reachable from a
   * test: `days = 7` maps to 2 under both `days >= 30 ? 3 : days >= 7 ? 2 : 1`
   * and the collapsed `days >= 30 ? 3 : 2`, so the obvious mutation was a no-op
   * and the 1-day and 30-day arms shipped unexercised. Both are reachable from
   * the page's own `EVICT_OPTIONS`.
   *
   * Each case is oracled against `evictOlderThan` at the same threshold — the
   * function the dialog is predicting — rather than against a hand-computed
   * number, so the preview cannot drift from the deletion.
   */
  for (const [days, removed, freed] of [
    [1, 3, 90], // #2 (3d) + #3 (10d) + #4 (40d)
    [7, 2, 70], // #3 + #4
    [30, 1, 40], // #4
  ] as const) {
    it(`previews exactly what evictOlderThan(${days}) removes`, () => {
      stage(Date.now());
      const preview = evictPreview(scanColdCache(root).ageHistogram, days);
      const actual = evictOlderThan(days, root);
      expect(preview.count).toBe(actual.removed);
      expect(preview.bytes).toBe(actual.freedBytes);
      // Pinned so a fixture that stopped straddling the boundary (making the
      // oracle agree trivially at 0) would be caught too.
      expect(actual.removed).toBe(removed);
      expect(actual.freedBytes).toBe(freed);
    });
  }

  it('does not grey out the Evict button when the only evictable objects sit in the 1-7d bucket', () => {
    // The live consequence of the untested 1-day arm: the page disables Evict
    // on `evictP.count === 0`, so a preview that skipped the 1-7d bucket left
    // occupied quota with no way to clear it — the same dead end the
    // sidecar-only fix closed.
    const histogram = [
      { label: '<1d', count: 1, bytes: 25 },
      { label: '1-7d', count: 5, bytes: 500 },
      { label: '7-30d', count: 0, bytes: 0 },
      { label: '>30d', count: 0, bytes: 0 },
    ];
    expect(evictPreview(histogram, 1)).toEqual({ count: 5, bytes: 500 });
    // …and the other two thresholds genuinely have nothing to offer here, so
    // the 1-day arm is the only thing standing between the user and the quota.
    expect(evictPreview(histogram, 7)).toEqual({ count: 0, bytes: 0 });
    expect(evictPreview(histogram, 30)).toEqual({ count: 0, bytes: 0 });
  });

  it('treats a sidecar-only tier as non-empty so its quota is clearable', () => {
    // Reachable state: blocks evicted, sidecars left behind. Gating the
    // destructive controls on `entryCount` showed occupied quota the UI refused
    // to clear, and the age chart claimed "no blocks persisted yet".
    const solo = join(root, sidecarName(9, 'gdn_state'));
    writeFileSync(solo, Buffer.alloc(64));

    const info = scanColdCache(root);
    expect(info.entryCount).toBe(0);
    expect(info.totalBytes).toBeGreaterThan(0);
    expect(coldObjectCounts(info).objects).toBe(1);
    expect(clearColdCache(root).removed).toBe(1);
  });
});

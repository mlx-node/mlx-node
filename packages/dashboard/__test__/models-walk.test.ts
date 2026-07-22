import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, expect, it, vi } from 'vite-plus/test';

// `node:fs`'s exports are non-configurable, so `vi.spyOn` cannot wrap them. Remap the
// module to delegating wrappers that COUNT the enumeration primitives instead, so a
// test can prove `walkDirStats` enumerates INCREMENTALLY (`opendirSync` +
// `Dir.readSync`, bounded by the entry budget) and never materializes the whole
// directory up front via `readdirSync(withFileTypes)`.
const fsCounters = vi.hoisted(() => ({ readdir: 0, opendir: 0, readSync: 0 }));

vi.mock('node:fs', async (importOriginal) => {
  const real = await importOriginal<typeof import('node:fs')>();
  return {
    ...real,
    readdirSync: (...args: Parameters<typeof real.readdirSync>) => {
      fsCounters.readdir++;
      return real.readdirSync(...(args as Parameters<typeof real.readdirSync>));
    },
    opendirSync: (...args: Parameters<typeof real.opendirSync>) => {
      fsCounters.opendir++;
      const dir = real.opendirSync(...(args as Parameters<typeof real.opendirSync>));
      const read = dir.readSync.bind(dir);
      dir.readSync = () => {
        fsCounters.readSync++;
        return read();
      };
      return dir;
    },
  };
});

// Bound AFTER the (hoisted) mock so the module under test uses the wrapped `node:fs`.
const { walkDirStats } = await import('../src/models.js');

let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'dash-walk-'));
  fsCounters.readdir = 0;
  fsCounters.opendir = 0;
  fsCounters.readSync = 0;
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

// The entry budget must bound BOTH memory and time: `readdirSync(withFileTypes)`
// first allocates the ENTIRE `Dirent[]` (millions of entries on a pathological
// high-fan-out dir) and blocks the synchronous `/api/models` handler before the
// per-entry budget can ever apply. The walk must instead enumerate incrementally so
// the budget stops it early.
it('bounds enumeration by the entry budget on a high-fan-out dir (incremental, not eager readdir)', () => {
  const budget = 5;
  const total = budget + 20; // 25 flat files — strictly more than the budget.
  for (let i = 0; i < total; i++) writeFileSync(join(dir, `f${i.toString().padStart(3, '0')}.bin`), Buffer.alloc(1));

  fsCounters.readdir = 0;
  fsCounters.opendir = 0;
  fsCounters.readSync = 0;
  const stats = walkDirStats(dir, budget);

  // Stops AT the budget: exactly `budget` files counted, `truncated` set.
  expect(stats.truncated).toBe(true);
  expect(stats.fileCount).toBe(budget);
  // Incremental strategy: `opendirSync` is used and `readdirSync` never materializes
  // the whole directory.
  expect(fsCounters.opendir).toBeGreaterThan(0);
  expect(fsCounters.readdir).toBe(0);
  // Enumeration is BOUNDED: at most `budget + 1` entries were read (the +1 is the
  // entry that trips the budget) — never all 25, proving the `Dirent[]` is not drained.
  expect(fsCounters.readSync).toBeLessThanOrEqual(budget + 1);
});

// A directory that fits the budget must count every file and NOT report truncation
// (the enumeration reaches the directory's genuine end-of-stream, not the cap).
it('counts every file and does not truncate when the dir fits the budget', () => {
  const files = 6;
  for (let i = 0; i < files; i++) writeFileSync(join(dir, `f${i}.bin`), Buffer.alloc(2));

  const stats = walkDirStats(dir, 100);
  expect(stats.truncated).toBe(false);
  expect(stats.fileCount).toBe(files);
  expect(stats.sizeBytes).toBe(files * 2);
});

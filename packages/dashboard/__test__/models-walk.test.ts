import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, expect, it, vi } from 'vite-plus/test';

// `node:fs`'s exports are non-configurable, so `vi.spyOn` cannot wrap them. Remap the
// module to delegating wrappers that COUNT the enumeration primitives instead, so a
// test can prove `walkDirStats` enumerates INCREMENTALLY (`opendirSync` +
// `Dir.readSync`, bounded by the entry budget) and never materializes the whole
// directory up front via `readdirSync(withFileTypes)`. The wrappers additionally
// track CONCURRENT open `Dir` handles (increment on `opendirSync`, decrement on
// `closeSync`) so a test can prove the walk holds at most ONE handle at a time —
// never one fd per tree level — and can inject `opendirSync`/`readSync` faults on a
// chosen path to prove a resource failure surfaces as `truncated` (incomplete),
// never a silent undercount.
const fsCounters = vi.hoisted(() => ({
  readdir: 0,
  opendir: 0,
  readSync: 0,
  openHandles: 0,
  maxOpenHandles: 0,
  /** Every path passed to `opendirSync` (to prove a swapped-in symlink is never followed). */
  opened: [] as string[],
}));

// When set, `opendirSync`/`Dir.readSync` on EXACTLY this path throws a resource
// error (EMFILE / EIO), simulating fd exhaustion or a mid-directory read fault.
// `lstatSymlinkOn` makes `lstatSync` on EXACTLY that path report a SYMLINK
// (`isDirectory() === false`) while the real on-disk entry is still a directory —
// modelling a concurrent swap of a queued real dir to a symlink between enqueue and
// pop, so a test can prove the walk does NOT `opendirSync`/follow it.
const fsFail = vi.hoisted(() => ({
  opendirThrowOn: null as string | null,
  readThrowOn: null as string | null,
  lstatSymlinkOn: null as string | null,
}));

vi.mock('node:fs', async (importOriginal) => {
  const real = await importOriginal<typeof import('node:fs')>();
  return {
    ...real,
    readdirSync: (...args: Parameters<typeof real.readdirSync>) => {
      fsCounters.readdir++;
      return real.readdirSync(...(args as Parameters<typeof real.readdirSync>));
    },
    lstatSync: ((...args: Parameters<typeof real.lstatSync>) => {
      const path = String(args[0]);
      const stat = real.lstatSync(...(args as Parameters<typeof real.lstatSync>));
      if (fsFail.lstatSymlinkOn !== null && path === fsFail.lstatSymlinkOn) {
        // Report a symlink at this path (no-follow lstat): keep the real dev/ino so
        // the walk's visited-set is unperturbed, but flip the type predicates so the
        // walk sees a swapped-in symlink, not a directory.
        return Object.assign(Object.create(Object.getPrototypeOf(stat)), stat, {
          isDirectory: () => false,
          isSymbolicLink: () => true,
          isFile: () => false,
        });
      }
      return stat;
    }) as typeof real.lstatSync,
    opendirSync: (...args: Parameters<typeof real.opendirSync>) => {
      fsCounters.opendir++;
      const path = String(args[0]);
      fsCounters.opened.push(path);
      if (fsFail.opendirThrowOn !== null && path === fsFail.opendirThrowOn) {
        const err = new Error(`EMFILE: too many open files, opendir '${path}'`) as NodeJS.ErrnoException;
        err.code = 'EMFILE';
        throw err;
      }
      const dir = real.opendirSync(...(args as Parameters<typeof real.opendirSync>));
      fsCounters.openHandles++;
      if (fsCounters.openHandles > fsCounters.maxOpenHandles) fsCounters.maxOpenHandles = fsCounters.openHandles;
      const read = dir.readSync.bind(dir);
      dir.readSync = () => {
        fsCounters.readSync++;
        if (fsFail.readThrowOn !== null && path === fsFail.readThrowOn) {
          const err = new Error(`EIO: i/o error, read '${path}'`) as NodeJS.ErrnoException;
          err.code = 'EIO';
          throw err;
        }
        return read();
      };
      const close = dir.closeSync.bind(dir);
      let closed = false;
      dir.closeSync = () => {
        if (!closed) {
          closed = true;
          fsCounters.openHandles--;
        }
        return close();
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
  fsCounters.openHandles = 0;
  fsCounters.maxOpenHandles = 0;
  fsCounters.opened = [];
  fsFail.opendirThrowOn = null;
  fsFail.readThrowOn = null;
  fsFail.lstatSymlinkOn = null;
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

// FD discipline: recursion that descends while the parent `Dir` handle is still open
// holds ONE fd per tree level, so a deep real-directory tree under a low
// `RLIMIT_NOFILE` makes a child `opendirSync` hit EMFILE (silently dropping the
// subtree). An ITERATIVE bounded worklist must instead close each dir BEFORE opening
// any child, so at most ONE `Dir` handle is ever open — no matter how deep the tree.
it('holds at most one Dir handle at a time across a deep nested tree (no fd-per-level)', () => {
  // A linear chain several levels deep, each level a REAL directory with a file, so
  // the walk must descend level by level. Recursion would keep every level's handle
  // open at the leaf (concurrency == depth+1); the worklist keeps it at 1.
  let cursor = dir;
  const depth = 6;
  let expectedFiles = 0;
  for (let level = 0; level < depth; level++) {
    writeFileSync(join(cursor, `file${level}.bin`), Buffer.alloc(3));
    expectedFiles++;
    cursor = join(cursor, `level${level}`);
    mkdirSync(cursor, { recursive: true });
  }
  writeFileSync(join(cursor, 'leaf.bin'), Buffer.alloc(3));
  expectedFiles++;

  const stats = walkDirStats(dir, 100_000);

  // Every level was visited (all files counted) …
  expect(stats.truncated).toBe(false);
  expect(stats.fileCount).toBe(expectedFiles);
  expect(stats.sizeBytes).toBe(expectedFiles * 3);
  // … yet the walk never held more than a single open `Dir` handle at once.
  expect(fsCounters.maxOpenHandles).toBe(1);
  // Sanity: it really did open one handle per directory (root + `depth` subdirs).
  expect(fsCounters.opendir).toBe(depth + 1);
});

// Resource failure = INCOMPLETE, never a silent false-complete. A nested
// `opendirSync` that throws EMFILE (fd exhaustion) must set `truncated` so the
// caller warns the reported size is a lower bound — not swallow the subtree and
// return an exact-looking undercount.
it('marks truncated when a nested opendirSync throws (EMFILE), not a silent undercount', () => {
  writeFileSync(join(dir, 'top.bin'), Buffer.alloc(4));
  const sub = join(dir, 'sub');
  mkdirSync(sub, { recursive: true });
  writeFileSync(join(sub, 'inner.bin'), Buffer.alloc(8));

  // The child dir cannot be opened (simulated fd exhaustion).
  fsFail.opendirThrowOn = sub;
  const stats = walkDirStats(dir, 100_000);

  // The subtree was dropped but the result is flagged INCOMPLETE (a lower bound),
  // not reported as an exact, complete size.
  expect(stats.truncated).toBe(true);
  // The root's own file was still counted (the walk did not crash or bail entirely).
  expect(stats.fileCount).toBe(1);
  expect(stats.sizeBytes).toBe(4);
});

// A `readSync` fault mid-directory likewise must mark the result incomplete rather
// than `return` silently with `truncated` still false.
it('marks truncated when a nested Dir.readSync throws mid-directory', () => {
  writeFileSync(join(dir, 'top.bin'), Buffer.alloc(4));
  const sub = join(dir, 'sub');
  mkdirSync(sub, { recursive: true });
  writeFileSync(join(sub, 'inner.bin'), Buffer.alloc(8));

  // Reading the child directory faults partway.
  fsFail.readThrowOn = sub;
  const stats = walkDirStats(dir, 100_000);

  expect(stats.truncated).toBe(true);
  // The root file still counted; the faulted child contributed nothing.
  expect(stats.fileCount).toBe(1);
  expect(stats.sizeBytes).toBe(4);
});

// No-follow across the whole enqueue→pop window: the walker queues child PATHS and
// opens them later. A child enqueued as a REAL directory (its `Dirent.isDirectory()`
// true during `readSync`) can be swapped to a symlink before it is popped; the pop's
// `lstatSync` (no-follow) then reports a symlink. The walk must reject it BEFORE
// `opendirSync` (which FOLLOWS symlinks), or it descends into and sizes the external
// target — a no-follow escape widened from the immediate-recursion window to the whole
// parent-enumeration interval.
it('does not opendir/follow a queued dir that is swapped to a symlink before it is popped', () => {
  writeFileSync(join(dir, 'top.bin'), Buffer.alloc(4));
  const sub = join(dir, 'sub');
  mkdirSync(sub, { recursive: true });
  // A large "external" payload the walk must NOT count if it follows the swapped link.
  writeFileSync(join(sub, 'external.bin'), Buffer.alloc(4096));

  // `sub` is a real dir at enumeration (so it is enqueued), but a concurrent swap
  // turns it into a symlink: its no-follow `lstatSync` now reports a symlink.
  fsFail.lstatSymlinkOn = sub;
  const stats = walkDirStats(dir, 100_000);

  // The swapped-in symlink was never opendir'd/followed …
  expect(fsCounters.opened).not.toContain(sub);
  // … so only the root's own file is counted — the external payload stays out.
  expect(stats.fileCount).toBe(1);
  expect(stats.sizeBytes).toBe(4);
  expect(stats.truncated).toBe(false);
});

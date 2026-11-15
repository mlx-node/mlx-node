import { spawn } from 'node:child_process';
import {
  chmodSync,
  closeSync,
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  renameSync,
  rmSync,
  symlinkSync,
  unlinkSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  closePrefixCacheWalks,
  OPEN_WALKS_MAX,
  PrefixCacheError,
  removePrefixNamespaceStep,
  scanPrefixCaches,
  takeNamespaceLock,
  type PrefixCacheNamespace,
  type PrefixScanOptions,
} from '../src/prefix-cache.js';

const NS_A = '0123456789abcdef0123456789abcdef';
const NS_B = 'fedcba9876543210fedcba9876543210';
const NS_C = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa';
const PERL = '/usr/bin/perl';
const DARWIN = process.platform === 'darwin';
const PERL_FLOCK =
  '$| = 1; use Fcntl ":flock"; open F, "+<", $ARGV[0] or exit 4; flock(F, LOCK_EX|LOCK_NB) or exit 3; print "ok\\n"; <STDIN>';

let tmp: string;
let root: string;
const held: number[] = [];

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), 'dash-prefix-'));
  root = join(tmp, 'root');
  mkdirSync(root);
});

afterEach(() => {
  closePrefixCacheWalks();
  for (const fd of held.splice(0)) closeSync(fd);
  rmSync(tmp, { recursive: true, force: true });
});

/** A namespace with `files` (name → bytes) and a `lock` holding `lock` bytes, or none for null. */
function namespace(name: string, files: Record<string, number>, lock: number | null = 0, at: string = root): string {
  const dir = join(at, name);
  mkdirSync(dir, { recursive: true });
  for (const [file, bytes] of Object.entries(files)) writeFileSync(join(dir, file), Buffer.alloc(bytes));
  if (lock !== null) writeFileSync(join(dir, 'lock'), Buffer.alloc(lock));
  return dir;
}

/** Sets both times of `path` and returns the mtime the file system stored, which every comparison uses. */
function setTime(path: string, ms: number): number {
  utimesSync(path, new Date(ms), new Date(ms));
  return lstatSync(path).mtimeMs;
}

function openedOf(dir: string): number {
  return lstatSync(join(dir, 'lock')).mtimeMs;
}

function scan(opts: PrefixScanOptions = {}, at: string = root) {
  return scanPrefixCaches([{ root: at, isDefault: true, engines: [] }], opts);
}

function row(result: ReturnType<typeof scan>, name: string): PrefixCacheNamespace {
  const found = result.roots[0]!.namespaces.find((ns) => ns.namespace === name);
  if (found === undefined) throw new Error(`no row ${name}`);
  return found;
}

function hold(dir: string): number {
  const fd = takeNamespaceLock(dir);
  held.push(fd);
  return fd;
}

function release(fd: number): void {
  held.splice(held.indexOf(fd), 1);
  closeSync(fd);
}

/** `flock(LOCK_EX | LOCK_NB)` in another process: 'ok' while it holds the lock, else its exit code. */
function perlLock(path: string): Promise<{ result: 'ok' | number; release: () => Promise<void> }> {
  const child = spawn(PERL, ['-e', PERL_FLOCK, path], { stdio: ['pipe', 'pipe', 'inherit'] });
  const exited = new Promise<number>((resolve) => child.once('exit', (code) => resolve(code ?? -1)));
  const release = async (): Promise<void> => {
    child.stdin.end();
    await exited;
  };
  return new Promise((resolve) => {
    let out = '';
    child.stdout.on('data', (chunk: Buffer) => {
      out += chunk.toString();
      if (out.includes('ok')) resolve({ result: 'ok', release });
    });
    void exited.then((code) => resolve({ result: code, release }));
  });
}

function caught(run: () => unknown): PrefixCacheError {
  try {
    run();
  } catch (err) {
    if (err instanceof PrefixCacheError) return err;
    throw err;
  }
  throw new Error('no PrefixCacheError');
}

describe('scanPrefixCaches', () => {
  it('lists each namespace with its bytes, files and last use', () => {
    const t = Date.now() - 3_600_000;
    const a = namespace(NS_A, { [`${'1'.repeat(64)}.safetensors`]: 100, [`${'2'.repeat(64)}.kv.safetensors`]: 50 }, 7);
    namespace(NS_B, { one: 10 });
    mkdirSync(join(root, 'notes'));
    mkdirSync(join(root, 'f'.repeat(31)));
    mkdirSync(join(root, 'D'.repeat(32)));
    writeFileSync(join(root, 'c'.repeat(32)), 'x');
    setTime(join(a, `${'1'.repeat(64)}.safetensors`), t);
    setTime(join(a, `${'2'.repeat(64)}.kv.safetensors`), t + 1000);
    const opened = setTime(join(a, 'lock'), t + 2000);
    const result = scan();
    expect(result.roots[0]!.namespaces.map((ns) => ns.namespace)).toEqual([NS_A, NS_B]);
    expect(row(result, NS_A)).toMatchObject({
      bytes: 150,
      files: 2,
      lastUsedMs: opened,
      openedMs: opened,
      complete: true,
    });
    expect(row(result, NS_B)).toMatchObject({ bytes: 10, files: 1, complete: true });
    expect(result.counting).toBe(false);
    expect(result.roots[0]!.exists).toBe(true);
    expect(result.roots[0]!.readError).toBeNull();
    expect(result.roots[0]!.volume?.sizeBytes).toBeGreaterThan(0);
  });

  it('a folder without a regular lock file is not listed and gets no lock', () => {
    namespace(NS_A, { one: 1 }, null);
    const b = namespace(NS_B, { one: 1 }, null);
    symlinkSync(join(tmp, 'elsewhere-lock'), join(b, 'lock'));
    const c = namespace(NS_C, { one: 1 }, null);
    mkdirSync(join(c, 'lock'));
    expect(scan().roots[0]!.namespaces).toEqual([]);
    expect(readdirSync(join(root, NS_A))).toEqual(['one']);
    expect(lstatSync(join(b, 'lock')).isSymbolicLink()).toBe(true);
    expect(lstatSync(join(c, 'lock')).isDirectory()).toBe(true);
  });

  it('a symlinked namespace is not listed', () => {
    const outside = namespace(NS_B, { big: 999 }, 0, join(tmp, 'outside'));
    symlinkSync(outside, join(root, NS_A));
    expect(scan().roots[0]!.namespaces).toEqual([]);
  });

  it('a link inside a namespace adds no bytes', () => {
    writeFileSync(join(tmp, 'target'), Buffer.alloc(4096));
    const dir = namespace(NS_A, { real: 5 });
    symlinkSync(join(tmp, 'target'), join(dir, 'link'));
    expect(row(scan(), NS_A)).toMatchObject({ bytes: 5, files: 1 });
  });

  it('follows a symlinked root', () => {
    namespace(NS_A, { one: 3 });
    const linked = join(tmp, 'linked-root');
    symlinkSync(root, linked);
    const result = scan({}, linked);
    expect(result.roots[0]!.namespaces.map((ns) => [ns.namespace, ns.bytes])).toEqual([[NS_A, 3]]);
  });

  it.runIf(DARWIN)('reads the lock of each namespace', () => {
    const a = namespace(NS_A, { one: 1 });
    namespace(NS_B, { one: 1 });
    namespace(NS_C, { one: 1 }, null);
    hold(a);
    const result = scan();
    expect(row(result, NS_A).lock).toBe('held');
    expect(row(result, NS_B).lock).toBe('free');
    expect(result.roots[0]!.namespaces.map((ns) => ns.namespace)).toEqual([NS_A, NS_B]);
  });

  it.runIf(DARWIN && existsSync(PERL))('a lock taken with flock in another process reads as held', async () => {
    const a = namespace(NS_A, { one: 1 });
    const perl = await perlLock(join(a, 'lock'));
    try {
      expect(perl.result).toBe('ok');
      expect(row(scan(), NS_A).lock).toBe('held');
    } finally {
      await perl.release();
    }
    expect(row(scan(), NS_A).lock).toBe('free');
  });

  it('a walk stops at its budget and the next scan continues it', () => {
    const files: Record<string, number> = {};
    for (let i = 0; i < 10; i += 1) files[`f${String(i)}`] = 10;
    namespace(NS_A, files);
    const first = scan({ budget: 4 });
    expect([row(first, NS_A).complete, first.counting]).toEqual([false, true]);
    expect(row(scan({ budget: 4 }), NS_A).complete).toBe(false);
    const last = scan({ budget: 4 });
    expect(row(last, NS_A)).toMatchObject({ complete: true, bytes: 100, files: 10 });
    expect(last.counting).toBe(false);
  });

  it('an unchanged directory is not walked again', () => {
    const dir = namespace(NS_A, { a: 1, b: 2 });
    const base = Date.now() - 600_000;
    setTime(dir, base);
    expect(scan().entriesRead).toBeGreaterThan(0);
    const again = scan();
    expect(again.entriesRead).toBe(0);
    expect(row(again, NS_A).bytes).toBe(3);
    writeFileSync(join(dir, 'c'), Buffer.alloc(4));
    setTime(dir, base + 60_000);
    expect(row(scan(), NS_A)).toMatchObject({ bytes: 7, files: 3 });
  });

  it('a changed directory keeps its last numbers until the new walk ends', () => {
    const dir = namespace(NS_A, { a: 10, b: 10, c: 10, d: 10 });
    const base = Date.now() - 600_000;
    setTime(dir, base);
    const first = row(scan(), NS_A);
    expect(first.bytes).toBe(40);
    writeFileSync(join(dir, 'e'), Buffer.alloc(10));
    setTime(dir, base + 60_000);
    const during = scan({ budget: 2 });
    expect(row(during, NS_A)).toMatchObject({ bytes: 40, files: 4, complete: true, measuredAtMs: first.measuredAtMs });
    expect(during.counting).toBe(false);
    expect(row(scan(), NS_A)).toMatchObject({ bytes: 50, files: 5 });
  });

  it('lists at most the cap and says more exist', () => {
    namespace(NS_A, {});
    namespace(NS_B, {});
    namespace(NS_C, {});
    const result = scan({ namespacesMax: 2 });
    expect(result.roots[0]!.namespaces.map((ns) => ns.namespace)).toEqual([NS_A, NS_C]);
    expect(result.roots[0]!.truncated).toBe(true);
  });

  it('a missing root reads as not created', () => {
    const result = scan({}, join(tmp, 'missing'));
    expect(result.roots[0]).toMatchObject({
      exists: false,
      readError: null,
      namespaces: [],
      volume: null,
      truncated: false,
    });
  });

  it('a root it cannot read says so and is not shown as empty', () => {
    namespace(NS_A, { one: 1 });
    writeFileSync(join(tmp, 'a-file'), 'x');
    expect(scan({}, join(tmp, 'a-file')).roots[0]).toMatchObject({ exists: true, readError: 'ENOTDIR' });
    chmodSync(root, 0o000);
    try {
      expect(scan().roots[0]).toMatchObject({ exists: true, readError: 'EACCES', namespaces: [] });
    } finally {
      chmodSync(root, 0o755);
    }
    expect(scan().roots[0]!.readError).toBeNull();
  });

  it('reads at most the cap of root entries per call and says more exist', () => {
    for (let i = 0; i < 6; i += 1) mkdirSync(join(root, `note-${String(i)}`));
    namespace(NS_A, {});
    namespace(NS_B, {});
    const capped = scan({ rootEntriesMax: 3 });
    expect(capped.roots[0]!.truncated).toBe(true);
    expect(capped.roots[0]!.namespaces.length).toBeLessThanOrEqual(2);
    const all = scan({ rootEntriesMax: 8 });
    expect(all.roots[0]!.truncated).toBe(false);
    expect(all.roots[0]!.namespaces.map((ns) => ns.namespace)).toEqual([NS_A, NS_B]);
  });

  it('a walk stops at its time per call and the next calls finish it', () => {
    const files: Record<string, number> = {};
    for (let i = 0; i < 10; i += 1) files[`f${String(i)}`] = 10;
    namespace(NS_A, files);
    let t = 0;
    const clock = (): number => (t += 10);
    const first = scan({ deadlineMs: 25, clock });
    expect(first.entriesRead).toBeGreaterThan(0);
    expect(first.entriesRead).toBeLessThan(11);
    expect(row(first, NS_A).complete).toBe(false);
    let calls = 1;
    let last = first;
    while (last.counting && calls < 11) {
      last = scan({ deadlineMs: 25, clock });
      calls += 1;
    }
    expect(row(last, NS_A)).toMatchObject({ complete: true, bytes: 100, files: 10 });
  });

  it(`opens at most ${String(OPEN_WALKS_MAX)} walks at once`, () => {
    const files: Record<string, number> = {};
    for (let i = 0; i < 10; i += 1) files[`f${String(i)}`] = 1;
    for (let i = OPEN_WALKS_MAX + 1; i >= 2; i -= 1) {
      namespace(String(i).repeat(32), files);
      expect(row(scan({ budget: 1 }), String(i).repeat(32)).complete).toBe(false);
    }
    const last = '1'.repeat(32);
    namespace(last, { one: 1 });
    const capped = scan({ budget: 1000 });
    expect(row(capped, last).complete).toBe(false);
    expect(row(capped, '2'.repeat(32)).complete).toBe(true);
    const next = scan({ budget: 1000 });
    expect(row(next, last)).toMatchObject({ complete: true, files: 1 });
    expect(next.counting).toBe(false);
  });

  it.runIf(DARWIN)('a re-walk is closed when its row is current again', () => {
    for (let i = 1; i <= OPEN_WALKS_MAX; i += 1) {
      const dir = namespace(String(i).repeat(32), { a: 1, b: 1, c: 1 });
      const fd = hold(dir);
      expect(row(scan(), String(i).repeat(32))).toMatchObject({ complete: true, lock: 'held' });
      release(fd);
      scan({ budget: 1 });
      hold(dir);
    }
    expect(scan().counting).toBe(false);
    namespace(NS_A, { x: 4 });
    const after = scan();
    expect(row(after, NS_A)).toMatchObject({ complete: true, bytes: 4 });
    expect(after.counting).toBe(false);
  });

  it.runIf(DARWIN)('the scan leaves every lock free', async () => {
    const a = namespace(NS_A, { one: 1 });
    const b = namespace(NS_B, { one: 1 });
    scan();
    for (const dir of [a, b]) {
      closeSync(takeNamespaceLock(dir));
    }
    if (existsSync(PERL)) {
      const perl = await perlLock(join(a, 'lock'));
      await perl.release();
      expect(perl.result).toBe('ok');
    }
  });

  it('a file that vanishes during a walk is skipped', () => {
    const files: Record<string, number> = {};
    for (let i = 0; i < 10; i += 1) files[`f${String(i)}`] = 8;
    const dir = namespace(NS_A, files);
    const base = Date.now() - 600_000;
    setTime(dir, base);
    const first = scan({ budget: 4 });
    expect(row(first, NS_A).complete).toBe(false);
    const seen = row(first, NS_A).files;
    for (const name of Object.keys(files)) unlinkSync(join(dir, name));
    setTime(dir, base + 60_000);
    expect(row(scan(), NS_A)).toMatchObject({ complete: true, files: seen });
    expect(row(scan(), NS_A)).toMatchObject({ complete: true, files: 0 });
  });

  it.runIf(DARWIN)('a cache an engine let go is walked again', () => {
    const dir = namespace(NS_A, { a: 1 });
    const t = Date.now() - 600_000;
    const at = setTime(join(dir, 'a'), t);
    setTime(join(dir, 'lock'), t);
    const fd = hold(dir);
    expect(row(scan(), NS_A)).toMatchObject({ lock: 'held', lastUsedMs: Math.max(at, openedOf(dir)) });
    const later = setTime(join(dir, 'a'), t + 300_000);
    release(fd);
    expect(row(scan(), NS_A)).toMatchObject({ lock: 'free', lastUsedMs: later });
  });

  it('a cache opened since the last walk is walked again', () => {
    const dir = namespace(NS_A, { a: 1 });
    const base = Date.now() - 600_000;
    setTime(join(dir, 'a'), base);
    setTime(join(dir, 'lock'), base);
    scan();
    const t0 = setTime(join(dir, 'lock'), base + 60_000);
    const t1 = setTime(join(dir, 'a'), base + 120_000);
    expect(row(scan(), NS_A)).toMatchObject({ lastUsedMs: t1, openedMs: t0 });
  });
});

describe('removePrefixNamespaceStep', () => {
  it.runIf(DARWIN)('deletes one idle namespace and leaves its neighbours', () => {
    const a = namespace(NS_A, { x: 10, y: 20 });
    const b = namespace(NS_B, { z: 5 });
    writeFileSync(join(root, 'loose'), 'keep');
    writeFileSync(join(tmp, 'beside'), 'keep');
    const neighbour = readdirSync(b).sort();
    const result = removePrefixNamespaceStep(root, NS_A, openedOf(a));
    expect(result).toEqual({ removed: 2, freedBytes: 30, done: true, left: 0, directoryRemoved: true });
    expect(existsSync(a)).toBe(false);
    expect(readdirSync(b).sort()).toEqual(neighbour);
    expect(readdirSync(root).sort()).toEqual([NS_B, 'loose']);
    expect(existsSync(join(tmp, 'beside'))).toBe(true);
  });

  it.runIf(DARWIN)('refuses while the lock is held and removes nothing', () => {
    const a = namespace(NS_A, { x: 1 });
    hold(a);
    expect(caught(() => removePrefixNamespaceStep(root, NS_A, openedOf(a))).kind).toBe('held');
    expect(readdirSync(a).sort()).toEqual(['lock', 'x']);
  });

  it.runIf(DARWIN)('refuses a name that is not a namespace', () => {
    writeFileSync(join(tmp, 'sentinel'), 'keep');
    for (const name of ['..', '.', 'a/b', 'a'.repeat(31), 'A'.repeat(32), '']) {
      expect(caught(() => removePrefixNamespaceStep(root, name, 0)).kind, name).toBe('bad-name');
    }
    expect(existsSync(join(tmp, 'sentinel'))).toBe(true);
    expect(existsSync(root)).toBe(true);
  });

  it.runIf(DARWIN)('never follows a link', () => {
    const outside = join(tmp, 'outside');
    mkdirSync(outside);
    writeFileSync(join(outside, 'keep.safetensors'), 'keep');
    writeFileSync(join(outside, 'lock'), '');
    symlinkSync(outside, join(root, NS_A));
    expect(caught(() => removePrefixNamespaceStep(root, NS_A, openedOf(outside))).kind).toBe('not-found');
    expect(readdirSync(outside).sort()).toEqual(['keep.safetensors', 'lock']);

    writeFileSync(join(tmp, 'target'), 'keep');
    const b = namespace(NS_B, {});
    symlinkSync(join(tmp, 'target'), join(b, 'link'));
    expect(removePrefixNamespaceStep(root, NS_B, openedOf(b))).toMatchObject({ removed: 1, freedBytes: 0, done: true });
    expect(existsSync(b)).toBe(false);
    expect(existsSync(join(tmp, 'target'))).toBe(true);
  });

  it.runIf(DARWIN)('deletes in steps', () => {
    const files: Record<string, number> = {};
    for (let i = 0; i < 10; i += 1) files[`f${String(i)}`] = 1;
    const a = namespace(NS_A, files);
    const opened = openedOf(a);
    const steps = [];
    for (let i = 0; i < 3; i += 1) {
      const step = removePrefixNamespaceStep(root, NS_A, opened, { maxFiles: 4 });
      steps.push([step.removed, step.done]);
    }
    expect(steps).toEqual([
      [4, false],
      [4, false],
      [2, true],
    ]);
    expect(existsSync(a)).toBe(false);
  });

  it.runIf(DARWIN)('leaves a folder inside and says so', () => {
    const a = namespace(NS_A, { x: 3 });
    mkdirSync(join(a, '.blocked.x'));
    writeFileSync(join(a, '.blocked.x', 'inner'), 'keep');
    expect(removePrefixNamespaceStep(root, NS_A, openedOf(a))).toEqual({
      removed: 1,
      freedBytes: 3,
      done: true,
      left: 1,
      directoryRemoved: false,
    });
    expect(readdirSync(a)).toEqual(['.blocked.x']);
    expect(readdirSync(join(a, '.blocked.x'))).toEqual(['inner']);
    expect(scan().roots[0]!.namespaces).toEqual([]);
  });

  it('refuses where the lock cannot be checked', () => {
    const a = namespace(NS_A, { x: 3 });
    expect(caught(() => removePrefixNamespaceStep(root, NS_A, openedOf(a), { canLock: false })).kind).toBe(
      'no-lock-check',
    );
    expect(readdirSync(a).sort()).toEqual(['lock', 'x']);
  });

  it.runIf(DARWIN)('refuses a cache opened after it was listed', () => {
    const a = namespace(NS_A, { x: 3 });
    const base = Date.now() - 600_000;
    const listedAt = setTime(join(a, 'lock'), base);
    const listed = row(scan(), NS_A).openedMs;
    expect(listed).toBe(listedAt);
    setTime(join(a, 'lock'), base + 60_000);
    expect(caught(() => removePrefixNamespaceStep(root, NS_A, listed)).kind).toBe('changed');
    expect(readdirSync(a).sort()).toEqual(['lock', 'x']);
  });

  it.runIf(DARWIN)('refuses a folder without a regular lock file and never makes one', () => {
    const files: Record<string, number> = {};
    for (let i = 0; i < 10; i += 1) files[`f${String(i)}`] = 1;
    const a = namespace(NS_A, files, null);
    const refused = caught(() => removePrefixNamespaceStep(root, NS_A, 0, { maxFiles: 4 }));
    expect([refused.kind, refused.message]).toEqual([
      'not-a-cache',
      'This folder has no lock file, so it is not a model cache. Nothing was removed.',
    ]);
    expect(readdirSync(a).sort()).toEqual(Object.keys(files).sort());

    writeFileSync(join(tmp, 'elsewhere-lock'), '');
    const b = namespace(NS_B, { y: 1 }, null);
    symlinkSync(join(tmp, 'elsewhere-lock'), join(b, 'lock'));
    const c = namespace(NS_C, { z: 1 }, null);
    mkdirSync(join(c, 'lock'));
    for (const [name, dir] of [
      [NS_B, b],
      [NS_C, c],
    ] as const) {
      const err = caught(() => removePrefixNamespaceStep(root, name, 0));
      expect([err.kind, err.message], name).toEqual([
        'not-a-cache',
        'Its lock is a link or a folder, not a file, so this is not a model cache. Nothing was removed.',
      ]);
      expect(readdirSync(dir).sort()).toEqual(name === NS_B ? ['lock', 'y'] : ['lock', 'z']);
    }
    expect(existsSync(join(tmp, 'elsewhere-lock'))).toBe(true);
  });

  it.runIf(DARWIN)('stops when the folder is swapped for a link during a step', () => {
    const files: Record<string, number> = {};
    for (let i = 0; i < 6; i += 1) files[`f${String(i)}`] = 2;
    const a = namespace(NS_A, files);
    const outside = namespace('outside', files, 0, tmp);
    const outsideBefore = readdirSync(outside).sort();
    const moved = join(tmp, 'moved');
    let entries = 0;
    const swap = (): void => {
      entries += 1;
      if (entries !== 2) return;
      renameSync(a, moved);
      symlinkSync(outside, a);
    };
    const err = caught(() => removePrefixNamespaceStep(root, NS_A, openedOf(a), { beforeEntry: swap }));
    expect(err.kind).toBe('moved');
    expect(readdirSync(outside).sort()).toEqual(outsideBefore);
    expect(readdirSync(moved).length).toBe(Object.keys(files).length);
    expect(readdirSync(root)).toEqual([NS_A]);
  });

  it.runIf(DARWIN)('stops when the folder is swapped for another folder during a step', () => {
    const files: Record<string, number> = {};
    for (let i = 0; i < 6; i += 1) files[`f${String(i)}`] = 2;
    const a = namespace(NS_A, files);
    const other = namespace('other', files, 0, tmp);
    const otherBefore = readdirSync(other).sort();
    const moved = join(tmp, 'moved');
    let entries = 0;
    const swap = (): void => {
      entries += 1;
      if (entries !== 3) return;
      renameSync(a, moved);
      renameSync(other, a);
    };
    const err = caught(() => removePrefixNamespaceStep(root, NS_A, openedOf(a), { beforeEntry: swap }));
    expect(err.kind).toBe('moved');
    expect(readdirSync(a).sort()).toEqual(otherBefore);
    expect(readdirSync(moved).length).toBe(Object.keys(files).length - 1);
  });

  it.runIf(DARWIN)('a step past its deadline still removes one file', () => {
    const a = namespace(NS_A, { x: 1, y: 1 });
    for (const sub of ['d1', 'd2', 'd3']) mkdirSync(join(a, sub));
    const opened = openedOf(a);
    const steps = [];
    for (let i = 0; i < 3; i += 1) {
      const step = removePrefixNamespaceStep(root, NS_A, opened, { deadlineMs: 0 });
      steps.push([step.removed, step.done, step.done ? step.left : null]);
    }
    expect(steps).toEqual([
      [1, false, null],
      [1, false, null],
      [0, true, 3],
    ]);
  });
});

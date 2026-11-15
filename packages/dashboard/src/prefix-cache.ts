/**
 * The model caches under `prefix-cache/`: one directory per model identity, named by 32 hex digits, each with a `lock`
 * file that a running engine holds with `flock(2)`. Every engine open makes that file, so a directory without a
 * regular `lock` file is not a model cache: it is not listed and never deleted, and no `lock` is ever created here.
 *
 * Only names, sizes, times, owner ids and the lock state are read. No cache object is ever opened; the files opened
 * are the empty `lock` and the namespace directory itself. The rules mirror `crates/mlx-core/src/disk_tier.rs`:
 * `namespace_name`, `usage`, `idle_caches`, `take_lock` and `named`.
 */

import {
  closeSync,
  constants,
  type Dir,
  fstatSync,
  lstatSync,
  opendirSync,
  openSync,
  realpathSync,
  rmdirSync,
  type Stats,
  statfsSync,
  statSync,
  unlinkSync,
} from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

import type { HostKind } from './live.js';

export type PrefixLockState = 'held' | 'free' | 'unknown';

export interface PrefixCacheNamespace {
  /** The directory's name: 32 lowercase hex digits. */
  namespace: string;
  /** The model a running engine reports for this cache; null when no running engine reports it. */
  model: string | null;
  /** Regular files but `lock`, by size. A lower bound while `complete` is false. */
  bytes: number;
  files: number;
  /** Newest mtime of the files and of `lock`: the last open, write or restore. */
  lastUsedMs: number | null;
  /** The lock file's time: when an engine last opened this cache. A delete sends it back. */
  openedMs: number;
  /** False until a first walk of this directory reached its end. */
  complete: boolean;
  /** When the walk behind `bytes` and `files` ended. */
  measuredAtMs: number;
  lock: PrefixLockState;
}

export interface PrefixCacheEngine {
  pid: number;
  host: HostKind;
  /**
   * The limit of this engine's one open persistent state tier, as the tier set it at its open. Null when the engine
   * has none, loads more than one such model (its figure is then a sum), or does not say.
   */
  limitBytes: number | null;
}

export interface PrefixCacheRoot {
  /** As configured: the default, the runtime's option, or an engine's `--cache-dir`. */
  root: string;
  exists: boolean;
  /** Why the folder could not be read (an error code, or `ENOTDIR` for a path that is not a folder); null otherwise. */
  readError: string | null;
  isDefault: boolean;
  /** Sorted by name. */
  namespaces: PrefixCacheNamespace[];
  /** More than the cap of namespaces, or of root entries read, exist: the rest are not listed. */
  truncated: boolean;
  /** `statfs` of the root; null when missing or unreadable. */
  volume: { availableBytes: number; sizeBytes: number } | null;
  /** Live engines whose cache dir is this root. */
  engines: PrefixCacheEngine[];
}

export interface PrefixCacheResponse {
  roots: PrefixCacheRoot[];
  /** A first walk is unfinished: one more GET continues it. */
  counting: boolean;
  /** Whether this system can check and take a namespace lock. */
  canLock: boolean;
}

export interface PrefixCacheDeleteResult {
  removed: number;
  freedBytes: number;
  /** This step reached the directory's end. False: call again with the same body. */
  done: boolean;
  /** Subdirectories left inside (never entered). */
  left: number;
  directoryRemoved: boolean;
}

export type PrefixCacheErrorKind =
  | 'bad-name'
  | 'no-lock-check'
  | 'not-found'
  | 'not-a-cache'
  | 'held'
  | 'changed'
  | 'moved';

export class PrefixCacheError extends Error {
  readonly kind: PrefixCacheErrorKind;

  constructor(kind: PrefixCacheErrorKind, message: string) {
    super(message);
    this.name = 'PrefixCacheError';
    this.kind = kind;
  }
}

export const NAMESPACES_MAX = 256;
export const WALK_BUDGET = 20_000;
/** The walk's time per call: the database worker that runs it also serves every SQLite route. */
export const WALK_DEADLINE_MS = 250;
/** Entries of one root read per call: a heartbeat can name any folder. */
export const ROOT_ENTRIES_MAX = 4096;
export const OPEN_WALKS_MAX = 8;
const DELETE_FILES_MAX = 4096;
const DELETE_DEADLINE_MS = 1000;
const LOCK_FILE = 'lock';
const NAMESPACE_LEN = 32;

// `O_SHLOCK` / `O_EXLOCK` of darwin's <fcntl.h>: `open(2)` takes a `flock(2)` lock with the open. `fs.constants` has
// neither, and other systems have no such flags, so the lock can be checked on darwin only.
const O_SHLOCK = 0x10;
const O_EXLOCK = 0x20;

export function canLockNamespaces(): boolean {
  return process.platform === 'darwin';
}

/** `cache_root` of disk_tier.rs with no `--cache-dir`. */
export function defaultPrefixCacheRoot(): string {
  return join(homedir(), '.mlx-node', 'cache', 'prefix-cache');
}

/** `namespace_name` of disk_tier.rs: 32 lowercase hex digits. */
export function isNamespaceName(name: string): boolean {
  if (name.length !== NAMESPACE_LEN) return false;
  for (let i = 0; i < name.length; i += 1) {
    const code = name.charCodeAt(i);
    const digit = code >= 0x30 && code <= 0x39;
    const lower = code >= 0x61 && code <= 0x66;
    if (!digit && !lower) return false;
  }
  return true;
}

function errCode(err: unknown): string | undefined {
  return (err as NodeJS.ErrnoException | null)?.code;
}

function busy(err: unknown): boolean {
  const code = errCode(err);
  return code === 'EAGAIN' || code === 'EWOULDBLOCK';
}

function lstatOrUndefined(path: string): Stats | undefined {
  try {
    return lstatSync(path, { throwIfNoEntry: false });
  } catch {
    return undefined;
  }
}

function ownedByMe(stat: Stats): boolean {
  const euid = process.geteuid?.();
  return euid === undefined || stat.uid === euid;
}

/** A real directory of this user: what `idle_caches` lists. */
function isOwnDirectory(stat: Stats | undefined): stat is Stats {
  return stat !== undefined && stat.isDirectory() && !stat.isSymbolicLink() && ownedByMe(stat);
}

/** The state of `<dir>/lock`, by a shared non-blocking lock taken and let go at once. */
export function probeLock(dir: string, canLock: boolean = canLockNamespaces()): PrefixLockState {
  if (!canLock) return 'unknown';
  let fd: number | undefined;
  try {
    fd = openSync(join(dir, LOCK_FILE), constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK | O_SHLOCK);
    return 'free';
  } catch (err) {
    return busy(err) ? 'held' : 'unknown';
  } finally {
    if (fd !== undefined) closeSync(fd);
  }
}

const NO_LOCK = 'This folder has no lock file, so it is not a model cache. Nothing was removed.';
const LOCK_NOT_FILE = 'Its lock is a link or a folder, not a file, so this is not a model cache. Nothing was removed.';

/**
 * `take_lock` of disk_tier.rs: the exclusive lock of `<dir>/lock`. Never creates it: a folder without a regular `lock`
 * file is `not-a-cache`. The caller closes `fd` to let it go. Throws `held` when another holder has it.
 */
export function takeNamespaceLock(dir: string): number {
  const path = join(dir, LOCK_FILE);
  const lock = lstatOrUndefined(path);
  if (lock === undefined) throw new PrefixCacheError('not-a-cache', NO_LOCK);
  if (!lock.isFile()) throw new PrefixCacheError('not-a-cache', LOCK_NOT_FILE);
  try {
    return openSync(path, constants.O_RDWR | constants.O_NOFOLLOW | constants.O_NONBLOCK | O_EXLOCK);
  } catch (err) {
    const code = errCode(err);
    if (busy(err)) throw new PrefixCacheError('held', 'This cache is in use by a running engine.');
    if (code === 'ENOENT') throw new PrefixCacheError('not-a-cache', NO_LOCK);
    if (code === 'ELOOP' || code === 'EISDIR') throw new PrefixCacheError('not-a-cache', LOCK_NOT_FILE);
    throw new PrefixCacheError('held', `The lock of this cache could not be taken: ${String(code ?? err)}`);
  }
}

// ---------------------------------------------------------------- scan

interface WalkKey {
  dev: number;
  ino: number;
  mtimeMs: number;
  openedMs: number;
}

interface Usage {
  bytes: number;
  files: number;
  newest: number | null;
  measuredAtMs: number;
}

interface OpenWalk {
  dir: Dir;
  key: WalkKey;
  bytes: number;
  files: number;
  newest: number | null;
}

interface WalkEntry {
  done?: Usage;
  doneKey?: WalkKey;
  lockAtDone?: PrefixLockState;
  walk?: OpenWalk;
}

const walks = new Map<string, WalkEntry>();

function sameKey(a: WalkKey | undefined, b: WalkKey): boolean {
  return a !== undefined && a.dev === b.dev && a.ino === b.ino && a.mtimeMs === b.mtimeMs && a.openedMs === b.openedMs;
}

function closeWalk(entry: WalkEntry): void {
  const walk = entry.walk;
  entry.walk = undefined;
  if (walk === undefined) return;
  try {
    walk.dir.closeSync();
  } catch {
    // Already closed.
  }
}

function openWalks(): number {
  let open = 0;
  for (const entry of walks.values()) if (entry.walk !== undefined) open += 1;
  return open;
}

/** Close every open walk and forget every count. */
export function closePrefixCacheWalks(): void {
  for (const entry of walks.values()) closeWalk(entry);
  walks.clear();
}

function finishWalk(
  entry: WalkEntry,
  usage: Omit<Usage, 'measuredAtMs'>,
  key: WalkKey,
  lock: PrefixLockState,
  now: number,
): void {
  closeWalk(entry);
  entry.done = { ...usage, measuredAtMs: now };
  entry.doneKey = key;
  entry.lockAtDone = lock;
}

/**
 * `usage` of disk_tier.rs over at most `budget` entries, stopping after an entry once `late()` says the call's time is
 * up. Returns the entries read.
 */
function stepWalk(
  entry: WalkEntry,
  path: string,
  budget: number,
  lock: PrefixLockState,
  now: number,
  late: () => boolean,
): number {
  const walk = entry.walk!;
  let used = 0;
  while (used < budget && !(used > 0 && late())) {
    let item;
    try {
      item = walk.dir.readSync();
    } catch {
      item = null;
    }
    if (item === null) {
      finishWalk(entry, { bytes: walk.bytes, files: walk.files, newest: walk.newest }, walk.key, lock, now);
      return used;
    }
    used += 1;
    let stat: Stats;
    try {
      stat = lstatSync(join(path, item.name));
    } catch {
      continue;
    }
    if (!stat.isFile()) continue;
    walk.newest = walk.newest === null ? stat.mtimeMs : Math.max(walk.newest, stat.mtimeMs);
    if (item.name !== LOCK_FILE) {
      walk.bytes += stat.size;
      walk.files += 1;
    }
  }
  return used;
}

export interface PrefixScanRoot {
  root: string;
  isDefault: boolean;
  engines: PrefixCacheEngine[];
  /** Namespace → model name, as the running engines of this root report them. */
  models?: ReadonlyMap<string, string>;
}

export interface PrefixScanOptions {
  budget?: number;
  namespacesMax?: number;
  rootEntriesMax?: number;
  now?: number;
  /** The walk's time per call; it always reads at least one entry. */
  deadlineMs?: number;
  clock?: () => number;
  canLock?: boolean;
}

interface Pending {
  path: string;
  entry: WalkEntry;
  key: WalkKey;
  lock: PrefixLockState;
  fresh: boolean;
}

function volumeOf(root: string): PrefixCacheRoot['volume'] {
  try {
    const fs = statfsSync(root);
    return { availableBytes: fs.bavail * fs.bsize, sizeBytes: fs.blocks * fs.bsize };
  } catch {
    return null;
  }
}

function newestOf(a: number | null, b: number): number {
  return a === null ? b : Math.max(a, b);
}

/** The namespace names among the first `max` entries of `root`, sorted; `more` when the root has further entries. */
function rootNames(root: string, max: number): { names: string[]; more: boolean } {
  const dir = opendirSync(root);
  try {
    const names: string[] = [];
    for (let read = 0; read < max; read += 1) {
      const item = dir.readSync();
      if (item === null) return { names: names.sort(), more: false };
      if (isNamespaceName(item.name)) names.push(item.name);
    }
    return { names: names.sort(), more: dir.readSync() !== null };
  } finally {
    dir.closeSync();
  }
}

/**
 * The namespaces of each root with their bytes, files, last use and lock. A directory's walk reads at most `budget`
 * entries per call over all roots and is continued by the next call; an unchanged directory is not walked again.
 */
export function scanPrefixCaches(
  roots: readonly PrefixScanRoot[],
  opts: PrefixScanOptions = {},
): { roots: PrefixCacheRoot[]; counting: boolean; entriesRead: number } {
  const now = opts.now ?? Date.now();
  const clock = opts.clock ?? Date.now;
  const deadline = clock() + (opts.deadlineMs ?? WALK_DEADLINE_MS);
  const late = (): boolean => clock() >= deadline;
  const canLock = opts.canLock ?? canLockNamespaces();
  const namespacesMax = opts.namespacesMax ?? NAMESPACES_MAX;
  const rootEntriesMax = opts.rootEntriesMax ?? ROOT_ENTRIES_MAX;
  let budget = opts.budget ?? WALK_BUDGET;
  const listed = new Set<string>();
  const shapes: Array<{ root: PrefixScanRoot; out: PrefixCacheRoot; rows: Array<{ name: string; pending: Pending }> }> =
    [];
  const work: Pending[] = [];

  for (const root of roots) {
    const out: PrefixCacheRoot = {
      root: root.root,
      exists: false,
      readError: null,
      isDefault: root.isDefault,
      namespaces: [],
      truncated: false,
      volume: null,
      engines: root.engines,
    };
    const rows: Array<{ name: string; pending: Pending }> = [];
    shapes.push({ root, out, rows });
    let real: string;
    let names: string[];
    try {
      const rootStat = statSync(root.root, { throwIfNoEntry: false });
      if (rootStat === undefined) continue;
      out.exists = true;
      if (!rootStat.isDirectory()) {
        out.readError = 'ENOTDIR';
        continue;
      }
      out.volume = volumeOf(root.root);
      real = realpathSync(root.root);
      const read = rootNames(real, rootEntriesMax);
      names = read.names;
      out.truncated = read.more;
    } catch (err) {
      out.exists = true;
      out.readError = errCode(err) ?? 'EIO';
      continue;
    }
    if (names.length > namespacesMax) out.truncated = true;
    for (const name of names.slice(0, namespacesMax)) {
      const path = join(real, name);
      const stat = lstatOrUndefined(path);
      if (!isOwnDirectory(stat)) continue;
      const lockStat = lstatOrUndefined(join(path, LOCK_FILE));
      if (lockStat === undefined || !lockStat.isFile()) continue;
      const openedMs = lockStat.mtimeMs;
      const lock = probeLock(path, canLock);
      const key: WalkKey = { dev: stat.dev, ino: stat.ino, mtimeMs: stat.mtimeMs, openedMs };
      listed.add(path);
      let entry = walks.get(path);
      if (entry === undefined) {
        entry = {};
        walks.set(path, entry);
      }
      const fresh =
        entry.done !== undefined && sameKey(entry.doneKey, key) && !(entry.lockAtDone === 'held' && lock !== 'held');
      if (entry.walk !== undefined && (fresh || entry.walk.key.dev !== key.dev || entry.walk.key.ino !== key.ino)) {
        closeWalk(entry);
      }
      const pending: Pending = { path, entry, key, lock, fresh };
      rows.push({ name, pending });
      if (!fresh) work.push(pending);
    }
  }

  for (const [path, entry] of walks) {
    if (listed.has(path)) continue;
    closeWalk(entry);
    walks.delete(path);
  }

  const ordered = [
    ...work.filter((p) => p.entry.done === undefined),
    ...work.filter((p) => p.entry.done !== undefined),
  ];
  let entriesRead = 0;
  for (const pending of ordered) {
    if (budget <= 0 || (entriesRead > 0 && late())) break;
    const { entry } = pending;
    if (entry.walk === undefined) {
      if (openWalks() >= OPEN_WALKS_MAX) continue;
      let dir: Dir;
      try {
        dir = opendirSync(pending.path);
      } catch {
        finishWalk(entry, { bytes: 0, files: 0, newest: null }, pending.key, pending.lock, now);
        continue;
      }
      entry.walk = { dir, key: pending.key, bytes: 0, files: 0, newest: null };
    }
    const used = stepWalk(entry, pending.path, budget, pending.lock, now, late);
    budget -= used;
    entriesRead += used;
  }

  let counting = false;
  for (const { root, out, rows } of shapes) {
    for (const { name, pending } of rows) {
      const { entry, key, lock } = pending;
      const done = entry.done;
      const usage =
        done !== undefined
          ? done
          : {
              bytes: entry.walk?.bytes ?? 0,
              files: entry.walk?.files ?? 0,
              newest: entry.walk?.newest ?? null,
              measuredAtMs: now,
            };
      const complete = done !== undefined;
      if (!complete) counting = true;
      out.namespaces.push({
        namespace: name,
        model: root.models?.get(name) ?? null,
        bytes: usage.bytes,
        files: usage.files,
        lastUsedMs: newestOf(usage.newest, key.openedMs),
        openedMs: key.openedMs,
        complete,
        measuredAtMs: usage.measuredAtMs,
        lock,
      });
    }
  }
  return { roots: shapes.map((shape) => shape.out), counting, entriesRead };
}

// ---------------------------------------------------------------- delete

export interface PrefixDeleteOptions {
  maxFiles?: number;
  deadlineMs?: number;
  now?: () => number;
  canLock?: boolean;
  /** Tests only: runs before each entry of the step is looked at. */
  beforeEntry?: (name: string) => void;
}

function forgetWalk(path: string): void {
  const entry = walks.get(path);
  if (entry === undefined) return;
  closeWalk(entry);
  walks.delete(path);
}

const GONE = 'This cache is already gone.';
const MOVED = 'This cache folder was moved or replaced during the delete. The rest was left as it is.';

/**
 * One step of removing `<root>/<namespace>` under its lock: every entry directly inside that is not a directory, then
 * `lock`, then the directory itself. It never descends and never follows a link. Refused for a folder without a
 * regular `lock` file, while an engine holds the lock, and when the lock's time is not `openedMs`: every engine open
 * sets it (`open_persistent_dir`).
 *
 * Node has no `unlinkat`, so each removal names its path again. The step holds the directory open and, right before
 * every unlink and the final rmdir, checks that the path still names that same directory and not a link; a folder
 * swapped during the step stops it with `moved`. What is left is the time between that check and the one unlink.
 */
export function removePrefixNamespaceStep(
  root: string,
  namespace: string,
  openedMs: number,
  opts: PrefixDeleteOptions = {},
): PrefixCacheDeleteResult {
  const now = opts.now ?? Date.now;
  const maxFiles = opts.maxFiles ?? DELETE_FILES_MAX;
  const deadline = now() + (opts.deadlineMs ?? DELETE_DEADLINE_MS);
  if (!isNamespaceName(namespace)) throw new PrefixCacheError('bad-name', 'Not a model cache name.');
  if (!(opts.canLock ?? canLockNamespaces())) {
    throw new PrefixCacheError('no-lock-check', 'The lock cannot be checked on this system.');
  }
  let dir: string;
  try {
    dir = join(realpathSync(root), namespace);
  } catch {
    throw new PrefixCacheError('not-found', GONE);
  }
  const before = lstatOrUndefined(dir);
  if (!isOwnDirectory(before)) throw new PrefixCacheError('not-found', GONE);
  let dirFd: number;
  try {
    dirFd = openSync(dir, constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW);
  } catch {
    throw new PrefixCacheError('not-found', GONE);
  }
  let removed = 0;
  let freedBytes = 0;
  let left = 0;
  let done = false;
  let directoryRemoved = false;
  try {
    const anchor = fstatSync(dirFd);
    if (!isOwnDirectory(anchor) || anchor.dev !== before.dev || anchor.ino !== before.ino) {
      throw new PrefixCacheError('not-found', GONE);
    }
    /** The path still names the directory this step holds open, and not a link. */
    const anchored = (): void => {
      const at = lstatOrUndefined(dir);
      if (at === undefined || !at.isDirectory() || at.dev !== anchor.dev || at.ino !== anchor.ino) {
        throw new PrefixCacheError('moved', MOVED);
      }
    };
    const fd = takeNamespaceLock(dir);
    const lockPath = join(dir, LOCK_FILE);
    try {
      const held = fstatSync(fd);
      const named = (): boolean => {
        const at = lstatOrUndefined(lockPath);
        return at !== undefined && at.dev === held.dev && at.ino === held.ino;
      };
      anchored();
      if (!named()) throw new PrefixCacheError('held', 'This cache is in use by a running engine.');
      if (held.mtimeMs !== openedMs) {
        throw new PrefixCacheError(
          'changed',
          'An engine opened this cache after it was listed. Refresh and look again.',
        );
      }
      const listing = opendirSync(dir);
      try {
        anchored();
        for (;;) {
          const item = listing.readSync();
          if (item === null) {
            done = true;
            break;
          }
          if (item.name === LOCK_FILE) continue;
          opts.beforeEntry?.(item.name);
          const path = join(dir, item.name);
          let stat: Stats;
          try {
            stat = lstatSync(path);
          } catch {
            continue;
          }
          if (stat.isDirectory()) {
            left += 1;
            continue;
          }
          anchored();
          try {
            unlinkSync(path);
          } catch {
            continue;
          }
          removed += 1;
          if (stat.isFile()) freedBytes += stat.size;
          if (removed >= maxFiles || now() >= deadline) break;
        }
        if (done) {
          anchored();
          if (!named()) throw new PrefixCacheError('moved', MOVED);
          unlinkSync(lockPath);
          anchored();
          try {
            rmdirSync(dir);
            directoryRemoved = true;
          } catch (err) {
            if (errCode(err) !== 'ENOTEMPTY' && errCode(err) !== 'EEXIST') throw err;
          }
        }
      } finally {
        listing.closeSync();
      }
    } finally {
      closeSync(fd);
    }
  } finally {
    closeSync(dirFd);
  }
  forgetWalk(dir);
  if (!done && removed === 0) throw new Error('A delete step removed nothing and did not reach the end.');
  return { removed, freedBytes, done, left, directoryRemoved };
}

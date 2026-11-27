/**
 * Work without a folder: each such conversation gets a new empty folder of its own under the scratch root
 * (`~/Portato`), named by the local minute of its first send. The app records each folder it makes
 * (`scratch-folders.json`: real path, device and inode) and removes one only when it is empty (`rmdir`, never
 * recursive): it never deletes a file in them. A folder with no record, or a new folder at a recorded path, is not the
 * app's.
 */

import { readFileSync, statSync } from 'node:fs';
import { mkdir, realpath, rmdir, stat } from 'node:fs/promises';
import { dirname, join } from 'node:path';

import { writeFileAtomic } from './paths.js';

/** The scratch root's name in the home folder: the app's name, so the user sees who made it. */
export const SCRATCH_FOLDER = 'Portato';

/** Names tried in one minute: the stamp, then `-2` to `-9`. */
const NAMES_PER_MINUTE = 9;

/** The longest file name the agent can make: it keeps a folder's conversations under its encoded path. */
const AGENT_NAME_BYTES = 255;

export function scratchRoot(home: string): string {
  return join(home, SCRATCH_FOLDER);
}

function two(n: number): string {
  return String(n).padStart(2, '0');
}

/** `YYYY-MM-DD-HHmm` in local time. */
export function stampName(ms: number): string {
  const at = new Date(ms);
  const day = `${at.getFullYear()}-${two(at.getMonth() + 1)}-${two(at.getDate())}`;
  return `${day}-${two(at.getHours())}${two(at.getMinutes())}`;
}

/**
 * Makes a new folder under `root` (the root itself when missing, mode 0700; an existing root is left as it is): the
 * stamp of `ms`, else `-2` to `-9`. The real path of the new folder, or null when none could be made.
 */
export async function makeScratch(root: string, ms: number): Promise<string | null> {
  let top: string;
  try {
    await mkdir(root, { recursive: true, mode: 0o700 });
    top = await realpath(root);
  } catch {
    return null;
  }
  const stamp = stampName(ms);
  for (let n = 1; n <= NAMES_PER_MINUTE; n += 1) {
    const path = join(top, n === 1 ? stamp : `${stamp}-${n}`);
    if (encodeURIComponent(path).length > AGENT_NAME_BYTES) return null;
    try {
      await mkdir(path, { mode: 0o700 });
      return path;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST') return null;
    }
  }
  return null;
}

/** A folder the app made: its real path, and the device and inode numbers it had when made (decimal text). */
export interface ScratchRecord {
  path: string;
  dev: string;
  ino: string;
}

type Identity = Pick<ScratchRecord, 'dev' | 'ino'>;

function identityOf(info: { dev: bigint; ino: bigint; isDirectory(): boolean }): Identity | null {
  return info.isDirectory() ? { dev: String(info.dev), ino: String(info.ino) } : null;
}

function identityNow(path: string): Identity | null {
  try {
    return identityOf(statSync(path, { bigint: true }));
  } catch {
    return null;
  }
}

/** The records of the file by path; a missing or broken file has none. */
export function readScratchRecordsSync(file: string): Map<string, ScratchRecord> {
  const out = new Map<string, ScratchRecord>();
  let raw: unknown;
  try {
    raw = JSON.parse(readFileSync(file, 'utf8'));
  } catch {
    return out;
  }
  if (typeof raw !== 'object' || raw === null) return out;
  const record = raw as { version?: unknown; folders?: unknown };
  if (record.version !== 1 || !Array.isArray(record.folders)) return out;
  for (const item of record.folders as unknown[]) {
    if (typeof item !== 'object' || item === null) continue;
    const { path, dev, ino } = item as { path?: unknown; dev?: unknown; ino?: unknown };
    if (typeof path !== 'string' || path === '' || typeof dev !== 'string' || typeof ino !== 'string') continue;
    out.set(path, { path, dev, ino });
  }
  return out;
}

/** The app made `cwd`: a record names it and the folder there is still the one it made. */
export function isOwned(cwd: string, records: ReadonlyMap<string, ScratchRecord>): boolean {
  const record = records.get(cwd);
  if (record === undefined) return false;
  const now = identityNow(cwd);
  return now !== null && now.dev === record.dev && now.ino === record.ino;
}

/** The records of the folders the app made: read once, the file written whole (mode 0600) after each change. */
export class ScratchFolders {
  readonly file: string;
  private readonly records: Map<string, ScratchRecord>;
  private chain: Promise<void> = Promise.resolve();

  constructor(file: string) {
    this.file = file;
    this.records = readScratchRecordsSync(file);
  }

  owns(cwd: string): boolean {
    return isOwned(cwd, this.records);
  }

  /** Records a folder just made; a write that fails takes the record out again and throws. */
  async add(path: string): Promise<void> {
    const identity = identityOf(await stat(path, { bigint: true }));
    if (identity === null) throw new Error('not a folder');
    this.records.set(path, { path, ...identity });
    try {
      await this.write();
    } catch (error) {
      this.records.delete(path);
      throw error;
    }
  }

  /** Forgets a folder the app removed; a write that fails leaves a record whose folder is gone. */
  async drop(path: string): Promise<void> {
    if (!this.records.delete(path)) return;
    await this.write().catch(() => {});
  }

  private write(): Promise<void> {
    const run = async (): Promise<void> => {
      await mkdir(dirname(this.file), { recursive: true, mode: 0o700 });
      await writeFileAtomic(this.file, `${JSON.stringify({ version: 1, folders: [...this.records.values()] })}\n`);
    };
    const next = this.chain.then(run, run);
    this.chain = next.catch(() => {});
    return next;
  }
}

/** Removes the folder only when it is empty. True when it was removed. */
export async function removeIfEmpty(path: string): Promise<boolean> {
  try {
    await rmdir(path);
    return true;
  } catch {
    return false;
  }
}

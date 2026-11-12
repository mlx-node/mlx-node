/**
 * The Work folder rule. The path comes from the page (a folder the user picked) or from a session's stored summary,
 * so it is untrusted: it must be an absolute, existing, readable folder, not the disk's top, not the home folder or a
 * folder that holds it, and neither inside nor holding the app's or the agent's data. The resolved path is what
 * reaches the agent.
 */

import { constants } from 'node:fs';
import { access, lstat, readdir, readlink, realpath, stat } from 'node:fs/promises';
import { basename, dirname, isAbsolute, join, resolve } from 'node:path';

import { ApiError } from '../api/errors.js';
import type { CwdProblem } from './types.js';

/** The fixed sentence of each refusal. */
export const CWD_PROBLEM_MESSAGES: Readonly<Record<CwdProblem, string>> = {
  'not-absolute': 'Choose the folder by its full path.',
  missing: 'That folder does not exist any more.',
  'not-directory': 'That is not a folder.',
  unreadable: 'That folder cannot be read.',
  root: 'The top of the disk cannot be a work folder.',
  home: 'Your home folder cannot be a work folder. Choose a folder inside it.',
  'app-data': 'That folder holds app or agent data.',
};

export interface WorkFolderRules {
  home: string;
  /** Folders that may not hold a work folder: the app data, the agent home, the Keychains folder. */
  refused: readonly string[];
}

export type WorkFolderCheck = { ok: true; path: string } | { ok: false; problem: CwdProblem };

/** A folder's identity on disk: the same for every path that reaches it, firmlinks such as `/System/Volumes/Data` included. */
async function identity(path: string): Promise<string | null> {
  try {
    const info = await stat(path, { bigint: true });
    return `${info.dev}:${info.ino}`;
  } catch {
    return null;
  }
}

/** The identities of `path` and of every folder above it, `path` first. */
async function chain(path: string): Promise<string[]> {
  const out: string[] = [];
  let current = path;
  for (;;) {
    const id = await identity(current);
    if (id !== null) out.push(id);
    const parent = dirname(current);
    if (parent === current) return out;
    current = parent;
  }
}

/** The identities of every folder above `folder` as written and above where its links lead, `folder` first. */
async function holders(folder: string): Promise<string[]> {
  const out = await chain(folder);
  const real = await realpath(folder).catch(() => null);
  if (real !== null) out.push(...(await chain(real)));
  return out;
}

/**
 * The folder may not be the home folder, hold it, or lie inside or hold a refused folder, nor be picked through a path
 * that runs through a refused folder. Folders are compared by identity, not by path text, so another path to the same
 * folder is refused too.
 */
async function placeProblem(path: string, real: string, rules: WorkFolderRules): Promise<CwdProblem | null> {
  const self = await identity(real);
  const above = await chain(real);
  const written = await chain(resolve(path));
  const home = await identity(rules.home);
  if (home !== null && self === home) return 'home';
  if (self !== null && (await holders(rules.home)).includes(self)) return 'home';
  for (const folder of rules.refused) {
    const id = await identity(folder);
    if (id !== null && (above.includes(id) || written.includes(id))) return 'app-data';
    if (self !== null && (await holders(folder)).includes(self)) return 'app-data';
  }
  return null;
}

/** The top of the macOS data volume, which holds every home folder. */
const DATA_VOLUME = '/System/Volumes/Data';

export async function checkWorkFolder(path: string, rules: WorkFolderRules): Promise<WorkFolderCheck> {
  if (!isAbsolute(path)) return { ok: false, problem: 'not-absolute' };
  let real: string;
  try {
    real = await realpath(path);
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    return { ok: false, problem: code === 'ENOENT' || code === 'ENOTDIR' ? 'missing' : 'unreadable' };
  }
  try {
    if (!(await stat(real)).isDirectory()) return { ok: false, problem: 'not-directory' };
    await access(real, constants.R_OK | constants.X_OK);
  } catch {
    return { ok: false, problem: 'unreadable' };
  }
  if (real === '/' || dirname(real) === real) return { ok: false, problem: 'root' };
  const volume = await identity(DATA_VOLUME);
  if (volume !== null && (await identity(real)) === volume) return { ok: false, problem: 'root' };
  const problem = await placeProblem(path, real, rules);
  return problem === null ? { ok: true, path: real } : { ok: false, problem };
}

function errorCode(error: unknown): string | undefined {
  return (error as NodeJS.ErrnoException | null)?.code;
}

/**
 * Where a write to `path` (absolute, `..` not collapsed) really lands once the disk follows its links: the real path of
 * its deepest existing part plus the rest, which the agent creates as folders. A dangling link is followed to its
 * target, a relative target read from the link's own folder. Null when it cannot be known. Only ENOENT and ENOTDIR
 * (and ELOOP from a link loop, which is then followed until the step budget ends) read as "not there"; any other
 * error throws, so the caller asks.
 */
export async function landingPath(path: string): Promise<string | null> {
  let rest: string[] = [];
  let current = path;
  for (let step = 0; step < 512; step++) {
    try {
      const real = await realpath(current);
      if (rest.length === 0) return real;
      const joined = join(real, ...rest);
      if (!rest.includes('..')) return joined;
      // A `..` past the created folders leads back to ones on disk, whose links the write follows.
      current = joined;
      rest = [];
      continue;
    } catch (error) {
      const code = errorCode(error);
      if (code !== 'ENOENT' && code !== 'ENOTDIR' && code !== 'ELOOP') throw error;
    }
    let info: Awaited<ReturnType<typeof lstat>> | null = null;
    try {
      info = await lstat(current);
    } catch (error) {
      const code = errorCode(error);
      if (code === 'ELOOP') return null;
      if (code !== 'ENOENT' && code !== 'ENOTDIR') throw error;
    }
    if (info !== null) {
      if (!info.isSymbolicLink()) return null;
      const target = await readlink(current);
      current = isAbsolute(target) ? target : `${dirname(current)}/${target}`;
      continue;
    }
    const parent = dirname(current);
    if (parent === current) return null;
    rest = [basename(current), ...rest];
    current = parent;
  }
  return null;
}

/** A file name as the agent compares it when the named file is missing: U+00A0 and U+202F read as a space. */
function spacedName(name: string): string {
  let out = '';
  for (const c of name) out += c === '\u00a0' || c === '\u202f' ? ' ' : c;
  return out.normalize('NFC');
}

/**
 * Where an edit of `path` lands, as `landingPath`; null also when the agent can edit another file. When no entry of the
 * folder has the written name, the agent edits the one entry whose name differs only by U+00A0 or U+202F against a
 * space (`try_resolve_unicode_filename`): any such entry makes the landing unknown.
 */
export async function editLanding(path: string): Promise<string | null> {
  const landing = await landingPath(path);
  if (landing === null) return null;
  const name = basename(path);
  let entries: string[];
  try {
    entries = await readdir(dirname(path));
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    return code === 'ENOENT' || code === 'ENOTDIR' ? landing : null;
  }
  if (entries.includes(name)) return landing;
  const wanted = spacedName(name);
  return entries.some((entry) => spacedName(entry) === wanted) ? null : landing;
}

/** The route failure of a refused folder: `E_BAD_REQUEST`, reason `cwd-invalid`, the problem's fixed sentence. */
export function cwdError(problem: CwdProblem): ApiError {
  return new ApiError('E_BAD_REQUEST', CWD_PROBLEM_MESSAGES[problem], 'cwd-invalid');
}

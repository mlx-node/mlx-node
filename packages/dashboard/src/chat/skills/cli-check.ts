/**
 * Whether a command is on a `PATH`: each `<dir>/<name>` is tested for a regular file the user may execute. Node only:
 * no shell, nothing is run.
 */

import { constants } from 'node:fs';
import { access, stat } from 'node:fs/promises';
import { isAbsolute, join, win32 } from 'node:path';

const DEFAULT_WIN_PATHEXT = '.COM;.EXE;.BAT;.CMD';

export interface FindOnPathOptions {
  platform?: NodeJS.Platform;
  pathExt?: string;
}

/** Case-insensitive `PATH`/`Path` lookup for process environments (Windows uses `Path`). */
export function readPathEnv(env: NodeJS.ProcessEnv): string | undefined {
  if (env.PATH !== undefined) return env.PATH;
  for (const [key, value] of Object.entries(env)) {
    if (key.toLowerCase() === 'path' && value !== undefined) return value;
  }
  return undefined;
}

function readPathExtEnv(env: NodeJS.ProcessEnv): string | undefined {
  if (env.PATHEXT !== undefined) return env.PATHEXT;
  for (const [key, value] of Object.entries(env)) {
    if (key.toLowerCase() === 'pathext' && value !== undefined) return value;
  }
  return undefined;
}

function windowsCandidates(name: string, rawPathExt: string): string[] {
  const exts = rawPathExt
    .split(';')
    .map((entry) => entry.trim())
    .filter((entry) => entry !== '')
    .map((entry) => (entry.startsWith('.') ? entry : `.${entry}`));
  if (exts.length === 0) return [name];
  const lowerName = name.toLowerCase();
  const candidates: string[] = [];
  const push = (candidate: string): void => {
    if (!candidates.includes(candidate)) candidates.push(candidate);
  };
  if (exts.some((ext) => lowerName.endsWith(ext.toLowerCase()))) {
    push(name);
  }
  for (const ext of exts) {
    const lowerExt = ext.toLowerCase();
    push(`${name}${lowerExt}`);
    if (ext !== lowerExt) push(`${name}${ext}`);
  }
  return candidates;
}

/** The first `<dir>/<name>` of `path` that is an executable regular file, or null. Relative entries are skipped. */
export async function findOnPath(
  name: string,
  path: string | undefined,
  options: FindOnPathOptions = {},
): Promise<string | null> {
  const platform = options.platform ?? process.platform;
  if (platform === 'win32') {
    if (name === '' || name.includes('/') || name.includes('\\') || name.includes(':') || path === undefined) {
      return null;
    }
    const rawExt =
      options.pathExt ??
      (platform === process.platform ? (readPathExtEnv(process.env) ?? DEFAULT_WIN_PATHEXT) : DEFAULT_WIN_PATHEXT);
    const names = windowsCandidates(name, rawExt);
    for (const rawDir of path.split(';')) {
      let dir = rawDir.trim();
      if (dir.length >= 2 && dir.startsWith('"') && dir.endsWith('"')) dir = dir.slice(1, -1);
      if (dir === '' || !win32.isAbsolute(dir)) continue;
      for (const file of names) {
        const candidate = join(dir, file);
        try {
          if (!(await stat(candidate)).isFile()) continue;
          await access(candidate, constants.F_OK);
          return candidate;
        } catch {
          // Not there, or not readable.
        }
      }
    }
    return null;
  }
  if (name === '' || name.includes('/') || path === undefined) return null;
  for (const dir of path.split(':')) {
    if (dir === '' || !isAbsolute(dir)) continue;
    const candidate = join(dir, name);
    try {
      if (!(await stat(candidate)).isFile()) continue;
      await access(candidate, constants.X_OK);
      return candidate;
    } catch {
      // Not there, or not executable.
    }
  }
  return null;
}

/**
 * The places Auto may answer for: the Work folder, and the temp roots the host gives as data. A path is placed by text
 * ({@link rootText}) and by where it lands on disk ({@link rootOf}); the Work folder always comes first.
 */

import type { WorkPlace } from './command-flags.js';

export interface AllowedRoot {
  /** Real path of the root's top folder. */
  top: string;
  /** Text spellings of the top a path may start with, the real one first: `['/private/tmp', '/tmp']`. */
  spellings: readonly string[];
  /** `work`: the Work folder. `temp`: shared by every program; the unit rules apply to every place in it. */
  kind: 'work' | 'temp';
}

/** `path` is `top` or below it, by text. */
export function below(path: string, top: string): boolean {
  return path === top || path.startsWith(top.endsWith('/') ? top : `${top}/`);
}

/**
 * The roots of a place: the Work folder first (with the spellings a temp root gives a Work folder under it), then the
 * temp roots that neither equal the Work folder nor lie inside it.
 */
export function roots(place: WorkPlace): AllowedRoot[] {
  const cwd = place.cwd;
  const temps = (place.temp ?? []).filter((t) => t.kind === 'temp' && !below(t.top, cwd));
  const spellings: string[] = [cwd];
  for (const t of temps) {
    if (!below(cwd, t.top)) continue;
    for (const s of t.spellings) {
      if (s === t.top) continue;
      const alias = `${s}${cwd.slice(t.top.length)}`;
      if (!spellings.includes(alias)) spellings.push(alias);
    }
  }
  return [{ top: cwd, spellings, kind: 'work' }, ...temps];
}

/** The temp roots of a place, as {@link roots} keeps them. */
export function tempRoots(place: WorkPlace): AllowedRoot[] {
  return roots(place).filter((r) => r.kind === 'temp');
}

/**
 * The first root a path names by text, and the path with the root's real top in place of the spelling; null in no root.
 * Exact text: no fold, no case change.
 */
export function rootText(path: string, place: WorkPlace): { root: AllowedRoot; path: string } | null {
  for (const root of roots(place)) {
    for (const s of root.spellings) {
      if (below(path, s)) return { root, path: `${root.top}${path.slice(s.length)}` };
    }
  }
  return null;
}

/** The first root a real path lies in, by tops only; `/dev` is in none. */
export function rootOf(real: string, place: WorkPlace): AllowedRoot | null {
  if (below(real, '/dev')) return null;
  for (const root of roots(place)) if (below(real, root.top)) return root;
  return null;
}

/** A real path under a temp root with the spelling people type (the root's last spelling) in place of its top. */
export function typedPath(real: string, place: WorkPlace): string {
  const root = rootOf(real, place);
  if (root === null || root.kind !== 'temp') return real;
  const typed = root.spellings[root.spellings.length - 1] ?? root.top;
  return `${typed}${real.slice(root.top.length)}`;
}

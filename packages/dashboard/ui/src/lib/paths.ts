/**
 * `path` with the home folder written as `~`, the way a shell prompt shows it.
 * Only a whole leading folder matches: `/Users/ann2/x` is not under `/Users/ann`.
 */
export function tildePath(path: string, home: string | null): string {
  if (home === null || !home.startsWith('/')) return path;
  let end = home.length;
  while (end > 1 && home[end - 1] === '/') end -= 1;
  const root = home.slice(0, end);
  if (root === '/') return path;
  if (path === root) return '~';
  if (path.startsWith(`${root}/`)) return `~${path.slice(root.length)}`;
  return path;
}

/** The scratch root's `~` form while the host state is not known; the host makes it `join(home, 'Portato')`. */
const SCRATCH_SHOWN = '~/Portato';

/** Where Work without a folder makes its folders, as people read it (`{root}` of the strings). */
export function scratchShown(root: string | null | undefined, home: string | null): string {
  return root === null || root === undefined ? SCRATCH_SHOWN : tildePath(root, home);
}

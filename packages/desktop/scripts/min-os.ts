/**
 * The macOS floor the bundle has to advertise.
 *
 * Release builds set `MACOSX_DEPLOYMENT_TARGET=26.0`, so the addon's
 * `LC_BUILD_VERSION` carries `minos 26.0` and dyld refuses to map it on anything
 * older. `@electron/packager` writes whatever Electron's own Info.plist template
 * says — 12.0, verified on a real build — so without this the app LAUNCHES on
 * macOS 12-15 and only then fails to load the addon. The supervisor reads that as
 * a crashed sidecar and restarts it, forever, with no message naming the cause.
 *
 * Measured off the Mach-O headers rather than read from `MACOSX_DEPLOYMENT_TARGET`:
 * the env var is an intent that a build can ignore, `LC_BUILD_VERSION` is what
 * actually shipped, and only the second one is what dyld enforces. It also means a
 * local build (addon `minos 11.0`) honestly advertises Electron's 12.0 instead of
 * a release floor it does not have.
 */

/**
 * Highest `minos` in `otool -l` / `vtool -show-build` output, or `null` when the
 * binary carries no `LC_BUILD_VERSION` at all.
 *
 * Highest rather than first: a universal binary prints one load-command block per
 * architecture, and they do not have to agree. Taking the first would pick
 * whichever slice `otool` happened to print first.
 */
export function parseMinOs(output: string): string | null {
  // `[ \t]` rather than `\s`: `\s` spans newlines, and a pattern that can cross a
  // line boundary would happily read the number off the FOLLOWING field.
  const found = [...output.matchAll(/^[ \t]*minos[ \t]+(\d+(?:\.\d+)*)[ \t]*$/gm)].map((m) => m[1]);
  return found.length === 0 ? null : maxOsVersion(found);
}

/** Component-wise numeric compare. Missing components are 0, so `12` === `12.0`. */
export function compareOsVersions(a: string, b: string): number {
  const left = a.split('.').map(Number);
  const right = b.split('.').map(Number);
  for (let i = 0; i < Math.max(left.length, right.length); i++) {
    const diff = (left[i] ?? 0) - (right[i] ?? 0);
    if (diff !== 0) return diff < 0 ? -1 : 1;
  }
  return 0;
}

/**
 * The floor is the MAXIMUM, not the minimum: every binary in the bundle has to
 * load, so the bundle can only run where the most demanding one runs.
 */
export function maxOsVersion(versions: readonly string[]): string {
  if (versions.length === 0) {
    throw new Error('maxOsVersion: no versions — nothing carried LC_BUILD_VERSION');
  }
  return versions.reduce((best, v) => (compareOsVersions(v, best) > 0 ? v : best));
}

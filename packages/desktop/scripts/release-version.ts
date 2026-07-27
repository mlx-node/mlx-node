/**
 * Version agreement across a desktop release.
 *
 * `tools bump` (packages/internal-tools/index.ts) bumps
 * `project.workspaces.filter((pkg) => pkg.manifest.private !== true)`, and
 * `@mlx-node/desktop` is `private: true` — it must never reach npm. So the desktop
 * manifest does NOT move when the repo version does. After `0.0.8`, a tagged
 * `v0.0.9` release would package a `0.0.8` app, name the DMG `0.0.8` and bump the
 * Homebrew cask to `0.0.8` — and nothing downstream would notice, because all
 * three would be consistently wrong. They all read the same stale manifest.
 *
 * The release tag is the source of truth. Everything the build produces is stamped
 * from it and then read BACK off the real artifact and compared, because "we
 * passed the right value to packager" and "the bundle says the right thing" are
 * different claims — the first is an intent, only the second ships.
 */

/**
 * semver core plus optional prerelease/build. Deliberately strict about leading
 * zeros: `0.01.0` and `0.1.0` would sort as one version and name two DMGs.
 */
const SEMVER = /^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$/;

export class ReleaseVersionError extends Error {
  constructor(
    message: string,
    readonly remedy: string,
  ) {
    super(message);
    this.name = 'ReleaseVersionError';
  }
}

/** Throws unless `value` is a semver string. Returns it, so it can wrap an expression. */
export function assertSemver(label: string, value: string): string {
  if (!SEMVER.test(value)) {
    throw new ReleaseVersionError(
      `${label} is not a version: ${JSON.stringify(value)}`,
      'expected MAJOR.MINOR.PATCH, optionally -prerelease',
    );
  }
  return value;
}

/**
 * `tools bump` tags `v${version}`, so the `v` is the normal case rather than the
 * tolerated one. A bare version is accepted too — a hand-cut tag is exactly when
 * this runs, and rejecting `0.0.9` would be pedantry that blocks a release.
 */
export function versionFromTag(tag: string): string {
  const trimmed = tag.trim();
  if (trimmed === '') {
    throw new ReleaseVersionError('empty release tag', 'expected a tag like v0.0.9');
  }
  const stripped = /^v/i.test(trimmed) ? trimmed.slice(1) : trimmed;
  if (!SEMVER.test(stripped)) {
    throw new ReleaseVersionError(
      `release tag ${JSON.stringify(tag)} is not a version`,
      'expected a tag like v0.0.9 — `tools bump <major|minor|patch>` produces one',
    );
  }
  return stripped;
}

/**
 * The one place the DMG filename is spelled. The workflow builds the name from
 * here and then parses it back with {@link versionFromDmgPath}, so a change to
 * either half that is not mirrored fails the gate instead of shipping a DMG whose
 * name disagrees with its contents.
 */
export function dmgFileName(version: string): string {
  return `mlx-node-${assertSemver('version', version)}-arm64.dmg`;
}

/** The inverse of {@link dmgFileName}. `null` when the name is not one of ours. */
export function versionFromDmgPath(path: string): string | null {
  const base = path.split('/').pop() ?? path;
  const match = /^mlx-node-(.+)-arm64\.dmg$/.exec(base);
  if (match === null) return null;
  return SEMVER.test(match[1]) ? match[1] : null;
}

export interface ReleaseVersions {
  /** From the release tag. `null` on `workflow_dispatch`, where there is no tag. */
  tag: string | null;
  /** `packages/desktop/package.json` — the one `tools bump` skips. */
  manifest: string;
  /** `CFBundleShortVersionString`, read back off the built `.app`. */
  bundleShort: string;
  /** `CFBundleVersion`, read back off the built `.app`. */
  bundleVersion: string;
  /** Parsed out of the DMG filename. `null` before the DMG exists. */
  dmg: string | null;
}

/**
 * Fail unless the tag, the manifest, the bundle's own metadata and the DMG name
 * are all the same version. Returns that version.
 *
 * The tag wins when present because it is the only one of the four that reliably
 * advances; the manifest is still required to match rather than being silently
 * overridden, since an artifact that disagrees with the repo it was built from is
 * a provenance bug even when the artifact is the correct one.
 */
export function assertVersionsAgree(versions: ReleaseVersions): string {
  assertSemver('packages/desktop/package.json version', versions.manifest);

  const expected = versions.tag === null ? versions.manifest : assertSemver('release tag version', versions.tag);
  const source = versions.tag === null ? 'packages/desktop/package.json (no release tag)' : 'the release tag';

  const checks: { label: string; value: string | null; remedy: string }[] = [
    {
      label: 'packages/desktop/package.json version',
      value: versions.manifest,
      // The trap this whole module exists for. Name the cause, not just the diff.
      remedy:
        '`tools bump` skips private workspaces and @mlx-node/desktop is private — ' +
        'set packages/desktop/package.json "version" to match the tag, then re-tag',
    },
    {
      label: 'CFBundleShortVersionString',
      value: versions.bundleShort,
      remedy: 'the packaged bundle was not stamped — check `package --app-version`',
    },
    {
      label: 'CFBundleVersion',
      value: versions.bundleVersion,
      remedy: 'the packaged bundle was not stamped — check `package --app-version`',
    },
    { label: 'DMG filename', value: versions.dmg, remedy: 'the DMG was named from a different version than it holds' },
  ];

  for (const check of checks) {
    if (check.value === null) continue;
    if (check.value !== expected) {
      throw new ReleaseVersionError(`${check.label} is ${check.value}, but ${source} says ${expected}`, check.remedy);
    }
  }

  return expected;
}

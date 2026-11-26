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
  return `Portato-${assertSemver('version', version)}-arm64.dmg`;
}

/** The inverse of {@link dmgFileName}. `null` when the name is not one of ours. */
export function versionFromDmgPath(path: string): string | null {
  const base = artifactBaseName(path);
  const match = /^Portato-(.+)-arm64\.dmg$/.exec(base);
  if (match === null) return null;
  return SEMVER.test(match[1]) ? match[1] : null;
}

/** Include architecture for electron-updater and versions for blockmap lookup. */
export function updateZipFileName(version: string): string {
  return `mlx-node-${assertSemver('version', version)}-darwin-arm64.zip`;
}

export function versionFromUpdateZipPath(path: string): string | null {
  const match = /^mlx-node-(.+)-darwin-arm64\.zip$/.exec(artifactBaseName(path));
  return match !== null && SEMVER.test(match[1]) ? match[1] : null;
}

/** Unsigned portable Windows installation archive. */
export function windowsZipFileName(version: string): string {
  return `Portato-${assertSemver('version', version)}-win32-arm64.zip`;
}

export function versionFromWindowsZipPath(path: string): string | null {
  const match = /^Portato-(.+)-win32-arm64\.zip$/.exec(artifactBaseName(path));
  return match !== null && SEMVER.test(match[1]) ? match[1] : null;
}

/** Unsigned Windows ARM64 MSIX package for Microsoft Store submission. */
export function windowsMsixFileName(version: string): string {
  return `Portato-${assertSemver('version', version)}-win32-arm64.msix`;
}

export function versionFromWindowsMsixPath(path: string): string | null {
  const match = /^Portato-(.+)-win32-arm64\.msix$/.exec(artifactBaseName(path));
  return match !== null && SEMVER.test(match[1]) ? match[1] : null;
}

const MSIX_MAX_PART = 65535;
const MSIX_MAX_SEMVER_MAJOR = MSIX_MAX_PART - 1;

/**
 * Convert a stable semver `MAJOR.MINOR.PATCH` into the four-part MSIX identity
 * version `(MAJOR + 1).MINOR.PATCH.0` required by the Microsoft Store.
 *
 * Microsoft Store Partner Center rejects MSIX packages whose `<Identity Version>`
 * starts with `0.` (Major must be `1..65535`, Minor and Build `0..65535`, and
 * Revision `0` reserved for the Store), forbids prerelease identifiers, and
 * forbids build metadata. Adding `1` to Major keeps Store versions strictly
 * monotonic across `0.x.y` -> `1.x.y` -> `N.x.y` without ever decreasing
 * (`0.0.21` -> `1.0.21.0`, `0.1.0` -> `1.1.0.0`, `1.0.0` -> `2.0.0.0`), while
 * capping semver Major at `65534` so `major + 1 <= 65535`.
 */
export function msixPackageVersion(version: string): string {
  assertSemver('version', version);
  if (version.includes('-')) {
    throw new ReleaseVersionError(
      `MSIX package version cannot use prerelease version ${JSON.stringify(version)}`,
      'Microsoft Store MSIX packages require a stable MAJOR.MINOR.PATCH version without prerelease identifiers',
    );
  }
  if (version.includes('+')) {
    throw new ReleaseVersionError(
      `MSIX package version cannot include build metadata in ${JSON.stringify(version)}`,
      'Microsoft Store MSIX packages require a plain MAJOR.MINOR.PATCH version without +build metadata',
    );
  }
  const [major, minor, patch] = version.split('.').map(Number);
  if (!Number.isInteger(major) || major < 0 || major > MSIX_MAX_SEMVER_MAJOR) {
    throw new ReleaseVersionError(
      `MSIX package version major component out of range (0..${MSIX_MAX_SEMVER_MAJOR} -> Store 1..${MSIX_MAX_PART}): ${JSON.stringify(version)}`,
      `semver MAJOR must be between 0 and ${MSIX_MAX_SEMVER_MAJOR} so shifted MSIX major (MAJOR + 1) stays within 1..${MSIX_MAX_PART}`,
    );
  }
  if (
    !Number.isInteger(minor) ||
    !Number.isInteger(patch) ||
    minor < 0 ||
    minor > MSIX_MAX_PART ||
    patch < 0 ||
    patch > MSIX_MAX_PART
  ) {
    throw new ReleaseVersionError(
      `MSIX package version component out of range (0..${MSIX_MAX_PART}): ${JSON.stringify(version)}`,
      `each MINOR and PATCH component must be between 0 and ${MSIX_MAX_PART} for MSIX`,
    );
  }
  return `${major + 1}.${minor}.${patch}.0`;
}

function artifactBaseName(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
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
  /** Parsed out of the Squirrel.Mac update archive filename. */
  zip: string | null;
  /** Parsed out of the portable Windows archive, when that job has produced it. */
  windowsZip?: string | null;
  /** Parsed out of the Windows MSIX package, when that job has produced it. */
  windowsMsix?: string | null;
}

/**
 * Fail unless the tag, manifest, bundle metadata, and DMG/update ZIP names
 * are all the same version. Returns that version.
 *
 * The tag wins when present because it is the only one that reliably
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
    { label: 'ZIP filename', value: versions.zip, remedy: 'the update ZIP must match the version of the app it holds' },
    {
      label: 'Windows ZIP filename',
      value: versions.windowsZip ?? null,
      remedy: 'the Windows ZIP must match the version of the app it holds',
    },
    {
      label: 'Windows MSIX filename',
      value: versions.windowsMsix ?? null,
      remedy: 'the Windows MSIX must match the version of the app it holds',
    },
  ];

  for (const check of checks) {
    if (check.value === null) continue;
    if (check.value !== expected) {
      throw new ReleaseVersionError(`${check.label} is ${check.value}, but ${source} says ${expected}`, check.remedy);
    }
  }

  return expected;
}

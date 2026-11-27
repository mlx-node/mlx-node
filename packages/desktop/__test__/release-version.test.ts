/**
 * `tools bump` skips `private: true` workspaces and @mlx-node/desktop is one, so
 * the desktop manifest does not move when the repo version does. Every downstream
 * consumer — the packaged plist, the DMG filename, the Homebrew cask — used to
 * read that same manifest, which is why a stale version would have been
 * consistently stale everywhere and looked correct.
 *
 * These are the checks that make the four values disagree loudly instead.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

import {
  assertSemver,
  assertVersionsAgree,
  dmgFileName,
  msixPackageVersion,
  ReleaseVersionError,
  type ReleaseVersions,
  versionFromDmgPath,
  versionFromTag,
  updateZipFileName,
  versionFromUpdateZipPath,
  versionFromWindowsMsixPath,
  versionFromWindowsZipPath,
  windowsMsixFileName,
  windowsZipFileName,
} from '../scripts/release-version.js';

/** Everything agreeing on 0.0.9 — the shape a good release has. */
const AGREED: ReleaseVersions = {
  tag: '0.0.9',
  manifest: '0.0.9',
  bundleShort: '0.0.9',
  bundleVersion: '0.0.9',
  dmg: '0.0.9',
  zip: '0.0.9',
};

describe('versionFromTag', () => {
  it('strips the v that `tools bump` writes', () => {
    // `git tag -s v${newVersion}` — internal-tools/index.ts
    expect(versionFromTag('v0.0.9')).toBe('0.0.9');
    expect(versionFromTag('  v1.2.3  ')).toBe('1.2.3');
  });

  it('accepts a hand-cut bare version', () => {
    expect(versionFromTag('0.0.9')).toBe('0.0.9');
    expect(versionFromTag('1.0.0-rc.1')).toBe('1.0.0-rc.1');
  });

  it('rejects anything that is not a version', () => {
    // A tag like `nightly` would otherwise become the DMG name and the cask
    // version, and semver-ordered consumers would never see the release again.
    for (const tag of ['', '   ', 'nightly', 'v1.2', 'v1.2.3.4', 'latest']) {
      expect(() => versionFromTag(tag)).toThrow();
    }
  });
});

describe('dmg naming', () => {
  it('round-trips', () => {
    expect(dmgFileName('0.0.9')).toBe('Portato-0.0.9-arm64.dmg');
    expect(versionFromDmgPath(dmgFileName('0.0.9'))).toBe('0.0.9');
    expect(versionFromDmgPath(`packages/desktop/out/${dmgFileName('1.2.3-rc.1')}`)).toBe('1.2.3-rc.1');
  });

  it('does not claim a foreign name is ours', () => {
    expect(versionFromDmgPath('Portato.dmg')).toBeNull();
    expect(versionFromDmgPath('Portato-nightly-arm64.dmg')).toBeNull();
    expect(versionFromDmgPath('mlx-node-0.0.9-arm64.dmg')).toBeNull();
    expect(versionFromDmgPath('/tmp/somethingelse-0.0.9-arm64.dmg')).toBeNull();
  });

  it('refuses to name a DMG from a non-version', () => {
    expect(() => dmgFileName('nightly')).toThrow();
  });
});

describe('assertVersionsAgree', () => {
  it('returns the version when all four agree', () => {
    expect(assertVersionsAgree(AGREED)).toBe('0.0.9');
  });

  it('fails on the stale private manifest — the bug this exists for', () => {
    // `tools bump patch` took the repo 0.0.8 -> 0.0.9 and tagged v0.0.9, but
    // packages/desktop is private so its manifest is still 0.0.8.
    expect(() => assertVersionsAgree({ ...AGREED, manifest: '0.0.8' })).toThrow(
      /packages\/desktop\/package\.json version is 0\.0\.8.*release tag.*0\.0\.9/s,
    );
  });

  it('fails when the bundle was not stamped with the tag', () => {
    // packager silently ignoring `appVersion`, or --app-version never reaching it.
    expect(() => assertVersionsAgree({ ...AGREED, bundleShort: '0.0.8' })).toThrow(/CFBundleShortVersionString/);
    expect(() => assertVersionsAgree({ ...AGREED, bundleVersion: '0.0.8' })).toThrow(/CFBundleVersion/);
  });

  it('fails when the DMG name disagrees with what it holds', () => {
    expect(() => assertVersionsAgree({ ...AGREED, dmg: '0.0.8' })).toThrow(/DMG filename/);
  });

  it('skips artifacts that do not exist yet', () => {
    // The pre-build `resolve` pass has no DMG. Absent must not read as "0.0.9 !=
    // null", or the fast pre-build check could never pass.
    expect(assertVersionsAgree({ ...AGREED, dmg: null })).toBe('0.0.9');
  });

  it('falls back to the manifest when there is no tag', () => {
    // workflow_dispatch dry runs. Still fails closed on a disagreeing artifact.
    expect(assertVersionsAgree({ ...AGREED, tag: null })).toBe('0.0.9');
    expect(() => assertVersionsAgree({ ...AGREED, tag: null, bundleShort: '0.0.8' })).toThrow(
      /no release tag.*0\.0\.9/s,
    );
  });

  it('rejects a manifest version that is not a version at all', () => {
    expect(() => assertVersionsAgree({ ...AGREED, tag: null, manifest: 'workspace:*' })).toThrow(
      /packages\/desktop\/package\.json/,
    );
  });

  it('carries a remedy naming the private-workspace cause', () => {
    // The failure message has to say WHY the manifest is stale, or the next
    // person re-tags and hits it again.
    try {
      assertVersionsAgree({ ...AGREED, manifest: '0.0.8' });
      throw new Error('expected a throw');
    } catch (err) {
      expect((err as { remedy?: string }).remedy).toMatch(/private/);
    }
  });
});

describe('assertSemver', () => {
  it('rejects leading zeros', () => {
    // `0.01.0` and `0.1.0` are one version to semver and two DMG filenames.
    expect(() => assertSemver('v', '0.01.0')).toThrow();
    expect(assertSemver('v', '0.1.0')).toBe('0.1.0');
  });
});

describe('assertVersionsAgree, untagged path', () => {
  it('validates the manifest even when there is no tag to compare it against', () => {
    // With a tag, a junk manifest is caught by the four-way compare: it simply
    // will not equal the tag. With NO tag the manifest BECOMES the expected
    // value, so every other check compares it against itself and agrees
    // trivially. The manifest's own semver check is the only thing standing
    // between a typo and a stamped, signed, published bundle -- and nothing
    // else in this module can cover it.
    const junk = {
      tag: null,
      manifest: 'nightly',
      bundleShort: 'nightly',
      bundleVersion: 'nightly',
      dmg: 'nightly',
      zip: 'nightly',
    };
    expect(() => assertVersionsAgree(junk)).toThrow(/packages\/desktop\/package\.json version/);
  });
});

describe('Windows ZIP release versions', () => {
  it('round-trips a Windows path and checks the version against the other artifacts', () => {
    const path = `packages\\desktop\\out\\${windowsZipFileName('0.0.14')}`;
    expect(windowsZipFileName('0.0.14')).toBe('Portato-0.0.14-win32-arm64.zip');
    expect(versionFromWindowsZipPath(path)).toBe('0.0.14');
    expect(assertVersionsAgree({ ...AGREED, windowsZip: '0.0.9' })).toBe('0.0.9');
    expect(() => assertVersionsAgree({ ...AGREED, windowsZip: '0.0.8' })).toThrow(/Windows ZIP filename/);
  });

  it('rejects other platforms, architectures, and invalid versions', () => {
    expect(versionFromWindowsZipPath('Portato-0.0.14-darwin-arm64.zip')).toBeNull();
    expect(versionFromWindowsZipPath('Portato-0.0.14-win32-x64.zip')).toBeNull();
    expect(versionFromWindowsZipPath('Portato-nightly-win32-arm64.zip')).toBeNull();
    expect(() => windowsZipFileName('nightly')).toThrow();
  });
});

describe('Windows MSIX release versions', () => {
  it('round-trips a Windows MSIX path and checks agreement with other artifacts', () => {
    const path = `packages\\desktop\\out\\${windowsMsixFileName('0.0.21')}`;
    expect(windowsMsixFileName('0.0.21')).toBe('Portato-0.0.21-win32-arm64.msix');
    expect(versionFromWindowsMsixPath(path)).toBe('0.0.21');
    expect(versionFromWindowsMsixPath(`/out/${windowsMsixFileName('1.2.3')}`)).toBe('1.2.3');
    expect(assertVersionsAgree({ ...AGREED, windowsMsix: '0.0.9' })).toBe('0.0.9');
    expect(() => assertVersionsAgree({ ...AGREED, windowsMsix: '0.0.8' })).toThrow(/Windows MSIX filename/);
  });

  it('rejects other platforms, extensions, architectures, and invalid versions', () => {
    expect(versionFromWindowsMsixPath('Portato-0.0.21-win32-arm64.zip')).toBeNull();
    expect(versionFromWindowsMsixPath('Portato-0.0.21-win32-x64.msix')).toBeNull();
    expect(versionFromWindowsMsixPath('Portato-0.0.21-darwin-arm64.msix')).toBeNull();
    expect(versionFromWindowsMsixPath('Portato-nightly-win32-arm64.msix')).toBeNull();
    expect(versionFromWindowsMsixPath('other-0.0.21-win32-arm64.msix')).toBeNull();
    expect(() => windowsMsixFileName('nightly')).toThrow(ReleaseVersionError);
  });

  it('maps stable semver to Store-valid non-zero Major 4-part MSIX versions', () => {
    expect(msixPackageVersion('0.0.21')).toBe('1.0.21.0');
    expect(msixPackageVersion('0.0.22')).toBe('1.0.22.0');
    expect(msixPackageVersion('0.1.0')).toBe('1.1.0.0');
    expect(msixPackageVersion('1.0.0')).toBe('2.0.0.0');
    expect(msixPackageVersion('1.2.3')).toBe('2.2.3.0');
    expect(msixPackageVersion('65534.65535.65535')).toBe('65535.65535.65535.0');
  });

  it('preserves strict monotonic ordering across 0.x -> 1.x -> 2.x upgrades', () => {
    const compareMsixVersion = (a: string, b: string): number => {
      const aParts = a.split('.').map(Number);
      const bParts = b.split('.').map(Number);
      for (let i = 0; i < 4; i += 1) {
        if (aParts[i] !== bParts[i]) return aParts[i] - bParts[i];
      }
      return 0;
    };

    const sequence = ['0.0.21', '0.0.22', '0.1.0', '0.99.99', '1.0.0', '1.0.1', '2.0.0'].map(msixPackageVersion);
    for (let i = 1; i < sequence.length; i += 1) {
      expect(compareMsixVersion(sequence[i - 1], sequence[i])).toBeLessThan(0);
    }
    for (const msixVersion of sequence) {
      const [major, , , revision] = msixVersion.split('.').map(Number);
      expect(major).toBeGreaterThanOrEqual(1);
      expect(revision).toBe(0);
    }
  });

  it('rejects prereleases, build metadata, invalid semver, and out-of-range uint16 parts', () => {
    expect(() => msixPackageVersion('0.0.22-rc.1')).toThrow(ReleaseVersionError);
    expect(() => msixPackageVersion('0.0.22-rc.1')).toThrow(/prerelease/i);
    expect(() => msixPackageVersion('1.2.3+build.4')).toThrow(ReleaseVersionError);
    expect(() => msixPackageVersion('1.2.3+build.4')).toThrow(/build metadata/i);
    expect(() => msixPackageVersion('65535.0.0')).toThrow(ReleaseVersionError);
    expect(() => msixPackageVersion('65535.0.0')).toThrow(/0\.\.65534/);
    expect(() => msixPackageVersion('65536.0.0')).toThrow(ReleaseVersionError);
    expect(() => msixPackageVersion('0.65536.0')).toThrow(/0\.\.65535/);
    expect(() => msixPackageVersion('0.0.70000')).toThrow(ReleaseVersionError);
    expect(() => msixPackageVersion('nightly')).toThrow(ReleaseVersionError);
    expect(() => msixPackageVersion('0.01.0')).toThrow(ReleaseVersionError);
  });
});

describe('update ZIP release versions', () => {
  it('names the archive for Apple Silicon discovery by update.electronjs.org', () => {
    expect(updateZipFileName('0.0.14')).toBe('mlx-node-0.0.14-darwin-arm64.zip');
    expect(versionFromUpdateZipPath('/out/mlx-node-0.0.14-darwin-arm64.zip')).toBe('0.0.14');
  });

  it('rejects archives for a different app, platform, architecture or version scheme', () => {
    for (const path of [
      'mlx-node-0.0.14-arm64.zip',
      'mlx-node-0.0.14-darwin-x64.zip',
      'other-0.0.14-darwin-arm64.zip',
      'mlx-node-nightly-darwin-arm64.zip',
    ]) {
      expect(versionFromUpdateZipPath(path)).toBeNull();
    }
    expect(() => updateZipFileName('nightly')).toThrow();
  });

  it('blocks a ZIP that advertises a different version from its bundle', () => {
    expect(() => assertVersionsAgree({ ...AGREED, zip: '0.0.8' })).toThrow(/ZIP filename/);
  });
});

/**
 * The guard that runs BEFORE a tag exists.
 *
 * The workflow checks versions before building from a pushed tag. Catch the
 * underlying manifest drift earlier, in ordinary pull request CI.
 *
 * The drift is created by `tools bump`, in an ordinary commit, days earlier —
 * so that is where it should be caught. `tools bump` reads its current version
 * from `packages/core/package.json` and writes the new one to every
 * `private !== true` workspace; @mlx-node/desktop is private, so it is left
 * behind. This test goes red on that commit, in the normal test gate, with the
 * whole remedy in one line.
 */
describe('the desktop manifest tracks the version `tools bump` writes', () => {
  const versionOf = (relative: string): string => {
    const path = fileURLToPath(new URL(relative, import.meta.url));
    return (JSON.parse(readFileSync(path, 'utf-8')) as { version: string }).version;
  };

  it('matches the public workspace version', () => {
    // `packages/core` is the manifest `tools bump` reads to compute the next
    // version, which makes it the canonical one to track.
    expect(versionOf('../package.json')).toBe(versionOf('../../core/package.json'));
  });

  it('is a version the release gate would accept', () => {
    // A tagged release stamps the TAG, not this — but `assertVersionsAgree`
    // still requires the two to be equal, so an unparseable manifest fails the
    // release even when the tag is perfect.
    expect(assertSemver('packages/desktop/package.json version', versionOf('../package.json'))).toMatch(/^\d/);
  });
});

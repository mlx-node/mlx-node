/**
 * `tools bump` skips `private: true` workspaces and @mlx-node/desktop is one, so
 * the desktop manifest does not move when the repo version does. Every downstream
 * consumer — the packaged plist, the DMG filename, the Homebrew cask — used to
 * read that same manifest, which is why a stale version would have been
 * consistently stale everywhere and looked correct.
 *
 * These are the checks that make the four values disagree loudly instead.
 */

import { describe, expect, it } from 'vite-plus/test';

import {
  assertSemver,
  assertVersionsAgree,
  dmgFileName,
  type ReleaseVersions,
  versionFromDmgPath,
  versionFromTag,
} from '../scripts/release-version.js';

/** Everything agreeing on 0.0.9 — the shape a good release has. */
const AGREED: ReleaseVersions = {
  tag: '0.0.9',
  manifest: '0.0.9',
  bundleShort: '0.0.9',
  bundleVersion: '0.0.9',
  dmg: '0.0.9',
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
    expect(dmgFileName('0.0.9')).toBe('mlx-node-0.0.9-arm64.dmg');
    expect(versionFromDmgPath(dmgFileName('0.0.9'))).toBe('0.0.9');
    expect(versionFromDmgPath(`packages/desktop/out/${dmgFileName('1.2.3-rc.1')}`)).toBe('1.2.3-rc.1');
  });

  it('does not claim a foreign name is ours', () => {
    expect(versionFromDmgPath('mlx-node.dmg')).toBeNull();
    expect(versionFromDmgPath('mlx-node-nightly-arm64.dmg')).toBeNull();
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
    const junk = { tag: null, manifest: 'nightly', bundleShort: 'nightly', bundleVersion: 'nightly', dmg: 'nightly' };
    expect(() => assertVersionsAgree(junk)).toThrow(/packages\/desktop\/package\.json version/);
  });
});

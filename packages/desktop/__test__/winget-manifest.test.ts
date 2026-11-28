import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { PayloadError } from '../scripts/payload.js';
import { ReleaseVersionError } from '../scripts/release-version.js';
import {
  DEFAULT_WINGET_PACKAGE_ID,
  generateWingetManifests,
  normalizeSha256,
  sha256File,
  WINGET_MANIFEST_VERSION,
  wingetManifestRelativeDir,
  writeWingetManifests,
} from '../scripts/winget-manifest.js';

const SAMPLE_SHA256_LOWER = '0123456789abcdef'.repeat(4);
const SAMPLE_SHA256_UPPER = SAMPLE_SHA256_LOWER.toUpperCase();

describe('WinGet manifest generation', () => {
  it('renders valid 1.12.0 version, installer, and locale manifests for the NSIS installer with uppercase SHA-256', () => {
    const generated = generateWingetManifests({
      version: '0.0.21',
      sha256: SAMPLE_SHA256_LOWER,
      releaseDate: '2026-11-24',
    });

    expect(generated.packageIdentifier).toBe(DEFAULT_WINGET_PACKAGE_ID);
    expect(DEFAULT_WINGET_PACKAGE_ID).toBe('LongYinan.Portato');
    expect(generated.sha256).toBe(SAMPLE_SHA256_UPPER);
    expect(generated.relativeDirectory).toBe('manifests/l/LongYinan/Portato/0.0.21');

    expect(generated.files.version.filename).toBe('LongYinan.Portato.yaml');
    expect(generated.files.version.content).toContain(`winget-manifest.version.${WINGET_MANIFEST_VERSION}.schema.json`);
    expect(generated.files.version.content).toContain('PackageIdentifier: LongYinan.Portato');
    expect(generated.files.version.content).toContain('PackageVersion: 0.0.21');
    expect(generated.files.version.content).toContain('DefaultLocale: en-US');
    expect(generated.files.version.content).toContain('ManifestType: version');
    expect(generated.files.version.content).toContain('ManifestVersion: 1.12.0');

    expect(generated.files.installer.filename).toBe('LongYinan.Portato.installer.yaml');
    expect(generated.files.installer.content).toContain('MinimumOSVersion: 10.0.26100.0');
    expect(generated.files.installer.content).toContain('InstallerLocale: en-US');
    expect(generated.files.installer.content).toContain('InstallerType: nullsoft');
    expect(generated.files.installer.content).toContain('Scope: user');
    expect(generated.files.installer.content).toContain('UpgradeBehavior: install');
    expect(generated.files.installer.content).toContain('AppsAndFeaturesEntries:');
    expect(generated.files.installer.content).toContain('- DisplayName: Portato');
    expect(generated.files.installer.content).toContain('  Publisher: mlx-node');
    expect(generated.files.installer.content).toContain('  DisplayVersion: 0.0.21');
    expect(generated.files.installer.content).toContain('  ProductCode: Portato');
    expect(generated.files.installer.content).toContain(
      "  DefaultInstallLocation: '%LOCALAPPDATA%\\Programs\\Portato'",
    );
    expect(generated.files.installer.content).not.toContain('NestedInstallerType');
    expect(generated.files.installer.content).not.toContain('ArchiveBinariesDependOnPath');
    expect(generated.files.installer.content).toContain('ReleaseDate: 2026-11-24');
    expect(generated.files.installer.content).toContain('Architecture: arm64');
    expect(generated.files.installer.content).toContain(
      'InstallerUrl: https://github.com/mlx-node/mlx-node/releases/download/v0.0.21/Portato-0.0.21-win32-arm64-setup.exe',
    );
    expect(generated.files.installer.content).toContain(`InstallerSha256: ${SAMPLE_SHA256_UPPER}`);

    expect(generated.files.locale.filename).toBe('LongYinan.Portato.locale.en-US.yaml');
    expect(generated.files.locale.content).toContain('Publisher: mlx-node');
    expect(generated.files.locale.content).toContain('PublisherUrl: https://github.com/mlx-node');
    expect(generated.files.locale.content).toContain('PackageName: Portato');
    expect(generated.files.locale.content).toContain('PackageUrl: https://github.com/mlx-node/mlx-node');
    expect(generated.files.locale.content).toContain('License: MIT');
    expect(generated.files.locale.content).toContain('ShortDescription: ');
    expect(generated.files.locale.content).toContain(
      'ReleaseNotesUrl: https://github.com/mlx-node/mlx-node/releases/tag/v0.0.21',
    );
  });

  it('places custom PackageIdentifier manifests under manifests/<first-letter-lowercase>/<Publisher>/<App>/<version>', () => {
    expect(wingetManifestRelativeDir('LongYinan.Portato', '1.2.3')).toBe('manifests/l/LongYinan/Portato/1.2.3');
    const generated = generateWingetManifests({
      version: '1.2.3',
      sha256: SAMPLE_SHA256_UPPER,
      packageIdentifier: 'LongYinan.Portato',
      repo: 'LongYinan/portato',
      releaseDate: '2026-11-24',
    });
    expect(generated.files.version.filename).toBe('LongYinan.Portato.yaml');
    expect(generated.files.installer.content).toContain(
      'InstallerUrl: https://github.com/LongYinan/portato/releases/download/v1.2.3/Portato-1.2.3-win32-arm64-setup.exe',
    );
    expect(generated.files.installer.content).toContain('  Publisher: LongYinan');
    expect(generated.files.locale.content).toContain('Publisher: LongYinan');
    expect(generated.files.locale.content).toContain('PublisherUrl: https://github.com/LongYinan');
  });

  it('uses an explicit release tag in InstallerUrl and ReleaseNotesUrl and rejects mismatched tags', () => {
    const bareTag = generateWingetManifests({
      version: '0.0.21',
      tag: '0.0.21',
      sha256: SAMPLE_SHA256_UPPER,
      releaseDate: '2026-11-24',
    });
    expect(bareTag.releaseTag).toBe('0.0.21');
    expect(bareTag.files.installer.content).toContain(
      'InstallerUrl: https://github.com/mlx-node/mlx-node/releases/download/0.0.21/Portato-0.0.21-win32-arm64-setup.exe',
    );
    expect(bareTag.files.locale.content).toContain(
      'ReleaseNotesUrl: https://github.com/mlx-node/mlx-node/releases/tag/0.0.21',
    );

    const blankTag = generateWingetManifests({
      version: '0.0.21',
      tag: '   ',
      sha256: SAMPLE_SHA256_UPPER,
      releaseDate: '2026-11-24',
    });
    expect(blankTag.releaseTag).toBe('v0.0.21');
    expect(blankTag.files.installer.content).toContain(
      'InstallerUrl: https://github.com/mlx-node/mlx-node/releases/download/v0.0.21/Portato-0.0.21-win32-arm64-setup.exe',
    );

    expect(() =>
      generateWingetManifests({
        version: '0.0.21',
        tag: 'v0.0.22',
        sha256: SAMPLE_SHA256_UPPER,
      }),
    ).toThrow(ReleaseVersionError);
    expect(() =>
      generateWingetManifests({
        version: '0.0.21',
        tag: 'nightly',
        sha256: SAMPLE_SHA256_UPPER,
      }),
    ).toThrow(ReleaseVersionError);
  });

  it('rejects prerelease versions, build metadata, invalid IDs, bad SHA-256 digests, and bad dates', () => {
    expect(() => generateWingetManifests({ version: '0.0.22-rc.1', sha256: SAMPLE_SHA256_UPPER })).toThrow(
      ReleaseVersionError,
    );
    expect(() => generateWingetManifests({ version: '0.0.22+meta', sha256: SAMPLE_SHA256_UPPER })).toThrow(
      ReleaseVersionError,
    );
    expect(() => generateWingetManifests({ version: 'nightly', sha256: SAMPLE_SHA256_UPPER })).toThrow(
      ReleaseVersionError,
    );

    for (const badId of ['', 'mlx-node', 'mlx..node', '../evil.app', 'bad id.pkg', '-bad.pkg', 'pub.app.']) {
      expect(() =>
        generateWingetManifests({ version: '0.0.21', sha256: SAMPLE_SHA256_UPPER, packageIdentifier: badId }),
      ).toThrow(PayloadError);
    }

    expect(() => normalizeSha256('not-hex')).toThrow(PayloadError);
    expect(() => normalizeSha256('abc123')).toThrow(PayloadError);
    expect(() =>
      generateWingetManifests({ version: '0.0.21', sha256: SAMPLE_SHA256_UPPER, releaseDate: '2026-02-30' }),
    ).toThrow(PayloadError);
    expect(() =>
      generateWingetManifests({ version: '0.0.21', sha256: SAMPLE_SHA256_UPPER, repo: 'invalid-repo' }),
    ).toThrow(PayloadError);
  });
});

describe('WinGet manifest disk output and SHA-256 hashing', () => {
  let root: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'mlx-winget-manifest-'));
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it('hashes an installer file and writes the three manifests in the winget-pkgs directory layout', () => {
    const fakeInstaller = join(root, 'Portato-0.0.21-win32-arm64-setup.exe');
    const contents = Buffer.from('signed windows nsis installer payload');
    writeFileSync(fakeInstaller, contents);
    const expectedSha = createHash('sha256').update(contents).digest('hex').toUpperCase();

    expect(sha256File(fakeInstaller)).toBe(expectedSha);

    const outDir = join(root, 'out');
    const written = writeWingetManifests(outDir, {
      version: '0.0.21',
      sha256: sha256File(fakeInstaller),
      releaseDate: '2026-11-24',
    });

    expect(written.directory).toBe(join(outDir, 'manifests', 'l', 'LongYinan', 'Portato', '0.0.21'));
    expect(readFileSync(written.paths.version, 'utf8')).toBe(written.files.version.content);
    expect(readFileSync(written.paths.installer, 'utf8')).toContain(`InstallerSha256: ${expectedSha}`);
    expect(readFileSync(written.paths.locale, 'utf8')).toBe(written.files.locale.content);
  });

  it('throws PayloadError when hashing a missing installer path', () => {
    expect(() => sha256File(join(root, 'missing-setup.exe'))).toThrow(PayloadError);
  });
});

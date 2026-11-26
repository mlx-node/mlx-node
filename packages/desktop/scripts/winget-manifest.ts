/**
 * Generate official WinGet 1.12.0 multi-file manifests for the portable
 * Windows ARM64 ZIP release (`microsoft/winget-pkgs`).
 *
 *   yarn workspace @mlx-node/desktop winget:manifests \
 *     --version 0.0.21 \
 *     --zip out/Portato-0.0.21-win32-arm64.zip \
 *     [--tag v0.0.21] \
 *     [--out out/winget] \
 *     [--id LongYinan.Portato] \
 *     [--repo mlx-node/mlx-node] \
 *     [--release-date YYYY-MM-DD]
 *
 * Or pass `--sha256 <64-hex>` instead of `--zip` when the archive digest is
 * already known.
 */

import { createHash } from 'node:crypto';
import { closeSync, existsSync, mkdirSync, openSync, readSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

import { APP_NAME } from '../src/app-name.js';
import { PayloadError } from './payload.js';
import { assertSemver, ReleaseVersionError, versionFromTag, windowsZipFileName } from './release-version.js';

export const WINGET_MANIFEST_VERSION = '1.12.0';
export const DEFAULT_WINGET_PACKAGE_ID = 'LongYinan.Portato';
export const DEFAULT_WINGET_REPO = 'mlx-node/mlx-node';
export const WINGET_SHORT_DESCRIPTION = 'Local AI inference and control panel for Apple Silicon and NVIDIA CUDA';

const PACKAGE_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9-]{0,31}(?:\.[A-Za-z0-9][A-Za-z0-9-]{0,31}){1,7}$/;
const SHA256_PATTERN = /^[0-9a-fA-F]{64}$/;
const REPO_PATTERN = /^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/;
const ISO_DATE_PATTERN = /^(\d{4})-(\d{2})-(\d{2})$/;
const HASH_CHUNK_BYTES = 4 * 1024 * 1024;

export interface WingetManifestInput {
  /** Stable semver release version (e.g. `0.0.21`). */
  version: string;
  /** Exact GitHub release tag hosting the assets (defaults to `v${version}`). */
  tag?: string;
  /** 64-character hex SHA-256 of the Windows ARM64 portable ZIP. */
  sha256: string;
  /** WinGet package identifier (`Publisher.Package`, default `LongYinan.Portato`). */
  packageIdentifier?: string;
  /** GitHub `owner/repo` hosting the release assets (default `mlx-node/mlx-node`). */
  repo?: string;
  /** ISO `YYYY-MM-DD` release date (defaults to today in UTC). */
  releaseDate?: string;
}

export interface GeneratedWingetManifests {
  packageIdentifier: string;
  version: string;
  releaseTag: string;
  sha256: string;
  releaseDate: string;
  /** Relative directory inside `microsoft/winget-pkgs` (using POSIX slashes). */
  relativeDirectory: string;
  files: {
    version: { filename: string; content: string };
    installer: { filename: string; content: string };
    locale: { filename: string; content: string };
  };
}

export interface WrittenWingetManifests extends GeneratedWingetManifests {
  directory: string;
  paths: {
    version: string;
    installer: string;
    locale: string;
  };
}

export function assertWingetVersion(version: string): string {
  assertSemver('version', version);
  if (version.includes('-')) {
    throw new ReleaseVersionError(
      `WinGet manifest cannot use prerelease version ${JSON.stringify(version)}`,
      'WinGet winget-pkgs releases require a stable MAJOR.MINOR.PATCH version without prerelease identifiers',
    );
  }
  if (version.includes('+')) {
    throw new ReleaseVersionError(
      `WinGet manifest cannot include build metadata in ${JSON.stringify(version)}`,
      'WinGet winget-pkgs releases require a plain MAJOR.MINOR.PATCH version without +build metadata',
    );
  }
  return version;
}

export function assertWingetPackageId(packageIdentifier: string): string {
  const trimmed = packageIdentifier.trim();
  if (!PACKAGE_ID_PATTERN.test(trimmed) || trimmed.length > 128) {
    throw new PayloadError(
      `Invalid WinGet PackageIdentifier: ${JSON.stringify(packageIdentifier)}`,
      'expected Publisher.Package (two or more dot-separated alphanumeric/hyphen segments, e.g. LongYinan.Portato)',
    );
  }
  return trimmed;
}

export function normalizeSha256(sha256: string): string {
  const trimmed = sha256.trim();
  if (!SHA256_PATTERN.test(trimmed)) {
    throw new PayloadError(
      `Invalid SHA-256 digest: ${JSON.stringify(sha256)}`,
      'expected a 64-character hexadecimal SHA-256 hash',
    );
  }
  return trimmed.toUpperCase();
}

export function assertReleaseDate(date: string): string {
  const trimmed = date.trim();
  const match = ISO_DATE_PATTERN.exec(trimmed);
  if (match === null) {
    throw new PayloadError(`Invalid release date: ${JSON.stringify(date)}`, 'expected YYYY-MM-DD');
  }
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const parsed = new Date(Date.UTC(year, month - 1, day));
  if (parsed.getUTCFullYear() !== year || parsed.getUTCMonth() !== month - 1 || parsed.getUTCDate() !== day) {
    throw new PayloadError(
      `Invalid calendar date: ${JSON.stringify(date)}`,
      'expected a real YYYY-MM-DD calendar date',
    );
  }
  return trimmed;
}

export function assertGitHubRepo(repo: string): string {
  const trimmed = repo.trim();
  if (!REPO_PATTERN.test(trimmed)) {
    throw new PayloadError(`Invalid GitHub repository: ${JSON.stringify(repo)}`, 'expected owner/repo');
  }
  return trimmed;
}

/**
 * Compute uppercase SHA-256 by streaming in 4 MiB chunks so multi-gigabyte
 * Windows CUDA archives never hit Node's 2 GiB `readFileSync` buffer limit.
 */
export function sha256File(filePath: string): string {
  if (!existsSync(filePath) || !statSync(filePath).isFile()) {
    throw new PayloadError(
      `Windows ZIP archive not found: ${filePath}`,
      'build the Windows ZIP before generating WinGet manifests',
    );
  }
  const hash = createHash('sha256');
  const buffer = Buffer.allocUnsafe(HASH_CHUNK_BYTES);
  const fd = openSync(filePath, 'r');
  try {
    for (;;) {
      const bytesRead = readSync(fd, buffer, 0, buffer.length, null);
      if (bytesRead === 0) break;
      hash.update(buffer.subarray(0, bytesRead));
    }
  } finally {
    closeSync(fd);
  }
  return hash.digest('hex').toUpperCase();
}

/**
 * Compute the `microsoft/winget-pkgs` manifest directory path:
 * `manifests/<first-letter-lowercase>/<Publisher>/<App>/<version>`.
 */
export function wingetManifestRelativeDir(packageIdentifier: string, version: string): string {
  const id = assertWingetPackageId(packageIdentifier);
  const stableVersion = assertWingetVersion(version);
  const segments = id.split('.');
  return ['manifests', id[0].toLowerCase(), ...segments, stableVersion].join('/');
}

function resolveWingetReleaseTag(version: string, tag?: string): string {
  const trimmed = tag?.trim() ?? '';
  if (trimmed === '') return `v${version}`;
  const parsedVersion = versionFromTag(trimmed);
  if (parsedVersion !== version) {
    throw new ReleaseVersionError(
      `WinGet manifest tag ${JSON.stringify(trimmed)} resolves to ${parsedVersion}, expected ${version}`,
      `pass --tag matching version ${version} (e.g. v${version} or ${version})`,
    );
  }
  return trimmed;
}

/** Render the three WinGet 1.12.0 multi-file YAML manifests in memory. */
export function generateWingetManifests(input: WingetManifestInput): GeneratedWingetManifests {
  const version = assertWingetVersion(input.version);
  const releaseTag = resolveWingetReleaseTag(version, input.tag);
  const packageIdentifier = assertWingetPackageId(input.packageIdentifier ?? DEFAULT_WINGET_PACKAGE_ID);
  const sha256 = normalizeSha256(input.sha256);
  const repo = assertGitHubRepo(input.repo ?? DEFAULT_WINGET_REPO);
  const releaseDate = assertReleaseDate(input.releaseDate ?? new Date().toISOString().slice(0, 10));

  const [owner] = repo.split('/');
  const zipName = windowsZipFileName(version);
  const installerUrl = `https://github.com/${repo}/releases/download/${releaseTag}/${zipName}`;
  const releaseNotesUrl = `https://github.com/${repo}/releases/tag/${releaseTag}`;
  const publisherUrl = `https://github.com/${owner}`;
  const packageUrl = `https://github.com/${repo}`;

  const versionContent =
    [
      `# yaml-language-server: $schema=https://aka.ms/winget-manifest.version.${WINGET_MANIFEST_VERSION}.schema.json`,
      '',
      `PackageIdentifier: ${packageIdentifier}`,
      `PackageVersion: ${version}`,
      'DefaultLocale: en-US',
      'ManifestType: version',
      `ManifestVersion: ${WINGET_MANIFEST_VERSION}`,
    ].join('\n') + '\n';

  const installerContent =
    [
      `# yaml-language-server: $schema=https://aka.ms/winget-manifest.installer.${WINGET_MANIFEST_VERSION}.schema.json`,
      '',
      `PackageIdentifier: ${packageIdentifier}`,
      `PackageVersion: ${version}`,
      'InstallerType: zip',
      'NestedInstallerType: portable',
      'NestedInstallerFiles:',
      `- RelativeFilePath: ${APP_NAME}-win32-arm64\\mlx-node.exe`,
      '  PortableCommandAlias: mlx-node',
      'ArchiveBinariesDependOnPath: true',
      `ReleaseDate: ${releaseDate}`,
      'Installers:',
      '- Architecture: arm64',
      `  InstallerUrl: ${installerUrl}`,
      `  InstallerSha256: ${sha256}`,
      'ManifestType: installer',
      `ManifestVersion: ${WINGET_MANIFEST_VERSION}`,
    ].join('\n') + '\n';

  const localeContent =
    [
      `# yaml-language-server: $schema=https://aka.ms/winget-manifest.defaultLocale.${WINGET_MANIFEST_VERSION}.schema.json`,
      '',
      `PackageIdentifier: ${packageIdentifier}`,
      `PackageVersion: ${version}`,
      'PackageLocale: en-US',
      `Publisher: ${owner}`,
      `PublisherUrl: ${publisherUrl}`,
      `PackageName: ${APP_NAME}`,
      `PackageUrl: ${packageUrl}`,
      'License: MIT',
      `ShortDescription: ${WINGET_SHORT_DESCRIPTION}`,
      `ReleaseNotesUrl: ${releaseNotesUrl}`,
      'ManifestType: defaultLocale',
      `ManifestVersion: ${WINGET_MANIFEST_VERSION}`,
    ].join('\n') + '\n';

  return {
    packageIdentifier,
    version,
    releaseTag,
    sha256,
    releaseDate,
    relativeDirectory: wingetManifestRelativeDir(packageIdentifier, version),
    files: {
      version: { filename: `${packageIdentifier}.yaml`, content: versionContent },
      installer: { filename: `${packageIdentifier}.installer.yaml`, content: installerContent },
      locale: { filename: `${packageIdentifier}.locale.en-US.yaml`, content: localeContent },
    },
  };
}

/** Write the generated manifests into `<outDir>/manifests/<letter>/<Publisher>/<App>/<version>/`. */
export function writeWingetManifests(outDir: string, input: WingetManifestInput): WrittenWingetManifests {
  const generated = generateWingetManifests(input);
  const directory = join(outDir, ...generated.relativeDirectory.split('/'));
  mkdirSync(directory, { recursive: true });

  const versionPath = join(directory, generated.files.version.filename);
  const installerPath = join(directory, generated.files.installer.filename);
  const localePath = join(directory, generated.files.locale.filename);

  writeFileSync(versionPath, generated.files.version.content, 'utf8');
  writeFileSync(installerPath, generated.files.installer.content, 'utf8');
  writeFileSync(localePath, generated.files.locale.content, 'utf8');

  return {
    ...generated,
    directory,
    paths: {
      version: versionPath,
      installer: installerPath,
      locale: localePath,
    },
  };
}

function isDirectExecution(): boolean {
  const entry = process.argv[1];
  if (!entry) return false;
  const resolvedEntry = resolve(entry);
  const self = fileURLToPath(import.meta.url);
  return process.platform === 'win32' ? resolvedEntry.toLowerCase() === self.toLowerCase() : resolvedEntry === self;
}

if (isDirectExecution()) {
  const HERE = dirname(fileURLToPath(import.meta.url));
  const APP_DIR = resolve(HERE, '..');

  const { values } = parseArgs({
    options: {
      version: { type: 'string' },
      tag: { type: 'string' },
      zip: { type: 'string' },
      sha256: { type: 'string' },
      out: { type: 'string', default: join(APP_DIR, 'out', 'winget') },
      id: { type: 'string' },
      repo: { type: 'string', default: DEFAULT_WINGET_REPO },
      'release-date': { type: 'string' },
    },
  });

  if (!values.version) {
    console.error(
      'usage: winget-manifest.ts --version <ver> (--zip <path-to-zip> | --sha256 <hex>) [--tag <tag>] [--out <dir>] [--id <pkg-id>] [--repo <owner/repo>] [--release-date <YYYY-MM-DD>]',
    );
    process.exit(2);
  }
  if (!values.zip && !values.sha256) {
    console.error('either --zip <path-to-zip> or --sha256 <hex> is required');
    process.exit(2);
  }
  if (values.zip && values.sha256) {
    console.error('pass either --zip or --sha256, not both');
    process.exit(2);
  }

  try {
    const sha256 = values.zip ? sha256File(values.zip) : values.sha256!;
    const packageIdentifier = values.id ?? (process.env.WINGET_PACKAGE_ID?.trim() || DEFAULT_WINGET_PACKAGE_ID);
    const written = writeWingetManifests(values.out, {
      version: values.version,
      tag: values.tag,
      sha256,
      packageIdentifier,
      repo: values.repo,
      releaseDate: values['release-date'],
    });
    console.log(`wrote WinGet ${WINGET_MANIFEST_VERSION} manifests to ${written.directory}`);
    console.log(`  ${written.paths.version}`);
    console.log(`  ${written.paths.installer}`);
    console.log(`  ${written.paths.locale}`);
  } catch (error) {
    if (error instanceof PayloadError || error instanceof ReleaseVersionError) {
      console.error(`::error::${error.message}`);
      console.error(`  ${error.remedy}`);
      process.exit(1);
    }
    throw error;
  }
}

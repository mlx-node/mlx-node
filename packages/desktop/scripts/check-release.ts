/**
 * The release gate for versions. Run from `.github/workflows/desktop-release.yml`.
 *
 *   check-release.ts resolve --tag <tag> --manifest <package.json>
 *   check-release.ts verify  --tag <tag> --manifest <package.json> --app <.app> [--dmg <dmg>] [--zip <zip>] [--windows-installer <exe>]
 *   check-release.ts verify-windows --tag <tag> --manifest <package.json> --app <packaged-dir> --installer <exe> [--msix <msix>]
 *
 * `resolve` runs BEFORE the ~16-minute build so a stale manifest costs a minute
 * instead of a full release cycle, and emits `version`, `dmg_name`, `zip_name`,
 * `windows_installer_name`, and `windows_msix_name` to `$GITHUB_OUTPUT` so every
 * later step reads one value rather than re-deriving it.
 *
 * `verify` runs after the artifacts exist and re-checks the same thing against
 * what was actually produced — the plist inside the `.app` and the artifact filenames.
 * Two checks, because `resolve` only proves the inputs agreed.
 *
 * `--tag` may be empty (`workflow_dispatch` has no tag); the manifest is then the
 * source and the tag comparison is skipped.
 */

import { execFileSync } from 'node:child_process';
import { appendFileSync, existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseArgs } from 'node:util';

import { PayloadError } from './payload.js';
import {
  assertVersionsAgree,
  dmgFileName,
  ReleaseVersionError,
  updateZipFileName,
  versionFromDmgPath,
  versionFromTag,
  versionFromUpdateZipPath,
  versionFromWindowsInstallerPath,
  versionFromWindowsMsixPath,
  windowsInstallerFileName,
  windowsMsixFileName,
} from './release-version.js';
import { verifyWindowsMsix } from './windows-msix.js';
import { verifyWindowsInstaller } from './windows-nsis.js';
import { readPackagedWindowsApp } from './windows-payload.js';

const { positionals, values } = parseArgs({
  allowPositionals: true,
  options: {
    tag: { type: 'string', default: '' },
    manifest: { type: 'string' },
    app: { type: 'string' },
    dmg: { type: 'string', default: '' },
    zip: { type: 'string', default: '' },
    installer: { type: 'string', default: '' },
    'windows-installer': { type: 'string', default: '' },
    msix: { type: 'string', default: '' },
  },
});

const command = positionals[0];
if (command !== 'resolve' && command !== 'verify' && command !== 'verify-windows') {
  console.error(
    'usage: check-release.ts <resolve|verify|verify-windows> --tag <tag> --manifest <package.json> [--app <path>] [--dmg <dmg>] [--zip <zip>] [--installer <exe>] [--windows-installer <exe>] [--msix <msix>]',
  );
  process.exit(2);
}
if (values.manifest === undefined) {
  console.error('--manifest is required');
  process.exit(2);
}

const manifestPath = values.manifest;
const manifest = readManifestVersion(manifestPath);

/**
 * Inside the handler, not above it: a tag like `nightly` is exactly the input this
 * gate exists for, and parsing it at module scope turned a release-blocking
 * mismatch into an unhandled rejection — a stack trace with no `::error::`
 * annotation, which reads on the job page as the workflow itself being broken.
 */
let tag: string | null = null;

try {
  // Empty means "not a release build", not "a tag we failed to read".
  tag = values.tag.trim() === '' ? null : versionFromTag(values.tag);
  const version = command === 'resolve' ? resolve() : command === 'verify' ? verify() : verifyWindows();
  console.log(`release version: ${version}`);
} catch (err) {
  if (err instanceof ReleaseVersionError || err instanceof PayloadError) {
    // `::error::` so it lands on the job summary rather than only in the log.
    console.error(`::error::${err.message}`);
    console.error(`  ${err.remedy}`);
    process.exit(1);
  }
  throw err;
}

function resolve(): string {
  // The bundle and archives do not exist yet; passing the expected version for them
  // would assert nothing, so they are declared absent and `verify` covers them.
  const version = assertVersionsAgree({
    tag,
    manifest,
    bundleShort: tag ?? manifest,
    bundleVersion: tag ?? manifest,
    dmg: null,
    zip: null,
  });
  const output = process.env.GITHUB_OUTPUT;
  if (output !== undefined && output !== '') {
    appendFileSync(
      output,
      `version=${version}\ndmg_name=${dmgFileName(version)}\nzip_name=${updateZipFileName(version)}\nwindows_installer_name=${windowsInstallerFileName(version)}\nwindows_msix_name=${windowsMsixFileName(version)}\n`,
    );
  }
  return version;
}

function verify(): string {
  if (values.app === undefined) {
    console.error('verify needs --app');
    process.exit(2);
  }
  const plist = join(values.app, 'Contents', 'Info.plist');
  if (!existsSync(plist)) {
    console.error(`::error::no Info.plist at ${plist} — is ${values.app} really the bundle?`);
    process.exit(1);
  }

  const dmg = values.dmg.trim();
  if (dmg !== '' && versionFromDmgPath(dmg) === null) {
    console.error(`::error::DMG ${dmg} is not named Portato-<version>-arm64.dmg`);
    process.exit(1);
  }

  const zip = values.zip.trim();
  if (zip !== '' && (versionFromUpdateZipPath(zip) === null || !existsSync(zip))) {
    console.error(`::error::update ZIP ${zip} must exist and be named Portato-<version>-darwin-arm64.zip`);
    process.exit(1);
  }
  const windowsInstaller = values['windows-installer'].trim();
  if (
    windowsInstaller !== '' &&
    (versionFromWindowsInstallerPath(windowsInstaller) === null || !existsSync(windowsInstaller))
  ) {
    console.error(
      `::error::Windows installer ${windowsInstaller} must exist and be named Portato-<version>-win32-arm64-setup.exe`,
    );
    process.exit(1);
  }

  return assertVersionsAgree({
    tag,
    manifest,
    bundleShort: plistString(plist, 'CFBundleShortVersionString'),
    bundleVersion: plistString(plist, 'CFBundleVersion'),
    dmg: dmg === '' ? null : versionFromDmgPath(dmg),
    zip: zip === '' ? null : versionFromUpdateZipPath(zip),
    windowsInstaller: windowsInstaller === '' ? null : versionFromWindowsInstallerPath(windowsInstaller),
  });
}

function verifyWindows(): string {
  if (values.app === undefined) {
    console.error('verify-windows needs --app');
    process.exit(2);
  }
  const installer = values.installer.trim();
  if (installer === '' || versionFromWindowsInstallerPath(installer) === null || !existsSync(installer)) {
    console.error(
      `::error::Windows installer ${installer} must exist and be named Portato-<version>-win32-arm64-setup.exe`,
    );
    process.exit(1);
  }
  const msix = values.msix.trim();
  if (msix !== '' && (versionFromWindowsMsixPath(msix) === null || !existsSync(msix))) {
    console.error(`::error::Windows MSIX ${msix} must exist and be named Portato-<version>-win32-arm64.msix`);
    process.exit(1);
  }
  const app = readPackagedWindowsApp(values.app);
  const version = assertVersionsAgree({
    tag,
    manifest,
    bundleShort: app.version,
    bundleVersion: app.version,
    dmg: null,
    zip: null,
    windowsInstaller: versionFromWindowsInstallerPath(installer),
    windowsMsix: msix === '' ? null : versionFromWindowsMsixPath(msix),
  });
  verifyWindowsInstaller({ installerPath: installer, expectedVersion: version });
  if (msix !== '') {
    verifyWindowsMsix({
      msixPath: msix,
      expectedVersion: version,
      runtimeFiles: app.runtimeFiles,
    });
  }
  return version;
}

/**
 * A function rather than an inline `typeof` guard: control-flow narrowing does not
 * reach into the hoisted declarations below, so an inline guard leaves `manifest`
 * as `unknown` everywhere it is actually used.
 */
function readManifestVersion(path: string): string {
  const version = (JSON.parse(readFileSync(path, 'utf-8')) as { version?: unknown }).version;
  if (typeof version !== 'string') {
    console.error(`${path} has no string "version"`);
    process.exit(2);
  }
  return version;
}

/** Read the real plist, not our memory of what we asked packager to write. */
function plistString(plist: string, key: string): string {
  return execFileSync('plutil', ['-extract', key, 'raw', '-o', '-', plist], { encoding: 'utf-8' }).trim();
}

/**
 * Build an unsigned, portable Windows ARM64 ZIP (and, for stable versions, an
 * unsigned Microsoft Store ARM64 MSIX) in packages/desktop/out.
 *
 *   yarn workspace @mlx-node/desktop package:windows --app-version 0.0.21
 *   yarn workspace @mlx-node/desktop package:windows --app-version 0.0.21 --skip-msix
 *
 * The ZIP contains mlx-node-win32-arm64/mlx-node.exe, the staged JS closure,
 * the addon, every CUDA/cuDNN/OpenBLAS runtime DLL, and the NVRTC headers.
 * Windows signing and automatic updates are deliberately not part of this path
 * (Partner Center signs the MSIX during Microsoft Store ingestion).
 */

import { execFileSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, mkdtempSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { packager } from '@electron/packager';

import { PayloadError } from './payload.js';
import { assertSemver, ReleaseVersionError, windowsMsixFileName, windowsZipFileName } from './release-version.js';
import { stageApp } from './stage-app.js';
import { verifyPackagedNativeRuntime } from './verify-cli.js';
import { buildWindowsMsix, type BuildWindowsMsixResult, verifyWindowsMsix } from './windows-msix.js';
import {
  assertWindowsAppRuntimeAssets,
  missingWindowsZipEntries,
  readPackagedWindowsApp,
  resolveWindowsPayload,
} from './windows-payload.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const APP_DIR = resolve(HERE, '..');
const REPO_ROOT = resolve(APP_DIR, '..', '..');
const OUT = join(APP_DIR, 'out');
const STAGE = join(OUT, '.windows-stage');
const STAGE_APP = join(STAGE, 'app');
const PRODUCT_NAME = 'mlx-node';
const TAR = join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'tar.exe');
const RUNTIME_ROOTS = ['@mlx-node/server', '@mlx-node/dashboard', '@mlx-node/cli', 'electron-updater'];

if (process.platform !== 'win32' || process.arch !== 'arm64') {
  console.error('\nWindows packaging must run with ARM64 Node.js on Windows ARM64.\n');
  process.exit(1);
}

const skipMsix = process.argv.includes('--skip-msix');
const manifestVersion = (await import(join(APP_DIR, 'package.json'), { with: { type: 'json' } })).default.version;
const versionFlag = process.argv.indexOf('--app-version');
if (versionFlag !== -1 && process.argv[versionFlag + 1] === undefined) {
  console.error('\n--app-version needs a value\n');
  process.exit(1);
}
const appVersion = assertSemver(
  versionFlag === -1 ? 'packages/desktop/package.json version' : '--app-version',
  versionFlag === -1 ? manifestVersion : process.argv[versionFlag + 1],
);

const iconPath = join(APP_DIR, 'build', 'icon.ico');
if (!existsSync(iconPath) || statSync(iconPath).size === 0) {
  console.error(`\nMissing or empty Windows app icon: ${iconPath}\n\n  run: yarn workspace @mlx-node/desktop icons\n`);
  process.exit(1);
}

const payload = (() => {
  try {
    return resolveWindowsPayload(REPO_ROOT);
  } catch (error) {
    if (error instanceof PayloadError) {
      console.error(`\n${error.message}\n\n  run: ${error.remedy}\n`);
      process.exit(1);
    }
    throw error;
  }
})();

rmSync(STAGE, { recursive: true, force: true });
const stagedNative = join(STAGE, 'native');
mkdirSync(stagedNative, { recursive: true });
for (const file of payload.runtimeFiles) {
  cpSync(join(payload.nativeDir, file), join(stagedNative, file));
}

// NVRTC looks for both the MLX/CCCL/CUTLASS tree and the CUDA toolkit headers.
// Overlay the MLX copies after the toolkit so its pinned CCCL version wins.
cpSync(payload.cudaIncludeDir, join(stagedNative, 'include'), { recursive: true });
for (const name of ['cccl', 'cute', 'cutlass']) {
  cpSync(join(payload.nativeDir, 'include', name), join(stagedNative, 'include', name), { recursive: true });
}

const staged = stageApp({
  repoRoot: REPO_ROOT,
  desktopDir: payload.appDir,
  stageDir: STAGE_APP,
  roots: RUNTIME_ROOTS,
  autoUpdates: false,
  platform: 'win32',
});
try {
  assertWindowsAppRuntimeAssets(STAGE_APP);
} catch (error) {
  if (error instanceof PayloadError) {
    console.error(`\n${error.message}\n\n  run: ${error.remedy}\n`);
    process.exit(1);
  }
  throw error;
}
console.log(`staged ${staged.externalCount} external + ${staged.workspaceCount} workspace packages`);
console.log(`staged ${payload.runtimeFiles.length - 1} native runtime DLLs`);

console.log('packaging Windows ARM64…');
const [appPath] = await packager({
  dir: STAGE_APP,
  out: OUT,
  platform: 'win32',
  arch: 'arm64',
  appVersion,
  executableName: PRODUCT_NAME,
  icon: iconPath,
  overwrite: true,
  asar: false,
  extraResource: [stagedNative, payload.wwwRoot],
  prune: false,
});
if (!appPath) throw new Error('Electron Packager returned no Windows app directory');

const resources = join(appPath, 'resources');
rmSync(join(resources, 'www'), { recursive: true, force: true });
cpSync(join(resources, 'web'), join(resources, 'www'), { recursive: true });
rmSync(join(resources, 'web'), { recursive: true, force: true });

let packaged: ReturnType<typeof readPackagedWindowsApp>;
try {
  packaged = readPackagedWindowsApp(appPath);
} catch (error) {
  if (error instanceof PayloadError) {
    console.error(`\n${error.message}\n\n  ${error.remedy}\n`);
    process.exit(1);
  }
  throw error;
}
if (packaged.version !== appVersion) {
  throw new Error(`Packaged app version ${packaged.version} does not match --app-version ${appVersion}`);
}

const tempHome = mkdtempSync(join(tmpdir(), 'mlx-windows-runtime-check-'));
try {
  verifyPackagedNativeRuntime({
    executable: join(appPath, 'mlx-node.exe'),
    appDir: join(appPath, 'resources', 'app'),
    home: tempHome,
    platform: 'win32',
  });
  console.log('packaged Windows native runtime verified (Keyring, Canvas, pdfjs-dist, Clairic PDF)');
} finally {
  rmSync(tempHome, { recursive: true, force: true });
}

const zipPath = join(OUT, windowsZipFileName(appVersion));
rmSync(zipPath, { force: true });
const appDirectoryName = basename(appPath);
execFileSync(TAR, ['-a', '-cf', zipPath, '-C', dirname(appPath), appDirectoryName], { stdio: 'inherit' });
const entries = execFileSync(TAR, ['-tf', zipPath], {
  encoding: 'utf8',
  maxBuffer: 32 * 1024 * 1024,
})
  .split(/\r?\n/)
  .filter(Boolean);
const missing = missingWindowsZipEntries(entries, appDirectoryName, packaged.runtimeFiles);
if (missing.length > 0) throw new Error(`Windows ZIP is missing required entries:\n${missing.join('\n')}`);

rmSync(STAGE, { recursive: true, force: true });

let msixResult: BuildWindowsMsixResult | null = null;
if (skipMsix) {
  console.log('skipping MSIX packaging (--skip-msix)');
} else if (appVersion.includes('-') || appVersion.includes('+')) {
  console.log(
    `skipping MSIX packaging for non-stable version ${appVersion} (Microsoft Store requires stable MAJOR.MINOR.PATCH mapped to (MAJOR + 1).MINOR.PATCH.0)`,
  );
} else {
  console.log('packaging Windows ARM64 MSIX…');
  try {
    msixResult = buildWindowsMsix({
      appPath,
      msixPath: join(OUT, windowsMsixFileName(appVersion)),
      version: appVersion,
      assetsDir: join(APP_DIR, 'build', 'msix'),
    });
    verifyWindowsMsix({
      msixPath: msixResult.msixPath,
      expectedVersion: appVersion,
      runtimeFiles: packaged.runtimeFiles,
      identity: msixResult.identity,
    });
  } catch (error) {
    if (error instanceof PayloadError || error instanceof ReleaseVersionError) {
      console.error(`\n${error.message}\n\n  ${error.remedy}\n`);
      process.exit(1);
    }
    throw error;
  }
}

const resultPath = join(OUT, 'windows-package-result.json');
writeFileSync(
  resultPath,
  `${JSON.stringify(
    {
      appPath,
      zipPath,
      msixPath: msixResult?.msixPath ?? null,
      msixVersion: msixResult?.msixVersion ?? null,
      msixIdentityName: msixResult?.identity.identityName ?? null,
      msixPublisher: msixResult?.identity.publisher ?? null,
      version: appVersion,
      platform: 'win32',
      arch: 'arm64',
      signed: false,
      runtimeFiles: packaged.runtimeFiles,
    },
    null,
    2,
  )}\n`,
);

console.log(`\n${zipPath}`);
if (msixResult !== null) console.log(msixResult.msixPath);
console.log(`\n  result: ${resultPath}\n`);

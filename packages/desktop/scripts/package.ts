/**
 * Build `packages/desktop/out/mlx-node.app`.
 *
 *   yarn workspace @mlx-node/desktop package            unsigned, for local runs
 *   yarn workspace @mlx-node/desktop package --sign     Developer ID + hardened runtime
 *
 * Signing needs an identity in the keychain and `APPLE_TEAM_ID`. Notarization is
 * NOT here — it belongs to `.github/workflows/desktop-release.yml`, which submits
 * and staples. Verify either result with `scripts/verify-bundle.sh`.
 */

import { execFileSync } from 'node:child_process';
import { cpSync, mkdirSync, rmSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { packager } from '@electron/packager';

import { bundledAddonPath, codesignArgs, NATIVE_FILES, PayloadError, resolvePayload } from './payload.js';
import { stageApp } from './stage-app.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const APP_DIR = resolve(HERE, '..');
const REPO_ROOT = resolve(APP_DIR, '..', '..');
const OUT = join(APP_DIR, 'out');
const STAGE = join(APP_DIR, 'out', '.stage');
const STAGE_APP = join(STAGE, 'app');

/**
 * What the three entries actually import at runtime. Deliberately a hand-kept
 * list rather than `packages/desktop/package.json` dependencies: that manifest
 * also carries `@mlx-node/lm` for the Phase 0 spike, which MAIN must never load.
 */
const RUNTIME_ROOTS = ['@mlx-node/server', '@mlx-node/dashboard'];
const BUNDLE_ID = 'ai.mlxnode.desktop';
const PRODUCT_NAME = 'mlx-node';

const sign = process.argv.includes('--sign');

function run(cmd: string, args: string[]): void {
  execFileSync(cmd, args, { stdio: 'inherit' });
}

const payload = (() => {
  try {
    return resolvePayload(REPO_ROOT);
  } catch (err) {
    if (err instanceof PayloadError) {
      console.error(`\n${err.message}\n\n  run: ${err.remedy}\n`);
      process.exit(1);
    }
    throw err;
  }
})();

// Stage the three native files into ONE directory. They must stay colocated:
// MLX loads mlx.metallib from beside the binary, and paged_attn.metallib is
// resolved dladdr-relative from the .node.
rmSync(STAGE, { recursive: true, force: true });
const stagedNative = join(STAGE, 'native');
mkdirSync(stagedNative, { recursive: true });
for (const file of NATIVE_FILES) {
  cpSync(payload.native[file], join(stagedNative, file));
}

// Two fixups on the staged copy, BEFORE packager or codesign sees it. Both are
// permanent properties of the artifact napi produces, verified on a real build:
//
//   LC_ID_DYLIB  /Users/<builder>/.../target/.../libmlx_core.dylib
//   Identifier   libmlx_core.dylib
//
// The first ships a developer's home directory (and the CI layout) to every user.
// The second makes a build-artifact filename this app's code-signing identity for
// its largest binary. Rewriting the install name must happen before signing —
// install_name_tool invalidates a signature.
const stagedAddon = join(stagedNative, 'mlx-core.darwin-arm64.node');
run('install_name_tool', ['-id', `@rpath/${NATIVE_FILES[0]}`, stagedAddon]);

const staged = stageApp({ repoRoot: REPO_ROOT, desktopDir: APP_DIR, stageDir: STAGE_APP, roots: RUNTIME_ROOTS });
console.log(`staged ${staged.externalCount} external + ${staged.workspaceCount} workspace packages`);
if (staged.excludedPrebuilt.length > 0) {
  console.log(`  excluded ${staged.excludedPrebuilt.join(', ')} — payload ships once, in Resources/native`);
}
if (staged.skippedOptional.length > 0) {
  // Never silent: "skipped 14 optional packages" is the difference between a
  // deliberate platform gate and a dependency that quietly failed to install.
  console.log(`  skipped ${staged.skippedOptional.length} uninstalled optional deps (other platforms)`);
}

console.log('packaging…');
const [outDir] = await packager({
  dir: STAGE_APP,
  out: OUT,
  platform: 'darwin',
  arch: 'arm64',
  appBundleId: BUNDLE_ID,
  appVersion: (await import(join(APP_DIR, 'package.json'), { with: { type: 'json' } })).default.version,
  icon: join(APP_DIR, 'build', 'icon.icns'),
  overwrite: true,
  // `asar: false` is deliberate. dlopen cannot read out of an asar archive, and
  // BOTH the addon and the sidecar entry that loads it would have to be unpacked
  // anyway. Given a 233 MB native payload that lives outside the archive
  // regardless, asar buys nothing here and costs a class of path bug that only
  // reproduces in a packaged build.
  asar: false,
  // Placed verbatim in Contents/Resources, outside anything packager rewrites.
  extraResource: [stagedNative, payload.wwwRoot],
  // OFF because the staged tree is ALREADY the exact resolved closure. Pruning
  // walks with flora-colossus from the app dir, and Yarn hoists workspace deps to
  // the repo root, so it would fail to locate @mlx-node/* and abort.
  prune: false,
  appCategoryType: 'public.app-category.developer-tools',
  darwinDarkModeSupport: true,
});

// packager returns the PLATFORM directory (`out/mlx-node-darwin-arm64`), not the
// bundle inside it. Treating the return value as the .app puts every later path
// one level too high, and the first symptom is an ENOENT on a file that is
// actually present.
const appPath = join(outDir, `${PRODUCT_NAME}.app`);

// extraResource keeps the source directory name; the app expects `www`.
const resources = join(appPath, 'Contents', 'Resources');
rmSync(join(resources, 'www'), { recursive: true, force: true });
cpSync(join(resources, 'web'), join(resources, 'www'), { recursive: true });
rmSync(join(resources, 'web'), { recursive: true, force: true });

console.log(`addon will load from: ${bundledAddonPath(resources)}`);

if (sign) {
  const identity = process.env.SIGN_IDENTITY ?? findIdentity();
  const entitlements = join(APP_DIR, 'build', 'entitlements.mac.plist');

  // Inside-out, and the addon FIRST: signing the outer bundle seals whatever its
  // nested Mach-O files look like at that moment, so anything signed afterwards
  // breaks the outer seal. The addon also takes --identifier, to replace the
  // `libmlx_core.dylib` build-artifact identity.
  run('codesign', codesignArgs({ file: stagedAddonInBundle(resources), identity, identifier: `${BUNDLE_ID}.mlx-core` }));

  const { sign: osxSign } = await import('@electron/osx-sign');
  await osxSign({
    app: appPath,
    identity,
    platform: 'darwin',
    optionsForFile: () => ({ entitlements, hardenedRuntime: true, timestamp: true }),
  });
  console.log('signed.');
} else {
  console.log('unsigned (pass --sign for Developer ID).');
}

rmSync(STAGE, { recursive: true, force: true });
console.log(`\n${appPath}\n\n  verify: ./packages/desktop/scripts/verify-bundle.sh "${appPath}"\n`);

function stagedAddonInBundle(resourcesPath: string): string {
  return bundledAddonPath(resourcesPath);
}

function findIdentity(): string {
  const out = execFileSync('security', ['find-identity', '-v', '-p', 'codesigning'], { encoding: 'utf-8' });
  const match = /"(Developer ID Application: [^"]+)"/.exec(out);
  if (match === null) {
    console.error('\nNo "Developer ID Application" identity in the keychain.\n');
    process.exit(1);
  }
  return match[1];
}

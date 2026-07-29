/**
 * Build `packages/desktop/out/mlx-node-darwin-arm64/mlx-node.app`.
 *
 *   yarn workspace @mlx-node/desktop package                       unsigned, for local runs
 *   yarn workspace @mlx-node/desktop package --sign                Developer ID + hardened runtime
 *   yarn workspace @mlx-node/desktop package --app-version 0.0.9   stamp a version other than the manifest's
 *
 * Signing needs an identity in the keychain and `APPLE_TEAM_ID`. Notarization is
 * NOT here — it belongs to `.github/workflows/desktop-release.yml`, which submits
 * and staples. Verify either result with `scripts/verify-bundle.ts`.
 *
 * The bundle path is NOT predictable from this file's constants — packager owns
 * the platform directory name — so it is written to `out/package-result.json`
 * instead of re-derived by every consumer.
 */

import { execFileSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { packager } from '@electron/packager';

import { maxOsVersion, parseMinOs } from './min-os.js';
import { bundledAddonPath, codesignArgs, NATIVE_FILES, PayloadError, resolvePayload } from './payload.js';
import { assertSemver } from './release-version.js';
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
const RESULT_FILE = join(OUT, 'package-result.json');

const sign = process.argv.includes('--sign');

// The manifest is `private: true`, and `tools bump` only bumps public workspaces,
// so on a tagged release the manifest version is stale. The release workflow
// passes the tag's version here; see `scripts/release-version.ts`.
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
console.log(`  pruned ${staged.prunedDirs} examples/docs/test dirs from staged packages`);
console.log(`  pruned ${staged.prunedFiles} .d.ts / source-map / tsbuildinfo files`);
if (staged.excludedProviderSdk.length > 0) {
  console.log(
    `  excluded ${staged.excludedProviderSdk.join(', ')} — cloud LLM SDKs, unreachable without a pi-ai stream()`,
  );
}
if (staged.prunedNested.length > 0) {
  console.log(
    `  removed ${staged.prunedNested.join(', ')} from nested node_modules — copied by directory, not by name`,
  );
}
if (staged.excludedThirdParty.length > 0) {
  console.log(`  excluded ${staged.excludedThirdParty.join(', ')} — leaks a build path, fails gate [3/5]`);
}
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
  appVersion,
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

// Asserted HERE, at the one place that knows how the path was derived. Without it
// a packager layout change surfaces three steps later as an ENOENT on some file
// deep inside a bundle that was never there, and the reported path looks correct.
//
// Info.plist and not just the directory: `appPath = outDir` — the mistake this
// whole file's comment warns about — names a directory that DOES exist, so an
// existence check alone passes and the run still dies later on a plausible path.
if (!existsSync(join(appPath, 'Contents', 'Info.plist'))) {
  console.error(`\npackager returned ${outDir} but ${appPath} is not a bundle — the layout changed.\n`);
  process.exit(1);
}

// extraResource keeps the source directory name; the app expects `www`.
const resources = join(appPath, 'Contents', 'Resources');
rmSync(join(resources, 'www'), { recursive: true, force: true });
cpSync(join(resources, 'web'), join(resources, 'www'), { recursive: true });
rmSync(join(resources, 'web'), { recursive: true, force: true });

console.log(`addon will load from: ${bundledAddonPath(resources)}`);

// Packager writes Electron's template floor (12.0). The addon is built with
// MACOSX_DEPLOYMENT_TARGET=26.0 on release, so on macOS 12-15 the app would open
// and only then fail to dlopen it — which the supervisor treats as a crashed
// sidecar and restarts forever. Must precede signing: editing Info.plist after a
// signature invalidates it.
const minOsProbes = [
  bundledAddonPath(resources),
  join(appPath, 'Contents', 'MacOS', PRODUCT_NAME),
  join(appPath, 'Contents', 'Frameworks', 'Electron Framework.framework', 'Versions', 'A', 'Electron Framework'),
].filter((f) => existsSync(f));
const floors = minOsProbes
  .map((f) => parseMinOs(execFileSync('otool', ['-l', f], { encoding: 'utf-8', maxBuffer: 64 * 1024 * 1024 })))
  .filter((v): v is string => v !== null);
if (floors.length === 0) {
  console.error(`\nno LC_BUILD_VERSION on any of:\n  ${minOsProbes.join('\n  ')}\n`);
  process.exit(1);
}
const minimumSystemVersion = maxOsVersion(floors);
run('plutil', [
  '-replace',
  'LSMinimumSystemVersion',
  '-string',
  minimumSystemVersion,
  join(appPath, 'Contents', 'Info.plist'),
]);
console.log(`LSMinimumSystemVersion: ${minimumSystemVersion} (highest minos of ${floors.length} binaries)`);

if (sign) {
  const identity = process.env.SIGN_IDENTITY ?? findIdentity();
  const entitlements = join(APP_DIR, 'build', 'entitlements.mac.plist');

  // Signed explicitly rather than with @electron/osx-sign. That library decides
  // what to sign by SNIFFING FILE CONTENT (isBinaryFile), and with `asar: false`
  // every asset is a loose file — so it tried to codesign
  // `.../doom-overlay/doom/build/doom.wasm` and then `.../assets/clankolas.png`
  // and died on each. `file` reports the actual Mach-O header instead, which is
  // the same detection the release gate uses, so what gets signed here and what
  // gets checked there cannot drift apart.
  const machO = execFileSync(
    'bash',
    [
      '-c',
      `find ${JSON.stringify(appPath)} -type f -print0 | xargs -0 file | awk '/Mach-O/ { i = index($0, ":"); if (i == 0) next; n = substr($0, 1, i - 1); sub(/ \\(for architecture [^)]*\\)$/, "", n); print n }' | sort -u`,
    ],
    { encoding: 'utf-8', maxBuffer: 64 * 1024 * 1024 },
  )
    .split('\n')
    .filter(Boolean);

  // Inside-out. Signing a bundle seals whatever its nested code looks like at that
  // moment, so anything signed afterwards breaks the outer seal. Deepest path
  // first gives that ordering without having to model the bundle tree.
  const byDepth = (a: string, b: string): number => b.split('/').length - a.split('/').length;

  for (const file of machO.sort(byDepth)) {
    // The addon carries `Identifier=libmlx_core.dylib`, a build-artifact name that
    // would otherwise become this app's identity for its largest binary.
    const identifier = file.endsWith(NATIVE_FILES[0]) ? `${BUNDLE_ID}.mlx-core` : undefined;
    run('codesign', codesignArgs({ file, identity, entitlements, identifier }));
  }

  // Nested .app bundles (the Electron helpers) then the outer bundle, same order.
  // The outer bundle is filtered out of the loop: `find` matches it too, and it
  // is sealed explicitly below so it is unambiguously last. Without the filter
  // it was re-sealed twice over — ~580 MB of redundant hashing per build.
  const bundles = execFileSync('bash', ['-c', `find ${JSON.stringify(appPath)} -name '*.app' -type d`], {
    encoding: 'utf-8',
  })
    .split('\n')
    .filter(Boolean)
    .filter((bundle) => bundle !== appPath);
  for (const bundle of bundles.sort(byDepth)) {
    run('codesign', codesignArgs({ file: bundle, identity, entitlements }));
  }
  run('codesign', codesignArgs({ file: appPath, identity, entitlements }));

  console.log(`signed ${machO.length} Mach-O + ${bundles.length + 1} bundles.`);
} else {
  console.log('unsigned (pass --sign for Developer ID).');
}

rmSync(STAGE, { recursive: true, force: true });

// One canonical path for everything downstream. The release workflow used to
// spell `out/mlx-node.app` in four places; packager puts the bundle inside a
// PLATFORM directory, so all four were wrong and every step after packaging
// failed. Consumers read this file rather than guessing the layout again.
writeFileSync(
  RESULT_FILE,
  `${JSON.stringify(
    {
      appPath,
      version: appVersion,
      minimumSystemVersion,
      bundleId: BUNDLE_ID,
      productName: PRODUCT_NAME,
      signed: sign,
    },
    null,
    2,
  )}\n`,
);

console.log(`\n${appPath}\n\n  verify: yarn oxnode packages/desktop/scripts/verify-bundle.ts "${appPath}"\n`);

function findIdentity(): string {
  const out = execFileSync('security', ['find-identity', '-v', '-p', 'codesigning'], { encoding: 'utf-8' });
  const match = /"(Developer ID Application: [^"]+)"/.exec(out);
  if (match === null) {
    console.error('\nNo "Developer ID Application" identity in the keychain.\n');
    process.exit(1);
  }
  return match[1];
}

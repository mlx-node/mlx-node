import { readFile, writeFile, copyFile, stat, mkdir, rm, readdir, cp } from 'node:fs/promises';
import { join, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

import { NapiCli, createBuildCommand } from '@napi-rs/cli';
import { format } from 'vite-plus/fmt';

import viteConfig from '../../vite.config';
import {
  assertAddonMinOs,
  assertMetallibFloor,
  assertMetallibIntegrity,
  assertPagedMetallibIntegrity,
  hostAppleTriple,
  MACOS_DEPLOYMENT_FLOOR,
  profileDirName,
  resolveTargetRoot,
  selectMetallib,
  selectPagedMetallib,
} from './metallib-select';

const __dirname = dirname(fileURLToPath(import.meta.url));
const packageDeclarationsPath = join(__dirname, 'index.d.cts');
const packageJsBindingPath = join(__dirname, 'index.cjs');
const crateDeclarationsPath = join(__dirname, '../../crates/mlx-core/index.d.cts');

async function assertDeclarationCopiesMatch(phase: string): Promise<void> {
  const [packageDeclarations, crateDeclarations] = await Promise.all([
    readFile(packageDeclarationsPath, 'utf-8'),
    readFile(crateDeclarationsPath, 'utf-8'),
  ]);
  if (packageDeclarations !== crateDeclarations) {
    throw new Error(
      `[build.ts declaration drift] packages/core/index.d.cts and crates/mlx-core/index.d.cts differ ${phase}. ` +
        'Regenerate the native declarations, synchronize both committed copies, and retry.',
    );
  }
}

// Non-macOS NAPI output omits target-gated macOS exports. Preserve the committed
// cross-platform binding surface instead of replacing it with that subset.
const preservedBindings =
  process.platform === 'linux' || process.platform === 'win32'
    ? await Promise.all(
        [packageDeclarationsPath, packageJsBindingPath].map(
          async (path) => [path, await readFile(path, 'utf8')] as const,
        ),
      )
    : null;

// Both paths are published/reviewed surfaces. Check the committed inputs
// before generation so an already-stale package copy cannot be hidden when
// napi overwrites it, then check again below against the freshly generated
// output so a native API change cannot silently leave the crate copy stale.
await assertDeclarationCopiesMatch('before native generation');

const buildCommand = createBuildCommand(process.argv.slice(2));
const cli = new NapiCli();
const buildOptions = buildCommand.getOptions();

const { task } = await cli.build({
  ...buildOptions,
  manifestPath: join(__dirname, '../../crates/mlx-core/Cargo.toml'),
  packageJsonPath: join(__dirname, 'package.json'),
  platform: true,
  outputDir: __dirname,
  jsBinding: 'index.cjs',
  dts: 'index.d.cts',
});
const outputs = await task;

for (const output of outputs) {
  if (output.kind !== 'node') {
    const { code } = await format(output.path, await readFile(output.path, 'utf-8'), viteConfig.fmt);
    await writeFile(output.path, code);
  }
  if (output.kind === 'dts') {
    const code = await readFile(output.path, 'utf-8');
    const replaced = code.replace('export declare const enum OutputFormat {', 'export enum OutputFormat {');
    await writeFile(output.path, replaced);
  }
}

if (preservedBindings) {
  await Promise.all(preservedBindings.map(([path, contents]) => writeFile(path, contents)));
}
await assertDeclarationCopiesMatch('after native generation');

const metalBuild = process.platform === 'darwin' && process.env.MLX_DISABLE_METAL == null;
if (metalBuild) {
  const nodeOutput = outputs.find((output) => output.kind === 'node');
  if (nodeOutput) {
    assertAddonMinOs(await readFile(nodeOutput.path), nodeOutput.path);
  }
}
await copyNativeAddon(outputs);
if (process.platform === 'win32' && process.arch === 'arm64') {
  await copyWindowsCudaRuntime();
}
// Copy mlx.metallib for colocated Metal shader loading
// MLX looks for metallib next to the binary, so we copy it here.
// Also copy paged_attn.metallib, which mlx_paged_dispatch.cpp loads via a
// `dladdr`-based colocated lookup at runtime.
//
// Both metallibs are copied to TWO destinations:
//   1. `packages/core/`       — sits next to the local
//      `mlx-core.darwin-arm64.node` for in-repo `yarn build:native`
//      developer flow.
//   2. `packages/core/npm/darwin-arm64/` — the platform-specific
//      optional package that gets published to npm. The user-facing
//      install path resolves the .node addon from this package, so
//      the metallibs MUST ship here too — otherwise the
//      `dladdr`-based runtime lookup in `mlx_paged_dispatch.cpp`
//      lands in the published package directory and finds no
//      `paged_attn.metallib`, throwing on first use.
//
// Both metallibs are required-on-darwin: `mlx.metallib` for stock
// MLX kernels, `paged_attn.metallib` for the paged-attention
// dispatch path used by `Qwen3Model` (where `use_paged_attention`
// is on by default for the legacy `PagedKVCache` route and by
// `use_block_paged_cache` on by default for the new vLLM-style path)
// and for every GGUF K-quant, segmented SDPA and mixed-affine kernel
// (prebuilt, no JIT fallback).
// We FAIL the build if either is missing so a packaging regression
// surfaces immediately rather than as a runtime throw at first use
// in a published install.
//
// The metallibs exist only on macOS (the Metal build). On the CUDA/Linux
// build there is no Metal toolchain and no metallib to copy, so skip the
// whole step (and its presence assert) on non-darwin platforms.
if (metalBuild) {
  await copyMetallibs(outputs);
} else if (process.platform === 'darwin') {
  await removeMetallibsForCpuOnlyBuild();
}

async function removeMetallibsForCpuOnlyBuild() {
  const destDirs = [__dirname, join(__dirname, 'npm', 'darwin-arm64')];
  const names = ['mlx.metallib', 'paged_attn.metallib'];
  for (const dest of destDirs) {
    for (const name of names) {
      await rm(join(dest, name), { force: true });
    }
  }
  console.log('Removed stale metallibs from MLX_DISABLE_METAL CPU-only build outputs.');
}

// Derive the matching `npm/<triple>/` directory from napi-rs's actual output.
// This must not use `process.arch`: an x64 Node under Rosetta can legitimately
// drive the configured aarch64 Rust build.
function nativeAddonTriple(actualName: string): string {
  const match = /^mlx-core\.(.+)\.node$/.exec(actualName);
  if (match == null) {
    throw new Error(`[build.ts] unrecognized native addon output: ${actualName}`);
  }
  const triple = match[1];
  if (process.platform === 'darwin' && !triple.startsWith('darwin-')) {
    throw new Error(`[build.ts] expected a Darwin addon, got ${actualName}`);
  }
  if (process.platform === 'linux' && !triple.startsWith('linux-')) {
    throw new Error(`[build.ts] expected a Linux addon, got ${actualName}`);
  }
  return triple;
}

async function copyNativeAddon(outputs: Awaited<typeof task>) {
  const nodeOutput = outputs.find((output) => output.kind === 'node');
  if (!nodeOutput) {
    throw new Error('[build.ts smoke check] native addon output missing from napi build');
  }
  const actualName = basename(nodeOutput.path);
  const triple = nativeAddonTriple(actualName);
  const expectedName = `mlx-core.${triple}.node`;
  if (actualName !== expectedName) {
    throw new Error(
      `[build.ts smoke check] expected native addon output ${expectedName}, got ${actualName} at ${nodeOutput.path}`,
    );
  }

  const npmPlatformDir = join(__dirname, 'npm', triple);
  // The darwin platform dir is committed (it carries the metallibs + a
  // README). The linux dir is not: its only published artifact would be a
  // .node that CI never builds, so we don't ship it as an optional package.
  // A from-source linux build still needs somewhere to land the .node, so
  // create the dir on demand (no-op when it already exists, e.g. darwin).
  await mkdir(npmPlatformDir, { recursive: true });
  const dst = join(npmPlatformDir, expectedName);
  await copyFile(nodeOutput.path, dst);
  console.log(`Copied ${expectedName} -> ${dst}`);
}

async function copyWindowsCudaRuntime() {
  const targetRoot = resolveTargetRoot({
    targetDir: buildOptions.targetDir,
    env: process.env,
    defaultRoot: join(__dirname, '../../target'),
  });
  const buildRoot = join(targetRoot, 'aarch64-pc-windows-msvc', profileDirName(buildOptions), 'build');
  const entries = await readdir(buildRoot);
  const scriptDirs = entries.filter((name) => name.startsWith('mlx-sys-')).map((name) => join(buildRoot, name));
  if (entries.includes('mlx-sys')) {
    const packageDir = join(buildRoot, 'mlx-sys');
    scriptDirs.push(...(await readdir(packageDir)).map((name) => join(packageDir, name)));
  }
  const installs = await Promise.all(
    scriptDirs.map(async (dir) => {
      const install = join(dir, 'out', 'mlx-cmake');
      const blas = await stat(join(install, 'openblas.dll')).catch((error: unknown) => {
        if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
        throw error;
      });
      return { install, mtime: blas?.mtimeMs ?? 0 };
    }),
  );
  const source = installs.filter(({ mtime }) => mtime > 0).sort((a, b) => b.mtime - a.mtime)[0]?.install;
  if (!source) throw new Error(`Cannot find the Windows MLX install in ${buildRoot}`);

  const cudaRoot = process.env.CUDA_PATH || process.env.CUDA_HOME;
  const cudnnRoot = process.env.CUDNN_PATH;
  if (!cudaRoot || !cudnnRoot) throw new Error('Set CUDA_PATH and CUDNN_PATH to the Windows ARM64 toolkits.');
  const cudaBin = join(cudaRoot, 'bin', 'arm64');
  const cudnnBin = join(cudnnRoot, 'bin', 'arm64');
  const cudaDlls = [
    'cudart64_13.dll',
    'cublas64_13.dll',
    'cublasLt64_13.dll',
    'cufft64_12.dll',
    'cusolver64_12.dll',
    'cusparse64_12.dll',
    'nvJitLink_130_0.dll',
    'nvrtc64_130_0.dll',
  ];
  const nvrtcBuiltins = (await readdir(cudaBin)).filter((name) => /^nvrtc-builtins64_.*\.dll$/i.test(name));
  const cudnnDlls = (await readdir(cudnnBin)).filter((name) => /^cudnn.*\.dll$/i.test(name));
  if (nvrtcBuiltins.length === 0 || !cudnnDlls.includes('cudnn64_9.dll')) {
    throw new Error('CUDA_PATH must contain NVRTC builtins and CUDNN_PATH must contain cuDNN 9.');
  }
  const destinations = [__dirname, join(__dirname, 'npm', 'win32-arm64-msvc')];
  for (const dest of destinations) {
    await copyFile(join(source, 'openblas.dll'), join(dest, 'openblas.dll'));
    for (const name of [...cudaDlls, ...nvrtcBuiltins]) {
      await copyFile(join(cudaBin, name), join(dest, name));
    }
    for (const name of cudnnDlls) {
      await copyFile(join(cudnnBin, name), join(dest, name));
    }
    for (const include of ['cccl', 'cute', 'cutlass']) {
      await cp(join(source, 'include', include), join(dest, 'include', include), { recursive: true });
    }
  }
}

// Publish/release fail-closed switch (set as `MLX_METALLIB_STRICT=1` in the
// CI workflow env — see .github/workflows/ci.yml). In strict mode the metallib
// gates FAIL CLOSED: a Metal-enabled addon that bakes no METAL_PATH aborts the
// build instead of scanning possibly-stale mlx-sys-* dirs, and a min-OS stamp
// that cannot be parsed aborts instead of skipping the deployment-floor check.
// Local `yarn build:native` leaves it unset and keeps the lenient
// warn-and-continue behavior so a CPU-only / exotic dev build still works.
function metallibStrictMode(): boolean {
  const v = process.env.MLX_METALLIB_STRICT;
  return v === '1' || v === 'true';
}

// The min-OS load floor build.rs compiled the metallibs for: an explicit
// MACOSX_DEPLOYMENT_TARGET, else MACOS_DEPLOYMENT_FLOOR (a Metal build below
// it fails in build.rs, so no lower floor reaches this gate).
function detectDeploymentFloor(): string {
  const env = process.env.MACOSX_DEPLOYMENT_TARGET;
  return env !== undefined && env !== '' ? env : MACOS_DEPLOYMENT_FLOOR;
}

async function copyMetallibs(outputs: Awaited<typeof task>) {
  const npmDarwinDir = join(__dirname, 'npm', 'darwin-arm64');
  const destDirs = [__dirname, npmDarwinDir];

  // Authoritative binding: the freshly built addon bakes the METAL_PATH of
  // the exact mlx-sys build it linked against; ship that build's metallibs.
  // The target-dir/triple/profile derivation only feeds the heuristic scan
  // used when no path is baked (see metallib-select.ts).
  const nodeOutput = outputs.find((output) => output.kind === 'node');
  if (!nodeOutput) {
    throw new Error('[build.ts smoke check] native addon output missing from napi build');
  }
  const targetRoot = resolveTargetRoot({
    targetDir: buildOptions.targetDir,
    env: process.env,
    defaultRoot: join(__dirname, '../../target'),
  });
  const triple = buildOptions.target ?? process.env.CARGO_BUILD_TARGET ?? hostAppleTriple();
  const profile = profileDirName(buildOptions);
  const strict = metallibStrictMode();
  const picked = selectMetallib({
    addonBinary: await readFile(nodeOutput.path),
    addonPath: nodeOutput.path,
    targetRoot,
    triple,
    profile,
    strict,
    warn: (msg) => console.warn(msg),
  });
  console.log(
    {
      baked: `Metallib bound via the addon's baked METAL_PATH: ${picked.metallibPath}`,
      prebuilt: `Metallib bound via the addon's prebuilt-MLX sentinel: ${picked.metallibPath}`,
      scan: `Metallib selected by directory scan (no baked METAL_PATH): ${picked.metallibPath}`,
    }[picked.source],
  );

  // Hard gates: never ship a truncated or stale-pin metallib (wrong kernels
  // paired with the fresh addon produce garbage inference with no error at
  // load time), nor one stamped above the intended deployment floor (it
  // would refuse to load on floor machines).
  const deploymentFloor = detectDeploymentFloor();
  const metallib = await readFile(picked.metallibPath);
  assertMetallibIntegrity(metallib, { path: picked.metallibPath });
  assertMetallibFloor(metallib, {
    path: picked.metallibPath,
    deploymentFloor,
    strict,
    warn: (msg) => console.warn(msg),
  });

  for (const dest of destDirs) {
    const dst = join(dest, 'mlx.metallib');
    await copyFile(picked.metallibPath, dst);
    console.log(`Copied mlx.metallib -> ${dst}`);
  }
  // paged_attn.metallib is also required for darwin, under the same
  // contract as mlx.metallib: build.rs writes the origin to the OUT_DIR
  // root and copies it into out/lib; both-present pairs must be
  // byte-identical, and the shipped file passes size/magic + floor gates
  // before any copy.
  const paged = selectPagedMetallib({
    outDir: picked.outDir,
    libDir: picked.libDir,
    warn: (msg) => console.warn(msg),
  });
  assertPagedMetallibIntegrity(paged.contents, { path: paged.path });
  assertMetallibFloor(paged.contents, {
    path: paged.path,
    deploymentFloor,
    strict,
    warn: (msg) => console.warn(msg),
  });
  for (const dest of destDirs) {
    const dst = join(dest, 'paged_attn.metallib');
    await copyFile(paged.path, dst);
    console.log(`Copied paged_attn.metallib -> ${dst}`);
  }
  // Final sanity-check: every destination must have BOTH files.
  // This catches a copy that silently overwrote or partially
  // failed; cheaper to fail the build than to publish a broken
  // optional package.
  await assertMetallibPresence(destDirs);
}

async function assertMetallibPresence(destDirs: string[]) {
  const required = ['mlx.metallib', 'paged_attn.metallib'];
  for (const dest of destDirs) {
    for (const name of required) {
      const p = join(dest, name);
      try {
        await stat(p);
      } catch {
        throw new Error(
          `[build.ts smoke check] expected ${name} at ${p} but it is missing. ` +
            `If this fires, the published npm package will not contain this metallib ` +
            `and the runtime dladdr lookup will throw on first use.`,
        );
      }
    }
    console.log(`Smoke check: ${dest} has all ${required.length} required metallibs.`);
  }
}

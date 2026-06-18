import { readFile, writeFile, copyFile, readdir, stat, mkdir } from 'node:fs/promises';
import { join, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

import { NapiCli, createBuildCommand } from '@napi-rs/cli';
import { format } from 'vite-plus/fmt';

import viteConfig from '../../vite.config';

const __dirname = dirname(fileURLToPath(import.meta.url));
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

await copyNativeAddon(outputs);
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
// `use_block_paged_cache` on by default for the new vLLM-style path).
// We FAIL the build if either is missing so a packaging regression
// surfaces immediately rather than as a runtime throw at first use
// in a published install.
//
// The metallibs exist only on macOS (the Metal build). On the CUDA/Linux
// build there is no Metal toolchain and no metallib to copy, so skip the
// whole step (and its presence assert) on non-darwin platforms.
if (process.platform === 'darwin') {
  await copyMetallibs();
}

// Derive the napi addon file name + the matching `npm/<triple>/` directory
// from the current platform/arch. napi-rs emits `mlx-core.<triple>.node`
// where the triple is e.g. `darwin-arm64` or `linux-arm64-gnu`.
function nativeAddonTriple(): string {
  if (process.platform === 'darwin') {
    return `darwin-${process.arch}`;
  }
  if (process.platform === 'linux') {
    // glibc only for this milestone (GB10 is glibc); musl is out of scope.
    return `linux-${process.arch}-gnu`;
  }
  throw new Error(`[build.ts] unsupported platform for native addon: ${process.platform}`);
}

async function copyNativeAddon(outputs: Awaited<typeof task>) {
  const nodeOutput = outputs.find((output) => output.kind === 'node');
  if (!nodeOutput) {
    throw new Error('[build.ts smoke check] native addon output missing from napi build');
  }
  const triple = nativeAddonTriple();
  const expectedName = `mlx-core.${triple}.node`;
  const actualName = basename(nodeOutput.path);
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

async function copyMetallibs() {
  const npmDarwinDir = join(__dirname, 'npm', 'darwin-arm64');
  const destDirs = [__dirname, npmDarwinDir];

  const targetDir = join(__dirname, '../../target');
  // Find mlx.metallib in the build directory.
  // Pattern: target/*/release/build/mlx-sys-*/out/lib/mlx.metallib
  const archDirs = await readdir(targetDir);
  for (const arch of archDirs) {
    const releaseDir = join(targetDir, arch, 'release', 'build');
    let buildDirs: string[];
    try {
      buildDirs = await readdir(releaseDir);
    } catch {
      // release/build dir doesn't exist for this arch
      continue;
    }
    for (const dir of buildDirs) {
      if (!dir.startsWith('mlx-sys-')) continue;
      const libDir = join(releaseDir, dir, 'out', 'lib');
      const mlxPath = join(libDir, 'mlx.metallib');
      try {
        await stat(mlxPath);
      } catch {
        // metallib not at this path, continue searching
        continue;
      }
      // mlx.metallib is required: copy to all destinations or fail.
      for (const dest of destDirs) {
        const dst = join(dest, 'mlx.metallib');
        await copyFile(mlxPath, dst);
        console.log(`Copied mlx.metallib -> ${dst}`);
      }
      // paged_attn.metallib is also required for darwin.
      // It lives next to mlx.metallib in the same lib dir.
      const pagedPath = join(libDir, 'paged_attn.metallib');
      try {
        await stat(pagedPath);
      } catch {
        throw new Error(
          `paged_attn.metallib not found at ${pagedPath}. The paged-attention ` +
            `compile path (mlx_paged_dispatch.cpp) loads this metallib via dladdr ` +
            `at runtime; without it, the addon throws on first paged-attention use. ` +
            `Check that mlx-sys/build.rs ran compile_paged_attn_metallib successfully.`,
        );
      }
      for (const dest of destDirs) {
        const dst = join(dest, 'paged_attn.metallib');
        await copyFile(pagedPath, dst);
        console.log(`Copied paged_attn.metallib -> ${dst}`);
      }
      // Final sanity-check: every destination must have BOTH files.
      // This catches a copy that silently overwrote or partially
      // failed; cheaper to fail the build than to publish a broken
      // optional package.
      await assertMetallibPresence(destDirs);
      return;
    }
  }
  throw new Error('mlx.metallib not found under any target/<arch>/release/build/mlx-sys-*/out/lib/');
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

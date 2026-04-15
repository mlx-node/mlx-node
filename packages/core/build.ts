import { readFile, writeFile, copyFile, readdir, stat } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

import { NapiCli, createBuildCommand } from '@napi-rs/cli';
import { format } from 'vite-plus/fmt';

import viteConfig from '../../vite.config';

const __dirname = dirname(fileURLToPath(import.meta.url));
const buildCommand = createBuildCommand(process.argv.slice(2));
const cli = new NapiCli();
const buildOptions = buildCommand.getOptions();

Object.assign(buildOptions, {
  manifestPath: join(__dirname, '../../crates/mlx-core/Cargo.toml'),
  packageJsonPath: join(__dirname, 'package.json'),
  platform: true,
  outputDir: __dirname,
});

const isWasmBuild = buildOptions.target === 'wasm32-wasip1-threads';

if (isWasmBuild) {
  // tokio_unstable: enables multi-thread tokio runtime on WASM (requires
  // asyncWorkPoolSize > 0 in emnapi instantiation so wasi_thread_spawn works).
  process.env.RUSTFLAGS = '--cfg tokio_unstable';
  // esaxx-rs (pulled in by tokenizers) uses C++ try/catch — needs -fexceptions
  // globally. mlx-sys/build.rs sets these for its own cmake/cc::Build, but
  // esaxx-rs uses its own cc::Build that reads TARGET_CXXFLAGS from env.
  process.env.TARGET_CXXFLAGS = '-fwasm-exceptions -fexceptions';
  Object.assign(buildOptions, {
    noJsBinding: true,
    noDefaultFeatures: true,
    features: ['browser'],
  });
} else {
  Object.assign(buildOptions, {
    jsBinding: 'index.cjs',
    dts: 'index.d.cts',
  });
}

// Snapshot hand-written wasi-worker-browser.mjs BEFORE cli.build() overwrites it.
// NAPI-RS always regenerates this file, but ours is the source of truth.
const wasiWorkerBrowserPath = join(__dirname, 'wasi-worker-browser.mjs');
const wasiWorkerBrowserSrc = isWasmBuild ? await readFile(wasiWorkerBrowserPath, 'utf-8').catch(() => null) : null;

const { task } = await cli.build(buildOptions);
const outputs = await task;

for (const output of outputs) {
  // Skip formatting wasi-worker-browser.mjs — hand-written, not auto-generated
  if (output.kind !== 'node' && !output.path.endsWith('wasi-worker-browser.mjs')) {
    const { code } = await format(output.path, await readFile(output.path, 'utf-8'), viteConfig.fmt);
    await writeFile(output.path, code);
  }
  if (output.kind === 'dts' && !isWasmBuild) {
    const code = await readFile(output.path, 'utf-8');
    const replaced = code.replace('export declare const enum OutputFormat {', 'export enum OutputFormat {');
    await writeFile(output.path, replaced);
  }
}

// Restore hand-written wasi-worker-browser.mjs after NAPI-RS overwrote it
if (wasiWorkerBrowserSrc) {
  await writeFile(wasiWorkerBrowserPath, wasiWorkerBrowserSrc);
}

// Copy mlx.metallib for colocated Metal shader loading
// MLX looks for metallib next to the binary, so we copy it here
// Skip for WASM targets (no Metal shaders)
const target = process.argv.find((a) => a.includes('wasm32'));
if (!target) {
  await copyMetallib();
} else {
  // Copy raw WASM binary to the runtime location (symlinked by browser demo)
  const rawWasm = join(
    __dirname,
    '../../target/wasm32-wasip1-threads',
    buildOptions.profile ?? 'release',
    'mlx_core.wasm',
  );
  const runtimeWasm = join(__dirname, 'mlx-core.wasm32-wasi.opt.wasm');
  await copyFile(rawWasm, runtimeWasm);
  console.log(`Copied WASM: ${rawWasm} -> ${runtimeWasm}`);
}

async function copyMetallib() {
  const targetDir = join(__dirname, '../../target');
  try {
    // Find mlx.metallib in the build directory
    // Pattern: target/*/release/build/mlx-sys-*/out/lib/mlx.metallib
    const archDirs = await readdir(targetDir);
    for (const arch of archDirs) {
      const releaseDir = join(targetDir, arch, 'release', 'build');
      try {
        const buildDirs = await readdir(releaseDir);
        for (const dir of buildDirs) {
          if (dir.startsWith('mlx-sys-')) {
            const metallibPath = join(releaseDir, dir, 'out', 'lib', 'mlx.metallib');
            try {
              await stat(metallibPath);
              await copyFile(metallibPath, './mlx.metallib');
              console.log('Copied mlx.metallib');
              return;
            } catch {
              // metallib not at this path, continue searching
            }
          }
        }
      } catch {
        // release/build dir doesn't exist for this arch
      }
    }
    throw new Error('Note: mlx.metallib not found');
  } catch {
    throw new Error('Note: mlx.metallib not found');
  }
}

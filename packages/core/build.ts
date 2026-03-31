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

// Copy mlx.metallib for colocated Metal shader loading
// MLX looks for metallib next to the binary, so we copy it here
// Skip for WASM targets (no Metal shaders)
const target = process.argv.find(a => a.includes('wasm32'));
if (!target) {
  await copyMetallib();
} else {
  // Patch generated WASM browser/worker files with extra imports needed by
  // MLX (C++ exception stubs, WebGPU bridge no-ops, GPU init).
  // These are WASM imports not provided by emnapi/WASI.
  await patchWasmEntries();
}

// Extra WASM imports needed by MLX that emnapi/WASI don't provide.
// Injected into both the browser entry and worker entry files.
const MLX_EXTRA_IMPORTS = `
      // C++ exception stubs (libc++abi leaves these as imports)
      __cxa_allocate_exception: () => 0,
      __cxa_throw: () => { throw new Error('C++ exception thrown in WASM'); },
      __cxa_init_primary_exception: (ptr) => ptr,
      // MLX GPU init — no-op (GPU initialized lazily via WebGPU bridge)
      _ZN3mlx4core3gpu4initEv: () => {},
      // WebGPU stubs (real bridge injected by consumer via overwriteImports)
      wgpuCreateInstance: () => 0, wgpuInstanceRequestAdapter: () => {},
      wgpuInstanceRelease: () => {}, wgpuAdapterRequestDevice: () => {},
      wgpuAdapterRelease: () => {}, wgpuDeviceSetUncapturedErrorCallback: () => {},
      wgpuDeviceSetDeviceLostCallback: () => {}, wgpuDeviceGetQueue: () => 0,
      mlx_webgpu_poll: () => {}, wgpuDeviceCreateComputePipeline: () => 0,
      wgpuComputePipelineGetBindGroupLayout: () => 0,
      wgpuDeviceCreateShaderModule: () => 0, wgpuQueueOnSubmittedWorkDone: () => {},
      wgpuAdapterGetProperties: () => {}, wgpuDeviceGetLimits: () => 0,
      wgpuCommandEncoderRelease: () => {}, wgpuComputePassEncoderEnd: () => {},
      wgpuComputePassEncoderRelease: () => {},
      wgpuDeviceCreateCommandEncoder: () => 0,
      wgpuCommandEncoderBeginComputePass: () => 0,
      wgpuComputePassEncoderSetPipeline: () => {},
      wgpuComputePassEncoderSetBindGroup: () => {},
      wgpuComputePassEncoderDispatchWorkgroups: () => {},
      wgpuCommandEncoderFinish: () => 0, wgpuQueueSubmit: () => {},
      wgpuCommandBufferRelease: () => {}, wgpuDeviceCreateBuffer: () => 0,
      wgpuBufferDestroy: () => {}, wgpuBufferRelease: () => {},
      wgpuCommandEncoderCopyBufferToBuffer: () => {},
      wgpuBufferMapAsync: () => {}, wgpuBufferGetConstMappedRange: () => 0,
      wgpuBufferUnmap: () => {}, wgpuBufferGetSize: () => 0,
      wgpuBindGroupRelease: () => {}, wgpuDeviceCreateBindGroup: () => 0,
      wgpuBufferGetMappedRange: () => 0,`;

async function patchWasmEntries() {
  for (const file of ['mlx-core.wasi-browser.js', 'wasi-worker-browser.mjs']) {
    const filePath = join(__dirname, file);
    try {
      let code = await readFile(filePath, 'utf-8');
      // Inject extra imports after the "memory: ..." line in overwriteImports
      if (!code.includes('__cxa_allocate_exception')) {
        code = code.replace(
          /memory:\s*(?:__sharedMemory|wasmMemory),?\n/,
          (match) => match + MLX_EXTRA_IMPORTS + '\n',
        );
        await writeFile(filePath, code);
        console.log(`Patched ${file} with MLX extra imports`);
      }
    } catch {
      // File might not exist for non-WASM builds
    }
  }
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

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
// MLX looks for metallib next to the binary, so we copy it here.
// Also copy paged_attn.metallib (Phase 2 of the paged-attention
// compile integration), which mlx_paged_dispatch.cpp loads via a
// `dladdr`-based colocated lookup at runtime.
await copyMetallibs();

async function copyMetallibs() {
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
            const libDir = join(releaseDir, dir, 'out', 'lib');
            const mlxPath = join(libDir, 'mlx.metallib');
            try {
              await stat(mlxPath);
              await copyFile(mlxPath, './mlx.metallib');
              console.log('Copied mlx.metallib');
              // Best-effort copy of the paged-attn metallib (Phase 2).
              // It lives next to mlx.metallib in the same lib dir.
              const pagedPath = join(libDir, 'paged_attn.metallib');
              try {
                await stat(pagedPath);
                await copyFile(pagedPath, './paged_attn.metallib');
                console.log('Copied paged_attn.metallib');
              } catch {
                console.warn('Note: paged_attn.metallib not found (Phase 2 paged-attn compile path will throw at first use)');
              }
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

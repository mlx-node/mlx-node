import { readFile, writeFile, copyFile, readdir, stat } from 'node:fs/promises';
import { join } from 'node:path';

import { NapiCli, createBuildCommand } from '@napi-rs/cli';
import { format } from 'oxfmt';

import viteConfig from '../../vite.config';

const buildCommand = createBuildCommand(process.argv.slice(2));
const cli = new NapiCli();
const buildOptions = buildCommand.getOptions();

const { task } = await cli.build({
  ...buildOptions,
  manifestPath: '../../crates/mlx-core/Cargo.toml',
  platform: true,
  outputDir: '.',
  jsBinding: 'index.cjs',
  dts: 'index.d.cts',
});
const outputs = await task;

for (const output of outputs) {
  if (output.kind !== 'node') {
    const { code } = await format(output.path, await readFile(output.path, 'utf-8'), viteConfig.fmt);
    await writeFile(output.path, code);
  }
}

// Copy mlx.metallib for colocated Metal shader loading
// MLX looks for metallib next to the binary, so we copy it here
await copyMetallib();

async function copyMetallib() {
  const targetDir = '../../target';
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

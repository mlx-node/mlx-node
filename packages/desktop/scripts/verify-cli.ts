/** Exercise the packaged CLI through the same launcher used by coding agents. */
import { execFileSync } from 'node:child_process';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { createCliLauncher } from '../src/cli-launcher.js';

export async function verifyBundledCli(appPath: string): Promise<void> {
  const home = await mkdtemp(join(tmpdir(), 'mlx-cli-package-check-'));
  const resources = join(appPath, 'Contents', 'Resources');
  try {
    const launcher = createCliLauncher(
      {
        executable: join(appPath, 'Contents', 'MacOS', 'mlx-node'),
        entry: join(resources, 'app', 'node_modules', '@mlx-node', 'cli', 'dist', 'cli.js'),
        nativeAddon: join(resources, 'native', 'mlx-core.darwin-arm64.node'),
        modelsDir: join(home, 'models'),
      },
      home,
    );
    const path = await launcher.prepare();
    // --version enters the full agent runtime before exiting, so missing lazy
    // dependencies and native runtime incompatibilities cannot hide behind help.
    const version = execFileSync(path, ['agent', '--version'], {
      env: { PATH: '/usr/bin:/bin:/usr/sbin:/sbin', HOME: home, PI_CODING_AGENT_DIR: join(home, 'agent') },
      encoding: 'utf8',
      timeout: 30_000,
      maxBuffer: 256 * 1024,
    }).trim();
    if (!/^\d+\.\d+\.\d+/.test(version)) throw new Error(`The packaged agent did not report its version: ${version}`);
    console.log(`packaged CLI verified without global Node/mlx (agent ${version})`);
  } finally {
    await rm(home, { recursive: true, force: true });
  }
}

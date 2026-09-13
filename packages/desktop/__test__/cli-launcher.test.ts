import { execFile } from 'node:child_process';
import { chmod, mkdir, mkdtemp, readFile, realpath, rm, stat, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { promisify } from 'node:util';

import { afterEach, describe, expect, it } from 'vite-plus/test';

import { createCliLauncher, type DesktopCliConfig } from '../src/cli-launcher.js';

const execute = promisify(execFile);
const roots: string[] = [];
afterEach(async () => {
  for (const root of roots.splice(0)) await rm(root, { recursive: true, force: true });
});

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), "mlx launcher ' $ test-"));
  roots.push(root);
  const entry = join(root, 'Application Space.app', 'cli.mjs');
  const nativeAddon = join(root, 'native.node');
  await mkdir(dirname(entry), { recursive: true });
  await writeFile(nativeAddon, 'fixture');
  await writeFile(
    entry,
    `
    import { appendFileSync } from 'node:fs';
    if (process.argv[2] === 'delegate' && process.argv[3] === '--help') {
      appendFileSync(${JSON.stringify(join(root, 'probes'))}, '1');
      console.log('Usage: mlx delegate\\nmlx delegate github');
    } else console.log(JSON.stringify({ args:process.argv.slice(2), cwd:process.cwd(),
      caller:process.env.CODEX_THREAD_ID, approval:process.env.MLX_AGENT_AUTO_APPROVE,
      models:process.env.MLX_MODELS_DIR, addon:process.env.NAPI_RS_NATIVE_LIBRARY_PATH,
      nodeMode:process.env.ELECTRON_RUN_AS_NODE }));
  `,
  );
  const config: DesktopCliConfig = {
    executable: process.execPath,
    entry,
    nativeAddon,
    modelsDir: join(root, 'models'),
  };
  return { root, config, launcher: createCliLauncher(config, root) };
}

describe('app-owned command', () => {
  it('runs without Node or mlx on PATH, quotes paths/arguments, and preserves caller permissions', async () => {
    const { root, config, launcher } = await fixture();
    const path = await launcher.prepare();
    const args = ['github', '--repo', 'owner/repo', 'PR #1: spaces; $(echo untouched)', ''];
    const { stdout } = await execute(path, args, {
      cwd: root,
      env: { PATH: '/usr/bin:/bin', CODEX_THREAD_ID: 'caller-thread' },
    });
    expect(JSON.parse(stdout)).toEqual({
      args,
      cwd: await realpath(root),
      caller: 'caller-thread',
      models: config.modelsDir,
      addon: config.nativeAddon,
      nodeMode: '1',
    });
    expect((await stat(path)).mode & 0o777).toBe(0o755);
  });

  it('coalesces probes and caches success until a runtime file changes', async () => {
    const { root, config, launcher } = await fixture();
    await Promise.all([launcher.prepare(), launcher.prepare()]);
    await launcher.prepare();
    expect(await readFile(join(root, 'probes'), 'utf8')).toBe('1');
    await writeFile(config.entry, (await readFile(config.entry, 'utf8')) + '\n// updated\n');
    await launcher.prepare();
    expect(await readFile(join(root, 'probes'), 'utf8')).toBe('11');
  });

  it('updates the stable launcher after the app moves and repairs its executable bit', async () => {
    const { root, config, launcher } = await fixture();
    await launcher.prepare();
    const moved = join(root, 'Moved.app', 'cli.mjs');
    await mkdir(dirname(moved), { recursive: true });
    await writeFile(moved, await readFile(config.entry));
    await rm(config.entry);
    const next = createCliLauncher({ ...config, entry: moved }, root);
    expect(await next.prepare()).toBe(launcher.path);
    await chmod(launcher.path, 0o600);
    await next.prepare();
    expect((await stat(launcher.path)).mode & 0o777).toBe(0o755);
    expect(await readFile(launcher.path, 'utf8')).toContain('Moved.app');
  });

  it('does not overwrite a foreign command or follow its symlink', async () => {
    const { root, launcher } = await fixture();
    await mkdir(dirname(launcher.path), { recursive: true });
    await writeFile(launcher.path, '#!/bin/sh\necho mine\n');
    await expect(launcher.prepare()).rejects.toThrow('Cannot replace');
    expect(await readFile(launcher.path, 'utf8')).toContain('echo mine');
    await rm(launcher.path);
    const target = join(root, 'foreign');
    await writeFile(target, 'preserve');
    await symlink(target, launcher.path);
    await expect(launcher.prepare()).rejects.toThrow('not a regular file');
    expect(await readFile(target, 'utf8')).toBe('preserve');
  });

  it('fails readiness when the runtime disappears or lacks delegation', async () => {
    const { config, launcher } = await fixture();
    await launcher.prepare();
    await writeFile(config.entry, "console.log('old command')");
    await expect(launcher.prepare()).rejects.toThrow('does not support delegation');
    await rm(config.entry);
    await expect(launcher.prepare()).rejects.toMatchObject({ code: 'ENOENT' });
  });
});

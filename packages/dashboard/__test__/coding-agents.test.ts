import { execFileSync } from 'node:child_process';
import { chmod, lstat, mkdir, mkdtemp, readFile, rm, stat, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { delegationPrompt } from '@mlx-node/agent/delegate';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { CodingAgentsService, MANAGED_BEGIN, MANAGED_END, managedBlock } from '../src/coding-agents.js';

const fsHooks = vi.hoisted(() => ({ beforeMkdir: undefined as (() => Promise<void>) | undefined }));
vi.mock('node:fs/promises', async (importOriginal) => {
  const fs = await importOriginal<typeof import('node:fs/promises')>();
  return {
    ...fs,
    // Lets a test edit the file between the service's read and its write.
    mkdir: async (...args: Parameters<typeof fs.mkdir>) => {
      const hook = fsHooks.beforeMkdir;
      fsHooks.beforeMkdir = undefined;
      await hook?.();
      return fs.mkdir(...args);
    },
  };
});

const cleanup: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const clean of cleanup.splice(0).reverse()) await clean();
});

async function setup(models = ['local-model'], prepare?: () => Promise<string>) {
  const home = await mkdtemp(join(tmpdir(), 'mlx-coding-agents-'));
  cleanup.push(() => rm(home, { recursive: true, force: true }));
  const command = join(home, '.mlx-node', 'bin', 'mlx');
  const env: NodeJS.ProcessEnv = {};
  const prepareCommand = vi.fn(prepare ?? (async () => command));
  const service = new CodingAgentsService({ home, env, listModels: async () => [...models], prepareCommand });
  cleanup.push(() => service.close());
  const claude = join(home, '.claude', 'CLAUDE.md');
  const write = async (text: string, path = claude) => {
    await mkdir(join(path, '..'), { recursive: true });
    await writeFile(path, text);
  };
  const row = async (id = 'claude') => (await service.refresh()).agents.find((agent) => agent.id === id)!;
  return { home, env, command, block: managedBlock(command), prepareCommand, service, claude, write, row };
}

describe('coding agent instruction paths', () => {
  it('uses CLAUDE_CONFIG_DIR and GROK_HOME, expanding ~', async () => {
    const { home, env, service } = await setup();
    env.CLAUDE_CONFIG_DIR = '~/claude config';
    env.GROK_HOME = join(home, 'grok');
    const [claude, codex, grok] = (await service.state()).agents;
    expect(claude.path).toBe(join(home, 'claude config', 'CLAUDE.md'));
    expect(codex.path).toBe(join(home, '.codex', 'AGENTS.md'));
    expect(grok.path).toBe(join(home, 'grok', 'AGENTS.md'));
  });

  it('uses a nonempty AGENTS.override.md under CODEX_HOME and ignores a blank one', async () => {
    const { home, env, write, row } = await setup();
    env.CODEX_HOME = join(home, 'codex');
    const override = join(home, 'codex', 'AGENTS.override.md');
    await write(' \n', override);
    expect((await row('codex')).path).toBe(join(home, 'codex', 'AGENTS.md'));
    await write('Override.', override);
    expect((await row('codex')).path).toBe(override);
  });

  it('reports an unreadable override as an error and never installs into the shadowed base file', async () => {
    const { home, write, service, row } = await setup();
    const override = join(home, '.codex', 'AGENTS.override.md');
    await write('Override.', override);
    await chmod(override, 0o000);
    cleanup.push(() => chmod(override, 0o600));
    expect(await row('codex')).toMatchObject({ path: override, status: 'error' });
    await expect(service.install('codex')).rejects.toMatchObject({ code: 'E_CONFLICT' });
    await expect(readFile(join(home, '.codex', 'AGENTS.md'))).rejects.toMatchObject({ code: 'ENOENT' });
  });

  it('installs once into a path shared by two agents', async () => {
    const { home, env, service, block } = await setup();
    env.CODEX_HOME = env.GROK_HOME = join(home, 'shared');
    const state = await service.install('codex');
    expect(state.agents.slice(1).map((agent) => agent.status)).toEqual(['installed', 'installed']);
    expect(await readFile(join(home, 'shared', 'AGENTS.md'), 'utf8')).toBe(`${block}\n`);
  });

  it('writes through a symlinked instruction file and keeps the link', async () => {
    const { home, claude, service, block } = await setup();
    const target = join(home, 'dotfiles', 'CLAUDE.md');
    await mkdir(join(home, 'dotfiles'));
    await mkdir(join(home, '.claude'));
    await writeFile(target, 'Mine.\n');
    await symlink(target, claude);
    await service.install('claude');
    expect((await lstat(claude)).isSymbolicLink()).toBe(true);
    expect(await readFile(target, 'utf8')).toBe(`Mine.\n\n${block}\n`);
  });

  it.each(['fifo', 'directory'])('reports a %s at the instruction path as an error without hanging', async (kind) => {
    const { claude, service, row } = await setup();
    await mkdir(join(claude, '..'), { recursive: true });
    if (kind === 'fifo') execFileSync('mkfifo', [claude]);
    else await mkdir(claude);
    expect(await row()).toMatchObject({ status: 'error', detail: expect.stringContaining('regular file') });
    await expect(service.install('claude')).rejects.toThrow('regular file');
  });
});

describe('coding agent availability', () => {
  it('needs an installed model before setup', async () => {
    const { service, claude } = await setup([]);
    const state = await service.state();
    expect(state).toMatchObject({ available: false, model: null, command: null });
    expect(state.unavailableReason).toContain('Install a local model first');
    expect(state.agents.every((agent) => agent.status === 'unchecked')).toBe(true);
    await expect(service.install('claude')).rejects.toThrow('Install a local model first');
    await expect(readFile(claude)).rejects.toMatchObject({ code: 'ENOENT' });
  });

  it('uses the persisted default model and reports it missing once removed', async () => {
    const { home, write, service } = await setup(['alpha', 'chosen']);
    await write('[models]\ndefault = "chosen"\n', join(home, '.mlx-agent', 'config.toml'));
    expect((await service.state()).model).toBe('chosen');
    await write('[models]\ndefault = "gone"\n', join(home, '.mlx-agent', 'config.toml'));
    expect(await service.state()).toMatchObject({ available: false, model: null });
    expect((await service.state()).unavailableReason).toContain('no longer installed');
  });

  it.each([
    ['fails', async () => Promise.reject(new Error('The app command is missing.')), 'The app command is missing.'],
    ['is not absolute', async () => 'bin/mlx', 'absolute path'],
  ])('blocks setup when the app command %s', async (_, prepare, reason) => {
    const { service, claude, write } = await setup(['local-model'], prepare);
    await write('Mine.');
    const state = await service.state();
    expect(state).toMatchObject({ available: false, model: 'local-model', command: null });
    expect(state.unavailableReason).toContain(reason);
    expect(state.agents[0].status).toBe('unchecked');
    await expect(service.install('claude')).rejects.toThrow(reason);
    expect(await readFile(claude, 'utf8')).toBe('Mine.');
  });
});

describe('managed instruction block', () => {
  it('reads the file again on every check', async () => {
    const { service, write, row } = await setup();
    await service.install('claude');
    expect((await row()).status).toBe('installed');
    await write('Only my preferences.');
    expect((await row()).status).toBe('not-installed');
  });

  it('reports a file without markers as not installed', async () => {
    const { write, row } = await setup();
    await write('Use yarn.\n');
    expect((await row()).status).toBe('not-installed');
  });

  it('does not recognize an unmarked delegation prompt', async () => {
    const { write, row, command } = await setup();
    await write(`${delegationPrompt(command)}\n`);
    expect((await row()).status).toBe('not-installed');
  });

  it('reports the exact block as installed', async () => {
    const { write, row, block } = await setup();
    await write(`Before.\n${block}\nAfter.\n`);
    expect((await row()).status).toBe('installed');
  });

  it.each([
    ['an old command path', managedBlock('/old/app/mlx')],
    ['edited text', `${MANAGED_BEGIN}\nmy own words\n${MANAGED_END}`],
  ])('updates a block with %s and keeps every byte outside the markers', async (_, stale) => {
    const { claude, write, row, service, block } = await setup();
    const before = 'Before.\r\n  ';
    const after = '  \nAfter, no trailing newline';
    await write(`${before}${stale}${after}`);
    const inode = (await stat(claude)).ino;
    expect((await row()).status).toBe('needs-update');
    expect((await service.install('claude')).agents[0].status).toBe('installed');
    expect(await readFile(claude, 'utf8')).toBe(`${before}${block}${after}`);
    expect((await stat(claude)).ino).toBe(inode);
  });

  it.each([
    ['begin without end', `${MANAGED_BEGIN}\nx\n`],
    ['end without begin', `x\n${MANAGED_END}\n`],
    ['two begin markers', `${MANAGED_BEGIN}\n${MANAGED_BEGIN}\nx\n${MANAGED_END}\n`],
    ['end before begin', `${MANAGED_END}\nx\n${MANAGED_BEGIN}\n`],
    ['stray end before a block', `${MANAGED_END}\n${MANAGED_BEGIN}\nx\n${MANAGED_END}\n`],
  ])('reports %s as an error and refuses to install', async (_, text) => {
    const { claude, write, row, service } = await setup();
    await write(`Mine.\n${text}`);
    expect((await row()).status).toBe('error');
    await expect(service.install('claude')).rejects.toMatchObject({
      code: 'E_CONFLICT',
      message: expect.stringContaining('markers'),
    });
    expect(await readFile(claude, 'utf8')).toBe(`Mine.\n${text}`);
  });

  it('fails the request when the file is not writable', async () => {
    const { claude, write, service } = await setup();
    await write('Mine.\n');
    await chmod(claude, 0o400);
    await expect(service.install('claude')).rejects.toMatchObject({ code: 'E_CONFLICT' });
    expect(await readFile(claude, 'utf8')).toBe('Mine.\n');
  });

  it('refuses to overwrite a file that changed during installation', async () => {
    const { claude, write, service } = await setup();
    await write('Mine.\n');
    fsHooks.beforeMkdir = () => writeFile(claude, 'Edited elsewhere.\n');
    await expect(service.install('claude')).rejects.toThrow('changed during installation');
    expect(await readFile(claude, 'utf8')).toBe('Edited elsewhere.\n');
  });

  it('creates a missing file and its parent directory', async () => {
    const { home, env, service, block } = await setup();
    env.CLAUDE_CONFIG_DIR = join(home, 'a', 'b');
    const path = join(home, 'a', 'b', 'CLAUDE.md');
    expect((await service.install('claude')).agents[0]).toMatchObject({ path, status: 'installed' });
    expect(await readFile(path, 'utf8')).toBe(`${block}\n`);
    expect((await stat(path)).mode & 0o777).toBe(0o600);
  });

  it.each(['Use yarn.', 'Use yarn.\n'])('appends after one blank line to %j', async (existing) => {
    const { claude, write, service, block } = await setup();
    await write(existing);
    await service.install('claude');
    expect(await readFile(claude, 'utf8')).toBe(`Use yarn.\n\n${block}\n`);
  });

  it('does not write an already installed file', async () => {
    const { claude, service } = await setup();
    await service.install('claude');
    const before = await readFile(claude, 'utf8');
    // A write to a read-only file would fail and surface as an error.
    await chmod(claude, 0o400);
    expect((await service.install('claude')).agents[0].status).toBe('installed');
    expect(await readFile(claude, 'utf8')).toBe(before);
  });
});

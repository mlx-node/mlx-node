import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, readFile, writeFile, rm, symlink, stat } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

import { DELEGATION_PROMPT, delegationCommand, delegationPrompt } from '@mlx-node/agent/delegate';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { COMMAND_SELECTION_SYSTEM, DETECTION_INPUT_PREFIX, DETECTION_SYSTEM } from '../src/coding-agent-detection.js';
import { CodingAgentsService } from '../src/coding-agents.js';

const cleanup: (() => Promise<void>)[] = [];
afterEach(async () => {
  for (const clean of cleanup.splice(0).reverse()) await clean();
});

async function setup(models = ['local-model']) {
  const home = await mkdtemp(join(tmpdir(), 'mlx-coding-agents-'));
  cleanup.push(() => rm(home, { recursive: true, force: true }));
  const command = join(home, '.mlx-node', 'bin', 'mlx');
  const prompt = delegationPrompt(command);
  const prepareCommand = vi.fn(async () => command);
  const complete = vi.fn(async (_connection, _system, messages) => {
    const content = messages[0].content as string;
    if (_system === COMMAND_SELECTION_SYSTEM) {
      const { source } = JSON.parse(content);
      const known = ['mlx', delegationCommand(command)];
      const executable = known.find((item) => source.includes(`${item} delegate github`));
      if (!executable) throw new Error('Test fixture requires an explicit selection for this command.');
      const prefix = `${executable} delegate github`;
      return JSON.stringify({
        text: source.includes(`${prefix} --caller-approved`) ? `${prefix} --caller-approved` : prefix,
        occurrence: 1,
      });
    }
    const currentCommand = JSON.parse(content.slice('Current app executable: '.length, content.indexOf('\n')));
    const currentPrompt = delegationPrompt(currentCommand);
    const lines = content
      .slice(content.indexOf(DETECTION_INPUT_PREFIX) + DETECTION_INPUT_PREFIX.length)
      .split('\n')
      .map((line) => line.slice(line.indexOf(' | ') + 3));
    const index = lines.findIndex((line) =>
      [currentPrompt, prompt, DELEGATION_PROMPT].some((known) => line.includes(known)),
    );
    const status = index < 0 ? 'not-installed' : lines[index].includes(currentPrompt) ? 'installed' : 'needs-update';
    return JSON.stringify({ status, startLine: index + 1, endLine: index + 1 });
  });
  const connect = vi.fn(async () => ({ url: 'http://127.0.0.1:8080', model: 'host-default', token: 'secret' }));
  const listModels = vi.fn(async () => [...models]);
  const env: NodeJS.ProcessEnv = {};
  const restart = () => {
    const service = new CodingAgentsService({ home, env, listModels, connect, complete, prepareCommand });
    cleanup.push(() => service.close());
    return service;
  };
  return {
    home,
    env,
    command,
    prompt,
    prepareCommand,
    complete,
    connect,
    service: restart(),
    models,
    listModels,
    restart,
  };
}

async function settled(service: CodingAgentsService, id = 'claude') {
  await vi.waitFor(async () => {
    const state = await service.state();
    expect(['waiting', 'checking', 'installing']).not.toContain(state.agents.find((row) => row.id === id)?.status);
  });
  return (await service.state()).agents.find((row) => row.id === id)!;
}

describe('local model coding-agent setup', () => {
  it('rechecks verdicts saved before the medium-thinking generation policy', async () => {
    const { home, service, restart, complete, command, prompt } = await setup();
    const path = join(home, '.claude', 'CLAUDE.md');
    await mkdir(join(home, '.claude'));
    await writeFile(path, prompt);
    await service.start('detect', 'claude');
    expect((await settled(service)).status).toBe('installed');
    await service.close();
    const hash = (text: string) => createHash('sha256').update(text).digest('hex');
    const oldKey = hash(
      JSON.stringify([
        DETECTION_SYSTEM,
        DETECTION_INPUT_PREFIX,
        DELEGATION_PROMPT,
        'local-model',
        path,
        hash(prompt),
        command,
      ]),
    );
    const cachePath = join(home, '.mlx-node', 'coding-agents.json');
    const cache = JSON.parse(await readFile(cachePath, 'utf8'));
    cache.entries[0].key = oldKey;
    cache.entries[0].installed = false;
    await writeFile(cachePath, JSON.stringify(cache));
    complete.mockClear();
    const restored = restart();
    await restored.start('detect', 'claude');
    expect((await settled(restored)).status).toBe('installed');
    expect(complete).toHaveBeenCalledTimes(1);
  });

  it('blocks setup without a working app command and invalidates an installed verdict', async () => {
    const { service, home, prepareCommand, complete } = await setup();
    await service.start('install', 'claude');
    expect((await settled(service)).status).toBe('installed');
    const before = await readFile(join(home, '.claude', 'CLAUDE.md'), 'utf8');
    prepareCommand.mockRejectedValue(new Error('The app command is missing.'));
    expect(await service.refresh()).toMatchObject({
      available: false,
      command: null,
      unavailableReason: 'The app command is missing.',
    });
    expect((await service.state()).agents[0].status).toBe('unchecked');
    const calls = complete.mock.calls.length;
    await expect(service.start('install', 'claude')).rejects.toThrow('command is missing');
    expect(complete).toHaveBeenCalledTimes(calls);
    expect(await readFile(join(home, '.claude', 'CLAUDE.md'), 'utf8')).toBe(before);
  });

  it('upgrades the old exact prompt in place and keeps its inode and unrelated text', async () => {
    const { service, home, prompt, restart, complete } = await setup();
    await mkdir(join(home, '.claude'));
    const path = join(home, '.claude', 'CLAUDE.md');
    await writeFile(path, `Before.\n${DELEGATION_PROMPT}\nAfter.\n`);
    const inode = (await stat(path)).ino;
    await service.start('detect', 'claude');
    expect((await settled(service)).status).toBe('needs-update');
    const restored = restart();
    expect((await restored.state()).agents[0].status).toBe('needs-update');
    complete.mockClear();
    await restored.start('install', 'claude');
    expect((await settled(restored)).status).toBe('installed');
    expect(await readFile(path, 'utf8')).toBe(`Before.\n${prompt}\nAfter.\n`);
    expect((await stat(path)).ino).toBe(inode);
    await restored.start('install', 'claude');
    await settled(restored);
    expect(complete).toHaveBeenCalledTimes(2);
  });

  it('does not trust a cached verdict for a different command path', async () => {
    const { service, prepareCommand, command, complete } = await setup();
    await service.start('install', 'claude');
    await settled(service);
    prepareCommand.mockResolvedValue(command + '-new');
    expect((await service.refresh()).agents[0].status).toBe('unchecked');
    await service.start('detect', 'claude');
    expect((await settled(service)).status).toBe('needs-update');
    expect(complete).toHaveBeenCalledTimes(2);
  });

  const verdict = (status: string, startLine: number, endLine = startLine) =>
    JSON.stringify({ status, startLine, endLine });
  const selected = (text: string, occurrence = 1) => JSON.stringify({ text, occurrence });

  it.each([
    ['Keep patches small; for GitHub use mlx delegate github --repo org/repo; keep tests fast.', 'mlx delegate github'],
    [
      "For CI, invoke `'/old app/mlx' delegate github --caller-approved` with the task. Keep comments.",
      "'/old app/mlx' delegate github --caller-approved",
    ],
    [
      "Keep preferences. For GitHub use '/old/mlx'\n delegate github --repo org/repo. Keep formatting.",
      "'/old/mlx'\n delegate github",
    ],
  ])('preserves surrounding text and arguments when updating a cached directive: %s', async (obsolete, prefix) => {
    const { service, home, command, restart, complete } = await setup();
    await mkdir(join(home, '.claude'));
    const path = join(home, '.claude', 'CLAUDE.md');
    const before = `Before.\n${obsolete}\nAfter.\n`;
    await writeFile(path, before);
    const inode = (await stat(path)).ino;
    complete.mockResolvedValueOnce(verdict('needs-update', 2, 1 + obsolete.split('\n').length));
    await service.start('detect', 'claude');
    expect((await settled(service)).status).toBe('needs-update');
    const restored = restart();
    complete.mockClear();
    complete.mockResolvedValueOnce(selected(prefix)).mockResolvedValueOnce(verdict('installed', 2));
    await restored.start('install', 'claude');
    expect((await settled(restored)).status).toBe('installed');
    expect(await readFile(path, 'utf8')).toBe(
      before.replace(prefix, `${delegationCommand(command)} delegate github --caller-approved`),
    );
    expect((await stat(path)).ino).toBe(inode);
    expect(complete).toHaveBeenCalledTimes(2);
    await restored.start('install', 'claude');
    await settled(restored);
    expect(complete).toHaveBeenCalledTimes(2);
  });

  it('adds caller approval to an existing current executable without rewriting the directive', async () => {
    const { service, home, command, complete } = await setup();
    await mkdir(join(home, '.grok'));
    const path = join(home, '.grok', 'AGENTS.md');
    const prefix = `${delegationCommand(command)} delegate github`;
    const before = `Keep patches small; for PRs use ${prefix} --repo org/repo.\n`;
    await writeFile(path, before);
    complete
      .mockResolvedValueOnce(verdict('needs-update', 1))
      .mockResolvedValueOnce(selected(prefix))
      .mockResolvedValueOnce(verdict('installed', 1));
    await service.start('install', 'grok');
    expect((await settled(service, 'grok')).status).toBe('installed');
    expect(await readFile(path, 'utf8')).toBe(before.replace(prefix, `${prefix} --caller-approved`));
  });

  it('rechecks an older cached update without source lines before replacing anything', async () => {
    const { service, home, prompt, restart, complete } = await setup();
    await mkdir(join(home, '.claude'));
    const path = join(home, '.claude', 'CLAUDE.md');
    await writeFile(path, `Keep my preferences.\n${DELEGATION_PROMPT}\n`);
    await service.start('detect', 'claude');
    await settled(service);
    await service.close();
    const cachePath = join(home, '.mlx-node', 'coding-agents.json');
    const cache = JSON.parse(await readFile(cachePath, 'utf8'));
    for (const entry of cache.entries) delete entry.source;
    await writeFile(cachePath, JSON.stringify(cache));
    complete.mockClear();
    const restored = restart();
    await restored.start('install', 'claude');
    expect((await settled(restored)).status).toBe('installed');
    expect(complete).toHaveBeenCalledTimes(3);
    expect(await readFile(path, 'utf8')).toBe(`Keep my preferences.\n${prompt}\n`);
  });

  it.each([false, true])('updates every stale command in any selection order (cached: %s)', async (cached) => {
    const { service, home, prompt, command, restart, complete } = await setup();
    await mkdir(join(home, '.claude'));
    const path = join(home, '.claude', 'CLAUDE.md');
    const first = "'/old/mlx'\n delegate github";
    const second = "'/other/mlx' delegate github";
    const before = `Before; for PRs use ${first}. Keep formatting.\n${prompt}\nFor CI use ${second}; keep tests fast.\nAfter.\n`;
    await writeFile(path, before);
    complete
      .mockResolvedValueOnce(verdict('needs-update', 4))
      .mockResolvedValueOnce(selected(second))
      .mockImplementationOnce(async () => {
        expect(await readFile(path, 'utf8')).toBe(before);
        return verdict('needs-update', 1, 2);
      })
      .mockResolvedValueOnce(selected(first))
      .mockResolvedValueOnce(verdict('installed', 1));
    if (cached) {
      await service.start('detect', 'claude');
      expect((await settled(service)).status).toBe('needs-update');
      await service.close();
    }
    const active = cached ? restart() : service;
    await active.start('install', 'claude');
    expect((await settled(active)).status).toBe('installed');
    const prefix = `${delegationCommand(command)} delegate github --caller-approved`;
    expect(await readFile(path, 'utf8')).toBe(before.replace(first, prefix).replace(second, prefix));
    expect(complete).toHaveBeenCalledTimes(5);
    complete.mockClear();
    const restored = restart();
    await restored.start('detect', 'claude');
    await settled(restored);
    expect(complete).not.toHaveBeenCalled();
  });

  it('stops without writing when the model selects generated replacement text as obsolete', async () => {
    const { service, home, command, complete } = await setup();
    await mkdir(join(home, '.claude'));
    const path = join(home, '.claude', 'CLAUDE.md');
    const before = "For PRs use '/old/mlx' delegate github.\n";
    await writeFile(path, before);
    complete
      .mockResolvedValueOnce(verdict('needs-update', 1))
      .mockResolvedValueOnce(selected("'/old/mlx' delegate github"))
      .mockResolvedValueOnce(verdict('needs-update', 1))
      // Even selecting just part of an earlier replacement must stop.
      .mockResolvedValueOnce(selected(delegationCommand(command)));
    await service.start('install', 'claude');
    expect((await settled(service)).detail).toContain('No changes were made');
    expect(await readFile(path, 'utf8')).toBe(before);
    expect(complete).toHaveBeenCalledTimes(4);
  });

  it('keeps the original file when the model invents selection text', async () => {
    const { service, home, complete } = await setup();
    await mkdir(join(home, '.claude'));
    const path = join(home, '.claude', 'CLAUDE.md');
    const before = "Keep patches small; use '/old/mlx' delegate github for PRs.\n";
    await writeFile(path, before);
    complete.mockResolvedValueOnce(verdict('needs-update', 1)).mockResolvedValueOnce(selected('invented'));
    await service.start('install', 'claude');
    expect((await settled(service)).detail).toContain('No changes were made');
    expect(await readFile(path, 'utf8')).toBe(before);
  });

  it('rejects unrelated source text even if later model answers would approve both replacements', async () => {
    const { service, home, complete } = await setup();
    await mkdir(join(home, '.claude'));
    const path = join(home, '.claude', 'CLAUDE.md');
    const before = "Keep patches small; use '/old/mlx' delegate github for PRs.\n";
    await writeFile(path, before);
    const inode = (await stat(path)).ino;
    complete
      .mockResolvedValueOnce(verdict('needs-update', 1))
      .mockResolvedValueOnce(selected('Keep patches small'))
      .mockResolvedValueOnce(verdict('needs-update', 1))
      .mockResolvedValueOnce(selected("'/old/mlx' delegate github"))
      .mockResolvedValueOnce(verdict('installed', 1));
    await service.start('install', 'claude');
    expect((await settled(service)).detail).toContain('No changes were made');
    expect(await readFile(path, 'utf8')).toBe(before);
    expect((await stat(path)).ino).toBe(inode);
    expect(complete).toHaveBeenCalledTimes(2);
  });

  it('keeps the original file if verification fails after multiple obsolete commands', async () => {
    const { service, home, complete } = await setup();
    await mkdir(join(home, '.claude'));
    const path = join(home, '.claude', 'CLAUDE.md');
    const before = "For PRs use '/old/mlx' delegate github.\nFor CI use '/other/mlx' delegate github.\n";
    await writeFile(path, before);
    complete
      .mockResolvedValueOnce(verdict('needs-update', 1))
      .mockResolvedValueOnce(selected("'/old/mlx' delegate github"))
      .mockResolvedValueOnce(verdict('needs-update', 2))
      .mockResolvedValueOnce(selected("'/other/mlx' delegate github"))
      .mockResolvedValueOnce(verdict('not-installed', 0));
    await service.start('install', 'claude');
    expect((await settled(service)).detail).toContain('No changes were made');
    expect(await readFile(path, 'utf8')).toBe(before);
    expect(complete).toHaveBeenCalledTimes(5);
  });

  it('preserves a current prompt example while updating the active directive', async () => {
    const { service, home, prompt, command, complete } = await setup();
    await mkdir(join(home, '.claude'));
    const path = join(home, '.claude', 'CLAUDE.md');
    const example = `\x60\x60\x60markdown\n${prompt}\n\x60\x60\x60\n`;
    await writeFile(path, `${example}For PRs use '/old/mlx' delegate github.\n`);
    complete
      .mockResolvedValueOnce(verdict('needs-update', 4))
      .mockResolvedValueOnce(selected("'/old/mlx' delegate github"))
      .mockResolvedValueOnce(verdict('installed', 4));
    await service.start('install', 'claude');
    expect((await settled(service)).status).toBe('installed');
    expect(await readFile(path, 'utf8')).toBe(
      `${example}For PRs use ${delegationCommand(command)} delegate github --caller-approved.\n`,
    );
  });

  it('blocks both detection and installation without a model, before inference or file writes', async () => {
    const { service, complete, connect, home } = await setup([]);
    expect((await service.state()).available).toBe(false);
    await expect(service.start('detect')).rejects.toThrow('Install a local model first');
    await expect(service.start('install', 'claude')).rejects.toThrow('Install a local model first');
    expect(complete).not.toHaveBeenCalled();
    expect(connect).not.toHaveBeenCalled();
    await expect(readFile(join(home, '.claude', 'CLAUDE.md'))).rejects.toMatchObject({ code: 'ENOENT' });
  });

  it('uses the persisted default rather than the host default or first model', async () => {
    const { service, home, complete } = await setup(['alpha', 'chosen']);
    await mkdir(join(home, '.mlx-node', 'agent'), { recursive: true });
    await writeFile(
      join(home, '.mlx-node', 'agent', 'settings.json'),
      JSON.stringify({ defaultProvider: 'mlx', defaultModel: 'chosen' }),
    );
    await mkdir(join(home, '.claude'));
    await writeFile(join(home, '.claude', 'CLAUDE.md'), 'Use yarn.');
    await service.start('detect', 'claude');
    expect((await settled(service)).status).toBe('not-installed');
    expect(complete.mock.calls[0][0].model).toBe('chosen');
    expect(complete.mock.calls[0][2][0].content).toContain(
      `Current app executable: ${JSON.stringify(join(home, '.mlx-node', 'bin', 'mlx'))}`,
    );
  });

  it.each(['path', 'file-url', 'tilde'])('uses the persisted default from a %s agent directory', async (kind) => {
    const { service, home, env, complete } = await setup(['alpha', 'chosen']);
    const dir = join(home, 'custom agent #配置');
    await mkdir(dir);
    await writeFile(join(dir, 'settings.json'), JSON.stringify({ defaultProvider: 'mlx', defaultModel: 'chosen' }));
    env.PI_CODING_AGENT_DIR =
      kind === 'file-url' ? pathToFileURL(dir).href : kind === 'tilde' ? '~/custom agent #配置' : dir;
    await mkdir(join(home, '.claude'));
    await writeFile(join(home, '.claude', 'CLAUDE.md'), 'Use yarn.');
    await service.start('detect', 'claude');
    expect((await settled(service)).status).toBe('not-installed');
    expect(complete.mock.calls[0][0].model).toBe('chosen');
    await service.start('install', 'claude');
    expect((await settled(service)).status).toBe('installed');
    expect(complete.mock.calls.every(([connection]) => connection.model === 'chosen')).toBe(true);
  });

  it('blocks setup for an invalid agent directory URL instead of choosing another model', async () => {
    const { service, env, complete, connect } = await setup(['alpha', 'chosen']);
    env.PI_CODING_AGENT_DIR = 'file:///%ZZ';
    await expect(service.start('detect', 'claude')).rejects.toThrow('Could not read the default local model');
    await expect(service.start('install', 'claude')).rejects.toThrow('Could not read the default local model');
    expect(complete).not.toHaveBeenCalled();
    expect(connect).not.toHaveBeenCalled();
  });

  it('recognizes a manually worded installation through the model without markers', async () => {
    const { service, home, complete, command } = await setup();
    const custom = `For GitHub work, call '${command}' delegate github with the repository and task.`;
    await mkdir(join(home, '.claude'));
    await writeFile(join(home, '.claude', 'CLAUDE.md'), custom);
    complete.mockResolvedValueOnce(JSON.stringify({ status: 'installed', startLine: 1, endLine: 1 }));
    await service.start('detect', 'claude');
    expect((await settled(service)).status).toBe('installed');
    expect(complete).toHaveBeenCalledTimes(1);
  });

  it('appends only the short prompt, preserves existing text, and does not duplicate installation', async () => {
    const { service, home, complete, prompt } = await setup();
    await mkdir(join(home, '.claude'));
    const path = join(home, '.claude', 'CLAUDE.md');
    await writeFile(path, '# Preferences\nUse yarn.');
    await service.start('install', 'claude');
    expect((await settled(service)).status).toBe('installed');
    expect(await readFile(path, 'utf8')).toBe(`# Preferences\nUse yarn.\n\n${prompt}\n`);
    expect(complete).toHaveBeenCalledTimes(2);
    await service.start('install', 'claude');
    await settled(service);
    expect((await readFile(path, 'utf8')).split(prompt)).toHaveLength(2);
  });

  it('selects the active Codex override and follows an existing file symlink', async () => {
    const { service, home } = await setup();
    await mkdir(join(home, '.codex'));
    const target = join(home, 'shared.md');
    await writeFile(target, 'Existing instructions.');
    await symlink(target, join(home, '.codex', 'AGENTS.override.md'));
    await service.start('install', 'codex');
    expect((await settled(service, 'codex')).path).toContain('AGENTS.override.md');
    expect(await readFile(target, 'utf8')).toContain(delegationPrompt(join(home, '.mlx-node', 'bin', 'mlx')));
    await expect(readFile(join(home, '.codex', 'AGENTS.md'))).rejects.toMatchObject({ code: 'ENOENT' });
  });

  it('does not turn invalid model answers or invented evidence into Install buttons', async () => {
    const { service, complete, home } = await setup();
    await mkdir(join(home, '.claude'));
    await writeFile(join(home, '.claude', 'CLAUDE.md'), 'Use yarn.');
    complete.mockResolvedValueOnce(JSON.stringify({ status: 'installed', startLine: 99, endLine: 99 }));
    await service.start('detect', 'claude');
    expect((await settled(service)).status).toBe('error');
  });

  it('rejects a file edit that races model detection', async () => {
    const { service, home, complete } = await setup();
    await mkdir(join(home, '.claude'));
    const path = join(home, '.claude', 'CLAUDE.md');
    await writeFile(path, 'Before.');
    complete.mockImplementationOnce(async () => {
      await writeFile(path, 'User edit.');
      return JSON.stringify({ status: 'not-installed', startLine: 0, endLine: 0 });
    });
    await service.start('install', 'claude');
    expect((await settled(service)).status).toBe('error');
    expect(await readFile(path, 'utf8')).toBe('User edit.');
  });

  it('invalidates a previous detection when the prompt is removed', async () => {
    const { service, home } = await setup();
    await service.start('install', 'claude');
    await settled(service);
    await writeFile(join(home, '.claude', 'CLAUDE.md'), 'Only my other preferences.');
    expect((await service.state()).agents[0].status).toBe('installed');
    expect((await service.refresh()).agents[0].status).toBe('unchecked');
  });

  it('checks each native global file independently', async () => {
    const { service } = await setup();
    await service.start('install', 'claude');
    await settled(service);
    await service.start('detect');
    await settled(service, 'grok');
    expect((await service.state()).agents.map((row) => row.status)).toEqual([
      'installed',
      'not-installed',
      'not-installed',
    ]);
    await service.start('install', 'grok');
    expect((await settled(service, 'grok')).path).toContain('.grok/AGENTS.md');
  });

  it('refuses oversized instructions without asking the model to classify a partial file', async () => {
    const { service, home, complete } = await setup();
    await mkdir(join(home, '.claude'));
    await writeFile(join(home, '.claude', 'CLAUDE.md'), 'x'.repeat(49 * 1024));
    await service.start('install', 'claude');
    expect((await settled(service)).detail).toContain('too large');
    expect(complete).not.toHaveBeenCalled();
  });

  it('does not install if the default model is removed during verification', async () => {
    const { service, home, complete, models } = await setup();
    complete.mockImplementationOnce(async () => {
      models.splice(0);
      return JSON.stringify({ status: 'not-installed', startLine: 0, endLine: 0 });
    });
    await service.start('install', 'claude');
    expect((await settled(service)).status).toBe('error');
    await expect(readFile(join(home, '.claude', 'CLAUDE.md'))).rejects.toMatchObject({ code: 'ENOENT' });
  });

  it('skips inference and connection startup for missing and empty files', async () => {
    const { service, complete, connect, home } = await setup();
    await mkdir(join(home, '.claude'));
    await writeFile(join(home, '.claude', 'CLAUDE.md'), ' \n\t');
    await service.start('detect');
    expect((await service.state()).agents.every((row) => row.status === 'not-installed')).toBe(true);
    expect(complete).not.toHaveBeenCalled();
    expect(connect).not.toHaveBeenCalled();
  });

  it('restores semantic results after restart without storing instruction text', async () => {
    const { service, home, complete, connect, restart } = await setup();
    await service.start('install', 'claude');
    await settled(service);
    const cachePath = join(home, '.mlx-node', 'coding-agents.json');
    const cache = await readFile(cachePath, 'utf8');
    expect(cache).not.toContain(DELEGATION_PROMPT);
    expect(cache).not.toContain(home);
    expect((await stat(cachePath)).mode & 0o777).toBe(0o600);
    await service.close();
    complete.mockClear();
    connect.mockClear();
    const next = restart();
    expect((await next.state()).agents[0].status).toBe('installed');
    await next.start('detect');
    expect(complete).not.toHaveBeenCalled();
    expect(connect).not.toHaveBeenCalled();
  });

  it('only invokes inference for a changed file and invalidates the cache when the model changes', async () => {
    const { service, home, complete, models } = await setup();
    await service.start('install', 'claude');
    await settled(service);
    complete.mockClear();
    await service.start('detect');
    expect(complete).not.toHaveBeenCalled();
    await writeFile(join(home, '.claude', 'CLAUDE.md'), 'Changed preferences.');
    await service.start('detect');
    expect((await settled(service)).status).toBe('not-installed');
    expect(complete).toHaveBeenCalledTimes(1);
    models.splice(0, 1, 'new-default');
    expect((await service.refresh()).agents[0].status).toBe('unchecked');
    await service.start('detect', 'claude');
    await settled(service);
    expect(complete).toHaveBeenCalledTimes(2);
    expect(complete.mock.calls[1][0].model).toBe('new-default');
  });

  it('reuses positive and negative verdicts for identical content across refreshes and restarts', async () => {
    const { service, home, complete, connect, restart } = await setup();
    await service.start('install', 'claude');
    const installed = await settled(service);
    await mkdir(join(home, '.codex'));
    const codexPath = join(home, '.codex', 'AGENTS.md');
    await writeFile(codexPath, 'Use oxnode for TypeScript.');
    await service.start('detect', 'codex');
    const absent = await settled(service, 'codex');
    expect(absent.status).toBe('not-installed');
    const claudePath = join(home, '.claude', 'CLAUDE.md');
    for (const path of [claudePath, codexPath]) await writeFile(path, await readFile(path));
    complete.mockClear();
    connect.mockClear();
    for (const next of [service, restart()]) {
      await next.refresh();
      await next.start('detect');
      const rows = (await next.state()).agents;
      expect(rows[0]).toMatchObject({ status: 'installed', checkedAt: installed.checkedAt });
      expect(rows[1]).toMatchObject({ status: 'not-installed', checkedAt: absent.checkedAt });
    }
    expect(complete).not.toHaveBeenCalled();
    expect(connect).not.toHaveBeenCalled();
  });

  it('polls only snapshots and labels queued checks as waiting', async () => {
    const { service, home, complete, listModels } = await setup();
    for (const dir of ['.claude', '.codex']) await mkdir(join(home, dir));
    await writeFile(join(home, '.claude', 'CLAUDE.md'), 'Use yarn.');
    await writeFile(join(home, '.codex', 'AGENTS.md'), 'Use oxnode.');
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    complete.mockImplementationOnce(async () => {
      await gate;
      return JSON.stringify({ status: 'not-installed', startLine: 0, endLine: 0 });
    });
    await service.start('detect');
    try {
      await vi.waitFor(() => expect(complete).toHaveBeenCalledTimes(1));
      const scans = listModels.mock.calls.length;
      for (let i = 0; i < 20; i++) {
        expect((await service.state()).agents.map((row) => row.status)).toEqual([
          'checking',
          'waiting',
          'not-installed',
        ]);
      }
      expect(listModels).toHaveBeenCalledTimes(scans);
    } finally {
      release();
    }
    await settled(service, 'codex');
    expect(complete).toHaveBeenCalledTimes(2);
  });

  it('treats a corrupt cache as unchecked and allows a forced recheck', async () => {
    const { service, home, complete, restart } = await setup();
    await service.start('install', 'claude');
    await settled(service);
    complete.mockClear();
    await service.start('detect', 'claude', true);
    await settled(service);
    expect(complete).toHaveBeenCalledTimes(1);
    await service.close();
    await writeFile(join(home, '.mlx-node', 'coding-agents.json'), '{broken');
    expect((await restart().state()).agents[0].status).toBe('unchecked');
  });
});

import { describe, expect, it, vi } from 'vite-plus/test';

import type { AgentRunDeps } from '../src/commands/agent/index.js';
import {
  DELEGATE_DEFAULT_ARGS,
  DELEGATE_SYSTEM_PROMPT,
  delegateAgentArgs,
  parseDelegateArgs,
  run,
} from '../src/commands/delegate.js';

describe('delegate agent arguments', () => {
  it('injects the worker profile and folds positionals into -p ahead of the parsed args', () => {
    const args = ['--reasoning-effort', 'high', '--resume', 'session-id', '@context.md', 'Explain this'];
    expect(delegateAgentArgs(args)).toEqual([
      ...DELEGATE_DEFAULT_ARGS,
      '-p',
      '@context.md Explain this',
      '--reasoning-effort',
      'high',
      '--resume',
      'session-id',
    ]);
    expect(DELEGATE_DEFAULT_ARGS).toEqual([
      '--system-prompt-override',
      DELEGATE_SYSTEM_PROMPT,
      '--tools',
      'read_file,run_terminal_command',
      '--permission-mode',
      'auto',
    ]);
  });

  it('keeps agent metadata and package commands usable without prompt mode', () => {
    for (const args of [
      ['--help'],
      ['--version'],
      ['export', 'session.jsonl'],
      ['config'],
      ['update'],
    ]) {
      expect(delegateAgentArgs(args)).toEqual(args);
    }
  });

  it.each([[], ['github', '--repo', 'owner/repo']])(
    'captures approval separately from model arguments: %j',
    (...prefix) => {
      const { args, callerApproved } = parseDelegateArgs([...prefix, '--caller-approved', 'Check PR #148']);
      expect(callerApproved).toBe(true);
      expect(args).not.toContain('--caller-approved');
      expect(args[args.indexOf('-p') + 1]).toBe('Check PR #148');
    },
  );

  it.each([
    ['--system-prompt-override', '--caller-approved'],
    ['github', '--append-system-prompt', '--caller-approved'],
    ['github', '--', '--caller-approved'],
    ['--models-dir', '--caller-approved'],
    ['github', '--repo=owner/repo', 'The prompt contains --caller-approved'],
  ])('does not authorize from an option value or prompt: %j', (...args) => {
    expect(parseDelegateArgs(args).callerApproved).toBe(false);
  });

  it('rejects ambiguous approval values', () => {
    expect(() => parseDelegateArgs(['github', '--caller-approved=false', 'Task'])).toThrow('takes no value');
  });

  it.each([
    [false, 'auto'],
    [true, 'bypassPermissions'],
  ])('forwards approval as the worker permission mode (%s → %s)', async (approved, mode) => {
    const launch = vi.fn<NonNullable<AgentRunDeps['launch']>>(async () => 0);
    await run(['github', ...(approved ? ['--caller-approved'] : []), '--repo', 'owner/repo', 'Check PR #148'], {
      resolveModelsDir: () => '/models',
      discoverModels: async () => [{ name: 'local' }],
      readPersistedDefault: () => ({ modelId: 'local' }),
      launch,
    });
    expect(launch).toHaveBeenCalledOnce();
    const options = launch.mock.calls[0]![0];
    expect(options).toMatchObject({ modelsDir: '/models' });
    expect(options.argv).toContain('--permission-mode');
    expect(options.argv[options.argv.indexOf('--permission-mode') + 1]).toBe(mode);
    expect(options.argv).not.toContain('--caller-approved');
    expect(options.argv[options.argv.indexOf('-p') + 1]).toContain('Check PR #148');
  });

  it('uses a focused worker prompt with the installed GitHub context', () => {
    const args = delegateAgentArgs(['github', '--repo', 'owner/repo', '--pr=42', 'Explain failed checks']);
    const context = args[args.indexOf('--append-system-prompt') + 1];
    expect(args.slice(0, DELEGATE_DEFAULT_ARGS.length)).toEqual(DELEGATE_DEFAULT_ARGS);
    expect(context).toContain('GitHub repository: owner/repo.');
    expect(context).toContain('Pull request: #42.');
    expect(context).toContain('read-only');
    expect(args[args.indexOf('-p') + 1]).toBe('Explain failed checks');
    expect(DELEGATE_SYSTEM_PROMPT).toContain('Do not invoke another agent');
  });

  it('accepts authorized-write context without granting permission or widening tools', () => {
    const args = delegateAgentArgs(['github', '--allow-write', '--repo=owner/repo', 'Post the approved comment']);
    expect(args[args.indexOf('--append-system-prompt') + 1]).toContain('explicitly authorized');
    expect(args).not.toContain('--allow-write');
    expect(args[args.indexOf('--tools') + 1]).toBe('read_file,run_terminal_command');
    expect(args[args.indexOf('--permission-mode') + 1]).toBe('auto');
  });

  it.each(['--system-prompt-override', '--append-system-prompt', '--model', '--resume', '--permission-mode'])(
    'does not consume a GitHub-looking value belonging to %s',
    (option) => {
      const args = delegateAgentArgs(['github', option, '--repo', '--pr', '42', 'Task']);
      expect(args.slice(-2)).toEqual([option, '--repo']);
      expect(args[args.indexOf('-p') + 1]).toBe('Task');
      expect(args[args.indexOf('--append-system-prompt') + 1]).toContain('Pull request: #42.');
    },
  );

  it('treats tokens after -- as literal prompt text', () => {
    const args = delegateAgentArgs(['github', '--repo', 'owner/repo', '--', '--model', '--repo', '--no-persist-cache']);
    expect(args).not.toContain('--');
    expect(args[args.indexOf('-p') + 1]).toBe('--model --repo --no-persist-cache');
    expect(args[args.indexOf('--append-system-prompt') + 1]).toContain('GitHub repository: owner/repo.');
  });

  it('keeps print prompts and conditional agent option values opaque', () => {
    const args = delegateAgentArgs([
      'github',
      '-p',
      'Task --repo other/repo',
      '--permission-mode',
      'auto',
      '--repo',
      'owner/repo',
    ]);
    expect(args[args.indexOf('-p') + 1]).toBe('Task --repo other/repo');
    expect(args[args.indexOf('--permission-mode') + 1]).toBe('auto');
    expect(args[args.indexOf('--append-system-prompt') + 1]).toContain('GitHub repository: owner/repo.');
  });

  it.each([
    ['github', '--repo'],
    ['github', '--repo', '--pr'],
    ['github', '--repo=bad'],
    ['github', '--pr'],
    ['github', '--pr', 'not-a-number'],
  ])('rejects invalid compatibility arguments before starting an agent: %j', (...args) => {
    expect(() => delegateAgentArgs(args)).toThrow();
  });

  it('prints delegate usage on --help and forwards --help to the agent binary without a server', async () => {
    const launch = vi.fn<NonNullable<AgentRunDeps['launch']>>(async () => 0);
    const stdout = vi.spyOn(console, 'log').mockImplementation(() => {});
    const previous = process.exitCode;
    try {
      await run(['--help'], { launch });
      expect(stdout).toHaveBeenCalledWith(expect.stringContaining('Usage: mlx delegate'));
      expect(launch).toHaveBeenCalledWith({ argv: ['--help'], needsServer: false });
      expect(process.exitCode).toBe(0);
    } finally {
      process.exitCode = previous;
      vi.restoreAllMocks();
    }
  });
});

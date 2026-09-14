import { describe, expect, it, vi } from 'vite-plus/test';

import { scanAgentArgs, withDefaultModel } from '../src/commands/agent/index.js';
import {
  DELEGATE_DEFAULT_ARGS,
  DELEGATE_SYSTEM_PROMPT,
  delegateAgentArgs,
  parseDelegateArgs,
  run,
} from '../src/commands/delegate.js';

describe('delegate agent arguments', () => {
  it('accepts a general prompt with normal agent options and stdin/file inputs', () => {
    const args = ['--thinking', 'high', '--mode', 'json', '--session', 'session-id', '@context.md', 'Explain this'];
    expect(delegateAgentArgs(args)).toEqual([...DELEGATE_DEFAULT_ARGS, ...args]);
  });

  it('keeps agent metadata and package commands usable without prompt mode', () => {
    for (const args of [
      ['--help'],
      ['--version'],
      ['--export', 'session.jsonl'],
      ['install', 'npm:extension'],
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
      expect(args.at(-1)).toBe('Check PR #148');
    },
  );

  it.each([
    ['--system-prompt', '--caller-approved'],
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

  it.each([false, true])('forwards approval only to the delegate runtime (%s)', async (approved) => {
    const runAgent = vi.fn<NonNullable<import('../src/commands/agent/index.js').AgentRunDeps['runAgent']>>(
      async () => {},
    );
    await run(['github', ...(approved ? ['--caller-approved'] : []), '--repo', 'owner/repo', 'Check PR #148'], {
      resolveModelsDir: () => '/models',
      discoverMlxModels: async () => [
        { discovered: { name: 'local', path: '/models/local', modelType: 'qwen3' }, piModel: {} } as never,
      ],
      readPersistedDefault: () => ({ provider: 'mlx', modelId: 'local' }),
      runAgent,
    });
    expect(runAgent).toHaveBeenCalledWith(
      expect.objectContaining({ mode: 'delegate', delegateCallerApproved: approved }),
    );
    expect(runAgent.mock.calls[0]?.[0]).not.toEqual(
      expect.objectContaining({ argv: expect.arrayContaining(['--caller-approved']) }),
    );
  });

  it('uses a focused worker prompt with the installed GitHub context', () => {
    const args = delegateAgentArgs(['github', '--repo', 'owner/repo', '--pr=42', 'Explain failed checks']);
    const context = args[args.indexOf('--append-system-prompt') + 1];
    expect(args.slice(0, DELEGATE_DEFAULT_ARGS.length)).toEqual(DELEGATE_DEFAULT_ARGS);
    expect(context).toContain('GitHub repository: owner/repo.');
    expect(context).toContain('Pull request: #42.');
    expect(context).toContain('read-only');
    expect(args.at(-1)).toBe('Explain failed checks');
    expect(DELEGATE_SYSTEM_PROMPT).toContain('Do not invoke another agent');
  });

  it('accepts authorized-write context without granting permission or disabling tools', () => {
    const args = delegateAgentArgs(['github', '--allow-write', '--repo=owner/repo', 'Post the approved comment']);
    expect(args[args.indexOf('--append-system-prompt') + 1]).toContain('explicitly authorized');
    expect(args).not.toContain('--allow-write');
    expect(args[args.indexOf('--tools') + 1]).toBe('read,bash');
    expect(args).not.toContain('--no-extensions');
  });

  it.each(['--system-prompt', '--append-system-prompt', '--model', '--session', '--extension'])(
    'does not consume a GitHub-looking value belonging to %s',
    (option) => {
      const args = delegateAgentArgs(['github', option, '--repo', '--pr', '42', 'Task']);
      expect(args.slice(-3)).toEqual([option, '--repo', 'Task']);
      expect(args[DELEGATE_DEFAULT_ARGS.length + 1]).toContain('Pull request: #42.');
    },
  );

  it('respects the end-of-options delimiter through model selection', () => {
    const args = delegateAgentArgs(['github', '--repo', 'owner/repo', '--', '--model', '--repo', '--no-persist-cache']);
    expect(args.slice(-4)).toEqual(['--', '--model', '--repo', '--no-persist-cache']);
    const scan = scanAgentArgs(args);
    expect(scan.persistPagedCache).toBe(true);
    expect(withDefaultModel(scan.passthrough, 'local').slice(0, 4)).toEqual([
      '--models',
      'mlx/*',
      '--model',
      'mlx/local',
    ]);
  });

  it('keeps print prompts and conditional agent option values opaque', () => {
    const args = delegateAgentArgs([
      'github',
      '--print',
      'Task --repo other/repo',
      '--use-theme',
      'dark',
      '--repo',
      'owner/repo',
    ]);
    expect(args.slice(-4)).toEqual(['--print', 'Task --repo other/repo', '--use-theme', 'dark']);
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
});

import { describe, expect, it } from 'vite-plus/test';

import { scanAgentArgs, withDefaultModel } from '../src/commands/agent/index.js';
import { DELEGATE_DEFAULT_ARGS, DELEGATE_SYSTEM_PROMPT, delegateAgentArgs } from '../src/commands/delegate.js';

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

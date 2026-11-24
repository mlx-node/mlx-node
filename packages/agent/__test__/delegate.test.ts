import { describe, expect, it } from 'vite-plus/test';

import { DELEGATION_PROMPT, delegationCommand, delegationPrompt, expandHome } from '../src/delegate.js';

describe('delegationCommand', () => {
  it('single-quotes POSIX paths and escapes embedded single quotes', () => {
    expect(delegationCommand('/Users/me/.mlx-node/bin/mlx', 'darwin')).toBe("'/Users/me/.mlx-node/bin/mlx'");
    expect(delegationCommand("/Users/o'reilly/bin/mlx", 'linux')).toBe("'/Users/o'\\''reilly/bin/mlx'");
    expect(delegationCommand('/Users/me/quote"dir/mlx', 'darwin')).toBe("'/Users/me/quote\"dir/mlx'");
  });

  it('formats PowerShell call-operator single-quoted Windows paths and rejects quotes/control chars', () => {
    expect(delegationCommand('C:\\Users\\My Name\\.mlx-node\\bin\\mlx.cmd', 'win32')).toBe(
      "& 'C:\\Users\\My Name\\.mlx-node\\bin\\mlx.cmd'",
    );
    expect(delegationCommand("C:\\Users\\O'Brien\\.mlx-node\\bin\\mlx.cmd", 'win32')).toBe(
      "& 'C:\\Users\\O''Brien\\.mlx-node\\bin\\mlx.cmd'",
    );
    expect(delegationCommand('C:\\Users\\O\u2019Brien\\.mlx-node\\bin\\mlx.cmd', 'win32')).toBe(
      "& 'C:\\Users\\O\u2019\u2019Brien\\.mlx-node\\bin\\mlx.cmd'",
    );
    expect(delegationCommand('\\\\server\\share\\mlx.cmd', 'win32')).toBe("& '\\\\server\\share\\mlx.cmd'");
    for (const bad of ['C:\\a"b\\mlx.cmd', 'C:\\a`b\\mlx.cmd', 'C:\\a\nb\\mlx.cmd', 'C:\\a\rb\\mlx.cmd', 'C:\\a\0b']) {
      expect(() => delegationCommand(bad, 'win32'), bad).toThrow('cannot be represented');
    }
  });

  it('rejects control characters and backticks on POSIX', () => {
    for (const bad of ['/a`b', '/a\nb', '/a\rb', '/a\0b']) {
      expect(() => delegationCommand(bad, 'darwin'), bad).toThrow('cannot be represented');
    }
  });
});

describe('delegationPrompt', () => {
  it('forwards the platform to delegationCommand', () => {
    const winPrompt = delegationPrompt('C:\\Users\\me\\.mlx-node\\bin\\mlx.cmd', 'win32');
    expect(winPrompt).toContain("`& 'C:\\Users\\me\\.mlx-node\\bin\\mlx.cmd' delegate github");
    expect(winPrompt).not.toBe(DELEGATION_PROMPT);

    // `$&` in a path must stay literal and not expand the matched `` `mlx delegate `` text.
    const dollarPath = 'C:\\Users\\me$&team\\.mlx-node\\bin\\mlx.cmd';
    expect(delegationPrompt(dollarPath, 'win32')).toContain(`\`& '${dollarPath}' delegate github`);
    expect(delegationPrompt('/Users/me$&team/bin/mlx', 'darwin')).toContain("`'/Users/me$&team/bin/mlx' delegate");

    const posixPrompt = delegationPrompt('/Users/me/.mlx-node/bin/mlx', 'darwin');
    expect(posixPrompt).toContain("`'/Users/me/.mlx-node/bin/mlx' delegate github");
  });
});

describe('expandHome', () => {
  it('expands ~ and ~/ on POSIX and leaves ~\\ literal', () => {
    expect(expandHome('~', '/Users/me', 'darwin')).toBe('/Users/me');
    expect(expandHome('~/.claude', '/Users/me', 'darwin')).toBe('/Users/me/.claude');
    expect(expandHome('~\\.claude', '/Users/me', 'darwin')).toBe('~\\.claude');
    expect(expandHome('/etc/claude', '/Users/me', 'darwin')).toBe('/etc/claude');
  });

  it('expands ~, ~/, and ~\\ on Windows', () => {
    expect(expandHome('~', 'C:\\Users\\me', 'win32')).toBe('C:\\Users\\me');
    expect(expandHome('~\\.claude', 'C:\\Users\\me', 'win32')).toBe('C:\\Users\\me\\.claude');
    expect(expandHome('~/.claude', 'C:\\Users\\me', 'win32')).toBe('C:\\Users\\me\\.claude');
    expect(expandHome('~other\\.claude', 'C:\\Users\\me', 'win32')).toBe('~other\\.claude');
    expect(expandHome('D:\\config\\claude', 'C:\\Users\\me', 'win32')).toBe('D:\\config\\claude');
  });
});

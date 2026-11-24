/**
 * `workCommandPath`: the `PATH` a Work command gets, asked of a real `/bin/zsh` or `/bin/bash` on a throwaway `HOME`
 * whose rc files the test writes (never the user's own). System files such as `/etc/zprofile` add entries, so the
 * checks compare order and membership, never the whole system `PATH`.
 */

import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { workCommandPath } from '../src/chat/agent/work-path.js';
import { until } from './helpers/chat-fake.js';

const NO_ZSH = !existsSync('/bin/zsh');

let root: string;
let home: string;
let app: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'mlx-work-path-'));
  home = join(root, 'home');
  app = join(root, 'app');
  mkdirSync(home);
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

function rc(name: string, text: string): void {
  writeFileSync(join(home, name), text);
}

function base(over: NodeJS.ProcessEnv = {}): NodeJS.ProcessEnv {
  return { PATH: `/usr/bin:/bin:${app}`, HOME: home, ...over };
}

function entries(path: string | undefined): string[] {
  return (path ?? '').split(':');
}

function gone(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return false;
  } catch {
    return true;
  }
}

describe('workCommandPath', () => {
  it.skipIf(NO_ZSH)("zsh: login files and the rc file, in the agent's order", async () => {
    rc('.zprofile', 'export PATH="$HOME/pro:$PATH"\n');
    rc('.zshrc', 'echo rc-noise-out\necho rc-noise-err >&2\nexport PATH="$HOME/rc:$PATH"\n');
    const path = await workCommandPath(base({ SHELL: '/bin/zsh' }), home);
    const list = entries(path);
    const at = (entry: string): number => list.indexOf(entry);
    for (const entry of [join(home, 'rc'), join(home, 'pro'), '/usr/bin', app]) {
      expect([entry, list.filter((e) => e === entry).length]).toEqual([entry, 1]);
    }
    expect(at(join(home, 'rc'))).toBeLessThan(at(join(home, 'pro')));
    expect(at(join(home, 'pro'))).toBeLessThan(at('/usr/bin'));
    expect(at('/usr/bin')).toBeLessThan(at(app));
    expect(path).not.toContain('noise');
  });

  it.skipIf(NO_ZSH)('zsh: what ~/.zshenv does in the command shell wins', async () => {
    rc('.zshenv', 'export PATH="/usr/bin:/bin:$HOME/envonly"\n');
    rc('.zshrc', 'export PATH="$HOME/rc:$PATH"\n');
    expect(await workCommandPath(base({ SHELL: '/bin/zsh' }), home)).toBe(`/usr/bin:/bin:${join(home, 'envonly')}`);
  });

  it.skipIf(NO_ZSH)('a login shell that fails gives the app PATH, and the command shell still answers', async () => {
    rc('.zprofile', 'exit 3\n');
    rc('.zshenv', 'export PATH="$HOME/envbin:$PATH"\n');
    expect(await workCommandPath(base({ SHELL: '/bin/zsh' }), home)).toBe(
      `${join(home, 'envbin')}:/usr/bin:/bin:${app}`,
    );
  });

  it.skipIf(NO_ZSH)('a login shell that prints its PATH and then exits non-zero gives the app PATH', async () => {
    rc('.zshrc', 'export PATH="$HOME/rcbin:$PATH"; trap "exit 4" EXIT\n');
    expect(await workCommandPath(base({ SHELL: '/bin/zsh' }), home)).toBe(`/usr/bin:/bin:${app}`);
  });

  it.skipIf(NO_ZSH)('rc output of any size before the PATH is skipped, as the agent reads it all', async () => {
    rc('.zshrc', 'head -c 2097152 /dev/zero\nexport PATH="$HOME/rc:$PATH"\n');
    expect(entries(await workCommandPath(base({ SHELL: '/bin/zsh' }), home))).toContain(join(home, 'rc'));
  });

  it.skipIf(NO_ZSH)('marked text over 1 MiB gives the app PATH', async () => {
    rc('.zshrc', "printf '\\001'\nhead -c 2097152 /dev/zero\n");
    expect(await workCommandPath(base({ SHELL: '/bin/zsh' }), home)).toBe(`/usr/bin:/bin:${app}`);
  });

  it.skipIf(NO_ZSH)('a login shell past the limit is stopped', async () => {
    rc('.zshrc', 'echo $$ > "$HOME/pid"\nsleep 20\n');
    const started = Date.now();
    expect(await workCommandPath(base({ SHELL: '/bin/zsh' }), home, 300)).toBe(`/usr/bin:/bin:${app}`);
    expect(Date.now() - started).toBeLessThan(3_000);
    const pid = Number(readFileSync(join(home, 'pid'), 'utf8').trim());
    await until(() => gone(pid), 2_000);
  });

  it.skipIf(NO_ZSH)('an rc job that keeps the pipe open counts as too slow', async () => {
    rc('.zshrc', 'sleep 20 &\nexport PATH="$HOME/rc:$PATH"\n');
    const started = Date.now();
    expect(await workCommandPath(base({ SHELL: '/bin/zsh' }), home, 300)).toBe(`/usr/bin:/bin:${app}`);
    expect(Date.now() - started).toBeLessThan(3_000);
  });

  it('bash for any other shell, with BASH_ENV in the command shell', async () => {
    rc('.bash_profile', 'export PATH="$HOME/pro:$PATH"\n');
    rc('.bashrc', 'export PATH="$HOME/rc:$PATH"\n');
    rc('benv', 'export PATH="$HOME/benv:$PATH"\n');
    const path = await workCommandPath(base({ SHELL: '/opt/homebrew/bin/fish', BASH_ENV: join(home, 'benv') }), home);
    const list = entries(path);
    expect(list[0]).toBe(join(home, 'benv'));
    expect(list).toEqual(expect.arrayContaining([join(home, 'rc'), join(home, 'pro')]));

    rmSync(home, { recursive: true });
    mkdirSync(home);
    expect(entries(await workCommandPath(base(), home)).at(-1)).toBe(app);
  });

  it.skipIf(NO_ZSH)('GROK_SHELL names the shell when it is that kind', async () => {
    mkdirSync(join(root, 'alt'));
    const wrapper = `#!/bin/sh\nPATH="$PATH:${join(root, 'via')}"\nexport PATH\nexec /bin/zsh "$@"\n`;
    for (const name of ['zsh', 'sh']) {
      writeFileSync(join(root, 'alt', name), wrapper);
      chmodSync(join(root, 'alt', name), 0o755);
    }
    const via = join(root, 'via');
    const named = await workCommandPath(base({ SHELL: '/bin/zsh', GROK_SHELL: join(root, 'alt', 'zsh') }), home);
    expect(entries(named)).toContain(via);
    const other = await workCommandPath(base({ SHELL: '/bin/zsh', GROK_SHELL: join(root, 'alt', 'sh') }), home);
    expect(entries(other)).not.toContain(via);
  });

  it.skipIf(NO_ZSH)('the shell does not get the app internals or provider keys', async () => {
    rc('.zshenv', 'echo "${NODE_OPTIONS:-none}|${FIRECRAWL_API_KEY:-none}" > "$HOME/seen"\n');
    await workCommandPath(base({ SHELL: '/bin/zsh', NODE_OPTIONS: '--inspect', FIRECRAWL_API_KEY: 'fc-key-1' }), home);
    expect(readFileSync(join(home, 'seen'), 'utf8').trim()).toBe('none|none');
  });

  it('a shell that cannot be started gives the app PATH', async () => {
    expect(await workCommandPath(base(), join(root, 'missing'))).toBe(`/usr/bin:/bin:${app}`);
  });
});

/**
 * `login-path.ts`: the login shell's `PATH` for the Work agent of the app. Real shells run on a throwaway `HOME` whose rc
 * files the test writes (never the user's own); fake shells are `/bin/sh` scripts named after the shell they stand for.
 */

import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { workAgentEnv } from '../src/chat/agent/env.js';
import {
  LAUNCHD_PATH,
  loginPathScript,
  markedPath,
  mergePath,
  needsLoginPath,
  resolveLoginPath,
} from '../src/chat/agent/login-path.js';
import { until } from './helpers/chat-fake.js';

const NO_ZSH = !existsSync('/bin/zsh');
const MARKERS = { begin: 'MLX_PATH_BEGIN_ab12', end: 'MLX_PATH_END_ab12' };
const endpoint = { url: 'http://127.0.0.1:4321', token: 'tok-INFERENCE' };

let root: string;
let home: string;
let bin: string;

beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'mlx-login-path-'));
  home = join(root, 'home');
  bin = join(root, 'bin');
  mkdirSync(home);
  mkdirSync(bin);
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

function rc(name: string, text: string): void {
  writeFileSync(join(home, name), text);
}

/** A `/bin/sh` script standing in for a shell; it gets `-ilc` as `$1` and the script as `$2`. */
function fakeShell(name: string, body: string): string {
  const path = join(bin, name);
  writeFileSync(path, `#!/bin/sh\n${body}\n`);
  chmodSync(path, 0o755);
  return path;
}

function gone(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return false;
  } catch {
    return true;
  }
}

function readPid(name: string): number {
  return Number(readFileSync(join(root, name), 'utf8').trim());
}

describe('markedPath', () => {
  // Mutation: slicing to the end of the output instead of the end marker → the rc noise after it joins the PATH.
  it('reads the text between the markers out of rc noise before and after', () => {
    const out = `Last login: today\nwelcome\n${MARKERS.begin}/a/bin:/usr/bin${MARKERS.end}\nbye from .zlogout\n`;
    expect(markedPath(out, MARKERS)).toBe('/a/bin:/usr/bin');
  });

  // Mutation: the end marker searched from the start of the output → an end printed first gives null.
  it('skips an end marker printed before the begin marker', () => {
    expect(markedPath(`${MARKERS.end}${MARKERS.begin}/a/bin${MARKERS.end}`, MARKERS)).toBe('/a/bin');
  });

  // Mutations: a missing marker not checked (slices from index -1), and an empty PATH taken as an answer.
  it('is null without the begin marker, without the end marker, or with nothing between', () => {
    expect(markedPath('noise only\n', MARKERS)).toBeNull();
    expect(markedPath(`/a/bin${MARKERS.end}`, MARKERS)).toBeNull();
    expect(markedPath(`noise${MARKERS.begin}/a/bin`, MARKERS)).toBeNull();
    expect(markedPath(`noise${MARKERS.begin}${MARKERS.end}noise`, MARKERS)).toBeNull();
  });
});

describe('mergePath', () => {
  // Mutations: the inherited entries first; no dedupe; empty entries kept.
  it('puts the login entries first, then the inherited ones it lacks, each once, with no empty entry', () => {
    expect(mergePath('/login/bin:/usr/bin', LAUNCHD_PATH)).toBe('/login/bin:/usr/bin:/bin:/usr/sbin:/sbin');
    expect(mergePath('/a::/b:/a:', ':/c:/b::')).toBe('/a:/b:/c');
    expect(mergePath('/a', undefined)).toBe('/a');
  });
});

describe('needsLoginPath', () => {
  // Mutations: either condition dropped, or Windows not excluded.
  it('in the app or under launchd’s PATH, never from a terminal and never on Windows', () => {
    expect(needsLoginPath({ PATH: '/Users/dev/bin:/usr/bin', MLX_DESKTOP_CLI: '{}' }, 'darwin')).toBe(true);
    expect(needsLoginPath({ PATH: LAUNCHD_PATH }, 'darwin')).toBe(true);
    expect(needsLoginPath({ PATH: `/opt/homebrew/bin:${LAUNCHD_PATH}` }, 'darwin')).toBe(false);
    expect(needsLoginPath({ PATH: LAUNCHD_PATH, MLX_DESKTOP_CLI: '{}' }, 'win32')).toBe(false);
  });
});

describe('loginPathScript', () => {
  // Mutation: fish given the POSIX script → its list PATH prints space-separated.
  it('prints PATH between the markers: "$PATH" for POSIX shells, a joined list for fish, nothing for others', () => {
    for (const shell of ['/bin/zsh', '/bin/bash', '/bin/sh', '/opt/homebrew/bin/bash', '/bin/ksh', '/bin/dash']) {
      expect([shell, loginPathScript(shell, MARKERS)]).toEqual([
        shell,
        `printf '%s%s%s' '${MARKERS.begin}' "$PATH" '${MARKERS.end}'`,
      ]);
    }
    expect(loginPathScript('/opt/homebrew/bin/fish', MARKERS)).toBe(
      `printf '%s%s%s' '${MARKERS.begin}' (string join : $PATH) '${MARKERS.end}'`,
    );
    for (const shell of ['/opt/homebrew/bin/nu', '/bin/tcsh', '/bin/csh', '/usr/local/bin/xonsh']) {
      expect([shell, loginPathScript(shell, MARKERS)]).toEqual([shell, null]);
    }
  });
});

describe('resolveLoginPath', () => {
  // Mutations: `-lc` (no `.zshrc`, where most PATH lines live), or the whole app env handed to the shell.
  it.skipIf(NO_ZSH)('runs an interactive login zsh with a small env and reads its PATH out of the noise', async () => {
    rc('.zprofile', 'echo profile-noise\nexport PATH="$HOME/pro:$PATH"\n');
    rc('.zshrc', 'echo rc-noise\nexport PATH="$HOME/rc:$PATH"\nenv > "$HOME/seen-env"\n');
    const logs: string[] = [];
    const path = await resolveLoginPath({
      base: {
        PATH: LAUNCHD_PATH,
        HOME: home,
        USER: 'dev',
        MLX_DESKTOP_CLI: '{}',
        NODE_OPTIONS: '--inspect',
        EXA_API_KEY: 'app-key',
      },
      home,
      shell: '/bin/zsh',
      log: (line) => logs.push(line),
    });
    expect(logs).toEqual([]);
    const list = (path ?? '').split(':');
    expect(list[0]).toBe(join(home, 'rc'));
    expect(list).toContain(join(home, 'pro'));
    expect(list.indexOf(join(home, 'rc'))).toBeLessThan(list.indexOf(join(home, 'pro')));
    expect(path).not.toContain('noise');
    const seen = readFileSync(join(home, 'seen-env'), 'utf8');
    expect(seen).toContain('TERM=dumb\n');
    expect(seen).toContain('SHELL=/bin/zsh\n');
    expect(seen).toContain('USER=dev\n');
    for (const name of ['MLX_DESKTOP_CLI', 'NODE_OPTIONS', 'EXA_API_KEY']) expect(seen).not.toContain(`${name}=`);
  });

  // Mutation: the parser hands back the whole output → the rc's printed key line lands in the Work PATH.
  it.skipIf(NO_ZSH)('takes only PATH: a key the rc file exports and prints never reaches Work', async () => {
    rc('.zshrc', 'export EXA_API_KEY=rc-key\necho EXA_API_KEY=$EXA_API_KEY\nexport PATH="$HOME/rc:$PATH"\n');
    const base = { PATH: LAUNCHD_PATH, HOME: home, EXA_API_KEY: 'app-key' };
    const loginPath = await resolveLoginPath({ base, home, shell: '/bin/zsh', log: () => {} });
    expect(loginPath?.startsWith(`${join(home, 'rc')}:`)).toBe(true);
    const env = workAgentEnv({ base, endpoint, agentHome: '/Users/dev/.mlx-agent', providerKeys: {}, loginPath });
    expect(env).not.toHaveProperty('EXA_API_KEY');
    expect(JSON.stringify(env)).not.toContain('rc-key');
    expect(env.PATH).toBe(mergePath(loginPath!, LAUNCHD_PATH));
    const chosen = workAgentEnv({
      base,
      endpoint,
      agentHome: '/Users/dev/.mlx-agent',
      providerKeys: { EXA_API_KEY: 'sel' },
      loginPath,
    });
    expect(chosen.EXA_API_KEY).toBe('sel');
  });

  // Mutations: no timeout (the call hangs), or only the shell killed (its `sleep` child lives on).
  it('a shell past the limit is killed with its children and gives null with one log line', async () => {
    const shell = fakeShell(
      'zsh',
      `echo $$ > "${root}/shell.pid"\n/bin/sleep 30 &\necho $! > "${root}/child.pid"\nwait`,
    );
    const logs: string[] = [];
    const started = Date.now();
    const path = await resolveLoginPath({
      base: { HOME: home },
      home,
      shell,
      timeoutMs: 300,
      log: (l) => logs.push(l),
    });
    expect(path).toBeNull();
    expect(Date.now() - started).toBeLessThan(3_000);
    expect(logs).toEqual([`[work] login shell ${shell} gave no PATH (no PATH in 300 ms); Work keeps the app PATH`]);
    await until(() => gone(readPid('shell.pid')) && gone(readPid('child.pid')), 3_000);
  });

  // Mutation: the answer read only when the shell closes its output → an rc job holding the pipe runs into the limit.
  it('answers at the end marker while the shell still holds its output open', async () => {
    const shell = fakeShell('bash', `echo $$ > "${root}/shell.pid"\n/bin/sh -c "$2"\n/bin/sleep 30`);
    const started = Date.now();
    const path = await resolveLoginPath({ base: { HOME: home, PATH: '/x/bin' }, home, shell, timeoutMs: 2_000 });
    expect(path).toBe('/x/bin');
    expect(Date.now() - started).toBeLessThan(1_500);
    await until(() => gone(readPid('shell.pid')), 5_000);
  });

  // Mutation: an exit with no marker taken as an empty answer.
  it('a shell that prints no PATH gives null and one log line', async () => {
    const shell = fakeShell('zsh', 'echo only noise\nexit 0');
    const logs: string[] = [];
    expect(await resolveLoginPath({ base: { HOME: home }, home, shell, log: (l) => logs.push(l) })).toBeNull();
    expect(logs).toEqual([
      `[work] login shell ${shell} gave no PATH (exited 0 with no PATH printed); Work keeps the app PATH`,
    ]);
  });

  // Mutation: an unknown shell run with the POSIX script.
  it('a shell that is neither POSIX nor fish is not started', async () => {
    const shell = fakeShell('nu', `/usr/bin/touch "${root}/ran"`);
    const logs: string[] = [];
    expect(await resolveLoginPath({ base: { HOME: home }, home, shell, log: (l) => logs.push(l) })).toBeNull();
    expect(existsSync(join(root, 'ran'))).toBe(false);
    expect(logs).toEqual([`[work] login shell ${shell} is neither POSIX nor fish; Work keeps the app PATH`]);
  });
});

/**
 * Never allow: the scopes a command card offers, which commands a rule matches, and the rule file (one per app,
 * per Work folder, read once, written whole after the write before it).
 */

import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  isRuleId,
  match,
  NEVER_CHARS_MAX,
  NEVER_NOTE_HEAD,
  NEVER_NOTE_TAIL,
  NEVER_RULES_MAX,
  NEVER_WORDS_MAX,
  neverNote,
  NeverRules,
  offer,
  parseNeverFile,
  type NeverRule,
} from '../src/chat/agent/never-rules.js';
import { chatPaths } from '../src/chat/paths.js';
import {
  NEVER_NOTE_HEAD as PAGE_HEAD,
  NEVER_NOTE_TAIL as PAGE_TAIL,
  neverNoteWords,
} from '../ui/src/lib/permission-text.js';

function rule(words: string[], id = `n_${'0'.repeat(31)}1`): NeverRule {
  return { id, words, at: 1 };
}

describe('offer: the two scopes of a command', () => {
  it('git push origin main → git push, or every word', () => {
    expect(offer('git push origin main')).toEqual({
      short: ['git', 'push'],
      exact: ['git', 'push', 'origin', 'main'],
    });
  });

  it('rm -rf build → rm, or every word', () => {
    expect(offer('rm -rf build')).toEqual({ short: ['rm'], exact: ['rm', '-rf', 'build'] });
  });

  it('a tool with subcommands takes the subcommand; an option in its place does not', () => {
    expect(offer('npm install left-pad')?.short).toEqual(['npm', 'install']);
    expect(offer('cargo test --release')?.short).toEqual(['cargo', 'test']);
    expect(offer('git -C repo push')?.short).toEqual(['git']);
  });

  it('skips assignments, plain wrappers and setup commands: the first real command', () => {
    expect(offer('FOO=1 git push origin')).toEqual({ short: ['git', 'push'], exact: ['git', 'push', 'origin'] });
    expect(offer('cd x && git push')).toEqual({ short: ['git', 'push'], exact: ['git', 'push'] });
    expect(offer('env A=1 timeout 30 nohup git push')?.exact).toEqual(['git', 'push']);
    expect(offer('cd x')).toBeNull();
  });

  it('unread text offers nothing', () => {
    expect(offer('echo $(whoami)')).toBeNull();
    expect(offer('git push &')).toBeNull();
  });

  it('cuts exact at the caps, never below short; a redacted or hidden part is not offered', () => {
    const many = Array.from({ length: 30 }, (_, i) => `a${i}`);
    expect(offer(`tool ${many.join(' ')}`)?.exact).toHaveLength(NEVER_WORDS_MAX);
    const long = 'x'.repeat(150);
    const cut = offer(`tool ${long} ${long}`)!;
    expect(cut.exact).toEqual(['tool', long]);
    expect(cut.exact.join(' ').length).toBeLessThanOrEqual(NEVER_CHARS_MAX);
    const key = 'sk-secret-0123456789';
    expect(offer(`curl -H ${key} https://x`, [key])).toEqual({ short: ['curl'], exact: ['curl', '-H'] });
    expect(offer(`${key} run`, [key])).toBeNull();
    expect(offer("echo 'a\u2800b'")).toEqual({ short: ['echo'], exact: ['echo'] });
    expect(offer("echo 'a\nb' c")).toEqual({ short: ['echo'], exact: ['echo'] });
    expect(offer('echo \u202eb')).toBeNull();
    expect(offer('x'.repeat(NEVER_CHARS_MAX + 1))).toBeNull();
  });
});

describe('match: a simple command that starts with the rule', () => {
  const push = rule(['git', 'push']);

  it('word-equal prefix only', () => {
    expect(match('git push origin main', [push])).toBe(push);
    expect(match('git  push', [push])).toBe(push);
    expect(match('git pushx', [push])).toBeNull();
    expect(match('git pull', [push])).toBeNull();
    expect(match('git', [push])).toBeNull();
  });

  it('quoted words read without their quotes', () => {
    expect(match('git "push" origin', [push])).toBe(push);
    expect(match("'git' push", [push])).toBe(push);
  });

  it('any simple command of a list or pipeline; assignments and plain wrappers are peeled', () => {
    expect(match('git status && git push', [push])).toBe(push);
    expect(match('echo hi; git push', [push])).toBe(push);
    expect(match('true | git push', [push])).toBe(push);
    expect(match('cd x && GIT_TRACE=1 git push', [push])).toBe(push);
    expect(match('env A=1 git push', [push])).toBe(push);
    expect(match('timeout 30 git push', [push])).toBe(push);
    expect(match('/usr/bin/git push', [push])).toBe(push);
  });

  it('a rule written as a path matches only that path', () => {
    const script = rule(['./deploy.sh']);
    expect(match('./deploy.sh prod', [script])).toBe(script);
    expect(match('deploy.sh prod', [script])).toBeNull();
  });

  it('unread with rules, nothing with none', () => {
    expect(match('echo $(git push)', [push])).toBe('unread');
    expect(match('echo $(git push)', [])).toBeNull();
    expect(match('git push', [])).toBeNull();
  });
});

describe('match: the ways around a rule', () => {
  const push = rule(['git', 'push']);
  const linux = { foldCase: false };

  // Mutation: drop `sudo` from the wrappers → each of these reads as the program `sudo`.
  it('sudo, with its options and assignments', () => {
    expect(match('sudo git push', [push], linux)).toBe(push);
    expect(match('sudo -E git push origin', [push], linux)).toBe(push);
    expect(match('sudo -u root -H git push', [push], linux)).toBe(push);
    expect(match('sudo -uroot git push', [push], linux)).toBe(push);
    expect(match('sudo --user=root -- git push', [push], linux)).toBe(push);
    expect(match('sudo --user root git push', [push], linux)).toBe(push);
    expect(match('sudo GIT_TRACE=1 git push', [push], linux)).toBe(push);
    expect(match('/usr/bin/sudo git push', [push], linux)).toBe(push);
    expect(match('sudo git pull', [push], linux)).toBeNull();
  });

  it('env, with its options and its NAME=value words', () => {
    expect(match('env git push', [push], linux)).toBe(push);
    expect(match('env A=1 B=two git push', [push], linux)).toBe(push);
    expect(match('env -i A=1 git push', [push], linux)).toBe(push);
    expect(match('env - git push', [push], linux)).toBe(push);
    expect(match('env -u HOME git push', [push], linux)).toBe(push);
    expect(match('env -C repo git push', [push], linux)).toBe(push);
    expect(match('env -- git push', [push], linux)).toBe(push);
    expect(match('/usr/bin/env A=1 git push', [push], linux)).toBe(push);
  });

  it('command, exec, nohup, time, nice and timeout', () => {
    expect(match('command git push', [push], linux)).toBe(push);
    expect(match('command -p git push', [push], linux)).toBe(push);
    expect(match('command -v git push', [push], linux)).toBeNull();
    expect(match('exec git push', [push], linux)).toBe(push);
    expect(match('exec -a deploy git push', [push], linux)).toBe(push);
    expect(match('exec -cl git push', [push], linux)).toBe(push);
    expect(match('nohup git push', [push], linux)).toBe(push);
    expect(match('nohup -- git push', [push], linux)).toBe(push);
    expect(match('time git push', [push], linux)).toBe(push);
    expect(match('time -p git push', [push], linux)).toBe(push);
    expect(match('nice -n 10 git push', [push], linux)).toBe(push);
    expect(match('nice -10 git push', [push], linux)).toBe(push);
    expect(match('timeout -s KILL 30 git push', [push], linux)).toBe(push);
    expect(match('timeout --preserve-status 5m git push', [push], linux)).toBe(push);
  });

  it('wrappers in a row, each read in turn; a rule on a wrapper itself still matches', () => {
    expect(match('sudo env A=1 nohup time git push', [push], linux)).toBe(push);
    expect(match('env A=1 sudo -n git push', [push], linux)).toBe(push);
    const sudo = rule(['sudo'], `n_${'0'.repeat(31)}2`);
    expect(match('env A=1 sudo git status', [sudo], linux)).toBe(sudo);
  });

  // Mutation: return null from `shellLine` → the line inside `-c` is never read.
  it('the line of sh -c, bash -c and zsh -c, read with the same scanner', () => {
    expect(match("sh -c 'git push'", [push], linux)).toBe(push);
    expect(match('bash -c "git push origin main"', [push], linux)).toBe(push);
    expect(match("zsh -c 'git push'", [push], linux)).toBe(push);
    expect(match("bash -lc 'cd repo && git push'", [push], linux)).toBe(push);
    expect(match("bash -e -o pipefail -c 'make; git push'", [push], linux)).toBe(push);
    expect(match("bash --norc -c 'git push'", [push], linux)).toBe(push);
    expect(match("/bin/sh -c -- 'git push'", [push], linux)).toBe(push);
    expect(match(`sudo bash -c "sh -c 'env A=1 git push'"`, [push], linux)).toBe(push);
    expect(match("bash -c 'git status' git push", [push], linux)).toBeNull();
    expect(match('bash deploy.sh git push', [push], linux)).toBeNull();
    expect(match("bash -c 'git pull'", [push], linux)).toBeNull();
  });

  it('a -c line the app cannot read is unread; a rule found elsewhere in the command wins', () => {
    expect(match(`bash -c 'echo $(whoami)'`, [push], linux)).toBe('unread');
    expect(match(`bash -c 'echo $(whoami)'; git push`, [push], linux)).toBe(push);
    expect(match(`bash -c 'echo $(whoami)'`, [], linux)).toBeNull();
  });

  it('wrappers past the layer budget are unread', () => {
    expect(match(`${'nohup '.repeat(7)}git push`, [push], linux)).toBe(push);
    expect(match(`${'nohup '.repeat(9)}git push`, [push], linux)).toBe('unread');
  });

  // Mutation: read an unknown wrapper option as "runs nothing" → `env -S "git push"` passes a folder with rules in Bypass.
  it('a wrapper option the app does not know is unread; a wrapper that runs nothing is not', () => {
    expect(match('sudo --frobnicate git push', [push], linux)).toBe('unread');
    expect(match('env -S "git push"', [push], linux)).toBe('unread');
    expect(match('env --split-string="git push"', [push], linux)).toBe('unread');
    expect(match('nohup -x git push', [push], linux)).toBe('unread');
    expect(match('timeout --bogus 5 git push', [push], linux)).toBe('unread');
    expect(match('timeout 0x10 git push', [push], linux)).toBe('unread');
    expect(match('env -S "git push"; git push', [push], linux)).toBe(push);
    const sudo = rule(['sudo'], `n_${'0'.repeat(31)}4`);
    expect(match('sudo --frobnicate git push', [sudo], linux)).toBe(sudo);
    expect(match('command -v git push', [push], linux)).toBeNull();
    expect(match('sudo -v', [push], linux)).toBeNull();
    expect(match('env A=1', [push], linux)).toBeNull();
    expect(match('env -S "git push"', [], linux)).toBeNull();
  });

  // Mutation: compare the program word as written on macOS → `GIT push` passes the rule.
  it('macOS finds a program by its name in any case: the program word is compared without case', () => {
    const mac = { foldCase: true };
    expect(match('GIT push', [push], mac)).toBe(push);
    expect(match('/usr/bin/Git push origin', [push], mac)).toBe(push);
    expect(match("SUDO Env A=1 BASH -c 'Git push'", [push], mac)).toBe(push);
    expect(match('git PUSH', [push], mac)).toBeNull();
    expect(match('GIT push', [push], linux)).toBeNull();
    expect(match('SUDO git push', [push], linux)).toBeNull();
    const script = rule(['./deploy.sh'], `n_${'0'.repeat(31)}3`);
    expect(match('./Deploy.SH prod', [script], mac)).toBe(script);
    expect(match('./Deploy.SH prod', [script], linux)).toBeNull();
  });

  it("the host's default is the platform's", () => {
    expect(match('GIT push', [push])).toBe(process.platform === 'darwin' ? push : null);
  });
});

describe('the note', () => {
  it('names the words; the page reads the same words back from it', () => {
    expect(neverNote('git push')).toBe(
      'I never allow commands that start with `git push` in this folder. Do not run it, and do not get the same result another way. Go on with the rest of the task, or stop and say what is blocked.',
    );
    expect(PAGE_HEAD).toBe(NEVER_NOTE_HEAD);
    expect(PAGE_TAIL).toBe(NEVER_NOTE_TAIL);
    expect(neverNoteWords(neverNote('git push'))).toBe('git push');
    expect(neverNoteWords(neverNote('echo `x`'))).toBe('echo `x`');
    expect(neverNoteWords('use pnpm, not npm')).toBeNull();
    expect(neverNoteWords(`${NEVER_NOTE_HEAD}${NEVER_NOTE_TAIL}`)).toBeNull();
  });
});

describe('the rule file', () => {
  let dir: string;
  let path: string;
  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), 'never-rules-'));
    path = join(dir, 'never-allow.json');
  });
  afterEach(() => rmSync(dir, { recursive: true, force: true }));

  it('the app keeps it in its chat folder', () => {
    expect(chatPaths('/r').neverAllow).toBe('/r/never-allow.json');
  });

  it('round-trips per folder, mode 0600; a folder compares as the disk does', async () => {
    const store = new NeverRules(path, { now: () => 5 });
    await store.ready();
    const { rule: made, saved } = store.add('/Users/me/Shop', ['git', 'push']);
    expect(await saved).toBe(true);
    expect(made).toMatchObject({ words: ['git', 'push'], at: 5 });
    expect(isRuleId(made!.id)).toBe(true);
    expect(statSync(path).mode & 0o777).toBe(0o600);
    expect(JSON.parse(readFileSync(path, 'utf8'))).toEqual({
      version: 1,
      folders: { '/Users/me/Shop': [{ id: made!.id, words: ['git', 'push'], at: 5 }] },
    });
    const again = new NeverRules(path);
    await again.ready();
    expect(again.rulesOf('/users/me/shop')).toEqual([made]);
    expect(again.rulesOf('/Users/me/other')).toEqual([]);
    expect(again.wordsOf('/Users/me/Shop', made!.id)).toBe('git push');
    expect(await again.remove('/Users/me/Shop', made!.id)).toBe('removed');
    expect(again.wordsOf('/Users/me/Shop', made!.id)).toBeNull();
    expect(JSON.parse(readFileSync(path, 'utf8'))).toEqual({ version: 1, folders: {} });
    expect(await again.remove('/Users/me/Shop', made!.id)).toBe('not-found');
  });

  it('the same words again are the rule there is; a full folder takes no more', async () => {
    const store = new NeverRules(path);
    await store.ready();
    const first = store.add('/p', ['rm']);
    await first.saved;
    const same = store.add('/p', ['rm']);
    expect(same.rule).toBe(first.rule);
    for (let i = 1; i < NEVER_RULES_MAX; i++) await store.add('/p', [`t${i}`]).saved;
    expect(store.rulesOf('/p')).toHaveLength(NEVER_RULES_MAX);
    const over = store.add('/p', ['one-more']);
    expect(over.rule).toBeNull();
    expect(await over.saved).toBe(false);
  });

  it('a rule matches from the moment it is added; a write that fails takes it out again', async () => {
    const store = new NeverRules(join(dir, 'missing', 'never-allow.json'));
    await store.ready();
    const { rule: made, saved } = store.add('/p', ['git', 'push']);
    expect(match('git push', store.rulesOf('/p'))).toBe(made);
    expect(await saved).toBe(false);
    expect(store.rulesOf('/p')).toEqual([]);
  });

  it('a broken file reads as no rules and is not written again until a rule is added', async () => {
    writeFileSync(path, '{"version":1,"folders":', { mode: 0o600 });
    const store = new NeverRules(path);
    await store.ready();
    expect(store.rulesOf('/p')).toEqual([]);
    expect(await store.remove('/p', `n_${'0'.repeat(32)}`)).toBe('not-found');
    expect(readFileSync(path, 'utf8')).toBe('{"version":1,"folders":');
    await store.add('/p', ['rm']).saved;
    expect(
      parseNeverFile(readFileSync(path, 'utf8'))
        .get('/p')
        ?.rules.map((r) => r.words),
    ).toEqual([['rm']]);
  });

  it('a rule id is n_ and 32 lowercase hex digits', () => {
    expect(isRuleId(`n_${'0123456789abcdef'.repeat(2)}`)).toBe(true);
    expect(isRuleId(`n_${'A'.repeat(32)}`)).toBe(false);
    expect(isRuleId(`n_${'a'.repeat(31)}`)).toBe(false);
    expect(isRuleId(`x_${'a'.repeat(32)}`)).toBe(false);
    expect(isRuleId(1)).toBe(false);
  });

  it('reads only the known shape: bad rules are skipped, caps hold', () => {
    const id = (n: number): string => `n_${n.toString(16).padStart(32, '0')}`;
    const text = JSON.stringify({
      version: 1,
      folders: {
        '/p': [
          { id: id(1), words: ['git', 'push'], at: 1 },
          { id: 'bad', words: ['rm'], at: 1 },
          { id: id(2), words: [], at: 1 },
          { id: id(3), words: ['a\u0001'], at: 1 },
          { id: id(4), words: ['x'.repeat(NEVER_CHARS_MAX + 1)], at: 1 },
          { id: id(5), words: Array.from({ length: NEVER_WORDS_MAX + 1 }, () => 'w'), at: 1 },
          { id: id(6), words: ['git', 'push'], at: 2 },
          { id: id(7), words: ['rm'], at: 'x' },
        ],
        relative: [{ id: id(8), words: ['rm'], at: 1 }],
        'C:\\Users\\me\\project': [{ id: id(9), words: ['git', 'push'], at: 1 }],
        '\\\\server\\share\\repo': [{ id: id(10), words: ['rm'], at: 1 }],
      },
    });
    const folders = parseNeverFile(text);
    expect([...folders.keys()]).toEqual(['/p', 'c:\\users\\me\\project', '\\\\server\\share\\repo']);
    expect(folders.get('/p')!.rules).toEqual([{ id: id(1), words: ['git', 'push'], at: 1 }]);
    expect(parseNeverFile('[]').size).toBe(0);
    expect(parseNeverFile(JSON.stringify({ version: 2, folders: {} })).size).toBe(0);
  });
});

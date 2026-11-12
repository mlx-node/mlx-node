/**
 * The Auto decision, one case per row of its rule table: what still asks (and why) in Auto, what the app
 * may answer `allow-once` itself, and that outside Auto every request is the plain card.
 */

import { describe, expect, it } from 'vite-plus/test';

import type { PermissionRequest } from '../src/chat/acp/messages.js';
import type { AutoScan } from '../src/chat/agent/auto-scan.js';
import {
  backgroundInput,
  AUTO_PER_TURN_MAX,
  decideAuto,
  type AutoFacts,
  type AutoInput,
} from '../src/chat/agent/auto.js';
import { workGuardHookName } from '../src/chat/agent/profiles.js';
import type { CommandFlag, PermissionInteraction } from '../src/chat/types.js';

const GUARD = workGuardHookName('mlx-work-guard-0123456789abcdef');
const OLDER_GUARD = workGuardHookName('mlx-work-guard-fedcba9876543210');
const OPTIONS = [
  { optionId: 'always-allow', name: 'always', kind: 'allow_always' },
  { optionId: 'allow-once', name: 'once', kind: 'allow_once' },
  { optionId: 'reject-once', name: 'no', kind: 'reject_once' },
];
const CLEAN: AutoScan = {
  unread: false,
  unknown: false,
  protectedName: false,
  background: false,
  keyName: false,
  git: false,
  changes: false,
  targets: [],
  checks: [],
  steps: 1,
};
const FACTS: AutoFacts = {
  scan: CLEAN,
  flags: [],
  placed: true,
  outside: false,
  protectedPath: false,
  links: false,
  repository: true,
  landing: null,
  landingOutside: false,
};

function request(tool: string, rawInput: unknown, over: Partial<PermissionRequest> = {}): PermissionRequest {
  return {
    sessionId: 's1',
    toolCallId: 'call-1',
    kind: 'execute',
    title: tool,
    tool: { name: tool, namespace: tool.includes('__') ? 'mcp' : 'grok_build' },
    rawInput,
    options: OPTIONS,
    ...over,
  };
}

function card(
  access: PermissionInteraction['access'],
  over: Partial<PermissionInteraction> = {},
): PermissionInteraction {
  return {
    id: 'i1',
    kind: 'permission',
    toolCallId: 'call-1',
    toolName: 'tool',
    access,
    title: 'title',
    input: '{}',
    command: access === 'bash' ? 'ls' : null,
    path: access === 'edit' ? 'a.txt' : null,
    resolvedPath: null,
    options: [],
    flags: [],
    ...over,
  };
}

/** A shell request that passes every row. */
function bash(over: Partial<AutoInput> = {}, scan: Partial<AutoScan> = {}, facts: Partial<AutoFacts> = {}) {
  return decideAuto({
    auto: true,
    watched: true,
    request: request(
      'run_terminal_command',
      { command: 'ls', is_background: false },
      { hookAsk: { hookName: GUARD, reason: null } },
    ),
    interaction: card('bash'),
    guardHook: GUARD,
    elsewhere: false,
    facts: { ...FACTS, ...facts, scan: { ...CLEAN, ...scan } },
    allowed: 0,
    ...over,
  });
}

/** An edit inside the folder that passes every row. */
function edit(over: Partial<AutoInput> = {}, facts: Partial<AutoFacts> = {}) {
  return decideAuto({
    auto: true,
    watched: true,
    request: request('write', { file_path: 'a.txt', content: 'x' }),
    interaction: card('edit'),
    guardHook: GUARD,
    elsewhere: false,
    facts: { ...FACTS, scan: null, landing: '/w/a.txt', ...facts },
    allowed: 0,
    ...over,
  });
}

function other(tool: string, rawInput: unknown, access: PermissionInteraction['access'] = 'other') {
  return decideAuto({
    auto: true,
    watched: true,
    request: request(tool, rawInput),
    interaction: card(access),
    guardHook: GUARD,
    elsewhere: false,
    facts: { ...FACTS, scan: null },
    allowed: 0,
  });
}

const ask = (why: string) => ({ allow: false, why });

describe('decideAuto', () => {
  it('off: every request is the plain card with no reason', () => {
    expect(bash({ auto: false })).toEqual({ allow: false, why: null });
    expect(edit({ auto: false })).toEqual({ allow: false, why: null });
  });

  it('rows 17 and 18: a plain command and an edit inside the folder are allowed', () => {
    expect(bash()).toEqual({ allow: true });
    expect(edit()).toEqual({ allow: true });
  });

  it('row 1: no window holds the host stream', () => {
    expect(bash({ watched: false })).toEqual(ask('no-window'));
    expect(edit({ watched: false })).toEqual(ask('no-window'));
  });

  it('row 2: allow-once is not offered, or not with its kind', () => {
    const none = request(
      'run_terminal_command',
      { command: 'ls' },
      {
        hookAsk: { hookName: GUARD, reason: null },
        options: [OPTIONS[2]!],
      },
    );
    expect(bash({ request: none })).toEqual(ask('not-covered'));
    const wrongKind = request(
      'run_terminal_command',
      { command: 'ls' },
      {
        hookAsk: { hookName: GUARD, reason: null },
        options: [{ optionId: 'allow-once', name: 'once', kind: 'allow_always' }, OPTIONS[2]!],
      },
    );
    expect(bash({ request: wrongKind })).toEqual(ask('not-covered'));
  });

  it('row 3: a shell request needs exactly this start’s guard; any other hook asks, of any kind', () => {
    const shell = (hookAsk?: PermissionRequest['hookAsk']) =>
      bash({ request: request('run_terminal_command', { command: 'ls' }, hookAsk === undefined ? {} : { hookAsk }) });
    expect(shell()).toEqual(ask('hook'));
    expect(shell({ hookName: OLDER_GUARD, reason: null })).toEqual(ask('hook'));
    expect(shell({ hookName: 'user/settings:pre_tool_use[0].hooks[0]', reason: null })).toEqual(ask('hook'));
    expect(shell({ hookName: GUARD, reason: 'blocked by policy' })).toEqual(ask('hook'));
    expect(bash({ guardHook: null })).toEqual(ask('hook'));
    expect(edit()).toEqual({ allow: true });
    const hooked = request('write', { file_path: 'a.txt' }, { hookAsk: { hookName: 'user/edit-hook', reason: null } });
    expect(edit({ request: hooked })).toEqual(ask('hook'));
    const guardedEdit = request('write', { file_path: 'a.txt' }, { hookAsk: { hookName: GUARD, reason: null } });
    expect(edit({ request: guardedEdit })).toEqual({ allow: true });
  });

  it('row 4: a helper that can work in another folder, and any workflow', () => {
    expect(other('spawn_subagent', { prompt: 'x', cwd: '/elsewhere' })).toEqual(ask('helper-start'));
    expect(other('spawn_subagent', { prompt: 'x', isolation: 'worktree' })).toEqual(ask('helper-start'));
    expect(other('spawn_subagent', 'not an object')).toEqual(ask('helper-start'));
    expect(other('workflow', { name: 'w' })).toEqual(ask('helper-start'));
  });

  it('row 5: tools other than commands and edits keep their card', () => {
    expect(other('monitor', { command: 'ls' })).toEqual(ask('not-covered'));
    expect(other('linear__save_issue', {}, 'mcp')).toEqual(ask('not-covered'));
    expect(other('use_tool', {}, 'mcp')).toEqual(ask('not-covered'));
    expect(other('spawn_subagent', { prompt: 'x' })).toEqual(ask('not-covered'));
    expect(other('scheduler_create', {})).toEqual(ask('not-covered'));
    expect(other('send_feedback', {})).toEqual(ask('not-covered'));
  });

  it('row 7: a helper may work in another folder in this session', () => {
    expect(bash({ elsewhere: true })).toEqual(ask('helper'));
    expect(edit({ elsewhere: true })).toEqual(ask('helper'));
  });

  it('row 8: a command the app cannot read in full', () => {
    expect(bash({ interaction: card('bash', { command: null }) })).toEqual(ask('unread'));
    expect(bash({ facts: { ...FACTS, scan: null } })).toEqual(ask('unread'));
    expect(bash({ interaction: card('bash', { flags: ['unchecked'] }) })).toEqual(ask('unread'));
    expect(bash({}, { unread: true })).toEqual(ask('unread'));
  });

  it.each<CommandFlag>([
    'elevated',
    'deletes',
    'outside-folder',
    'installs',
    'credentials',
    'prints-secrets',
    'provider-jobs',
  ])('row 9: the flag %s alone asks', (flag) => {
    expect(bash({ interaction: card('bash', { flags: [flag] }) })).toEqual(ask('risky'));
  });

  it('row 9: a word that names a key variable asks with its own reason, since no warning line shows', () => {
    expect(bash({}, { keyName: true })).toEqual(ask('key'));
    expect(bash({ interaction: card('bash', { flags: ['prints-secrets'] }) }, { keyName: true })).toEqual(ask('risky'));
  });

  it('row 10: a protected name in a word, a target or the edit landing', () => {
    expect(bash({}, { protectedName: true })).toEqual(ask('protected'));
    expect(bash({}, {}, { protectedPath: true })).toEqual(ask('protected'));
    expect(edit({}, { protectedPath: true })).toEqual(ask('protected'));
  });

  it('row 11: a head or option not on the list, and git where the folder holds no repository', () => {
    expect(bash({}, { unknown: true })).toEqual(ask('not-known'));
    expect(bash({}, { git: true }, { repository: false })).toEqual(ask('not-known'));
    expect(bash({}, { git: true }, { repository: true })).toEqual({ allow: true });
  });

  it('row 12: a background start, in the command or in the tool input', () => {
    expect(bash({}, { background: true })).toEqual(ask('background'));
    for (const input of [
      { command: 'ls', is_background: true },
      { command: 'ls', is_background: 'true' },
      { command: 'ls', is_background: 1 },
      { command: 'ls', block_until_ms: 0 },
      { command: 'ls', block_until_ms: '0' },
      { command: 'ls', block_until_ms: 'soon' },
    ]) {
      const req = request('run_terminal_command', input, { hookAsk: { hookName: GUARD, reason: null } });
      expect([input, bash({ request: req })]).toEqual([input, ask('background')]);
    }
    expect(backgroundInput({ command: 'ls', is_background: false, block_until_ms: 30_000 })).toBe(false);
    expect(backgroundInput({ command: 'ls', block_until_ms: '120000' })).toBe(false);
  });

  it('row 13: a write target that cannot be placed, lands outside on disk, or has two hard links', () => {
    expect(bash({}, { targets: null })).toEqual(ask('unknown-target'));
    expect(bash({}, { checks: null })).toEqual(ask('unknown-target'));
    expect(bash({}, {}, { placed: false })).toEqual(ask('unknown-target'));
    expect(bash({}, {}, { outside: true })).toEqual(ask('unknown-target'));
    expect(bash({}, {}, { links: true })).toEqual(ask('unknown-target'));
  });

  it('row 14: an edit whose landing is unknown or that has two hard links', () => {
    expect(edit({}, { landing: null })).toEqual(ask('unknown-landing'));
    expect(edit({}, { links: true })).toEqual(ask('unknown-landing'));
  });

  it('row 15: an edit that lands outside the folder', () => {
    expect(edit({}, { landingOutside: true })).toEqual(ask('risky'));
    expect(edit({ interaction: card('edit', { flags: ['outside-folder'] }) })).toEqual(ask('risky'));
  });

  it('an error inside the decision asks; outside Auto it is still the plain card', () => {
    const throwing = new Proxy({} as AutoFacts, {
      get() {
        throw new Error('broken facts');
      },
    });
    expect(bash({ facts: throwing })).toEqual(ask('unread'));
    expect(edit({ facts: throwing })).toEqual(ask('unread'));
    expect(bash({ auto: false, facts: throwing })).toEqual({ allow: false, why: null });
  });

  it('row 16: after the cap of calls in one answer, each call asks', () => {
    expect(bash({ allowed: AUTO_PER_TURN_MAX - 1 })).toEqual({ allow: true });
    expect(bash({ allowed: AUTO_PER_TURN_MAX })).toEqual(ask('many'));
    expect(edit({ allowed: AUTO_PER_TURN_MAX })).toEqual(ask('many'));
  });
});

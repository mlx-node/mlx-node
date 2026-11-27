/**
 * The slash command rule: the agent's own parse, every built-in of the agent in exactly one class, and the allow rule
 * per mode. The built-in list is compared with the fixture taken from the agent's source, so a built-in the agent
 * adds fails here until it is classed.
 */

import { readFileSync } from 'node:fs';

import { describe, expect, it } from 'vite-plus/test';

import { readCommandList, readCommandsUpdate } from '../src/chat/acp/messages.js';
import {
  AGENT_BUILTINS,
  ALLOWED_BUILTINS,
  allowedCommands,
  APP_COMMANDS,
  appCommandOf,
  BLOCKED,
  checkCommand,
  HIDDEN,
  parseSlash,
  readCommands,
  type AgentCommand,
} from '../src/chat/commands.js';
import { routeError } from '../src/chat/route-errors.js';
import { checkText, promptInput } from '../src/chat/route-input.js';
import { slashToken } from '../ui/src/lib/chat-input.js';
import { slashDraft, slashRows } from '../ui/src/pages/chat/slash.js';

interface FixtureCommand {
  name: string;
  description: string;
  hint: string | null;
  aliases: string[];
  gate: string;
}

const FIXTURE = JSON.parse(readFileSync(new URL('./fixtures/agent-builtins.json', import.meta.url), 'utf8')) as {
  commands: FixtureCommand[];
};

function refusedOf(text: string): string | undefined {
  try {
    checkText(text);
    return undefined;
  } catch (error) {
    return routeError('chat', error).reason;
  }
}

function row(name: string, kind: AgentCommand['kind'], extra: Partial<AgentCommand> = {}): AgentCommand {
  return { name, kind, description: `${name} words`, hint: null, qualifiedName: null, ...extra };
}

describe('parseSlash: the agent’s parse_slash_prefix', () => {
  it('reads the name up to the first white space, trims, lower-cases; null where the agent sends plain text', () => {
    const table: Array<[string, { name: string; args: string } | null]> = [
      ['/goal ship it', { name: 'goal', args: 'ship it' }],
      ['  /Goal\tship  it  ', { name: 'goal', args: 'ship  it' }],
      ['\n/goal\n\nmore', { name: 'goal', args: 'more' }],
      ['/A', { name: 'a', args: '' }],
      // NBSP, ideographic space and the line separator are Rust white space: trimmed, and they end the name.
      [' /goal x', { name: 'goal', args: 'x' }],
      ['/goal x', { name: 'goal', args: 'x' }],
      ['　/workflow go', { name: 'workflow', args: 'go' }],
      ['/goal x', { name: 'goal', args: 'x' }],
      // A zero-width space is not white space: it stays in the name.
      ['/comm​it', { name: 'comm​it', args: '' }],
      ['/acme:Login now', { name: 'acme:login', args: 'now' }],
      ['/', null],
      ['   /   ', null],
      ['/ goal', null],
      ['/a/b', null],
      ['/usr/bin is broken', null],
      ['/a\\b', null],
      // U+FEFF is not Rust white space: the text does not start with `/` for the agent.
      ['﻿/goal x', null],
      ['goal', null],
      ['say /goal', null],
      ['', null],
    ];
    for (const [text, parsed] of table) expect([text, parseSlash(text)]).toEqual([text, parsed]);
  });

  it('the page’s first word of a draft matches the same start', () => {
    expect(slashToken('/go')).toEqual({ start: 0, end: 3, typed: 'go' });
    expect(slashToken('  /goal ship')).toEqual({ start: 2, end: 7, typed: 'goal' });
    expect(slashToken('/')).toEqual({ start: 0, end: 1, typed: '' });
    expect(slashToken('hello /goal')).toBeNull();
    expect(slashToken('')).toBeNull();
  });
});

describe('the classes of the agent’s built-ins', () => {
  it('AGENT_BUILTINS is the agent’s list, names and aliases', () => {
    expect(AGENT_BUILTINS.map((b) => ({ name: b.name, aliases: [...b.aliases] }))).toEqual(
      FIXTURE.commands.map((c) => ({ name: c.name, aliases: c.aliases })),
    );
    expect(AGENT_BUILTINS).toHaveLength(19);
    expect(AGENT_BUILTINS.flatMap((b) => b.aliases)).toHaveLength(5);
  });

  it('every built-in name and alias is in exactly one of app / allowed / blocked / hidden', () => {
    const app = new Set(Object.entries(APP_COMMANDS).flatMap(([name, aliases]) => [name, ...aliases]));
    const allowed = new Set(ALLOWED_BUILTINS.work);
    for (const builtin of AGENT_BUILTINS) {
      for (const name of [builtin.name, ...builtin.aliases]) {
        const classes = [app.has(name), allowed.has(name), BLOCKED.has(name), HIDDEN.has(name)].filter(Boolean);
        expect([name, classes.length]).toEqual([name, 1]);
      }
    }
    // App commands the agent does not have stay out of the agent's classes.
    for (const name of app)
      expect([name, allowed.has(name) || BLOCKED.has(name) || HIDDEN.has(name)]).toEqual([name, false]);
    expect(ALLOWED_BUILTINS.chat).toEqual([]);
    expect([...allowed].sort()).toEqual(['deep-research', 'goal', 'loop', 'workflow']);
  });

  it('app commands by name and by the other names they are typed as', () => {
    expect(
      ['compact', 'context', 'usage', 'cost', 'fork', 'plan', 'recap', 'rewind', 'undo'].map(appCommandOf),
    ).toEqual(['compact', 'context', 'context', 'context', 'fork', 'plan', 'recap', 'rewind', 'rewind']);
    expect(appCommandOf('goal')).toBeNull();
  });
});

describe('the allow rule', () => {
  const listed = [
    row('goal', 'builtin'),
    row('commit', 'skill', { qualifiedName: 'user:commit' }),
    row('release', 'workflow'),
    // Rows named like refused built-ins or aliases: the agent never sends them bare, the rule refuses them anyway.
    row('yolo', 'skill'),
    row('status', 'skill'),
    row('compact', 'skill'),
    row('mystery', 'other'),
  ];

  it('refuses the blocked, hidden and app names in both modes, whatever the list says', () => {
    for (const mode of ['chat', 'work'] as const) {
      for (const text of [
        '/yolo',
        '/always-approve on',
        '/plugin add x',
        '/mem',
        '/status',
        '/info',
        '/compact',
        '/undo',
        '/plan',
        '/plan fix the build',
      ]) {
        expect([mode, text, checkCommand(mode, text, listed)]).toEqual([mode, text, 'refused']);
      }
    }
  });

  it('Chat sends no agent command; Work what the list holds, its four built-ins too', () => {
    expect(checkCommand('chat', '/goal ship it', listed)).toBe('refused');
    expect(checkCommand('chat', '/commit', listed)).toBe('refused');
    expect(checkCommand('chat', 'hello', null)).toBe('plain');
    for (const text of ['/goal ship it', '/loop 5m check', '/deep-research why', '/workflow runs', '/GOAL x']) {
      expect([text, checkCommand('work', text, null)]).toEqual([text, 'needs-list']);
    }
    // Mutation: pass an allowed built-in with no row → `/loop` reaches an agent whose session has no scheduler.
    expect(checkCommand('work', '/goal ship it', listed)).toBe('allowed');
    expect(checkCommand('work', '/GOAL x', listed)).toBe('allowed');
    for (const text of ['/loop 5m check', '/deep-research why', '/workflow runs']) {
      expect([text, checkCommand('work', text, listed)]).toEqual([text, 'refused']);
    }
    // Only the agent's own built-in row offers it: a skill of that name does not.
    expect(checkCommand('work', '/loop 5m', [row('loop', 'skill')])).toBe('refused');
    expect(checkCommand('work', '/loop 5m', [row('loop', 'builtin')])).toBe('allowed');
    expect(checkCommand('work', '/commit fix', listed)).toBe('allowed');
    expect(checkCommand('work', '/user:commit fix', listed)).toBe('allowed');
    expect(checkCommand('work', '/Release now', listed)).toBe('allowed');
    expect(checkCommand('work', '/mystery', listed)).toBe('refused');
    expect(checkCommand('work', '/clear', listed)).toBe('refused');
    expect(checkCommand('work', '/commit', null)).toBe('needs-list');
    expect(checkCommand('work', '/commit', [])).toBe('refused');
    expect(checkCommand('work', '/a/b', listed)).toBe('refused');
    expect(checkCommand('work', '﻿/commit', listed)).toBe('refused');
  });

  it('`/plan` is an app command: refused as prompt text even before the list came or when a skill is named plan', () => {
    // Mutation: drop `plan` from APP_COMMANDS → Work with no list says 'needs-list' and a `plan` skill row says 'allowed'.
    for (const mode of ['chat', 'work'] as const) {
      expect([mode, checkCommand(mode, '/plan', null)]).toEqual([mode, 'refused']);
      expect([mode, checkCommand(mode, '/plan fix the build', [row('plan', 'skill')])]).toEqual([mode, 'refused']);
    }
  });

  it('lists in Work the allowed built-ins, skills and workflows; nothing refused, nothing in Chat', () => {
    const rows = [
      ...listed,
      row('always-approve', 'builtin'),
      row('session-info', 'builtin'),
      row('loop', 'builtin', { hint: '[interval] <prompt>' }),
      row('two words', 'skill'),
      row('goal', 'skill'),
    ];
    expect(allowedCommands('work', rows)).toEqual([
      { name: 'goal', kind: 'builtin', description: 'goal words', hint: null },
      { name: 'commit', kind: 'skill', description: 'commit words', hint: null },
      { name: 'release', kind: 'workflow', description: 'release words', hint: null },
      { name: 'loop', kind: 'builtin', description: 'loop words', hint: '[interval] <prompt>' },
    ]);
    expect(allowedCommands('chat', rows)).toEqual([]);
  });

  it('the route refuses what no mode allows, and lets the rest through to the service', () => {
    for (const text of [
      '/yolo',
      ' /yolo',
      '\n/always-approve on',
      '﻿/plugins add x',
      '/compact',
      '/context',
      '/a/b',
      '/',
    ]) {
      expect([text, refusedOf(text)]).toEqual([text, 'slash-text']);
    }
    expect(refusedOf('/plan fix the build')).toBe('slash-text');
    for (const text of ['/goal ship it', '/commit', '/whatever', 'a word, then /yolo']) {
      expect([text, refusedOf(text)]).toEqual([text, undefined]);
    }
    expect(promptInput({ text: '/goal x' }).text).toBe('/goal x');
  });
});

describe('the agent’s rows', () => {
  it('classes a row by name, then by its _meta', () => {
    const rows = readCommandList({
      commands: [
        { name: 'goal', description: 'Set a goal', input: { hint: '<objective>' } },
        { name: 'deep-research', description: 'Research', input: null, _meta: { workflowSource: 'builtin' } },
        {
          name: 'Commit',
          description: 'Write a commit message',
          input: null,
          _meta: { scope: 'user', path: '/s/SKILL.md', bareName: 'Commit', qualifiedName: 'user:Commit' },
        },
        { name: 'Release', description: 'Workflow: cut a release', input: null, _meta: { workflowSource: 'user' } },
        { name: 'future', description: 'A built-in the app does not know', input: null },
        { name: 7, description: 'not a row' },
      ],
      tools: ['bash'],
    });
    expect(rows).not.toBeNull();
    expect(readCommands(rows!)).toEqual([
      { name: 'goal', kind: 'builtin', description: 'Set a goal', hint: '<objective>', qualifiedName: null },
      { name: 'deep-research', kind: 'builtin', description: 'Research', hint: null, qualifiedName: null },
      {
        name: 'commit',
        kind: 'skill',
        description: 'Write a commit message',
        hint: null,
        qualifiedName: 'user:commit',
      },
      { name: 'release', kind: 'workflow', description: 'Workflow: cut a release', hint: null, qualifiedName: null },
      {
        name: 'future',
        kind: 'other',
        description: 'A built-in the app does not know',
        hint: null,
        qualifiedName: null,
      },
    ]);
    expect(readCommandList({ result: { commands: [] } })).toBeNull();
    expect(readCommandsUpdate({ sessionUpdate: 'available_commands_update', availableCommands: [] })).toEqual([]);
    expect(readCommandsUpdate({ sessionUpdate: 'agent_message_chunk' })).toBeNull();
  });
});

describe('`/plan` on the page', () => {
  it('is offered in Work only; typed by hand in Chat it is an unknown command, like `/goal`', () => {
    // Mutation: drop `workOnly` from the plan row → Chat lists `/plan` and reads the typed one as an app command.
    expect(slashRows('work', null).rows.some((r) => r.name === 'plan')).toBe(true);
    expect(slashRows('chat', null).rows.some((r) => r.name === 'plan')).toBe(false);
    expect(slashDraft('/plan fix the build', 'work', null)).toEqual({ kind: 'app', op: 'plan', args: 'fix the build' });
    expect(slashDraft('/plan', 'chat', null)).toEqual({ kind: 'unknown' });
    expect(slashDraft('/goal x', 'chat', null)).toEqual({ kind: 'unknown' });
  });
});

/**
 * The permission history file, `ROOT/permission/<id>.jsonl`: each line kind is written with its fields in a fixed
 * order and read back field by field; a `withdrawn` line drops the allowed line it names; a bad line is skipped; a
 * link at the file's place is refused.
 */

import { createHash } from 'node:crypto';
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  truncateSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  changedSettings,
  claudeJsonPart,
  markOf,
  PermissionLog,
  readPermissionLine,
  readPermissionLines,
  settingsPrints,
  SETTINGS_KEYS,
  shownPath,
  storedServer,
  type PermissionLine,
  type SettingsPrints,
  type ShownLine,
} from '../src/chat/permission-log.js';
import type { PermissionMark } from '../src/chat/types.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import { deniedWord, mcpLine, placeMarks, rewindRule, subagentDeniedText } from '../ui/src/lib/permission-text.js';
import type { TranscriptEntry } from '../ui/src/lib/types.js';

const PRINT = 'a'.repeat(64);
const PRINTS = Object.fromEntries(SETTINGS_KEYS.map((k) => [k, PRINT])) as SettingsPrints;
const PRINTS_TEXT = JSON.stringify(PRINTS);
const SETTINGS: PermissionLine = { kind: 'settings', at: 2, prints: PRINTS, changed: ['home:.claude.json'] };
const SETTINGS_TEXT = `{"kind":"settings","at":2,"prints":${PRINTS_TEXT},"changed":["home:.claude.json"]}`;

let root = '';
beforeEach(() => {
  root = mkdtempSync(join(tmpdir(), 'mlx-mode-log-'));
});
afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

describe('the lines', () => {
  it('each kind is written with its fields in order and read back', async () => {
    const log = new PermissionLog(join(root, 'permission'));
    const lines: PermissionLine[] = [
      { kind: 'mode', at: 1, mode: 'bypass', from: 'ask', by: 'user', prints: PRINTS, userHooks: true },
      { kind: 'allowed', at: 2, mode: 'bypass', toolCallId: 'call_1', access: 'mcp' },
      { kind: 'allowed', at: 3, mode: 'bypass', toolCallId: 'call_2', access: 'bash' },
      { kind: 'withdrawn', at: 4, of: 3, toolCallId: 'call_2' },
      { kind: 'mode', at: 5, mode: 'auto', from: 'bypass', by: 'user', prints: PRINTS, editsUnasked: true },
      { kind: 'allowed', at: 6, mode: 'auto', toolCallId: 'call_3', access: 'edit' },
      { kind: 'settings', at: 7, prints: { ...PRINTS, 'folder:.mcp.json': 'missing' }, changed: ['folder:.mcp.json'] },
      { kind: 'mode', at: 8, mode: 'ask', from: 'auto', by: 'restart' },
    ];
    for (const line of lines) await log.append('conv-1', line);
    const text = readFileSync(join(root, 'permission', 'conv-1.jsonl'), 'utf8')
      .split('\n')
      .filter((l) => l !== '');
    expect(text.map((l) => Object.keys(JSON.parse(l) as object).join(','))).toEqual([
      'kind,at,mode,from,by,prints,userHooks',
      'kind,at,mode,toolCallId,access',
      'kind,at,mode,toolCallId,access',
      'kind,at,of,toolCallId',
      'kind,at,mode,from,by,prints,editsUnasked',
      'kind,at,mode,toolCallId,access',
      'kind,at,prints,changed',
      'kind,at,mode,from,by',
    ]);
    expect(text.map(readPermissionLine)).toEqual(lines);
    expect(await log.read('conv-1')).toEqual(lines.filter((l) => l.at !== 3 && l.at !== 4));
  });

  it('the temp key is written and read on an Auto mode line and an Auto allowed line, and reaches the mark', async () => {
    const log = new PermissionLog(join(root, 'permission'));
    const lines: PermissionLine[] = [
      { kind: 'mode', at: 1, mode: 'auto', from: 'ask', by: 'user', temp: true },
      { kind: 'allowed', at: 2, mode: 'auto', toolCallId: 'call_1', access: 'bash', temp: true },
      { kind: 'allowed', at: 3, mode: 'auto', toolCallId: 'call_2', access: 'edit' },
    ];
    for (const line of lines) await log.append('conv-t', line);
    const text = readFileSync(join(root, 'permission', 'conv-t.jsonl'), 'utf8')
      .split('\n')
      .filter((l) => l !== '');
    expect(text.map((l) => Object.keys(JSON.parse(l) as object).join(','))).toEqual([
      'kind,at,mode,from,by,temp',
      'kind,at,mode,toolCallId,access,temp',
      'kind,at,mode,toolCallId,access',
    ]);
    expect(text.map(readPermissionLine)).toEqual(lines);
    const place = { home: root, agentHome: root, cwd: root };
    expect(lines.map((l) => markOf(l as ShownLine, place))).toEqual([
      { kind: 'mode', at: 1, mode: 'auto', from: 'ask', by: 'user', temp: true },
      { kind: 'allowed', at: 2, mode: 'auto', toolCallId: 'call_1', access: 'bash', card: null, temp: true },
      { kind: 'allowed', at: 3, mode: 'auto', toolCallId: 'call_2', access: 'edit', card: null },
    ]);
  });

  it("the child key is written last on a helper's allowed line, read back and kept on the mark; a line without it still reads", async () => {
    const log = new PermissionLog(join(root, 'permission'));
    const child = '01a20d98-cc9b-7833-95a9-46637f8506d6';
    const lines: PermissionLine[] = [
      { kind: 'allowed', at: 1, mode: 'bypass', toolCallId: 'call_1', access: 'bash', child },
      { kind: 'allowed', at: 2, mode: 'auto', toolCallId: 'call_2', access: 'edit', temp: true, child },
      { kind: 'allowed', at: 3, mode: 'bypass', toolCallId: 'call_3', access: 'bash' },
      { kind: 'allowed', at: 4, mode: 'bypass', toolCallId: 'call_4', access: 'bash', child: 'not a safe id' },
    ];
    for (const line of lines) await log.append('conv-c', line);
    const text = readFileSync(join(root, 'permission', 'conv-c.jsonl'), 'utf8')
      .split('\n')
      .filter((l) => l !== '');
    expect(text.map((l) => Object.keys(JSON.parse(l) as object).join(','))).toEqual([
      'kind,at,mode,toolCallId,access,child',
      'kind,at,mode,toolCallId,access,temp,child',
      'kind,at,mode,toolCallId,access',
      'kind,at,mode,toolCallId,access',
    ]);
    const unsafe: PermissionLine = { kind: 'allowed', at: 4, mode: 'bypass', toolCallId: 'call_4', access: 'bash' };
    expect(text.map(readPermissionLine)).toEqual([...lines.slice(0, 3), unsafe]);
    expect(readPermissionLine('{"kind":"allowed","at":5,"mode":"bypass","toolCallId":"c","access":"bash"}')).toEqual({
      kind: 'allowed',
      at: 5,
      mode: 'bypass',
      toolCallId: 'c',
      access: 'bash',
    });
    const place = { home: root, agentHome: root, cwd: root };
    expect(markOf(lines[0] as ShownLine, place)).toEqual({
      kind: 'allowed',
      at: 1,
      mode: 'bypass',
      toolCallId: 'call_1',
      access: 'bash',
      card: null,
      child,
    });
    expect(markOf(lines[2] as ShownLine, place)).not.toHaveProperty('child');
  });

  it('withdrawn drops the last earlier allowed line with its id and at === of; another of drops nothing', () => {
    const allowed = (at: number, id: string): string =>
      JSON.stringify({ kind: 'allowed', at, mode: 'bypass', toolCallId: id, access: 'bash' });
    const withdrawn = (at: number, of: number, id: string): string =>
      JSON.stringify({ kind: 'withdrawn', at, of, toolCallId: id });
    const read = (lines: string[]): number[] => readPermissionLines(lines.join('\n')).map((l) => l.at);
    expect(read([allowed(1, 'a'), allowed(2, 'a'), withdrawn(3, 2, 'a')])).toEqual([1]);
    expect(read([allowed(2, 'a'), allowed(2, 'a'), withdrawn(3, 2, 'a')])).toEqual([2]);
    expect(read([allowed(1, 'a'), withdrawn(3, 9, 'a')])).toEqual([1]);
    expect(read([allowed(1, 'a'), withdrawn(3, 1, 'b')])).toEqual([1]);
    expect(read([withdrawn(3, 1, 'a'), allowed(1, 'a')])).toEqual([1]);
    expect(read([withdrawn(3, 1, 'a')])).toEqual([]);
  });

  it.each<[string, string]>([
    ['an unknown mode', '{"kind":"mode","at":1,"mode":"yolo","from":"ask","by":"user"}'],
    ['the old switch kind', '{"kind":"switch","at":1,"on":true,"by":"user"}'],
    ['a missing from', '{"kind":"mode","at":1,"mode":"auto","by":"user"}'],
    [
      'a bad print',
      `{"kind":"mode","at":1,"mode":"auto","from":"ask","by":"user","prints":{"home:.claude.json":"xyz"}}`,
    ],
    [
      'a missing key in the prints',
      `{"kind":"settings","at":1,"prints":${JSON.stringify({ ...PRINTS, 'folder:.mcp.json': undefined })},"changed":["home:.claude.json"]}`,
    ],
    [
      'an unknown key in the prints',
      `{"kind":"settings","at":1,"prints":${JSON.stringify({ ...PRINTS, 'home:.bashrc': PRINT })},"changed":["home:.claude.json"]}`,
    ],
    ['no changed file', `{"kind":"settings","at":1,"prints":${PRINTS_TEXT},"changed":[]}`],
    ['an unknown changed file', `{"kind":"settings","at":1,"prints":${PRINTS_TEXT},"changed":["/etc/passwd"]}`],
    [
      'changed files out of order',
      `{"kind":"settings","at":1,"prints":${PRINTS_TEXT},"changed":["folder:.mcp.json","home:.claude.json"]}`,
    ],
    [
      'the old mode line with one print',
      `{"kind":"mode","at":1,"mode":"auto","from":"ask","by":"user","print":"${PRINT}"}`,
    ],
    ['the old settings line with one print', `{"kind":"settings","at":1,"print":"${PRINT}"}`],
    ['extra text', '{"kind":"mode","at":1,"mode":"auto","from":"ask","by":"user"} x'],
    ['an extra key', '{"kind":"mode","at":1,"mode":"auto","from":"ask","by":"user","on":true}'],
    ['an allowed line with no mode', '{"kind":"allowed","at":1,"toolCallId":"c","access":"bash"}'],
    ['an Auto line for a tool', '{"kind":"allowed","at":1,"mode":"auto","toolCallId":"c","access":"mcp"}'],
    ['a bad by', '{"kind":"mode","at":1,"mode":"auto","from":"ask","by":"model"}'],
    ['a negative time', `{"kind":"settings","at":-1,"prints":${PRINTS_TEXT},"changed":["home:.claude.json"]}`],
    ['temp on a mode line to Ask', '{"kind":"mode","at":1,"mode":"ask","from":"auto","by":"user","temp":true}'],
    ['temp on a Bypass line', '{"kind":"allowed","at":1,"mode":"bypass","toolCallId":"c","access":"bash","temp":true}'],
    ['temp that is not true', '{"kind":"allowed","at":1,"mode":"auto","toolCallId":"c","access":"bash","temp":false}'],
    ['temp on a withdrawn line', '{"kind":"withdrawn","at":1,"of":1,"toolCallId":"c","temp":true}'],
    [
      'a child that is not text',
      '{"kind":"allowed","at":1,"mode":"bypass","toolCallId":"c","access":"bash","child":7}',
    ],
    [
      'a child that is not a safe id',
      '{"kind":"allowed","at":1,"mode":"bypass","toolCallId":"c","access":"bash","child":"../x"}',
    ],
    ['a child on a withdrawn line', '{"kind":"withdrawn","at":1,"of":1,"toolCallId":"c","child":"s1"}'],
    [
      'temp on a settings line',
      `{"kind":"settings","at":1,"prints":${PRINTS_TEXT},"changed":["home:.claude.json"],"temp":true}`,
    ],
    ['not JSON', 'mode auto'],
  ])('%s is skipped', (_name, text) => {
    expect(readPermissionLine(text)).toBeNull();
    expect(readPermissionLines(`${text}\n${SETTINGS_TEXT}`)).toEqual([SETTINGS]);
  });

  it('a link at the file place is refused; a read through it gives nothing', async () => {
    const dir = join(root, 'permission');
    mkdirSync(dir, { recursive: true });
    const target = join(root, 'elsewhere.jsonl');
    writeFileSync(target, `${SETTINGS_TEXT}\n`);
    symlinkSync(target, join(dir, 'conv-1.jsonl'));
    const log = new PermissionLog(dir);
    await expect(log.append('conv-1', SETTINGS)).rejects.toThrow();
    expect(await log.read('conv-1')).toEqual([]);
    expect(
      readFileSync(target, 'utf8')
        .split('\n')
        .filter((l) => l !== ''),
    ).toHaveLength(1);
  });

  it('a line queued before remove is written then removed; a line after remove is refused and makes no file', async () => {
    const dir = join(root, 'permission');
    const log = new PermissionLog(dir);
    const file = join(dir, 'conv-1.jsonl');
    const before = log.append('conv-1', SETTINGS);
    const removed = log.remove('conv-1');
    const after = log.append('conv-1', { kind: 'mode', at: 2, mode: 'ask', from: 'ask', by: 'user' });
    await before;
    await removed;
    await expect(after).rejects.toThrow();
    expect(existsSync(file)).toBe(false);
    await log.append('conv-2', SETTINGS);
    expect(existsSync(join(dir, 'conv-2.jsonl'))).toBe(true);
  });
});

describe('the settings prints', () => {
  it('~/.claude.json: only the top MCP servers and the MCP keys of the folders count, keys in any order', () => {
    const folders = ['/w/app', '/w'];
    const base = claudeJsonPart(
      JSON.stringify({
        numStartups: 1,
        mcpServers: { a: { command: 'a', args: ['1'] } },
        projects: { '/w/app': { mcpServers: {}, history: [] }, '/other': { mcpServers: { b: {} } } },
      }),
      folders,
    );
    expect(base).not.toBeNull();
    const same = claudeJsonPart(
      JSON.stringify({
        projects: { '/other': { mcpServers: { c: {} } }, '/w/app': { history: ['x'], mcpServers: {} } },
        mcpServers: { a: { args: ['1'], command: 'a' } },
        numStartups: 2,
        pluginUsage: { p: 1 },
      }),
      folders,
    );
    expect(same).toBe(base);
    const rootEntry = claudeJsonPart(
      JSON.stringify({
        mcpServers: { a: { command: 'a', args: ['1'] } },
        projects: { '/w/app': { mcpServers: {} }, '/w': { disabledMcpjsonServers: ['d'] } },
      }),
      folders,
    );
    expect(rootEntry).not.toBe(base);
    expect(claudeJsonPart('{"mcpServers":', folders)).toBeNull();
    expect(claudeJsonPart('[1]', folders)).toBe(claudeJsonPart('{}', folders));
  });

  it('a home path is shown with ~; a path beside the home is not', () => {
    const place = { home: '/Users/bob', agentHome: '/Users/bob/.mlx-agent', cwd: '/Users/bob/app' };
    expect(shownPath(place, '/Users/bob')).toBe('~');
    expect(shownPath(place, '/Users/bob/.claude.json')).toBe('~/.claude.json');
    expect(shownPath(place, '/Users/bob2/.claude.json')).toBe('/Users/bob2/.claude.json');
    expect(shownPath(place, '/etc/grok/managed_config.toml')).toBe('/etc/grok/managed_config.toml');
  });

  it('one print per file: missing, unreadable, content; a file over 16 MB counts by size and time only', async () => {
    const home = join(root, 'home');
    const cwd = join(home, 'app');
    mkdirSync(join(cwd, '.claude'), { recursive: true });
    const place = { home, agentHome: join(root, 'agent'), cwd };
    const first = await settingsPrints(place);
    expect(Object.keys(first)).toEqual([...SETTINGS_KEYS]);
    expect(first['folder:.mcp.json']).toBe('missing');
    writeFileSync(join(cwd, '.mcp.json'), '{}');
    mkdirSync(join(home, '.claude.json'));
    const second = await settingsPrints(place);
    expect(second['folder:.mcp.json']).toBe(createHash('sha256').update('{}').digest('hex'));
    expect(second['home:.claude.json']).toBe('unreadable');
    expect(changedSettings(first, second)).toEqual(['home:.claude.json', 'folder:.mcp.json']);

    const big = join(cwd, '.claude', 'settings.json');
    writeFileSync(big, '');
    truncateSync(big, 16 * 1024 * 1024 + 1);
    utimesSync(big, 1_000, 1_000);
    const large = await settingsPrints(place);
    writeFileSync(big, 'x', { flag: 'r+' });
    utimesSync(big, 1_000, 1_000);
    expect((await settingsPrints(place))['folder:.claude/settings.json']).toBe(large['folder:.claude/settings.json']);
    utimesSync(big, 2_000, 2_000);
    expect(changedSettings(large, await settingsPrints(place))).toEqual(['folder:.claude/settings.json']);
  });

  // Mutation: the key left at `folder:.grok/config.toml` → the folder's agent config is never printed.
  it("the folder's agent config is its `.mlx-agent/config.toml`", async () => {
    const home = join(root, 'agent-config-home');
    const cwd = join(home, 'app');
    mkdirSync(join(cwd, '.mlx-agent'), { recursive: true });
    mkdirSync(join(cwd, '.grok'), { recursive: true });
    writeFileSync(join(cwd, '.grok', 'config.toml'), 'old = 1\n');
    const place = { home, agentHome: join(root, 'agent'), cwd };
    expect((await settingsPrints(place))['folder:.mlx-agent/config.toml']).toBe('missing');
    writeFileSync(join(cwd, '.mlx-agent', 'config.toml'), '[plugins]\n');
    expect((await settingsPrints(place))['folder:.mlx-agent/config.toml']).toBe(
      createHash('sha256').update('[plugins]\n').digest('hex'),
    );
  });
});

describe('denied lines: a call denied with a note', () => {
  it('is written with its fields in order, read back, and becomes a mark with no card; never with the note', async () => {
    const log = new PermissionLog(join(root, 'permission'));
    const child = '01a20d98-cc9b-7833-95a9-46637f8506d6';
    const lines: PermissionLine[] = [
      { kind: 'denied', at: 1, toolCallId: 'call_1', access: 'bash', by: 'note' },
      { kind: 'denied', at: 2, toolCallId: 'call_2', access: 'edit', by: 'note', child },
    ];
    for (const line of lines) await log.append('conv-d', line);
    const text = readFileSync(join(root, 'permission', 'conv-d.jsonl'), 'utf8')
      .split('\n')
      .filter((l) => l !== '');
    expect(text.map((l) => Object.keys(JSON.parse(l) as object).join(','))).toEqual([
      'kind,at,toolCallId,access,by',
      'kind,at,toolCallId,access,by,child',
    ]);
    expect(await log.read('conv-d')).toEqual(lines);
    const place = { home: root, agentHome: root, cwd: root };
    expect(lines.map((l) => markOf(l as ShownLine, place))).toEqual([
      { kind: 'denied', at: 1, toolCallId: 'call_1', access: 'bash', by: 'note', card: null },
      { kind: 'denied', at: 2, toolCallId: 'call_2', access: 'edit', by: 'note', card: null, child },
    ]);
  });

  it.each<[string, string]>([
    ['an unknown by', '{"kind":"denied","at":1,"toolCallId":"c","access":"bash","by":"rule"}'],
    ['no by', '{"kind":"denied","at":1,"toolCallId":"c","access":"bash"}'],
    ['the note text', '{"kind":"denied","at":1,"toolCallId":"c","access":"bash","by":"note","note":"use pnpm"}'],
    ['an unknown access', '{"kind":"denied","at":1,"toolCallId":"c","access":"shell","by":"note"}'],
    [
      'a child that is not a safe id',
      '{"kind":"denied","at":1,"toolCallId":"c","access":"bash","by":"note","child":"../x"}',
    ],
  ])('%s is skipped', (_name, text) => {
    expect(readPermissionLine(text)).toBeNull();
  });
});

describe('denied lines: a call a never-allow rule denied', () => {
  const RULE = `n_${'ab'.repeat(16)}`;

  it('holds the rule id, never its words; the mark reads the words the rule file holds now', async () => {
    const log = new PermissionLog(join(root, 'permission'));
    const child = '01a20d98-cc9b-7833-95a9-46637f8506d6';
    const lines: PermissionLine[] = [
      { kind: 'denied', at: 1, toolCallId: 'call_1', access: 'bash', by: 'rule', ruleId: RULE },
      { kind: 'denied', at: 2, toolCallId: 'call_2', access: 'bash', by: 'rule', ruleId: RULE, child },
    ];
    for (const line of lines) await log.append('conv-r', line);
    const text = readFileSync(join(root, 'permission', 'conv-r.jsonl'), 'utf8')
      .split('\n')
      .filter((l) => l !== '');
    expect(text.map((l) => Object.keys(JSON.parse(l) as object).join(','))).toEqual([
      'kind,at,toolCallId,access,by,ruleId',
      'kind,at,toolCallId,access,by,ruleId,child',
    ]);
    expect(await log.read('conv-r')).toEqual(lines);
    const place = { home: root, agentHome: root, cwd: root };
    const kept = (id: string): string | null => (id === RULE ? 'git push' : null);
    expect(lines.map((l) => markOf(l as ShownLine, place, kept))).toEqual([
      {
        kind: 'denied',
        at: 1,
        toolCallId: 'call_1',
        access: 'bash',
        by: 'rule',
        ruleId: RULE,
        words: 'git push',
        card: null,
      },
      {
        kind: 'denied',
        at: 2,
        toolCallId: 'call_2',
        access: 'bash',
        by: 'rule',
        ruleId: RULE,
        words: 'git push',
        card: null,
        child,
      },
    ]);
    expect(markOf(lines[0] as ShownLine, place)).toMatchObject({ ruleId: RULE, words: null });
  });

  it.each<[string, string]>([
    ['a rule with no id', '{"kind":"denied","at":1,"toolCallId":"c","access":"bash","by":"rule"}'],
    [
      'a rule id that is not a safe id',
      '{"kind":"denied","at":1,"toolCallId":"c","access":"bash","by":"rule","ruleId":"../x"}',
    ],
    [
      'a note with a rule id',
      `{"kind":"denied","at":1,"toolCallId":"c","access":"bash","by":"note","ruleId":"${RULE}"}`,
    ],
    [
      'the rule words',
      `{"kind":"denied","at":1,"toolCallId":"c","access":"bash","by":"rule","ruleId":"${RULE}","words":"git push"}`,
    ],
  ])('%s is skipped', (_name, text) => {
    expect(readPermissionLine(text)).toBeNull();
  });

  it('the row word: the note, the rule, or a rule that is gone', () => {
    const base = { kind: 'denied', at: 1, toolCallId: 'c', access: 'bash', card: null } as const;
    expect(deniedWord({ ...base, by: 'note' })).toBe(CHAT.permission.deniedNote);
    expect(deniedWord({ ...base, by: 'rule', ruleId: RULE, words: 'git push' })).toBe(
      fill(CHAT.never.row, { words: 'git push' }),
    );
    expect(deniedWord({ ...base, by: 'rule', ruleId: RULE, words: null })).toBe(CHAT.never.rowGone);
    expect(deniedWord({ ...base, by: 'rule', words: null })).toBe(CHAT.never.rowGone);
  });

  it('a rule mark goes on its tool row like a note mark', () => {
    const entries: TranscriptEntry[] = [
      { kind: 'user', ts: 10, text: 'go', promptIndex: 0 },
      { kind: 'tool', ts: 30, id: 'c1', name: 'run_terminal_command', title: 'git push', status: 'failed', endTs: 35 },
    ];
    const mark: PermissionMark = {
      kind: 'denied',
      at: 31,
      toolCallId: 'c1',
      access: 'bash',
      by: 'rule',
      ruleId: RULE,
      words: 'git push',
      card: null,
    };
    expect(placeMarks(entries, [mark]).denied).toEqual(new Map([['c1', mark]]));
  });
});

describe('placeMarks: denied marks', () => {
  const tool = (id: string, ts: number): TranscriptEntry => ({
    kind: 'tool',
    ts,
    id,
    name: 'run_terminal_command',
    title: 'npm install left-pad',
    status: 'failed',
    input: '{"command":"npm install left-pad"}',
    endTs: ts + 5,
  });
  const denied = (at: number, toolCallId: string, child?: string): PermissionMark => ({
    kind: 'denied',
    at,
    toolCallId,
    access: 'bash',
    by: 'note',
    card: null,
    ...(child === undefined ? {} : { child }),
  });

  // Mutation: place no denied mark by its row → it becomes a loose line.
  it("marks its own tool row, a helper's row, and is a line when neither is in the thread", () => {
    const spawn: TranscriptEntry = {
      kind: 'subagent',
      ts: 20,
      childId: 's1',
      type: 'general-purpose',
      description: 'd',
    };
    const entries: TranscriptEntry[] = [
      { kind: 'user', ts: 10, text: 'go', promptIndex: 0 },
      spawn,
      tool('c1', 30),
      { kind: 'assistant', ts: 50, text: 'ok' },
    ];
    const own = denied(31, 'c1');
    const helper = denied(25, 'h1', 's1');
    const same = denied(35, 'c1', 's1');
    const loose = denied(40, 'h2', 'gone');
    const placed = placeMarks(entries, [own, helper, same, loose]);
    expect(placed.denied).toEqual(new Map([['c1', own]]));
    expect(placed.deniedBySubagent).toEqual(new Map([['s1', [helper, same]]]));
    expect(placed.before.get(3)).toEqual([loose]);
    expect(placed.allowed.size).toBe(0);
    expect(placed.bySubagent.size).toBe(0);
  });

  it('the fact of a helper row counts its denied calls', () => {
    expect(subagentDeniedText(1)).toBe(CHAT.permission.subagentDeniedOne);
    expect(subagentDeniedText(3)).toBe(fill(CHAT.permission.subagentDenied, { n: 3 }));
  });
});

describe('rewind lines: the rule a rewind leaves', () => {
  // Mutation: write the rewind line with a `path` → the key order check fails.
  it('is written with its fields in order, read back, and becomes a mark; never with a path or a text', async () => {
    const log = new PermissionLog(join(root, 'permission'));
    const lines: PermissionLine[] = [
      { kind: 'allowed', at: 5, mode: 'bypass', toolCallId: 'call_1', access: 'bash' },
      { kind: 'rewind', at: 9, mode: 'conversation', files: 0, from: 4 },
      { kind: 'rewind', at: 12, mode: 'files', files: 3 },
    ];
    for (const line of lines) await log.append('conv-r', line);
    const text = readFileSync(join(root, 'permission', 'conv-r.jsonl'), 'utf8')
      .split('\n')
      .filter((l) => l !== '');
    expect(text.slice(1).map((l) => Object.keys(JSON.parse(l) as object).join(','))).toEqual([
      'kind,at,mode,files,from',
      'kind,at,mode,files',
    ]);
    expect(await log.read('conv-r')).toEqual(lines);
    const place = { home: root, agentHome: root, cwd: root };
    expect(lines.slice(1).map((l) => markOf(l as ShownLine, place))).toEqual([
      { kind: 'rewind', at: 9, mode: 'conversation', files: 0, from: 4 },
      { kind: 'rewind', at: 12, mode: 'files', files: 3 },
    ]);
  });

  it.each<[string, string]>([
    ['an unknown mode', '{"kind":"rewind","at":1,"mode":"conversation_only","files":0}'],
    ['no files', '{"kind":"rewind","at":1,"mode":"conversation"}'],
    ['a negative count', '{"kind":"rewind","at":1,"mode":"all","files":-1}'],
    ['a count that is not whole', '{"kind":"rewind","at":1,"mode":"all","files":1.5}'],
    ['a from that is not a time', '{"kind":"rewind","at":1,"mode":"all","files":0,"from":"x"}'],
    ['a path', '{"kind":"rewind","at":1,"mode":"files","files":1,"path":"/etc/hosts"}'],
  ])('%s is skipped', (_name, text) => {
    expect(readPermissionLine(text)).toBeNull();
  });
});

describe('placeMarks: rewind rules', () => {
  const tool = (id: string, ts: number): TranscriptEntry => ({
    kind: 'tool',
    ts,
    id,
    name: 'run_terminal_command',
    title: 'ls',
    status: 'completed',
    endTs: ts + 1,
  });

  // Mutation: drop the span check → the rewound allowed mark comes back as a loose line.
  it('the rule is a line where the rewind happened; an allowed or denied call the rewind removed is not drawn', () => {
    const entries: TranscriptEntry[] = [
      { kind: 'user', ts: 10, text: 'go', promptIndex: 0 },
      tool('c1', 12),
      { kind: 'assistant', ts: 14, text: 'ok' },
      { kind: 'user', ts: 40, text: 'again', promptIndex: 1 },
    ];
    const kept: PermissionMark = {
      kind: 'allowed',
      at: 13,
      mode: 'bypass',
      toolCallId: 'c1',
      access: 'bash',
      card: null,
    };
    const gone: PermissionMark = {
      kind: 'allowed',
      at: 22,
      mode: 'bypass',
      toolCallId: 'c9',
      access: 'bash',
      card: null,
    };
    const goneDenied: PermissionMark = {
      kind: 'denied',
      at: 24,
      toolCallId: 'c8',
      access: 'edit',
      by: 'note',
      card: null,
    };
    const after: PermissionMark = {
      kind: 'allowed',
      at: 35,
      mode: 'bypass',
      toolCallId: 'c7',
      access: 'bash',
      card: null,
    };
    const rule: PermissionMark = { kind: 'rewind', at: 30, mode: 'conversation', files: 0, from: 20 };
    const files: PermissionMark = { kind: 'rewind', at: 50, mode: 'files', files: 2 };
    const placed = placeMarks(entries, [kept, gone, goneDenied, rule, after, files]);
    expect(placed.allowed).toEqual(new Map([['c1', kept]]));
    expect(placed.before.get(3)).toEqual([rule, after]);
    expect(placed.end).toEqual([files]);
    expect(placed.denied.size).toBe(0);
  });

  it('the rule words follow the mode and the count of files', () => {
    expect(rewindRule({ kind: 'rewind', at: 1, mode: 'conversation', files: 0 })).toBe(CHAT.history.rule);
    expect(rewindRule({ kind: 'rewind', at: 1, mode: 'all', files: 2 })).toBe(
      fill(CHAT.history.ruleFiles, { files: fill(CHAT.history.nFiles, { n: 2 }) }),
    );
    expect(rewindRule({ kind: 'rewind', at: 1, mode: 'all', files: 0 })).toBe(CHAT.history.rule);
    expect(rewindRule({ kind: 'rewind', at: 1, mode: 'files', files: 1 })).toBe(
      fill(CHAT.history.ruleFilesOnly, { files: CHAT.history.oneFile }),
    );
    expect(rewindRule({ kind: 'rewind', at: 1, mode: 'files', files: 3 })).toBe(
      fill(CHAT.history.ruleFilesOnly, { files: fill(CHAT.history.nFiles, { n: 3 }) }),
    );
  });
});

describe('MCP lines: a server asked for input', () => {
  it('is written with its fields in order, read back, and becomes the same mark; never a typed value', async () => {
    const log = new PermissionLog(join(root, 'permission'));
    const lines: PermissionLine[] = [
      { kind: 'mcp', at: 1, server: 'github', mode: 'form', outcome: 'accept' },
      { kind: 'mcp', at: 2, server: 'github', mode: 'form', outcome: 'decline' },
      { kind: 'mcp', at: 3, server: 'linear', mode: 'url', outcome: 'cancel' },
      { kind: 'mcp', at: 4, server: 'linear', mode: 'form', outcome: 'expired' },
      { kind: 'mcp', at: 5, server: 'github', mode: 'url', outcome: 'accept' },
      { kind: 'mcp', at: 6, server: 'github', mode: 'url', outcome: 'completed' },
    ];
    for (const line of lines) await log.append('conv-m', line);
    const text = readFileSync(join(root, 'permission', 'conv-m.jsonl'), 'utf8')
      .split('\n')
      .filter((l) => l !== '');
    expect(text.map((l) => Object.keys(JSON.parse(l) as object).join(','))).toEqual(
      lines.map(() => 'kind,at,server,mode,outcome'),
    );
    expect(await log.read('conv-m')).toEqual(lines);
    const place = { home: root, agentHome: root, cwd: root };
    expect(lines.map((l) => markOf(l as ShownLine, place))).toEqual(lines);
  });

  it('keeps the server name printable and at most 64 characters', async () => {
    expect(storedServer('git\u202Ehub\nx')).toBe('git\\u{202E}hub x');
    expect(storedServer('s'.repeat(70))).toBe('s'.repeat(64));
    const log = new PermissionLog(join(root, 'permission'));
    await log.append('conv-s', {
      kind: 'mcp',
      at: 1,
      server: `bad\u0007${'n'.repeat(80)}`,
      mode: 'form',
      outcome: 'accept',
    });
    const [line] = await log.read('conv-s');
    expect(line).toEqual({
      kind: 'mcp',
      at: 1,
      server: storedServer(`bad\u0007${'n'.repeat(80)}`),
      mode: 'form',
      outcome: 'accept',
    });
  });

  it.each<[string, string]>([
    ['an unknown outcome', '{"kind":"mcp","at":1,"server":"github","mode":"url","outcome":"opened"}'],
    [
      'a form that finished in the browser',
      '{"kind":"mcp","at":1,"server":"github","mode":"form","outcome":"completed"}',
    ],
    ['an unknown mode', '{"kind":"mcp","at":1,"server":"github","mode":"link","outcome":"accept"}'],
    ['no server', '{"kind":"mcp","at":1,"mode":"form","outcome":"accept"}'],
    [
      'a server over 64 characters',
      `{"kind":"mcp","at":1,"server":"${'s'.repeat(65)}","mode":"form","outcome":"accept"}`,
    ],
    [
      'a control character in the server',
      '{"kind":"mcp","at":1,"server":"a\\u0007b","mode":"form","outcome":"accept"}',
    ],
    [
      'a typed value',
      '{"kind":"mcp","at":1,"server":"github","mode":"form","outcome":"accept","content":{"title":"x"}}',
    ],
  ])('%s is skipped', (_name, text) => {
    expect(readPermissionLine(text)).toBeNull();
  });

  it('the line words: the server, its mode, how it ended', () => {
    const mark = (mode: 'form' | 'url', outcome: 'accept' | 'decline' | 'cancel' | 'expired') =>
      mcpLine({ kind: 'mcp', at: 1, server: 'github', mode, outcome });
    expect(mark('form', 'accept')).toEqual({
      label: fill(CHAT.mcp.line.form, { server: 'github' }),
      outcome: CHAT.mcp.line.accept,
    });
    expect(mark('form', 'decline').outcome).toBe(CHAT.mcp.line.decline);
    expect(mark('form', 'cancel').outcome).toBe(CHAT.mcp.line.cancel);
    expect(mark('form', 'expired').outcome).toBe(CHAT.mcp.line.expired);
    expect(mark('url', 'accept')).toEqual({
      label: fill(CHAT.mcp.line.link, { server: 'github' }),
      outcome: CHAT.mcp.line.opened,
    });
    expect(mark('url', 'decline').outcome).toBe(CHAT.mcp.line.decline);
    expect(mcpLine({ kind: 'mcp', at: 1, server: 'github', mode: 'url', outcome: 'completed' })).toEqual({
      label: 'github',
      outcome: CHAT.mcp.line.completed,
    });
  });
});

describe('placeMarks: MCP lines', () => {
  // Mutation: drop 'mcp' from the span check → the rewound line comes back.
  it('is a line placed by its time; one inside a rewound span is not drawn', () => {
    const entries: TranscriptEntry[] = [
      { kind: 'user', ts: 10, text: 'go', promptIndex: 0 },
      { kind: 'assistant', ts: 14, text: 'ok' },
      { kind: 'user', ts: 40, text: 'again', promptIndex: 1 },
    ];
    const kept: PermissionMark = { kind: 'mcp', at: 12, server: 'github', mode: 'form', outcome: 'accept' };
    const gone: PermissionMark = { kind: 'mcp', at: 22, server: 'github', mode: 'form', outcome: 'decline' };
    const late: PermissionMark = { kind: 'mcp', at: 45, server: 'github', mode: 'form', outcome: 'expired' };
    const rule: PermissionMark = { kind: 'rewind', at: 30, mode: 'conversation', files: 0, from: 20 };
    const placed = placeMarks(entries, [kept, gone, rule, late]);
    expect(placed.before.get(1)).toEqual([kept]);
    expect(placed.before.get(2)).toEqual([rule]);
    expect(placed.end).toEqual([late]);
  });
});

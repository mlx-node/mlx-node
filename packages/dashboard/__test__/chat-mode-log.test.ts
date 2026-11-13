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
  type PermissionLine,
  type SettingsPrints,
  type ShownLine,
} from '../src/chat/permission-log.js';

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
});

/**
 * The permission history file, `ROOT/permission/<id>.jsonl`: each line kind is written with its fields in a fixed
 * order and read back field by field; a `withdrawn` line drops the allowed line it names; a bad line is skipped; a
 * link at the file's place is refused.
 */

import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import {
  PermissionLog,
  readPermissionLine,
  readPermissionLines,
  type PermissionLine,
} from '../src/chat/permission-log.js';

const PRINT = 'a'.repeat(64);

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
      { kind: 'mode', at: 1, mode: 'bypass', from: 'ask', by: 'user', print: PRINT, userHooks: true },
      { kind: 'allowed', at: 2, mode: 'bypass', toolCallId: 'call_1', access: 'mcp' },
      { kind: 'allowed', at: 3, mode: 'bypass', toolCallId: 'call_2', access: 'bash' },
      { kind: 'withdrawn', at: 4, of: 3, toolCallId: 'call_2' },
      { kind: 'mode', at: 5, mode: 'auto', from: 'bypass', by: 'user', print: PRINT, editsUnasked: true },
      { kind: 'allowed', at: 6, mode: 'auto', toolCallId: 'call_3', access: 'edit' },
      { kind: 'settings', at: 7, print: PRINT },
      { kind: 'mode', at: 8, mode: 'ask', from: 'auto', by: 'restart' },
    ];
    for (const line of lines) await log.append('conv-1', line);
    const text = readFileSync(join(root, 'permission', 'conv-1.jsonl'), 'utf8')
      .split('\n')
      .filter((l) => l !== '');
    expect(text.map((l) => Object.keys(JSON.parse(l) as object).join(','))).toEqual([
      'kind,at,mode,from,by,print,userHooks',
      'kind,at,mode,toolCallId,access',
      'kind,at,mode,toolCallId,access',
      'kind,at,of,toolCallId',
      'kind,at,mode,from,by,print,editsUnasked',
      'kind,at,mode,toolCallId,access',
      'kind,at,print',
      'kind,at,mode,from,by',
    ]);
    expect(text.map(readPermissionLine)).toEqual(lines);
    expect(await log.read('conv-1')).toEqual(lines.filter((l) => l.at !== 3 && l.at !== 4));
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
    ['a bad print', `{"kind":"mode","at":1,"mode":"auto","from":"ask","by":"user","print":"xyz"}`],
    ['a bad print on settings', '{"kind":"settings","at":1,"print":"0"}'],
    ['extra text', '{"kind":"mode","at":1,"mode":"auto","from":"ask","by":"user"} x'],
    ['an extra key', '{"kind":"mode","at":1,"mode":"auto","from":"ask","by":"user","on":true}'],
    ['an allowed line with no mode', '{"kind":"allowed","at":1,"toolCallId":"c","access":"bash"}'],
    ['an Auto line for a tool', '{"kind":"allowed","at":1,"mode":"auto","toolCallId":"c","access":"mcp"}'],
    ['a bad by', '{"kind":"mode","at":1,"mode":"auto","from":"ask","by":"model"}'],
    ['a negative time', '{"kind":"settings","at":-1,"print":"' + PRINT + '"}'],
    ['not JSON', 'mode auto'],
  ])('%s is skipped', (_name, text) => {
    expect(readPermissionLine(text)).toBeNull();
    expect(readPermissionLines(`${text}\n{"kind":"settings","at":2,"print":"${PRINT}"}`)).toEqual([
      { kind: 'settings', at: 2, print: PRINT },
    ]);
  });

  it('a link at the file place is refused; a read through it gives nothing', async () => {
    const dir = join(root, 'permission');
    mkdirSync(dir, { recursive: true });
    const target = join(root, 'elsewhere.jsonl');
    writeFileSync(target, `{"kind":"settings","at":2,"print":"${PRINT}"}\n`);
    symlinkSync(target, join(dir, 'conv-1.jsonl'));
    const log = new PermissionLog(dir);
    await expect(log.append('conv-1', { kind: 'settings', at: 3, print: PRINT })).rejects.toThrow();
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
    const before = log.append('conv-1', { kind: 'settings', at: 1, print: PRINT });
    const removed = log.remove('conv-1');
    const after = log.append('conv-1', { kind: 'mode', at: 2, mode: 'ask', from: 'ask', by: 'user' });
    await before;
    await removed;
    await expect(after).rejects.toThrow();
    expect(existsSync(file)).toBe(false);
    await log.append('conv-2', { kind: 'settings', at: 3, print: PRINT });
    expect(existsSync(join(dir, 'conv-2.jsonl'))).toBe(true);
  });
});

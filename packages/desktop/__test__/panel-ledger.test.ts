/**
 * MAIN's panel ledger: one encoder that lets only numbers, booleans, list members, UUIDs, row ids and route patterns
 * through, and the reader that turns intent and exit rows back into the ends the next panel shows.
 */

import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vite-plus/test';

import {
  createPanelLedger,
  encodeRow,
  isRoutePattern,
  isUuid,
  parseRow,
  readAppEnds,
  type LedgerRow,
} from '../src/panel-ledger.js';

const CONV = '019a1f2e-7b3c-7d4e-8f50-61728394a5b6';
const OTHER = '019a1f2e-7b3c-7d4e-8f50-000000000001';
const ENVELOPE = { t: 1_700_000_000_000, m: 5, p: '77.1000', n: 1 };

const dirs: string[] = [];
afterEach(() => {
  for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
});
function tempDir(): string {
  const dir = mkdtempSync(join(tmpdir(), 'mlx-ledger-'));
  dirs.push(dir);
  return dir;
}

describe('the encoder', () => {
  it('drops unknown keys, off-list enums, bad ids and a route with a digit, and counts each', () => {
    const encoded = encodeRow(ENVELOPE, {
      k: 'intent',
      what: 'panel.stop',
      cause: 'main.panel_no_pong',
      stage: 'sleeping',
      ageMs: 120_000,
      open: 1,
      convs: [CONV, 'not-a-uuid'],
      route: 'GET /api/sessions/019a1f2e',
      thread: 'worker',
      prompt: 'rm -rf /',
    });
    expect(encoded).not.toBeNull();
    expect(JSON.parse(encoded!.line)).toEqual({
      v: 1,
      t: ENVELOPE.t,
      m: 5,
      p: '77.1000',
      n: 1,
      k: 'intent',
      what: 'panel.stop',
      cause: 'main.panel_no_pong',
      ageMs: 120_000,
      open: 1,
      thread: 'worker',
    });
    expect(encoded!.dropped).toBe(4);
  });

  it('accepts route patterns from the table and nothing that carries a value', () => {
    for (const route of [
      'GET /api/sessions/:id',
      'POST /api/chat/conversations/:id/interactions/:iid',
      'GET /api/health',
    ]) {
      expect(isRoutePattern(route), route).toBe(true);
    }
    for (const route of [
      'GET /api/sessions/019a1f2e-7b3c',
      'GET /api/models/qwen3.5-4b',
      'GET /api/sessions?limit=20',
      'GET /Users/me/secret',
      'FETCH /api/models',
      'GET /a/b/c/d/e/f/g/h/i',
      'GET api/models',
      'GET /api//models',
    ]) {
      expect(isRoutePattern(route), route).toBe(false);
    }
  });

  it('a fuzz of quoted text, paths, URLs and base64 leaves none of it in the file', () => {
    const dir = tempDir();
    let t = 0;
    const ledger = createPanelLedger({
      dir,
      pid: 77,
      ppid: 1,
      ver: '1.0.0',
      startMs: 1_000,
      now: () => (t += 1),
      mono: () => t,
    });
    const poison = [
      '"quoted prompt text"',
      '/Users/someone/Projects/secret/file.ts',
      'https://example.com/?token=abc',
      'c2VjcmV0LWtleS1iYXNlNjQ=',
      'sk-ant-0123456789',
      'tok-INFERENCE-0123456789abcdef',
    ];
    const keys = [
      'route',
      'wroute',
      'thread',
      'stage',
      'cause',
      'what',
      'reason',
      'by',
      'target',
      'child',
      'ver',
      'role',
      'state',
      'op',
    ];
    for (const value of poison) {
      for (const key of keys) {
        ledger.record({
          k: 'slow',
          op: 'loop',
          target: 'panel',
          state: 'open',
          ageMs: 1,
          [key]: value,
        } as unknown as LedgerRow);
        ledger.record({
          k: 'child.exit',
          role: 'panel',
          asked: false,
          cause: 'panel.exit_unasked',
          uptimeMs: 1,
          crashes: 1,
          open: 0,
          convs: [value],
          cancelled: [value, CONV],
          [key]: value,
        } as unknown as LedgerRow);
      }
      ledger.record({
        k: 'intent',
        what: 'panel.stop',
        cause: 'user.app_quit',
        open: 0,
        convs: [CONV],
        note: value,
      } as unknown as LedgerRow);
    }
    ledger.close();
    const text = readFileSync(join(dir, 'main-1000-77.1.jsonl'), 'utf8');
    for (const value of poison) expect(text).not.toContain(value);
    for (const fragment of ['Users', 'https', 'token', 'sk-ant', 'INFERENCE', 'quoted'])
      expect(text).not.toContain(fragment);
    expect(text).toContain('"k":"note","what":"refused"');
    // Every line parses back with nothing but the closed tables.
    for (const line of text.split('\n').filter((l) => l !== '')) expect(parseRow(line)).not.toBeNull();
    expect(statSync(join(dir, 'main-1000-77.1.jsonl')).mode & 0o777).toBe(0o600);
  });

  it('checks UUIDs by hand', () => {
    expect(isUuid(CONV)).toBe(true);
    for (const id of [
      '',
      CONV.toUpperCase().replace('-', '_'),
      `${CONV}x`,
      'workflow-completed-run-1',
      CONV.slice(1),
    ]) {
      expect(isUuid(id), id).toBe(false);
    }
  });

  it('never throws when the folder cannot be made; the failure is counted and said once', () => {
    const lines: string[] = [];
    const ledger = createPanelLedger({
      dir: '/dev/null/ledger',
      pid: 1,
      ppid: 1,
      ver: '1',
      startMs: 1,
      now: () => 1,
      mono: () => 1,
      log: (line) => lines.push(line),
    });
    expect(ledger.record({ k: 'child.spawn', role: 'panel', gen: 1 })).toBe('1.1#2');
    expect(ledger.failures()).toBeGreaterThan(0);
    expect(lines).toHaveLength(1);
  });
});

function line(n: number, t: number, row: Record<string, unknown>, p = '77.1000'): string {
  return encodeRow({ t, m: t, p, n }, row)!.line;
}

describe('readAppEnds', () => {
  it('joins each exit to its intent, newest first, skips other kinds, and accepts a torn last line', async () => {
    const dir = tempDir();
    const older = [
      line(1, 100, { k: 'proc', role: 'main', host: 'desktop', pid: 70, ppid: 1, ver: '1.0.0', seq: 1 }, '70.50'),
      line(2, 110, { k: 'child.spawn', role: 'panel', gen: 1 }, '70.50'),
      line(
        3,
        900,
        { k: 'intent', what: 'panel.stop', cause: 'user.app_quit', open: 1, convs: [OTHER], target: '70.50#2' },
        '70.50',
      ),
      line(
        4,
        950,
        {
          k: 'child.exit',
          role: 'panel',
          asked: true,
          cause: 'user.app_quit',
          by: '70.50#3',
          uptimeMs: 840,
          crashes: 0,
          open: 1,
          convs: [OTHER],
          code: 0,
        },
        '70.50',
      ),
    ].join('');
    writeFileSync(join(dir, 'main-50-70.1.jsonl'), older);
    const newer = [
      line(1, 1_000, { k: 'proc', role: 'main', host: 'desktop', pid: 77, ppid: 1, ver: '1.0.0', seq: 1 }),
      line(2, 1_010, { k: 'child.spawn', role: 'panel', gen: 1 }),
      line(3, 2_000, { k: 'slow', op: 'loop', target: 'panel', state: 'open', ageMs: 10_000 }),
      line(4, 5_000, {
        k: 'intent',
        what: 'panel.stop',
        cause: 'main.panel_no_pong',
        stage: 'turn',
        ageMs: 120_000,
        open: 1,
        convs: [CONV],
        target: '77.1000#2',
      }),
      line(5, 7_000, {
        k: 'child.exit',
        role: 'panel',
        child: '77.1000#2',
        asked: true,
        cause: 'main.panel_no_pong',
        by: '77.1000#4',
        sinceIntentMs: 2_000,
        uptimeMs: 6_000,
        crashes: 1,
        open: 1,
        convs: [CONV],
        cancelled: [CONV],
        code: 0,
      }),
      line(6, 7_010, { k: 'child.spawn', role: 'panel', gen: 2 }),
      line(7, 20_000, {
        k: 'child.exit',
        role: 'panel',
        asked: false,
        cause: 'panel.exit_unasked',
        uptimeMs: 13_000,
        crashes: 1,
        open: 0,
        convs: [],
        code: 1,
      }),
      '{"v":1,"t":21000,"m":21000,"p":"77.1000","n":8,"k":"child.ex',
    ].join('');
    writeFileSync(join(dir, 'main-1000-77.1.jsonl'), newer);
    writeFileSync(
      join(dir, 'panel-1000-78.1.jsonl'),
      line(1, 1, {
        k: 'child.exit',
        role: 'panel',
        asked: false,
        cause: 'panel.exit_unasked',
        uptimeMs: 1,
        crashes: 1,
        open: 0,
        convs: [],
      }),
    );

    expect(await readAppEnds(dir)).toEqual([
      {
        trigger: 'crash',
        decidedAt: 20_000 - 12_000,
        exitAt: 20_000,
        silentMs: null,
        code: 1,
        convs: [],
        cancelled: null,
      },
      {
        trigger: 'no-ping',
        decidedAt: 5_000,
        exitAt: 7_000,
        silentMs: 120_000,
        code: 0,
        convs: [CONV],
        cancelled: [CONV],
      },
      { trigger: 'user', decidedAt: 900, exitAt: 950, silentMs: null, code: 0, convs: [OTHER], cancelled: null },
    ]);
  });

  it('reads only the last 1 MiB of a big file', async () => {
    const dir = tempDir();
    const quit = (n: number, t: number, conv: string): string =>
      line(n, t, {
        k: 'child.exit',
        role: 'panel',
        asked: true,
        cause: 'user.app_quit',
        uptimeMs: 1,
        crashes: 0,
        open: 1,
        convs: [conv],
        code: 0,
      });
    const filler: string[] = [];
    for (let n = 2; filler.length * 100 < 2 * 1024 * 1024; n += 1) {
      filler.push(line(n, 2_000, { k: 'slow', op: 'loop', target: 'panel', state: 'open', ageMs: 10_000 }));
    }
    writeFileSync(
      join(dir, 'main-1000-77.1.jsonl'),
      [quit(1, 1_000, OTHER), ...filler, quit(filler.length + 2, 3_000, CONV)].join(''),
    );
    const ends = await readAppEnds(dir);
    expect(ends.map((end) => end.convs)).toEqual([[CONV]]);
  });

  it('is empty for a missing or unset folder', async () => {
    expect(await readAppEnds(undefined)).toEqual([]);
    expect(await readAppEnds(join(tempDir(), 'missing'))).toEqual([]);
  });
});

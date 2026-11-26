/**
 * Pictures in tool results on the real agent binary (`MLX_AGENT_E2E_BIN`) with a scripted engine and no model: the
 * model reads a PNG and pages 4 to 6 of a 12-page PDF (past 10 pages the agent needs `pages`). The read lines have the fake agent's shape (`pictureRead` step); live and
 * stored give the same `pictures` and no base64 anywhere in the entries; the session reader serves each picture's
 * bytes from `updates.jsonl` as the wire carried them. Skipped without the variable.
 */

import { mkdirSync, mkdtempSync, readdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { crc32, deflateSync } from 'node:zlib';

import { describe, expect, it } from 'vite-plus/test';

import { AgentSessionsReader } from '../src/agent-sessions/reader.js';
import { parseUpdates, TranscriptReducer } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { AGENT_ARGS } from '../src/chat/agent/profiles.js';
import { fakeHost, noProviders, until, type FakeHost, type Json } from './helpers/chat-fake.js';
import { hasToolResult, lastMessage, startScriptedEngine, type ModelReply } from './helpers/scripted-engine.js';

const BIN = process.env.MLX_AGENT_E2E_BIN;
const MODEL = 'qwen3.5-4b';
const AGENT_WAIT_MS = 30_000;
const PROMPT = 'Read the screenshot and the spec.';

type Tool = Extract<TranscriptEntry, { kind: 'tool' }>;

/** A plain RGB PNG of the given sides. */
function png(width: number, height: number): Buffer {
  const chunk = (type: string, data: Buffer): Buffer => {
    const length = Buffer.alloc(4);
    length.writeUInt32BE(data.length);
    const body = Buffer.concat([Buffer.from(type, 'ascii'), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(body));
    return Buffer.concat([length, body, crc]);
  };
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8;
  header[9] = 2;
  const rows: Buffer[] = [];
  for (let y = 0; y < height; y += 1) {
    const row = Buffer.alloc(1 + width * 3);
    for (let x = 0; x < width; x += 1) {
      row[1 + x * 3] = (x * 5 + y) & 255;
      row[2 + x * 3] = (y * 2) & 255;
      row[3 + x * 3] = 96;
    }
    rows.push(row);
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', header),
    chunk('IDAT', deflateSync(Buffer.concat(rows))),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

/** A PDF of `pages` pages, each a filled square, with a correct cross-reference table. */
function pdf(pages: number): Buffer {
  const objects: string[] = [];
  const kids = Array.from({ length: pages }, (_, i) => `${3 + i * 2} 0 R`).join(' ');
  objects.push('<< /Type /Catalog /Pages 2 0 R >>');
  objects.push(`<< /Type /Pages /Kids [${kids}] /Count ${pages} >>`);
  for (let i = 0; i < pages; i += 1) {
    const stream = `${(i * 0.3).toFixed(1)} 0.4 0.8 rg 20 20 160 160 re f`;
    objects.push(`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << >> /Contents ${4 + i * 2} 0 R >>`);
    objects.push(`<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`);
  }
  let text = '%PDF-1.4\n';
  const offsets: number[] = [];
  objects.forEach((body, i) => {
    offsets.push(text.length);
    text += `${i + 1} 0 obj\n${body}\nendobj\n`;
  });
  const xref = text.length;
  text += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  for (const offset of offsets) text += `${String(offset).padStart(10, '0')} 00000 n \n`;
  text += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return Buffer.from(text, 'latin1');
}

/** Runs one turn and allows every request it makes, until it ends. */
async function runAllowing(fake: FakeHost, sessionId: string, text: string): Promise<string> {
  const turn = fake.host.startTurn(sessionId, text, { modelId: MODEL, cwd: fake.project });
  let answered = 0;
  const ended: { outcome: string | null } = { outcome: null };
  void turn.done.then((end) => {
    ended.outcome = end.outcome ?? 'no outcome';
  });
  for (;;) {
    await until(() => ended.outcome !== null || fake.events.interactions.length > answered, AGENT_WAIT_MS);
    for (; answered < fake.events.interactions.length; answered += 1) {
      fake.host.answer(sessionId, fake.events.interactions[answered]!.interaction.id, 'allow-once');
    }
    if (ended.outcome !== null) return ended.outcome;
  }
}

/** The completed update of each read call on the wire, in order. */
function readResults(fake: FakeHost, sessionId: string): Json[] {
  return fake.events.updates
    .filter((u) => u.sessionId === sessionId && !u.line.xai)
    .map((u) => u.line.update)
    .filter((update) => update.sessionUpdate === 'tool_call_update' && Array.isArray(update.content))
    .filter((update) =>
      (update.content as Json[]).some((item) => (item.content as Json | undefined)?.type === 'image'),
    );
}

/** Every tool line of each call that read a picture, in order, by call. */
function readCalls(fake: FakeHost, sessionId: string): Json[][] {
  const ids = new Set(readResults(fake, sessionId).map((update) => update.toolCallId));
  const byCall = new Map<unknown, Json[]>();
  for (const u of fake.events.updates) {
    if (u.sessionId !== sessionId || u.line.xai) continue;
    const update = u.line.update;
    if (!ids.has(update.toolCallId)) continue;
    if (!byCall.has(update.toolCallId)) byCall.set(update.toolCallId, []);
    byCall.get(update.toolCallId)!.push(update);
  }
  return [...byCall.values()];
}

/** The fake agent's `pictureRead` steps of the Work words "picture" and "readpdf", played on the fake host. */
async function fakeReadCalls(): Promise<Json[][]> {
  const fake = await fakeHost('chat-harness', { mode: 'work' });
  try {
    const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
    expect(await runAllowing(fake, sessionId, 'picture')).toBe('completed');
    expect(await runAllowing(fake, sessionId, 'readpdf')).toBe('completed');
    return readCalls(fake, sessionId);
  } finally {
    await fake.close();
  }
}

/** The keys of a value, and of its nested records and arrays of records, as paths. */
function shapeOf(value: unknown, at = ''): string[] {
  if (Array.isArray(value)) return value.length === 0 ? [`${at}[]`] : shapeOf(value[0], `${at}[]`);
  if (typeof value !== 'object' || value === null) return [`${at}:${typeof value}`];
  return Object.keys(value)
    .sort()
    .flatMap((key) => shapeOf((value as Json)[key], `${at}.${key}`));
}

describe.skipIf(BIN === undefined)('Pictures in tool results on the real agent', () => {
  it('a PNG and pages 4 to 6 of a 12-page PDF: fake-agent shape, live = stored, no base64 in entries, the reader serves the bytes', async () => {
    const fakeCalls = await fakeReadCalls();
    expect(fakeCalls).toHaveLength(2);

    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-tool-pictures-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    const script: ModelReply[] = [
      { blocks: [{ type: 'tool_use', name: 'read_file', input: { target_file: 'docs/shot.png' } }] },
      { blocks: [{ type: 'tool_use', name: 'read_file', input: { target_file: 'docs/spec.pdf', pages: '4-6' } }] },
    ];
    let calls = 0;
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: () =>
        startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q) => {
          if (q.tools.length === 1 && q.tools[0] === 'session_title') {
            return { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Read probe' } }] };
          }
          if (calls > 0 && !hasToolResult(lastMessage(q))) return { blocks: [{ type: 'text', text: 'Done.' }] };
          const step = script[calls];
          calls += 1;
          return step ?? { blocks: [{ type: 'text', text: 'Both read.' }] };
        }),
      env: { TMPDIR: `${agentTmp}/` },
    });
    try {
      const project = realpathSync(real.project);
      mkdirSync(join(project, 'docs'), { recursive: true });
      writeFileSync(join(project, 'docs/shot.png'), png(64, 48));
      writeFileSync(join(project, 'docs/spec.pdf'), pdf(12));
      const { sessionId } = await real.host.createSession({ modelId: MODEL, cwd: real.project });
      expect(await runAllowing(real, sessionId, PROMPT)).toBe('completed');

      const wire = readResults(real, sessionId);
      expect(wire).toHaveLength(2);
      // The fake agent's lines have the real agent's shape, picture and PDF alike: the call, its title, its result.
      const realCalls = readCalls(real, sessionId);
      expect(realCalls.map((lines) => lines.map((line) => shapeOf(line)))).toEqual(
        fakeCalls.map((lines) => lines.map((line) => shapeOf(line))),
      );
      expect(realCalls.map((lines) => lines.map((line) => line.title ?? null))).toEqual([
        ['read_file', 'Read `docs/shot.png`', null],
        ['read_file', 'Read `docs/spec.pdf`', null],
      ]);
      const [shot, spec] = wire as [Json, Json];
      expect((shot.content as Json[]).map((item) => (item.content as Json).mimeType)).toEqual(['image/png']);
      expect((spec.content as Json[]).map((item) => (item.content as Json).mimeType)).toEqual([
        'image/jpeg',
        'image/jpeg',
        'image/jpeg',
      ]);
      expect(spec.rawOutput).toMatchObject({ type: 'ReadFile', PdfPageImages: { total_pages: 12 } });
      expect(realCalls[1]!.map((line) => (line.rawInput as Json | undefined)?.pages ?? null)).toEqual([
        '4-6',
        '4-6',
        null,
      ]);

      let stored = '';
      await until(() => {
        const files = readdirSync(real.agentHome, { recursive: true, encoding: 'utf8' })
          .filter((f) => f.endsWith('updates.jsonl'))
          .map((f) => readFileSync(join(real.agentHome, f), 'utf8'))
          .filter((text) => text.includes('PdfPageImages'));
        stored = files.length === 1 ? files[0]! : '';
        return stored !== '';
      }, AGENT_WAIT_MS);

      const live = new TranscriptReducer();
      live.userPrompt(PROMPT);
      for (const u of real.events.updates) if (u.sessionId === sessionId) live.apply(u.line);
      const reads = (entries: readonly TranscriptEntry[]): Tool[] =>
        entries.filter((e): e is Tool => e.kind === 'tool' && e.pictures !== undefined);
      const liveReads = reads(live.snapshot().entries);
      const diskReads = reads(parseUpdates(stored).transcript);
      expect(diskReads).toEqual(liveReads);
      expect(liveReads.map((t) => [t.name, t.status, t.output ?? null, t.pdfPages ?? null])).toEqual([
        ['read_file', 'completed', null, null],
        ['read_file', 'completed', null, 12],
      ]);
      expect(liveReads[1]!.pictures!.map((p) => [p.n, p.mediaType, p.page])).toEqual([
        [0, 'image/jpeg', 4],
        [1, 'image/jpeg', 5],
        [2, 'image/jpeg', 6],
      ]);
      expect(liveReads[0]!.pictures).toEqual([{ n: 0, mediaType: 'image/png', bytes: png(64, 48).length }]);
      const shotData = ((shot.content as Json[])[0]!.content as Json).data as string;
      expect(JSON.stringify(live.snapshot())).not.toContain(shotData.slice(0, 64));
      expect(JSON.stringify(parseUpdates(stored))).not.toContain(shotData.slice(0, 64));

      // The reader serves what the wire carried, from disk.
      const reader = new AgentSessionsReader({ agentHome: real.agentHome });
      expect(reader.picture(sessionId, String(shot.toolCallId), 0)).toEqual({
        ok: true,
        mediaType: 'image/png',
        data: shotData,
      });
      const page2 = ((spec.content as Json[])[1]!.content as Json).data as string;
      expect(reader.picture(sessionId, String(spec.toolCallId), 1)).toEqual({
        ok: true,
        mediaType: 'image/jpeg',
        data: page2,
      });
      expect(reader.picture(sessionId, String(spec.toolCallId), 3)).toEqual({ ok: false, error: 'not-found' });
    } finally {
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  }, 120_000);
});

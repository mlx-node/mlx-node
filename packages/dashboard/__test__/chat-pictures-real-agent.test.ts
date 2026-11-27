/**
 * Picture notices on the real agent binary (`MLX_AGENT_E2E_BIN`) with a scripted engine and no model: a prompt with
 * a picture wider than the agent's 2,000 px side and one under its 8 px floor gets `image_compressed` and
 * `image_dropped`, with the fields the fake agent's harness lines have. The agent makes both after the prompt's chunks,
 * and its store can write the last chunk after them; the reducer gives the same list live and from `updates.jsonl`,
 * the two notices at the end of the one turn. Skipped without the variable.
 */

import { mkdirSync, mkdtempSync, readdirSync, readFileSync, realpathSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { crc32, deflateSync } from 'node:zlib';

import { describe, expect, it } from 'vite-plus/test';

import { parseUpdates, TranscriptReducer } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { AGENT_ARGS } from '../src/chat/agent/profiles.js';
import { fakeHost, noProviders, until, type Json } from './helpers/chat-fake.js';
import { startScriptedEngine, type ModelReply } from './helpers/scripted-engine.js';

const BIN = process.env.MLX_AGENT_E2E_BIN;
const MODEL = 'qwen3.5-4b';
const AGENT_WAIT_MS = 30_000;
const PROMPT = 'Picture probe: what is in these?';

type Picture = Extract<TranscriptEntry, { kind: 'notice'; code: 'picture' }>;

/** A plain RGB PNG of the given sides. */
function png(width: number, height: number): string {
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
      row[1 + x * 3] = (x * 7 + y) & 255;
      row[2 + x * 3] = (y * 3) & 255;
      row[3 + x * 3] = 128;
    }
    rows.push(row);
  }
  const signature = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  return Buffer.concat([
    signature,
    chunk('IHDR', header),
    chunk('IDAT', deflateSync(Buffer.concat(rows))),
    chunk('IEND', Buffer.alloc(0)),
  ]).toString('base64');
}

/** The harness's own picture lines of the fake agent, by `sessionUpdate`. */
function fakeLines(): Map<string, Json> {
  const scenario = JSON.parse(
    readFileSync(join(import.meta.dirname, 'fixtures/chat-scenarios/chat-harness.json'), 'utf8'),
  ) as { prompts: Array<{ match: string; promptPictures?: Json[]; steps: Array<{ picture?: Json }> }> };
  const word = scenario.prompts.find((p) => p.match === 'picture')!;
  const lines = new Map<string, Json>();
  for (const line of [...(word.promptPictures ?? []), ...word.steps.flatMap((s) => (s.picture ? [s.picture] : []))]) {
    const { sessionUpdate = 'image_dropped', ...fields } = line;
    lines.set(String(sessionUpdate), { sessionUpdate, ...fields });
  }
  return lines;
}

function keysOf(value: unknown): string[] {
  return Object.keys(value as Json).sort();
}

describe.skipIf(BIN === undefined)('Picture notices on the real agent', () => {
  it('a downscaled and a dropped prompt picture: fake-agent fields; stored among the prompt chunks, live = stored', async () => {
    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-pictures-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: () =>
        startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q): ModelReply => {
          if (q.tools.length === 1 && q.tools[0] === 'session_title') {
            return { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Picture probe' } }] };
          }
          return { blocks: [{ type: 'text', text: 'One picture.' }] };
        }),
      env: { TMPDIR: `${agentTmp}/` },
    });
    try {
      const { sessionId } = await real.host.createSession({ modelId: MODEL, cwd: real.project });
      const turn = real.host.startTurn(sessionId, PROMPT, { modelId: MODEL, cwd: real.project }, [
        { kind: 'image', mediaType: 'image/png', data: png(2_100, 300) },
        { kind: 'image', mediaType: 'image/png', data: png(4, 4) },
      ]);
      expect((await turn.done).outcome).toBe('completed');

      // On the wire: the two one-shots, with the fields of the fake agent's lines.
      const wire = real.events.updates
        .filter((u) => u.sessionId === sessionId && u.line.xai)
        .map((u) => u.line.update)
        .filter((update) => update.sessionUpdate === 'image_compressed' || update.sessionUpdate === 'image_dropped');
      expect(wire.map((update) => update.sessionUpdate)).toEqual(['image_compressed', 'image_dropped']);
      const fake = fakeLines();
      const [shrunk, dropped] = wire as [Json, Json];
      expect(keysOf(shrunk)).toEqual(keysOf(fake.get('image_compressed')));
      const images = shrunk.images as Json[];
      expect(images).toHaveLength(1);
      expect(keysOf(images[0])).toEqual(keysOf((fake.get('image_compressed')!.images as Json[])[0]));
      expect(images[0]).toMatchObject({
        index: 1,
        original_width: 2_100,
        original_height: 300,
        compressed_width: 2_000,
      });
      expect(String(shrunk.message).startsWith('Downscaled Image 1: ')).toBe(true);
      expect(keysOf(dropped)).toEqual(keysOf(fake.get('image_dropped')));
      const notes = dropped.notes as unknown[];
      expect(notes).toHaveLength(1);
      expect(String(notes[0]).startsWith('Image 2 was dropped before send: ')).toBe(true);

      // On disk: made after the prompt's chunks, stored after its first one; the store's buffer can put a later chunk
      // after them (the store holds them until the prompt is whole).
      let stored = '';
      await until(() => {
        const files = readdirSync(real.agentHome, { recursive: true, encoding: 'utf8' })
          .filter((f) => f.endsWith('updates.jsonl'))
          .map((f) => readFileSync(join(real.agentHome, f), 'utf8'))
          .filter((text) => text.includes(PROMPT) && text.includes('image_dropped'));
        stored = files.length === 1 ? files[0]! : '';
        return stored !== '';
      }, AGENT_WAIT_MS);
      const kinds = stored
        .split('\n')
        .filter((line) => line !== '')
        .map((line) => String((JSON.parse(line) as { params?: { update?: Json } }).params?.update?.sessionUpdate));
      expect(kinds.indexOf('user_message_chunk')).toBeLessThan(kinds.indexOf('image_compressed'));
      expect(kinds.indexOf('image_compressed')).toBeLessThan(kinds.indexOf('image_dropped'));
      expect(kinds.filter((kind) => kind === 'user_message_chunk')).toHaveLength(3);

      // One reducer: the page's live list and the stored list give the same notices, at the turn's end.
      const live = new TranscriptReducer();
      live.userPrompt(PROMPT);
      for (const u of real.events.updates) if (u.sessionId === sessionId) live.apply(u.line);
      const liveEntries = live.snapshot().entries;
      const disk = parseUpdates(stored).transcript;
      const pictures = (entries: readonly TranscriptEntry[]): Picture[] =>
        entries.filter((e): e is Picture => e.kind === 'notice' && e.code === 'picture');
      expect(pictures(liveEntries)).toEqual(pictures(disk));
      expect(pictures(disk).map((p) => [p.what, p.count, p.sizes?.length ?? 0])).toEqual([
        ['shrunk', 1, 1],
        ['dropped', 1, 0],
      ]);
      for (const entries of [liveEntries, disk]) {
        expect(entries.map((e) => e.kind)).toEqual(['user', 'assistant', 'notice', 'notice', 'turn-end']);
      }
      // The prompt's three chunks are still one message on disk, with both pictures.
      expect(disk[0]).toMatchObject({ kind: 'user', text: PROMPT });
      expect(disk[0]!.kind === 'user' ? disk[0]!.attachments?.map((a) => a.kind) : null).toEqual(['image', 'image']);
    } finally {
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  }, 120_000);
});

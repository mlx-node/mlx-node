/**
 * Send now on the real agent binary (`MLX_AGENT_E2E_BIN`) with a scripted engine and no model: `_x.ai/interject`
 * answers `{result: {status: "queued"}}`, the model gets the text at the agent's next step, and the agent stores it as
 * `_meta.interjection` with the typed text in `displayText`, which the reducer reads as the host's own steer row. A Stop
 * drops a steer the model did not read; one sent with no turn runs as its own turn, `interject-fallback-`, which the
 * live wake row names `message` and the disk keeps as a typed prompt. Skipped without the variable.
 */

import { mkdirSync, mkdtempSync, readdirSync, readFileSync, realpathSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { parseUpdates, TranscriptReducer } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import type { AcpClient } from '../src/chat/acp/client.js';
import { AGENT_ARGS } from '../src/chat/agent/profiles.js';
import { fakeHost, noProviders, until, type FakeHost, type Json } from './helpers/chat-fake.js';
import { lastMessage, messageText, startScriptedEngine, type ModelReply } from './helpers/scripted-engine.js';

const BIN = process.env.MLX_AGENT_E2E_BIN;
const MODEL = 'qwen3.5-4b';
const AGENT_WAIT_MS = 30_000;
const STEER = 'use pnpm instead';
const DROPPED = 'stop, use yarn';
const IDLE = 'one more thing while idle';

function storedText(agentHome: string, marker: string): string {
  const files = readdirSync(agentHome, { recursive: true, encoding: 'utf8' })
    .filter((f) => f.endsWith('updates.jsonl'))
    .map((f) => readFileSync(join(agentHome, f), 'utf8'))
    .filter((text) => text.includes(marker));
  return files.length === 1 ? files[0]! : '';
}

/** The stored user chunks of the file, raw. */
function userChunks(text: string): Json[] {
  return text
    .split('\n')
    .filter((line) => line !== '')
    .map((line) => JSON.parse(line) as { params?: { update?: Json } })
    .flatMap((line) => {
      const update = line.params?.update;
      return update?.sessionUpdate === 'user_message_chunk' ? [update] : [];
    });
}

function untimed(entry: TranscriptEntry): Omit<TranscriptEntry, 'ts'> {
  const { ts: _ts, ...rest } = entry;
  return rest;
}

function liveEntries(fake: FakeHost, sessionId: string): TranscriptEntry[] {
  const reducer = new TranscriptReducer();
  for (const u of fake.events.updates) if (u.sessionId === sessionId) reducer.apply(u.line);
  return reducer.snapshot().entries;
}

describe.skipIf(BIN === undefined)('Send now on the real agent', () => {
  it('interject reply, the model reads it, the stored row equals the host row; Stop drops; idle runs its own turn', async () => {
    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-steer-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    /** The text of the last user message of each model call that was not a title call. */
    const asked: string[] = [];
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: () =>
        startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q): ModelReply => {
          if (q.tools.length === 1 && q.tools[0] === 'session_title') {
            return { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Steer probe' } }] };
          }
          const last = messageText(lastMessage(q));
          asked.push(last);
          if (last.includes(STEER)) return { blocks: [{ type: 'text', text: 'Using pnpm.' }] };
          if (last.includes('Steer probe')) return { blocks: [{ type: 'text', text: 'Working.' }], holdMs: 2_500 };
          if (last.includes('Stop probe')) return { blocks: [{ type: 'text', text: 'Slow.' }], holdMs: 8_000 };
          return { blocks: [{ type: 'text', text: 'Read it.' }] };
        }),
      env: { TMPDIR: `${agentTmp}/` },
    });
    try {
      const { sessionId } = await real.host.createSession({ modelId: MODEL, cwd: real.project });
      const load = { modelId: MODEL, cwd: real.project };
      expect(await real.host.steer(sessionId, 'nothing runs')).toBe('not-now');

      // 1. Into a running answer: the reply is `queued`, the model reads it at the next step.
      const first = real.host.startTurn(sessionId, 'Steer probe', load);
      await until(() => asked.some((text) => text.includes('Steer probe')), AGENT_WAIT_MS);
      expect(await real.host.steer(sessionId, STEER)).toBe('sent');
      expect((await first.done).outcome).toBe('completed');
      const read = asked.find((text) => text.includes(STEER));
      expect(read).toContain('<user_query>');

      const host = new TranscriptReducer();
      const steerRow = host.snapshot().entries.length;
      host.steer(STEER, 1);
      const expected = untimed(host.snapshot().entries[steerRow]!);
      await until(() => userChunks(storedText(real.agentHome, 'Steer probe')).length >= 2, AGENT_WAIT_MS);
      const stored = userChunks(storedText(real.agentHome, 'Steer probe')).find(
        (chunk) => (chunk._meta as Json | undefined)?.interjection === true,
      );
      expect(stored).toMatchObject({
        _meta: { interjection: true },
        content: { type: 'text', _meta: { displayText: STEER } },
      });
      expect(((stored!.content as Json).text as string).includes('<user_query>')).toBe(true);
      const disk = parseUpdates(storedText(real.agentHome, 'Steer probe')).transcript;
      expect(disk.filter((e) => e.kind === 'user' && e.interjection === true).map(untimed)).toEqual([expected]);

      // 2. A Stop before the model read it: nothing stored, no turn of its own.
      const second = real.host.startTurn(sessionId, 'Stop probe', load);
      await until(() => asked.some((text) => text.includes('Stop probe')), AGENT_WAIT_MS);
      expect(await real.host.steer(sessionId, DROPPED)).toBe('sent');
      expect(real.host.cancel(sessionId)).toBe(true);
      expect((await second.done).outcome).toBe('cancelled');
      await new Promise((resolve) => setTimeout(resolve, 1_500));
      expect(asked.some((text) => text.includes(DROPPED))).toBe(false);
      expect(real.host.isBusy(sessionId)).toBe(false);
      expect(storedText(real.agentHome, 'Steer probe')).not.toContain(DROPPED);

      // 3. Sent with no turn (the host never does; this is the race it guards): the agent's own turn, `message`.
      const running = Reflect.get(real.host, 'running') as { process: { client: AcpClient } };
      const client = running.process.client;
      expect(await client.request('_x.ai/interject', { sessionId, text: IDLE }, null)).toEqual({
        result: { status: 'queued' },
      });
      // The typed two, then the agent's own.
      await until(() => real.events.ends.filter((e) => e.sessionId === sessionId).length === 3, AGENT_WAIT_MS);
      expect(asked.some((text) => text.includes(IDLE))).toBe(true);
      const wakes = liveEntries(real, sessionId).filter((e) => e.kind === 'wake');
      expect(wakes.at(-1)).toMatchObject({ kind: 'wake', cause: 'message' });
      await until(() => storedText(real.agentHome, 'Steer probe').includes(IDLE), AGENT_WAIT_MS);
      const typed = parseUpdates(storedText(real.agentHome, 'Steer probe')).transcript.filter(
        (e) => e.kind === 'user' && e.text === IDLE,
      );
      expect(typed).toHaveLength(1);
      expect(typed[0]).not.toHaveProperty('interjection');
      expect((typed[0] as Extract<TranscriptEntry, { kind: 'user' }>).promptIndex).toEqual(expect.any(Number));
    } finally {
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  }, 120_000);
});

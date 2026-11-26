/**
 * Recap on the real agent binary (`MLX_AGENT_E2E_BIN`) with a scripted engine and no model: `_x.ai/recap` with
 * `auto: false` acks, a session with no finished answer gets `session_recap_unavailable` (stored), one with an answer
 * gets one model call with no tools and one `session_recap` line, and the live lines and `updates.jsonl` give the same
 * recap entry. A typed message while the recap is written drops it: the wait ends `cancelled` and no row comes. Skipped
 * without the variable.
 */

import { mkdirSync, mkdtempSync, readdirSync, readFileSync, realpathSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { parseUpdates, TranscriptReducer } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { AGENT_ARGS } from '../src/chat/agent/profiles.js';
import { fakeHost, noProviders, until, type FakeHost } from './helpers/chat-fake.js';
import { startScriptedEngine, type ModelRequest, type ScriptedEngine } from './helpers/scripted-engine.js';

const BIN = process.env.MLX_AGENT_E2E_BIN;
const MODEL = 'qwen3.5-4b';
const AGENT_WAIT_MS = 30_000;
const MARKER = 'Recap probe: fix the upload retry.';
const RECAP = 'We fixed the upload retry; next, update the docs.';

/** The agent's recap call: its own system prompt and a transcript, no tools. */
const isRecap = (q: ModelRequest): boolean => q.tools.length === 0 && q.system.includes('recap body');

function stored(agentHome: string): { text: string; entries: TranscriptEntry[] } {
  const files = readdirSync(agentHome, { recursive: true, encoding: 'utf8' })
    .filter((f) => f.endsWith('updates.jsonl'))
    .map((f) => readFileSync(join(agentHome, f), 'utf8'))
    .filter((text) => text.includes(MARKER) || text.includes('session_recap'));
  expect(files).toHaveLength(1);
  return { text: files[0]!, entries: parseUpdates(files[0]!).transcript };
}

function live(fake: FakeHost, sessionId: string): TranscriptEntry[] {
  const reducer = new TranscriptReducer();
  for (const u of fake.events.updates) if (u.sessionId === sessionId) reducer.apply(u.line);
  return reducer.snapshot().entries;
}

const recaps = (entries: readonly TranscriptEntry[]): TranscriptEntry[] => entries.filter((e) => e.kind === 'recap');

describe.skipIf(BIN === undefined)('Recap on the real agent', () => {
  it('no answer yet: unavailable; after one: one model call and one stored row; a sent message drops a recap', async () => {
    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-recap-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    let holdRecap = 0;
    let engine: ScriptedEngine | null = null;
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: async () => {
        engine = await startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q) => {
          if (q.tools.length === 1 && q.tools[0] === 'session_title') {
            return { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Recap probe' } }] };
          }
          if (isRecap(q)) return { blocks: [{ type: 'text', text: RECAP }], holdMs: holdRecap };
          return { blocks: [{ type: 'text', text: 'Fixed.' }], inputTokens: 4_000 };
        });
        return engine;
      },
      env: { TMPDIR: `${agentTmp}/` },
    });
    try {
      const load = { modelId: MODEL, cwd: real.project };
      const { sessionId } = await real.host.createSession(load);
      // Only a manual recap (`auto: false`) is told it has nothing: the agent sends no line for an automatic one.
      expect(await real.host.recap(sessionId, load).done).toBe('none');
      const kinds = (): string[] =>
        real.events.updates.filter((u) => u.sessionId === sessionId).map((u) => String(u.line.update.sessionUpdate));
      expect(kinds()).toContain('session_recap_unavailable');
      expect(engine!.modelRequests.some(isRecap)).toBe(false);

      expect((await real.host.startTurn(sessionId, MARKER, load).done).outcome).toBe('completed');
      expect(await real.host.recap(sessionId, load).done).toBe('shown');
      expect(engine!.modelRequests.filter(isRecap)).toHaveLength(1);
      expect(recaps(live(real, sessionId))).toEqual([{ kind: 'recap', ts: expect.any(Number), text: RECAP }]);
      await until(() => stored(real.agentHome).text.includes('"session_recap"'), AGENT_WAIT_MS);
      const disk = stored(real.agentHome);
      // The shapes the fake agent plays.
      const lines = disk.text
        .split('\n')
        .filter((l) => l.includes('session_recap'))
        .map((l) => JSON.parse(l) as { method: string; params: { update: Record<string, unknown> } });
      expect(lines.map((l) => [l.method, l.params.update])).toEqual([
        ['_x.ai/session/update', { sessionUpdate: 'session_recap_unavailable' }],
        ['_x.ai/session/update', { sessionUpdate: 'session_recap', summary: RECAP, auto: false }],
      ]);
      expect(recaps(disk.entries)).toEqual(recaps(live(real, sessionId)));
      expect(disk.entries.filter((e) => e.kind !== 'user')).toEqual(
        live(real, sessionId).filter((e) => e.kind !== 'user'),
      );

      holdRecap = 3_000;
      const before = engine!.modelRequests.filter(isRecap).length;
      const run = real.host.recap(sessionId, load);
      await until(() => engine!.modelRequests.filter(isRecap).length > before, AGENT_WAIT_MS);
      const turn = real.host.startTurn(sessionId, 'And the docs?', load);
      expect(await run.done).toBe('cancelled');
      expect((await turn.done).outcome).toBe('completed');
      const held = engine!.modelRequests.filter(isRecap).at(-1)!;
      await until(() => held.doneAt !== null || held.closedEarly, AGENT_WAIT_MS);
      await until(() => kinds().filter((k) => k === 'session_recap_unavailable').length === 2, AGENT_WAIT_MS);
      expect(recaps(live(real, sessionId))).toHaveLength(1);
      expect(real.events.recaps.filter((r) => r.recap === null).map((r) => r.end)).toEqual([
        'none',
        'shown',
        'cancelled',
      ]);
    } finally {
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  }, 120_000);
});

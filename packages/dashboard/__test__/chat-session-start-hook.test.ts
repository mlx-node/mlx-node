/**
 * A `session_start` hook batch: the agent runs it off its command loop while `session/new` is still open
 * (`DispatchSessionStartHook`), so its stored `hook_execution`, and the `hook_annotation` of each hook's own message
 * after it, can come before the reply names the session. The host hands such lines on while a `session/new` is in
 * flight; the service keeps them until it holds the conversation. The rows then stand in the first turn where the
 * stored list has them. With the real agent (`MLX_AGENT_E2E_BIN`): a failing `SessionStart` hook and one that prints a
 * `systemMessage`. Skipped without the variable.
 */

import { mkdirSync, mkdtempSync, readdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { parseUpdates, TranscriptReducer } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { AGENT_ARGS } from '../src/chat/agent/profiles.js';
import type { ChatService } from '../src/chat/service.js';
import type { ConversationEvent } from '../src/chat/types.js';
import { applyConversationEvent, type ConversationState } from '../ui/src/lib/chat-store.js';
import { fakeHost, fakeService, noProviders, until, type Json } from './helpers/chat-fake.js';
import { startScriptedEngine, type ModelReply } from './helpers/scripted-engine.js';

const BIN = process.env.MLX_AGENT_E2E_BIN;
const MODEL = 'qwen3.5-4b';
const RUN = {
  name: 'global/env:session_start[0].hooks[0]',
  status: { status: 'failed', error: 'exit code 1: no .env here', elapsed_ms: 12 },
};
/** A hook's own `systemMessage`: the agent sends it as a note right after the batch. */
const NOTE = 'Loaded the team notes.';

function sessionStart(afterReply: boolean): (scenario: Json) => void {
  return (scenario) => {
    scenario.session = {
      ...(scenario.session as Json),
      sessionStart: { runs: [RUN], notes: [NOTE], ...(afterReply ? { afterReply } : {}) },
    };
  };
}

function watch(service: ChatService, id: string): { current: () => ConversationState | null; off: () => void } {
  let state: ConversationState | null = null;
  const off = service.watchConversation(id, (event: ConversationEvent) => {
    const next = applyConversationEvent(state, event);
    if (next !== 'resync') state = next;
  });
  return { current: () => state, off };
}

const shape = (entries: readonly TranscriptEntry[]): unknown[] =>
  entries.map((e) => {
    if (e.kind === 'user' || e.kind === 'assistant') return [e.kind, e.text];
    if (e.kind === 'notice' && e.code === 'hook') return [e.kind, e.event, e.hook, e.detail, e.times];
    if (e.kind === 'notice' && e.code === 'hook-note') return [e.kind, e.text];
    return [e.kind];
  });

describe('A session_start hook batch on the fake agent', () => {
  // Mutation: drop the host's hand-off of an unknown session's `session_start` line → the line never comes; drop the
  // `note` hand-off → the hook's own message never comes.
  it('the host hands on the batch and its note that came before the session/new reply, under the new id', async () => {
    const fake = await fakeHost('chat-basic', { mutate: sessionStart(false) });
    try {
      const created = await fake.host.createSession({ modelId: MODEL }).then((session) => ({
        session,
        seen: fake.events.updates.length,
      }));
      const lines = fake.events.updates.filter((u) => String(u.line.update.sessionUpdate).startsWith('hook_'));
      expect(lines).toEqual([
        {
          sessionId: created.session.sessionId,
          line: expect.objectContaining({
            xai: true,
            update: expect.objectContaining({ sessionUpdate: 'hook_execution', event_name: 'session_start' }),
          }),
        },
        {
          sessionId: created.session.sessionId,
          line: expect.objectContaining({
            xai: true,
            update: { sessionUpdate: 'hook_annotation', message: NOTE },
          }),
        },
      ]);
      expect(created.seen).toBeGreaterThan(fake.events.updates.indexOf(lines[1]!));
      // Another session's line with no session/new open is not handed on.
      expect(fake.events.updates.every((u) => u.sessionId === created.session.sessionId)).toBe(true);
    } finally {
      await fake.close();
    }
  }, 20_000);

  // Mutation: drop `keepEarly` in the service's `onUpdate` → the live thread has no row and differs from the stored one.
  it.each([
    ['before', false],
    ['after', true],
  ] as const)(
    'Chat: the line %s the reply gives the failed row in the first turn, live as after a restart',
    async (_when, afterReply) => {
      const fake = fakeService('chat-basic', { mutate: sessionStart(afterReply) });
      try {
        const { conversation } = await fake.service.create({ mode: 'chat', text: 'hello there' });
        const id = conversation.id;
        const page = watch(fake.service, id);
        await until(() => {
          const state = page.current();
          return state !== null && state.turn === null && state.entries.some((e) => e.kind === 'turn-end');
        }, 8_000);
        const live = page.current()!.entries;
        const hookRows = live.filter((e) => e.kind === 'notice');
        expect(shape(hookRows)).toEqual([
          ['notice', 'session_start', 'global/env', 'exit code 1: no .env here', 1],
          ['notice', NOTE],
        ]);
        const kinds = live.map((e) => e.kind);
        expect(kinds.indexOf('notice')).toBe(kinds.indexOf('turn-end') - 2);
        page.off();

        await fake.service.close();
        const after = watch(fake.restart(), id);
        await until(() => (after.current()?.entries.length ?? 0) > 0, 8_000);
        expect(shape(after.current()!.entries)).toEqual(shape(live));
        after.off();
      } finally {
        await fake.close();
      }
    },
    30_000,
  );
});

describe.skipIf(BIN === undefined)('A session_start hook batch on the real agent', () => {
  it('a failing SessionStart hook and one with a message: the lines reach the session, the rows stand live as stored', async () => {
    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-session-start-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: () =>
        startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q): ModelReply => {
          if (q.tools.length === 1 && q.tools[0] === 'session_title') {
            return { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Start probe' } }] };
          }
          return { blocks: [{ type: 'text', text: 'Hello.' }] };
        }),
      env: { TMPDIR: `${agentTmp}/` },
    });
    try {
      const hooksDir = join(real.agentHome, 'hooks');
      mkdirSync(hooksDir, { recursive: true });
      writeFileSync(
        join(hooksDir, 'probe.json'),
        JSON.stringify({
          hooks: {
            SessionStart: [
              {
                hooks: [
                  { type: 'command', command: "echo 'no .env here' >&2; exit 1" },
                  { type: 'command', command: `echo '${JSON.stringify({ systemMessage: NOTE })}'` },
                ],
              },
            ],
          },
        }),
      );
      const cwd = { cwd: real.project };
      const { sessionId } = await real.host.createSession({ modelId: MODEL, ...cwd });
      const startLine = (): boolean =>
        real.events.updates.some(
          (u) =>
            u.sessionId === sessionId &&
            u.line.update.sessionUpdate === 'hook_execution' &&
            u.line.update.event_name === 'session_start',
        );
      await until(startLine, 10_000);
      expect(
        (await real.host.startTurn(sessionId, 'Start probe: hello.', { modelId: MODEL, ...cwd }).done).outcome,
      ).toBe('completed');

      let stored = '';
      await until(() => {
        const files = readdirSync(real.agentHome, { recursive: true, encoding: 'utf8' })
          .filter((f) => f.endsWith('updates.jsonl'))
          .map((f) => readFileSync(join(real.agentHome, f), 'utf8'))
          .filter((text) => text.includes('session_start') && text.includes('turn_completed'));
        stored = files.length === 1 ? files[0]! : '';
        return stored !== '';
      }, 30_000);
      const live = new TranscriptReducer();
      const lines = real.events.updates.filter((u) => u.sessionId === sessionId).map((u) => u.line);
      const first = lines.findIndex(
        (l) => l.update.sessionUpdate !== 'hook_execution' && l.update.sessionUpdate !== 'hook_annotation',
      );
      for (const line of lines.slice(0, first)) live.apply(line);
      live.userPrompt('Start probe: hello.');
      for (const line of lines.slice(first)) live.apply(line);
      const shown = live.snapshot().entries;
      const disk = parseUpdates(stored).transcript;
      const rows = (entries: readonly TranscriptEntry[]) => entries.filter((e) => e.kind === 'notice');
      expect(rows(shown)).toEqual(rows(disk));
      expect(rows(disk)).toEqual([
        expect.objectContaining({ code: 'hook', event: 'session_start', detail: 'exit code 1: no .env here' }),
        expect.objectContaining({ code: 'hook-note', text: NOTE }),
      ]);
    } finally {
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  }, 120_000);
});

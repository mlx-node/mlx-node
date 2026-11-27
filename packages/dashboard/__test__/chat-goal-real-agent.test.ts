/**
 * The goal loop on the real agent binary (`MLX_AGENT_E2E_BIN`) with a scripted engine and no model, the planner off:
 * `/goal <objective>` stores its first `goal_updated` and then its goal rules as the prompt, which the reducer reads
 * back as the typed text; the rounds run inside the typed `session/prompt` (one end), each goal line stored as sent (the
 * live-only ticks come from helpers' progress); Send now reaches a running round; Stop pauses the goal after the turn's end; `/goal resume` runs it
 * again under its own rules; `/goal clear` is a host turn that clears it. Disk and live show one goal row. Skipped
 * without the variable.
 */

import { mkdirSync, mkdtempSync, readdirSync, readFileSync, realpathSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { parseUpdates, TranscriptReducer } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { AGENT_ARGS } from '../src/chat/agent/profiles.js';
import { fakeHost, noProviders, until, type FakeHost, type Json } from './helpers/chat-fake.js';
import { lastMessage, messageText, startScriptedEngine, type ModelReply } from './helpers/scripted-engine.js';

const BIN = process.env.MLX_AGENT_E2E_BIN;
const MODEL = 'qwen3.5-4b';
const AGENT_WAIT_MS = 30_000;
const OBJECTIVE = 'write a haiku about tests';
const TYPED = `/goal ${OBJECTIVE}`;
const STEER = 'keep it to three lines';

type Goal = Extract<TranscriptEntry, { kind: 'goal' }>;

function storedText(agentHome: string): string {
  const files = readdirSync(agentHome, { recursive: true, encoding: 'utf8' })
    .filter((f) => f.endsWith('updates.jsonl'))
    .map((f) => readFileSync(join(agentHome, f), 'utf8'))
    .filter((text) => text.includes(OBJECTIVE));
  return files.length === 1 ? files[0]! : '';
}

/** The stored lines' updates, in order. */
function storedUpdates(text: string): Json[] {
  return text
    .split('\n')
    .filter((line) => line !== '')
    .flatMap((line) => {
      const update = (JSON.parse(line) as { params?: { update?: Json } }).params?.update;
      return update === undefined ? [] : [update];
    });
}

/** The goal lines the host passed on live, in order. */
function liveGoalLines(fake: FakeHost, sessionId: string): Json[] {
  return fake.events.updates
    .filter((u) => u.sessionId === sessionId && u.line.update.sessionUpdate === 'goal_updated')
    .map((u) => u.line.update);
}

/** The page's list: the host pushes each typed prompt itself, then the agent's lines. */
function liveEntries(fake: FakeHost, sessionId: string, typed: readonly string[]): TranscriptEntry[] {
  const reducer = new TranscriptReducer();
  let next = 0;
  for (const u of fake.events.updates) {
    if (u.sessionId !== sessionId) continue;
    if (u.line.update.sessionUpdate === 'goal_updated' && next < typed.length && u.line.update.status === 'active') {
      // A prompt that sets or resumes the goal: the host's push came before the agent's first line.
      if (next === 0 || u.line.update.last_event === 'goal_resumed') reducer.userPrompt(typed[next++]!);
    }
    reducer.apply(u.line);
  }
  return reducer.snapshot().entries;
}

const goals = (entries: readonly TranscriptEntry[]): Goal[] => entries.filter((e): e is Goal => e.kind === 'goal');
const shown = (g: Goal) => ({ goalId: g.goalId, objective: g.objective, status: g.status, phase: g.phase });

describe.skipIf(BIN === undefined)('the goal loop on the real agent', () => {
  it('set, rounds in the typed answer, Send now, Stop pauses, resume runs it, clear; one row live and stored', async () => {
    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-goal-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    /** The last user message of each round's model call (not the title, not the per-round check). */
    const rounds: string[] = [];
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: () =>
        startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q): ModelReply => {
          if (q.tools.length === 1 && q.tools[0] === 'session_title') {
            return { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Goal probe' } }] };
          }
          const last = messageText(lastMessage(q));
          // The per-round check (`goal_evaluator.rs`): its verdict as strict JSON.
          if (last.startsWith('{"objective"')) {
            const verdict = { decision: 'continue', evidence: 'one line so far', next_step: 'go on', blocker_key: '' };
            return { blocks: [{ type: 'text', text: JSON.stringify(verdict) }] };
          }
          rounds.push(last);
          const n = rounds.length;
          return {
            blocks: [{ type: 'text', text: `Line ${String(n)}.` }],
            holdMs: n === 3 ? 3_000 : n > 3 ? 8_000 : 300,
          };
        }),
      env: { TMPDIR: `${agentTmp}/`, GROK_GOAL_PLANNER: '0' },
    });
    try {
      const { sessionId } = await real.host.createSession({ modelId: MODEL, cwd: real.project });
      const load = { modelId: MODEL, cwd: real.project };
      const ends = () => real.events.ends.filter((e) => e.sessionId === sessionId);
      const lastStatus = () => liveGoalLines(real, sessionId).at(-1)?.status;

      // 1. The rounds run inside the typed prompt: three model calls, no end yet; the goal runs.
      const first = real.host.startTurn(sessionId, TYPED, load);
      await until(() => rounds.length >= 3, AGENT_WAIT_MS * 2);
      expect(ends()).toEqual([]);
      expect(real.host.isBusy(sessionId)).toBe(true);
      expect(real.host.goalRuns(sessionId)).toBe(true);

      // 2. Send now reaches the running round: the next round's call reads it.
      expect(await real.host.steer(sessionId, STEER)).toBe('sent');
      await until(() => rounds.some((text) => text.includes(STEER)), AGENT_WAIT_MS);

      // 3. Stop: the turn ends cancelled, then the agent stores the goal paused.
      expect(real.host.cancel(sessionId)).toBe(true);
      expect((await first.done).outcome).toBe('cancelled');
      await until(() => lastStatus() === 'user_paused', AGENT_WAIT_MS);
      expect(real.host.goalRuns(sessionId)).toBe(false);
      expect(ends()).toHaveLength(1);

      // 4. Resume runs the goal again; a Stop pauses it again.
      const before = rounds.length;
      const resumed = real.host.startTurn(sessionId, '/goal resume', load);
      await until(() => rounds.length > before, AGENT_WAIT_MS);
      expect(real.host.goalRuns(sessionId)).toBe(true);
      expect(real.host.cancel(sessionId)).toBe(true);
      expect((await resumed.done).outcome).toBe('cancelled');
      await until(() => lastStatus() === 'user_paused', AGENT_WAIT_MS);

      // 5. Clear: a host turn, no model call; the goal line with no id.
      const asked = rounds.length;
      expect((await real.host.startTurn(sessionId, '/goal clear', load).done).outcome).toBe('completed');
      await until(() => lastStatus() === 'cleared', AGENT_WAIT_MS);
      expect(liveGoalLines(real, sessionId).at(-1)).toMatchObject({ goal_id: '', objective: '', status: 'cleared' });
      expect(rounds).toHaveLength(asked);

      // The stored file: the first goal line before the prompt, the prompt the goal rules, the resume's its own.
      // The host turn's typed text is stored after the goal line it made.
      await until(() => storedText(real.agentHome).includes('"text":"/goal clear"'), AGENT_WAIT_MS);
      const text = storedText(real.agentHome);
      const updates = storedUpdates(text);
      const firstGoal = updates.findIndex((u) => u.sessionUpdate === 'goal_updated');
      const prompts = updates.filter(
        (u) => u.sessionUpdate === 'user_message_chunk' && (u._meta as Json | undefined)?.promptIndex !== undefined,
      );
      const promptText = (u: Json) => ((u.content as Json).text as string) ?? '';
      expect(firstGoal).toBeLessThan(updates.indexOf(prompts[0]!));
      expect(promptText(prompts[0]!).startsWith(`<system-reminder>\nA goal has been set: ${OBJECTIVE}\n`)).toBe(true);
      const resumeRules = prompts.find((u) => promptText(u).endsWith('\nContinue working now.\n</system-reminder>'));
      expect(resumeRules).toBeDefined();
      // With no helper running the agent sends no tick: every live goal line is stored, in order.
      const statuses = (lines: Json[]) => lines.map((u) => [u.status, u.last_event ?? null]);
      expect(statuses(liveGoalLines(real, sessionId))).toEqual(
        statuses(updates.filter((u) => u.sessionUpdate === 'goal_updated')),
      );

      // One goal row, read the same from disk and live; the stored rules read as what was typed.
      const disk = parseUpdates(text).transcript;
      const typed = disk.flatMap((e) => (e.kind === 'user' && e.interjection !== true ? [e.text] : []));
      expect(typed).toEqual([TYPED, '/goal resume', '/goal clear']);
      expect(disk.flatMap((e) => (e.kind === 'user' && e.interjection === true ? [e.text] : []))).toEqual([STEER]);
      expect(goals(disk).map(shown)).toEqual([
        { goalId: expect.any(String), objective: OBJECTIVE, status: 'cleared', phase: 'idle' },
      ]);
      const row = disk.findIndex((e) => e.kind === 'goal');
      expect(disk[row - 1]).toMatchObject({ kind: 'user', text: TYPED });
      const live = liveEntries(real, sessionId, [TYPED, '/goal resume']);
      expect(goals(live).map(shown)).toEqual(goals(disk).map(shown));
    } finally {
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  }, 180_000);
});

/**
 * Questions on the real agent binary (`MLX_AGENT_E2E_BIN`) with a scripted engine and no model: Work offers the tool,
 * the agent's `_x.ai/ask_user_question` opens a card, the page's indexes go back as the agent's own strings, the model
 * gets the agent's sentence, and the reducer gives the same `ask` from the live lines and from `updates.jsonl`. Skip and
 * Stop end with "no answer". A picked option's preview goes back word for word; in plan mode Talk it over and Plan now
 * go back as the agent's two plan replies. Skipped without the variable.
 */

import { mkdirSync, mkdtempSync, readdirSync, readFileSync, realpathSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { parseUpdates, TranscriptReducer } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { AGENT_ARGS } from '../src/chat/agent/profiles.js';
import type { QuestionInteraction } from '../src/chat/types.js';
import { fakeHost, noProviders, until, type FakeHost, type Json } from './helpers/chat-fake.js';
import {
  hasToolResult,
  lastMessage,
  messageText,
  startScriptedEngine,
  type ModelReply,
} from './helpers/scripted-engine.js';

const BIN = process.env.MLX_AGENT_E2E_BIN;
const MODEL = 'qwen3.5-4b';
const AGENT_WAIT_MS = 30_000;
const CANCEL_TEXT =
  'User declined to answer the questions. Continue with the task using your best judgment, or ask different questions.';

type Tool = Extract<TranscriptEntry, { kind: 'tool' }>;

/** The model's input, as the agent's tool schema takes it (`multi_select`). */
const TWO: Json = {
  questions: [
    {
      question: 'Which database should the cart use?',
      options: [
        { label: 'Postgres (Recommended)', description: 'Relational.' },
        { label: 'Redis', description: 'In memory.' },
      ],
    },
    {
      question: 'Which checks should run before a merge?',
      options: [
        { label: 'Unit tests', description: 'Fast.' },
        { label: 'Type check', description: '' },
        { label: 'Lint', description: '' },
      ],
      multi_select: true,
    },
  ],
};
const ONE: Json = {
  questions: [{ question: 'Deploy after merge?', options: [{ label: 'Yes', description: '' }] }],
};
const SIDEBAR = '┌──────┬─────────┐\n│ nav  │ content │\n└──────┴─────────┘';
/** A single-select question with previews, then a multi-select one. */
const LAYOUT: Json = {
  questions: [
    {
      question: 'Which layout should the shop use?',
      options: [
        { label: 'Sidebar layout', description: 'Nav on the left.', preview: SIDEBAR },
        { label: 'Top bar layout', description: 'Nav on top.', preview: 'nav\ncontent' },
      ],
    },
    {
      question: 'Which checks should run before a merge?',
      options: [
        { label: 'Unit tests', description: 'Fast.' },
        { label: 'Type check', description: '' },
        { label: 'Lint', description: '' },
      ],
      multi_select: true,
    },
  ],
};

function liveAsks(fake: FakeHost, sessionId: string): Tool[] {
  const reducer = new TranscriptReducer();
  for (const u of fake.events.updates) if (u.sessionId === sessionId) reducer.apply(u.line);
  return reducer.snapshot().entries.filter((e): e is Tool => e.kind === 'tool' && e.name === 'ask_user_question');
}

function storedAsks(agentHome: string, marker: string): Tool[] {
  const files = readdirSync(agentHome, { recursive: true, encoding: 'utf8' })
    .filter((f) => f.endsWith('updates.jsonl'))
    .map((f) => readFileSync(join(agentHome, f), 'utf8'))
    .filter((text) => text.includes(marker));
  expect(files).toHaveLength(1);
  return parseUpdates(files[0]!).transcript.filter(
    (e): e is Tool => e.kind === 'tool' && e.name === 'ask_user_question',
  );
}

async function card(fake: FakeHost, n: number): Promise<QuestionInteraction> {
  await until(() => fake.events.interactions.length >= n, AGENT_WAIT_MS);
  const interaction = fake.events.interactions[n - 1]!.interaction;
  expect(interaction.kind).toBe('question');
  return interaction as QuestionInteraction;
}

describe.skipIf(BIN === undefined)('Questions on the real agent', () => {
  it('Work offers the tool; answer by index, Skip and Stop; the same ask live and from disk', async () => {
    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-ask-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    let offered: string[] = [];
    let system = '';
    /** The text the model got back from each question call, in order. */
    const results: string[] = [];
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: () =>
        startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q): ModelReply => {
          if (q.tools.length === 1 && q.tools[0] === 'session_title') {
            return { blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Ask probe' } }] };
          }
          offered = q.tools;
          system = q.system;
          const last = lastMessage(q);
          if (hasToolResult(last)) {
            results.push(messageText(last));
            return { blocks: [{ type: 'text', text: 'Went on.' }] };
          }
          const input = messageText(last).includes('two questions') ? TWO : ONE;
          return { blocks: [{ type: 'tool_use', name: 'ask_user_question', input }] };
        }),
      env: { TMPDIR: `${agentTmp}/` },
    });
    try {
      const { sessionId } = await real.host.createSession({ modelId: MODEL, cwd: real.project });

      // 1. Two questions, answered by index: a pick, picks with own text.
      const first = real.host.startTurn(sessionId, 'Ask me two questions.', { modelId: MODEL, cwd: real.project });
      const q1 = await card(real, 1);
      expect(offered).toContain('ask_user_question');
      expect(system).toContain('Ask the user a question only when a wrong guess would waste real work.');
      expect(q1).toMatchObject({
        kind: 'question',
        planMode: false,
        questions: [
          {
            text: 'Which database should the cart use?',
            multi: false,
            options: [
              { label: 'Postgres (Recommended)', description: 'Relational.', preview: null },
              { label: 'Redis', description: 'In memory.', preview: null },
            ],
          },
          { text: 'Which checks should run before a merge?', multi: true },
        ],
      });
      real.host.answerQuestion(sessionId, q1.id, { answers: [{ picks: [1] }, { picks: [2, 0], other: 'and e2e' }] });
      expect((await first.done).outcome).toBe('completed');
      expect(results[0]).toBe(
        'User has answered your questions: "Which database should the cart use?"="Redis", ' +
          '"Which checks should run before a merge?"="Unit tests, Lint" user notes: and e2e. ' +
          "You can now continue with the user's answers in mind.",
      );
      const live1 = liveAsks(real, sessionId);
      expect(live1).toHaveLength(1);
      expect(live1[0]!.id).toBe(q1.toolCallId);
      expect(live1[0]!.ask).toEqual({
        questions: [
          { text: 'Which database should the cart use?', multi: false, options: ['Postgres (Recommended)', 'Redis'] },
          {
            text: 'Which checks should run before a merge?',
            multi: true,
            options: ['Unit tests', 'Type check', 'Lint'],
          },
        ],
        result: {
          kind: 'answered',
          answers: [
            { picks: ['Redis'], other: null },
            { picks: ['Unit tests', 'Lint'], other: 'and e2e' },
          ],
        },
      });

      // 2. Skip: "no answer".
      const second = real.host.startTurn(sessionId, 'Ask me one question.', { modelId: MODEL, cwd: real.project });
      const q2 = await card(real, 2);
      real.host.answerQuestion(sessionId, q2.id, { reply: 'skip' });
      expect((await second.done).outcome).toBe('completed');
      expect(results[1]).toBe(CANCEL_TEXT);
      expect(liveAsks(real, sessionId)[1]!.ask?.result).toEqual({ kind: 'none' });

      // 3. Stop while the card is open: the card closes as cancelled.
      const third = real.host.startTurn(sessionId, 'Ask me one question.', { modelId: MODEL, cwd: real.project });
      const q3 = await card(real, 3);
      expect(real.host.cancel(sessionId)).toBe(true);
      expect((await third.done).outcome).toBe('cancelled');
      await until(() => real.events.closed.some((c) => c.id === q3.id), AGENT_WAIT_MS);
      expect(real.events.closed.find((c) => c.id === q3.id)?.reason).toBe('cancelled');

      const live = liveAsks(real, sessionId);
      expect(live).toHaveLength(3);
      // A stopped call never ends: no result line live or on disk, so its row says "not finished" in both.
      expect(live[2]!.status).toBe('pending');
      expect(live[2]!.ask?.result).toBeNull();
      expect(results).toHaveLength(2);
      await until(() => storedAsks(real.agentHome, 'Ask me two questions.').length === 3, AGENT_WAIT_MS);
      const stored = storedAsks(real.agentHome, 'Ask me two questions.');
      expect(stored.slice(0, 2)).toEqual(live.slice(0, 2));
      const view = (tools: Tool[]) => tools.map((t) => ({ id: t.id, status: t.status, ask: t.ask }));
      expect(view(stored)).toEqual(view(live));
    } finally {
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  }, 120_000);

  it('a preview goes back word for word; in plan mode Talk it over and Plan now are the agent plan replies', async () => {
    const outer = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-e2e-ask-plan-')));
    const agentTmp = join(outer, 'tmp');
    mkdirSync(agentTmp);
    const results: string[] = [];
    const offered: string[][] = [];
    const real = await fakeHost('chat-harness', {
      mode: 'work',
      providers: noProviders(),
      launch: { command: BIN!, args: [...AGENT_ARGS], source: 'env', tag: null, versionVerified: false },
      startEngine: () =>
        startScriptedEngine([{ id: MODEL, contextWindow: 262_144 }], (q): ModelReply => {
          if (q.tools.length === 1 && q.tools[0] === 'session_title') {
            return {
              blocks: [{ type: 'tool_use', name: 'session_title', input: { session_title: 'Ask plan probe' } }],
            };
          }
          const last = lastMessage(q);
          if (hasToolResult(last)) {
            results.push(messageText(last));
            return { blocks: [{ type: 'text', text: 'Went on.' }] };
          }
          offered.push(q.tools);
          return { blocks: [{ type: 'tool_use', name: 'ask_user_question', input: LAYOUT }] };
        }),
      env: { TMPDIR: `${agentTmp}/` },
    });
    try {
      const { sessionId } = await real.host.createSession({ modelId: MODEL, cwd: real.project });
      const load = { modelId: MODEL, cwd: real.project };

      // 1. Not in plan mode: a pick with a preview, own text alone.
      const first = real.host.startTurn(sessionId, 'PREVIEW-PROBE: ask me.', load);
      const q1 = await card(real, 1);
      expect(q1.planMode).toBe(false);
      expect(q1.questions[0]!.options.map((o) => o.preview)).toEqual([SIDEBAR, 'nav\ncontent']);
      expect(q1.questions[1]!.options.map((o) => o.preview)).toEqual([null, null, null]);
      expect(() => real.host.answerQuestion(sessionId, q1.id, { reply: 'discuss' })).toThrow();
      real.host.answerQuestion(sessionId, q1.id, { answers: [{ picks: [0] }, { picks: [], other: 'e2e' }] });
      expect((await first.done).outcome).toBe('completed');
      expect(results[0]).toBe(
        'User has answered your questions: "Which layout should the shop use?"="Sidebar layout" selected preview:\n' +
          `${SIDEBAR}, "Which checks should run before a merge?"="Other" user notes: e2e. ` +
          "You can now continue with the user's answers in mind.",
      );

      // 2. Plan mode, Talk it over with the answers so far: labels only, notes dropped.
      const second = real.host.startTurn(sessionId, 'DISCUSS-PROBE: ask me.', load, [], { plan: true });
      const q2 = await card(real, 2);
      expect(q2.planMode).toBe(true);
      real.host.answerQuestion(sessionId, q2.id, {
        reply: 'discuss',
        answers: [{ picks: [1] }, { picks: [2, 0], other: 'and e2e' }],
      });
      expect((await second.done).outcome).toBe('completed');
      expect(results[1]).toBe(
        'The user wants to clarify these questions.\n' +
          '    This means they may have additional information, context or questions for you.\n' +
          '    Take their response into account and then reformulate the questions if appropriate.\n' +
          '    Start by asking them what they would like to clarify.\n\n' +
          '    Questions asked:\n' +
          '- "Which layout should the shop use?"\n  Answer: Top bar layout\n' +
          '- "Which checks should run before a merge?"\n  Answer: Unit tests, Lint',
      );

      // 3. Plan mode, Plan now with nothing answered.
      const third = real.host.startTurn(sessionId, 'PLANNOW-PROBE: ask me.', load, [], { plan: true });
      const q3 = await card(real, 3);
      expect(q3.planMode).toBe(true);
      real.host.answerQuestion(sessionId, q3.id, { reply: 'plan-now', answers: [null, null] });
      expect((await third.done).outcome).toBe('completed');
      expect(results[2]).toBe(
        'The user has indicated they have provided enough answers for the plan interview.\n' +
          'Stop asking clarifying questions and proceed to finish the plan with the information you have.\n\n' +
          'Questions asked and answers provided:\n' +
          '- "Which layout should the shop use?"\n  (No answer provided)\n' +
          '- "Which checks should run before a merge?"\n  (No answer provided)',
      );
      expect(offered.every((tools) => tools.includes('ask_user_question'))).toBe(true);

      const live = liveAsks(real, sessionId);
      expect(live.map((t) => t.ask?.result)).toEqual([
        {
          kind: 'answered',
          answers: [
            { picks: ['Sidebar layout'], other: null },
            { picks: [], other: 'e2e' },
          ],
        },
        { kind: 'discuss' },
        { kind: 'plan-now' },
      ]);
      await until(() => storedAsks(real.agentHome, 'PREVIEW-PROBE').length === 3, AGENT_WAIT_MS);
      const stored = storedAsks(real.agentHome, 'PREVIEW-PROBE');
      const view = (tools: Tool[]) => tools.map((t) => ({ id: t.id, status: t.status, ask: t.ask }));
      expect(view(stored)).toEqual(view(live));
    } finally {
      await real.close();
      rmSync(outer, { recursive: true, force: true });
    }
  }, 120_000);
});

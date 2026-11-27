/**
 * Questions of the agent (Work): the fake agent's `_x.ai/ask_user_question` becomes a card, the page's indexes go back
 * as the agent's own strings, and every way the card ends without an answer gives "no answer" (`cancelled`).
 */

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vite-plus/test';

import { parseUpdates, TranscriptReducer } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { ASKS_PER_TURN_MAX, InteractionError } from '../src/chat/agent/agent-host.js';
import { REDACTED } from '../src/chat/redact.js';
import { interactionInput } from '../src/chat/route-input.js';
import type { ChatService } from '../src/chat/service.js';
import type { ConversationEvent, Interaction, QuestionInteraction } from '../src/chat/types.js';
import { applyConversationEvent, type ConversationState } from '../ui/src/lib/chat-store.js';
import {
  fakeHost,
  fakeService,
  noProviders,
  storedWhen,
  until,
  type FakeHost,
  type FakeHostOptions,
  type Json,
} from './helpers/chat-fake.js';

const MODEL = 'qwen3.5-4b';
const KEY = 'exa-KEY-0123456789abcdef';
const CANCELLED = { outcome: 'cancelled' };

const Q_DB = `Which database should the cart use? The old one used ${KEY}.`;
const Q_CHECKS = 'Which checks should run before a merge?';
const Q_MORE = 'Anything else?';

/** A question step of the fake agent: one single-select, one multi-select, one with no options. */
function threeQuestions(extra: Json = {}): Json {
  return {
    question: {
      id: 'toolu_ask',
      questions: [
        {
          question: Q_DB,
          options: [
            { label: 'Postgres (Recommended)', description: 'Relational. Already used by orders.' },
            { label: 'Redis', description: 'In memory. Lost on restart.' },
          ],
        },
        {
          question: Q_CHECKS,
          multi: true,
          options: [
            { label: 'Unit tests', description: 'Fast.' },
            { label: 'Type check', description: 'tsc over all packages.' },
            { label: 'Lint', description: '' },
          ],
        },
        { question: Q_MORE, options: [] },
      ],
      ...extra,
    },
  };
}

function oneQuestion(text: string, extra: Json = {}): Json {
  return { question: { questions: [{ question: text, options: [{ label: 'Yes', description: '' }] }], ...extra } };
}

/** The first prompt runs `steps`, then says a line and ends. */
function steps(...first: Json[]): (scenario: Json) => void {
  return (scenario) => {
    const prompt = (scenario.prompts as Json[])[0]!;
    prompt.steps = [...first, { text: 'Went on.' }, { end: { stopReason: 'end_turn' } }];
  };
}

async function withWork(run: (fake: FakeHost) => Promise<void>, options: FakeHostOptions = {}) {
  const fake = await fakeHost('work-permission', { mode: 'work', providers: noProviders(), ...options });
  try {
    await run(fake);
  } finally {
    await fake.close();
  }
}

async function session(fake: FakeHost): Promise<string> {
  return (await fake.host.createSession({ modelId: MODEL, cwd: fake.project })).sessionId;
}

function start(fake: FakeHost, id: string, text = 'go') {
  return fake.host.startTurn(id, text, { modelId: MODEL, cwd: fake.project });
}

async function card(fake: FakeHost, n = 1): Promise<QuestionInteraction> {
  await until(() => fake.events.interactions.length >= n);
  const interaction = fake.events.interactions[n - 1]!.interaction;
  expect(interaction.kind).toBe('question');
  return interaction as QuestionInteraction;
}

/** What the fake agent got back for each question call, in order. */
function replies(fake: FakeHost): Array<{ answer: unknown; late: boolean }> {
  return fake
    .records()
    .filter((r) => r.kind === 'question-answer')
    .map((r) => ({ answer: r.answer, late: r.late as boolean }));
}

type Tool = Extract<TranscriptEntry, { kind: 'tool' }>;

function liveAsk(fake: FakeHost, sessionId: string): Tool[] {
  const reducer = new TranscriptReducer();
  for (const u of fake.events.updates) if (u.sessionId === sessionId) reducer.apply(u.line);
  return reducer.snapshot().entries.filter((e): e is Tool => e.kind === 'tool' && e.name === 'ask_user_question');
}

/** The stored question rows once the line {@link steps} ends with is on disk. */
async function storedAsk(fake: FakeHost): Promise<Tool[]> {
  const files = await storedWhen(
    fake.agentHome,
    () => true,
    (text) => text.includes('"text":"Went on."'),
  );
  expect(files).toHaveLength(1);
  return parseUpdates(files[0]!).transcript.filter(
    (e): e is Tool => e.kind === 'tool' && e.name === 'ask_user_question',
  );
}

function notOffered(run: () => void): void {
  let caught: unknown;
  try {
    run();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(InteractionError);
  expect((caught as InteractionError).kind).toBe('not-offered');
}

describe('a question of the agent (Work)', () => {
  const dirs: string[] = [];
  afterEach(() => {
    for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  it('opens a card with the texts redacted; the answer goes back by index in the agent own strings', async () => {
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const before = Date.now();
        const turn = start(fake, id);
        const q = await card(fake);
        expect(q.toolCallId).toBe('toolu_ask');
        expect(q.openedAt).toBeGreaterThanOrEqual(before);
        expect(q.planMode).toBe(false);
        expect(q.from).toBeUndefined();
        expect(q.questions.map((x) => [x.multi, x.options.map((o) => o.label)])).toEqual([
          [false, ['Postgres (Recommended)', 'Redis']],
          [true, ['Unit tests', 'Type check', 'Lint']],
          [false, []],
        ]);
        expect(q.questions[0]!.text).toContain(REDACTED);
        expect(q.questions[0]!.text).not.toContain(KEY);
        expect(q.questions[0]!.options[0]).toEqual({
          label: 'Postgres (Recommended)',
          description: 'Relational. Already used by orders.',
          preview: null,
        });
        expect(fake.host.active()).toMatchObject([{ sessionId: id, waiting: true, asks: 1 }]);
        // The live signals stay with the host.
        expect(
          fake.events.updates.filter((u) =>
            ['pending_interaction', 'interaction_resolved'].includes(String(u.line.update.sessionUpdate)),
          ),
        ).toEqual([]);

        fake.host.answerQuestion(id, q.id, {
          answers: [{ picks: [0] }, { picks: [2, 0], other: '  also the e2e smoke test ' }, { picks: [], other: 'no' }],
        });
        expect((await turn.done).outcome).toBe('completed');
        expect(fake.events.closed).toEqual([{ sessionId: id, id: q.id, reason: 'answered' }]);
        expect(replies(fake)).toEqual([
          {
            late: false,
            answer: {
              outcome: 'accepted',
              answers: { [Q_DB]: ['Postgres (Recommended)'], [Q_CHECKS]: ['Unit tests', 'Lint'], [Q_MORE]: ['Other'] },
              annotations: { [Q_CHECKS]: { notes: 'also the e2e smoke test' }, [Q_MORE]: { notes: 'no' } },
            },
          },
        ]);

        const live = liveAsk(fake, id);
        expect(live).toHaveLength(1);
        expect(live[0]!.ask).toEqual({
          questions: [
            { text: Q_DB, multi: false, options: ['Postgres (Recommended)', 'Redis'] },
            { text: Q_CHECKS, multi: true, options: ['Unit tests', 'Type check', 'Lint'] },
            { text: Q_MORE, multi: false, options: [] },
          ],
          result: {
            kind: 'answered',
            answers: [
              { picks: ['Postgres (Recommended)'], other: null },
              { picks: ['Unit tests', 'Lint'], other: 'also the e2e smoke test' },
              { picks: [], other: 'no' },
            ],
          },
        });
        expect(await storedAsk(fake)).toEqual(live);
      },
      { providers: noProviders({ secretNeedles: () => [KEY] }), mutate: steps(threeQuestions()) },
    );
  });

  it('an answer that does not fit the card is not offered and the card stays; Skip sends "no answer"', async () => {
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const q = await card(fake);
        const answer = (answers: Array<{ picks: number[]; other?: string } | null>) => () =>
          fake.host.answerQuestion(id, q.id, { answers });
        notOffered(answer([{ picks: [2] }, null, null]));
        notOffered(answer([{ picks: [0, 1] }, null, null]));
        notOffered(answer([{ picks: [0], other: 'and this' }, null, null]));
        notOffered(answer([null, { picks: [1, 1] }, null]));
        notOffered(answer([null, null, { picks: [], other: '   ' }]));
        notOffered(answer([{ picks: [0] }]));
        notOffered(() => fake.host.answerQuestion(id, q.id, { answers: [null, null, null] }));
        // A permission body on a question card is no answer to it.
        expect(() => fake.host.answer(id, q.id, 'allow-once')).toThrow(InteractionError);
        expect(fake.host.openInteractions(id)).toHaveLength(1);

        fake.host.answerQuestion(id, q.id, { reply: 'skip' });
        expect((await turn.done).outcome).toBe('completed');
        expect(replies(fake)).toEqual([{ answer: CANCELLED, late: false }]);
        expect(liveAsk(fake, id)[0]!.ask?.result).toEqual({ kind: 'none' });
      },
      { mutate: steps(threeQuestions()) },
    );
  });

  it('Chat never shows a card: the request is answered "no answer"', async () => {
    const fake = await fakeHost('chat-basic', {
      mode: 'chat',
      providers: noProviders(),
      mutate: (scenario) => {
        const prompt =
          (scenario.prompts as Json[]).find((p) => (p.match ?? '') === '') ?? (scenario.prompts as Json[])[0]!;
        prompt.steps = [oneQuestion('Postgres?'), { text: 'ok' }, { end: { stopReason: 'end_turn' } }];
      },
    });
    try {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const turn = fake.host.startTurn(sessionId, 'hello', { modelId: MODEL });
      expect((await turn.done).outcome).toBe('completed');
      expect(fake.events.interactions).toEqual([]);
      expect(replies(fake)).toEqual([{ answer: CANCELLED, late: false }]);
    } finally {
      await fake.close();
    }
  });

  it('Stop closes the card as cancelled and the agent gets "no answer"', async () => {
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const q = await card(fake);
        expect(fake.host.cancel(id)).toBe(true);
        expect((await turn.done).outcome).toBe('cancelled');
        expect(fake.events.closed).toEqual([{ sessionId: id, id: q.id, reason: 'cancelled' }]);
        expect(replies(fake)).toEqual([{ answer: CANCELLED, late: false }]);
        expect(liveAsk(fake, id)[0]!.ask?.result).toBeNull();
      },
      { mutate: steps(oneQuestion('Postgres?')) },
    );
  });

  it('the agent giving up (its time limit) closes the card as expired; the call ends with no answer', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-ask-gate-'));
    dirs.push(dir);
    const gate = join(dir, 'give-up');
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const q = await card(fake);
        writeFileSync(gate, '');
        await until(() => fake.events.closed.length === 1);
        expect(fake.events.closed).toEqual([{ sessionId: id, id: q.id, reason: 'expired' }]);
        expect(fake.host.openInteractions(id)).toEqual([]);
        expect((await turn.done).outcome).toBe('completed');
        await until(() => replies(fake).length === 1);
        expect(replies(fake)).toEqual([{ answer: CANCELLED, late: true }]);
        expect(liveAsk(fake, id)[0]!.ask?.result).toEqual({ kind: 'none' });
      },
      { mutate: steps(oneQuestion('Postgres?', { giveUpOn: gate })) },
    );
  });

  it(`the call after ${ASKS_PER_TURN_MAX} in one answer is answered "no answer" with no card`, async () => {
    const calls = Array.from({ length: ASKS_PER_TURN_MAX + 1 }, (_, i) => oneQuestion(`Question ${i + 1}?`));
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        for (let n = 1; n <= ASKS_PER_TURN_MAX; n += 1) {
          const q = await card(fake, n);
          fake.host.answerQuestion(id, q.id, { answers: [{ picks: [0] }] });
        }
        expect((await turn.done).outcome).toBe('completed');
        expect(fake.events.interactions).toHaveLength(ASKS_PER_TURN_MAX);
        expect(replies(fake).map((r) => r.answer)).toEqual([
          ...Array.from({ length: ASKS_PER_TURN_MAX }, (_, i) => ({
            outcome: 'accepted',
            answers: { [`Question ${i + 1}?`]: ['Yes'] },
          })),
          CANCELLED,
        ]);
        // A new answer counts again.
        const next = start(fake, id);
        for (let n = ASKS_PER_TURN_MAX + 1; n <= 2 * ASKS_PER_TURN_MAX; n += 1) {
          const q = await card(fake, n);
          fake.host.answerQuestion(id, q.id, { reply: 'skip' });
        }
        expect((await next.done).outcome).toBe('completed');
        expect(fake.events.interactions).toHaveLength(2 * ASKS_PER_TURN_MAX);
        expect(replies(fake).map((r) => r.answer)).toHaveLength(2 * ASKS_PER_TURN_MAX + 2);
        expect(replies(fake).at(-1)!.answer).toEqual(CANCELLED);
      },
      { mutate: steps(...calls) },
    );
  }, 20_000);

  it('a request over the card limits opens no card: "no answer" and a conversation notice', async () => {
    const many = {
      question: {
        questions: Array.from({ length: 11 }, (_, i) => ({ question: `Q${i}?`, options: [] })),
      },
    };
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        expect((await turn.done).outcome).toBe('completed');
        expect(fake.events.interactions).toEqual([]);
        expect(replies(fake)).toEqual([{ answer: CANCELLED, late: false }]);
        expect(fake.events.notices).toEqual([
          { sessionId: id, notice: { code: 'question-too-long', tone: 'warning' } },
        ]);
      },
      { mutate: steps(many) },
    );
  });

  it('with no window the card is withdrawn and the agent gets "no answer"', async () => {
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const q = await card(fake);
        fake.host.denyOpen();
        expect((await turn.done).outcome).toBe('completed');
        expect(fake.events.closed).toEqual([{ sessionId: id, id: q.id, reason: 'expired' }]);
        expect(replies(fake)).toEqual([{ answer: CANCELLED, late: false }]);
      },
      { mutate: steps(oneQuestion('Postgres?')) },
    );
  });

  it('in plan mode the card says so', async () => {
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const q = await card(fake);
        expect(q.planMode).toBe(true);
        fake.host.answerQuestion(id, q.id, { reply: 'skip' });
        expect((await turn.done).outcome).toBe('completed');
      },
      { mutate: steps({ planMode: 'plan' }, oneQuestion('Postgres?')) },
    );
  });

  it('plan mode: Talk it over and Plan now send the answers so far; the row reads them the same live and from disk', async () => {
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const first = await card(fake, 1);
        expect(first.planMode).toBe(true);
        // Not on a card that is not in plan mode, and not with answers that do not fit it.
        notOffered(() =>
          fake.host.answerQuestion(id, first.id, { reply: 'discuss', answers: [{ picks: [0, 1] }, null, null] }),
        );
        fake.host.answerQuestion(id, first.id, {
          reply: 'discuss',
          answers: [{ picks: [1] }, { picks: [2, 0], other: 'and e2e' }, null],
        });
        const second = await card(fake, 2);
        fake.host.answerQuestion(id, second.id, { reply: 'plan-now' });
        expect((await turn.done).outcome).toBe('completed');
        expect(replies(fake)).toEqual([
          {
            late: false,
            answer: {
              outcome: 'chat_about_this',
              partial_answers: { [Q_DB]: 'Redis', [Q_CHECKS]: 'Unit tests, Lint' },
            },
          },
          { late: false, answer: { outcome: 'skip_interview', partial_answers: {} } },
        ]);
        const live = liveAsk(fake, id);
        expect(live.map((t) => t.ask?.result)).toEqual([{ kind: 'discuss' }, { kind: 'plan-now' }]);
        expect(await storedAsk(fake)).toEqual(live);
      },
      {
        providers: noProviders({ secretNeedles: () => [KEY] }),
        mutate: steps({ planMode: 'plan' }, threeQuestions(), oneQuestion('Ship it?')),
      },
    );
  });

  it('not in plan mode the plan replies are not offered', async () => {
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const q = await card(fake);
        expect(q.planMode).toBe(false);
        notOffered(() => fake.host.answerQuestion(id, q.id, { reply: 'discuss' }));
        notOffered(() => fake.host.answerQuestion(id, q.id, { reply: 'plan-now', answers: [{ picks: [0] }] }));
        expect(fake.host.openInteractions(id)).toHaveLength(1);
        fake.host.answerQuestion(id, q.id, { reply: 'skip' });
        expect((await turn.done).outcome).toBe('completed');
        expect(replies(fake)).toEqual([{ answer: CANCELLED, late: false }]);
      },
      { mutate: steps(oneQuestion('Postgres?')) },
    );
  });

  it("a picked option's preview goes back word for word; the row reads the pick the same live and from disk", async () => {
    const preview = `┌──────┬─────────┐\n│ nav  │ content │\n└──────┴─────────┘ uses ${KEY}`;
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const q = await card(fake);
        expect(q.questions[0]!.options[0]!.preview).toBe(
          `┌──────┬─────────┐\n│ nav  │ content │\n└──────┴─────────┘ uses ${REDACTED}`,
        );
        expect(q.questions[0]!.options[1]!.preview).toBeNull();
        fake.host.answerQuestion(id, q.id, { answers: [{ picks: [0] }] });
        expect((await turn.done).outcome).toBe('completed');
        expect(replies(fake)).toEqual([
          {
            late: false,
            answer: {
              outcome: 'accepted',
              answers: { 'Which layout?': ['Sidebar layout'] },
              annotations: { 'Which layout?': { preview } },
            },
          },
        ]);
        const live = liveAsk(fake, id);
        expect(live[0]!.ask?.result).toEqual({
          kind: 'answered',
          answers: [{ picks: ['Sidebar layout'], other: null }],
        });
        expect(await storedAsk(fake)).toEqual(live);
      },
      {
        providers: noProviders({ secretNeedles: () => [KEY] }),
        mutate: steps({
          question: {
            questions: [
              {
                question: 'Which layout?',
                options: [
                  { label: 'Sidebar layout', description: 'Nav on the left.', preview },
                  { label: 'Top bar layout', description: 'Nav on top.' },
                ],
              },
            ],
          },
        }),
      },
    );
  });
});

class Page {
  readonly events: ConversationEvent[] = [];
  state: ConversationState | null = null;
  readonly detach: () => void;

  constructor(service: ChatService, id: string) {
    this.detach = service.watchConversation(id, (event) => {
      this.events.push(event);
      const next = applyConversationEvent(this.state, event);
      if (next !== 'resync') this.state = next;
    });
  }

  open(): Interaction[] {
    return this.state?.interactions ?? [];
  }
}

describe('the service and a question card', () => {
  it('the answer route takes the question body; a question does not block Ask → Auto; the notice goes at the next message', async () => {
    const fake = fakeService('work-permission', {
      mutate: (scenario) => {
        scenario.prompts = [
          {
            match: 'ask',
            steps: [oneQuestion('Postgres?'), { text: 'Done.' }, { end: { stopReason: 'end_turn' } }],
          },
          {
            match: 'many',
            steps: [
              { question: { questions: Array.from({ length: 11 }, (_, i) => ({ question: `Q${i}?`, options: [] })) } },
              { text: 'Done.' },
              { end: { stopReason: 'end_turn' } },
            ],
          },
          { match: '', steps: [{ text: 'Done.' }, { end: { stopReason: 'end_turn' } }] },
        ];
      },
    });
    const project = join(fake.root, 'home', 'code', 'app');
    mkdirSync(project, { recursive: true });
    const window = fake.service.watchHost(() => {});
    try {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: 'ask', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.open().length === 1, 5_000);
      const q = page.open()[0]!;
      expect(q.kind).toBe('question');
      expect(await svc.setPermissionMode(id, 'auto', true)).toEqual({ mode: 'auto' });
      await svc.answer(id, q.id, { answers: [{ picks: [0] }] });
      await until(() => page.open().length === 0 && page.state?.turn === null, 5_000);
      expect(fake.records().find((r) => r.kind === 'question-answer')?.answer).toEqual({
        outcome: 'accepted',
        answers: { 'Postgres?': ['Yes'] },
      });

      await svc.prompt(id, 'many');
      await until(() => page.state?.notices.some((n) => n.code === 'question-too-long') === true, 5_000);
      await until(() => page.state?.turn === null, 5_000);
      await svc.prompt(id, 'then');
      await until(() => page.state?.notices.length === 0, 5_000);
      page.detach();
    } finally {
      window();
      await fake.close();
    }
  }, 30_000);

  it('the plan replies go from the route body through the service to the agent', async () => {
    const fake = fakeService('work-permission', {
      mutate: (scenario) => {
        scenario.prompts = [
          {
            match: '',
            steps: [
              { planMode: 'plan' },
              oneQuestion('Plan it?'),
              oneQuestion('Talk first?'),
              { text: 'Done.' },
              { end: { stopReason: 'end_turn' } },
            ],
          },
        ];
      },
    });
    const project = join(fake.root, 'home', 'code', 'app');
    mkdirSync(project, { recursive: true });
    const window = fake.service.watchHost(() => {});
    try {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: 'plan it', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.open().length === 1, 5_000);
      const first = page.open()[0]!;
      expect(first.kind === 'question' && first.planMode).toBe(true);
      await svc.answer(id, first.id, interactionInput({ reply: 'plan-now', answers: [{ picks: [0] }] }));
      await until(() => page.open().length === 1 && page.open()[0]!.id !== first.id, 5_000);
      await svc.answer(id, page.open()[0]!.id, interactionInput({ reply: 'discuss' }));
      await until(() => page.open().length === 0 && page.state?.turn === null, 5_000);
      expect(fake.records().flatMap((r) => (r.kind === 'question-answer' ? [r.answer] : []))).toEqual([
        { outcome: 'skip_interview', partial_answers: { 'Plan it?': 'Yes' } },
        { outcome: 'chat_about_this', partial_answers: {} },
      ]);
      page.detach();
    } finally {
      window();
      await fake.close();
    }
  }, 30_000);
});

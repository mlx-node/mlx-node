/**
 * The goal loop in the Chat service on the scripted fake agent: the strip's presses (Pause is Stop while an answer
 * runs, else each sends the agent's own `/goal pause|resume|clear` through the checks of a typed message), their
 * refusals (Chat, a busy conversation), the queue that waits while the goal runs and goes once it paused or ended,
 * Send now between two rounds, and Rewind refused at or before the row of an open goal.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { ApiError } from '../src/api/errors.js';
import { checkCommand } from '../src/chat/commands.js';
import { routeError } from '../src/chat/route-errors.js';
import type { ChatService } from '../src/chat/service.js';
import type { ConversationEvent, QueueState } from '../src/chat/types.js';
import { applyConversationEvent, type ConversationState } from '../ui/src/lib/chat-store.js';
import { fakeService, until, type FakeService, type Json } from './helpers/chat-fake.js';

const sleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

type Goal = Extract<TranscriptEntry, { kind: 'goal' }>;

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

  ends(): number {
    return this.events.filter((e) => e.type === 'turn' && e.turn === null).length;
  }

  queue(): QueueState {
    return this.state?.queue ?? { items: [], held: null, count: 0 };
  }

  entries(): readonly TranscriptEntry[] {
    return this.state?.entries ?? [];
  }

  goals(): Goal[] {
    return this.entries().filter((e): e is Goal => e.kind === 'goal');
  }

  /** Index of the first entry that matches. */
  at(match: (e: TranscriptEntry) => boolean): number {
    return this.entries().findIndex(match);
  }
}

async function apiError(promise: Promise<unknown>): Promise<ApiError> {
  try {
    await promise;
  } catch (error) {
    return routeError('chat', error);
  }
  throw new Error('expected a failure');
}

/** The texts of every `session/prompt` the agent got, in order. */
function prompts(fake: FakeService): string[] {
  return fake
    .sent()
    .filter((m) => m.method === 'session/prompt')
    .map((m) => (((m.params as Json).prompt as Json[])[0]!.text as string) ?? '');
}

const END = [
  { text: 'Done.', chunk: 5 },
  { turnCompleted: { inputTokens: 10, outputTokens: 2 } },
  { end: { stopReason: 'end_turn' } },
];

async function withWork(run: (fake: FakeService, project: string, gate: string) => Promise<void>): Promise<void> {
  let gate = '';
  const fake = fakeService('work-workflow-run', {
    mutate: (scenario) => {
      gate = join(String(scenario.recordTo), '..', 'gate');
      scenario.prompts = [
        {
          match: '/goal fix',
          goal: { wakeRounds: 2, wakeRoundMs: 60, gapMs: 250, end: 'complete' },
          // A page attaches only after its async reads, so an ungated turn can end before the page counts its end.
          steps: [{ holdOn: gate }, { goalRound: { ms: 30, text: 'Round 0.' } }, ...END],
        },
        {
          match: '/goal stall',
          goal: { after: { ms: 400, status: 'no_progress_paused', pauseMessage: 'No progress in 3 rounds.' } },
          steps: [{ holdOn: gate }, ...END],
        },
        { match: '/goal hold', goal: {}, steps: [{ holdOn: gate }, { goalRound: { ms: 20 } }, ...END] },
        { match: 'hold', steps: [{ holdOn: gate }, ...END] },
        { match: '', steps: END },
      ];
    },
  });
  const project = join(fake.root, 'home', 'code', 'app');
  mkdirSync(project, { recursive: true });
  try {
    await run(fake, project, gate);
  } finally {
    await fake.close();
  }
}

describe('the goal strip presses', () => {
  // Mutation: drop the cancel branch of `goal()` → Pause while the round runs sends `/goal pause`, which waits.
  it('Pause while the goal runs is Stop: the agent pauses the goal; the queue is held as after a Stop', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: '/goal hold the line', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      await svc.prompt(id, 'after the goal', [], undefined, 'queue');
      expect(await svc.goal(id, 'pause')).toEqual({ stopped: true });
      await until(() => page.goals()[0]?.status === 'user_paused', 5_000);
      expect(prompts(fake)).toEqual(['/goal hold the line']);
      expect(page.queue().held).toEqual({ why: 'stopped' });
      expect(page.goals()).toHaveLength(1);
      page.detach();
    });
  }, 20_000);

  // Mutation: send the action word as typed (`pause` with no `/goal`) → the agent gets a plain message.
  it('idle: Resume, Pause and Clear send exactly the agent’s own commands, which the allow rule lets through', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: '/goal hold the line', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      await svc.cancel(id);
      await until(() => page.goals()[0]?.status === 'user_paused' && page.state?.turn === null, 5_000);
      expect(await svc.goal(id, 'resume')).toEqual({ turnId: expect.any(String) });
      await until(() => page.goals()[0]?.status === 'active' && page.ends() === 2, 5_000);
      expect(await svc.goal(id, 'pause')).toEqual({ turnId: expect.any(String) });
      await until(() => page.goals()[0]?.status === 'user_paused' && page.ends() === 3, 5_000);
      expect(await svc.goal(id, 'clear')).toEqual({ turnId: expect.any(String) });
      await until(() => page.goals()[0]?.status === 'cleared' && page.ends() === 4, 5_000);
      expect(prompts(fake)).toEqual(['/goal hold the line', '/goal resume', '/goal pause', '/goal clear']);
      const rows = [{ name: 'goal', kind: 'builtin' as const, description: '', hint: null, qualifiedName: null }];
      for (const text of ['/goal pause', '/goal resume', '/goal clear'])
        expect([text, checkCommand('work', text, rows)]).toEqual([text, 'allowed']);
      // One row for the goal, where it began, right after its message.
      const goal = page.at((e) => e.kind === 'goal');
      expect(page.entries()[goal - 1]).toMatchObject({ kind: 'user', text: '/goal hold the line' });
      expect(page.goals()).toHaveLength(1);
      page.detach();
    });
  }, 20_000);

  // Mutation: send Resume and Clear with `whenBusy: 'queue'` → they wait in the list (202) instead of 409.
  it('Resume and Clear while an answer runs: 409 conversation-busy and nothing sent; Chat: 400', async () => {
    await withWork(async (fake, project, gate) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: '/goal hold the line', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      for (const action of ['resume', 'clear'] as const) {
        expect(await apiError(svc.goal(id, action))).toMatchObject({ status: 409, reason: 'conversation-busy' });
      }
      writeFileSync(gate, '');
      await until(() => page.state?.turn === null && page.ends() === 1, 5_000);
      expect(prompts(fake)).toEqual(['/goal hold the line']);
      page.detach();
    });
    const chat = fakeService('chat-basic');
    try {
      const { conversation } = await chat.service.create({ mode: 'chat', text: 'hello' });
      expect(await apiError(chat.service.goal(conversation.id, 'pause'))).toMatchObject({ status: 400 });
    } finally {
      await chat.close();
    }
  }, 20_000);
});

describe('the queue while a goal runs', () => {
  // Mutation: drop `goalRuns` from the drain's `free()` → the row goes in the gap before the first round.
  it('a queued row waits through the rounds the agent starts by itself and goes once the goal is done', async () => {
    await withWork(async (fake, project, gate) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: '/goal fix the cache', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      expect(await svc.prompt(id, 'after the goal', [], undefined, 'queue')).toEqual({ queued: expect.any(String) });
      writeFileSync(gate, '');
      await until(() => page.ends() === 1, 5_000);
      // The gap before the first round: no answer runs, the goal does.
      await sleep(120);
      expect(prompts(fake)).toEqual(['/goal fix the cache']);
      await until(() => page.ends() === 4, 10_000);
      expect(prompts(fake)).toEqual(['/goal fix the cache', 'after the goal']);
      const wakes = page.entries().filter((e) => e.kind === 'wake');
      expect(wakes.map((e) => (e.kind === 'wake' ? e.cause : null))).toEqual(['goal', 'goal']);
      const lastRound = page.at((e) => e.kind === 'assistant' && e.text.includes('Round 2.'));
      const row = page.at((e) => e.kind === 'user' && e.text === 'after the goal');
      expect(lastRound).toBeGreaterThan(-1);
      expect(row).toBeGreaterThan(lastRound);
      expect(page.goals().map((g) => g.status)).toEqual(['complete']);
      expect(page.queue()).toEqual({ items: [], held: null, count: 0 });
      page.detach();
    });
  }, 20_000);

  // Mutation: drop the drain on a `goal_updated` that leaves the goal not running → the row waits for ever.
  it('the goal pauses with no answer running: the row that waited goes then', async () => {
    await withWork(async (fake, project, gate) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: '/goal stall the build', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      await svc.prompt(id, 'after the pause', [], undefined, 'queue');
      writeFileSync(gate, '');
      await until(() => page.ends() === 1, 5_000);
      await sleep(150);
      expect(prompts(fake)).toEqual(['/goal stall the build']);
      await until(() => page.ends() === 2, 5_000);
      expect(prompts(fake)).toEqual(['/goal stall the build', 'after the pause']);
      expect(page.goals()[0]).toMatchObject({ status: 'no_progress_paused', pauseMessage: 'No progress in 3 rounds.' });
      page.detach();
    });
  }, 20_000);

  // Mutation: drop the goal term of the host's `steerable` → the press in the gap is refused and the row stays.
  it('Send now in the gap before a round goes to the agent', async () => {
    await withWork(async (fake, project, gate) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: '/goal fix the cache', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      const { queued } = (await svc.prompt(id, 'use pnpm', [], undefined, 'queue')) as { queued: string };
      writeFileSync(gate, '');
      await until(() => page.ends() === 1, 5_000);
      expect(page.state?.turn).toBeNull();
      expect(await svc.queueSend(id, queued)).toEqual({ steered: true });
      expect(
        fake
          .sent()
          .filter((m) => m.method === '_x.ai/interject')
          .map((m) => (m.params as Json).text),
      ).toEqual(['use pnpm']);
      expect(page.queue().items).toEqual([]);
      page.detach();
    });
  }, 20_000);
});

describe('rewind and the open goal', () => {
  // Mutation: drop the goal wall of the rewind plan → the plan for the goal's own message is made.
  it('a plan at or before the open goal’s row is refused; once the goal is cleared it is made', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: 'one', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.state?.turn === null && page.entries().some((e) => e.kind === 'assistant'), 5_000);
      await svc.prompt(id, '/goal hold the line');
      await until(() => page.state?.turn?.phase === 'running', 5_000);
      await svc.cancel(id);
      await until(() => page.goals()[0]?.status === 'user_paused' && page.state?.turn === null, 5_000);
      const goalMessage = page.at((e) => e.kind === 'user' && e.text === '/goal hold the line');
      for (const entry of [0, goalMessage]) {
        expect(await apiError(svc.rewindPlan(id, entry, page.entries()[entry]!.ts))).toMatchObject({
          status: 409,
          reason: 'rewind-goal',
        });
      }
      await svc.goal(id, 'clear');
      await until(() => page.goals()[0]?.status === 'cleared' && page.state?.turn === null, 5_000);
      expect(await svc.rewindPlan(id, goalMessage, page.entries()[goalMessage]!.ts)).toMatchObject({
        token: expect.any(String),
      });
      page.detach();
    });
  }, 20_000);
});

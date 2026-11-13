/**
 * The Work permission laws, checked on the scripted fake agent's own record: a tool that needs permission runs only
 * after the page allowed that request (or after an "always" answer the page gave), a Chat session never gets a Work
 * card or a chosen folder, and an option the card did not list is refused without reaching the agent.
 */

import { realpathSync } from 'node:fs';

import { describe, expect, it } from 'vite-plus/test';

import { InteractionError } from '../src/chat/agent/agent-host.js';
import { createConversationInput } from '../src/chat/route-input.js';
import type { PermissionInteraction } from '../src/chat/types.js';
import { fakeHost, noProviders, until, type FakeHost, type FakeHostOptions, type Json } from './helpers/chat-fake.js';

const MODEL = 'qwen3.5-4b';
const ALLOWS = ['allow-once', 'always-allow', 'allow-edits-session'];
const GRANTS = ['always-allow', 'allow-edits-session'];

/** An answer the page gave, and how many record lines existed when it was given. */
interface PageAnswer {
  toolCallId: string;
  optionId: string;
  atRecord: number;
}

async function withWork(name: string, run: (fake: FakeHost) => Promise<void>, options: FakeHostOptions = {}) {
  const fake = await fakeHost(name, { mode: 'work', providers: noProviders(), ...options });
  try {
    await run(fake);
  } finally {
    await fake.close();
  }
}

function pageAnswers(fake: FakeHost, sessionId: string) {
  const log: PageAnswer[] = [];
  return {
    log,
    answer(card: PermissionInteraction, optionId: string): void {
      log.push({ toolCallId: card.toolCallId, optionId, atRecord: fake.records().length });
      fake.host.answer(sessionId, card.id, optionId);
    },
  };
}

async function card(fake: FakeHost, n: number): Promise<PermissionInteraction> {
  await until(() => fake.events.interactions.length >= n);
  const interaction = fake.events.interactions[n - 1]!.interaction;
  expect(interaction.kind).toBe('permission');
  return interaction as PermissionInteraction;
}

function execs(fake: FakeHost): Json[] {
  return fake.records().filter((r) => r.kind === 'exec');
}

/**
 * Every `exec` of the record that no page answer allowed: its request (or the request that gave its grant) must have
 * been answered by the host with an allowing option the page chose for that same tool call, before the tool ran. In
 * Auto or Bypass, a mark of that tool call, done before the reply, allows it with `allow-once` only.
 */
function unallowed(records: readonly Json[], page: readonly PageAnswer[], marks: readonly PageAnswer[] = []): Json[] {
  const allowedBy = (from: number, before: number, requestId: unknown, options: readonly string[]): string | null => {
    const ask = records.findIndex((r, i) => i >= from && i < before && r.kind === 'ask' && r.id === requestId);
    if (ask === -1) return null;
    const reply = records.findIndex((r, i) => {
      const msg = r.msg as Json | undefined;
      return i > ask && i < before && r.kind === 'in' && msg?.method === undefined && msg?.id === requestId;
    });
    if (reply === -1) return null;
    const outcome = ((records[reply]!.msg as Json).result as Json | undefined)?.outcome as Json | undefined;
    const optionId = outcome?.outcome === 'selected' ? (outcome.optionId as string) : null;
    if (optionId === null || !options.includes(optionId)) return null;
    const toolCallId = records[ask]!.toolCallId as string;
    const fromPage = page.some((a) => a.toolCallId === toolCallId && a.optionId === optionId && a.atRecord <= reply);
    const byMode = optionId === 'allow-once' && marks.some((m) => m.toolCallId === toolCallId && m.atRecord <= reply);
    return fromPage || byMode ? toolCallId : null;
  };
  const out: Json[] = [];
  records.forEach((r, i) => {
    if (r.kind !== 'exec') return;
    let from = 0;
    for (let k = i; k >= 0; k--) {
      if (records[k]!.kind === 'start') {
        from = k;
        break;
      }
    }
    const ok =
      r.via !== null
        ? allowedBy(from, i, r.via, ALLOWS) === r.toolCallId
        : r.grantedBy !== null && allowedBy(from, i, r.grantedBy, GRANTS) !== null;
    if (!ok) out.push(r);
  });
  return out;
}

describe('no Work tool runs without an answer from the page', () => {
  it('always: the first call waits for the page; later calls run on the grant the page gave', async () => {
    await withWork('work-always', async (fake) => {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      const page = pageAnswers(fake, sessionId);
      const turn = fake.host.startTurn(sessionId, 'go', { modelId: MODEL, cwd: fake.project });
      const first = await card(fake, 1);
      expect(execs(fake)).toEqual([]);
      page.answer(first, 'always-allow');
      expect((await turn.done).outcome).toBe('completed');
      expect((await fake.host.startTurn(sessionId, 'again', { modelId: MODEL, cwd: fake.project }).done).outcome).toBe(
        'completed',
      );
      expect(execs(fake)).toHaveLength(4);
      expect(unallowed(fake.records(), page.log)).toEqual([]);
    });
  });

  it('allow once and allow all edits: each run follows its own answer', async () => {
    for (const [scenario, option, runs] of [
      ['work-always', 'allow-once', 2],
      ['work-edit', 'allow-edits-session', 2],
    ] as const) {
      await withWork(scenario, async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        const page = pageAnswers(fake, sessionId);
        const turn = fake.host.startTurn(sessionId, 'go', { modelId: MODEL, cwd: fake.project });
        page.answer(await card(fake, 1), option);
        if (option === 'allow-once') page.answer(await card(fake, 2), option);
        expect((await turn.done).outcome).toBe('completed');
        expect(execs(fake)).toHaveLength(runs);
        expect(unallowed(fake.records(), page.log)).toEqual([]);
      });
    }
  });

  it('Deny, Stop and a page that is gone run nothing', async () => {
    for (const end of ['deny', 'stop', 'gone'] as const) {
      await withWork('work-flagged', async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        const page = pageAnswers(fake, sessionId);
        const turn = fake.host.startTurn(sessionId, 'go', { modelId: MODEL, cwd: fake.project });
        const open = await card(fake, 1);
        if (end === 'deny') page.answer(open, 'reject-once');
        else if (end === 'stop') fake.host.cancel(sessionId);
        else fake.host.denyOpen();
        expect((await turn.done).outcome).toBe(end === 'stop' ? 'cancelled' : 'denied');
        expect(execs(fake)).toEqual([]);
      });
    }
  });

  it('the check itself flags a run the page did not allow', () => {
    const records: Json[] = [
      { kind: 'start' },
      { kind: 'ask', id: 0, method: 'session/request_permission', toolCallId: 't1' },
      { kind: 'in', msg: { id: 0, result: { outcome: { outcome: 'selected', optionId: 'allow-once' } } } },
      { kind: 'exec', toolCallId: 't1', via: 0, grantedBy: null },
    ];
    expect(unallowed(records, [{ toolCallId: 't1', optionId: 'allow-once', atRecord: 2 }])).toEqual([]);
    expect(unallowed(records, [])).toHaveLength(1);
    expect(unallowed(records, [{ toolCallId: 't2', optionId: 'allow-once', atRecord: 2 }])).toHaveLength(1);
    expect(unallowed(records, [{ toolCallId: 't1', optionId: 'allow-once', atRecord: 3 }])).toHaveLength(1);
    expect(unallowed(records, [], [{ toolCallId: 't1', optionId: 'allow-once', atRecord: 2 }])).toEqual([]);
    expect(unallowed(records, [], [{ toolCallId: 't1', optionId: 'allow-once', atRecord: 3 }])).toHaveLength(1);
    const always = records.map((r) =>
      r.kind === 'in'
        ? { kind: 'in', msg: { id: 0, result: { outcome: { outcome: 'selected', optionId: 'always-allow' } } } }
        : r,
    );
    expect(unallowed(always, [], [{ toolCallId: 't1', optionId: 'allow-once', atRecord: 2 }])).toHaveLength(1);
  });

  it.each(['auto', 'bypass'] as const)(
    'in %s: every run follows a page answer or a mark made before the reply',
    async (mode) => {
      const marks: PageAnswer[] = [];
      const held: { fake: FakeHost | null } = { fake: null };
      await withWork(
        'work-permission',
        async (fake) => {
          held.fake = fake;
          const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
          const page = pageAnswers(fake, sessionId);
          const turn = fake.host.startTurn(sessionId, 'go', { modelId: MODEL, cwd: fake.project });
          if (mode === 'auto') page.answer(await card(fake, 1), 'allow-once');
          expect((await turn.done).outcome).toBe('completed');
          expect(execs(fake)).toHaveLength(3);
          const byMode = mode === 'auto' ? 2 : 3;
          expect(marks).toHaveLength(byMode);
          expect(unallowed(fake.records(), page.log, marks)).toEqual([]);
          expect(unallowed(fake.records(), page.log)).toHaveLength(byMode);
        },
        {
          permission: { modeOf: () => mode, watched: () => true },
          onModeAllowed: async (_id, mark) => {
            await new Promise((resolve) => setImmediate(resolve));
            marks.push({ toolCallId: mark.toolCallId, optionId: 'allow-once', atRecord: held.fake!.records().length });
            return true;
          },
          mutate: (scenario) => {
            const step = (command: string): Json => ({
              permission: {
                toolName: 'run_terminal_command',
                kind: 'execute',
                title: 'Execute',
                rawInput: { variant: 'Bash', command, description: 'run', is_background: false },
                output: 'ok',
              },
            });
            (scenario.prompts as Json[])[0]!.steps = [
              step('rm a.txt'),
              step('ls'),
              step('cat a.txt'),
              { end: { stopReason: 'end_turn' } },
            ];
          },
        },
      );
    },
  );
});

describe('a Chat session never gets a Work card or a chosen folder', () => {
  it('a shell request in Chat opens no card, runs nothing, and the session stays in the Chat workspace', async () => {
    const fake = await fakeHost('chat-shell-attempt');
    try {
      const { sessionId, cwd } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      const workspace = realpathSync(fake.paths.chatWorkspace);
      expect(cwd).toBe(workspace);
      expect((await fake.host.startTurn(sessionId, 'cat', { modelId: MODEL, cwd: fake.project }).done).outcome).toBe(
        'tool-refused',
      );
      await fake.host.startTurn('01a1cb42-0000-7000-8000-00000000c0de', 'hi', { modelId: MODEL, cwd: fake.project })
        .done;
      const sessions = fake.sent().filter((m) => m.method === 'session/new' || m.method === 'session/load');
      expect(sessions.map((m) => m.method)).toEqual(['session/new', 'session/load']);
      for (const m of sessions) expect((m.params as Json).cwd).toBe(workspace);
      expect(JSON.stringify(fake.sent())).not.toContain(fake.project);
      expect(fake.events.interactions).toEqual([]);
      expect(execs(fake)).toEqual([]);
    } finally {
      await fake.close();
    }
  });

  it('the create route takes a folder only in Work', () => {
    expect(() => createConversationInput({ mode: 'chat', text: 'hi', cwd: '/Users/me/code' })).toThrow(
      'Field "cwd" is only accepted in Work mode.',
    );
    expect(createConversationInput({ mode: 'work', text: 'hi', cwd: '/Users/me/code' })).toMatchObject({
      cwd: '/Users/me/code',
    });
  });
});

describe('an answer must be an option the card listed', () => {
  it('options the agent offered but the card dropped, made-up ids and another card’s ids are refused', async () => {
    await withWork(
      'work-permission',
      async (fake) => {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        const turn = fake.host.startTurn(sessionId, 'go', { modelId: MODEL, cwd: fake.project });
        const open = await card(fake, 1);
        expect(open.options.map((o) => o.optionId)).toEqual(['always-allow', 'allow-once', 'reject-once']);
        for (const optionId of ['enable-always-approve', 'reject-always', 'allow-edits-session', 'yes', '']) {
          let error: unknown = null;
          try {
            fake.host.answer(sessionId, open.id, optionId);
          } catch (caught) {
            error = caught;
          }
          expect(error, optionId).toBeInstanceOf(InteractionError);
          expect((error as InteractionError).kind, optionId).toBe('not-offered');
        }
        expect(fake.host.hasOpenInteractions()).toBe(true);
        expect(fake.sent().filter((m) => m.method === undefined && m.result !== undefined)).toEqual([]);
        expect(execs(fake)).toEqual([]);
        fake.host.answer(sessionId, open.id, 'reject-once');
        expect((await turn.done).outcome).toBe('denied');
        expect(execs(fake)).toEqual([]);
      },
      {
        mutate: (scenario) => {
          const step = ((scenario.prompts as Json[])[0]!.steps as Json[])[0]!.permission as Json;
          step.offerEnableAlwaysApprove = true;
        },
      },
    );
  });
});

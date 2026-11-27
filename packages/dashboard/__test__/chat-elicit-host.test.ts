/**
 * MCP input, Work: the fake agent's `_x.ai/mcp/elicit` becomes a form or a link card in every mode; the page's
 * values are checked against the form and go back as the server's own values; a link's Open page goes back as a bare
 * `accept`, and the server's `elicit_complete` gives one "finished" line, only for a page this app opened; every way
 * the card ends gives the server an answer and leaves one history line; Chat never shows a card. No typed value
 * reaches an event, a history line or an error.
 */

import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vite-plus/test';

import { InteractionError } from '../src/chat/agent/agent-host.js';
import { markOf, readPermissionLines, storedServer } from '../src/chat/permission-log.js';
import { REDACTED } from '../src/chat/redact.js';
import { interactionInput } from '../src/chat/route-input.js';
import type { ChatService } from '../src/chat/service.js';
import type { ConversationEvent, Interaction, McpInteraction } from '../src/chat/types.js';
import { applyConversationEvent, type ConversationState } from '../ui/src/lib/chat-store.js';
import {
  fakeHost,
  fakeService,
  noProviders,
  until,
  type FakeHost,
  type FakeHostOptions,
  type Json,
} from './helpers/chat-fake.js';

const MODEL = 'qwen3.5-4b';
const KEY = 'exa-KEY-0123456789abcdef';
const TYPED = 'TYPED-VALUE-7f3a';
/** A server name with a hidden character: the history keeps it as a code, live and on disk alike. */
const HIDDEN_SERVER = 'git\u202Ehub';
const CANCEL = { outcome: 'cancel' };
const DECLINE = { outcome: 'decline' };

const SCHEMA: Json = {
  type: 'object',
  properties: {
    repo: { type: 'string', title: 'Repository', enum: ['shop-api', 'shop-web'], enumNames: ['Shop API', 'Shop web'] },
    title: { type: 'string', title: 'Title', minLength: 3, description: `Short. Not ${KEY}.` },
    labels: {
      type: 'array',
      title: 'Labels',
      items: {
        anyOf: [
          { const: 'bug', title: 'Bug' },
          { const: 'feature', title: 'Feature' },
        ],
      },
      maxItems: 1,
    },
    priority: { type: 'integer', title: 'Priority', minimum: 1, maximum: 5 },
    mine: { type: 'boolean', title: 'Assign to me', default: true },
    blob: { type: 'object', title: 'Blob' },
  },
  required: ['repo', 'title'],
};

function elicit(extra: Json = {}): Json {
  return {
    elicit: {
      id: 'mcp-elicit-1',
      server: 'github',
      message: `Which repository should the issue go to? Token ${KEY}`,
      schema: SCHEMA,
      ...extra,
    },
  };
}

/** The permission log once it holds `n` `mcp` lines: each line is written after its mark is sent. */
async function mcpLog(fake: { paths: { permissionLog: string } }, id: string, n: number): Promise<string> {
  const path = join(fake.paths.permissionLog, `${id}.jsonl`);
  const text = (): string => (existsSync(path) ? readFileSync(path, 'utf8') : '');
  await until(
    () =>
      text()
        .split('\n')
        .filter((l) => l.includes('"kind":"mcp"')).length === n,
    5_000,
  );
  return text();
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

async function card(fake: FakeHost, n = 1): Promise<McpInteraction> {
  await until(() => fake.events.interactions.length >= n);
  const interaction = fake.events.interactions[n - 1]!.interaction;
  expect(interaction.kind).toBe('mcp-input');
  return interaction as McpInteraction;
}

/** What the fake agent got back for each request, in order. */
function replies(fake: FakeHost): Array<{ answer: unknown; late: boolean }> {
  return fake
    .records()
    .filter((r) => r.kind === 'elicit-answer')
    .map((r) => ({ answer: r.answer, late: r.late as boolean }));
}

function notOffered(run: () => unknown): void {
  let caught: unknown;
  try {
    run();
  } catch (error) {
    caught = error;
  }
  expect(caught).toBeInstanceOf(InteractionError);
  expect((caught as InteractionError).kind).toBe('not-offered');
}

const GOOD = [1, 'Cart loses items', [0], '3', false];

describe('an MCP server asks for input (Work)', () => {
  const dirs: string[] = [];
  afterEach(() => {
    for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  it('opens a form card with its texts redacted; Send goes back as the server own values', async () => {
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const before = Date.now();
        const turn = start(fake, id);
        const c = await card(fake);
        expect(c).toMatchObject({ kind: 'mcp-input', mode: 'form', toolCallId: 'mcp-elicit-1', server: 'github' });
        expect(c.openedAt).toBeGreaterThanOrEqual(before);
        expect(c.from).toBeUndefined();
        expect(c.unusable).toBeUndefined();
        expect(c.message).toContain(REDACTED);
        expect(c.message).not.toContain(KEY);
        expect(c.form!.leftOut).toBe(1);
        expect(c.form!.fields).toEqual([
          {
            name: 'repo',
            title: 'Repository',
            description: null,
            required: true,
            type: 'choice',
            options: [{ label: 'Shop API' }, { label: 'Shop web' }],
            initial: null,
          },
          {
            name: 'title',
            title: 'Title',
            description: `Short. Not ${REDACTED}.`,
            required: true,
            type: 'text',
            format: null,
            minLength: 3,
            maxLength: null,
            initial: '',
          },
          {
            name: 'labels',
            title: 'Labels',
            description: null,
            required: false,
            type: 'choices',
            options: [{ label: 'Bug' }, { label: 'Feature' }],
            minItems: null,
            maxItems: 1,
            initial: [],
          },
          {
            name: 'priority',
            title: 'Priority',
            description: null,
            required: false,
            type: 'integer',
            min: 1,
            max: 5,
            initial: '',
          },
          { name: 'mine', title: 'Assign to me', description: null, required: false, type: 'boolean', initial: true },
        ]);
        expect(fake.host.active()).toMatchObject([{ sessionId: id, waiting: true, asks: 1 }]);
        // The live signals stay with the host.
        expect(
          fake.events.updates.filter((u) =>
            ['pending_interaction', 'interaction_resolved'].includes(String(u.line.update.sessionUpdate)),
          ),
        ).toEqual([]);

        expect(fake.host.answerElicit(id, c.id, { mcp: 'accept', values: GOOD })).toBeNull();
        expect((await turn.done).outcome).toBe('completed');
        expect(fake.events.closed).toEqual([{ sessionId: id, id: c.id, reason: 'answered' }]);
        expect(replies(fake)).toEqual([
          {
            late: false,
            answer: {
              outcome: 'accept',
              content: { repo: 'shop-web', title: 'Cart loses items', labels: ['bug'], priority: 3, mine: false },
            },
          },
        ]);
        expect(fake.events.mcp).toEqual([
          { sessionId: id, closed: { server: 'github', mode: 'form', outcome: 'accept' } },
        ]);
      },
      { providers: noProviders({ secretNeedles: () => [KEY] }), mutate: steps(elicit()) },
    );
  });

  it('values that break the form come back by field and the card stays; a wrong shape is not offered; Decline', async () => {
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const c = await card(fake);
        expect(fake.host.answerElicit(id, c.id, { mcp: 'accept', values: [null, 'ab', [0, 1], '9', true] })).toEqual([
          { code: 'required' },
          { code: 'minLength', n: 3 },
          { code: 'maxItems', n: 1 },
          { code: 'max', n: 5 },
          null,
        ]);
        expect(fake.host.answerElicit(id, c.id, { mcp: 'accept', values: [0, TYPED, [], '2.5', true] })).toEqual([
          null,
          null,
          null,
          { code: 'integer' },
          null,
        ]);
        notOffered(() => fake.host.answerElicit(id, c.id, { mcp: 'accept', values: [0, 'abc'] }));
        notOffered(() => fake.host.answerElicit(id, c.id, { mcp: 'accept', values: [0, 7, [], '', true] }));
        notOffered(() => fake.host.answerElicit(id, c.id, { mcp: 'accept', values: [2, 'abc', [], '', true] }));
        notOffered(() => fake.host.answerElicit(id, c.id, { mcp: 'accept', values: [0, 'abc', [], '', 'yes'] }));
        // A permission body on an MCP card is no answer to it; an MCP body on no card is not found.
        expect(() => fake.host.answer(id, c.id, 'allow-once')).toThrow(InteractionError);
        expect(() => fake.host.answerElicit(id, 'nope', { mcp: 'decline' })).toThrow(InteractionError);
        expect(fake.host.openInteractions(id)).toHaveLength(1);
        expect(replies(fake)).toEqual([]);

        expect(fake.host.answerElicit(id, c.id, { mcp: 'decline' })).toBeNull();
        expect((await turn.done).outcome).toBe('completed');
        expect(replies(fake)).toEqual([{ answer: DECLINE, late: false }]);
        expect(fake.events.mcp.map((m) => m.closed.outcome)).toEqual(['decline']);
        // No typed value reaches an event.
        expect(JSON.stringify(fake.events)).not.toContain(TYPED);
      },
      { mutate: steps(elicit()) },
    );
  });

  it('a form with no schema is an empty form: Send sends empty content', async () => {
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const c = await card(fake);
        expect(c.form).toEqual({ fields: [], leftOut: 0 });
        expect(fake.host.answerElicit(id, c.id, { mcp: 'accept', values: [] })).toBeNull();
        expect((await turn.done).outcome).toBe('completed');
        expect(replies(fake)).toEqual([{ answer: { outcome: 'accept', content: {} }, late: false }]);
      },
      { mutate: steps(elicit({ schema: undefined })) },
    );
  });

  it('a form the app cannot draw still opens a card, with Decline only', async () => {
    const needed = { type: 'object', properties: { blob: { type: 'object' } }, required: ['blob'] };
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const field = await card(fake);
        expect(field.unusable).toBe('field');
        expect(field.form).toBeUndefined();
        notOffered(() => fake.host.answerElicit(id, field.id, { mcp: 'accept', values: [] }));
        fake.host.answerElicit(id, field.id, { mcp: 'decline' });
        const schema = await card(fake, 2);
        expect(schema.unusable).toBe('schema');
        fake.host.answerElicit(id, schema.id, { mcp: 'decline' });
        expect((await turn.done).outcome).toBe('completed');
        expect(replies(fake).map((r) => r.answer)).toEqual([DECLINE, DECLINE]);
      },
      {
        mutate: steps(
          elicit({ id: 'mcp-elicit-a', schema: needed }),
          elicit({ id: 'mcp-elicit-b', schema: { type: 'array' } }),
        ),
      },
    );
  });

  it('a link opens a card with the whole address; Open page goes back as a bare accept; done is one line', async () => {
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const c = await card(fake);
        expect(c).toMatchObject({ kind: 'mcp-input', mode: 'url', toolCallId: 'mcp-elicit-1', server: 'github' });
        expect(c.link).toEqual({
          url: 'https://github.com/login/oauth/authorize?client_id=abc&state=x%20y',
          host: 'github.com',
          secure: true,
        });
        expect(c.form).toBeUndefined();
        expect(c.unusable).toBeUndefined();
        expect(c.message).toContain(REDACTED);
        expect(c.message).not.toContain(KEY);
        expect(fake.host.active()).toMatchObject([{ sessionId: id, waiting: true, asks: 1 }]);
        // A link takes no values.
        notOffered(() => fake.host.answerElicit(id, c.id, { mcp: 'accept', values: [TYPED] }));
        expect(replies(fake)).toEqual([]);

        expect(fake.host.answerElicit(id, c.id, { mcp: 'accept', values: [] })).toBeNull();
        expect((await turn.done).outcome).toBe('completed');
        const [reply] = replies(fake);
        expect(reply).toEqual({ answer: { outcome: 'accept' }, late: false });
        expect(Object.keys(reply!.answer as Json)).toEqual(['outcome']);
        expect(fake.events.closed).toEqual([{ sessionId: id, id: c.id, reason: 'answered' }]);
        expect(fake.events.mcp).toEqual([
          { sessionId: id, closed: { server: 'github', mode: 'url', outcome: 'accept' } },
        ]);
        // The server says it is done twice: one line.
        await until(() => fake.events.mcpDone.length === 1);
        await new Promise((resolve) => setTimeout(resolve, 100));
        expect(fake.events.mcpDone).toEqual([{ sessionId: id, server: 'github' }]);
      },
      {
        providers: noProviders({ secretNeedles: () => [KEY] }),
        mutate: steps(
          elicit({
            mode: 'url',
            message: `Sign in to GitHub. ${KEY}`,
            url: '  https://GitHub.com/login/oauth/authorize?client_id=abc&state=x y',
            elicitationId: 'e-7',
            complete: [{}, {}],
          }),
        ),
      },
    );
  });

  it('done counts only for a page this app opened, from the server that asked', async () => {
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const declined = await card(fake);
        fake.host.answerElicit(id, declined.id, { mcp: 'decline' });
        const opened = await card(fake, 2);
        fake.host.answerElicit(id, opened.id, { mcp: 'accept' });
        const older = await card(fake, 3);
        fake.host.answerElicit(id, older.id, { mcp: 'accept' });
        expect((await turn.done).outcome).toBe('completed');
        expect(replies(fake).map((r) => r.answer)).toEqual([DECLINE, { outcome: 'accept' }, { outcome: 'accept' }]);
        await until(() => fake.events.mcpDone.length === 1);
        await new Promise((resolve) => setTimeout(resolve, 100));
        // Only the one with no server name, as an older agent sends it.
        expect(fake.events.mcpDone).toEqual([{ sessionId: id, server: 'gitlab' }]);
        expect(fake.events.mcp.map((m) => [m.closed.mode, m.closed.outcome])).toEqual([
          ['url', 'decline'],
          ['url', 'accept'],
          ['url', 'accept'],
        ]);
      },
      {
        mutate: steps(
          elicit({ id: 'mcp-elicit-a', mode: 'url', elicitationId: 'e-a', complete: {} }),
          elicit({
            id: 'mcp-elicit-b',
            mode: 'url',
            elicitationId: 'e-b',
            complete: [
              { elicitationId: 'e-a' },
              { elicitationId: 'e-x' },
              { server: 'linear' },
              { sessionId: 'another-session' },
            ],
          }),
          elicit({
            id: 'mcp-elicit-c',
            mode: 'url',
            server: 'gitlab',
            elicitationId: 'e-c',
            complete: { server: null },
          }),
        ),
      },
    );
  });

  it('two servers that open a page under the same id each get their own done', async () => {
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const github = await card(fake);
        fake.host.answerElicit(id, github.id, { mcp: 'accept' });
        const gitlab = await card(fake, 2);
        fake.host.answerElicit(id, gitlab.id, { mcp: 'accept' });
        expect((await turn.done).outcome).toBe('completed');
        await until(() => fake.events.mcpDone.length === 2, 5_000);
        expect(fake.events.mcpDone).toEqual([
          { sessionId: id, server: 'gitlab' },
          { sessionId: id, server: 'github' },
        ]);
      },
      {
        mutate: steps(
          elicit({ id: 'mcp-elicit-a', mode: 'url', elicitationId: 'e-1', complete: { afterMs: 500 } }),
          elicit({ id: 'mcp-elicit-b', mode: 'url', server: 'gitlab', elicitationId: 'e-1', complete: {} }),
        ),
      },
    );
  });

  it('a page opened before the agent process ended gets no done from the next process', async () => {
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const first = start(fake, id, 'first');
        const opened = await card(fake);
        fake.host.answerElicit(id, opened.id, { mcp: 'accept' });
        expect((await first.done).outcome).toBe('completed');
        await fake.host.restart();
        const second = start(fake, id, 'second');
        const declined = await card(fake, 2);
        fake.host.answerElicit(id, declined.id, { mcp: 'decline' });
        const other = await card(fake, 3);
        fake.host.answerElicit(id, other.id, { mcp: 'accept' });
        expect((await second.done).outcome).toBe('completed');
        await until(() => fake.events.mcpDone.length === 1, 5_000);
        await new Promise((resolve) => setTimeout(resolve, 100));
        // The new process's done for its declined "e-1" is not the old process's opened "e-1".
        expect(fake.events.mcpDone).toEqual([{ sessionId: id, server: 'gitlab' }]);
      },
      {
        mutate: (scenario) => {
          const end = [{ text: 'Went on.' }, { end: { stopReason: 'end_turn' } }];
          scenario.prompts = [
            { match: 'first', steps: [elicit({ id: 'mcp-elicit-a', mode: 'url', elicitationId: 'e-1' }), ...end] },
            {
              match: 'second',
              steps: [
                elicit({ id: 'mcp-elicit-b', mode: 'url', elicitationId: 'e-1', complete: {} }),
                elicit({ id: 'mcp-elicit-c', mode: 'url', server: 'gitlab', elicitationId: 'e-2', complete: {} }),
                ...end,
              ],
            },
          ];
        },
      },
    );
  });

  it('a link that is not a safe web address opens a card with Decline only; plain http is flagged', async () => {
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const script = await card(fake);
        expect(script.unusable).toBe('address');
        expect(script.link).toBeUndefined();
        notOffered(() => fake.host.answerElicit(id, script.id, { mcp: 'accept' }));
        fake.host.answerElicit(id, script.id, { mcp: 'decline' });
        const keyed = await card(fake, 2);
        expect(keyed.unusable).toBe('address');
        expect(JSON.stringify(keyed)).not.toContain(KEY);
        fake.host.answerElicit(id, keyed.id, { mcp: 'decline' });
        const plain = await card(fake, 3);
        expect(plain.link).toEqual({ url: 'http://intranet.local/sign-in', host: 'intranet.local', secure: false });
        fake.host.answerElicit(id, plain.id, { mcp: 'decline' });
        expect((await turn.done).outcome).toBe('completed');
        expect(replies(fake).map((r) => r.answer)).toEqual([DECLINE, DECLINE, DECLINE]);
      },
      {
        providers: noProviders({ secretNeedles: () => [KEY] }),
        mutate: steps(
          elicit({ id: 'mcp-elicit-a', mode: 'url', url: 'javascript:alert(1)' }),
          elicit({ id: 'mcp-elicit-b', mode: 'url', url: `https://example.com/cb?token=${KEY}` }),
          elicit({ id: 'mcp-elicit-c', mode: 'url', url: 'http://intranet.local/sign-in' }),
        ),
      },
    );
  });

  // Mutation: keep refusing an empty id in `readElicitRequest` → the request is cancelled with no card.
  it('a link with an empty elicitation id is a card to decline, not a silent cancel; an xn-- site is flagged', async () => {
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const empty = await card(fake);
        expect(empty.unusable).toBe('address');
        expect(empty.link).toBeUndefined();
        notOffered(() => fake.host.answerElicit(id, empty.id, { mcp: 'accept' }));
        fake.host.answerElicit(id, empty.id, { mcp: 'decline' });
        const lookalike = await card(fake, 2);
        expect(lookalike.link).toEqual({
          url: 'https://xn--gthub-n2e.com/login',
          host: 'xn--gthub-n2e.com',
          secure: true,
          lookalike: true,
        });
        fake.host.answerElicit(id, lookalike.id, { mcp: 'decline' });
        expect((await turn.done).outcome).toBe('completed');
        expect(replies(fake).map((r) => r.answer)).toEqual([DECLINE, DECLINE]);
      },
      {
        mutate: steps(
          elicit({ id: 'mcp-elicit-a', mode: 'url', url: 'https://github.com/login', elicitationId: '' }),
          elicit({ id: 'mcp-elicit-b', mode: 'url', url: 'https://gіthub.com/login' }),
        ),
      },
    );
  });

  it('Stop closes a link card as cancelled and the server gets cancel', async () => {
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const c = await card(fake);
        expect(fake.host.cancel(id)).toBe(true);
        expect((await turn.done).outcome).toBe('cancelled');
        expect(fake.events.closed).toEqual([{ sessionId: id, id: c.id, reason: 'cancelled' }]);
        expect(replies(fake)).toEqual([{ answer: CANCEL, late: false }]);
        expect(fake.events.mcp).toEqual([
          { sessionId: id, closed: { server: 'github', mode: 'url', outcome: 'cancel' } },
        ]);
        await new Promise((resolve) => setTimeout(resolve, 100));
        expect(fake.events.mcpDone).toEqual([]);
      },
      { mutate: steps(elicit({ mode: 'url', complete: {} })) },
    );
  });

  it('Chat never shows a card: the request is answered cancel', async () => {
    const fake = await fakeHost('chat-basic', {
      mode: 'chat',
      providers: noProviders(),
      mutate: (scenario) => {
        const prompt =
          (scenario.prompts as Json[]).find((p) => (p.match ?? '') === '') ?? (scenario.prompts as Json[])[0]!;
        prompt.steps = [elicit(), { text: 'ok' }, { end: { stopReason: 'end_turn' } }];
      },
    });
    try {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const turn = fake.host.startTurn(sessionId, 'hello', { modelId: MODEL });
      expect((await turn.done).outcome).toBe('completed');
      expect(fake.events.interactions).toEqual([]);
      expect(replies(fake)).toEqual([{ answer: CANCEL, late: false }]);
    } finally {
      await fake.close();
    }
  });

  it('Stop closes the card as cancelled and the server gets cancel', async () => {
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const c = await card(fake);
        expect(fake.host.cancel(id)).toBe(true);
        expect((await turn.done).outcome).toBe('cancelled');
        expect(fake.events.closed).toEqual([{ sessionId: id, id: c.id, reason: 'cancelled' }]);
        expect(replies(fake)).toEqual([{ answer: CANCEL, late: false }]);
        expect(fake.events.mcp.map((m) => m.closed.outcome)).toEqual(['cancel']);
      },
      { mutate: steps(elicit()) },
    );
  });

  it('the server stopping its wait closes the card as expired', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-elicit-gate-'));
    dirs.push(dir);
    const gate = join(dir, 'give-up');
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const c = await card(fake);
        writeFileSync(gate, '');
        await until(() => fake.events.closed.length === 1);
        expect(fake.events.closed).toEqual([{ sessionId: id, id: c.id, reason: 'expired' }]);
        expect(fake.host.openInteractions(id)).toEqual([]);
        expect(() => fake.host.answerElicit(id, c.id, { mcp: 'decline' })).toThrow(InteractionError);
        expect((await turn.done).outcome).toBe('completed');
        await until(() => replies(fake).length === 1);
        expect(replies(fake)).toEqual([{ answer: CANCEL, late: true }]);
        expect(fake.events.mcp.map((m) => m.closed.outcome)).toEqual(['expired']);
      },
      { mutate: steps(elicit({ giveUpOn: gate })) },
    );
  });

  it('with no window the card closes and the server gets cancel', async () => {
    await withWork(
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const c = await card(fake);
        fake.host.denyOpen();
        expect((await turn.done).outcome).toBe('completed');
        expect(fake.events.closed).toEqual([{ sessionId: id, id: c.id, reason: 'expired' }]);
        expect(replies(fake)).toEqual([{ answer: CANCEL, late: false }]);
        expect(fake.events.mcp.map((m) => m.closed.outcome)).toEqual(['expired']);
      },
      { mutate: steps(elicit()) },
    );
  });

  it('the answer body: accept with values, decline alone; no error names a value', () => {
    expect(interactionInput({ mcp: 'accept', values: [1, 'x', [0, 2], null, true] })).toEqual({
      mcp: 'accept',
      values: [1, 'x', [0, 2], null, true],
    });
    expect(interactionInput({ mcp: 'accept' })).toEqual({ mcp: 'accept', values: [] });
    expect(interactionInput({ mcp: 'decline' })).toEqual({ mcp: 'decline' });
    const refused = (body: unknown): string => {
      try {
        interactionInput(body);
      } catch (error) {
        return (error as Error).message;
      }
      throw new Error('accepted');
    };
    const messages = [
      refused({ mcp: 'decline', values: [] }),
      refused({ mcp: 'maybe' }),
      refused({ mcp: 'accept', values: [`${TYPED}${'x'.repeat(4_096)}`] }),
      refused({ mcp: 'accept', values: [{ v: TYPED }] }),
      refused({ mcp: 'accept', values: [[0, 0]] }),
      refused({ mcp: 'accept', values: [32] }),
      refused({ mcp: 'accept', values: Array.from({ length: 33 }, () => null) }),
      refused({ mcp: 'accept', values: [], extra: TYPED }),
    ];
    for (const message of messages) expect(message).not.toContain(TYPED);
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

describe('the service and an MCP card', () => {
  it('the route takes the MCP body; a refused form comes back as fields; one history line per card, no value', async () => {
    const fake = fakeService('work-permission', {
      mutate: (scenario) => {
        scenario.prompts = [
          { match: 'form', steps: [elicit(), { text: 'Done.' }, { end: { stopReason: 'end_turn' } }] },
          {
            match: 'again',
            steps: [
              elicit({ id: 'mcp-elicit-2', server: HIDDEN_SERVER }),
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
      const { conversation } = await svc.create({ mode: 'work', text: 'form', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.open().length === 1, 5_000);
      const c = page.open()[0]!;
      expect(c.kind).toBe('mcp-input');
      // An MCP card does not block Ask → Auto.
      expect(await svc.setPermissionMode(id, 'auto', true)).toEqual({ mode: 'auto' });
      expect(await svc.answer(id, c.id, { mcp: 'accept', values: [0, TYPED.slice(0, 2), [], '', true] })).toEqual({
        fields: [null, { code: 'minLength', n: 3 }, null, null, null],
      });
      expect(page.open()).toHaveLength(1);
      expect(await svc.answer(id, c.id, { mcp: 'accept', values: [0, TYPED, [], '', true] })).toEqual({});
      await until(() => page.open().length === 0 && page.state?.turn === null, 5_000);
      expect(fake.records().find((r) => r.kind === 'elicit-answer')?.answer).toEqual({
        outcome: 'accept',
        content: { repo: 'shop-api', title: TYPED, mine: true },
      });
      await until(() => (page.state?.permissionMarks ?? []).some((m) => m.kind === 'mcp'), 5_000);

      await svc.prompt(id, 'again');
      await until(() => page.open().length === 1, 5_000);
      await svc.answer(id, page.open()[0]!.id, { mcp: 'decline' });
      await until(() => page.state?.turn === null, 5_000);
      await until(() => (page.state?.permissionMarks ?? []).filter((m) => m.kind === 'mcp').length === 2, 5_000);

      const marks = (page.state?.permissionMarks ?? []).filter((m) => m.kind === 'mcp');
      const shownServer = storedServer(HIDDEN_SERVER);
      expect(shownServer).not.toBe(HIDDEN_SERVER);
      expect(marks.map((m) => ({ ...m, at: 0 }))).toEqual([
        { kind: 'mcp', at: 0, server: 'github', mode: 'form', outcome: 'accept' },
        { kind: 'mcp', at: 0, server: shownServer, mode: 'form', outcome: 'decline' },
      ]);
      const file = await mcpLog(fake, id, 2);
      const lines = file
        .split('\n')
        .filter((l) => l.includes('"mcp"'))
        .map((l) => JSON.parse(l) as Json);
      expect(lines.map((l) => ({ ...l, at: 0 }))).toEqual([
        { kind: 'mcp', at: 0, server: 'github', mode: 'form', outcome: 'accept' },
        { kind: 'mcp', at: 0, server: shownServer, mode: 'form', outcome: 'decline' },
      ]);
      // A reload reads the same marks the page got live.
      const place = { home: fake.root, agentHome: fake.root, cwd: project };
      const disk = readPermissionLines(file)
        .filter((line) => line.kind === 'mcp')
        .map((line) => markOf(line, place));
      expect(disk).toEqual(marks);
      expect(file).not.toContain(TYPED);
      expect(JSON.stringify(page.events)).not.toContain(TYPED);
      page.detach();
    } finally {
      window();
      await fake.close();
    }
  }, 30_000);

  it('a link: the route takes a bare accept; "opened" then "finished in the browser", live and after a reload', async () => {
    const fake = fakeService('work-permission', {
      mutate: (scenario) => {
        scenario.prompts = [
          {
            match: 'link',
            steps: [
              elicit({ mode: 'url', url: 'https://github.com/login', elicitationId: 'e-9', complete: {} }),
              { text: 'Signed in.' },
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
      const { conversation } = await svc.create({ mode: 'work', text: 'link', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.open().length === 1, 5_000);
      const c = page.open()[0]!;
      expect(c).toMatchObject({ kind: 'mcp-input', mode: 'url', link: { host: 'github.com', secure: true } });
      expect(await svc.answer(id, c.id, interactionInput({ mcp: 'accept' }))).toEqual({});
      await until(() => (page.state?.permissionMarks ?? []).filter((m) => m.kind === 'mcp').length === 2, 5_000);
      await until(() => page.state?.turn === null, 5_000);
      const marks = (page.state?.permissionMarks ?? []).filter((m) => m.kind === 'mcp');
      expect(marks.map((m) => ({ ...m, at: 0 }))).toEqual([
        { kind: 'mcp', at: 0, server: 'github', mode: 'url', outcome: 'accept' },
        { kind: 'mcp', at: 0, server: 'github', mode: 'url', outcome: 'completed' },
      ]);
      const file = await mcpLog(fake, id, 2);
      const place = { home: fake.root, agentHome: fake.root, cwd: project };
      const disk = readPermissionLines(file)
        .filter((line) => line.kind === 'mcp')
        .map((line) => markOf(line, place));
      expect(disk).toEqual(marks);
      // The address and the request's id stay out of the history.
      expect(file).not.toContain('github.com/login');
      expect(file).not.toContain('e-9');
      page.detach();
    } finally {
      window();
      await fake.close();
    }
  }, 30_000);
});

/**
 * The agent's command list on the host: pulled (soft) once a Work session is loaded, replaced by a later push, taken
 * before the live filter (never a transcript line); a push for a session the host does not hold is dropped; Chat lists
 * no agent command whatever the agent sends. Slash text is checked against the list: a cold conversation's turn is
 * checked once its session loaded, and a refused one ends unsent before `session/prompt`.
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { TranscriptReducer } from '../src/agent-sessions/transcript.js';
import { AgentHostError } from '../src/chat/agent/agent-host.js';
import { routeError } from '../src/chat/route-errors.js';
import {
  fakeHost,
  fakeService,
  until,
  type FakeHost,
  type FakeHostOptions,
  type FakeService,
  type Json,
} from './helpers/chat-fake.js';

const MODEL = 'qwen3.5-4b';
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function withHost(run: (fake: FakeHost) => Promise<void>, options: FakeHostOptions = {}) {
  const fake = await fakeHost(options.mode === 'work' ? 'work-length' : 'chat-basic', options);
  try {
    await run(fake);
  } finally {
    await fake.close();
  }
}

/** The scenario's skills and workflows, and one answer for every prompt. */
function listed(scenario: Json): void {
  scenario.prompts = [{ match: '', steps: [{ text: 'Done.' }, { end: { stopReason: 'end_turn' } }] }];
  scenario.session = {
    ...(scenario.session as Json),
    commands: {
      skills: [{ name: 'commit', description: 'Write a commit message\nfrom the diff', hint: '[scope]' }],
      workflows: [{ name: 'release', description: 'cut a release' }],
    },
  };
}

function codeOf(run: () => unknown): string | null {
  try {
    run();
  } catch (error) {
    return error instanceof AgentHostError ? error.code : String(error);
  }
  return null;
}

const prompts = (fake: FakeHost | FakeService): string[] =>
  fake
    .sent()
    .filter((m) => m.method === 'session/prompt')
    .map((m) => ((m.params as Json).prompt as Json[])[0]!.text as string);

const pulls = (fake: FakeHost): Json[] => fake.sent().filter((m) => m.method === '_x.ai/commands/list');

const WORK_ROWS = [
  {
    name: 'deep-research',
    kind: 'builtin',
    description: 'Research with bounded parallel agents, cross-check evidence, and write a cited report',
    hint: '<query>',
  },
  {
    name: 'workflow',
    kind: 'builtin',
    description: 'Launch a saved workflow, list runs, or manage a run (pause, resume, stop, save)',
    hint: '<name> [--agent-budget N] [--effort LEVEL] [args] | runs | pause|resume|stop|save [name]',
  },
  {
    name: 'goal',
    kind: 'builtin',
    description: 'Set, manage, or check an autonomous goal',
    hint: '<objective> [--budget <tokens>] | status | pause | resume | clear',
  },
  { name: 'loop', kind: 'builtin', description: 'Run a prompt on a recurring interval', hint: '[interval] <prompt>' },
  { name: 'commit', kind: 'skill', description: 'Write a commit message\nfrom the diff', hint: '[scope]' },
  {
    name: 'release',
    kind: 'workflow',
    description: 'Workflow: cut a release',
    hint: '[--agent-budget N] [--effort LEVEL] [args]',
  },
];

describe('the command list on the host', () => {
  it('Work pulls the list once the session is ready; the page gets the rows it may send', async () => {
    await withHost(
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        await until(() => fake.events.commands.length > 0);
        expect(pulls(fake)).toEqual([expect.objectContaining({ params: { sessionId: id } })]);
        expect(fake.events.commands).toEqual([{ sessionId: id, commands: WORK_ROWS }]);
        expect(fake.host.commandsOf(id)).toEqual(WORK_ROWS);
        // The list is the host's alone: no update line of it reaches the reducer.
        const turn = fake.host.startTurn(id, 'hello', { modelId: MODEL, cwd: fake.project });
        expect((await turn.done).outcome).toBe('completed');
        expect(fake.events.timeline).not.toContain('available_commands_update');
        // The same list pushed again at the turn's end is no change.
        expect(fake.events.commands).toHaveLength(1);
      },
      { mode: 'work', mutate: listed },
    );
  }, 20_000);

  // Mutation: drop the `null` hand-off at the agent's exit → the page keeps a list no agent stands behind.
  it('the agent exiting clears the list: the page gets none; the next start pulls it again', async () => {
    await withHost(
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        await until(() => fake.events.commands.length === 1);
        const pid = fake.records().find((r) => r.kind === 'start')!.pid as number;
        process.kill(pid, 'SIGKILL');
        await until(() => fake.events.commands.length === 2);
        expect(fake.events.commands[1]).toEqual({ sessionId: id, commands: null });
        expect(fake.host.commandsOf(id)).toBeNull();
        const turn = fake.host.startTurn(id, 'hello', { modelId: MODEL, cwd: fake.project });
        expect((await turn.done).outcome).toBe('completed');
        await until(() => fake.events.commands.length === 3);
        expect(fake.events.commands[2]).toEqual({ sessionId: id, commands: WORK_ROWS });
      },
      { mode: 'work', mutate: listed },
    );
  }, 20_000);

  it('a later push replaces the list', async () => {
    await withHost(
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        await until(() => fake.events.commands.length === 1);
        const turn = fake.host.startTurn(id, 'add a skill', { modelId: MODEL, cwd: fake.project });
        expect((await turn.done).outcome).toBe('completed');
        expect(fake.events.commands).toHaveLength(2);
        const last = fake.events.commands.at(-1)!;
        expect(last.sessionId).toBe(id);
        expect(last.commands!.filter((c) => c.kind !== 'builtin').map((c) => c.name)).toEqual(['review-pr']);
        expect(fake.host.commandsOf(id)).toEqual(last.commands);
        // The pushed list is the one the next message is checked against.
        expect(codeOf(() => fake.host.startTurn(id, '/commit', { modelId: MODEL, cwd: fake.project }))).toBe(
          'slash-text',
        );
        const review = fake.host.startTurn(id, '/review-pr 12', { modelId: MODEL, cwd: fake.project });
        expect((await review.done).outcome).toBe('completed');
        expect(prompts(fake).at(-1)).toBe('/review-pr 12');
      },
      {
        mode: 'work',
        mutate: (scenario) => {
          listed(scenario);
          scenario.prompts = [
            {
              match: 'add a skill',
              steps: [
                { commands: { skills: [{ name: 'review-pr', description: 'Review a pull request' }] } },
                { text: 'Added.' },
                { end: { stopReason: 'end_turn' } },
              ],
            },
          ];
        },
      },
    );
  }, 20_000);

  it('a push that carries the answer’s token count moves the context and makes no entry; one without stays the host’s', async () => {
    await withHost(
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        await until(() => fake.events.commands.length === 1);
        const turn = fake.host.startTurn(id, 'count', { modelId: MODEL, cwd: fake.project });
        expect((await turn.done).outcome).toBe('completed');
        const lines = fake.events.updates.filter((u) => u.sessionId === id).map((u) => u.line);
        const reducer = new TranscriptReducer();
        for (const line of lines) reducer.apply(line);
        expect(reducer.context()).toBe(9_000);
        expect(fake.events.timeline.filter((kind) => kind === 'available_commands_update')).toHaveLength(1);
        expect(reducer.snapshot().entries.map((e) => e.kind)).toEqual(['assistant']);
        expect(fake.host.commandsOf(id)?.map((c) => c.name)).toContain('review-pr');
      },
      {
        mode: 'work',
        mutate: (scenario) => {
          listed(scenario);
          scenario.prompts = [
            {
              match: 'count',
              steps: [
                { text: 'Counted.', totalTokens: 4_000 },
                { commands: { skills: [{ name: 'review-pr', description: 'Review a pull request' }] } },
                {
                  commands: { skills: [{ name: 'review-pr', description: 'Review a pull request' }] },
                  totalTokens: 9_000,
                },
                { end: { stopReason: 'end_turn' } },
              ],
            },
          ];
        },
      },
    );
  }, 20_000);

  it('a push before the session is known is dropped; a pull that misses keeps no list and leaves the agent', async () => {
    await withHost(
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        await sleep(300);
        // The agent pushed its list before `session/new` answered, and the pull never answered.
        expect(pulls(fake)).toHaveLength(1);
        expect(fake.events.commands).toEqual([]);
        expect(fake.host.commandsOf(id)).toBeNull();
        // Slash text only a listed skill could allow waits for the list, then ends unsent; the agent lives on.
        const refused = fake.host.startTurn(id, '/commit fix', { modelId: MODEL, cwd: fake.project });
        expect(await refused.done).toMatchObject({ outcome: 'refused', unsent: true, failure: { code: 'slash-text' } });
        expect(prompts(fake)).toEqual([]);
        expect(fake.host.modeState().state).toBe('ready');
        expect(fake.records().filter((r) => r.kind === 'start')).toHaveLength(1);
        // An allowed built-in goes only when the session's list offers it: with no list it ends unsent too.
        const early = fake.host.startTurn(id, '/goal ship it', { modelId: MODEL, cwd: fake.project });
        expect(await early.done).toMatchObject({ outcome: 'refused', unsent: true, failure: { code: 'slash-text' } });
        expect(prompts(fake)).toEqual([]);
        // The push at the end of an answer brings the list.
        const hello = fake.host.startTurn(id, 'hello', { modelId: MODEL, cwd: fake.project });
        expect((await hello.done).outcome).toBe('completed');
        await until(() => fake.events.commands.length === 1);
        expect(fake.events.commands[0]!.commands!.map((c) => c.name)).toEqual(
          expect.arrayContaining(['goal', 'commit']),
        );
        const goal = fake.host.startTurn(id, '/goal ship it', { modelId: MODEL, cwd: fake.project });
        expect((await goal.done).outcome).toBe('completed');
        const commit = fake.host.startTurn(id, '/commit fix', { modelId: MODEL, cwd: fake.project });
        expect((await commit.done).outcome).toBe('completed');
        expect(prompts(fake)).toEqual(['hello', '/goal ship it', '/commit fix']);
      },
      {
        mode: 'work',
        timings: { commandsListMs: 100 },
        mutate: (scenario) => {
          listed(scenario);
          scenario.faults = { hangOn: '_x.ai/commands/list' };
          scenario.prompts = [
            { match: '/goal', steps: [{ text: 'Goal set.' }, { end: { stopReason: 'end_turn' } }] },
            { match: '', steps: [{ text: 'Done.' }, { end: { stopReason: 'end_turn' } }] },
          ];
        },
      },
    );
  }, 20_000);

  /** `listed`, plus a `read-pdf` workflow row whose file is the agent home's `workflows/read-pdf.rhai`. */
  const withReadPdf = (scenario: Json) => {
    listed(scenario);
    const commands = (scenario.session as Json).commands as Json;
    commands.workflows = [...(commands.workflows as Json[]), { name: 'read-pdf', description: 'read', inHome: true }];
  };
  const names = (fake: FakeHost, id: string) => fake.host.commandsOf(id)?.map((c) => c.name);

  // Mutations: drop `withoutAppWorkflow` at the pull → `read-pdf` is listed and `/read-pdf` goes; at the push → it
  // comes back after the turn.
  it('the app’s own read-pdf is neither listed nor sent typed; its /workflow start still goes', async () => {
    await withHost(
      async (fake) => {
        const load = { modelId: MODEL, cwd: fake.project };
        const { sessionId: id } = await fake.host.createSession(load);
        await until(() => fake.events.commands.length === 1);
        expect(names(fake, id)).toEqual(['deep-research', 'workflow', 'goal', 'loop', 'commit', 'release']);
        expect(codeOf(() => fake.host.startTurn(id, '/read-pdf {"pdf":"p1"}', load))).toBe('slash-text');
        const hello = fake.host.startTurn(id, 'hello', load);
        expect((await hello.done).outcome).toBe('completed');
        expect(names(fake, id)).not.toContain('read-pdf');
        expect(codeOf(() => fake.host.startTurn(id, '/READ-PDF', load))).toBe('slash-text');
        const start = fake.host.startTurn(id, '/workflow read-pdf {"pdf":"p1"}', load);
        expect((await start.done).outcome).toBe('completed');
        expect(prompts(fake)).toEqual(['hello', '/workflow read-pdf {"pdf":"p1"}']);
      },
      { mode: 'work', mutate: withReadPdf },
    );
  }, 20_000);

  // Mutation: `appReadPdfPaths` skips the marker check → a person's own `read-pdf` is hidden and refused.
  it('a person’s own read-pdf in the agent home, without the app’s marker, is listed and sent', async () => {
    await withHost(
      async (fake) => {
        const own = 'workflow(#{ name: "read-pdf" });\n';
        mkdirSync(join(fake.agentHome, 'workflows'), { recursive: true });
        writeFileSync(join(fake.agentHome, 'workflows', 'read-pdf.rhai'), own);
        const load = { modelId: MODEL, cwd: fake.project };
        const { sessionId: id } = await fake.host.createSession(load);
        await until(() => fake.events.commands.length === 1);
        expect(names(fake, id)).toContain('read-pdf');
        const typed = fake.host.startTurn(id, '/read-pdf go', load);
        expect((await typed.done).outcome).toBe('completed');
        expect(prompts(fake)).toEqual(['/read-pdf go']);
        expect(readFileSync(join(fake.agentHome, 'workflows', 'read-pdf.rhai'), 'utf8')).toBe(own);
      },
      { mode: 'work', mutate: withReadPdf },
    );
  }, 20_000);

  it('a cold conversation’s slash text is checked once its session loaded', async () => {
    await withHost(
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        await until(() => fake.events.commands.length === 1);
        await fake.host.restart();
        expect(fake.host.isHot(id)).toBe(false);
        expect(fake.host.commandsOf(id)).toBeNull();
        const commit = fake.host.startTurn(id, '/commit fix', { modelId: MODEL, cwd: fake.project });
        expect((await commit.done).outcome).toBe('completed');
        expect(prompts(fake)).toEqual(['/commit fix']);
        expect(pulls(fake)).toHaveLength(2);

        await fake.host.restart();
        const unknown = fake.host.startTurn(id, '/nothere x', { modelId: MODEL, cwd: fake.project });
        expect(await unknown.done).toMatchObject({ outcome: 'refused', unsent: true, failure: { code: 'slash-text' } });
        expect(fake.sent().filter((m) => m.method === 'session/load')).toHaveLength(2);
        expect(prompts(fake)).toEqual(['/commit fix']);
      },
      { mode: 'work', mutate: listed },
    );
  }, 20_000);

  it('a built-in runs as the agent’s host turn and ends the answer', async () => {
    await withHost(
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        const turn = fake.host.startTurn(id, '/workflow runs', { modelId: MODEL, cwd: fake.project });
        expect((await turn.done).outcome).toBe('completed');
        expect(fake.records().filter((r) => r.kind === 'host-turn')).toEqual([
          expect.objectContaining({ name: 'workflow', text: '/workflow runs' }),
        ]);
        const said = fake.events.updates
          .map((u) => u.line.update)
          .filter((u) => u.sessionUpdate === 'agent_message_chunk');
        expect(said).toEqual([
          expect.objectContaining({ content: { type: 'text', text: 'No runs.' }, _meta: { hostTurn: true } }),
        ]);
      },
      {
        mode: 'work',
        mutate: (scenario) => {
          listed(scenario);
          scenario.hostTurnText = { workflow: 'No runs.' };
        },
      },
    );
  }, 20_000);

  it('Chat lists no agent command whatever the agent sends, and pulls none', async () => {
    await withHost(
      async (fake) => {
        const { sessionId: id } = await fake.host.createSession({ modelId: MODEL });
        const turn = fake.host.startTurn(id, 'hello', { modelId: MODEL });
        expect((await turn.done).outcome).toBe('completed');
        expect(pulls(fake)).toEqual([]);
        expect(fake.events.commands).toEqual([]);
        expect(fake.host.commandsOf(id)).toEqual([]);
        for (const text of ['/goal ship it', '/commit', '/loop 5m x']) {
          expect([text, codeOf(() => fake.host.startTurn(id, text, { modelId: MODEL }))]).toEqual([text, 'slash-text']);
        }
        expect(prompts(fake)).toEqual(['hello']);
      },
      {
        mutate: (scenario) => {
          listed(scenario);
          (scenario.session as Json).commands = {
            ...((scenario.session as Json).commands as Json),
            gates: ['AlwaysOn', 'Goal', 'Scheduler', 'WorkflowLaunches', 'WorkflowManagement', 'Hooks', 'Plugins'],
          };
        },
      },
    );
  }, 20_000);
});

describe('the service checks slash text before it sends or queues', () => {
  it('a busy Work conversation queues a listed command and refuses an unlisted one', async () => {
    let gate = '';
    const fake = fakeService('work-workflow-run', {
      mutate: (scenario) => {
        listed(scenario);
        gate = join(String(scenario.recordTo), '..', 'gate');
        scenario.prompts = [
          { match: 'hold', steps: [{ holdOn: gate }, { text: 'Done.' }, { end: { stopReason: 'end_turn' } }] },
          { match: '', steps: [{ text: 'Done.' }, { end: { stopReason: 'end_turn' } }] },
        ];
      },
    });
    const project = join(fake.root, 'home', 'code', 'app');
    mkdirSync(project, { recursive: true });
    try {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: 'hold on', cwd: project });
      await until(() => prompts(fake).length === 1, 10_000);
      const refused = await svc.prompt(conversation.id, '/nothere x', [], undefined, 'queue').then(
        () => null,
        (error: unknown) => routeError('chat', error).reason,
      );
      expect(refused).toBe('slash-text');
      expect(await svc.prompt(conversation.id, '/commit fix', [], undefined, 'queue')).toEqual({
        queued: expect.any(String),
      });
      writeFileSync(gate, '');
      await until(() => prompts(fake).length === 2, 10_000);
      expect(prompts(fake)).toEqual(['hold on', '/commit fix']);
    } finally {
      await fake.close();
    }
  }, 30_000);
});

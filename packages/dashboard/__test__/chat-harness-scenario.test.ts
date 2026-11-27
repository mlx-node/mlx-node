import { mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { parseUpdates, TranscriptReducer, type UpdateLine } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { NeverRules, neverNote } from '../src/chat/agent/never-rules.js';
import type { Interaction, McpInteraction, TranscriptOp } from '../src/chat/types.js';
import { activityOf, wakeReportOf, wakeReportsOf } from '../ui/src/lib/activity.js';
import { applyConversationEvent, type ConversationState } from '../ui/src/lib/chat-store.js';
import { fill } from '../ui/src/lib/chat-strings.js';
import { WAKE_ENDED, WAKE_REPORTED, wakeTitle } from '../ui/src/lib/sessions.js';
import { fakeHost, fakeService, until, type FakeHost, type Json } from './helpers/chat-fake.js';

const MODEL = 'qwen3.5-4b';

function texts(fake: FakeHost): string {
  return fake.events.updates
    .map((u) => {
      const content = u.line.update.content as Json | undefined;
      return typeof content?.text === 'string' ? content.text : '';
    })
    .join('');
}

/** The dev harness script: every prompt word reaches the outcome the Chat page is built to show. */
describe('chat-harness scenario', () => {
  it('each prompt word ends its turn as scripted', async () => {
    const fake = await fakeHost('chat-harness');
    try {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      const run = async (text: string) => (await fake.host.startTurn(sessionId, text, { modelId: MODEL }).done).outcome;

      expect(await run('search rust')).toBe('completed');
      expect(fake.events.web.map((w) => w.result.kind)).toEqual(['search', 'search']);
      expect(texts(fake)).toContain('Sources:');

      expect(await run('read https://docs.example.org/guide')).toBe('completed');
      expect(fake.events.web.map((w) => w.result.kind)).toEqual(['search', 'search', 'page']);

      expect(await run('run a shell')).toBe('tool-refused');
      expect(await run('make an error')).toBe('model-error');
      expect(await run('a long one')).toBe('output-limit');
      expect(await run('hello')).toBe('completed');
      expect(texts(fake)).toContain('```rust');

      const slow = fake.host.startTurn(sessionId, 'slow please', { modelId: MODEL });
      await until(() => texts(fake).includes('Two.'), 5_000);
      expect(fake.host.cancel(sessionId)).toBe(true);
      expect((await slow.done).outcome).toBe('cancelled');

      const said = texts(fake);
      const stall = fake.host.startTurn(sessionId, 'stall here', { modelId: MODEL });
      await new Promise((resolve) => setTimeout(resolve, 300));
      expect(texts(fake)).toBe(said);
      expect(fake.host.cancel(sessionId)).toBe(true);
      expect((await stall.done).outcome).toBe('cancelled');
    } finally {
      await fake.close();
    }
  });

  it('Work lists the harness skills and workflow; a built-in plays the agent’s host turn; Chat lists none', async () => {
    const fake = await fakeHost('chat-harness', { mode: 'work' });
    try {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      await until(() => fake.events.commands.length > 0);
      expect(fake.host.commandsOf(sessionId)?.map((c) => `${c.kind} ${c.name}`)).toEqual([
        'builtin deep-research',
        'builtin workflow',
        'builtin goal',
        'builtin loop',
        'skill commit',
        'skill review-pr',
        'workflow release',
      ]);
      const goal = fake.host.startTurn(sessionId, '/goal', { modelId: MODEL, cwd: fake.project });
      expect((await goal.done).outcome).toBe('completed');
      expect(texts(fake)).toContain('Goal paused. Use /goal resume to continue.');
    } finally {
      await fake.close();
    }
    const chat = await fakeHost('chat-harness');
    try {
      const { sessionId } = await chat.host.createSession({ modelId: MODEL });
      expect((await chat.host.startTurn(sessionId, 'hello', { modelId: MODEL }).done).outcome).toBe('completed');
      expect(chat.host.commandsOf(sessionId)).toEqual([]);
    } finally {
      await chat.close();
    }
  });

  // Mutation: pass every allowed built-in → `/loop` goes to a session whose agent no longer offers it.
  it('"noloop": the agent stops offering /loop; the host then refuses it as an unknown command, /goal still goes', async () => {
    const fake = await fakeHost('chat-harness', { mode: 'work' });
    try {
      const load = { modelId: MODEL, cwd: fake.project };
      const { sessionId } = await fake.host.createSession(load);
      await until(() => fake.events.commands.length > 0);
      expect((await fake.host.startTurn(sessionId, 'noloop', load).done).outcome).toBe('completed');
      await until(() => !(fake.host.commandsOf(sessionId) ?? []).some((c) => c.name === 'loop'));
      expect(fake.host.commandsOf(sessionId)?.map((c) => c.name)).toEqual([
        'deep-research',
        'workflow',
        'goal',
        'commit',
        'review-pr',
        'release',
      ]);
      expect(() => fake.host.startTurn(sessionId, '/loop 5m check the build', load)).toThrow(
        expect.objectContaining({ code: 'slash-text' }),
      );
      expect((await fake.host.startTurn(sessionId, '/goal', load).done).outcome).toBe('completed');
    } finally {
      await fake.close();
    }
  });

  it('each Work prompt word asks what it is scripted to ask', async () => {
    const fake = await fakeHost('chat-harness', { mode: 'work' });
    try {
      const { sessionId, agentName } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      expect(agentName).toBe('grok-build-plan');
      let seen = 0;
      const run = async (text: string, answer: (card: Interaction) => void) => {
        const started = fake.host.startTurn(sessionId, text, { modelId: MODEL, cwd: fake.project });
        await until(() => fake.events.interactions.length > seen);
        const card = fake.events.interactions[seen]!.interaction;
        seen += 1;
        answer(card);
        return { card, outcome: (await started.done).outcome };
      };
      const allow = (card: Interaction) => fake.host.answer(sessionId, card.id, 'allow-once');

      const ran = await run('run the tests', allow);
      expect(ran).toMatchObject({ card: { command: 'git status', flags: [] }, outcome: 'completed' });
      expect(ran.card.kind === 'permission' && ran.card.options.map((o) => o.optionId)).toContain('always-allow');

      const wrote = await run('write a note', allow);
      expect(wrote).toMatchObject({ card: { access: 'edit', path: 'notes.md' }, outcome: 'completed' });

      const risky = await run('something risky', (card) => fake.host.answer(sessionId, card.id, 'reject-once'));
      expect(risky).toMatchObject({ card: { flags: ['elevated', 'deletes', 'outside-folder', 'installs'] } });
      expect(risky.outcome).toBe('denied');

      const plan = await run('plan it', (card) => fake.host.answerPlan(sessionId, card.id, 'approved'));
      expect(plan).toMatchObject({ card: { kind: 'plan' }, outcome: 'completed' });
      // The model entered plan mode itself; approving the plan left it.
      const modes = fake.events.updates
        .filter((u) => u.line.update.sessionUpdate === 'current_mode_update')
        .map((u) => u.line.update.currentModeId);
      expect(modes).toEqual(['plan', 'default']);

      const asked = await run('a question for me', (card) =>
        fake.host.answerQuestion(sessionId, card.id, { answers: [{ picks: [1] }, { picks: [0, 2] }, null] }),
      );
      expect(asked.outcome).toBe('completed');
      expect(asked.card.kind === 'question' && asked.card.questions.map((q) => [q.text, q.multi])).toEqual([
        ['Which database should the cart use?', false],
        ['Which checks should run before a merge?', true],
        ['Deploy after merge?', false],
      ]);
      expect(
        fake
          .records()
          .filter((r) => r.kind === 'question-answer')
          .map((r) => r.answer),
      ).toEqual([
        {
          outcome: 'accepted',
          answers: {
            'Which database should the cart use?': ['Redis'],
            'Which checks should run before a merge?': ['Unit tests', 'Lint'],
          },
        },
      ]);

      const form = await run('an mcp form', (card) => {
        expect(card.kind).toBe('mcp-input');
        const values = [0, 'Cart is empty', '', [0], '3', '', '', true];
        expect(fake.host.answerElicit(sessionId, card.id, { mcp: 'accept', values })).toBeNull();
      });
      expect(form.outcome).toBe('completed');
      expect(form.card.kind === 'mcp-input' && form.card.form?.fields.map((f) => [f.title, f.type])).toEqual([
        ['Repository', 'choice'],
        ['Title', 'text'],
        ['Details', 'text'],
        ['Labels', 'choices'],
        ['Priority', 'integer'],
        ['Due', 'text'],
        ['Notify', 'text'],
        ['Assign to me', 'boolean'],
      ]);
      expect(
        fake
          .records()
          .filter((r) => r.kind === 'elicit-answer')
          .map((r) => r.answer),
      ).toEqual([
        {
          outcome: 'accept',
          content: { repo: 'shop-web', title: 'Cart is empty', labels: ['bug'], priority: 3, assign: true },
        },
      ]);

      const signIn = await run('signin', (card) => {
        expect(card).toMatchObject({
          kind: 'mcp-input',
          mode: 'url',
          server: 'github',
          link: {
            url: 'https://github.com/login/oauth/authorize?client_id=mlx-dev&state=fixture',
            host: 'github.com',
            secure: true,
          },
        });
        expect(fake.host.answerElicit(sessionId, card.id, { mcp: 'accept' })).toBeNull();
      });
      expect(signIn.outcome).toBe('completed');
      expect(
        fake
          .records()
          .filter((r) => r.kind === 'elicit-answer')
          .map((r) => r.answer)
          .at(-1),
      ).toEqual({ outcome: 'accept' });

      const key = await run('show the key', allow);
      expect(key).toMatchObject({ card: { command: 'echo $EXA_API_KEY', flags: ['prints-secrets'] } });

      const known = ['ls', 'cat README.md', 'echo done > notes.txt', 'notes.md', 'curl -s https://example.com'];
      const started = fake.host.startTurn(sessionId, 'auto', { modelId: MODEL, cwd: fake.project });
      for (const want of known) {
        await until(() => fake.events.interactions.length > seen);
        const card = fake.events.interactions[seen]!.interaction;
        seen += 1;
        expect(card.kind === 'permission' && (card.command ?? card.path)).toBe(want);
        allow(card);
      }
      expect((await started.done).outcome).toBe('completed');

      const everything = ["printf '%d\\n' 5", '~/notes/out.txt', 'github__create_issue', 'touch b'];
      const all = fake.host.startTurn(sessionId, 'everything', { modelId: MODEL, cwd: fake.project });
      for (const want of everything) {
        await until(() => fake.events.interactions.length > seen);
        const card = fake.events.interactions[seen]!.interaction;
        seen += 1;
        expect(card.kind === 'permission' && (card.command ?? card.path ?? card.toolName)).toBe(want);
        allow(card);
      }
      expect((await all.done).outcome).toBe('completed');

      const other = await run('hello', allow);
      expect(other).toMatchObject({ card: { command: 'ls' }, outcome: 'completed' });
    } finally {
      await fake.close();
    }
  });

  it('"interview": the model enters plan mode and asks with previews; Talk it over goes back as its plan reply', async () => {
    const fake = await fakeHost('chat-harness', { mode: 'work' });
    try {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      const turn = fake.host.startTurn(sessionId, 'interview me', { modelId: MODEL, cwd: fake.project });
      await until(() => fake.events.interactions.length === 1);
      const card = fake.events.interactions[0]!.interaction;
      expect(card.kind).toBe('question');
      if (card.kind !== 'question') return;
      expect(card.planMode).toBe(true);
      expect(card.questions.map((q) => [q.text, q.multi, q.options.map((o) => o.preview !== null)])).toEqual([
        ['Which layout should the shop use?', false, [true, true, false]],
        ['Which pages come first?', true, [false, false, false]],
      ]);
      fake.host.answerQuestion(sessionId, card.id, { reply: 'discuss', answers: [{ picks: [1] }, null] });
      expect((await turn.done).outcome).toBe('completed');
      expect(
        fake
          .records()
          .filter((r) => r.kind === 'question-answer')
          .map((r) => r.answer),
      ).toEqual([
        { outcome: 'chat_about_this', partial_answers: { 'Which layout should the shop use?': 'Top bar layout' } },
      ]);
      const reducer = new TranscriptReducer();
      for (const u of fake.events.updates) if (u.sessionId === sessionId) reducer.apply(u.line);
      const ask = reducer.snapshot().entries.find((e) => e.kind === 'tool' && e.name === 'ask_user_question');
      expect(ask?.kind === 'tool' && ask.ask?.result).toEqual({ kind: 'discuss' });
    } finally {
      await fake.close();
    }
  });

  it('"colors": the flagged ticks draw the same live output as the unflagged ones, as plain text', async () => {
    const play = async (flagged: boolean) => {
      const fake = await fakeHost('chat-harness', {
        mode: 'work',
        mutate: (scenario) => {
          if (!flagged) (scenario.initialize as Json).incrementalBashOutput = false;
          const work = (scenario.work as Json).prompts as Json[];
          const steps = work.find((p) => p.match === 'colors')!.steps as Json[];
          (steps.find((s) => s.shell !== undefined)!.shell as Json).tickMs = 5;
        },
      });
      try {
        const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
        const turn = fake.host.startTurn(sessionId, 'colors', { modelId: MODEL, cwd: fake.project });
        expect((await turn.done).outcome).toBe('completed');
        const init = fake.sent().find((m) => m.method === 'initialize')!.params as Json;
        expect((init.clientCapabilities as Json)._meta).toEqual({ 'x.ai/incrementalBashOutput': true });
        const lines: UpdateLine[] = fake.events.updates.filter((u) => u.sessionId === sessionId).map((u) => u.line);
        const ticks = lines
          .filter((l) => l.update.sessionUpdate === 'tool_call_update' && l.update.status === 'in_progress')
          .map((l) => l.update.rawOutput as Json);
        const reducer = new TranscriptReducer();
        const ops: TranscriptOp[] = lines.flatMap((line) => reducer.apply(line));
        const drawn: string[] = [];
        for (const op of ops) {
          if ((op.op !== 'push' && op.op !== 'set') || op.entry.kind !== 'tool' || op.entry.output === undefined)
            continue;
          if (drawn[drawn.length - 1] !== op.entry.output) drawn.push(op.entry.output);
        }
        const files = readdirSync(fake.agentHome, { recursive: true, encoding: 'utf8' })
          .filter((f) => f.endsWith('updates.jsonl'))
          .map((f) => readFileSync(join(fake.agentHome, f), 'utf8'))
          .filter((text) => text.includes('run_terminal_command'));
        expect(files).toHaveLength(1);
        const storedTool = parseUpdates(files[0]!).transcript.find((e) => e.kind === 'tool');
        const liveTool = reducer.snapshot().entries.find((e) => e.kind === 'tool');
        expect(files[0]!).not.toContain('in_progress');
        expect(storedTool).toMatchObject({ kind: 'tool', status: 'completed', output: drawn.at(-1) });
        expect(storedTool).toEqual(liveTool);
        return { ticks, drawn };
      } finally {
        await fake.close();
      }
    };
    const today = await play(false);
    const flagged = await play(true);
    expect(today.ticks).toHaveLength(3);
    expect(flagged.ticks).toHaveLength(3);
    for (const raw of today.ticks) {
      expect((raw.output as number[]).length).toBeGreaterThan(0);
      expect(raw).not.toHaveProperty('output_delta');
    }
    for (const raw of flagged.ticks) {
      expect(raw.output).toEqual([]);
      expect((raw.output_delta as number[]).length).toBeGreaterThan(0);
    }
    expect(flagged.drawn).toEqual(today.drawn);
    expect(flagged.drawn).toHaveLength(4);
    expect(flagged.drawn.at(-1)).toBe(
      ' PASS  __test__/line-diff.test.ts (12 tests)\n PASS  __test__/queue.test.ts (8 tests)\n\n' +
        ' Test Files  2 passed\n      Tests  20 passed\n',
    );
  });

  it.each([
    ['bypass', 0],
    ['auto', 3],
  ] as const)('"everything" in %s: %i cards', async (mode, count) => {
    const fake = await fakeHost('chat-harness', {
      mode: 'work',
      permission: { modeOf: () => mode, watched: () => true },
      onModeAllowed: async () => true,
    });
    try {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      const turn = fake.host.startTurn(sessionId, 'everything', { modelId: MODEL, cwd: fake.project });
      for (let n = 0; n < count; n++) {
        await until(() => fake.events.interactions.length > n);
        fake.host.answer(sessionId, fake.events.interactions[n]!.interaction.id, 'allow-once');
      }
      expect((await turn.done).outcome).toBe('completed');
      expect(fake.events.interactions).toHaveLength(count);
      const asked = fake.events.interactions.map((e) =>
        e.interaction.kind === 'permission' ? e.interaction.autoAsk : null,
      );
      expect(asked).toEqual(mode === 'auto' ? ['not-known', 'risky', 'not-covered'] : []);
    } finally {
      await fake.close();
    }
  });

  it('"everything" with Bypass picked on the New screen: four allowed marks and no card', async () => {
    const fake = fakeService('chat-harness');
    const window = fake.service.watchHost(() => {});
    try {
      const project = join(fake.root, 'home', 'code', 'app');
      mkdirSync(project, { recursive: true });
      await fake.service.setDraftPermissionMode('d1', 'bypass', true, project, null);
      const { conversation } = await fake.service.create({
        mode: 'work',
        text: 'everything',
        cwd: project,
        draft: 'd1',
      });
      expect(conversation.permissionMode).toBe('bypass');
      let state: ConversationState | null = null;
      let cards = 0;
      const off = fake.service.watchConversation(conversation.id, (event) => {
        if (event.type === 'interaction') cards += 1;
        const next = applyConversationEvent(state, event);
        if (next !== 'resync') state = next;
      });
      const done = (): boolean => state !== null && state.turn === null && state.entries.length > 1;
      await until(done, 10_000);
      const marks = (state as ConversationState | null)?.permissionMarks ?? [];
      expect(marks.filter((m) => m.kind === 'allowed').map((m) => m.kind === 'allowed' && m.mode)).toEqual([
        'bypass',
        'bypass',
        'bypass',
        'bypass',
      ]);
      expect([cards, (state as ConversationState | null)?.interactions.length]).toEqual([0, 0]);
      off();
    } finally {
      window();
      await fake.close();
    }
  }, 30_000);

  it('"needs": a subagent, the turn and a workflow helper ask; the subagent gives up and its card closes as expired', async () => {
    const fake = await fakeHost('chat-harness', {
      mode: 'work',
      mutate: (scenario) => {
        const work = (scenario.work as Json).prompts as Json[];
        const steps = work.find((p) => p.match === 'needs')!.steps as Json[];
        (steps.find((s) => s.giveUp !== undefined)!.giveUp as Json).afterMs = 50;
        (steps.find((s) => s.workflow !== undefined)!.workflow as Json).completeAfterMs = 0;
      },
    });
    try {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      const turn = fake.host.startTurn(sessionId, 'needs', { modelId: MODEL, cwd: fake.project });
      await until(() => fake.events.interactions.length === 3);
      const [helper, own, run] = fake.events.interactions.map((e) => e.interaction);
      expect(helper).toMatchObject({
        command: 'cargo test -p app',
        from: { label: 'fix tests', run: null, child: 'helper-needs' },
      });
      expect(own).toMatchObject({ command: 'rm -rf build/' });
      expect(own!.kind === 'permission' && own.from).toBeUndefined();
      expect(run).toMatchObject({ command: 'cargo publish', from: { label: 'packer', run: 'release' } });
      await until(() => fake.events.closed.length === 1);
      expect(fake.events.closed).toEqual([{ sessionId, id: helper!.id, reason: 'expired' }]);
      fake.host.answer(sessionId, own!.id, 'allow-once');
      fake.host.answer(sessionId, run!.id, 'allow-once');
      expect((await turn.done).outcome).toBe('completed');
      await until(() => fake.records().some((r) => r.kind === 'child-answer'));
      expect(fake.records().find((r) => r.kind === 'child-answer')?.answer).toBe('allow-once');
    } finally {
      await fake.close();
    }
  });
  // Mutation: let the fake skip the scenario's `changed` → src/log.ts reads clean.
  it('Rewind after "diff": the dry run names the four files it wrote, src/log.ts changed outside; Files only puts them back', async () => {
    const fake = await fakeHost('chat-harness', { mode: 'work' });
    try {
      const load = { modelId: MODEL, cwd: fake.project };
      const { sessionId } = await fake.host.createSession(load);
      const turn = fake.host.startTurn(sessionId, 'diff', load);
      for (let n = 0; n < 5; n += 1) {
        await until(() => fake.events.interactions.length > n);
        fake.host.answer(sessionId, fake.events.interactions[n]!.interaction.id, 'allow-once');
      }
      expect((await turn.done).outcome).toBe('completed');
      const preview = await fake.host.rewindPreview(sessionId, load, 0);
      expect('files' in preview && [...preview.files].sort((a, b) => (a.path < b.path ? -1 : 1))).toEqual([
        { path: 'README.md', state: 'clean' },
        { path: 'docs/notes.md', state: 'clean' },
        { path: 'src/app.ts', state: 'clean' },
        { path: 'src/log.ts', state: 'modified' },
      ]);
      const files = 'files' in preview ? preview.files : [];
      const done = await fake.host.rewind(sessionId, load, {
        target: 0,
        mode: 'files',
        tasks: [],
        schedules: [],
        runs: [],
        files,
      }).done;
      expect(done).toEqual({ outcome: 'done', restored: 4 });
      expect(await fake.host.rewindPreview(sessionId, load, 0)).toEqual({ files: [] });
    } finally {
      await fake.close();
    }
  });

  it('"diff": five edits as the agent plays them; live and stored give the same diffs, the change shown before Allow', async () => {
    const fake = await fakeHost('chat-harness', { mode: 'work' });
    try {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      const turn = fake.host.startTurn(sessionId, 'diff', { modelId: MODEL, cwd: fake.project });
      const reducer = new TranscriptReducer();
      let fed = 0;
      const feed = (): void => {
        const lines = fake.events.updates.filter((u) => u.sessionId === sessionId).map((u) => u.line);
        for (; fed < lines.length; fed += 1) reducer.apply(lines[fed]!);
      };
      for (let n = 0; n < 5; n += 1) {
        await until(() => fake.events.interactions.length > n || fake.events.updates.length > 200);
        const card = fake.events.interactions[n]!.interaction;
        feed();
        const asked = reducer.snapshot().entries.find((e) => e.kind === 'tool' && e.id === card.toolCallId);
        expect(card).toMatchObject({ kind: 'permission', access: 'edit' });
        expect(asked).toMatchObject({ kind: 'tool', status: 'pending' });
        expect(asked?.kind === 'tool' && asked.edits?.[0]?.hunks.length).toBeGreaterThan(0);
        fake.host.answer(sessionId, card.id, 'allow-once');
      }
      expect((await turn.done).outcome).toBe('completed');
      feed();
      const live = reducer.snapshot().entries.filter((e) => e.kind === 'tool');
      const files = readdirSync(fake.agentHome, { recursive: true, encoding: 'utf8' })
        .filter((f) => f.endsWith('updates.jsonl'))
        .map((f) => readFileSync(join(fake.agentHome, f), 'utf8'))
        .filter((text) => text.includes('docs/notes.md'));
      expect(files).toHaveLength(1);
      const stored = parseUpdates(files[0]!).transcript.filter((e) => e.kind === 'tool');
      expect(stored).toEqual(live);
      const shapes = stored.map((e) =>
        e.kind === 'tool'
          ? [e.status, e.edits?.map((d) => [d.create === true, d.added, d.removed, d.hunks.length]), e.output ?? null]
          : null,
      );
      expect(shapes).toEqual([
        ['completed', [[false, 2, 1, 1]], null],
        ['completed', [[false, 2, 2, 2]], null],
        ['completed', [[true, 4, 0, 1]], null],
        ['completed', [[false, 1, 1, 1]], null],
        ['failed', [[false, 1, 1, 1]], 'No matches found for old_string.'],
      ]);
      const first = stored[0]!.kind === 'tool' ? stored[0]!.edits![0]! : null;
      expect(first?.hunks[0]).toEqual({
        start: [2, 2],
        body: [
          " import { log } from './log';",
          ' ',
          " const host = 'localhost';",
          '-export const port = 3000;',
          '+export const port = 8080;',
          '+export const retries = 3;',
          ' ',
          ' export function start(): void {',
          "   log('starting');",
        ].join('\n'),
      });
      expect(JSON.stringify(stored)).not.toContain('EditsApplied');
    } finally {
      await fake.close();
    }
  });

  it('"leftpad": denied with a note the agent goes on with pnpm; a plain Deny ends the answer', async () => {
    const fake = await fakeHost('chat-harness', { mode: 'work' });
    try {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      const noted = fake.host.startTurn(sessionId, 'add leftpad', { modelId: MODEL, cwd: fake.project });
      await until(() => fake.events.interactions.length === 1);
      const card = fake.events.interactions[0]!.interaction;
      expect(card).toMatchObject({ command: 'npm install left-pad' });
      fake.host.answer(sessionId, card.id, 'reject-once', 'use pnpm, not npm');
      expect((await noted.done).outcome).toBe('completed');
      expect(texts(fake)).toContain('Added left-pad with pnpm.');
      const plain = fake.host.startTurn(sessionId, 'add leftpad', { modelId: MODEL, cwd: fake.project });
      await until(() => fake.events.interactions.length === 2);
      fake.host.answer(sessionId, fake.events.interactions[1]!.interaction.id, 'reject-once');
      expect((await plain.done).outcome).toBe('denied');
    } finally {
      await fake.close();
    }
  });

  it('"oddlink": a site in its xn-- form is flagged; a link with an empty id is a card to decline', async () => {
    const fake = await fakeHost('chat-harness', { mode: 'work' });
    try {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      const turn = fake.host.startTurn(sessionId, 'oddlink', { modelId: MODEL, cwd: fake.project });
      await until(() => fake.events.interactions.length === 1);
      const odd = fake.events.interactions[0]!.interaction as McpInteraction;
      expect(odd.link).toMatchObject({ host: 'xn--gthub-n2e.com', secure: true, lookalike: true });
      fake.host.answerElicit(sessionId, odd.id, { mcp: 'decline' });
      await until(() => fake.events.interactions.length === 2);
      const empty = fake.events.interactions[1]!.interaction as McpInteraction;
      expect(empty).toMatchObject({ mode: 'url', unusable: 'address' });
      fake.host.answerElicit(sessionId, empty.id, { mcp: 'decline' });
      expect((await turn.done).outcome).toBe('completed');
      expect(texts(fake)).toContain('Both pages were answered.');
    } finally {
      await fake.close();
    }
  });

  it('"helperform": after the answer ends, the run helper\'s MCP server asks: the card is the run\'s', async () => {
    const fake = await fakeHost('chat-harness', { mode: 'work' });
    try {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      const turn = fake.host.startTurn(sessionId, 'helperform', { modelId: MODEL, cwd: fake.project });
      await until(() => fake.events.interactions.length === 1);
      fake.host.answer(sessionId, fake.events.interactions[0]!.interaction.id, 'allow-once');
      expect((await turn.done).outcome).toBe('completed');
      await until(() => fake.events.interactions.length === 2, 8_000);
      const card = fake.events.interactions[1]!.interaction as McpInteraction;
      expect(card).toMatchObject({ kind: 'mcp-input', server: 'github', run: 'deep-research' });
      expect(fake.host.answerElicit(sessionId, card.id, { mcp: 'accept', values: [1] })).toBeNull();
      await until(() => fake.records().some((r) => r.kind === 'elicit-answer'));
      expect(fake.records().find((r) => r.kind === 'elicit-answer')!.answer).toEqual({
        outcome: 'accept',
        content: { repo: 'shop-api' },
      });
    } finally {
      await fake.close();
    }
  }, 20_000);

  it('"wrapped": sudo, env and bash -c around a never-allowed npm publish: denied with no card, the agent goes on', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'harness-never-'));
    const never = new NeverRules(join(dir, 'never-allow.json'));
    await never.ready();
    const fake = await fakeHost('chat-harness', {
      mode: 'work',
      never,
      permission: { modeOf: () => 'ask', watched: () => true },
      onModeAllowed: async () => true,
    });
    try {
      const { sessionId, cwd } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      await never.add(cwd, ['npm', 'publish']).saved;
      const turn = fake.host.startTurn(sessionId, 'wrapped', { modelId: MODEL, cwd: fake.project });
      expect((await turn.done).outcome).toBe('completed');
      expect(fake.events.interactions).toHaveLength(0);
      expect(texts(fake)).toContain('The publish was not allowed, so I left it.');
      const replies = fake
        .sent()
        .filter((m) => m.method === undefined && JSON.stringify(m.result ?? '').includes('reject-once'));
      expect(JSON.stringify(replies)).toContain(JSON.stringify(neverNote('npm publish')).slice(1, -1));
    } finally {
      await fake.close();
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('"slow" in Work takes a Send now; Stop before its read drops it, as the agent does', async () => {
    const fake = await fakeHost('chat-harness', { mode: 'work' });
    try {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      const slow = fake.host.startTurn(sessionId, 'slow please', { modelId: MODEL, cwd: fake.project });
      await until(() => fake.host.promptOpen(sessionId) && texts(fake).includes('Read the file first.'), 5_000);
      expect(await fake.host.steer(sessionId, 'also update the changelog')).toBe('sent');
      expect(
        fake
          .records()
          .filter((r) => r.kind === 'interject')
          .map((r) => r.text),
      ).toEqual(['also update the changelog']);
      expect(fake.host.cancel(sessionId)).toBe(true);
      expect((await slow.done).outcome).toBe('cancelled');
      await new Promise((resolve) => setTimeout(resolve, 100));
      expect(fake.records().some((r) => r.kind === 'steer-read' || r.kind === 'steer-turn')).toBe(false);
    } finally {
      await fake.close();
    }
  });

  // Mutation: drop the fake agent's `slowLoad` wait → the load answers at once and the recap is written.
  it('"lateopen": after a restart the agent opens the conversation late; a recap ends load-slow and the load lands', async () => {
    const fake = await fakeHost('chat-harness', {
      mode: 'work',
      timings: { loadSoftMs: 300 },
      mutate: (scenario) => {
        const word = ((scenario.work as Json).prompts as Json[]).find((p) => p.match === 'lateopen')!;
        (word.steps as Json[])[0] = { slowLoad: 1_500 };
      },
    });
    try {
      const load = { modelId: MODEL, cwd: fake.project };
      const markers = (): string[] =>
        (readdirSync(fake.root, { recursive: true }) as string[]).filter((path) => path.endsWith('slow_load.json'));
      const { sessionId } = await fake.host.createSession(load);
      expect((await fake.host.startTurn(sessionId, 'lateopen', load).done).outcome).toBe('completed');
      expect(markers()).toHaveLength(1);
      await fake.host.restart();
      expect(await fake.host.recap(sessionId, load).done).toBe('load-slow');
      const starts = fake.records().filter((r) => r.kind === 'start').length;
      await until(() => fake.host.isHot(sessionId), 5_000);
      expect(fake.records().filter((r) => r.kind === 'start')).toHaveLength(starts);
      expect(fake.sent().filter((m) => m.method === '_x.ai/recap')).toEqual([]);
      // The marker is used once.
      expect(markers()).toEqual([]);
    } finally {
      await fake.close();
    }
  }, 20_000);

  it('"retry" and "retryfail", Chat and Work: two stalls then the answer; three server errors then the agent gives up', async () => {
    // The harness waits seconds between tries so the line can be seen; here they are short.
    const quick = (scenario: Json): void => {
      const lists = [scenario.prompts, (scenario.work as Json).prompts] as Json[][];
      for (const list of lists)
        for (const prompt of list)
          if (prompt.match === 'retry' || prompt.match === 'retryfail')
            for (const step of prompt.steps as Json[]) if (typeof step.sleepMs === 'number') step.sleepMs = 20;
    };
    for (const mode of ['chat', 'work'] as const) {
      const fake = await fakeHost('chat-harness', { mode, mutate: quick });
      try {
        const cwd = mode === 'work' ? { cwd: fake.project } : {};
        const { sessionId } = await fake.host.createSession({ modelId: MODEL, ...cwd });
        const reducer = new TranscriptReducer();
        let fed = 0;
        const feed = (): void => {
          const lines = fake.events.updates.filter((u) => u.sessionId === sessionId).map((u) => u.line);
          for (; fed < lines.length; fed += 1) reducer.apply(lines[fed]!);
        };
        reducer.userPrompt('retry please');
        expect((await fake.host.startTurn(sessionId, 'retry please', { modelId: MODEL, ...cwd }).done).outcome).toBe(
          'completed',
        );
        feed();
        reducer.userPrompt('retryfail now');
        expect((await fake.host.startTurn(sessionId, 'retryfail now', { modelId: MODEL, ...cwd }).done).outcome).toBe(
          'model-error',
        );
        feed();
        const notices = reducer.snapshot().entries.filter((e) => e.kind === 'notice' && e.code === 'retry');
        expect(notices.map((n) => [n.cause, n.attempts, n.max, n.outcome ?? null])).toEqual([
          ['stalled', 2, 3, 'recovered'],
          ['server', 3, 3, 'gave-up'],
        ]);
        expect(notices[1]).toMatchObject({ detail: 'API error (status 500 Internal Server Error): upstream failed' });
        // As the agent after `failed`: its turn end says error, so the stored turn reads as the live one did.
        const ends = reducer.snapshot().entries.filter((e) => e.kind === 'turn-end');
        expect(ends.at(-1)).toMatchObject({
          stopReason: 'error',
          error: 'API error (status 500 Internal Server Error): upstream failed',
        });
        const files = readdirSync(fake.root, { recursive: true, encoding: 'utf8' })
          .filter((f) => f.endsWith('updates.jsonl'))
          .map((f) => readFileSync(join(fake.root, f), 'utf8'))
          .filter((text) => text.includes('retry_state'));
        expect(files).toHaveLength(1);
        const stored = parseUpdates(files[0]!).transcript;
        expect(stored.filter((e) => e.kind === 'notice')).toEqual(notices);
        expect(stored.filter((e) => e.kind === 'turn-end')).toEqual(ends);
      } finally {
        await fake.close();
      }
    }
  });

  it('"picture" and "nomodel", Chat and Work: picture rows live and stored alike; a blocked prompt stores nothing', async () => {
    for (const mode of ['chat', 'work'] as const) {
      const fake = await fakeHost('chat-harness', { mode });
      try {
        const cwd = mode === 'work' ? { cwd: fake.project } : {};
        const { sessionId } = await fake.host.createSession({ modelId: MODEL, ...cwd });
        const reducer = new TranscriptReducer();
        reducer.userPrompt('picture please');
        const pictured = await fake.host.startTurn(sessionId, 'picture please', { modelId: MODEL, ...cwd }).done;
        expect(pictured.outcome).toBe('completed');
        for (const { line } of fake.events.updates.filter((u) => u.sessionId === sessionId)) reducer.apply(line);
        const entries = reducer.snapshot().entries;
        const pictures = entries.filter((e) => e.kind === 'notice' && e.code === 'picture');
        expect(pictures.map((p) => [p.what, p.count])).toEqual(
          mode === 'chat'
            ? [
                ['shrunk', 1],
                ['dropped', 1],
              ]
            : [['dropped', 1]],
        );
        // One-shot rows: with no call after them they go at the turn's end, live and from disk.
        const kinds = entries.map((e) => e.kind);
        const end = kinds.indexOf('turn-end');
        expect(kinds.slice(end - pictures.length, end)).toEqual(pictures.map(() => 'notice'));
        const blocked = await fake.host.startTurn(sessionId, 'nomodel now', { modelId: MODEL, ...cwd }).done;
        expect(blocked).toMatchObject({
          outcome: 'no-model',
          model: { previous: null, next: null, reason: expect.stringContaining('allowed_models') },
        });
        const files = readdirSync(fake.root, { recursive: true, encoding: 'utf8' })
          .filter((f) => f.endsWith('updates.jsonl'))
          .map((f) => readFileSync(join(fake.root, f), 'utf8'));
        expect(files).toHaveLength(1);
        expect(files[0]).not.toContain('nomodel');
        expect(files[0]).not.toContain('model_auto_switched');
        expect(parseUpdates(files[0]!).transcript.filter((e) => e.kind === 'notice')).toEqual(pictures);
        // Work reads the picture first: its row carries it, sizes only, live and stored alike.
        const reads = (list: readonly TranscriptEntry[]) =>
          list.filter((e) => e.kind === 'tool' && e.pictures !== undefined);
        expect(reads(parseUpdates(files[0]!).transcript)).toEqual(reads(entries));
        expect(reads(entries).map((e) => (e.kind === 'tool' ? [e.name, e.pictures, e.output ?? null] : null))).toEqual(
          mode === 'work' ? [['read_file', [{ n: 0, mediaType: 'image/png', bytes: expect.any(Number) }], null]] : [],
        );
        expect(files[0]!.includes('"ImageContent"')).toBe(mode === 'work');
      } finally {
        await fake.close();
      }
    }
  });

  it('"hookstop", "hookblock" and "hookdeny": the hook line, the failed row, the note, the blocked prompt and call, live and stored alike', async () => {
    // The harness waits seconds on a hook so its line can be seen; here the waits are short.
    const quick = (scenario: Json): void => {
      const lists = [scenario.prompts, (scenario.work as Json).prompts] as Json[][];
      for (const list of lists)
        for (const prompt of list) {
          if (typeof prompt.match !== 'string' || !prompt.match.startsWith('hook')) continue;
          if (prompt.promptBlocked !== undefined) (prompt.promptBlocked as Json).waitMs = 20;
          for (const step of prompt.steps as Json[]) {
            if (typeof step.sleepMs === 'number') step.sleepMs = 20;
            if (step.hookDenied !== undefined) (step.hookDenied as Json).waitMs = 20;
          }
        }
    };
    for (const mode of ['chat', 'work'] as const) {
      const fake = await fakeHost('chat-harness', { mode, mutate: quick });
      try {
        const cwd = mode === 'work' ? { cwd: fake.project } : {};
        const { sessionId } = await fake.host.createSession({ modelId: MODEL, ...cwd });
        const reducer = new TranscriptReducer();
        let fed = 0;
        const feed = (): void => {
          const lines = fake.events.updates.filter((u) => u.sessionId === sessionId).map((u) => u.line);
          for (; fed < lines.length; fed += 1) reducer.apply(lines[fed]!);
        };
        reducer.userPrompt('hookstop please');
        expect((await fake.host.startTurn(sessionId, 'hookstop please', { modelId: MODEL, ...cwd }).done).outcome).toBe(
          'completed',
        );
        feed();
        expect(
          fake.events.hookRuns.map((h) => h.hookRun && { event: h.hookRun.event, count: h.hookRun.count }),
        ).toEqual([{ event: 'stop', count: 2 }, null]);
        const blocked = await fake.host.startTurn(sessionId, 'hookblock now', { modelId: MODEL, ...cwd }).done;
        expect(blocked.outcome).toBe('hook-stopped');
        feed();
        expect(fake.events.hookRuns.at(-1)?.hookRun).toBeNull();
        if (mode === 'work') {
          reducer.userPrompt('hookdeny please');
          const deny = await fake.host.startTurn(sessionId, 'hookdeny please', { modelId: MODEL, ...cwd }).done;
          expect(deny.outcome).toBe('completed');
          feed();
        }
        const live = reducer.snapshot().entries;
        expect(live.filter((e) => e.kind === 'notice')).toEqual([
          expect.objectContaining({
            code: 'hook',
            event: 'stop',
            hook: 'global/lint',
            detail: 'exit code 127: sh: eslint: command not found',
            times: 1,
          }),
          expect.objectContaining({ code: 'hook-note', text: 'Remember to run the tests before you push.' }),
          expect.objectContaining({ code: 'hook-note', text: 'Prompt blocked by global/policy: no deploys on Friday' }),
        ]);
        const ends = live.filter((e) => e.kind === 'turn-end');
        expect(ends[1]).toMatchObject({ stopReason: 'cancelled', hookStopped: true });
        expect(ends[0]).not.toHaveProperty('hookStopped');
        const tools = live.filter((e) => e.kind === 'tool');
        expect(tools.map((t) => t.blocked ?? null)).toEqual(
          mode === 'work'
            ? ['`run_terminal_command` blocked by global/no-rm: deleting build output is not allowed here']
            : [],
        );
        const files = readdirSync(fake.root, { recursive: true, encoding: 'utf8' })
          .filter((f) => f.endsWith('updates.jsonl'))
          .map((f) => readFileSync(join(fake.root, f), 'utf8'));
        expect(files).toHaveLength(1);
        expect(files[0]).not.toContain('hook_run_started');
        expect(files[0]).not.toContain('hookblock');
        const stored = parseUpdates(files[0]!).transcript;
        expect(stored.filter((e) => e.kind !== 'user')).toEqual(live.filter((e) => e.kind !== 'user'));
      } finally {
        await fake.close();
      }
    }
  });

  it('"ended": two commands end with no wake; the agent reports both in one turn, named live and stored alike', async () => {
    const quick = (scenario: Json): void => {
      const word = ((scenario.work as Json).prompts as Json[]).find((p) => p.match === 'ended')!;
      for (const step of word.steps as Json[]) {
        const background = step.background as Json | undefined;
        if (background !== undefined) background.endAfterMs = background.command === 'npm test' ? 50 : 100;
        if (step.drain !== undefined) step.drain = { ...(step.drain as Json), afterMs: 400, wakeEndAfterMs: 0 };
      }
    };
    const fake = await fakeHost('chat-harness', { mode: 'work', mutate: quick });
    try {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      const reducer = new TranscriptReducer();
      reducer.userPrompt('ended');
      expect((await fake.host.startTurn(sessionId, 'ended', { modelId: MODEL, cwd: fake.project }).done).outcome).toBe(
        'completed',
      );
      await until(() => fake.events.ends.length === 2, 5_000);
      for (const u of fake.events.updates) if (u.sessionId === sessionId) reducer.apply(u.line);
      const title = fill(WAKE_ENDED.named, {
        what: fill(WAKE_ENDED.commands, { n: 2 }),
        labels: 'npm test, cargo build',
      });
      const wakes = (entries: readonly TranscriptEntry[]) =>
        entries.flatMap((e) => (e.kind === 'wake' ? [{ cause: e.cause, tasks: e.tasks, title: wakeTitle(e) }] : []));
      const live = wakes(reducer.snapshot().entries);
      expect(live).toMatchObject([{ cause: 'notifications', title }]);
      const files = readdirSync(fake.root, { recursive: true, encoding: 'utf8' })
        .filter((f) => f.endsWith('updates.jsonl'))
        .map((f) => readFileSync(join(fake.root, f), 'utf8'))
        .filter((text) => text.includes('notifications-'));
      expect(files).toHaveLength(1);
      expect(wakes(parseUpdates(files[0]!).transcript)).toEqual(live);
    } finally {
      await fake.close();
    }
  });

  it('"background": a dev server, a build watch, a subagent and a schedule hold the agent; the rail rows come from them', async () => {
    const fake = await fakeHost('chat-harness', { mode: 'work', timings: { activityEveryMs: 50 } });
    try {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      const turn = fake.host.startTurn(sessionId, 'background', { modelId: MODEL, cwd: fake.project });
      expect((await turn.done).outcome).toBe('completed');
      // The subagent ticks: its first numbers reach the live facts. A task's or a schedule's start can send the
      // live facts before the first tick does.
      const ticked = () => fake.events.activity.find((e) => e.live.helpers.length > 0);
      await until(() => ticked() !== undefined, 5_000);
      expect(ticked()!.live.helpers).toMatchObject([{ turns: 1, toolCalls: 1, contextPct: 6 }]);
      expect(fake.host.modeState()).toMatchObject({ tasks: 2, background: 1 });
      expect(fake.host.active()).toMatchObject([{ sessionId, background: true, helpers: true, tasks: 2 }]);
      const reducer = new TranscriptReducer();
      for (const u of fake.events.updates) reducer.apply(u.line);
      const entries = reducer.snapshot().entries;
      const tasks = entries.flatMap((e) => (e.kind === 'tool' && e.task !== undefined ? [e.task] : []));
      expect(tasks.map((t) => [t.kind, t.label, t.endTs])).toEqual([
        ['command', 'Start the dev server', undefined],
        ['monitor', 'Watch the build log', undefined],
      ]);
      expect(entries.some((e) => e.kind === 'subagent' && e.description === 'Find the flaky test')).toBe(true);
      expect(entries.some((e) => e.kind === 'tool' && e.schedule?.prompt === 'Check the deploy')).toBe(true);
    } finally {
      await fake.close();
    }
  });

  it('"background": the schedule fires once; its row reads Running now while the run it started runs', async () => {
    const quick = (scenario: Json): void => {
      const word = ((scenario.work as Json).prompts as Json[]).find((p) => p.match === 'background')!;
      for (const step of word.steps as Json[]) {
        if (step.schedule !== undefined) step.schedule = { ...(step.schedule as Json), fireAfterMs: 100, runMs: 300 };
      }
    };
    const fake = await fakeHost('chat-harness', { mode: 'work', mutate: quick, timings: { activityEveryMs: 50 } });
    try {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      expect(
        (await fake.host.startTurn(sessionId, 'background', { modelId: MODEL, cwd: fake.project }).done).outcome,
      ).toBe('completed');
      await until(() => fake.host.liveActivity(sessionId).schedules[0]?.firing != null, 5_000);
      const run = fake.host.liveActivity(sessionId).schedules[0]!.firing!;
      const reducer = new TranscriptReducer();
      for (const u of fake.events.updates) if (u.sessionId === sessionId) reducer.apply(u.line);
      const entries = reducer.snapshot().entries;
      expect(
        entries.some((e) => e.kind === 'subagent' && e.childId === run && e.description === 'Check the deploy'),
      ).toBe(true);
      const rows = activityOf({ entries, runs: [], live: fake.host.liveActivity(sessionId), turnRunning: false });
      expect(rows.scheduled.map((row) => [row.title, row.state])).toEqual([['Check the deploy', 'firing']]);
      await until(() => fake.host.liveActivity(sessionId).schedules[0]?.firing === null, 5_000);
    } finally {
      await fake.close();
    }
  });

  it('"background": the build watch reports seven lines; the rail keeps the last five, and one wake reads them', async () => {
    const quick = (scenario: Json): void => {
      const word = ((scenario.work as Json).prompts as Json[]).find((p) => p.match === 'background')!;
      for (const step of word.steps as Json[]) {
        if (step.monitorEvents !== undefined)
          step.monitorEvents = { ...(step.monitorEvents as Json), afterMs: 300, everyMs: 10, wakeAfterMs: 0 };
      }
    };
    const fake = await fakeHost('chat-harness', { mode: 'work', mutate: quick, timings: { activityEveryMs: 50 } });
    try {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
      expect(
        (await fake.host.startTurn(sessionId, 'background', { modelId: MODEL, cwd: fake.project }).done).outcome,
      ).toBe('completed');
      await until(() => fake.host.liveActivity(sessionId).reports.length === 1, 5_000);
      const live = fake.host.liveActivity(sessionId);
      const watch = live.tasks.find((t) => t.kind === 'monitor')!;
      expect(watch.events).toBe(7);
      expect(watch.recent.map((e) => e.text)).toEqual([
        "src/routes/cart.ts(48,17): error TS2345: Argument of type 'string' is not assignable to parameter of type 'number'.",
        '[build] 1 error, 1 warning',
        '[build] Watching for file changes…',
        '[build] File change detected. Compiling…',
        "src/routes/cart.ts(48,17): error TS2345: Argument of type 'string' is not assignable to parameter of type 'number'.",
      ]);
      expect(live.reports[0]!.monitors.map((m) => m.id)).toEqual([watch.id]);
      const reducer = new TranscriptReducer();
      reducer.userPrompt('background');
      for (const u of fake.events.updates) if (u.sessionId === sessionId) reducer.apply(u.line);
      const entries = reducer.snapshot().entries;
      const wake = entries.find((e) => e.kind === 'wake');
      if (wake?.kind !== 'wake') throw new Error('no wake row');
      expect(wakeTitle(wake, wakeReportOf(wake, wakeReportsOf(live, entries)))).toBe(
        fill(WAKE_ENDED.named, { what: WAKE_REPORTED.one, labels: 'Watch the build log' }),
      );
    } finally {
      await fake.close();
    }
  });
});

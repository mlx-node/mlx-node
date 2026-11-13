import { mkdirSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import type { Interaction } from '../src/chat/types.js';
import { applyConversationEvent, type ConversationState } from '../ui/src/lib/chat-store.js';
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
});

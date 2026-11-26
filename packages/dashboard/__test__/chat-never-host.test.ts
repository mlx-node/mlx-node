/**
 * Never allow on the scripted fake agent: "Never allow" on a command card keeps the rule and denies the call
 * with the rule's note, so the answer goes on; a later match gets no card in any mode and from any agent; a model
 * that keeps trying is stopped; Bypass asks for a command the app cannot read when the folder has rules.
 */

import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { parseUpdates } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { InteractionError, type RuleDeny } from '../src/chat/agent/agent-host.js';
import { neverNote, NeverRules } from '../src/chat/agent/never-rules.js';
import type { PermissionInteraction, PermissionMark, PermissionMode } from '../src/chat/types.js';
import { applyConversationEvent, type ConversationState } from '../ui/src/lib/chat-store.js';
import {
  fakeHost,
  fakeService,
  noProviders,
  until,
  type FakeHost,
  type FakeHostOptions,
  type FakeService,
  type Json,
} from './helpers/chat-fake.js';

const MODEL = 'qwen3.5-4b';
const END = [{ text: 'Done.', chunk: 5 }, { end: { stopReason: 'end_turn' } }];

function bashStep(command: string): Json {
  return {
    permission: {
      toolName: 'run_terminal_command',
      kind: 'execute',
      title: `Execute \`${command}\``,
      rawInput: { variant: 'Bash', command, description: 'run', is_background: false },
      output: 'ok',
    },
  };
}

/** A `monitor` call as v0.3.4 asks for it: its `command` runs in the background terminal; the Work guard asks. */
function monitorStep(command: unknown): Json {
  return {
    permission: {
      toolName: 'monitor',
      kind: 'execute',
      title: 'monitor',
      rawInput: { variant: 'Monitor', command, description: 'watch', timeout_ms: 36_000_000, persistent: false },
      hookAsk: 'guard',
      output: 'started',
    },
  };
}

function script(steps: Json[]): (scenario: Json) => void {
  return (scenario) => {
    scenario.prompts = [{ match: '', steps: [...steps, ...END] }];
  };
}

let dir: string;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'never-host-'));
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

interface Run {
  fake: FakeHost;
  id: string;
  /** The session's Work folder, as the host keeps it (its real path). */
  cwd: string;
  never: NeverRules;
  denied: RuleDeny[];
  mode: { now: PermissionMode };
}

async function withNever(
  steps: Json[] | null,
  run: (r: Run) => Promise<void>,
  options: FakeHostOptions & { permissionMode?: PermissionMode; name?: string; store?: NeverRules } = {},
): Promise<void> {
  const never = options.store ?? new NeverRules(join(dir, 'never-allow.json'));
  await never.ready();
  const denied: RuleDeny[] = [];
  const mode: { now: PermissionMode } = { now: options.permissionMode ?? 'ask' };
  const fake = await fakeHost(options.name ?? 'work-permission', {
    mode: 'work',
    providers: noProviders(),
    mutate: steps === null ? options.mutate : script(steps),
    permission: { modeOf: () => mode.now, watched: () => true },
    onModeAllowed: async () => true,
    never,
    onRuleDenied: (_id, deny) => denied.push(deny),
  });
  try {
    const created = await fake.host.createSession({ modelId: MODEL, cwd: fake.project });
    await run({ fake, id: created.sessionId, cwd: created.cwd, never, denied, mode });
  } finally {
    await fake.close();
  }
}

function start(fake: FakeHost, id: string, text = 'go') {
  return fake.host.startTurn(id, text, { modelId: MODEL, cwd: fake.project });
}

async function nth(fake: FakeHost, n: number): Promise<PermissionInteraction> {
  await until(() => fake.events.interactions.length >= n, 5_000);
  return fake.events.interactions[n - 1]!.interaction as PermissionInteraction;
}

/** The results the host sent back for the agent's own requests, in order. */
function answers(fake: FakeHost): unknown[] {
  return fake
    .sent()
    .filter((m) => m.method === undefined && m.result !== undefined && typeof m.result === 'object')
    .map((m) => m.result)
    .filter((r) => 'outcome' in (r as Json));
}

const NOTED = (words: string): Json => ({
  outcome: { outcome: 'selected', optionId: 'reject-once' },
  _meta: { followup_message: neverNote(words) },
});
const PLAIN_DENY = { outcome: { outcome: 'selected', optionId: 'reject-once' } };

describe('Never allow on a card', () => {
  it('offers both scopes; keeps the rule, denies with its note, and the answer goes on', async () => {
    await withNever(
      [bashStep('git push origin main'), bashStep('git push --force')],
      async ({ fake, id, cwd, never, denied }) => {
        const turn = start(fake, id);
        const card = await nth(fake, 1);
        expect(card.never).toEqual({ short: 'git push', exact: 'git push origin main' });
        const { denied: facts, saved } = fake.host.answerNever(id, card.id, 'short');
        expect(facts).toMatchObject({ toolCallId: card.toolCallId, access: 'bash', words: 'git push', noted: true });
        expect(await saved).toBe(true);
        expect(never.rulesOf(cwd).map((r) => r.words)).toEqual([['git', 'push']]);
        expect(facts.ruleId).toBe(never.rulesOf(cwd)[0]!.id);
        expect((await turn.done).outcome).toBe('completed');
        // The second command matched the rule: no card, the same note.
        expect(fake.events.interactions).toHaveLength(1);
        expect(answers(fake)).toEqual([NOTED('git push'), NOTED('git push')]);
        expect(denied).toHaveLength(1);
        expect(denied[0]).toMatchObject({ words: 'git push', noted: true, ruleId: facts.ruleId });
        expect(denied[0]!.card?.command).toBe('git push --force');
        expect(fake.records().filter((r) => r.kind === 'note')).toHaveLength(2);
      },
    );
  });

  it('the exact scope keeps every word; a command that offers one scope shows it twice', async () => {
    await withNever([bashStep('git push origin main'), bashStep('git push')], async ({ fake, id, cwd, never }) => {
      const turn = start(fake, id);
      const card = await nth(fake, 1);
      await fake.host.answerNever(id, card.id, 'exact').saved;
      expect(never.rulesOf(cwd).map((r) => r.words)).toEqual([['git', 'push', 'origin', 'main']]);
      const second = await nth(fake, 2);
      expect(second.never).toEqual({ short: 'git push', exact: 'git push' });
      fake.host.answer(id, second.id, 'allow-once');
      expect((await turn.done).outcome).toBe('completed');
    });
  });

  it('an edit card offers no rule, and a never answer to it is refused', async () => {
    await withNever(
      [
        {
          permission: {
            toolName: 'write',
            kind: 'edit',
            title: 'Write a.txt',
            rawInput: { variant: 'Write', file_path: 'a.txt', content: 'x\n' },
            output: 'wrote',
          },
        },
      ],
      async ({ fake, id }) => {
        const turn = start(fake, id);
        const card = await nth(fake, 1);
        expect(card.never).toBeUndefined();
        expect(() => fake.host.answerNever(id, card.id, 'short')).toThrow(InteractionError);
        fake.host.answer(id, card.id, 'allow-once');
        expect((await turn.done).outcome).toBe('completed');
      },
    );
  });

  it('a rule the file could not keep still denies this call with the note; the next match asks again', async () => {
    const store = new NeverRules(join(dir, 'missing', 'never-allow.json'));
    // The rule matches until its write fails: a request sent before that is denied by it, so the next one waits.
    const gate = join(dir, 'next');
    await withNever(
      [bashStep('git push'), { holdOn: gate }, bashStep('git push')],
      async ({ fake, id, cwd, never }) => {
        const turn = start(fake, id);
        const card = await nth(fake, 1);
        expect(await fake.host.answerNever(id, card.id, 'short').saved).toBe(false);
        expect(never.rulesOf(cwd)).toEqual([]);
        writeFileSync(gate, '');
        const second = await nth(fake, 2);
        expect(answers(fake)).toEqual([NOTED('git push')]);
        fake.host.answer(id, second.id, 'allow-once');
        expect((await turn.done).outcome).toBe('completed');
      },
      { store },
    );
  });
});

describe('a later match', () => {
  for (const mode of ['ask', 'auto', 'bypass'] as const) {
    it(`gets no card in ${mode}: denied with the note, and the answer goes on`, async () => {
      await withNever(
        [bashStep('cd sub && git push origin')],
        async ({ fake, id, cwd, never, denied }) => {
          const made = never.add(cwd, ['git', 'push']);
          await made.saved;
          const turn = start(fake, id);
          expect((await turn.done).outcome).toBe('completed');
          expect(fake.events.interactions).toHaveLength(0);
          expect(answers(fake)).toEqual([NOTED('git push')]);
          expect(denied).toEqual([
            expect.objectContaining({ ruleId: made.rule!.id, words: 'git push', noted: true, access: 'bash' }),
          ]);
          expect(denied[0]!.child).toBeUndefined();
        },
        { permissionMode: mode },
      );
    });
  }

  for (const mode of ['ask', 'auto', 'bypass'] as const) {
    it(`a monitor's command too, in ${mode}: no card, denied with the note`, async () => {
      await withNever(
        [monitorStep('git push origin && echo pushed')],
        async ({ fake, id, cwd, never, denied }) => {
          const made = never.add(cwd, ['git', 'push']);
          await made.saved;
          const turn = start(fake, id);
          expect((await turn.done).outcome).toBe('completed');
          expect(fake.events.interactions).toHaveLength(0);
          expect(answers(fake)).toEqual([NOTED('git push')]);
          expect(fake.records().filter((r) => r.kind === 'exec')).toEqual([]);
          expect(denied).toEqual([
            expect.objectContaining({ ruleId: made.rule!.id, words: 'git push', noted: true, access: 'other' }),
          ]);
          expect(denied[0]!.card?.toolName).toBe('monitor');
        },
        { permissionMode: mode },
      );
    });
  }

  it("a helper's request too: no card, the note goes to the helper", async () => {
    await withNever(
      null,
      async ({ fake, id, cwd, never, denied }) => {
        await never.add(cwd, ['git', 'push']).saved;
        const turn = start(fake, id, 'research');
        const startCard = await nth(fake, 1);
        expect(startCard.toolName).toBe('workflow');
        fake.host.answer(id, startCard.id, 'allow-once');
        await until(() => fake.records().some((r) => r.kind === 'child-answer'), 8_000);
        expect(fake.records().find((r) => r.kind === 'child-answer')).toMatchObject({
          answer: 'reject-once',
          note: neverNote('git push'),
        });
        expect(fake.events.interactions).toHaveLength(1);
        expect(denied).toHaveLength(1);
        expect(denied[0]!.child).toBeDefined();
        expect(denied[0]!.child).not.toBe(id);
        await turn.done;
      },
      {
        name: 'work-workflow-run',
        mutate: (scenario) => {
          const steps = (scenario.prompts as Json[])[0]!.steps as Json[];
          for (const step of steps)
            if (step.workflow !== undefined) (step.workflow as Json).command = 'git push origin';
        },
      },
    );
  });

  it('the fourth hit in one answer is a plain deny, and the answer ends never-loop', async () => {
    const pushes = [1, 2, 3, 4].map((n) => bashStep(`git push try${n}`));
    await withNever(pushes, async ({ fake, id, cwd, never, denied }) => {
      await never.add(cwd, ['git', 'push']).saved;
      const turn = start(fake, id);
      expect((await turn.done).outcome).toBe('never-loop');
      expect(fake.events.interactions).toHaveLength(0);
      expect(answers(fake)).toEqual([NOTED('git push'), NOTED('git push'), NOTED('git push'), PLAIN_DENY]);
      expect(denied.map((d) => d.noted)).toEqual([true, true, true, false]);
    });
  });

  it('a removed rule: the next request opens a card again', async () => {
    await withNever([bashStep('git push')], async ({ fake, id, cwd, never }) => {
      const made = never.add(cwd, ['git', 'push']);
      await made.saved;
      expect(await never.remove(cwd, made.rule!.id)).toBe('removed');
      const turn = start(fake, id);
      const card = await nth(fake, 1);
      expect(card.command).toBe('git push');
      fake.host.answer(id, card.id, 'allow-once');
      expect((await turn.done).outcome).toBe('completed');
    });
  });

  it('a rule of another folder does not apply', async () => {
    await withNever([bashStep('git push')], async ({ fake, id, cwd, never }) => {
      await never.add(join(cwd, 'elsewhere'), ['git', 'push']).saved;
      const turn = start(fake, id);
      const card = await nth(fake, 1);
      fake.host.answer(id, card.id, 'allow-once');
      expect((await turn.done).outcome).toBe('completed');
    });
  });
});

describe('Bypass and a command the app cannot read', () => {
  it('asks when the folder has rules: the card says why', async () => {
    await withNever(
      [bashStep('echo $(whoami)')],
      async ({ fake, id, cwd, never }) => {
        await never.add(cwd, ['git', 'push']).saved;
        const turn = start(fake, id);
        const card = await nth(fake, 1);
        expect(card.neverUnread).toBe(true);
        expect(card.never).toBeUndefined();
        fake.host.answer(id, card.id, 'allow-once');
        expect((await turn.done).outcome).toBe('completed');
      },
      { permissionMode: 'bypass' },
    );
  });

  it('a monitor whose command the app cannot read asks too; one with no command text asks as well', async () => {
    await withNever(
      [monitorStep('tail -f $(ls *.log)'), monitorStep(['tail', '-f', 'x.log'])],
      async ({ fake, id, cwd, never }) => {
        await never.add(cwd, ['git', 'push']).saved;
        const turn = start(fake, id);
        for (const n of [1, 2]) {
          const card = await nth(fake, n);
          expect(card.neverUnread).toBe(true);
          fake.host.answer(id, card.id, 'allow-once');
        }
        expect((await turn.done).outcome).toBe('completed');
      },
      { permissionMode: 'bypass' },
    );
  });

  it('allows as before when the folder has none', async () => {
    await withNever(
      [bashStep('echo $(whoami)')],
      async ({ fake, id }) => {
        const turn = start(fake, id);
        expect((await turn.done).outcome).toBe('completed');
        expect(fake.events.interactions).toHaveLength(0);
        expect(answers(fake)).toEqual([{ outcome: { outcome: 'selected', optionId: 'allow-once' } }]);
      },
      { permissionMode: 'bypass' },
    );
  });
});

describe('the service', () => {
  /** The service's stream of a conversation, folded by the page's reducer. */
  function watchState(fake: FakeService, id: string): { state: () => ConversationState | null; off: () => void } {
    let state: ConversationState | null = null;
    const off = fake.service.watchConversation(id, (event) => {
      const next = applyConversationEvent(state, event);
      if (next !== 'resync') state = next;
    });
    return { state: () => state, off };
  }

  function storedUpdates(fake: FakeService, id: string): string {
    const root = join(fake.root, 'agent-home', 'sessions');
    for (const name of readdirSync(root)) {
      const path = join(root, name, id, 'updates.jsonl');
      if (existsSync(path)) return readFileSync(path, 'utf8');
    }
    throw new Error('no stored session');
  }

  const ruleMarks = (marks: readonly PermissionMark[]): Array<Extract<PermissionMark, { kind: 'denied' }>> =>
    marks.filter((m): m is Extract<PermissionMark, { kind: 'denied' }> => m.kind === 'denied' && m.by === 'rule');
  const notes = (entries: readonly TranscriptEntry[]): TranscriptEntry[] =>
    entries.filter((e) => e.kind === 'user' && e.note === true);

  it('a card answer and a later match: marks, rule rows, history lines, the count; a removal reads "rule removed"', async () => {
    const fake = fakeService('work-permission', {
      mutate: script([bashStep('git push origin main'), bashStep('git push --force')]),
    });
    const project = join(fake.root, 'home', 'code', 'app');
    mkdirSync(project, { recursive: true });
    const window = fake.service.watchHost(() => undefined);
    try {
      const { conversation } = await fake.service.create({ mode: 'work', text: 'push it', cwd: project });
      const id = conversation.id;
      const page = watchState(fake, id);
      try {
        await until(() => (page.state()?.interactions.length ?? 0) === 1, 5_000);
        const card = page.state()!.interactions[0] as PermissionInteraction;
        const reply = await fake.service.answer(id, card.id, { never: 'short' });
        expect(reply).toEqual({ never: { saved: true, ruleId: expect.any(String), words: 'git push' } });
        const ruleId = (reply.never as { ruleId: string }).ruleId;
        await until(() => page.state()?.turn === null && page.state()?.lastEnd !== undefined, 8_000);
        expect(page.state()!.lastEnd?.outcome).toBe('completed');
        expect(page.state()!.conversation.neverRules).toBe(1);
        const marks = ruleMarks(page.state()!.permissionMarks);
        expect(marks).toEqual([
          expect.objectContaining({
            toolCallId: card.toolCallId,
            ruleId,
            words: 'git push',
            card: { ...card, options: [] },
          }),
          expect.objectContaining({ ruleId, words: 'git push' }),
        ]);
        expect(marks[1]!.card?.command).toBe('git push --force');
        const live = notes(page.state()!.entries);
        expect(live.map((e) => (e.kind === 'user' ? e.text : ''))).toEqual([
          neverNote('git push'),
          neverNote('git push'),
        ]);
        // A history line is written after its mark is sent, so the second one can land after the answer's end.
        const logText = (): string => {
          const path = join(fake.paths.permissionLog, `${id}.jsonl`);
          return existsSync(path) ? readFileSync(path, 'utf8') : '';
        };
        await until(
          () =>
            logText()
              .split('\n')
              .filter((l) => l !== '').length === 2,
          5_000,
        );
        const lines = logText()
          .split('\n')
          .filter((l) => l !== '')
          .map((l) => JSON.parse(l) as Json);
        expect(lines).toEqual([
          { kind: 'denied', at: marks[0]!.at, toolCallId: card.toolCallId, access: 'bash', by: 'rule', ruleId },
          { kind: 'denied', at: marks[1]!.at, toolCallId: marks[1]!.toolCallId, access: 'bash', by: 'rule', ruleId },
        ]);
        expect(logText()).not.toContain('git push');
        // The agent stored each note twice; the reducer reads them as the live rows.
        const disk = notes(parseUpdates(storedUpdates(fake, id)).transcript);
        expect(disk.map((e) => ({ ...e, ts: 0 }))).toEqual(live.map((e) => ({ ...e, ts: 0 })));
        expect(await fake.service.neverRules(id)).toEqual({
          folder: page.state()!.conversation.cwd,
          rules: [{ id: ruleId, words: 'git push', at: expect.any(Number) }],
        });
        await fake.service.removeNeverRule(id, ruleId);
        await until(() => page.state()?.conversation.neverRules === undefined, 5_000);
        expect(ruleMarks(page.state()!.permissionMarks).map((m) => m.words)).toEqual([null, null]);
        await expect(fake.service.removeNeverRule(id, ruleId)).rejects.toMatchObject({ code: 'E_NOT_FOUND' });
      } finally {
        page.off();
      }
      // A Control Panel restart reads the marks again: the rule is gone.
      await fake.service.close();
      fake.restart();
      const again = watchState(fake, id);
      try {
        await until(() => again.state() !== null, 5_000);
        expect(ruleMarks(again.state()!.permissionMarks).map((m) => [m.ruleId !== undefined, m.words])).toEqual([
          [true, null],
          [true, null],
        ]);
        expect(again.state()!.conversation.neverRules).toBeUndefined();
        expect(notes(again.state()!.entries)).toHaveLength(2);
      } finally {
        again.off();
      }
    } finally {
      window();
      await fake.close();
    }
  }, 30_000);

  it('a kept rule reads its words after a restart, and every conversation of the folder gets the count', async () => {
    const fake = fakeService('work-permission', { mutate: script([bashStep('rm -rf build')]) });
    const project = join(fake.root, 'home', 'code', 'app');
    mkdirSync(project, { recursive: true });
    const window = fake.service.watchHost(() => undefined);
    try {
      const first = await fake.service.create({ mode: 'work', text: 'clean', cwd: project });
      const page = watchState(fake, first.conversation.id);
      let ruleId = '';
      try {
        await until(() => (page.state()?.interactions.length ?? 0) === 1, 5_000);
        const card = page.state()!.interactions[0] as PermissionInteraction;
        expect(card.never).toEqual({ short: 'rm', exact: 'rm -rf build' });
        const reply = await fake.service.answer(first.conversation.id, card.id, { never: 'exact' });
        ruleId = (reply.never as { ruleId: string }).ruleId;
        await until(() => page.state()?.turn === null && page.state()?.lastEnd !== undefined, 8_000);
      } finally {
        page.off();
      }
      const second = await fake.service.create({ mode: 'work', text: 'clean', cwd: project });
      expect(second.conversation.neverRules).toBe(1);
      const other = watchState(fake, second.conversation.id);
      try {
        await until(() => other.state()?.turn === null && other.state()?.lastEnd !== undefined, 8_000);
        // The second answer's command matched the rule made in the first conversation.
        expect(ruleMarks(other.state()!.permissionMarks)).toEqual([
          expect.objectContaining({ ruleId, words: 'rm -rf build' }),
        ]);
      } finally {
        other.off();
      }
      await fake.service.close();
      fake.restart();
      const again = watchState(fake, first.conversation.id);
      try {
        await until(() => again.state() !== null, 5_000);
        expect(ruleMarks(again.state()!.permissionMarks)).toEqual([
          expect.objectContaining({ ruleId, words: 'rm -rf build', card: null }),
        ]);
        expect(again.state()!.conversation.neverRules).toBe(1);
      } finally {
        again.off();
      }
    } finally {
      window();
      await fake.close();
    }
  }, 30_000);
});

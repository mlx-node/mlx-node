/**
 * Work permission and plan requests on the scripted fake agent: what the card shows (the whole command or path, the
 * options the agent offered, the warnings), the answers and how each ends the turn, the "always" options the agent
 * keeps, the deny when no page is left to answer, and the notice when a command prints a provider key.
 */

import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import fsPromises from 'node:fs/promises';
import { syncBuiltinESMExports } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { afterEach, describe, expect, it } from 'vite-plus/test';

import { parseUpdates, TranscriptReducer } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { AgentHostError, FAILURE_MESSAGES, InteractionError } from '../src/chat/agent/agent-host.js';
import { PDF_READER_TEXT, readPdfScript } from '../src/chat/agent/pdf-tool.js';
import { WORK_GUARD_PREFIX, workGuardHookName, workGuardManifest } from '../src/chat/agent/profiles.js';
import { REDACTED } from '../src/chat/redact.js';
import type { McpInteraction, PermissionInteraction, PlanInteraction } from '../src/chat/types.js';
import { wakeReportOf, wakeReportsOf } from '../ui/src/lib/activity.js';
import { applyConversationEvent, type ConversationState } from '../ui/src/lib/chat-store.js';
import { fill } from '../ui/src/lib/chat-strings.js';
import { WAKE_ENDED, WAKE_REPORTED, wakeTitle } from '../ui/src/lib/sessions.js';
import {
  fakeHost,
  fakeService,
  noProviders,
  ended,
  storedWhen,
  until,
  type FakeHost,
  type FakeHostOptions,
  type FakeService,
  type Json,
} from './helpers/chat-fake.js';
import { runView } from './helpers/run-views.js';

const MODEL = 'qwen3.5-4b';
const KEY = 'exa-KEY-0123456789abcdef';

async function withWork(name: string, run: (fake: FakeHost) => Promise<void>, options: FakeHostOptions = {}) {
  const fake = await fakeHost(name, { mode: 'work', providers: noProviders(), ...options });
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

async function nth<T extends 'permission' | 'plan'>(
  fake: FakeHost,
  n: number,
  kind: T,
): Promise<T extends 'plan' ? PlanInteraction : PermissionInteraction> {
  await until(() => fake.events.interactions.length >= n);
  const interaction = fake.events.interactions[n - 1]!.interaction;
  expect(interaction.kind).toBe(kind);
  return interaction as T extends 'plan' ? PlanInteraction : PermissionInteraction;
}

/** The results the host sent back for the agent's own requests, in order. */
function answers(fake: FakeHost): unknown[] {
  return fake
    .sent()
    .filter((m) => m.method === undefined && m.result !== undefined && typeof m.result === 'object')
    .map((m) => m.result)
    .filter((r) => 'outcome' in (r as Json));
}

function texts(fake: FakeHost): string {
  return fake.events.updates
    .map((u) => {
      const content = u.line.update.content as Json | undefined;
      return typeof content?.text === 'string' ? content.text : '';
    })
    .join('');
}

function permissionStep(scenario: Json, n = 0): Json {
  return ((scenario.prompts as Json[])[0]!.steps as Json[])[n]!.permission as Json;
}

function withCommand(command: string): (scenario: Json) => void {
  return (scenario) => {
    const step = permissionStep(scenario);
    step.title = 'Execute';
    step.rawInput = { variant: 'Bash', command, description: 'run', is_background: false };
  };
}

describe('the Work guard', () => {
  it('the host writes the guard plugin and sends it first; the shell card drops the guard hook line', async () => {
    await withWork('work-always', async (fake) => {
      const id = await session(fake);
      const file = join(fake.paths.workGuard, 'plugin.json');
      const name = (JSON.parse(readFileSync(file, 'utf8')) as { name: string }).name;
      expect(name.startsWith(WORK_GUARD_PREFIX)).toBe(true);
      expect(readFileSync(file, 'utf8')).toBe(workGuardManifest(name));
      expect(statSync(file).mode & 0o777).toBe(0o600);
      const created = fake.sent().find((m) => m.method === 'session/new') as Json;
      expect(((created.params as Json)._meta as Json).pluginDirs).toEqual([fake.paths.workGuard]);
      const turn = start(fake, id);
      const card = await nth(fake, 1, 'permission');
      const asks = fake.records().filter((r) => r.kind === 'ask');
      expect(asks.map((r) => r.hookAsk)).toEqual([workGuardHookName(name)]);
      expect(card).toMatchObject({ access: 'bash', command: 'ls', title: 'Execute `ls`' });
      fake.host.answer(id, card.id, 'allow-once');
      fake.host.answer(id, (await nth(fake, 2, 'permission')).id, 'allow-once');
      expect((await turn.done).outcome).toBe('completed');
      expect(fake.events.interactions.map((e) => (e.interaction as PermissionInteraction).title)).toEqual([
        'Execute `ls`',
        'Execute `ls`',
      ]);
    });
  });
});

describe('the PDF tool files', () => {
  // Mutation: the files not written at the Work agent start → red.
  it('the Work agent start writes the workflow and its reader agent into the agent home', async () => {
    await withWork('work-always', async (fake) => {
      expect(fake.host.pdfTool()).toBeNull();
      await session(fake);
      expect(fake.host.pdfTool()).toBe(true);
      expect(readFileSync(join(fake.agentHome, 'workflows', 'read-pdf.rhai'), 'utf8')).toBe(
        readPdfScript(fake.paths.workPdf),
      );
      expect(readFileSync(join(fake.agentHome, 'agents', 'pdf-reader.md'), 'utf8')).toBe(PDF_READER_TEXT);
    });
  });

  // Mutation: a person's file replaced at the agent start → red.
  it("a file of the person's under either name stays as it is, and the tool is off", async () => {
    const theirs = '---\nname: read-pdf\n---\nmine\n';
    await withWork(
      'work-always',
      async (fake) => {
        await session(fake);
        expect(fake.host.pdfTool()).toBe(false);
        expect(readFileSync(join(fake.agentHome, 'agents', 'pdf-reader.md'), 'utf8')).toBe(theirs);
        expect(existsSync(join(fake.agentHome, 'workflows', 'read-pdf.rhai'))).toBe(false);
      },
      {
        mutate: (scenario) => {
          const home = join(dirname(String(scenario.recordTo)), 'agent-home');
          mkdirSync(join(home, 'agents'), { recursive: true });
          writeFileSync(join(home, 'agents', 'pdf-reader.md'), theirs);
        },
      },
    );
  });
});

describe('Work permission cards', () => {
  it('a bash request offers its always option; the agent keeps the grant for the next call and the next turn', async () => {
    await withWork('work-always', async (fake) => {
      const id = await session(fake);
      const turn = start(fake, id);
      const card = await nth(fake, 1, 'permission');
      expect(card).toMatchObject({ access: 'bash', command: 'ls', path: null, flags: [] });
      expect(card.options.map((o) => [o.optionId, o.kind])).toEqual([
        ['always-allow', 'allow_always'],
        ['allow-once', 'allow_once'],
        ['reject-once', 'reject_once'],
      ]);
      fake.host.answer(id, card.id, 'always-allow');
      expect((await turn.done).outcome).toBe('completed');
      expect(fake.events.interactions).toHaveLength(1);
      expect(texts(fake)).toContain('Listed twice.');
      expect((await start(fake, id, 'again').done).outcome).toBe('completed');
      expect(fake.events.interactions).toHaveLength(1);
      expect(answers(fake)).toEqual([{ outcome: { outcome: 'selected', optionId: 'always-allow' } }]);
    });
  });

  it('allow-once asks again on the next call', async () => {
    await withWork('work-always', async (fake) => {
      const id = await session(fake);
      const turn = start(fake, id);
      fake.host.answer(id, (await nth(fake, 1, 'permission')).id, 'allow-once');
      fake.host.answer(id, (await nth(fake, 2, 'permission')).id, 'allow-once');
      expect((await turn.done).outcome).toBe('completed');
    });
  });

  it('a file write shows the whole path and the session-wide edit option, which the agent honours', async () => {
    await withWork('work-edit', async (fake) => {
      const id = await session(fake);
      const turn = start(fake, id);
      const card = await nth(fake, 1, 'permission');
      expect(card).toMatchObject({ access: 'edit', toolName: 'write', command: null, path: 'notes.md', flags: [] });
      expect(card.options.map((o) => o.optionId)).toEqual(['allow-edits-session', 'allow-once', 'reject-once']);
      fake.host.answer(id, card.id, 'allow-edits-session');
      expect((await turn.done).outcome).toBe('completed');
      expect(fake.events.interactions).toHaveLength(1);
      expect(texts(fake)).toContain('Wrote two files.');
    });
  });

  it('a write outside the work folder is flagged; an always option of the wrong access is never listed', async () => {
    await withWork(
      'work-edit',
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const card = await nth(fake, 1, 'permission');
        expect(card.path).toBe('/etc/hosts');
        expect(card.flags).toEqual(['outside-folder']);
        expect(card.options.map((o) => o.optionId)).toEqual(['allow-once', 'reject-once']);
        expect(() => fake.host.answer(id, card.id, 'always-allow')).toThrow();
        fake.host.answer(id, card.id, 'reject-once');
        expect((await turn.done).outcome).toBe('denied');
      },
      {
        mutate: (scenario) => {
          const step = permissionStep(scenario);
          step.rawInput = { variant: 'Write', file_path: '/etc/hosts', content: 'x' };
          step.options = [
            { optionId: 'always-allow', name: 'always', kind: 'allow_always' },
            { optionId: 'allow-once', name: 'Yes', kind: 'allow_once' },
            { optionId: 'reject-once', name: 'No', kind: 'reject_once' },
          ];
        },
      },
    );
  });

  it('a write through a link in the work folder shows where it lands and is flagged as outside', async () => {
    await withWork(
      'work-edit',
      async (fake) => {
        const outside = join(fake.root, 'outside');
        mkdirSync(outside);
        symlinkSync(outside, join(fake.project, 'docs'));
        const id = await session(fake);
        const turn = start(fake, id);
        const card = await nth(fake, 1, 'permission');
        expect([card.path, card.resolvedPath, card.flags]).toEqual([
          'docs/x.plist',
          join(realpathSync(outside), 'x.plist'),
          ['outside-folder'],
        ]);
        fake.host.answer(id, card.id, 'reject-once');
        expect((await turn.done).outcome).toBe('denied');
      },
      {
        mutate: (scenario) => {
          permissionStep(scenario).rawInput = { variant: 'Write', file_path: 'docs/x.plist', content: 'x' };
        },
      },
    );
  });

  it('an edit whose `..` follows a linked folder out of the work folder shows where it lands, flagged', async () => {
    await withWork(
      'work-edit',
      async (fake) => {
        const outside = join(fake.root, 'outside');
        mkdirSync(join(outside, 'deep'), { recursive: true });
        writeFileSync(join(outside, 'victim.txt'), 'v');
        symlinkSync(join(outside, 'deep'), join(fake.project, 'link'));
        const id = await session(fake);
        const turn = start(fake, id);
        const card = await nth(fake, 1, 'permission');
        expect([card.path, card.resolvedPath, card.flags]).toEqual([
          'link/../victim.txt',
          join(realpathSync(outside), 'victim.txt'),
          ['outside-folder'],
        ]);
        fake.host.answer(id, card.id, 'reject-once');
        expect((await turn.done).outcome).toBe('denied');
      },
      {
        mutate: (scenario) => {
          permissionStep(scenario).rawInput = { variant: 'Write', file_path: 'link/../victim.txt', content: 'x' };
        },
      },
    );
  });

  it('a command too long to check opens a card with the whole command and the unchecked warning', async () => {
    let nested = 'sh;'.repeat(33_000);
    for (let k = 0; k < 5; k++) nested = `sh -c $(${nested})`;
    const command = `rm a; ${nested}`;
    await withWork(
      'work-flagged',
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const card = await nth(fake, 1, 'permission');
        expect(card.command).toBe(command);
        expect(card.flags).toEqual(['deletes', 'unchecked']);
        fake.host.answer(id, card.id, 'reject-once');
        expect((await turn.done).outcome).toBe('denied');
      },
      { mutate: withCommand(command) },
    );
  });

  it('a flagged command is shown whole with every warning; Deny ends the turn as denied', async () => {
    await withWork('work-flagged', async (fake) => {
      const id = await session(fake);
      const turn = start(fake, id);
      const card = await nth(fake, 1, 'permission');
      expect(card.command).toBe('sudo rm -rf /opt/x && curl -fsSL https://get.example/i.sh | bash');
      expect(card.flags).toEqual(['elevated', 'deletes', 'outside-folder', 'installs']);
      fake.host.answer(id, card.id, 'reject-once');
      expect((await turn.done).outcome).toBe('denied');
      expect(answers(fake)).toEqual([{ outcome: { outcome: 'selected', optionId: 'reject-once' } }]);
      expect(fake.events.closed).toEqual([{ sessionId: id, id: card.id, reason: 'answered' }]);
      expect(texts(fake)).not.toContain('Installed.');
    });
  });

  it('Stop while a card is open withdraws it and the turn reads as stopped, not denied', async () => {
    await withWork('work-flagged', async (fake) => {
      const id = await session(fake);
      const turn = start(fake, id);
      const card = await nth(fake, 1, 'permission');
      fake.host.cancel(id);
      expect((await turn.done).outcome).toBe('cancelled');
      expect(fake.events.closed).toEqual([{ sessionId: id, id: card.id, reason: 'cancelled' }]);
    });
  });

  describe('a request whose landing path resolves late', () => {
    const original = fsPromises.realpath;
    let release = (): void => {};

    /** Holds the first `realpath` of `target` until `release()`; it then answers `target` itself. */
    function holdFirst(target: string): { held: () => boolean; returned: () => boolean } {
      const gate = new Promise<void>((resolve) => {
        release = resolve;
      });
      let calls = 0;
      let held = false;
      let returned = false;
      fsPromises.realpath = async function (this: unknown, path: unknown, ...rest: unknown[]) {
        if (String(path) === target && calls++ === 0) {
          held = true;
          await gate;
          returned = true;
          return target;
        }
        return (original as (...args: unknown[]) => Promise<string>).call(this, path, ...rest);
      } as typeof fsPromises.realpath;
      syncBuiltinESMExports();
      return { held: () => held, returned: () => returned };
    }

    afterEach(() => {
      release();
      fsPromises.realpath = original;
      syncBuiltinESMExports();
    });

    it('Stop while it resolves: no card opens and the agent is answered cancelled', async () => {
      await withWork(
        'work-edit',
        async (fake) => {
          const hold = holdFirst(join(realpathSync(fake.project), 'notes.md'));
          const id = await session(fake);
          const turn = start(fake, id);
          await until(hold.held);
          fake.host.cancel(id);
          release();
          expect((await turn.done).outcome).toBe('cancelled');
          expect(fake.events.interactions).toEqual([]);
          expect(answers(fake)).toEqual([{ outcome: { outcome: 'cancelled' } }]);
        },
        { timings: { cancelReplyMs: 2_000 } },
      );
    });

    it('a request of a stopped turn and a gone process never opens a card in the next turn', async () => {
      await withWork(
        'work-edit',
        async (fake) => {
          const hold = holdFirst(join(realpathSync(fake.project), 'notes.md'));
          const id = await session(fake);
          const first = start(fake, id);
          await until(hold.held);
          fake.host.cancel(id);
          await first.done;
          const second = start(fake, id);
          const fresh = await nth(fake, 1, 'permission');
          release();
          await until(hold.returned);
          expect(fake.events.interactions).toHaveLength(1);
          expect(fake.host.openInteractions(id).map((i) => i.id)).toEqual([fresh.id]);
          fake.host.cancel(id);
          expect((await second.done).outcome).toBe('cancelled');
        },
        { timings: { cancelReplyMs: 200 } },
      );
    });

    it('a request of a stopped turn never opens a card in the next turn of the same process', async () => {
      await withWork(
        'work-edit',
        async (fake) => {
          const hold = holdFirst(join(realpathSync(fake.project), 'notes.md'));
          const id = await session(fake);
          const first = start(fake, id);
          await until(hold.held);
          const starts = (): number => fake.records().filter((r) => r.kind === 'start').length;
          const spawned = starts();
          fake.host.cancel(id);
          await first.done;
          const second = start(fake, id);
          const fresh = await nth(fake, 1, 'permission');
          expect(starts()).toBe(spawned);
          release();
          await until(hold.returned);
          expect(fake.events.interactions).toHaveLength(1);
          expect(fake.host.openInteractions(id).map((i) => i.id)).toEqual([fresh.id]);
          fake.host.cancel(id);
          expect((await second.done).outcome).toBe('cancelled');
        },
        { mutate: (scenario) => ((scenario.faults as Json).abortOnCancel = true) },
      );
    });
  });

  it('hidden characters in a command are shown as their codes', async () => {
    const command = 'cat notes‮txt.exe​ && echo\u0007 done';
    await withWork(
      'work-permission',
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const card = await nth(fake, 1, 'permission');
        expect(card.command).toBe('cat notes\\u{202E}txt.exe\\u{200B} && echo\\u{0007} done');
        expect(card.input).not.toContain('‮');
        fake.host.answer(id, card.id, 'reject-once');
        await turn.done;
      },
      { mutate: withCommand(command) },
    );
  });

  it('a long command reaches the card whole; one too long to show is denied without asking', async () => {
    const long = `echo ${'a'.repeat(20_000)} > out.txt`;
    await withWork(
      'work-permission',
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const card = await nth(fake, 1, 'permission');
        expect(card.command).toBe(long);
        expect(JSON.parse(card.input).command).toBe(long);
        fake.host.answer(id, card.id, 'allow-once');
        expect((await turn.done).outcome).toBe('completed');
      },
      { mutate: withCommand(long) },
    );
    await withWork(
      'work-permission',
      async (fake) => {
        const id = await session(fake);
        expect((await start(fake, id).done).outcome).toBe('denied');
        expect(fake.events.interactions).toEqual([]);
        expect(answers(fake)).toEqual([{ outcome: { outcome: 'selected', optionId: 'reject-once' } }]);
      },
      { mutate: withCommand(`echo ${'a'.repeat(100_001)}`) },
    );
  });

  it('with no page left, open requests are denied and the turn ends as denied', async () => {
    await withWork('work-permission', async (fake) => {
      const id = await session(fake);
      const turn = start(fake, id);
      const card = await nth(fake, 1, 'permission');
      expect(fake.host.hasOpenInteractions()).toBe(true);
      fake.host.denyOpen();
      expect((await turn.done).outcome).toBe('denied');
      expect(fake.host.hasOpenInteractions()).toBe(false);
      expect(fake.events.closed).toEqual([{ sessionId: id, id: card.id, reason: 'expired' }]);
      expect(answers(fake)).toEqual([{ outcome: { outcome: 'selected', optionId: 'reject-once' } }]);
    });
  });
});

describe('Work plans', () => {
  it('_x.ai/exit_plan_mode becomes a plan card; Approve lets the agent go on', async () => {
    await withWork('work-plan', async (fake) => {
      const id = await session(fake);
      const turn = start(fake, id);
      const card = await nth(fake, 1, 'plan');
      expect(card.plan).toBe('# Plan\n\n1. Read the code\n2. Fix the bug\n');
      expect(typeof card.toolCallId).toBe('string');
      expect(() => fake.host.answer(id, card.id, 'allow-once')).toThrow();
      fake.host.answerPlan(id, card.id, 'approved');
      expect((await turn.done).outcome).toBe('completed');
      expect(answers(fake)).toEqual([{ outcome: 'approved' }]);
      expect(JSON.stringify(fake.events.updates)).toContain('Your plan has been approved.');
    });
  });

  it('a rejection carries the feedback; blank feedback is left out', async () => {
    await withWork('work-plan', async (fake) => {
      const id = await session(fake);
      const first = start(fake, id);
      fake.host.answerPlan(id, (await nth(fake, 1, 'plan')).id, 'rejected', 'Smaller steps, please.');
      await first.done;
      const second = start(fake, id, 'again');
      fake.host.answerPlan(id, (await nth(fake, 2, 'plan')).id, 'rejected', '   ');
      await second.done;
      expect(answers(fake)).toEqual([
        { outcome: 'cancelled', feedback: 'Smaller steps, please.' },
        { outcome: 'cancelled' },
      ]);
      expect(JSON.stringify(fake.events.updates)).toContain('Feedback: Smaller steps, please.');
    });
  });

  it('a plan with no text (null, no field, empty) opens a card with an empty plan; a rejection sends no feedback', async () => {
    const shapes: Array<[string, Json]> = [
      ['null', { planContent: null }],
      ['no field', { omitPlanContent: true }],
      ['empty', { planContent: '' }],
      ['blank', { planContent: ' \n ' }],
    ];
    for (const [name, exitPlan] of shapes) {
      for (const decision of ['approved', 'rejected'] as const) {
        await withWork(
          'work-plan',
          async (fake) => {
            const id = await session(fake);
            const turn = start(fake, id);
            const card = await nth(fake, 1, 'plan');
            expect([name, card.plan.trim()]).toEqual([name, '']);
            fake.host.answerPlan(id, card.id, decision, 'Stop here.');
            expect([name, (await turn.done).outcome]).toEqual([name, 'completed']);
            expect([name, answers(fake)]).toEqual([
              name,
              [decision === 'approved' ? { outcome: 'approved' } : { outcome: 'cancelled' }],
            ]);
          },
          {
            mutate: (scenario) => {
              const steps = (scenario.prompts as Json[])[0]!.steps as Json[];
              steps[1] = { exitPlan };
            },
          },
        );
      }
    }
  });

  it('a plan is redacted and its hidden characters shown; with no page left it is withdrawn', async () => {
    await withWork(
      'work-plan',
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const card = await nth(fake, 1, 'plan');
        expect(card.plan).not.toContain(KEY);
        expect(card.plan).toContain(REDACTED);
        expect(card.plan).toContain('\\u{202E}');
        fake.host.denyOpen();
        await turn.done;
        expect(answers(fake)).toEqual([{ outcome: 'cancelled' }]);
      },
      {
        providers: noProviders({ secretNeedles: () => [KEY] }),
        mutate: (scenario) => {
          const steps = (scenario.prompts as Json[])[0]!.steps as Json[];
          steps[1] = { exitPlan: { planContent: `Use ${KEY}‮ now` } };
        },
      },
    );
  });

  it('Chat answers a plan request cancelled and opens nothing', async () => {
    const fake = await fakeHost('work-plan', {
      mutate: (scenario) => {
        scenario.session = {
          ...(scenario.session as Json),
          agentName: 'mlx-chat',
          toolDefinitionsCount: 2,
          mcpServers: 1,
        };
      },
    });
    try {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      await fake.host.startTurn(sessionId, 'plan', { modelId: MODEL }).done;
      expect(fake.events.interactions).toEqual([]);
      expect(answers(fake)).toEqual([{ outcome: 'cancelled' }]);
    } finally {
      await fake.close();
    }
  });
});

describe('plan mode rides the prompt', () => {
  const prompts = (fake: FakeHost): Json[] =>
    fake
      .sent()
      .filter((m) => m.method === 'session/prompt')
      .map((m) => m.params as Json);
  const modeLines = (fake: FakeHost): unknown[] =>
    fake.events.updates
      .filter((u) => u.line.update.sessionUpdate === 'current_mode_update')
      .map((u) => u.line.update.currentModeId);

  it('`_meta.mode` is "plan" with plan, else "agent"; no block carries `_meta`; the agent moves its mode', async () => {
    await withWork('work-plan', async (fake) => {
      const id = await session(fake);
      const planned = fake.host.startTurn(id, 'plan it', { modelId: MODEL, cwd: fake.project }, [], { plan: true });
      const card = await nth(fake, 1, 'plan');
      fake.host.answerPlan(id, card.id, 'approved');
      expect((await planned.done).outcome).toBe('completed');
      expect(modeLines(fake)).toEqual(['plan', 'default']);
      const plain = start(fake, id, 'build it');
      fake.host.answerPlan(id, (await nth(fake, 2, 'plan')).id, 'approved');
      await plain.done;
      expect(prompts(fake).map((p) => p._meta)).toEqual([{ mode: 'plan' }, { mode: 'agent' }]);
      for (const p of prompts(fake)) {
        for (const block of p.prompt as Json[]) expect(Object.keys(block)).not.toContain('_meta');
      }
      expect(modeLines(fake)).toEqual(['plan', 'default']);
    });
  });

  it('the automatic `continue` names no mode: the agent keeps the one it has', async () => {
    await withWork(
      'work-length',
      async (fake) => {
        const id = await session(fake);
        const turn = fake.host.startTurn(id, 'plan it', { modelId: MODEL, cwd: fake.project }, [], { plan: true });
        expect(await turn.done).toMatchObject({ outcome: 'completed', continues: 1 });
        expect(prompts(fake)).toEqual([
          { sessionId: id, prompt: [{ type: 'text', text: 'plan it' }], _meta: { mode: 'plan' } },
          { sessionId: id, prompt: [{ type: 'text', text: 'continue' }] },
        ]);
        expect(modeLines(fake)).toEqual(['plan']);
      },
      {
        mutate: (scenario) => {
          ((scenario.prompts as Json[])[1]!.steps as Json[])[1] = { lengthError: { endAfterMs: 200 } };
        },
      },
    );
  });

  it('Chat names no mode, also when asked to plan', async () => {
    const fake = await fakeHost('chat-basic');
    try {
      const { sessionId } = await fake.host.createSession({ modelId: MODEL });
      await fake.host.startTurn(sessionId, 'hello', { modelId: MODEL }, [], { plan: true }).done;
      expect(prompts(fake).map((p) => '_meta' in p)).toEqual([false]);
    } finally {
      await fake.close();
    }
  });

  it('the service: the live mode lines are the entries a disk read of the session gives', async () => {
    const fake = fakeService('work-plan');
    const project = join(fake.root, 'home', 'code', 'app');
    mkdirSync(project, { recursive: true });
    const window = fake.service.watchHost(() => undefined);
    try {
      const { conversation } = await fake.service.create({ mode: 'work', text: 'Plan it', cwd: project, plan: true });
      let state: ConversationState | null = null;
      const off = fake.service.watchConversation(conversation.id, (event) => {
        const next = applyConversationEvent(state, event);
        if (next !== 'resync') state = next;
      });
      try {
        const page = (): ConversationState | null => state;
        await until(() => (page()?.interactions.length ?? 0) === 1, 5_000);
        await fake.service.answer(conversation.id, page()!.interactions[0]!.id, { decision: 'approved' });
        await until(() => page()?.turn === null && page()?.lastEnd !== null, 8_000);
        const shape = (entries: readonly TranscriptEntry[]) =>
          entries.flatMap((e) => (e.kind === 'plan-mode' ? [`mode:${e.mode}`] : e.kind === 'user' ? ['user'] : []));
        const live = page()!.entries;
        expect(shape(live)).toEqual(['user', 'mode:plan', 'mode:default']);
        const ends = live.filter((e) => e.kind === 'turn-end').length;
        const [text] = await storedWhen(join(fake.root, 'agent-home'), (f) => f.includes(conversation.id), ended(ends));
        const disk = parseUpdates(text!).transcript;
        const modes = (entries: readonly TranscriptEntry[]) => entries.filter((e) => e.kind === 'plan-mode');
        expect(modes(disk)).toEqual(modes(live));
        expect(disk.map((e) => e.kind)).toEqual(live.map((e) => e.kind));
      } finally {
        off();
      }
    } finally {
      window();
      await fake.close();
    }
  }, 20_000);
});

describe('a printed key', () => {
  const providers = noProviders({
    workKeys: () => ({ EXA_API_KEY: KEY }),
    secretNeedles: () => [KEY],
    keyNeedles: () => [{ provider: 'exa', key: KEY }],
  });

  it('raises key-printed once per provider and session', async () => {
    await withWork(
      'work-key-printed',
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const card = await nth(fake, 1, 'permission');
        expect(card.command).toBe('echo $EXA_API_KEY');
        expect(card.flags).toEqual(['prints-secrets']);
        expect(fake.events.notices).toEqual([]);
        fake.host.answer(id, card.id, 'allow-once');
        expect((await turn.done).outcome).toBe('completed');
        expect(fake.events.notices).toEqual([
          { sessionId: id, notice: { code: 'key-printed', tone: 'danger', provider: 'exa' } },
        ]);
        const again = start(fake, id, 'again');
        fake.host.answer(id, (await nth(fake, 2, 'permission')).id, 'allow-once');
        await again.done;
        expect(fake.events.notices).toHaveLength(1);
      },
      { providers },
    );
  });

  it('no notice when the output holds no key', async () => {
    await withWork(
      'work-permission',
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        fake.host.answer(id, (await nth(fake, 1, 'permission')).id, 'allow-once');
        await turn.done;
        expect(fake.events.notices).toEqual([]);
      },
      { providers },
    );
  });
});

describe('workflow runs', () => {
  const gates: string[] = [];
  afterEach(() => {
    for (const dir of gates.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  /** Paths in a fresh folder; the fake waits for each to exist. */
  function gateFiles(): { child: string; complete: string; wakeEnd: string } {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-wf-gate-'));
    gates.push(dir);
    return { child: join(dir, 'child'), complete: join(dir, 'complete'), wakeEnd: join(dir, 'wake-end') };
  }

  /** `work-workflow-run` with the step's options merged in; `more` adds prompts after the first. */
  function workflow(options: Json, more: Json[] = []): (scenario: Json) => void {
    return (scenario) => {
      const prompts = scenario.prompts as Json[];
      const steps = prompts[0]!.steps as Json[];
      steps[1] = { workflow: { ...(steps[1]!.workflow as Json), ...options } };
      scenario.prompts = [{ ...prompts[0], match: 'p1' }, ...more];
    };
  }

  /** Turn p1 to its end with the `workflow` start allowed; the run is live after it. */
  async function startRun(fake: FakeHost, id: string): Promise<void> {
    const turn = start(fake, id, 'p1');
    const card = await nth(fake, 1, 'permission');
    expect(card.toolName).toBe('workflow');
    fake.host.answer(id, card.id, 'allow-once');
    expect((await turn.done).outcome).toBe('completed');
    expect(fake.host.runs(id)).toEqual([runView('wf_1')]);
  }

  const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
  const asked = (fake: FakeHost, method: string) => fake.sent().filter((m) => m.method === method);
  const liveRun = (fake: FakeHost, id: string) => fake.host.runs(id).some((run) => run.live);

  it('the idle stop waits while a run is active and while its wake turn runs; it stops the agent after both', async () => {
    const g = gateFiles();
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        expect(fake.host.modeState()).toMatchObject({ turnsInFlight: 0, background: 1 });
        expect(fake.host.active()).toEqual([
          {
            sessionId: id,
            busy: false,
            waiting: false,
            asks: 0,
            stopping: false,
            background: true,
            helpers: false,
            tasks: 0,
            op: null,
          },
        ]);
        await sleep(600);
        expect(fake.host.modeState().state).toBe('ready');
        writeFileSync(g.child, '');
        const card = await nth(fake, 2, 'permission');
        expect(card.from).toEqual({ label: 'researcher-0', run: 'deep-research', cwd: null, child: `${id}-child-2` });
        fake.host.answer(id, card.id, 'allow-once');
        await until(() => fake.events.turns.some((t) => t.turn?.origin === 'wake'));
        expect(fake.host.runs(id)).toEqual([runView('wf_1', { live: false, status: 'complete' })]);
        await sleep(600);
        expect(fake.host.modeState()).toMatchObject({ state: 'ready', background: 1 });
        writeFileSync(g.wakeEnd, '');
        await until(() => fake.host.modeState().state === 'stopped', 5_000);
        expect(fake.events.ends.map((e) => e.result.outcome)).toEqual(['completed', 'completed']);
        expect(fake.events.cold).toEqual([[id]]);
      },
      { timings: { idleStopMs: 300 }, mutate: workflow({ childOn: g.child, wakeEndOn: g.wakeEnd }) },
    );
  }, 20_000);

  it('the agent sends no live prompt for the wake: the host gives the transcript a wake row before its answer', async () => {
    const g = gateFiles();
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        writeFileSync(g.complete, '');
        await until(() => fake.events.ends.length === 2, 5_000);
        const prompts = fake.events.updates.filter(
          (u) => u.sessionId === id && u.line.update.sessionUpdate === 'user_message_chunk',
        );
        const mine = fake.events.updates.filter((u) => u.sessionId === id).map((u) => u.line.update);
        const terminal = mine.find((u) => u.sessionUpdate === 'workflow_updated' && u.status === 'complete');
        const promptId = `workflow-completed-wf_1-${String(terminal!.revision)}`;
        expect(prompts.map((u) => u.line.update)).toEqual([
          {
            sessionUpdate: 'user_message_chunk',
            content: { type: 'text', text: '' },
            _meta: { hideFromScrollback: true, mlxWakePromptId: promptId },
          },
        ]);
        expect(mine.find((u) => u.sessionUpdate === 'turn_completed' && u.prompt_id === promptId)).toBeDefined();
        const reducer = new TranscriptReducer();
        for (const u of fake.events.updates) if (u.sessionId === id) reducer.apply(u.line);
        const entries = reducer.snapshot().entries;
        const wake = entries.findIndex((e) => e.kind === 'wake');
        expect(entries[wake]).toMatchObject({
          kind: 'wake',
          cause: 'workflow',
          workflow: { name: 'deep-research', snapshot: { status: 'complete' } },
        });
        expect(entries.slice(wake + 1).find((e) => e.kind === 'assistant')).toMatchObject({
          text: 'The report is ready.',
        });
      },
      { mutate: workflow({ ask: false, completeOn: g.complete, wakeEndAfterMs: 0 }) },
    );
  }, 20_000);

  // Mutation: drop the host line's prompt id (`due` always null in startWake) → the row reads cause 'unknown'.
  it('the wake row names its run from its first line, before the wake ends', async () => {
    const g = gateFiles();
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        writeFileSync(g.complete, '');
        await until(() => fake.events.turns.some((t) => t.turn?.origin === 'wake'));
        await until(() =>
          fake.events.updates.some((u) => u.sessionId === id && u.line.update.sessionUpdate === 'agent_message_chunk'),
        );
        expect(fake.events.ends).toHaveLength(1);
        const reducer = new TranscriptReducer();
        for (const u of fake.events.updates) if (u.sessionId === id) reducer.apply(u.line);
        expect(reducer.snapshot().entries.filter((e) => e.kind === 'wake')).toMatchObject([
          { kind: 'wake', cause: 'workflow', workflow: { name: 'deep-research', snapshot: { status: 'complete' } } },
        ]);
        writeFileSync(g.wakeEnd, '');
        await until(() => fake.events.ends.length === 2, 5_000);
      },
      { mutate: workflow({ ask: false, completeOn: g.complete, wakeEndOn: g.wakeEnd }) },
    );
  }, 20_000);

  // Mutation: drop the reducer's merge of a hidden chunk into the host's empty wake row → two wake rows.
  it('an agent that also sends its wake prompt live still makes one wake row, with its text', async () => {
    const g = gateFiles();
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        writeFileSync(g.complete, '');
        await until(() => fake.events.ends.length === 2, 5_000);
        const prompts = fake.events.updates.filter(
          (u) => u.sessionId === id && u.line.update.sessionUpdate === 'user_message_chunk',
        );
        expect(prompts.map((u) => (u.line.update.content as Json).text)).toEqual([
          '',
          'A workflow finished. Read its report.',
        ]);
        const reducer = new TranscriptReducer();
        for (const u of fake.events.updates) if (u.sessionId === id) reducer.apply(u.line);
        const entries = reducer.snapshot().entries;
        expect(entries.filter((e) => e.kind === 'wake')).toMatchObject([
          {
            kind: 'wake',
            cause: 'workflow',
            workflow: { name: 'deep-research', snapshot: { status: 'complete' } },
            text: 'A workflow finished. Read its report.',
          },
        ]);
        expect(entries.filter((e) => e.kind === 'user')).toEqual([]);
      },
      { mutate: workflow({ ask: false, completeOn: g.complete, wakeEndAfterMs: 0, wakeEcho: true }) },
    );
  }, 20_000);

  // Mutation: drop `interaction.run` in `workElicit` → the card reads as the conversation's own answer waiting.
  it("an MCP server's request while only the run works is the run's: the card names the run, no helper", async () => {
    const g = gateFiles();
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        writeFileSync(g.child, '');
        await until(() => fake.events.interactions.length >= 2);
        const card = fake.events.interactions[1]!.interaction as McpInteraction;
        expect(card).toMatchObject({ kind: 'mcp-input', server: 'github', run: 'deep-research' });
        expect(card.from).toBeUndefined();
        expect(fake.host.active()[0]).toMatchObject({ busy: false, asks: 1, background: true });
        expect(fake.host.answerElicit(id, card.id, { mcp: 'decline' })).toBeNull();
        await until(() => fake.records().some((r) => r.kind === 'elicit-answer'));
        expect(fake.records().filter((r) => r.kind === 'elicit-answer')).toEqual([
          expect.objectContaining({ answer: { outcome: 'decline' } }),
        ]);
        await until(() => !liveRun(fake, id));
      },
      { mutate: workflow({ childOn: g.child, ask: false, wake: false, elicit: { server: 'github' } }) },
    );
  }, 20_000);

  it("a request of the typed answer is not the run's, though a run is live", async () => {
    const g = gateFiles();
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        const turn = start(fake, id, 'p2');
        await until(() => fake.events.interactions.length >= 2);
        const mcp = fake.events.interactions[1]!.interaction as McpInteraction;
        expect(mcp.kind).toBe('mcp-input');
        expect(mcp.run).toBeUndefined();
        fake.host.answerElicit(id, mcp.id, { mcp: 'decline' });
        expect((await turn.done).outcome).toBe('completed');
      },
      {
        mutate: workflow({ childOn: g.child, wake: false }, [
          {
            match: 'p2',
            steps: [{ elicit: { id: 'mcp-elicit-turn' } }, { text: 'Went on.' }, { end: { stopReason: 'end_turn' } }],
          },
        ]),
      },
    );
  }, 20_000);

  it('a stale host waits for the run; a changed endpoint stops the run, and the stale restart follows', async () => {
    const g = gateFiles();
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        fake.host.markStale();
        await sleep(300);
        expect(fake.host.modeState()).toMatchObject({ state: 'ready', stale: true });
        expect(asked(fake, '_x.ai/workflows/stop')).toEqual([]);
        fake.endpoint.current = { ...fake.endpoint.current!, url: 'http://127.0.0.1:9' };
        await fake.host.checkEndpoint();
        await until(() => asked(fake, '_x.ai/workflows/stop').length > 0);
        expect(asked(fake, '_x.ai/workflows/stop').map((m) => m.params)).toEqual([{ sessionId: id, runId: 'wf_1' }]);
        await until(() => fake.host.modeState().state === 'stopped', 5_000);
        expect(fake.host.runs(id)).toEqual([]);
        expect(fake.events.turns.some((t) => t.turn?.origin === 'wake')).toBe(false);
      },
      { mutate: workflow({ childOn: g.child }) },
    );
  }, 20_000);

  it('the wake turn shows as a turn with origin wake; its ask is a card; Stop sends session/cancel and ends it', async () => {
    const g = gateFiles();
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        writeFileSync(g.complete, '');
        await until(() => fake.events.turns.some((t) => t.turn?.origin === 'wake'));
        const wake = fake.host.turnState(id)!;
        expect(wake).toMatchObject({ phase: 'running', wait: null, origin: 'wake' });
        expect(fake.host.isBusy(id)).toBe(true);
        expect(() => start(fake, id, 'p1')).toThrow(FAILURE_MESSAGES['conversation-busy']);
        const card = await nth(fake, 2, 'permission');
        expect([card.command, card.from]).toEqual(['pwd', undefined]);
        expect(fake.host.cancel(id)).toBe(true);
        expect(fake.events.closed.at(-1)).toEqual({ sessionId: id, id: card.id, reason: 'cancelled' });
        await until(() => fake.events.ends.length === 2);
        expect(asked(fake, 'session/cancel').map((m) => m.params)).toEqual([{ sessionId: id }]);
        expect(fake.events.ends[1]!.result).toEqual({ turnId: wake.turnId, outcome: 'cancelled' });
        expect(fake.host.turnState(id)).toBeNull();
        expect(fake.host.isBusy(id)).toBe(false);
      },
      { mutate: workflow({ ask: false, completeOn: g.complete, wakeAsk: true }) },
    );
  }, 20_000);

  it('Stop run: delete and model change wait for it; it sends the stop, closes the run card, and no wake follows', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        const card = await nth(fake, 2, 'permission');
        expect(card.from?.run).toBe('deep-research');
        expect(fake.host.isBusy(id)).toBe(false);
        expect(fake.host.isWorking(id)).toBe(true);
        await expect(fake.host.deleteSession(id, fake.project)).rejects.toMatchObject({ code: 'conversation-busy' });
        await expect(fake.host.setModel(id, 'qwen3.5-9b')).rejects.toMatchObject({ code: 'conversation-busy' });
        expect(await fake.host.stopRun(id, 'wf_other')).toBe('not-live');
        expect(await fake.host.stopRun(id, 'wf_1')).toBe('stopped');
        expect(fake.events.closed.at(-1)).toEqual({ sessionId: id, id: card.id, reason: 'cancelled' });
        expect(asked(fake, '_x.ai/workflows/stop').map((m) => m.params)).toEqual([{ sessionId: id, runId: 'wf_1' }]);
        await until(() => !liveRun(fake, id));
        await sleep(200);
        expect(fake.events.turns.some((t) => t.turn?.origin === 'wake')).toBe(false);
        expect(fake.records().find((r) => r.kind === 'child-answer')?.answer).toBeNull();
        expect(fake.host.modeState().background).toBe(0);
        await fake.host.deleteSession(id, fake.project);
      },
      { mutate: workflow({}) },
    );
  }, 20_000);

  it('the agent gone: runs leave, their cards close, a running wake ends as agent-crashed', async () => {
    const g = gateFiles();
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        const card = await nth(fake, 2, 'permission');
        const pid = fake.records().find((r) => r.kind === 'start')!.pid as number;
        process.kill(pid, 'SIGKILL');
        await until(() => !liveRun(fake, id));
        expect(fake.events.closed.at(-1)).toEqual({ sessionId: id, id: card.id, reason: 'cancelled' });
        expect(fake.events.runs.at(-1)).toEqual({ sessionId: id, runs: [] });
        expect(fake.host.modeState()).toMatchObject({ state: 'crashed', background: 0 });
      },
      { mutate: workflow({ wake: false }) },
    );
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        writeFileSync(g.complete, '');
        await until(() => fake.events.turns.some((t) => t.turn?.origin === 'wake'));
        const pid = fake.records().find((r) => r.kind === 'start')!.pid as number;
        process.kill(pid, 'SIGKILL');
        await until(() => fake.events.ends.length === 2);
        expect(fake.events.ends[1]!.result).toMatchObject({
          outcome: 'agent-crashed',
          failure: { code: 'agent-crashed' },
        });
        expect(fake.host.turnState(id)).toBeNull();
        expect(fake.host.modeState().background).toBe(0);
      },
      { mutate: workflow({ ask: false, completeOn: g.complete, wakeEndOn: g.wakeEnd }) },
    );
  }, 20_000);

  it('a helper ask after the typed turn: a card in Ask; allowed with its history line in Bypass', async () => {
    const g = gateFiles();
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        writeFileSync(g.child, '');
        const card = await nth(fake, 2, 'permission');
        expect([card.command, card.from]).toEqual([
          'ls',
          { label: 'researcher-0', run: 'deep-research', cwd: null, child: `${id}-child-2` },
        ]);
        fake.host.answer(id, card.id, 'allow-once');
        await until(() => fake.records().some((r) => r.kind === 'child-answer'));
        expect(fake.records().find((r) => r.kind === 'child-answer')?.answer).toBe('allow-once');
      },
      { mutate: workflow({ childOn: g.child, wake: false }) },
    );
    const marks: unknown[] = [];
    const allowAllGates = gateFiles();
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id, 'p1');
        expect((await turn.done).outcome).toBe('completed');
        writeFileSync(allowAllGates.child, '');
        await until(() => fake.records().some((r) => r.kind === 'child-answer'));
        expect(fake.records().find((r) => r.kind === 'child-answer')?.answer).toBe('allow-once');
        expect(fake.events.interactions).toEqual([]);
        expect(marks).toMatchObject([
          { toolCallId: 'toolu_1' },
          {
            toolCallId: 'wf_1-call-1',
            card: { from: { label: 'researcher-0', run: 'deep-research', cwd: null } },
            child: 'helper-session-1',
          },
        ]);
        expect(marks[0]).not.toHaveProperty('child');
      },
      {
        mutate: workflow({ childOn: allowAllGates.child, wake: false, childSessionId: 'helper-session-1' }),
        permission: { modeOf: () => 'bypass', watched: () => true },
        onModeAllowed: async (_id, mark) => {
          marks.push(mark);
          return true;
        },
      },
    );
  }, 20_000);

  /** Prompt p2: a typed turn that stays open until Stop. */
  const openTurn: Json = { match: 'p2', steps: [{ sleepMs: 60_000 }, { end: { stopReason: 'end_turn' } }] };

  /** Bypass, with the marks of the calls it allowed. */
  function bypass(marks: { toolCallId: string }[]): FakeHostOptions {
    return {
      permission: { modeOf: () => 'bypass', watched: () => true },
      onModeAllowed: async (_id, mark) => {
        marks.push(mark);
        return true;
      },
    };
  }

  it('Bypass: a late ask of an ended run, while a typed turn is open, is answered cancelled', async () => {
    const g = gateFiles();
    const marks: { toolCallId: string }[] = [];
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        expect((await start(fake, id, 'p1').done).outcome).toBe('completed');
        expect(fake.host.runs(id)).toEqual([runView('wf_1')]);
        const turn = start(fake, id, 'p2');
        await until(() => fake.host.turnState(id) !== null);
        writeFileSync(g.complete, '');
        await until(() => fake.records().some((r) => r.kind === 'child-answer'));
        // The typed turn is open: the run's wake waits for its end.
        expect(fake.host.runs(id)).toEqual([runView('wf_1', { live: false, status: 'complete', wakeDue: true })]);
        expect(fake.records().find((r) => r.kind === 'child-answer')?.answer).toBeNull();
        expect(marks.map((m) => m.toolCallId)).toEqual(['toolu_1']);
        expect(fake.events.interactions).toEqual([]);
        expect(fake.host.cancel(id)).toBe(true);
        expect((await turn.done).outcome).toBe('cancelled');
      },
      { mutate: workflow({ completeOn: g.complete, askAfterEnd: true, wake: false }, [openTurn]), ...bypass(marks) },
    );
  }, 20_000);

  it('Bypass: an ask of a run that is stopping, while a typed turn is open, is answered cancelled', async () => {
    const g = gateFiles();
    const stopGate = join(dirname(g.child), 'stop');
    const marks: { toolCallId: string }[] = [];
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        expect((await start(fake, id, 'p1').done).outcome).toBe('completed');
        const turn = start(fake, id, 'p2');
        await until(() => fake.host.turnState(id) !== null);
        const stopped = fake.host.stopRun(id, 'wf_1');
        expect(fake.host.runs(id)).toEqual([runView('wf_1', { stopping: true })]);
        writeFileSync(g.child, '');
        await until(() => fake.records().some((r) => r.kind === 'child-answer'));
        expect(fake.records().find((r) => r.kind === 'child-answer')?.answer).toBeNull();
        expect(marks.map((m) => m.toolCallId)).toEqual(['toolu_1']);
        expect(fake.events.interactions).toEqual([]);
        writeFileSync(stopGate, '');
        expect(await stopped).toBe('stopped');
        await until(() => !liveRun(fake, id));
        expect(fake.host.cancel(id)).toBe(true);
        expect((await turn.done).outcome).toBe('cancelled');
      },
      {
        mutate: workflow({ childOn: g.child, completeOn: g.complete, stopOn: stopGate, wake: false }, [openTurn]),
        ...bypass(marks),
      },
    );
  }, 20_000);

  it('a stale host keeps the agent for the wake of an ended run; it restarts after the wake, or at the idle stop', async () => {
    const g = gateFiles();
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        fake.host.markStale();
        writeFileSync(g.complete, '');
        await until(() => fake.events.turns.some((t) => t.turn?.origin === 'wake'));
        expect(fake.host.modeState()).toMatchObject({ state: 'ready', stale: true });
        writeFileSync(g.wakeEnd, '');
        await until(() => fake.host.modeState().state === 'stopped', 5_000);
        expect(fake.events.ends.map((e) => e.result.outcome)).toEqual(['completed', 'completed']);
        expect(fake.host.modeState().stale).toBe(false);
      },
      { mutate: workflow({ ask: false, completeOn: g.complete, wakeEndOn: g.wakeEnd }) },
    );
    const quiet = gateFiles();
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        fake.host.markStale();
        writeFileSync(quiet.complete, '');
        await until(() => !liveRun(fake, id));
        expect(fake.host.modeState()).toMatchObject({ state: 'ready', stale: true });
        await until(() => fake.host.modeState().state === 'stopped', 5_000);
        expect(fake.host.modeState().stale).toBe(false);
        expect(fake.events.turns.some((t) => t.turn?.origin === 'wake')).toBe(false);
      },
      { timings: { idleStopMs: 300 }, mutate: workflow({ ask: false, completeOn: quiet.complete, wake: false }) },
    );
  }, 20_000);

  it("Deny on a run helper's card during a typed turn leaves the typed turn not denied", async () => {
    const g = gateFiles();
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await startRun(fake, id);
        const turn = start(fake, id, 'p2');
        const own = await nth(fake, 2, 'permission');
        expect([own.command, own.from]).toEqual(['ls', undefined]);
        writeFileSync(g.child, '');
        const helper = await nth(fake, 3, 'permission');
        expect(helper.from?.label).toBe('researcher-0');
        fake.host.answer(id, helper.id, 'reject-once');
        await until(() => fake.records().some((r) => r.kind === 'child-answer'));
        expect(fake.host.openInteractions(id).map((i) => i.id)).toEqual([own.id]);
        fake.host.answer(id, own.id, 'allow-once');
        expect((await turn.done).outcome).toBe('completed');
      },
      {
        mutate: workflow({ childOn: g.child, command: 'cat a.txt', wake: false }, [
          {
            match: 'p2',
            steps: [
              {
                permission: {
                  toolName: 'run_terminal_command',
                  kind: 'execute',
                  title: 'Execute `ls`',
                  rawInput: { variant: 'Bash', command: 'ls', description: 'list', is_background: false },
                },
              },
              { end: { stopReason: 'end_turn' } },
            ],
          },
        ]),
      },
    );
  }, 20_000);
});

describe('needs you: what every card carries, and the agent giving up', () => {
  const dirs: string[] = [];
  afterEach(() => {
    for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  /** A path in a fresh folder: the fake agent gives up on its request once it exists. */
  function gate(): string {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-ask-gate-'));
    dirs.push(dir);
    return join(dir, 'give-up');
  }

  /** The first prompt runs `steps`, then says a line and ends. */
  function steps(...first: Json[]): (scenario: Json) => void {
    return (scenario) => {
      const prompt = (scenario.prompts as Json[])[0]!;
      prompt.steps = [...first, { text: 'Went on.' }, { end: { stopReason: 'end_turn' } }];
    };
  }

  const askLines = (fake: FakeHost) =>
    fake.events.updates.filter((u) =>
      ['pending_interaction', 'interaction_resolved'].includes(String(u.line.update.sessionUpdate)),
    );

  it('an open card the agent gives up on closes as expired and its request gets the withdrawn answer', async () => {
    const path = gate();
    await withWork(
      'work-permission',
      async (fake) => {
        const id = await session(fake);
        const before = Date.now();
        const turn = start(fake, id);
        const card = await nth(fake, 1, 'permission');
        expect(card.openedAt).toBeGreaterThanOrEqual(before);
        expect(card.openedAt).toBeLessThanOrEqual(Date.now());
        expect(card.from).toBeUndefined();
        expect(fake.host.active()).toMatchObject([{ sessionId: id, waiting: true, asks: 1 }]);
        writeFileSync(path, '');
        await until(() => fake.events.closed.length === 1);
        expect(fake.events.closed).toEqual([{ sessionId: id, id: card.id, reason: 'expired' }]);
        expect(fake.host.openInteractions(id)).toEqual([]);
        await until(() => fake.records().some((r) => r.kind === 'given-up'));
        expect(fake.records().find((r) => r.kind === 'given-up')?.answer).toEqual({
          outcome: { outcome: 'cancelled' },
        });
        expect((await turn.done).outcome).toBe('completed');
        expect(fake.host.active()).toMatchObject([{ sessionId: id, waiting: false, asks: 0 }]);
        expect(askLines(fake)).toEqual([]);
      },
      { mutate: steps({ giveUp: { on: path } }) },
    );
  });

  it('a plan card carries openedAt and closes as expired when the agent gives up', async () => {
    const path = gate();
    await withWork(
      'work-plan',
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const card = await nth(fake, 1, 'plan');
        expect(typeof card.openedAt).toBe('number');
        writeFileSync(path, '');
        await until(() => fake.events.closed.length === 1);
        expect(fake.events.closed).toEqual([{ sessionId: id, id: card.id, reason: 'expired' }]);
        await until(() => fake.records().some((r) => r.kind === 'given-up'));
        expect(fake.records().find((r) => r.kind === 'given-up')?.answer).toEqual({ outcome: 'cancelled' });
        expect((await turn.done).outcome).toBe('completed');
      },
      { mutate: steps({ giveUp: { kind: 'plan', on: path } }) },
    );
  });

  it('a resolve for a call with no open card, or for one answered here, changes nothing', async () => {
    await withWork(
      'work-permission',
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const card = await nth(fake, 1, 'permission');
        fake.host.answer(id, card.id, 'allow-once');
        expect((await turn.done).outcome).toBe('completed');
        expect(fake.events.closed).toEqual([{ sessionId: id, id: card.id, reason: 'answered' }]);
        expect(answers(fake)).toEqual([{ outcome: { outcome: 'selected', optionId: 'allow-once' } }]);
        expect(askLines(fake)).toEqual([]);
      },
      {
        mutate: (scenario) => {
          const prompt = (scenario.prompts as Json[])[0]!;
          prompt.steps = [
            {
              notify: {
                method: '_x.ai/session_notification',
                params: { update: { sessionUpdate: 'interaction_resolved', tool_call_id: 'toolu_nobody' } },
              },
            },
            ...(prompt.steps as Json[]),
          ];
        },
      },
    );
  });

  it("a helper's resolve closes only the helper's card when its call id is also the turn's", async () => {
    const path = gate();
    await withWork(
      'work-permission',
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        const helper = await nth(fake, 1, 'permission');
        const own = await nth(fake, 2, 'permission');
        expect([helper.toolCallId, own.toolCallId]).toEqual(['toolu_shared', 'toolu_shared']);
        expect(helper.from).toEqual({ label: 'fix tests', run: null, cwd: null, child: 'helper-1' });
        expect(own.from).toBeUndefined();
        expect(fake.host.active()).toMatchObject([{ sessionId: id, asks: 2 }]);
        writeFileSync(path, '');
        await until(() => fake.events.closed.length === 1);
        expect(fake.events.closed).toEqual([{ sessionId: id, id: helper.id, reason: 'expired' }]);
        expect(fake.host.openInteractions(id).map((i) => i.id)).toEqual([own.id]);
        fake.host.answer(id, own.id, 'allow-once');
        expect((await turn.done).outcome).toBe('completed');
        expect(fake.records().find((r) => r.kind === 'given-up')?.answer).toEqual({
          outcome: { outcome: 'cancelled' },
        });
      },
      {
        mutate: (scenario) => {
          const prompt = (scenario.prompts as Json[])[0]!;
          const steps = prompt.steps as Json[];
          prompt.steps = [
            {
              giveUp: {
                id: 'toolu_shared',
                background: true,
                on: path,
                requester: { sessionId: 'helper-1', label: 'fix tests' },
              },
            },
            { permission: { ...(steps[0]!.permission as Json), id: 'toolu_shared' } },
            ...steps.slice(1),
          ];
        },
      },
    );
  });

  it('a request the agent gives up on while the host still decides it opens no card', async () => {
    const path = gate();
    let mode: 'bypass' | 'ask' = 'bypass';
    let writes = 0;
    let release = (): void => {};
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    await withWork(
      'work-permission',
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id);
        await until(() => writes === 1);
        writeFileSync(path, '');
        // The second request comes after the resolve on the same pipe, so the host has read the resolve by then.
        await until(() => writes === 2);
        mode = 'ask';
        release();
        const card = await nth(fake, 1, 'permission');
        expect(card.toolCallId).toBe('toolu_kept');
        fake.host.answer(id, card.id, 'allow-once');
        expect((await turn.done).outcome).toBe('completed');
        expect(fake.events.interactions).toHaveLength(1);
        expect(fake.events.closed).toEqual([{ sessionId: id, id: card.id, reason: 'answered' }]);
        await until(() => fake.records().some((r) => r.kind === 'given-up'));
        expect(fake.records().find((r) => r.kind === 'given-up')?.answer).toEqual({
          outcome: { outcome: 'cancelled' },
        });
      },
      {
        permission: { modeOf: () => mode, watched: () => true },
        onModeAllowed: async () => {
          writes += 1;
          await held;
          return true;
        },
        mutate: (scenario) => {
          const prompt = (scenario.prompts as Json[])[0]!;
          const steps = prompt.steps as Json[];
          prompt.steps = [
            { giveUp: { id: 'toolu_gone', on: path } },
            { permission: { ...(steps[0]!.permission as Json), id: 'toolu_kept' } },
            ...steps.slice(1),
          ];
        },
      },
    );
  });
});

describe('deny with a note', () => {
  const NOTE = 'use pnpm, not npm';
  /** `work-deny` with a second request after the first, in the same turn. */
  const twoAsks = (scenario: Json): void => {
    const prompt = (scenario.prompts as Json[])[0]!;
    const steps = prompt.steps as Json[];
    const second: Json = {
      permission: {
        toolName: 'run_terminal_command',
        kind: 'execute',
        title: 'Execute `pnpm add left-pad`',
        rawInput: { variant: 'Bash', command: 'pnpm add left-pad', description: 'add', is_background: false },
      },
    };
    prompt.steps = [steps[0]!, second, ...steps.slice(1)];
  };

  it('sends the note as followup_message; the answer is not denied, and a later request of it still asks', async () => {
    await withWork(
      'work-deny',
      async (fake) => {
        const id = await session(fake);
        const turn = start(fake, id, 'clean');
        const card = await nth(fake, 1, 'permission');
        const denied = fake.host.answer(id, card.id, 'reject-once', NOTE);
        expect(denied).toEqual({ toolCallId: card.toolCallId, access: 'bash', card: { ...card, options: [] } });
        expect(fake.events.closed).toEqual([{ sessionId: id, id: card.id, reason: 'answered' }]);
        const second = await nth(fake, 2, 'permission');
        expect(second.command).toBe('pnpm add left-pad');
        expect(answers(fake)).toEqual([
          { outcome: { outcome: 'selected', optionId: 'reject-once' }, _meta: { followup_message: NOTE } },
        ]);
        expect(fake.records().filter((r) => r.kind === 'note')).toEqual([
          { kind: 'note', toolCallId: card.toolCallId, note: NOTE },
        ]);
        fake.host.answer(id, second.id, 'allow-once');
        expect((await turn.done).outcome).toBe('completed');
      },
      { mutate: twoAsks },
    );
  });

  it('a blank note is a plain Deny: no _meta, and the answer ends denied', async () => {
    await withWork('work-deny', async (fake) => {
      const id = await session(fake);
      const turn = start(fake, id, 'clean');
      const card = await nth(fake, 1, 'permission');
      expect(fake.host.answer(id, card.id, 'reject-once', '  \n　﻿')).toBeNull();
      expect((await turn.done).outcome).toBe('denied');
      expect(answers(fake)).toEqual([{ outcome: { outcome: 'selected', optionId: 'reject-once' } }]);
      expect(fake.records().some((r) => r.kind === 'note')).toBe(false);
    });
  });

  it('a note with any option but reject-once is refused and the card stays open', async () => {
    await withWork('work-deny', async (fake) => {
      const id = await session(fake);
      const turn = start(fake, id, 'clean');
      const card = await nth(fake, 1, 'permission');
      expect(() => fake.host.answer(id, card.id, 'allow-once', NOTE)).toThrow(InteractionError);
      expect(fake.host.openInteractions(id).map((i) => i.id)).toEqual([card.id]);
      fake.host.answer(id, card.id, 'reject-once');
      expect((await turn.done).outcome).toBe('denied');
    });
  });

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
    for (const dir of readdirSync(root)) {
      const path = join(root, dir, id, 'updates.jsonl');
      if (existsSync(path)) return readFileSync(path, 'utf8');
    }
    throw new Error('no stored session');
  }

  it('the service: the mark, one note row for its own call, the history line; the stored note reads as the same row', async () => {
    const fake = fakeService('work-deny');
    const project = join(fake.root, 'home', 'code', 'app');
    mkdirSync(project, { recursive: true });
    const window = fake.service.watchHost(() => undefined);
    try {
      const { conversation } = await fake.service.create({ mode: 'work', text: 'clean', cwd: project });
      const page = watchState(fake, conversation.id);
      try {
        await until(() => (page.state()?.interactions.length ?? 0) === 1, 5_000);
        const card = page.state()!.interactions[0] as PermissionInteraction;
        await fake.service.answer(conversation.id, card.id, { optionId: 'reject-once', note: NOTE });
        const notes = (): TranscriptEntry[] =>
          (page.state()?.entries ?? []).filter((e) => e.kind === 'user' && e.note === true);
        await until(() => notes().length === 1, 5_000);
        expect(page.state()!.permissionMarks).toEqual([
          {
            kind: 'denied',
            at: expect.any(Number),
            toolCallId: card.toolCallId,
            access: 'bash',
            by: 'note',
            card: { ...card, options: [] },
          },
        ]);
        await until(() => page.state()?.turn === null, 8_000);
        expect(page.state()!.lastEnd?.outcome).toBe('completed');
        const live = notes();
        expect(live).toEqual([{ kind: 'user', ts: expect.any(Number), text: NOTE, promptIndex: null, note: true }]);
        const line = readFileSync(join(fake.paths.permissionLog, `${conversation.id}.jsonl`), 'utf8');
        expect(JSON.parse(line) as unknown).toEqual({
          kind: 'denied',
          at: page.state()!.permissionMarks[0]!.at,
          toolCallId: card.toolCallId,
          access: 'bash',
          by: 'note',
        });
        expect(line).not.toContain(NOTE);
        const stored = storedUpdates(fake, conversation.id);
        expect(stored.split(`"text":"${NOTE}"`).length - 1).toBe(2);
        const disk = parseUpdates(stored).transcript.filter((e) => e.kind === 'user');
        expect(disk.map((e) => (e.kind === 'user' ? [e.text, e.note === true] : null))).toEqual([
          ['clean', false],
          [NOTE, true],
        ]);
        expect({ ...disk[1], ts: 0 }).toEqual({ ...live[0], ts: 0 });
      } finally {
        page.off();
      }
    } finally {
      window();
      await fake.close();
    }
  }, 20_000);

  it("the service: a helper's call denied with a note gets a mark with its child and no note row", async () => {
    const fake = fakeService('work-workflow-run');
    const project = join(fake.root, 'home', 'code', 'app');
    mkdirSync(project, { recursive: true });
    const window = fake.service.watchHost(() => undefined);
    try {
      const { conversation } = await fake.service.create({ mode: 'work', text: 'research', cwd: project });
      const page = watchState(fake, conversation.id);
      const open = (): PermissionInteraction[] =>
        (page.state()?.interactions ?? []).filter((i): i is PermissionInteraction => i.kind === 'permission');
      try {
        await until(() => open().some((i) => i.toolName === 'workflow'), 5_000);
        const startCard = open().find((i) => i.toolName === 'workflow')!;
        await fake.service.answer(conversation.id, startCard.id, { optionId: 'allow-once' });
        await until(() => open().some((i) => i.from !== undefined), 8_000);
        const ask = open().find((i) => i.from !== undefined)!;
        await fake.service.answer(conversation.id, ask.id, { optionId: 'reject-once', note: NOTE });
        await until(() => fake.records().some((r) => r.kind === 'child-answer'), 5_000);
        expect(fake.records().find((r) => r.kind === 'child-answer')).toMatchObject({
          answer: 'reject-once',
          note: NOTE,
        });
        const denied = page.state()!.permissionMarks.filter((m) => m.kind === 'denied');
        expect(denied).toEqual([
          {
            kind: 'denied',
            at: expect.any(Number),
            toolCallId: ask.toolCallId,
            access: 'bash',
            by: 'note',
            card: { ...ask, options: [] },
            child: ask.from!.child,
          },
        ]);
        // A new watcher's sync holds the view as it is now; the open stream sends ops a flush later.
        const now = watchState(fake, conversation.id);
        try {
          await until(() => now.state() !== null, 5_000);
          expect(now.state()!.entries.some((e) => e.kind === 'user' && e.note === true)).toBe(false);
          expect(now.state()!.entries.some((e) => e.kind === 'user' && e.text === NOTE)).toBe(false);
        } finally {
          now.off();
        }
      } finally {
        page.off();
      }
    } finally {
      window();
      await fake.close();
    }
  }, 30_000);
});

describe('background work holds the agent process (H-a)', () => {
  const gates: string[] = [];
  afterEach(() => {
    for (const dir of gates.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  /** A fresh folder; the fake waits for each gate file in it to exist. */
  function gateDir(): (name: string) => string {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-bg-gate-'));
    gates.push(dir);
    return (name) => join(dir, name);
  }

  /** One prompt per word: its background step, then a short answer. Any other prompt gets "ok". */
  function prompts(words: Record<string, Json>): (scenario: Json) => void {
    return (scenario) => {
      scenario.prompts = Object.entries(words).map(([match, step]) => ({
        match,
        steps: [step, { text: 'Started.' }, { end: { stopReason: 'end_turn' } }],
      }));
    };
  }

  const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
  const pids = (fake: FakeHost): unknown[] =>
    fake
      .records()
      .filter((r) => r.kind === 'start')
      .map((r) => r.pid);

  async function turn(fake: FakeHost, id: string, text: string): Promise<void> {
    expect((await start(fake, id, text).done).outcome).toBe('completed');
  }

  it('a stale host keeps the agent for a live command, monitor or helper; their end line restarts it', async () => {
    const steps: Array<[string, (endOn: string) => Json]> = [
      ['cmd', (endOn) => ({ background: { endOn, wake: false } })],
      ['mon', (endOn) => ({ background: { monitor: true, endOn, wake: false } })],
      ['helper', (endOn) => ({ helper: { endOn, wake: false } })],
    ];
    for (const [word, step] of steps) {
      const gate = gateDir();
      const end = gate('end');
      await withWork(
        'work-workflow-run',
        async (fake) => {
          const id = await session(fake);
          await turn(fake, id, word);
          const helper = word === 'helper';
          expect(fake.host.engineBusy(id)).toBe(helper);
          expect(fake.host.isWorking(id)).toBe(true);
          expect(fake.host.active()).toEqual([
            {
              sessionId: id,
              busy: false,
              waiting: false,
              asks: 0,
              stopping: false,
              background: helper,
              helpers: helper,
              tasks: helper ? 0 : 1,
              op: null,
            },
          ]);
          fake.host.markStale();
          await sleep(300);
          expect(fake.host.modeState()).toMatchObject({ state: 'ready', stale: true, tasks: helper ? 0 : 1 });
          writeFileSync(end, '');
          await until(() => fake.host.modeState().state === 'stopped', 5_000);
          expect(fake.host.modeState()).toMatchObject({ stale: false, tasks: 0, background: 0 });
        },
        { mutate: prompts({ [word]: step(end) }) },
      );
    }
  }, 30_000);

  it('a due wake holds the restart: a command that ends with will_wake keeps the agent until its wake ends', async () => {
    const gate = gateDir();
    const [end, wakeEnd] = [gate('end'), gate('wake-end')];
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await turn(fake, id, 'cmd');
        fake.host.markStale();
        writeFileSync(end, '');
        await until(() => fake.events.turns.some((t) => t.turn?.origin === 'wake'), 5_000);
        expect(fake.host.modeState()).toMatchObject({ state: 'ready', stale: true, tasks: 0 });
        expect(pids(fake)).toHaveLength(1);
        writeFileSync(wakeEnd, '');
        await until(() => fake.host.modeState().state === 'stopped', 5_000);
        expect(fake.events.ends.map((e) => e.result.outcome)).toEqual(['completed', 'completed']);
      },
      { mutate: prompts({ cmd: { background: { endOn: end, wakeAfterMs: 300, wakeEndOn: wakeEnd } } }) },
    );
  }, 20_000);

  it('a schedule alone does not hold the stale restart', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await turn(fake, id, 'sched');
        expect(fake.host.isWorking(id)).toBe(false);
        fake.host.markStale();
        await until(() => fake.host.modeState().state === 'stopped', 5_000);
      },
      { mutate: prompts({ sched: { schedule: {} } }) },
    );
  });

  it('a settings change with a live command: a new turn runs on the old process, the restart waits for the command', async () => {
    const gate = gateDir();
    const end = gate('end');
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await turn(fake, id, 'cmd');
        fake.host.markStale();
        await turn(fake, id, 'hello');
        expect(pids(fake)).toHaveLength(1);
        expect(fake.host.modeState()).toMatchObject({ state: 'ready', stale: true, engineLost: false, tasks: 1 });
        writeFileSync(end, '');
        await until(() => fake.host.modeState().state === 'stopped', 5_000);
      },
      { mutate: prompts({ cmd: { background: { endOn: end, wake: false } } }) },
    );
  });

  it('the engine restarted under a live command: the agent stays, history ops refuse, the next message restarts it', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await turn(fake, id, 'cmd');
        const before = fake.endpoint.current!;
        fake.endpoint.current = { ...before, pid: before.pid + 1 };
        await fake.host.checkEndpoint();
        await sleep(300);
        expect(fake.host.modeState()).toMatchObject({ state: 'ready', stale: true, engineLost: true, tasks: 1 });
        let refused: unknown = null;
        try {
          fake.host.compact(id, { modelId: MODEL, cwd: fake.project }, null);
        } catch (error) {
          refused = error;
        }
        expect(refused).toBeInstanceOf(AgentHostError);
        expect((refused as AgentHostError).code).toBe('engine-lost');
        await sleep(200);
        expect(pids(fake)).toHaveLength(1);
        expect(fake.sent().some((m) => m.method === '_x.ai/compact_conversation')).toBe(false);
        await turn(fake, id, 'hello');
        expect(pids(fake)).toHaveLength(2);
        expect(fake.host.modeState()).toMatchObject({ state: 'ready', stale: false, engineLost: false, tasks: 0 });
      },
      { mutate: prompts({ cmd: { background: { wake: false } } }) },
    );
  }, 20_000);

  it('a turn that finds the engine moved itself (no endpoint check before) restarts over a live command', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await turn(fake, id, 'cmd');
        const before = fake.endpoint.current!;
        fake.endpoint.current = { ...before, pid: before.pid + 1 };
        expect(fake.host.modeState()).toMatchObject({ stale: false, engineLost: false, tasks: 1 });
        await turn(fake, id, 'hello');
        expect(pids(fake)).toHaveLength(2);
        expect(fake.host.modeState()).toMatchObject({ stale: false, engineLost: false, tasks: 0 });
      },
      { mutate: prompts({ cmd: { background: { wake: false } } }) },
    );
  }, 20_000);

  it('a model change and a delete: a command blocks the delete only; a helper blocks both', async () => {
    const gate = gateDir();
    const end = gate('end');
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        await turn(fake, id, 'cmd');
        await fake.host.setModel(id, MODEL);
        await expect(fake.host.deleteSession(id, fake.project)).rejects.toMatchObject({ code: 'conversation-busy' });
        await turn(fake, id, 'helper');
        await expect(fake.host.setModel(id, MODEL)).rejects.toMatchObject({ code: 'conversation-busy' });
        writeFileSync(end, '');
        await until(() => !fake.host.engineBusy(id), 5_000);
        await fake.host.setModel(id, MODEL);
      },
      { mutate: prompts({ cmd: { background: { wake: false } }, helper: { helper: { endOn: end, wake: false } } }) },
    );
  }, 20_000);
});

describe('the wake row names what ended (H-b)', () => {
  const dirs: string[] = [];
  afterEach(() => {
    for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  function gate(name: string): string {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-wake-gate-'));
    dirs.push(dir);
    return join(dir, name);
  }

  /** One prompt per word with its steps, then a short answer. */
  function prompts(words: Record<string, Json[]>): (scenario: Json) => void {
    return (scenario) => {
      scenario.prompts = Object.entries(words).map(([match, steps]) => ({
        match,
        steps: [...steps, { text: 'Started.' }, { turnCompleted: {} }, { end: { stopReason: 'end_turn' } }],
      }));
    };
  }

  type Wake = Extract<TranscriptEntry, { kind: 'wake' }>;
  const wakesOf = (entries: readonly TranscriptEntry[]): Array<Omit<Wake, 'ts' | 'text'>> =>
    entries.flatMap((e) => {
      if (e.kind !== 'wake') return [];
      const { ts: _ts, text: _text, ...rest } = e;
      return [rest];
    });

  /** The page's reducer: the host's typed prompt, then every line the host sent. */
  function live(fake: FakeHost, id: string, typed: string): TranscriptEntry[] {
    const reducer = new TranscriptReducer();
    reducer.userPrompt(typed);
    for (const u of fake.events.updates) if (u.sessionId === id) reducer.apply(u.line);
    return reducer.snapshot().entries;
  }

  /** The stored transcript once it holds `ends` turn ends. */
  async function stored(fake: FakeHost, id: string, ends: number): Promise<TranscriptEntry[]> {
    const files = () =>
      readdirSync(fake.root, { recursive: true, encoding: 'utf8' })
        .filter((f) => f.endsWith('updates.jsonl') && f.includes(id))
        .map((f) => parseUpdates(readFileSync(join(fake.root, f), 'utf8')).transcript);
    // The agent stores a line after it sends it: the wake's end can reach the host first.
    await until(() => files().some((t) => t.filter((e) => e.kind === 'turn-end').length === ends), 5_000);
    const found = files();
    expect(found).toHaveLength(1);
    return found[0]!;
  }

  /** The host's own wake lines (the agent sends a wake's prompt to disk only). */
  function hostLines(fake: FakeHost, id: string): Json[] {
    return fake.events.updates
      .filter((u) => u.sessionId === id && u.line.update.sessionUpdate === 'user_message_chunk')
      .map((u) => u.line.update._meta as Json);
  }

  // Mutation: stamp only a run's id in startWake → the row reads 'unknown' until the wake ends.
  it('a will_wake end makes the wake row name the command, the monitor or the helper before its turn ends', async () => {
    const cases: Array<[string, (end: string, wakeEnd: string) => Json, Omit<Wake, 'ts' | 'text'>]> = [
      [
        'cmd',
        (endOn, wakeEndOn) => ({ background: { endOn, wakeEndOn } }),
        { kind: 'wake', cause: 'task', task: { id: 'task-toolu_1', kind: 'command', label: 'Start the dev server' } },
      ],
      [
        'mon',
        (endOn, wakeEndOn) => ({ background: { monitor: true, endOn, wakeEndOn } }),
        { kind: 'wake', cause: 'task', task: { id: 'task-toolu_1', kind: 'monitor', label: 'Watch the build log' } },
      ],
      [
        'helper',
        (endOn, wakeEndOn) => ({ helper: { endOn, wakeEndOn } }),
        { kind: 'wake', cause: 'subagent', subagent: { childId: '', description: 'Find the flaky test' } },
      ],
    ];
    for (const [word, step, expected] of cases) {
      const [end, wakeEnd] = [gate('end'), gate('wake-end')];
      await withWork(
        'work-workflow-run',
        async (fake) => {
          const id = await session(fake);
          expect((await start(fake, id, word).done).outcome).toBe('completed');
          writeFileSync(end, '');
          await until(() => fake.events.turns.some((t) => t.turn?.origin === 'wake'), 5_000);
          const turnEnds = fake.events.updates.filter(
            (u) => u.sessionId === id && u.line.update.sessionUpdate === 'turn_completed',
          );
          expect(turnEnds).toHaveLength(1);
          const want =
            expected.cause === 'subagent'
              ? { ...expected, subagent: { childId: `${id}-helper-toolu_1`, description: 'Find the flaky test' } }
              : expected;
          const promptId =
            expected.cause === 'subagent' ? `subagent-completed-${id}-helper-toolu_1` : 'task-completed-task-toolu_1';
          expect(hostLines(fake, id)).toEqual([{ hideFromScrollback: true, mlxWakePromptId: promptId }]);
          expect(wakesOf(live(fake, id, word))).toEqual([want]);
          writeFileSync(wakeEnd, '');
          await until(() => fake.events.ends.length === 2, 5_000);
          expect(wakesOf(live(fake, id, word))).toEqual([want]);
          expect(wakesOf(await stored(fake, id, 2))).toEqual([want]);
        },
        { mutate: prompts({ [word]: [step(end, wakeEnd)] }) },
      );
    }
  }, 30_000);

  // Mutation: make an end with `will_wake: false` due (`wakeDueOf`) → the drain's host line carries a task id.
  it('a wake with nothing due is stamped with nothing; its end lists the commands that ended with no wake', async () => {
    const [a, b, drain] = [gate('a'), gate('b'), gate('drain')];
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        expect((await start(fake, id, 'ended').done).outcome).toBe('completed');
        // The fake watches each gate on its own: b opens only after a's end was read, so the ends come in order.
        writeFileSync(a, '');
        await until(() => fake.host.modeState().tasks === 1, 5_000);
        writeFileSync(b, '');
        await until(() => fake.host.modeState().tasks === 0, 5_000);
        writeFileSync(drain, '');
        await until(() => fake.events.ends.length === 2, 5_000);
        expect(hostLines(fake, id)).toEqual([{ hideFromScrollback: true }]);
        const want = {
          kind: 'wake',
          cause: 'notifications',
          tasks: [
            { id: 'task-toolu_1', kind: 'command', label: 'npm test' },
            { id: 'task-toolu_2', kind: 'command', label: 'cargo build' },
          ],
        };
        expect(wakesOf(live(fake, id, 'ended'))).toEqual([want]);
        expect(wakesOf(await stored(fake, id, 2))).toEqual([want]);
      },
      {
        mutate: prompts({
          ended: [
            { background: { command: 'npm test', description: 'npm test', endOn: a, wake: false } },
            { background: { command: 'cargo build', description: 'cargo build', endOn: b, wake: false } },
            { drain: { on: drain, wakeEndAfterMs: 0 } },
          ],
        }),
      },
    );
  }, 20_000);
});

describe('helper progress (#10)', () => {
  const dirs: string[] = [];
  afterEach(() => {
    for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  function gate(name: string): string {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-progress-gate-'));
    dirs.push(dir);
    return join(dir, name);
  }

  /** The prompt "go": these steps, then a short answer. */
  function prompts(steps: Json[]): (scenario: Json) => void {
    return (scenario) => {
      scenario.prompts = [
        { match: 'go', steps: [...steps, { text: 'Started.' }, { end: { stopReason: 'end_turn' } }] },
      ];
    };
  }

  const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
  /** Lines the host passed on to the reducer that are ticks: none may be. */
  const passedTicks = (fake: FakeHost) =>
    fake.events.updates.filter((u) => u.line.update.sessionUpdate === 'subagent_progress');

  // Mutation: send onActivity at each tick (no timer) → ten events.
  it('ten ticks in one window make one live event with the last numbers; no tick reaches the reducer', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        expect((await start(fake, id).done).outcome).toBe('completed');
        await until(() => fake.events.activity.length > 0, 5_000);
        await sleep(500);
        const tick = {
          childId: 'helper-1',
          turns: 1,
          toolCalls: 10,
          tokens: 1_000,
          contextPct: 3,
          errors: 0,
          failedCalls: 0,
        };
        expect(fake.events.activity).toEqual([
          {
            sessionId: id,
            live: {
              helpers: [expect.objectContaining(tick)],
              agents: [{ childId: 'helper-1', run: null, attempts: ['helper-1'], stopping: false, messages: 'on' }],
              tasks: [],
              schedules: [],
              reports: [],
              stoppedByYou: [],
            },
          },
        ]);
        expect(fake.host.liveActivity(id).helpers).toEqual([expect.objectContaining(tick)]);
        expect(passedTicks(fake)).toEqual([]);
      },
      {
        timings: { activityEveryMs: 1_500 },
        mutate: prompts([
          { helper: { child: 'helper-1' } },
          {
            helperProgress: { ticks: Array.from({ length: 10 }, (_, i) => ({ tool_call_count: i + 1 })), everyMs: 10 },
          },
        ]),
      },
    );
  }, 20_000);

  // Mutation: keep a tick of any child (no tracker check) → the stranger's 98 calls show.
  it('a tick a helper of the helper sends, or one for a helper this conversation did not start, reaches nothing', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        expect((await start(fake, id).done).outcome).toBe('completed');
        await until(() => fake.events.activity.length > 0, 5_000);
        await sleep(400);
        // The helper's start makes an event of its own, before any tick.
        const ticked = fake.events.activity.filter((e) => e.live.helpers.length > 0);
        expect(ticked.map((e) => e.live.helpers.map((h) => [h.childId, h.toolCalls]))).toEqual([[['helper-1', 5]]]);
        expect(passedTicks(fake)).toEqual([]);
      },
      {
        timings: { activityEveryMs: 200 },
        mutate: prompts([
          { helper: { child: 'helper-1' } },
          { helperProgress: { sessionId: 'helper-1', child: 'helper-1-nested', ticks: [{ tool_call_count: 99 }] } },
          { helperProgress: { child: 'stranger', ticks: [{ tool_call_count: 98 }] } },
          { helperProgress: { child: 'helper-1', afterMs: 300, ticks: [{ tool_call_count: 5 }] } },
        ]),
      },
    );
  }, 20_000);

  // Mutation: drop the tickedCount check in trackWork → no event after the end, the page keeps the old tick.
  it('a helper that ends leaves the live facts; its row keeps its turns and tool calls, live and on disk', async () => {
    const end = gate('end');
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        expect((await start(fake, id).done).outcome).toBe('completed');
        await until(() => fake.events.activity.length === 1, 5_000);
        expect(fake.events.activity[0]!.live.helpers.map((h) => h.toolCalls)).toEqual([2]);
        writeFileSync(end, '');
        await until(() => fake.events.activity.length === 2, 5_000);
        expect(fake.events.activity[1]!.live).toEqual({
          helpers: [],
          agents: [],
          tasks: [],
          schedules: [],
          reports: [],
          stoppedByYou: [],
        });
        const helper = (entries: readonly TranscriptEntry[]) => {
          const found = entries.find((e) => e.kind === 'subagent');
          if (found?.kind !== 'subagent') throw new Error('no subagent row');
          return { status: found.status, turns: found.turns, toolCalls: found.toolCalls, tokens: found.tokens };
        };
        const reducer = new TranscriptReducer();
        reducer.userPrompt('go');
        for (const u of fake.events.updates) if (u.sessionId === id) reducer.apply(u.line);
        const want = { status: 'completed', turns: 2, toolCalls: 3, tokens: 1_200 };
        expect(helper(reducer.snapshot().entries)).toEqual(want);
        const files = readdirSync(fake.root, { recursive: true, encoding: 'utf8' })
          .filter((f) => f.endsWith('updates.jsonl') && f.includes(id))
          .map((f) => readFileSync(join(fake.root, f), 'utf8'));
        expect(files).toHaveLength(1);
        expect(files[0]).not.toContain('subagent_progress');
        expect(helper(parseUpdates(files[0]!).transcript)).toEqual(want);
      },
      {
        timings: { activityEveryMs: 200 },
        mutate: prompts([
          { helper: { child: 'helper-1', endOn: end, wake: false } },
          { helperProgress: { ticks: [{ tool_call_count: 2 }] } },
        ]),
      },
    );
  }, 20_000);

  // Mutation: leave `live` out of hostView → the second page's sync has no helper.
  it('a page that attaches while a helper ticks gets its numbers in the sync; the activity event replaces them', async () => {
    const fake = fakeService('work-workflow-run', {
      timings: { activityEveryMs: 100 },
      mutate: prompts([
        { helper: { child: 'helper-1' } },
        { helperProgress: { ticks: [{ tool_call_count: 1 }, { tool_call_count: 4 }], everyMs: 1_500 } },
      ]),
    });
    const project = join(fake.root, 'home', 'code', 'app');
    mkdirSync(project, { recursive: true });
    try {
      const { conversation } = await fake.service.create({ mode: 'work', text: 'go', cwd: project });
      let state: ConversationState | null = null;
      const off = fake.service.watchConversation(conversation.id, (event) => {
        const next = applyConversationEvent(state, event);
        if (next !== 'resync') state = next;
      });
      try {
        const calls = (): number[] => (state as ConversationState | null)?.live.helpers.map((h) => h.toolCalls) ?? [];
        await until(() => calls()[0] === 1, 8_000);
        let second: ConversationState | null = null;
        const again = fake.service.watchConversation(conversation.id, (event) => {
          const next = applyConversationEvent(second, event);
          if (next !== 'resync') second = next;
        });
        try {
          await until(() => second !== null, 5_000);
          expect((second as ConversationState | null)?.live.helpers.map((h) => [h.childId, h.toolCalls])).toEqual([
            ['helper-1', 1],
          ]);
          await until(() => calls()[0] === 4, 8_000);
          expect((second as ConversationState | null)?.live.helpers.map((h) => h.toolCalls)).toEqual([4]);
        } finally {
          again();
        }
      } finally {
        off();
      }
    } finally {
      await fake.close();
    }
  }, 20_000);
});

describe('Stop a command or monitor (T1)', () => {
  const gates: string[] = [];
  afterEach(() => {
    for (const dir of gates.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  function gate(name: string): string {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-stop-gate-'));
    gates.push(dir);
    return join(dir, name);
  }

  /** Every prompt starts one background step, then answers "Started.". */
  function background(step: Json, faults: Json = {}): (scenario: Json) => void {
    return (scenario) => {
      scenario.prompts = [
        { match: '', steps: [{ background: step }, { text: 'Started.' }, { end: { stopReason: 'end_turn' } }] },
      ];
      scenario.faults = { ...(scenario.faults as Json | undefined), ...faults };
    };
  }

  const kills = (fake: FakeHost): Json[] =>
    fake
      .sent()
      .filter((m) => m.method === '_x.ai/task/kill')
      .map((m) => m.params as Json);
  const xaiLines = (fake: FakeHost, kind: string): Json[] =>
    fake.events.updates.filter((u) => u.line.xai && u.line.update.sessionUpdate === kind).map((u) => u.line.update);
  const pids = (fake: FakeHost): unknown[] =>
    fake
      .records()
      .filter((r) => r.kind === 'start')
      .map((r) => r.pid);

  async function started(fake: FakeHost): Promise<{ id: string; task: string }> {
    const id = await session(fake);
    expect((await start(fake, id).done).outcome).toBe('completed');
    const task = xaiLines(fake, 'task_backgrounded')[0]!.task_id as string;
    expect(fake.host.liveActivity(id).tasks).toEqual([
      { id: task, kind: 'command', stopping: false, events: 0, recent: [] },
    ]);
    return { id, task };
  }

  function entriesOf(fake: FakeHost, id: string): TranscriptEntry[] {
    const reducer = new TranscriptReducer();
    reducer.userPrompt('go');
    for (const u of fake.events.updates) if (u.sessionId === id) reducer.apply(u.line);
    return reducer.snapshot().entries;
  }

  // Mutation: send the kill before `stopping` is set → the row keeps its Stop while the kill is out.
  it('killed: the row reads stopping at the press, the kill names clientUi, the end line ends it, a wake tells the model', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const { id, task } = await started(fake);
        const stop = fake.host.stopTask(id, task);
        // At once, not at the next one-second window: the press is the user's.
        expect(fake.events.activity.at(-1)).toEqual({
          sessionId: id,
          live: {
            helpers: [],
            agents: [],
            tasks: [{ id: task, kind: 'command', stopping: true, events: 0, recent: [] }],
            schedules: [],
            reports: [],
            stoppedByYou: [],
          },
        });
        expect(await stop).toBe('stopped');
        expect(kills(fake)).toEqual([{ sessionId: id, taskId: task, source: 'clientUi' }]);
        await until(() => fake.host.modeState().tasks === 0, 5_000);
        const end = xaiLines(fake, 'task_completed')[0]!;
        expect(end).toMatchObject({ will_wake: true, task_snapshot: { explicitly_killed: true } });
        await until(() => fake.events.ends.length === 2, 5_000);
        expect(fake.events.ends.map((e) => e.result.outcome)).toEqual(['completed', 'completed']);
        expect(fake.events.turns.some((t) => t.turn?.origin === 'wake')).toBe(true);
        await until(() => fake.events.activity.at(-1)?.live.tasks.length === 0, 5_000);
        expect(fake.host.isWorking(id)).toBe(false);
        const tool = entriesOf(fake, id).find((e) => e.kind === 'tool' && e.task !== undefined);
        expect(tool?.kind === 'tool' ? tool.task : null).toMatchObject({ id: task, killed: true });
        expect(tool?.kind === 'tool' ? tool.task?.endTs : undefined).toBeTypeOf('number');
      },
      { mutate: background({}) },
    );
  }, 20_000);

  // Mutation: drop `source` from the kill → the rewind's stop starts a wake turn about a command the model forgot.
  it('teardown (the rewind): the agent counts it delivered and starts no wake', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const { id, task } = await started(fake);
        expect(await fake.host.stopTask(id, task, 'teardown')).toBe('stopped');
        expect(kills(fake)).toEqual([{ sessionId: id, taskId: task, source: 'teardown' }]);
        await until(() => fake.host.modeState().tasks === 0, 5_000);
        expect(xaiLines(fake, 'task_completed')[0]).toMatchObject({
          will_wake: false,
          task_snapshot: { explicitly_killed: true, kill_result_delivered: true },
        });
        await new Promise((resolve) => setTimeout(resolve, 400));
        expect(fake.events.ends).toHaveLength(1);
        expect(fake.host.wakeDue(id)).toBe(false);
      },
      { mutate: background({}) },
    );
  }, 20_000);

  it('already_exited is a success: the task stays stopping until its own end line', async () => {
    const end = gate('end');
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const { id, task } = await started(fake);
        expect(await fake.host.stopTask(id, task)).toBe('stopped');
        expect(fake.host.liveActivity(id).tasks).toEqual([
          { id: task, kind: 'command', stopping: true, events: 0, recent: [] },
        ]);
        expect(fake.host.isWorking(id)).toBe(true);
        writeFileSync(end, '');
        await until(() => fake.host.liveActivity(id).tasks.length === 0, 5_000);
      },
      { mutate: background({ endOn: end, wake: false }, { killReply: 'already_exited' }) },
    );
  }, 20_000);

  // Mutation: keep the task on not_found → it holds the agent process for ever.
  // Mutation: drop the work step on not_found → the page's state keeps the task and the idle stop never comes.
  it('not_found: nothing to stop, and the task leaves the live facts and no longer holds the agent', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const { id, task } = await started(fake);
        await new Promise((resolve) => setTimeout(resolve, 600));
        expect(fake.host.modeState()).toMatchObject({ state: 'ready', tasks: 1 });
        expect(fake.host.isWorking(id)).toBe(true);
        expect(await fake.host.stopTask(id, task)).toBe('not-live');
        expect(fake.host.isWorking(id)).toBe(false);
        expect(fake.host.modeState().tasks).toBe(0);
        expect(fake.events.activity.at(-1)?.live.tasks).toEqual([]);
        expect(fake.events.states.at(-1)?.tasks).toBe(0);
        await until(() => fake.host.modeState().state === 'stopped', 5_000);
      },
      { timings: { idleStopMs: 300 }, mutate: background({ wake: false }, { killReply: 'not_found' }) },
    );
  }, 20_000);

  // Mutation: leave `stopping` set on a failure → the row stays "Stopping…" with no Stop.
  it('an error reply or a JSON-RPC error fails, and the task gets its Stop back', async () => {
    for (const killReply of ['error', 'rpc-error']) {
      await withWork(
        'work-workflow-run',
        async (fake) => {
          const { id, task } = await started(fake);
          expect(await fake.host.stopTask(id, task)).toBe('failed');
          expect(fake.events.activity.slice(-2).map((e) => e.live.tasks[0]?.stopping)).toEqual([true, false]);
          expect(fake.host.liveActivity(id).tasks).toEqual([
            { id: task, kind: 'command', stopping: false, events: 0, recent: [] },
          ]);
          expect(fake.host.modeState()).toMatchObject({ state: 'ready', tasks: 1 });
        },
        { mutate: background({ wake: false }, { killReply }) },
      );
    }
  }, 30_000);

  // Mutation: a hard deadline for the kill → the agent process, with every command it holds, is stopped.
  it('no reply in time fails softly: the agent and the command live on', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const { id, task } = await started(fake);
        expect(await fake.host.stopTask(id, task)).toBe('failed');
        await new Promise((resolve) => setTimeout(resolve, 300));
        expect(fake.host.modeState()).toMatchObject({ state: 'ready', tasks: 1 });
        expect(pids(fake)).toHaveLength(1);
        expect(fake.host.liveActivity(id).tasks).toEqual([
          { id: task, kind: 'command', stopping: false, events: 0, recent: [] },
        ]);
      },
      { mutate: background({ wake: false }, { hangOn: '_x.ai/task/kill' }), timings: { taskKillMs: 200 } },
    );
  }, 20_000);

  it('nothing to stop: an unknown conversation or task, or a task of a gone process; no kill is sent', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const { id, task } = await started(fake);
        expect(await fake.host.stopTask('01a00000-0000-7000-8000-000000000000', task)).toBe('not-live');
        expect(await fake.host.stopTask(id, 'other-task')).toBe('not-live');
        await fake.host.restart();
        expect(fake.host.isHot(id)).toBe(false);
        expect(await fake.host.stopTask(id, task)).toBe('not-live');
        expect(kills(fake)).toEqual([]);
      },
      { mutate: background({ wake: false }) },
    );
  }, 20_000);
});

describe('Stop or message a subagent', () => {
  const CHILD = 'helper-1';
  const ADDRESS = `addr-${CHILD}`;
  const gates: string[] = [];
  afterEach(() => {
    for (const dir of gates.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  function gate(name: string): string {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-subagent-gate-'));
    gates.push(dir);
    return join(dir, name);
  }

  /** Every prompt starts one background helper (no wake at its end), then answers "Started.". */
  function helper(step: Json = {}, faults: Json = {}): (scenario: Json) => void {
    return (scenario) => {
      scenario.prompts = [
        {
          match: '',
          steps: [
            { helper: { child: CHILD, wake: false, ...step } },
            { text: 'Started.' },
            { end: { stopReason: 'end_turn' } },
          ],
        },
      ];
      scenario.faults = { ...(scenario.faults as Json | undefined), ...faults };
    };
  }

  const calls = (fake: FakeHost, method: string): Json[] =>
    fake
      .sent()
      .filter((m) => m.method === method)
      .map((m) => m.params as Json);
  const pids = (fake: FakeHost): unknown[] =>
    fake
      .records()
      .filter((r) => r.kind === 'start')
      .map((r) => r.pid);

  async function started(fake: FakeHost): Promise<string> {
    const id = await session(fake);
    expect((await start(fake, id).done).outcome).toBe('completed');
    expect(fake.host.liveActivity(id).agents).toEqual([
      { childId: CHILD, run: null, attempts: [CHILD], stopping: false, messages: 'on' },
    ]);
    return id;
  }

  function entriesOf(fake: FakeHost, id: string): TranscriptEntry[] {
    const reducer = new TranscriptReducer();
    reducer.userPrompt('go');
    for (const u of fake.events.updates) if (u.sessionId === id) reducer.apply(u.line);
    return reducer.snapshot().entries;
  }

  // Mutation: send the cancel before `stopping` is set → the row keeps Stop and Message while the cancel is out.
  // Mutation: name the cancel by `sessionId` only → the agent cannot parse it and the Stop fails.
  it('stop: the row reads stopping at the press, the cancel names the subagent, its end line ends the row', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await started(fake);
        const stop = fake.host.stopSubagent(id, CHILD);
        expect(fake.events.activity.at(-1)).toEqual({
          sessionId: id,
          live: {
            helpers: [],
            agents: [{ childId: CHILD, run: null, attempts: [CHILD], stopping: true, messages: 'on' }],
            tasks: [],
            schedules: [],
            reports: [],
            stoppedByYou: [],
          },
        });
        expect(await stop).toBe('stopped');
        expect(calls(fake, '_x.ai/subagent/cancel')).toEqual([{ subagentId: CHILD }]);
        await until(() => fake.host.liveActivity(id).agents.length === 0, 5_000);
        await until(() => fake.events.activity.at(-1)?.live.agents.length === 0, 5_000);
        const row = entriesOf(fake, id).find((e) => e.kind === 'subagent');
        expect(row?.kind === 'subagent' ? [row.status, row.endTs !== undefined] : null).toEqual(['cancelled', true]);
        expect(fake.host.isWorking(id)).toBe(false);
      },
      { mutate: helper() },
    );
  }, 20_000);

  // Mutation: keep the helper on already_finished or not_found → it holds the agent process until the idle guard.
  it('already_finished and not_found: nothing to stop, and the helper leaves the live facts and the process', async () => {
    for (const cancelReply of ['already_finished', 'not_found']) {
      await withWork(
        'work-workflow-run',
        async (fake) => {
          const id = await started(fake);
          expect(fake.host.isWorking(id)).toBe(true);
          expect(await fake.host.stopSubagent(id, CHILD)).toBe('not-live');
          expect(fake.host.liveActivity(id).agents).toEqual([]);
          expect(fake.events.activity.at(-1)?.live.agents).toEqual([]);
          expect(fake.host.isWorking(id)).toBe(false);
        },
        { mutate: helper({}, { cancelReply }) },
      );
    }
  }, 30_000);

  // Mutation: leave `stopping` set on a failure → the row stays "Stopping…" with no buttons.
  // Mutation: a hard deadline for the cancel → the agent process, with every helper it runs, is stopped.
  it('an error reply, a JSON-RPC error or no reply in time fails softly: the buttons come back, the agent lives on', async () => {
    const cases: Array<[Json, NonNullable<FakeHostOptions['timings']>]> = [
      [{ cancelReply: 'error' }, {}],
      [{ cancelReply: 'rpc-error' }, {}],
      [{ hangOn: '_x.ai/subagent/cancel' }, { subagentCancelMs: 200 }],
    ];
    for (const [faults, timings] of cases) {
      await withWork(
        'work-workflow-run',
        async (fake) => {
          const id = await started(fake);
          expect(await fake.host.stopSubagent(id, CHILD)).toBe('failed');
          expect(fake.events.activity.slice(-2).map((e) => e.live.agents[0]?.stopping)).toEqual([true, false]);
          await new Promise((resolve) => setTimeout(resolve, 300));
          expect(fake.host.liveActivity(id).agents).toEqual([
            { childId: CHILD, run: null, attempts: [CHILD], stopping: false, messages: 'on' },
          ]);
          expect(pids(fake)).toHaveLength(1);
          expect(fake.host.isWorking(id)).toBe(true);
        },
        { mutate: helper({}, faults), timings },
      );
    }
  }, 40_000);

  it('nothing to stop: an unknown conversation or subagent, or one of a gone process; no cancel is sent', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await started(fake);
        expect(await fake.host.stopSubagent('01a00000-0000-7000-8000-000000000000', CHILD)).toBe('not-live');
        expect(await fake.host.stopSubagent(id, 'stranger')).toBe('not-live');
        await fake.host.restart();
        expect(await fake.host.stopSubagent(id, CHILD)).toBe('not-live');
        expect(calls(fake, '_x.ai/subagent/cancel')).toEqual([]);
      },
      { mutate: helper() },
    );
  }, 20_000);

  // Mutation: send `queue: true` → the text waits for a turn of the helper that never comes (its turn end is its end).
  // Mutation: drop GROK_ACTIVE_AGENT_MESSAGES from the Work env → the fake answers method not found (`off`).
  it('a message steers into the helper by its live address; the address never reaches the page', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await started(fake);
        expect(await fake.host.messageSubagent(id, CHILD, 'Skip the tests folder.')).toBe('sent');
        expect(calls(fake, '_x.ai/subagent/message')).toEqual([
          { sessionId: id, agentAddress: ADDRESS, content: [{ type: 'text', text: 'Skip the tests folder.' }] },
        ]);
        const spawned = fake.events.updates.find((u) => u.line.update.sessionUpdate === 'subagent_spawned');
        expect(spawned?.line.update.agentAddress).toBe(ADDRESS);
        expect(JSON.stringify(fake.events.activity)).not.toContain(ADDRESS);
        expect(JSON.stringify(entriesOf(fake, id))).not.toContain(ADDRESS);
        expect(JSON.stringify(fake.host.liveActivity(id))).not.toContain(ADDRESS);
      },
      { mutate: helper() },
    );
  }, 20_000);

  // Mutation: map `not_active` to gone → the page drops a draft for a helper that runs.
  // Mutation: read a missing kind as sent → an error envelope reads "Sent to …".
  it("each answer of the agent reads as the page's result", async () => {
    const replies: Array<[string, string]> = [
      ['not_active', 'not-ready'],
      ['saturated', 'busy'],
      ['not_accepted_before_deadline', 'late'],
      ['admission_uncertain', 'unsure'],
      ['limit', 'too-long'],
      ['unsupported_content', 'error'],
      ['error', 'error'],
      ['rejected', 'gone'],
      ['channel_closed', 'gone'],
    ];
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await started(fake);
        const got: string[] = [];
        for (let i = 0; i < replies.length; i++) got.push(await fake.host.messageSubagent(id, CHILD, `try ${i}`));
        expect(got).toEqual(replies.map(([, result]) => result));
        expect(await fake.host.messageSubagent(id, CHILD, 'again')).toBe('sent');
        expect(calls(fake, '_x.ai/subagent/message')).toHaveLength(replies.length + 1);
      },
      { mutate: helper({}, { messageReplies: replies.map(([kind]) => kind) }) },
    );
  }, 20_000);

  // Mutation: count UTF-16 units → 16,385 two-byte letters go out and the agent refuses them.
  it('the 32 KiB limit counts UTF-8 bytes: one byte past it is too long and is not sent', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await started(fake);
        expect(await fake.host.messageSubagent(id, CHILD, 'é'.repeat(16_385))).toBe('too-long');
        expect(calls(fake, '_x.ai/subagent/message')).toEqual([]);
        expect(await fake.host.messageSubagent(id, CHILD, 'é'.repeat(16_384))).toBe('sent');
      },
      { mutate: helper() },
    );
  }, 20_000);

  // Mutation: forget `off` → each press asks the agent again. Mutation: no push → the row keeps Message on.
  // Mutation: keep `off` in the host, not the process → a newer agent started later never gets a message.
  it('method not found: messages are off for that process, the row says so at once, and nothing more is sent', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await started(fake);
        expect(await fake.host.messageSubagent(id, CHILD, 'hello')).toBe('off');
        expect(fake.events.activity.at(-1)?.live.agents).toEqual([
          { childId: CHILD, run: null, attempts: [CHILD], stopping: false, messages: 'off' },
        ]);
        expect(await fake.host.messageSubagent(id, CHILD, 'hello')).toBe('off');
        expect(calls(fake, '_x.ai/subagent/message')).toHaveLength(1);
        await fake.host.restart();
        expect((await start(fake, id).done).outcome).toBe('completed');
        expect(fake.host.liveActivity(id).agents.map((s) => s.messages)).toEqual(['on']);
      },
      { mutate: helper({}, { messagesOff: true }) },
    );
  }, 30_000);

  // Mutation: a hard deadline for the message → the agent process and its helper are stopped.
  it('no reply in time is unsure: the message may have arrived, nothing is sent again, the agent lives on', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await started(fake);
        expect(await fake.host.messageSubagent(id, CHILD, 'hello')).toBe('unsure');
        await new Promise((resolve) => setTimeout(resolve, 300));
        expect(calls(fake, '_x.ai/subagent/message')).toHaveLength(1);
        expect(pids(fake)).toHaveLength(1);
        expect(fake.host.liveActivity(id).agents).toEqual([
          { childId: CHILD, run: null, attempts: [CHILD], stopping: false, messages: 'on' },
        ]);
      },
      { mutate: helper({}, { hangOn: '_x.ai/subagent/message' }), timings: { subagentMessageMs: 200 } },
    );
  }, 20_000);

  // Mutation: send while the helper is stopping or after its end → the agent wakes an ended helper as a new turn whose
  // result reaches no one.
  it('gone: a helper that is stopping or ended, a stranger, an unknown conversation; nothing is sent', async () => {
    const end = gate('end');
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await started(fake);
        expect(await fake.host.messageSubagent('01a00000-0000-7000-8000-000000000000', CHILD, 'hi')).toBe('gone');
        expect(await fake.host.messageSubagent(id, 'stranger', 'hi')).toBe('gone');
        const stop = fake.host.stopSubagent(id, CHILD);
        expect(await fake.host.messageSubagent(id, CHILD, 'hi')).toBe('gone');
        expect(await stop).toBe('failed');
        writeFileSync(end, '');
        await until(() => fake.host.liveActivity(id).agents.length === 0, 5_000);
        expect(await fake.host.messageSubagent(id, CHILD, 'hi')).toBe('gone');
        expect(calls(fake, '_x.ai/subagent/message')).toEqual([]);
      },
      { mutate: helper({ endOn: end }, { cancelReply: 'error' }) },
    );
  }, 20_000);

  // Mutation: send with no address → the agent refuses an empty address as not owned (gone), and the page drops the draft.
  it('a helper whose live start named no address says no-address, and no message goes out', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        expect((await start(fake, id).done).outcome).toBe('completed');
        expect(fake.host.liveActivity(id).agents).toEqual([
          { childId: CHILD, run: null, attempts: [CHILD], stopping: false, messages: 'no-address' },
        ]);
        expect(await fake.host.messageSubagent(id, CHILD, 'hi')).toBe('error');
        expect(calls(fake, '_x.ai/subagent/message')).toEqual([]);
      },
      { mutate: helper({ address: false }) },
    );
  }, 20_000);
});

describe('Delete a schedule (T3)', () => {
  /** Every prompt makes one schedule (and maybe a background command), then answers "Scheduled.". */
  function scheduling(schedule: Json, more: { faults?: Json; command?: boolean; persist?: boolean } = {}) {
    return (scenario: Json) => {
      scenario.prompts = [
        {
          match: '',
          steps: [
            ...(more.command === true ? [{ background: { taskId: 't1', wake: false } }] : []),
            { schedule: { taskId: 's1', ...schedule } },
            { text: 'Scheduled.' },
            { end: { stopReason: 'end_turn' } },
          ],
        },
      ];
      if (more.persist === true) scenario.persist = true;
      scenario.faults = { ...(scenario.faults as Json | undefined), ...more.faults };
    };
  }

  const deletes = (fake: FakeHost): Json[] =>
    fake
      .sent()
      .filter((m) => m.method === '_x.ai/scheduler/delete')
      .map((m) => m.params as Json);
  const phases = (fake: FakeHost, id: string): Array<string | null> =>
    fake.events.activity
      .filter((e) => e.sessionId === id)
      .map((e) => e.live.schedules.find((s) => s.id === 's1')?.deleting ?? null);
  const xaiLines = (fake: FakeHost, kind: string): Json[] =>
    fake.events.updates.filter((u) => u.line.xai && u.line.update.sessionUpdate === kind).map((u) => u.line.update);
  const starts = (fake: FakeHost): number => fake.records().filter((r) => r.kind === 'start').length;

  async function scheduled(fake: FakeHost): Promise<string> {
    const id = await session(fake);
    expect((await start(fake, id).done).outcome).toBe('completed');
    await until(() => fake.host.liveActivity(id).schedules.length === 1);
    expect(fake.host.liveActivity(id).schedules).toEqual([{ id: 's1', deleting: null, firing: null }]);
    return id;
  }

  const load = (fake: FakeHost) => ({ modelId: MODEL, cwd: fake.project });

  // Mutation: send the delete before the phase is set → the row keeps its Delete while the call is out.
  it('a held schedule: removing at the press, the delete names it, the end line ends it, and it no longer holds the agent', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await scheduled(fake);
        const end = fake.host.deleteSchedule(id, 's1', load(fake));
        expect(fake.events.activity.at(-1)).toEqual({
          sessionId: id,
          live: {
            helpers: [],
            agents: [],
            tasks: [],
            schedules: [{ id: 's1', deleting: 'removing', firing: null }],
            reports: [],
            stoppedByYou: [],
          },
        });
        expect(await end).toBe('deleted');
        expect(deletes(fake)).toEqual([{ sessionId: id, taskId: 's1' }]);
        // The reply can come before the end line: the schedule left the live facts at the reply.
        expect(fake.host.liveActivity(id).schedules).toEqual([]);
        await until(() => xaiLines(fake, 'scheduled_task_deleted').length === 1);
        expect(xaiLines(fake, 'scheduled_task_deleted')).toEqual([
          expect.objectContaining({ task_id: 's1', reason: 'deleted' }),
        ]);
        expect(fake.host.liveActivity(id).schedules).toEqual([]);
        expect(fake.events.activity.at(-1)?.live.schedules).toEqual([]);
        expect(starts(fake)).toBe(1);
        // The schedule held off the idle stop; with it gone the agent stops.
        await until(() => fake.host.modeState().state === 'stopped', 5_000);
      },
      { mutate: scheduling({}), timings: { idleStopMs: 300 } },
    );
  }, 20_000);

  // Mutation: keep a schedule the agent says it does not hold → it holds the idle stop for ever.
  it('gone: the agent holds no such schedule; an error reply or no reply in time fails softly and the row is back', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await scheduled(fake);
        expect(await fake.host.deleteSchedule(id, 'other', load(fake))).toBe('gone');
        expect(fake.host.liveActivity(id).schedules).toEqual([{ id: 's1', deleting: null, firing: null }]);
      },
      { mutate: scheduling({}) },
    );
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await scheduled(fake);
        expect(await fake.host.deleteSchedule(id, 's1', load(fake))).toBe('failed');
        expect(phases(fake, id).slice(-2)).toEqual(['removing', null]);
        expect(fake.host.liveActivity(id).schedules).toEqual([{ id: 's1', deleting: null, firing: null }]);
      },
      { mutate: scheduling({}, { faults: { scheduleDeleteReply: 'error' } }) },
    );
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await scheduled(fake);
        expect(await fake.host.deleteSchedule(id, 's1', load(fake))).toBe('failed');
        await new Promise((resolve) => setTimeout(resolve, 300));
        expect(fake.host.modeState().state).toBe('ready');
        expect(starts(fake)).toBe(1);
        expect(fake.host.liveActivity(id).schedules).toEqual([{ id: 's1', deleting: null, firing: null }]);
      },
      {
        mutate: scheduling({}, { faults: { hangOn: '_x.ai/scheduler/delete' } }),
        timings: { scheduleDeleteMs: 200 },
      },
    );
  }, 40_000);

  // Mutation: keep a held schedule the agent says is gone → it holds the idle stop for ever.
  it('gone for a schedule the host held: it leaves the live facts and no longer holds the agent', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await scheduled(fake);
        expect(await fake.host.deleteSchedule(id, 's1', load(fake))).toBe('gone');
        expect(fake.host.liveActivity(id).schedules).toEqual([]);
        await until(() => fake.host.modeState().state === 'stopped', 5_000);
      },
      { mutate: scheduling({}, { faults: { scheduleDeleteReply: 'gone' } }), timings: { idleStopMs: 300 } },
    );
  }, 20_000);

  // Mutation: leave a Delete in flight out of `holdsIdle` → another answer's end starts the idle stop under the Delete.
  // Mutation: leave a Delete in flight out of `hasWork` → a settings change restarts the agent under the Delete.
  it('a Delete in flight holds the agent: neither the idle stop nor a settings change stops it before the reply', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const a = await session(fake);
        expect((await start(fake, a).done).outcome).toBe('completed');
        const b = await session(fake);
        const end = fake.host.deleteSchedule(a, 's9', load(fake));
        expect((await start(fake, b).done).outcome).toBe('completed');
        await new Promise((resolve) => setTimeout(resolve, 400));
        expect(fake.host.modeState().state).toBe('ready');
        fake.host.markStale();
        await new Promise((resolve) => setTimeout(resolve, 300));
        expect(fake.host.modeState()).toMatchObject({ state: 'ready', stale: true });
        expect(starts(fake)).toBe(1);
        expect(await end).toBe('failed');
        await until(() => fake.host.modeState().state === 'stopped', 5_000);
      },
      {
        mutate: (scenario) => {
          scenario.prompts = [{ match: '', steps: [{ text: 'ok' }, { end: { stopReason: 'end_turn' } }] }];
          scenario.faults = { hangOn: '_x.ai/scheduler/delete' };
        },
        timings: { idleStopMs: 100, scheduleDeleteMs: 1_500 },
      },
    );
  }, 20_000);

  // Mutation: leave a Delete in flight out of `deleteSession` → the agent deletes the session beside the Delete's load.
  it('a Delete in flight refuses the conversation delete until its reply', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await scheduled(fake);
        const end = fake.host.deleteSchedule(id, 's1', load(fake));
        await expect(fake.host.deleteSession(id, fake.project)).rejects.toMatchObject({ code: 'conversation-busy' });
        expect(fake.sent().filter((m) => m.method === '_x.ai/session/delete')).toEqual([]);
        expect(await end).toBe('failed');
        await fake.host.deleteSession(id, fake.project);
        expect(fake.sent().filter((m) => m.method === '_x.ai/session/delete')).toHaveLength(1);
      },
      {
        mutate: scheduling({}, { faults: { hangOn: '_x.ai/scheduler/delete' } }),
        timings: { scheduleDeleteMs: 300 },
      },
    );
  }, 20_000);

  // Mutation: delete with no load on a conversation the agent does not hold → the agent answers "session not found".
  it('a paused schedule (no agent holds the conversation): opening while it loads, then removing, then deleted', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await scheduled(fake);
        await fake.host.restart();
        expect(fake.host.isHot(id)).toBe(false);
        expect(fake.host.liveActivity(id).schedules).toEqual([]);
        const end = fake.host.deleteSchedule(id, 's1', load(fake));
        expect(fake.events.activity.at(-1)?.live.schedules).toEqual([{ id: 's1', deleting: 'opening', firing: null }]);
        // A page that attaches now reads it from its `sync`.
        expect(fake.host.liveActivity(id).schedules).toEqual([{ id: 's1', deleting: 'opening', firing: null }]);
        expect(await end).toBe('deleted');
        expect(fake.host.isHot(id)).toBe(true);
        const order = fake
          .sent()
          .map((m) => m.method)
          .filter((m) => m === 'session/load' || m === '_x.ai/scheduler/delete');
        expect(order).toEqual(['session/load', '_x.ai/scheduler/delete']);
        expect(starts(fake)).toBe(2);
        // The load's re-announce made the schedule live; the delete ended it.
        expect(phases(fake, id)).toEqual(expect.arrayContaining(['opening', 'removing']));
        expect(phases(fake, id).at(-1)).toBeNull();
        expect(fake.host.liveActivity(id).schedules).toEqual([]);
      },
      { mutate: scheduling({}, { persist: true }) },
    );
  }, 20_000);

  // Mutation: drop the `lostTasks()` wall in `deleteSchedule()` → the load restarts the agent and stops the command.
  it('the engine restarted under a live command: the Delete is refused, nothing is sent and the agent stays', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await scheduled(fake);
        await until(() => fake.host.liveActivity(id).tasks.length === 1);
        const engine = fake.endpoint.current!;
        fake.endpoint.current = { ...engine, pid: engine.pid + 1 };
        await fake.host.checkEndpoint();
        await until(() => fake.host.modeState().engineLost === true);
        await expect(fake.host.deleteSchedule(id, 's1', load(fake))).rejects.toMatchObject({ code: 'engine-lost' });
        await new Promise((resolve) => setTimeout(resolve, 200));
        expect(deletes(fake)).toEqual([]);
        expect(starts(fake)).toBe(1);
        expect(fake.host.liveActivity(id).tasks).toHaveLength(1);
        expect(fake.host.liveActivity(id).schedules).toEqual([{ id: 's1', deleting: null, firing: null }]);
      },
      { mutate: scheduling({}, { command: true }) },
    );
  }, 20_000);

  // Mutation: take `scheduled_task_fired` with no look at its run → the row never reads "Running now".
  it('a fire: firing names its run until that run ends', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await scheduled(fake);
        await until(() => fake.host.liveActivity(id).schedules[0]?.firing != null, 5_000);
        const run = fake.host.liveActivity(id).schedules[0]!.firing!;
        expect(xaiLines(fake, 'subagent_spawned').map((l) => l.child_session_id)).toContain(run);
        await until(() => fake.host.liveActivity(id).schedules[0]?.firing === null, 5_000);
        expect(xaiLines(fake, 'subagent_finished').map((l) => l.child_session_id)).toContain(run);
      },
      { mutate: scheduling({ fireAfterMs: 100, runMs: 400 }) },
    );
  }, 20_000);

  // Mutation: let the reducer read `scheduled_task_fired` again → the live entry moves its next time and disk does not.
  it('a fire changes no entry, live or on disk; the next time it names rides the live facts', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await scheduled(fake);
        await until(() => fake.host.liveActivity(id).schedules[0]?.nextTs !== undefined, 5_000);
        const fired = fake.host.liveActivity(id).schedules[0]!.nextTs!;
        const scheduleOf = (entries: readonly TranscriptEntry[]) => {
          for (const e of entries) if (e.kind === 'tool' && e.schedule !== undefined) return e.schedule;
          return null;
        };
        const reducer = new TranscriptReducer();
        reducer.userPrompt('go');
        for (const u of fake.events.updates) if (u.sessionId === id) reducer.apply(u.line);
        expect(fake.events.updates.some((u) => u.line.update.sessionUpdate === 'scheduled_task_fired')).toBe(true);
        const files = await storedWhen(
          fake.root,
          (f) => f.includes(id),
          (text) => text.includes('"text":"Scheduled."'),
        );
        expect(files).toHaveLength(1);
        expect(files[0]).not.toContain('scheduled_task_fired');
        const live = scheduleOf(reducer.snapshot().entries);
        expect(live).toEqual(scheduleOf(parseUpdates(files[0]!).transcript));
        expect(live?.nextTs).toEqual(expect.any(Number));
        expect(fired).toBeGreaterThan(live!.nextTs!);
      },
      { mutate: scheduling({ fireAfterMs: 100, runMs: 400 }, { persist: true }) },
    );
  }, 20_000);
});

describe('monitor events (T2)', () => {
  const dirs: string[] = [];
  afterEach(() => {
    for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  function gate(name: string): string {
    const dir = mkdtempSync(join(tmpdir(), 'mlx-monitor-gate-'));
    dirs.push(dir);
    return join(dir, name);
  }

  /** The prompt "go": a monitor that runs on, these steps, then a short answer. */
  function prompts(steps: Json[]): (scenario: Json) => void {
    return (scenario) => {
      scenario.prompts = [
        {
          match: 'go',
          steps: [
            { background: { monitor: true, taskId: 'mon-1' } },
            ...steps,
            { text: 'Watching.' },
            { turnCompleted: {} },
            { end: { stopReason: 'end_turn' } },
          ],
        },
      ];
    };
  }

  const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
  /** Lines the host passed on to the reducer that are monitor events: none may be. */
  const passedEvents = (fake: FakeHost) =>
    fake.events.updates.filter((u) => u.line.update.sessionUpdate === 'monitor_event');

  // Mutation: let the line through to onUpdate (no early return) → the reducer gets a fact the disk lacks.
  // Mutation: drop the isTaskTick take → the first output tick starts a wake that never ends.
  it('events reach the live facts (count, last 5) in one window, never the reducer; a stranger’s count nowhere', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        expect((await start(fake, id).done).outcome).toBe('completed');
        await until(() => fake.events.activity.some((e) => e.live.tasks[0]?.events === 7), 5_000);
        await sleep(300);
        const [task] = fake.host.liveActivity(id).tasks;
        expect(task).toMatchObject({ id: 'mon-1', kind: 'monitor', events: 7 });
        expect(task!.recent.map((e) => e.text)).toEqual(['e3', 'e4', 'e5', 'e6', 'e7']);
        // The stream of events made one event after the start's, not one per line.
        expect(fake.events.activity.map((e) => e.live.tasks.map((t) => t.events))).toEqual([[0], [7]]);
        expect(fake.host.liveActivity(id).reports).toEqual([]);
        expect(passedEvents(fake)).toEqual([]);
        // Each event came after the output tick of the monitor's call: the host takes it, so it starts no wake and
        // the monitor's row stays as the disk ends it.
        const ticks = fake.events.updates.filter(
          (u) => u.line.update.sessionUpdate === 'tool_call_update' && u.line.update.status === 'in_progress',
        );
        expect(ticks).toEqual([]);
        expect(fake.events.turns.some((t) => t.turn?.origin === 'wake')).toBe(false);
        expect(fake.host.engineBusy(id)).toBe(false);
        const reducer = new TranscriptReducer();
        reducer.userPrompt('go');
        for (const u of fake.events.updates) if (u.sessionId === id) reducer.apply(u.line);
        const row = reducer.snapshot().entries.find((e) => e.kind === 'tool' && e.task?.id === 'mon-1');
        expect(row).toMatchObject({ status: 'completed' });
        expect(row?.kind === 'tool' ? row.output : null).not.toContain('e7');
      },
      {
        timings: { activityEveryMs: 400 },
        mutate: prompts([
          { monitorEvents: { afterMs: 700, everyMs: 10, events: ['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7'] } },
          { monitorEvents: { task: 'stranger', afterMs: 700, events: ['x'] } },
          { monitorEvents: { sessionId: 'helper-1', afterMs: 700, events: ['y'] } },
        ]),
      },
    );
  }, 20_000);

  // Mutation: stamp nothing when only a report names the wake → the row reads "Woken by a background event" live.
  it('a wake after events: the report reaches the page at once, the row is named from its first line; disk agrees on its cause', async () => {
    const wakeEnd = gate('wake-end');
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        expect((await start(fake, id).done).outcome).toBe('completed');
        await until(() => fake.events.turns.some((t) => t.turn?.origin === 'wake'), 5_000);
        const reports = fake.host.liveActivity(id).reports;
        expect(reports).toEqual([
          { wakeTs: expect.any(Number), monitors: [{ id: 'mon-1', description: 'Watch the build log' }] },
        ]);
        // Sent at once, not at the end of the 60 s window the events opened.
        expect(fake.events.activity.at(-1)!.live.reports).toEqual(reports);
        const hostLine = fake.events.updates.find(
          (u) => u.sessionId === id && u.line.update.sessionUpdate === 'user_message_chunk',
        )!;
        expect(hostLine.line.update._meta).toEqual({ hideFromScrollback: true, mlxWakePromptId: 'notifications-' });
        expect(hostLine.line.meta).toEqual({ agentTimestampMs: reports[0]!.wakeTs });

        const entriesOf = (): TranscriptEntry[] => {
          const reducer = new TranscriptReducer();
          reducer.userPrompt('go');
          for (const u of fake.events.updates) if (u.sessionId === id) reducer.apply(u.line);
          return reducer.snapshot().entries;
        };
        const wakeOf = (entries: readonly TranscriptEntry[]) => {
          const wake = entries.find((e) => e.kind === 'wake');
          if (wake?.kind !== 'wake') throw new Error('no wake row');
          return wake;
        };
        const running = entriesOf();
        expect(wakeOf(running)).toMatchObject({ cause: 'notifications', ts: reports[0]!.wakeTs });
        const named = wakeReportOf(wakeOf(running), wakeReportsOf(fake.host.liveActivity(id), running));
        expect(named).toEqual([{ id: 'mon-1', kind: 'monitor', label: 'Watch the build log' }]);
        expect(wakeTitle(wakeOf(running), named)).toBe(
          fill(WAKE_ENDED.named, { what: WAKE_REPORTED.one, labels: 'Watch the build log' }),
        );

        writeFileSync(wakeEnd, '');
        await until(() => fake.events.ends.length === 2, 5_000);
        const ended = wakeOf(entriesOf());
        expect(ended).toMatchObject({ cause: 'notifications', ts: reports[0]!.wakeTs });
        expect(ended.tasks).toBeUndefined();
        const stored = () =>
          readdirSync(fake.root, { recursive: true, encoding: 'utf8' })
            .filter((f) => f.endsWith('updates.jsonl') && f.includes(id))
            .map((f) => readFileSync(join(fake.root, f), 'utf8'));
        // The agent stores a line after it sends it: the wake's end can reach the host first.
        await until(() => stored()[0]?.includes('"prompt_id":"notifications-') === true, 5_000);
        const files = stored();
        expect(files).toHaveLength(1);
        expect(files[0]).not.toContain('monitor_event');
        const disk = wakeOf(parseUpdates(files[0]!).transcript);
        expect([disk.cause, disk.tasks]).toEqual([ended.cause, ended.tasks]);
        // The page's stored copy: no report, no monitor named.
        expect(wakeTitle(disk)).toBe('Woken by background notifications');
        expect(passedEvents(fake)).toEqual([]);
      },
      {
        timings: { activityEveryMs: 60_000 },
        mutate: prompts([
          { monitorEvents: { afterMs: 300, events: ['error TS2345'], wake: true, wakeEndOn: wakeEnd } },
        ]),
      },
    );
  }, 20_000);

  // Mutation: drop checkPrinted from the tick's take → a key a monitor prints raises nothing.
  it('a provider key in a taken tick still raises key-printed', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        expect((await start(fake, id).done).outcome).toBe('completed');
        await until(() => fake.events.notices.length > 0, 5_000);
        expect(fake.events.notices).toEqual([
          { sessionId: id, notice: { code: 'key-printed', tone: 'danger', provider: 'exa' } },
        ]);
      },
      {
        providers: noProviders({
          workKeys: () => ({ EXA_API_KEY: KEY }),
          secretNeedles: () => [KEY],
          keyNeedles: () => [{ provider: 'exa', key: KEY }],
        }),
        mutate: prompts([{ monitorEvents: { afterMs: 300, events: [`EXA_API_KEY=${KEY}`] } }]),
      },
    );
  }, 20_000);
});

describe('the goal loop (#17)', () => {
  const END = [{ text: 'Done.' }, { turnCompleted: {} }, { end: { stopReason: 'end_turn' } }];
  const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));
  type Goal = Extract<TranscriptEntry, { kind: 'goal' }>;

  function goalPrompts(goal: Json, steps: Json[] = END): (scenario: Json) => void {
    return (scenario) => {
      scenario.prompts = [{ match: '/goal', goal, steps }];
    };
  }

  function live(fake: FakeHost, id: string, typed: string): TranscriptEntry[] {
    const reducer = new TranscriptReducer();
    reducer.userPrompt(typed);
    for (const u of fake.events.updates) if (u.sessionId === id) reducer.apply(u.line);
    return reducer.snapshot().entries;
  }

  /** The stored transcript once it holds `ends` turn ends. */
  async function disk(fake: FakeHost, id: string, ends: number): Promise<TranscriptEntry[]> {
    const files = await storedWhen(fake.root, (f) => f.includes(id), ended(ends));
    expect(files).toHaveLength(1);
    return parseUpdates(files[0]!).transcript;
  }

  const goals = (entries: readonly TranscriptEntry[]): Goal[] => entries.filter((e): e is Goal => e.kind === 'goal');

  // Mutation: drop the host's `goal_updated` read → `goalRuns` stays false and the queue drains into the gap.
  it('rounds the agent starts after the turn arrive as wakes with cause goal; one goal row, the same on disk', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        const typed = '/goal fix the cache';
        expect((await start(fake, id, typed).done).outcome).toBe('completed');
        // The gap before the first round: no answer runs, the goal does.
        expect(fake.host.isBusy(id)).toBe(false);
        expect(fake.host.goalRuns(id)).toBe(true);
        await until(() => fake.events.ends.length === 3, 5_000);
        expect(fake.host.goalRuns(id)).toBe(false);
        const now = live(fake, id, typed);
        expect(now.filter((e) => e.kind === 'wake').map((e) => (e.kind === 'wake' ? e.cause : null))).toEqual([
          'goal',
          'goal',
        ]);
        expect(goals(now)).toEqual([
          expect.objectContaining({ objective: 'fix the cache', status: 'complete', tokensUsed: 3_600 }),
        ]);
        // The row sits where the goal began: after its message, before its first answer.
        const row = now.findIndex((e) => e.kind === 'goal');
        expect(now[row - 1]).toMatchObject({ kind: 'user', text: typed });
        const stored = await disk(fake, id, 3);
        // The agent stores its goal rules as the prompt; the row reads as what was typed.
        expect(stored.find((e) => e.kind === 'user')).toMatchObject({ text: typed });
        expect(goals(stored)).toEqual(goals(now));
        expect(stored.map((e) => e.kind)).toEqual(now.map((e) => e.kind));
      },
      {
        mutate: goalPrompts({ wakeRounds: 2, wakeRoundMs: 30, gapMs: 200, end: 'complete' }, [
          { goalRound: { ms: 30, ticks: 3, tokens: 1_200 } },
          ...END,
        ]),
      },
    );
  }, 20_000);

  it('Send now reaches the round inside the typed turn and a round the agent started', async () => {
    await withWork(
      'work-workflow-run',
      async (fake) => {
        const id = await session(fake);
        const typed = '/goal fix the cache';
        const turn = start(fake, id, typed);
        await until(() => fake.events.updates.some((u) => u.line.update.sessionUpdate === 'goal_updated'), 5_000);
        expect(await fake.host.steer(id, 'use pnpm')).toBe('sent');
        expect((await turn.done).outcome).toBe('completed');
        await until(() => fake.events.turns.some((t) => t.turn?.origin === 'wake'), 5_000);
        expect(await fake.host.steer(id, 'and yarn')).toBe('sent');
        await until(() => fake.events.ends.length === 2, 5_000);
        await sleep(50);
        const read = fake.records().filter((r) => r.kind === 'steer-read');
        expect(read.map((r) => r.text)).toEqual(['use pnpm', 'and yarn']);
        const stored = await disk(fake, id, 2);
        const said = stored.flatMap((e) => (e.kind === 'user' ? [e.text] : []));
        expect(said).toEqual([typed, 'use pnpm', 'and yarn']);
        expect(goals(stored)).toEqual(goals(live(fake, id, typed)));
        expect(goals(stored)).toHaveLength(1);
      },
      {
        mutate: goalPrompts({ wakeRounds: 1, wakeRoundMs: 300, gapMs: 50 }, [
          { goalRound: { ms: 300, ticks: 2 } },
          { readSteers: true },
          ...END,
        ]),
      },
    );
  }, 20_000);
});

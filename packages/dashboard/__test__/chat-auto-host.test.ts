/**
 * Auto in the Work host, on the scripted fake agent: the calls the app answers `allow-once` itself (each after its
 * history line), the cards that still open with their reason (they keep every offered option, and only Deny and Allow
 * once can be answered while the mode is Auto), the mode read after the disk wait, the cap per answer, a closed
 * window, a history line that was not written, and a helper that can work in another folder. The agent is told
 * nothing.
 */

import { spawn } from 'node:child_process';
import { linkSync, mkdirSync, realpathSync, symlinkSync, writeFileSync } from 'node:fs';
import fsPromises from 'node:fs/promises';
import { syncBuiltinESMExports } from 'node:module';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vite-plus/test';

import { AUTO_PER_TURN_MAX } from '../src/chat/agent/auto.js';
import type { PermissionInteraction, PermissionMark, PermissionMode } from '../src/chat/types.js';
import { fakeHost, noProviders, until, type FakeHost, type FakeHostOptions, type Json } from './helpers/chat-fake.js';

const MODEL = 'qwen3.5-4b';
const EDIT_OPTIONS = [
  { optionId: 'allow-edits-session', name: 'Yes, allow all edits this session', kind: 'allow_always' },
  { optionId: 'allow-once', name: 'Yes', kind: 'allow_once' },
  { optionId: 'reject-once', name: 'No', kind: 'reject_once' },
];
const END = [{ text: 'Done.', chunk: 5 }, { end: { stopReason: 'end_turn' } }];

type Allowed = Extract<PermissionMark, { kind: 'allowed' }>;

interface Switch {
  mode: PermissionMode;
  watched: boolean;
  /** What the history line answers: false is a line that could not be written. */
  write: boolean;
  marks: Allowed[];
  /** Per mark: how many frames the host had written to the agent when its line was done. */
  loggedAt: number[];
  frames: string[];
}

function bashStep(command: string, over: Json = {}): Json {
  return {
    permission: {
      toolName: 'run_terminal_command',
      kind: 'execute',
      title: `Execute \`${command}\``,
      rawInput: { variant: 'Bash', command, description: 'run', is_background: false },
      output: 'ok',
      ...over,
    },
  };
}

function editStep(path: string): Json {
  return {
    permission: {
      toolName: 'write',
      kind: 'edit',
      title: `Write ${path}`,
      rawInput: { variant: 'Write', file_path: path, content: 'x\n' },
      options: EDIT_OPTIONS,
      output: 'wrote',
    },
  };
}

function toolStep(toolName: string, rawInput: Json, over: Json = {}): Json {
  return { permission: { toolName, kind: 'other', title: toolName, rawInput, output: 'done', ...over } };
}

function script(steps: Json[]): (scenario: Json) => void {
  return (scenario) => {
    scenario.prompts = [{ match: '', steps: [...steps, ...END] }];
  };
}

async function withAuto(
  steps: Json[],
  run: (fake: FakeHost, sw: Switch, id: string) => Promise<void>,
  options: FakeHostOptions & { prepare?: (project: string) => void; name?: string } = {},
): Promise<void> {
  const sw: Switch = { mode: 'auto', watched: true, write: true, marks: [], loggedAt: [], frames: [] };
  const fake = await fakeHost(options.name ?? 'work-permission', {
    mode: 'work',
    providers: noProviders(),
    mutate: options.name === undefined ? script(steps) : options.mutate,
    permission: { modeOf: () => sw.mode, watched: () => sw.watched },
    onModeAllowed: async (_id, mark) => {
      await new Promise((resolve) => setImmediate(resolve));
      if (!sw.write) {
        sw.mode = 'ask';
        return false;
      }
      sw.marks.push(mark);
      sw.loggedAt.push(sw.frames.length);
      return true;
    },
    spawn: (command) => {
      const child = spawn(command.command, [...command.args], {
        env: command.env,
        cwd: command.cwd,
        stdio: ['pipe', 'pipe', 'pipe'],
      });
      const write = child.stdin.write.bind(child.stdin) as (chunk: unknown, ...rest: unknown[]) => boolean;
      child.stdin.write = ((chunk: unknown, ...rest: unknown[]) => {
        sw.frames.push(String(chunk));
        return write(chunk, ...rest);
      }) as typeof child.stdin.write;
      return child;
    },
    ...options,
  });
  try {
    options.prepare?.(fake.project);
    const id = (await fake.host.createSession({ modelId: MODEL, cwd: fake.project })).sessionId;
    await run(fake, sw, id);
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

const ALLOW_ONCE = { outcome: { outcome: 'selected', optionId: 'allow-once' } };

describe('Auto', () => {
  it('ls, npm test and an edit inside run with no card: allow-once each, after its history line, with the card text', async () => {
    await withAuto([bashStep('ls'), bashStep('npm test'), editStep('notes.md')], async (fake, sw, id) => {
      expect((await start(fake, id).done).outcome).toBe('completed');
      expect(fake.events.interactions).toEqual([]);
      expect(answers(fake)).toEqual([ALLOW_ONCE, ALLOW_ONCE, ALLOW_ONCE]);
      const asks = fake.records().filter((r) => r.kind === 'ask');
      const execs = fake.records().filter((r) => r.kind === 'exec');
      expect(execs.map((r) => r.via)).toEqual(asks.map((r) => r.id));
      expect(sw.marks.map((m) => [m.access, m.card?.command ?? m.card?.path, m.card?.options])).toEqual([
        ['bash', 'ls', []],
        ['bash', 'npm test', []],
        ['edit', 'notes.md', []],
      ]);
      expect(sw.marks.every((m) => m.card?.autoAsk === undefined)).toBe(true);
      expect(sw.marks.map((m) => m.toolCallId)).toEqual(asks.map((r) => r.toolCallId));
      // Each reply frame is written after the history line of its call.
      asks.forEach((ask, k) => {
        const reply = sw.frames.findIndex((f) => f.includes(`"id":${String(ask.id)},`) && f.includes('"result"'));
        expect([k, reply >= sw.loggedAt[k]!]).toEqual([k, true]);
      });
    });
  });

  it.each<[string, Json[], string, ((project: string) => void) | undefined]>([
    ['rm', [bashStep('rm a.txt')], 'risky', undefined],
    ['an edit outside', [editStep('/etc/hosts')], 'risky', undefined],
    [
      'a write through a link that leads outside',
      [bashStep('echo x >> k/a')],
      'unknown-target',
      (project) => {
        const outside = join(project, '..', 'outside');
        mkdirSync(outside, { recursive: true });
        symlinkSync(outside, join(project, 'k'));
      },
    ],
    [
      'an edit of a file with two hard links',
      [editStep('h')],
      'unknown-landing',
      (project) => {
        writeFileSync(join(project, '..', 'f'), 'f');
        linkSync(join(project, '..', 'f'), join(project, 'h'));
      },
    ],
    ['an edit of .git/hooks/pre-commit', [editStep('.git/hooks/pre-commit')], 'protected', undefined],
    ['monitor', [toolStep('monitor', { command: 'ls' }, { hookAsk: 'guard' })], 'not-covered', undefined],
    ['an MCP tool', [toolStep('linear__save_issue', { title: 'x' })], 'not-covered', undefined],
    ['spawn_subagent', [toolStep('spawn_subagent', { prompt: 'look' })], 'not-covered', undefined],
    [
      'spawn_subagent with a cwd',
      [toolStep('spawn_subagent', { prompt: 'look', cwd: '/' })],
      'helper-start',
      undefined,
    ],
    [
      'the guard name of an older start',
      [bashStep('ls', { hookAsk: 'plugin/mlx-work-guard-0123456789abcdef/plugin:pre_tool_use[0].hooks[0]' })],
      'hook',
      undefined,
    ],
    ['a shell request with no guard', [bashStep('ls', { hookAsk: null })], 'hook', undefined],
    [
      'a hook of the user with a reason',
      [bashStep('ls', { hookAsk: 'guard', hookReason: 'policy' })],
      'hook',
      undefined,
    ],
    [
      'a background start',
      [
        bashStep('sleep 5', {
          rawInput: { variant: 'Bash', command: 'sleep 5', description: 'run', is_background: true },
        }),
      ],
      'background',
      undefined,
    ],
    ['npx', [bashStep('npx cowsay')], 'risky', undefined],
    ['a command not on the list', [bashStep('sed -i s/a/b/ a.txt')], 'not-known', undefined],
  ])(
    '%s: a card with its reason; it keeps every offered option, and only Deny and Allow once answer it',
    async (_name, steps, why, prepare) => {
      await withAuto(
        steps,
        async (fake, sw, id) => {
          const turn = start(fake, id);
          const card = await nth(fake, 1);
          expect(card.autoAsk).toBe(why);
          const always = card.access === 'edit' ? 'allow-edits-session' : 'always-allow';
          expect(card.options.map((o) => o.optionId)).toEqual([always, 'allow-once', 'reject-once']);
          expect(() => fake.host.answer(id, card.id, always)).toThrow('That option was not offered.');
          expect(sw.marks).toEqual([]);
          fake.host.answer(id, card.id, 'reject-once');
          expect((await turn.done).outcome).toBe('denied');
        },
        prepare === undefined ? {} : { prepare },
      );
    },
  );

  it('a plan is the plan card, unchanged', async () => {
    await withAuto(
      [],
      async (fake, _sw, id) => {
        const turn = start(fake, id);
        await until(() => fake.events.interactions.length === 1, 5_000);
        const card = fake.events.interactions[0]!.interaction;
        expect(card.kind).toBe('plan');
        expect('autoAsk' in card).toBe(false);
        fake.host.answerPlan(id, card.id, 'approved');
        expect((await turn.done).outcome).toBe('completed');
      },
      { name: 'work-plan' },
    );
  });

  it('after the user allows a helper with its own folder, every call of the session asks', async () => {
    await withAuto(
      [toolStep('spawn_subagent', { prompt: 'look', cwd: '/' }), bashStep('ls'), editStep('notes.md')],
      async (fake, sw, id) => {
        const turn = start(fake, id);
        const spawnCard = await nth(fake, 1);
        expect(spawnCard.autoAsk).toBe('helper-start');
        fake.host.answer(id, spawnCard.id, 'allow-once');
        const shell = await nth(fake, 2);
        expect(shell.autoAsk).toBe('helper');
        fake.host.answer(id, shell.id, 'allow-once');
        expect((await nth(fake, 3)).autoAsk).toBe('helper');
        fake.host.answer(id, fake.events.interactions[2]!.interaction.id, 'allow-once');
        expect((await turn.done).outcome).toBe('completed');
        expect(sw.marks).toEqual([]);
      },
    );
  });

  it(`the call after ${AUTO_PER_TURN_MAX} asks (many); an allow by the user starts the count again`, async () => {
    const steps = Array.from({ length: AUTO_PER_TURN_MAX + 3 }, () => bashStep('true'));
    await withAuto(steps, async (fake, sw, id) => {
      const turn = start(fake, id);
      const card = await nth(fake, 1);
      expect(card.autoAsk).toBe('many');
      expect(sw.marks).toHaveLength(AUTO_PER_TURN_MAX);
      fake.host.answer(id, card.id, 'allow-once');
      expect((await turn.done).outcome).toBe('completed');
      expect(fake.events.interactions).toHaveLength(1);
      expect(sw.marks).toHaveLength(AUTO_PER_TURN_MAX + 2);
    });
  }, 30_000);

  it('with no window holding the host stream, each call is a card (no-window)', async () => {
    await withAuto([bashStep('ls')], async (fake, sw, id) => {
      sw.watched = false;
      const turn = start(fake, id);
      const card = await nth(fake, 1);
      expect(card.autoAsk).toBe('no-window');
      fake.host.answer(id, card.id, 'reject-once');
      expect((await turn.done).outcome).toBe('denied');
    });
  });

  it('a history line that was not written: the mode is Ask and the plain card opens', async () => {
    await withAuto([bashStep('ls')], async (fake, sw, id) => {
      sw.write = false;
      const turn = start(fake, id);
      const card = await nth(fake, 1);
      expect(card.autoAsk).toBeUndefined();
      expect(card.options.map((o) => o.optionId)).toEqual(['always-allow', 'allow-once', 'reject-once']);
      expect(sw.mode).toBe('ask');
      fake.host.answer(id, card.id, 'allow-once');
      expect((await turn.done).outcome).toBe('completed');
      expect(answers(fake)).toEqual([ALLOW_ONCE]);
    });
  });

  it('Ask: the plain card with every offered option and no reason', async () => {
    await withAuto([bashStep('ls')], async (fake, sw, id) => {
      sw.mode = 'ask';
      const turn = start(fake, id);
      const card = await nth(fake, 1);
      expect(card.autoAsk).toBeUndefined();
      expect(card.options.map((o) => o.optionId)).toEqual(['always-allow', 'allow-once', 'reject-once']);
      fake.host.answer(id, card.id, 'reject-once');
      await turn.done;
    });
  });

  it('a session that passed its guard is reported ready', async () => {
    await withAuto([], async (fake, _sw, id) => {
      expect(fake.events.ready).toEqual([id]);
    });
  });

  it('requests that arrive together cannot pass the cap: each call is counted at its decision', async () => {
    const before = AUTO_PER_TURN_MAX - 2;
    const steps: Json[] = [
      ...Array.from({ length: before }, () => bashStep('true')),
      { parallel: ['true', 'true', 'true', 'true', 'true'] },
    ];
    let release = (): void => {};
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    let held = 0;
    let written = 0;
    await withAuto(
      steps,
      async (fake, _sw, id) => {
        const turn = start(fake, id);
        await until(() => held + fake.events.interactions.length >= 5, 20_000);
        release();
        await until(() => fake.records().filter((r) => r.kind === 'parallel').length === 2, 10_000);
        const cards = fake.events.interactions.map((e) => e.interaction as PermissionInteraction);
        expect([written, cards.map((c) => c.autoAsk)]).toEqual([AUTO_PER_TURN_MAX, ['many', 'many', 'many']]);
        for (const card of cards) fake.host.answer(id, card.id, 'reject-once');
        await turn.done;
        const parallel = fake.records().filter((r) => r.kind === 'parallel');
        expect(parallel.filter((r) => r.answer === 'allow-once')).toHaveLength(2);
      },
      {
        onModeAllowed: async (_id, mark) => {
          if (mark.toolCallId.includes('_p')) {
            held += 1;
            await gate;
          }
          written += 1;
          return true;
        },
      },
    );
  }, 60_000);

  it('moved to Ask while the history line is written: the plain card, no mark, and the line is withdrawn', async () => {
    const marked: Allowed[] = [];
    const withdrawn: Allowed[] = [];
    let sw: Switch | null = null;
    await withAuto(
      [bashStep('ls')],
      async (fake, current, id) => {
        sw = current;
        const turn = start(fake, id);
        const card = await nth(fake, 1);
        expect(card.autoAsk).toBeUndefined();
        expect(card.options.map((o) => o.optionId)).toEqual(['always-allow', 'allow-once', 'reject-once']);
        expect(answers(fake)).toEqual([]);
        expect(marked).toEqual([]);
        expect(withdrawn.map((m) => [m.mode, m.card?.command])).toEqual([['auto', 'ls']]);
        fake.host.answer(id, card.id, 'reject-once');
        expect((await turn.done).outcome).toBe('denied');
      },
      {
        onModeAllowed: async () => {
          await new Promise((resolve) => setImmediate(resolve));
          if (sw !== null) sw.mode = 'ask';
          return true;
        },
        onModeMarked: (_id, mark) => marked.push(mark),
        onModeWithdrawn: (_id, mark) => withdrawn.push(mark),
      },
    );
  });

  it('the mark of an allowed call is given in the tick of its reply, after its line', async () => {
    const order: string[] = [];
    await withAuto(
      [bashStep('ls')],
      async (fake, _sw, id) => {
        expect((await start(fake, id).done).outcome).toBe('completed');
        expect(order).toEqual(['line', 'mark']);
      },
      {
        onModeAllowed: async () => {
          order.push('line');
          return true;
        },
        onModeMarked: () => order.push('mark'),
      },
    );
  });

  it('each conversation of one agent process uses its own mode and its own count', async () => {
    const on = new Set<string>();
    const seen: string[] = [];
    await withAuto(
      [bashStep('ls')],
      async (fake, _sw, a) => {
        const b = (await fake.host.createSession({ modelId: MODEL, cwd: fake.project })).sessionId;
        on.add(a);
        const off = start(fake, b);
        const card = await nth(fake, 1);
        expect([fake.events.interactions[0]!.sessionId, card.autoAsk]).toEqual([b, undefined]);
        fake.host.answer(b, card.id, 'reject-once');
        expect((await off.done).outcome).toBe('denied');
        expect((await start(fake, a).done).outcome).toBe('completed');
        expect(fake.events.interactions).toHaveLength(1);
        expect(seen).toEqual([a]);
      },
      {
        permission: { modeOf: (sessionId: string) => (on.has(sessionId) ? 'auto' : 'ask'), watched: () => true },
        onModeAllowed: async (sessionId) => {
          seen.push(sessionId);
          return true;
        },
      },
    );
  });

  it('a search_replace whose file is missing while a look-alike name is there: the landing is unknown', async () => {
    const lookAlike = (project: string): void => {
      const outside = join(project, '..', 'victim.txt');
      writeFileSync(outside, 'v');
      symlinkSync(outside, join(project, 'n b.txt'));
    };
    const replace = {
      permission: {
        toolName: 'search_replace',
        kind: 'edit',
        title: 'Edit n b.txt',
        rawInput: { variant: 'SearchReplace', file_path: 'n b.txt', old_string: 'v', new_string: 'w' },
        options: EDIT_OPTIONS,
        output: 'edited',
      },
    };
    for (const auto of [true, false]) {
      await withAuto(
        [replace],
        async (fake, sw, id) => {
          sw.mode = auto ? 'auto' : 'ask';
          const turn = start(fake, id);
          const card = await nth(fake, 1);
          expect([card.landingUnknown, card.resolvedPath, card.autoAsk]).toEqual([
            true,
            null,
            auto ? 'unknown-landing' : undefined,
          ]);
          expect(sw.marks).toEqual([]);
          fake.host.answer(id, card.id, 'reject-once');
          expect((await turn.done).outcome).toBe('denied');
        },
        { prepare: lookAlike },
      );
    }
  });

  describe('calls that run at the same time', () => {
    const folder = (project: string): void => {
      mkdirSync(join(project, 'd'), { recursive: true });
      writeFileSync(join(project, 'a.txt'), 'a');
    };

    it('a mv and a write into its new name, asked together: one runs, the other is a card (unknown-target)', async () => {
      await withAuto(
        [
          {
            parallel: [
              { command: 'mv d e', complete: false },
              { command: 'echo x > e/f', complete: false },
            ],
          },
        ],
        async (fake, sw, id) => {
          const turn = start(fake, id);
          const card = await nth(fake, 1);
          expect([card.autoAsk, sw.marks.length]).toEqual(['unknown-target', 1]);
          fake.host.answer(id, card.id, 'reject-once');
          await turn.done;
        },
        { prepare: folder },
      );
    });

    it('two reads asked together both run with no card', async () => {
      await withAuto(
        [
          {
            parallel: [
              { command: 'cat a.txt', complete: false },
              { command: 'ls d', complete: false },
            ],
          },
        ],
        async (fake, sw, id) => {
          expect((await start(fake, id).done).outcome).toBe('completed');
          expect([fake.events.interactions.length, sw.marks.length]).toEqual([0, 2]);
        },
        { prepare: folder },
      );
    });

    it('a call the user allowed that has not ended: the next call is a card (unknown-target)', async () => {
      await withAuto(
        [{ parallel: [{ command: 'sed -i s/a/b/ a.txt', complete: false }] }, bashStep('ls')],
        async (fake, sw, id) => {
          const turn = start(fake, id);
          const first = await nth(fake, 1);
          expect(first.autoAsk).toBe('not-known');
          fake.host.answer(id, first.id, 'allow-once');
          const second = await nth(fake, 2);
          expect([second.command, second.autoAsk, sw.marks.length]).toEqual(['ls', 'unknown-target', 0]);
          fake.host.answer(id, second.id, 'reject-once');
          await turn.done;
        },
        { prepare: folder },
      );
    });

    it('a mv that has ended: the next mv runs with no card', async () => {
      await withAuto(
        [{ parallel: ['mv d e'] }, bashStep('mv e f')],
        async (fake, sw, id) => {
          expect((await start(fake, id).done).outcome).toBe('completed');
          expect([fake.events.interactions.length, sw.marks.map((m) => m.card?.command)]).toEqual([
            0,
            ['mv d e', 'mv e f'],
          ]);
        },
        { prepare: folder },
      );
    });
  });

  describe('the mode is read after the disk wait', () => {
    const original = fsPromises.realpath;
    let release = (): void => {};

    afterEach(() => {
      release();
      fsPromises.realpath = original;
      syncBuiltinESMExports();
    });

    it('moved to Ask while a write target is placed: the call is the plain card', async () => {
      await withAuto([bashStep('touch a.txt')], async (fake, sw, id) => {
        const target = join(realpathSync(fake.project), 'a.txt');
        const gate = new Promise<void>((resolve) => {
          release = resolve;
        });
        let held = false;
        fsPromises.realpath = async function (this: unknown, path: unknown, ...rest: unknown[]) {
          if (String(path) === target && !held) {
            held = true;
            await gate;
          }
          return (original as (...args: unknown[]) => Promise<string>).call(this, path, ...rest);
        } as typeof fsPromises.realpath;
        syncBuiltinESMExports();
        const turn = start(fake, id);
        await until(() => held, 5_000);
        sw.mode = 'ask';
        release();
        const card = await nth(fake, 1);
        expect(card.autoAsk).toBeUndefined();
        expect(sw.marks).toEqual([]);
        fake.host.answer(id, card.id, 'reject-once');
        expect((await turn.done).outcome).toBe('denied');
      });
    });
  });
});

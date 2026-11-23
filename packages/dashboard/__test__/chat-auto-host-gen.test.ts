/**
 * Auto and the calls that run in the Work folder, on the scripted fake agent: facts read before a call that may
 * change the folder started or ended are stale, whichever conversation or turn the call belongs to, and a call stays
 * in its folder until it reports an end. The fake runs nothing: the test makes each allowed call's effect itself (a
 * rename or a link inside the fake's throwaway folder).
 */

import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, realpathSync, renameSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import fsPromises from 'node:fs/promises';
import { syncBuiltinESMExports } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vite-plus/test';

import type { PermissionMark, PermissionInteraction } from '../src/chat/types.js';
import {
  fakeHost,
  INFERENCE_TOKEN,
  noProviders,
  until,
  type FakeHost,
  type FakeHostOptions,
  type Json,
} from './helpers/chat-fake.js';

const MODEL = 'qwen3.5-4b';
const END = [{ text: 'Done.', chunk: 5 }, { end: { stopReason: 'end_turn' } }];
type Allowed = Extract<PermissionMark, { kind: 'allowed' }>;

interface Switch {
  auto: boolean;
  marks: Allowed[];
  hold: ((mark: Allowed) => Promise<void>) | null;
}

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

async function withHost(
  prompts: Json[],
  run: (fake: FakeHost, sw: Switch) => Promise<void>,
  prepare: (project: string, root: string) => void,
  extra: Partial<FakeHostOptions> = {},
): Promise<void> {
  const sw: Switch = { auto: true, marks: [], hold: null };
  const fake = await fakeHost('work-permission', {
    ...extra,
    mode: 'work',
    providers: noProviders(),
    mutate: (scenario: Json) => {
      scenario.prompts = prompts;
    },
    permission: { modeOf: () => (sw.auto ? 'auto' : 'ask'), watched: () => true },
    onModeAllowed: async (_id, mark) => {
      await new Promise((resolve) => setImmediate(resolve));
      if (sw.hold !== null) await sw.hold(mark);
      sw.marks.push(mark);
      return true;
    },
    spawn: (command) =>
      spawn(command.command, [...command.args], {
        env: command.env,
        cwd: command.cwd,
        stdio: ['pipe', 'pipe', 'pipe'],
      }),
  });
  try {
    prepare(fake.project, fake.root);
    await run(fake, sw);
  } finally {
    await fake.close();
  }
}

const completed = (fake: FakeHost, suffix: string): boolean =>
  fake.events.updates.some(
    (u) =>
      (u.line.update as Json).status === 'completed' && String((u.line.update as Json).toolCallId).endsWith(suffix),
  );

async function nth(fake: FakeHost, n: number): Promise<PermissionInteraction> {
  await until(() => fake.events.interactions.length >= n, 5_000);
  return fake.events.interactions[n - 1]!.interaction as PermissionInteraction;
}

const linkOut = (project: string, root: string): void => {
  mkdirSync(join(root, 'out'), { recursive: true });
  mkdirSync(join(project, 'd'), { recursive: true });
  writeFileSync(join(project, 'a.txt'), 'a');
  symlinkSync(join(root, 'out'), join(project, 'd', 'k'));
};

const outFolder = (project: string, root: string): void => {
  mkdirSync(join(root, 'out'), { recursive: true });
  writeFileSync(join(project, 'a.txt'), 'a');
};

describe('calls that change the Work folder', () => {
  const original = fsPromises.lstat;
  let release = (): void => {};
  afterEach(() => {
    release();
    fsPromises.lstat = original;
    syncBuiltinESMExports();
  });

  it('a move that is allowed and ends while another call is placed makes that call ask', async () => {
    await withHost(
      [{ match: '', steps: [{ parallel: ['mv d e', { command: 'echo x > e/k/f', complete: false }] }, ...END] }],
      async (fake, sw) => {
        const id = (await fake.host.createSession({ modelId: MODEL, cwd: fake.project })).sessionId;
        const gate = new Promise<void>((r) => {
          release = r;
        });
        let visits = 0;
        let held = false;
        fsPromises.lstat = async function (this: unknown, path: unknown, ...rest: unknown[]) {
          if (String(path).endsWith('/e/k/f') && ++visits === 2) {
            held = true;
            await gate;
          }
          return (original as (...a: unknown[]) => Promise<unknown>).call(this, path, ...rest);
        } as typeof fsPromises.lstat;
        syncBuiltinESMExports();
        const turn = fake.host.startTurn(id, 'go', { modelId: MODEL, cwd: fake.project });
        await until(() => held && completed(fake, '_p0'), 5_000);
        renameSync(join(fake.project, 'd'), join(fake.project, 'e'));
        release();
        const card = await nth(fake, 1);
        expect([card.command, card.autoAsk]).toEqual(['echo x > e/k/f', 'unknown-target']);
        expect(sw.marks.map((m) => m.card?.command)).toEqual(['mv d e']);
        fake.host.answer(id, card.id, 'reject-once');
        await turn.done;
      },
      linkOut,
    );
  });

  it('a call the user allows while another waits on its history line makes that one ask', async () => {
    let enter = (): void => {};
    const entered = new Promise<void>((r) => {
      enter = r;
    });
    let releaseLine = (): void => {};
    const line = new Promise<void>((r) => {
      releaseLine = r;
    });
    await withHost(
      [{ match: '', steps: [{ parallel: ['ln -s ../out e', { command: 'echo x > e/f', complete: false }] }, ...END] }],
      async (fake, sw) => {
        sw.hold = async (mark) => {
          if (mark.card?.command === 'echo x > e/f') {
            enter();
            await line;
          }
        };
        const id = (await fake.host.createSession({ modelId: MODEL, cwd: fake.project })).sessionId;
        const turn = fake.host.startTurn(id, 'go', { modelId: MODEL, cwd: fake.project });
        const card = await nth(fake, 1);
        await entered;
        fake.host.answer(id, card.id, 'allow-once');
        await until(() => completed(fake, '_p0'), 5_000);
        symlinkSync(join(fake.root, 'out'), join(fake.project, 'e'));
        releaseLine();
        const second = await nth(fake, 2);
        expect([second.command, second.autoAsk]).toEqual(['echo x > e/f', 'unknown-target']);
        fake.host.answer(id, second.id, 'reject-once');
        await turn.done;
        const answers = fake.records().filter((r) => r.kind === 'parallel');
        expect(answers.find((r) => String(r.toolCallId).endsWith('_p1'))?.answer).toBeNull();
      },
      outFolder,
    );
  });

  it("another conversation's running call on the same folder makes a call ask", async () => {
    await withHost(
      [
        {
          match: 'x',
          steps: [{ parallel: [{ command: 'ln -s ../out e', complete: false }] }, bashStep('rm a.txt'), ...END],
        },
        { match: 'y', steps: [bashStep('echo x > e/f'), ...END] },
      ],
      async (fake) => {
        const x = (await fake.host.createSession({ modelId: MODEL, cwd: fake.project })).sessionId;
        const y = (await fake.host.createSession({ modelId: MODEL, cwd: fake.project })).sessionId;
        const tx = fake.host.startTurn(x, 'x', { modelId: MODEL, cwd: fake.project });
        const first = await nth(fake, 1);
        fake.host.answer(x, first.id, 'allow-once');
        const second = await nth(fake, 2);
        const ty = fake.host.startTurn(y, 'y', { modelId: MODEL, cwd: fake.project });
        const third = await nth(fake, 3);
        expect([third.command, third.autoAsk]).toEqual(['echo x > e/f', 'unknown-target']);
        fake.host.answer(y, third.id, 'reject-once');
        fake.host.answer(x, second.id, 'reject-once');
        await Promise.all([tx.done, ty.done]);
      },
      outFolder,
    );
  });

  it('a call still running when the user stops the turn makes the next turn ask', async () => {
    await withHost(
      [
        {
          match: 'one',
          steps: [{ parallel: [{ command: 'ln -s ../out e', complete: false }] }, bashStep('rm a.txt'), ...END],
        },
        { match: 'two', steps: [bashStep('echo x > e/f'), ...END] },
      ],
      async (fake) => {
        const id = (await fake.host.createSession({ modelId: MODEL, cwd: fake.project })).sessionId;
        const t1 = fake.host.startTurn(id, 'one', { modelId: MODEL, cwd: fake.project });
        const first = await nth(fake, 1);
        fake.host.answer(id, first.id, 'allow-once');
        await nth(fake, 2);
        fake.host.cancel(id);
        await t1.done;
        const t2 = fake.host.startTurn(id, 'two', { modelId: MODEL, cwd: fake.project });
        const third = await nth(fake, 3);
        expect([third.command, third.autoAsk]).toEqual(['echo x > e/f', 'unknown-target']);
        fake.host.answer(id, third.id, 'reject-once');
        await t2.done;
      },
      outFolder,
    );
  });

  it('a running call reaches a Work folder it holds or that holds it; one the user allowed reaches every folder', async () => {
    const cases = [
      { first: 'mv a.txt b.txt', user: false, folder: 'project/sub', asks: true },
      { first: 'mv a.txt b.txt', user: false, folder: 'other', asks: false },
      { first: 'ln -s ../out e', user: true, folder: 'other', asks: true },
    ];
    for (const c of cases) {
      await withHost(
        [
          { match: 'x', steps: [{ parallel: [{ command: c.first, complete: false }] }, bashStep('rm a.txt'), ...END] },
          { match: 'y', steps: [bashStep('echo x > n.txt'), ...END] },
        ],
        async (fake, sw) => {
          const there = join(fake.root, c.folder);
          const x = (await fake.host.createSession({ modelId: MODEL, cwd: fake.project })).sessionId;
          const y = (await fake.host.createSession({ modelId: MODEL, cwd: there })).sessionId;
          const tx = fake.host.startTurn(x, 'x', { modelId: MODEL, cwd: fake.project });
          if (c.user) fake.host.answer(x, (await nth(fake, 1)).id, 'allow-once');
          const hold = await nth(fake, c.user ? 2 : 1);
          expect(hold.command).toBe('rm a.txt');
          const ty = fake.host.startTurn(y, 'y', { modelId: MODEL, cwd: there });
          if (c.asks) {
            const card = await nth(fake, c.user ? 3 : 2);
            expect([c.folder, card.command, card.autoAsk]).toEqual([c.folder, 'echo x > n.txt', 'unknown-target']);
            fake.host.answer(y, card.id, 'reject-once');
          }
          await ty.done;
          fake.host.answer(x, hold.id, 'reject-once');
          await tx.done;
          const marked = sw.marks.map((m) => m.card?.command);
          expect([c.folder, marked.includes('echo x > n.txt')]).toEqual([c.folder, !c.asks]);
        },
        (project, root) => {
          outFolder(project, root);
          mkdirSync(join(project, 'sub'), { recursive: true });
          mkdirSync(join(root, 'other'), { recursive: true });
        },
      );
    }
  });

  it('after an "always" grant, a call the agent runs with no request makes a call ask while it runs', async () => {
    for (const grant of ['always-allow', 'allow-once'] as const) {
      await withHost(
        [
          {
            match: '',
            steps: [
              bashStep('ln -s ../out e'),
              {
                tool: {
                  id: 'toolu_quiet',
                  toolName: 'run_terminal_command',
                  rawInput: { command: 'ln -s ../out e' },
                  status: 'in_progress',
                },
              },
              bashStep('echo x > n.txt'),
              ...END,
            ],
          },
        ],
        async (fake, sw) => {
          sw.auto = false;
          const id = (await fake.host.createSession({ modelId: MODEL, cwd: fake.project })).sessionId;
          const turn = fake.host.startTurn(id, 'go', { modelId: MODEL, cwd: fake.project });
          const first = await nth(fake, 1);
          fake.host.answer(id, first.id, grant);
          sw.auto = true;
          if (grant === 'always-allow') {
            const second = await nth(fake, 2);
            expect([second.command, second.autoAsk]).toEqual(['echo x > n.txt', 'unknown-target']);
            fake.host.answer(id, second.id, 'reject-once');
          }
          await turn.done;
          expect([grant, sw.marks.map((m) => m.card?.command)]).toEqual([
            grant,
            grant === 'always-allow' ? [] : ['echo x > n.txt'],
          ]);
        },
        outFolder,
      );
    }
  });
});

describe('the temp root is one place for every conversation', () => {
  /** A throwaway temp root (never the machine's temp folder): `scan/a.txt`, and a Work folder `proj` inside it. */
  function tempRoot(): string {
    const top = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-auto-host-temp-')));
    mkdirSync(join(top, 'scan'), { mode: 0o755 });
    writeFileSync(join(top, 'scan', 'a.txt'), 'x\n');
    mkdirSync(join(top, 'proj'), { mode: 0o755 });
    writeFileSync(join(top, 'proj', 'a.txt'), 'x\n');
    return top;
  }

  const cases = [
    { name: 'a changing call under the temp root', first: 'scan/n.txt', cwd: 'project' },
    { name: 'a project tool in any Work folder', first: null, cwd: 'project' },
    { name: 'a changing call in a Work folder inside the temp root', first: 'proj/n.txt', cwd: 'temp-proj' },
  ] as const;

  for (const c of cases) {
    it(`${c.name} holds an Auto call that reaches the temp root in another conversation; one that stays in its own folder runs`, async () => {
      const top = tempRoot();
      const first =
        c.first === null ? 'npm test' : c.cwd === 'temp-proj' ? 'touch n.txt' : `echo x > ${top}/${c.first}`;
      try {
        await withHost(
          [
            { match: 'x', steps: [{ parallel: [{ command: first, complete: false }] }, bashStep('rm a.txt'), ...END] },
            { match: 'y', steps: [bashStep('cat a.txt'), bashStep(`cat ${top}/scan/a.txt`), ...END] },
          ],
          async (fake, sw) => {
            const xCwd = c.cwd === 'temp-proj' ? join(top, 'proj') : fake.project;
            const there = join(fake.root, 'other');
            const x = (await fake.host.createSession({ modelId: MODEL, cwd: xCwd })).sessionId;
            const y = (await fake.host.createSession({ modelId: MODEL, cwd: there })).sessionId;
            const tx = fake.host.startTurn(x, 'x', { modelId: MODEL, cwd: xCwd });
            const hold = await nth(fake, 1);
            expect(hold.command).toBe('rm a.txt');
            const ty = fake.host.startTurn(y, 'y', { modelId: MODEL, cwd: there });
            const card = await nth(fake, 2);
            expect([c.name, card.command, card.autoAsk]).toEqual([c.name, `cat ${top}/scan/a.txt`, 'unknown-target']);
            fake.host.answer(y, card.id, 'reject-once');
            await ty.done;
            fake.host.answer(x, hold.id, 'reject-once');
            await tx.done;
            const marked = sw.marks.map((m) => [m.card?.command, m.temp === true]);
            expect([c.name, marked]).toEqual([
              c.name,
              [
                [first, c.first !== null && c.cwd === 'project'],
                ['cat a.txt', false],
              ],
            ]);
          },
          (project, root) => {
            outFolder(project, root);
            writeFileSync(join(project, 'package.json'), JSON.stringify({ scripts: { test: 'true' } }));
            mkdirSync(join(root, 'other'), { recursive: true });
            writeFileSync(join(root, 'other', 'a.txt'), 'x\n');
          },
          { tempRoots: [{ top, spellings: [top], kind: 'temp' }] },
        );
      } finally {
        rmSync(top, { recursive: true, force: true });
      }
    });
  }
});

describe('what ps shows of the app', () => {
  it('the argument lists of the agent and of the web tool server it starts hold no card needle', async () => {
    const key = 'sk-KEY-0123456789abcdef0123';
    const args: string[][] = [];
    const fake = await fakeHost('chat-basic', {
      mode: 'chat',
      providers: noProviders({
        chatWeb: () => ({ searchProvider: 'brave', searchKey: key, fetchProvider: 'none', fetchKey: '' }),
        workKeys: () => ({ BRAVE_API_KEY: key }),
        secretNeedles: () => [key],
      }),
      spawn: (command) => {
        args.push([command.command, ...command.args]);
        return spawn(command.command, [...command.args], { env: command.env, cwd: command.cwd, stdio: 'pipe' });
      },
    });
    try {
      await fake.host.createSession({ modelId: MODEL });
      const sent = fake.sent().filter((m) => m.method === 'session/new');
      expect(sent.length).toBeGreaterThan(0);
      const servers = sent.flatMap((m) => ((m.params as Json).mcpServers as Json[]) ?? []);
      expect(servers.length).toBeGreaterThan(0);
      for (const server of servers) args.push([String(server.command), ...((server.args as string[]) ?? [])]);
      expect(args.length).toBeGreaterThan(1);
      for (const list of args) {
        for (const word of list) {
          expect([word, word.includes(key) || word.includes(INFERENCE_TOKEN)]).toEqual([word, false]);
        }
      }
    } finally {
      await fake.close();
    }
  });
});

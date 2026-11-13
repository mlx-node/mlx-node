/**
 * The permission mode in the Chat service on the fake agent: every transition and its refusals, the epoch that lets a
 * move down win over a move up sent before it, the open cards the move to Bypass settles, the events and host state,
 * the metadata-only history file (written before each reply; a failed write moves the mode to Ask; a line whose call
 * got no allow is withdrawn), the settings check, the user's hooks, delete, and a Control Panel restart that leaves
 * every conversation in Ask.
 */

import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import { ApiError } from '../src/api/errors.js';
import type { AllowedRoot } from '../src/chat/agent/roots.js';
import { MARK_CARDS_KEPT, Conversation } from '../src/chat/conversation.js';
import { PermissionLog, type PermissionLine } from '../src/chat/permission-log.js';
import { routeError } from '../src/chat/route-errors.js';
import type { ChatService } from '../src/chat/service.js';
import type {
  ChatHostEvent,
  ConversationEvent,
  Interaction,
  PermissionInteraction,
  PermissionMark,
  PermissionMode,
} from '../src/chat/types.js';
import { resolveTempRoot } from '../src/chat/work-folder.js';
import { applyConversationEvent, type ConversationState } from '../ui/src/lib/chat-store.js';
import { fakeService, INFERENCE_TOKEN, until, type FakeService, type Json } from './helpers/chat-fake.js';

const END = [{ text: 'Done.', chunk: 5 }, { end: { stopReason: 'end_turn' } }];

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

  lastOutcome(): string | undefined {
    const end = this.events.filter((e) => e.type === 'turn' && e.turn === null).at(-1);
    return end?.type === 'turn' ? end.outcome : undefined;
  }

  cards(): Interaction[] {
    const seen = new Map<string, Interaction>();
    for (const e of this.events) {
      const list = e.type === 'interaction' ? [e.interaction] : e.type === 'sync' ? e.interactions : [];
      for (const card of list) if (!seen.has(card.id)) seen.set(card.id, card);
    }
    return [...seen.values()];
  }

  closed(id: string): string | undefined {
    const e = this.events.find((x) => x.type === 'interaction-closed' && x.id === id);
    return e?.type === 'interaction-closed' ? e.reason : undefined;
  }

  open(): Interaction[] {
    return this.state?.interactions ?? [];
  }

  marks(): PermissionMark[] {
    return this.state?.permissionMarks ?? [];
  }

  mode(): PermissionMode | undefined {
    return this.state?.conversation.permissionMode;
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

/** Scripts by prompt text; the first prompt of a conversation (`ls`) is one `ls` card. */
const SCRIPTS = (scenario: Json): void => {
  const reject = [{ optionId: 'reject-once', name: 'no', kind: 'reject_once' }];
  const neither = [{ optionId: 'always-allow', name: 'always', kind: 'allow_always' }];
  scenario.prompts = [
    { match: 'two', steps: [{ parallel: ['ls', 'cat a.txt'] }, ...END] },
    { match: 'deny-only', steps: [bashStep('ls', { options: reject }), ...END] },
    { match: 'neither', steps: [bashStep('ls', { options: neither }), ...END] },
    { match: 'printf', steps: [bashStep("printf '%d\\n' 5"), ...END] },
    { match: '', steps: [bashStep('ls'), ...END] },
  ];
};

async function withWork(
  run: (fake: FakeService, project: string) => Promise<void>,
  options: Parameters<typeof fakeService>[1] = {},
  name = 'work-permission',
): Promise<void> {
  const fake = fakeService(name, { mutate: SCRIPTS, ...options });
  const project = join(fake.root, 'home', 'code', 'app');
  mkdirSync(project, { recursive: true });
  // A window holds the host stream: Auto answers only then.
  const window = hostState(fake.service);
  try {
    await run(fake, project);
  } finally {
    window.off();
    await fake.close();
  }
}

/** A Work conversation whose first answer (one `ls` card) was allowed by the user. */
async function started(fake: FakeService, project: string): Promise<{ id: string; page: Page }> {
  const svc = fake.service;
  const { conversation } = await svc.create({ mode: 'work', text: 'ls', cwd: project });
  const page = new Page(svc, conversation.id);
  await until(() => page.cards().length === 1, 5_000);
  await svc.answer(conversation.id, page.cards()[0]!.id, { optionId: 'allow-once' });
  await until(() => page.ends() === 1, 5_000);
  return { id: conversation.id, page };
}

function logLines(fake: FakeService, id: string): Json[] {
  const file = join(fake.paths.permissionLog, `${id}.jsonl`);
  if (!existsSync(file)) return [];
  return readFileSync(file, 'utf8')
    .split('\n')
    .filter((l) => l !== '')
    .map((l) => JSON.parse(l) as Json);
}

/** The mode lines of the file as `from>mode:by`. */
function modeLines(fake: FakeService, id: string): string[] {
  return logLines(fake, id)
    .filter((l) => l.kind === 'mode')
    .map((l) => `${String(l.from)}>${String(l.mode)}:${String(l.by)}`);
}

function modeMarks(page: Page): string[] {
  return page
    .marks()
    .filter((m): m is Extract<PermissionMark, { kind: 'mode' }> => m.kind === 'mode')
    .map((m) => `${m.from}>${m.mode}:${m.by}`);
}

function hostState(svc: ChatService): {
  last: () => Extract<ChatHostEvent, { type: 'state' }> | undefined;
  off: () => void;
} {
  const events: ChatHostEvent[] = [];
  const off = svc.watchHost((e) => events.push(e));
  return {
    last: () => events.filter((e): e is Extract<ChatHostEvent, { type: 'state' }> => e.type === 'state').at(-1),
    off,
  };
}

/** Replaces `PermissionLog.prototype.append` while `run` runs. */
async function patchAppend(
  wrap: (line: PermissionLine, original: () => Promise<void>) => Promise<void>,
  run: () => Promise<void>,
): Promise<void> {
  const original = Object.getOwnPropertyDescriptor(PermissionLog.prototype, 'append')!;
  PermissionLog.prototype.append = function (this: PermissionLog, cid: string, line: PermissionLine) {
    return wrap(
      line,
      () => Reflect.apply(original.value as PermissionLog['append'], this, [cid, line]) as Promise<void>,
    );
  };
  try {
    await run();
  } finally {
    Object.defineProperty(PermissionLog.prototype, 'append', original);
  }
}

/** A gate: `wait` resolves after `open`. */
function gate(): { wait: Promise<void>; open: () => void } {
  let open = (): void => {};
  const wait = new Promise<void>((resolve) => {
    open = resolve;
  });
  return { wait, open };
}

/** Holds the service's folder check of the next move up until released. */
function holdFolderCheck(svc: ChatService): { entered: () => boolean; release: () => void } {
  const g = gate();
  let entered = false;
  const target = svc as unknown as { folderState: (c: unknown) => Promise<unknown> };
  const original = target.folderState.bind(svc);
  target.folderState = async (c: unknown) => {
    if (!entered) {
      entered = true;
      await g.wait;
    }
    return original(c);
  };
  return { entered: () => entered, release: g.open };
}

describe('the transitions', () => {
  it('Ask → Auto: ls runs with no card; the info, the marks, the host list and the file follow; Auto → Ask asks again', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const host = hostState(svc);
      const { id, page } = await started(fake, project);
      expect(page.mode()).toBe('ask');
      expect(await svc.setPermissionMode(id, 'auto', true)).toEqual({ mode: 'auto' });
      await until(() => host.last()?.state.permissionRaised.some((r) => r.id === id && r.mode === 'auto') === true);
      expect(page.mode()).toBe('auto');
      expect(page.marks()).toEqual([{ kind: 'mode', at: expect.any(Number), mode: 'auto', from: 'ask', by: 'user' }]);
      await svc.prompt(id, 'again');
      await until(() => page.ends() === 2, 5_000);
      expect([page.lastOutcome(), page.cards().length]).toEqual(['completed', 1]);
      const allowed = page.marks()[1] as Extract<PermissionMark, { kind: 'allowed' }>;
      expect(allowed).toMatchObject({
        kind: 'allowed',
        mode: 'auto',
        access: 'bash',
        card: { command: 'ls', options: [] },
      });
      const lines = logLines(fake, id);
      expect(lines.map((l) => l.kind)).toEqual(['mode', 'allowed']);
      expect(Object.keys(lines[0]!)).toEqual(['kind', 'at', 'mode', 'from', 'by', 'prints']);
      expect(lines[1]).toEqual({
        kind: 'allowed',
        at: allowed.at,
        mode: 'auto',
        toolCallId: allowed.toolCallId,
        access: 'bash',
      });
      expect(JSON.stringify(lines)).not.toContain('"ls"');

      expect(await svc.setPermissionMode(id, 'ask', false)).toEqual({ mode: 'ask' });
      expect(page.mode()).toBe('ask');
      await until(() => host.last()?.state.permissionRaised.length === 0, 5_000);
      await until(() => modeMarks(page).length === 2, 5_000);
      expect(modeMarks(page)).toEqual(['ask>auto:user', 'auto>ask:user']);
      await svc.prompt(id, 'third');
      await until(() => page.cards().length === 2, 5_000);
      expect((page.cards()[1] as PermissionInteraction).autoAsk).toBeUndefined();
      await svc.answer(id, page.cards()[1]!.id, { optionId: 'reject-once' });
      await until(() => page.ends() === 3, 5_000);
      page.detach();
      host.off();
    });
  }, 30_000);

  it('Ask → Bypass, Bypass → Auto, Auto → Bypass, Bypass → Ask, and same → same writes nothing', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const host = hostState(svc);
      const { id, page } = await started(fake, project);
      expect(await svc.setPermissionMode(id, 'bypass', true)).toEqual({ mode: 'bypass' });
      await until(() => host.last()?.state.permissionRaised.some((r) => r.mode === 'bypass') === true);
      await svc.prompt(id, 'printf');
      await until(() => page.ends() === 2, 5_000);
      expect([page.lastOutcome(), page.cards().length]).toEqual(['completed', 1]);
      expect(page.marks().at(-1)).toMatchObject({
        kind: 'allowed',
        mode: 'bypass',
        card: { command: "printf '%d\\n' 5" },
      });

      expect(await svc.setPermissionMode(id, 'auto', false)).toEqual({ mode: 'auto' });
      expect(page.mode()).toBe('auto');
      await svc.prompt(id, 'printf');
      await until(() => page.cards().length === 2, 5_000);
      expect((page.cards()[1] as PermissionInteraction).autoAsk).toBe('not-known');
      await svc.answer(id, page.cards()[1]!.id, { optionId: 'reject-once' });
      await until(() => page.ends() === 3, 5_000);

      expect(await svc.setPermissionMode(id, 'bypass', true)).toEqual({ mode: 'bypass' });
      const before = logLines(fake, id).length;
      expect(await svc.setPermissionMode(id, 'bypass', true)).toEqual({ mode: 'bypass' });
      expect(await svc.setPermissionMode(id, 'auto', false)).toEqual({ mode: 'auto' });
      expect(await svc.setPermissionMode(id, 'auto', false)).toEqual({ mode: 'auto' });
      expect(await svc.setPermissionMode(id, 'ask', false)).toEqual({ mode: 'ask' });
      expect(await svc.setPermissionMode(id, 'ask', false)).toEqual({ mode: 'ask' });
      await until(() => modeLines(fake, id).length === 5, 5_000);
      expect(logLines(fake, id).length).toBe(before + 2);
      expect(modeLines(fake, id)).toEqual([
        'ask>bypass:user',
        'bypass>auto:user',
        'auto>bypass:user',
        'bypass>auto:user',
        'auto>ask:user',
      ]);
      page.detach();
      host.off();
    });
  }, 40_000);

  it('the mode of one conversation does not reach another: its ls is still a card', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const a = await started(fake, project);
      const b = await started(fake, project);
      expect(await svc.setPermissionMode(a.id, 'bypass', true)).toEqual({ mode: 'bypass' });
      await svc.prompt(a.id, 'again');
      await until(() => a.page.ends() === 2, 5_000);
      expect([a.page.lastOutcome(), a.page.cards().length]).toEqual(['completed', 1]);
      await svc.prompt(b.id, 'again');
      await until(() => b.page.cards().length === 2, 5_000);
      expect([b.page.mode(), (b.page.cards()[1] as PermissionInteraction).autoAsk]).toEqual(['ask', undefined]);
      await svc.answer(b.id, b.page.cards()[1]!.id, { optionId: 'reject-once' });
      await until(() => b.page.ends() === 2, 5_000);
      a.page.detach();
      b.page.detach();
    });
  }, 30_000);
});

describe('refusals', () => {
  it('an unknown id, an open card for Ask → Auto, and a folder that went away (up only; a move down works)', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      expect((await apiError(svc.setPermissionMode('no-such-id', 'auto', true))).code).toBe('E_NOT_FOUND');
      expect((await apiError(svc.setPermissionMode('no-such-id', 'ask', false))).code).toBe('E_NOT_FOUND');

      const { conversation } = await svc.create({ mode: 'work', text: 'ls', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.cards().length === 1, 5_000);
      const open = await apiError(svc.setPermissionMode(id, 'auto', true));
      expect([open.code, open.reason]).toEqual(['E_CONFLICT', 'card-open']);
      await svc.answer(id, page.cards()[0]!.id, { optionId: 'reject-once' });
      await until(() => page.ends() === 1, 5_000);
      expect(await svc.setPermissionMode(id, 'auto', true)).toEqual({ mode: 'auto' });
      chmodSync(project, 0o000);
      try {
        const gone = await apiError(svc.setPermissionMode(id, 'bypass', true));
        expect([gone.code, gone.reason]).toEqual(['E_BAD_REQUEST', 'cwd-invalid']);
        expect(await svc.setPermissionMode(id, 'ask', false)).toEqual({ mode: 'ask' });
      } finally {
        chmodSync(project, 0o755);
      }
      expect(page.mode()).toBe('ask');
      page.detach();
    });
  }, 30_000);

  it('a mode line that cannot be written: 503 and the mode stays', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { id, page } = await started(fake, project);
      await patchAppend(
        async (line, original) => {
          if (line.kind === 'mode') throw new Error('EIO');
          return original();
        },
        async () => {
          const failed = await apiError(svc.setPermissionMode(id, 'bypass', true));
          expect([failed.code, failed.message]).toEqual(['E_UNAVAILABLE', 'The permission mode could not be changed.']);
        },
      );
      expect([page.mode(), svc.modeOf(id), (await svc.state()).permissionRaised]).toEqual(['ask', 'ask', []]);
      page.detach();
    });
  }, 30_000);

  it('a card that opens while the Auto line is written: Auto is refused and the record closed', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { id, page } = await started(fake, project);
      let entered = false;
      await patchAppend(
        async (line, original) => {
          if (line.kind === 'mode' && line.mode === 'auto') {
            entered = true;
            await until(() => page.cards().length === 2, 5_000);
          }
          return original();
        },
        async () => {
          const switching = apiError(svc.setPermissionMode(id, 'auto', true));
          await until(() => entered, 5_000);
          await svc.prompt(id, 'again');
          const refused = await switching;
          expect([refused.code, refused.reason]).toEqual(['E_CONFLICT', 'card-open']);
        },
      );
      const card = page.cards()[1] as PermissionInteraction;
      expect([page.mode(), (await svc.state()).permissionRaised, card.autoAsk]).toEqual(['ask', [], undefined]);
      expect(modeMarks(page)).toEqual(['ask>auto:user', 'auto>ask:user']);
      expect(modeLines(fake, id)).toEqual(['ask>auto:user', 'auto>ask:user']);
      await svc.answer(id, card.id, { optionId: 'reject-once' });
      await until(() => page.ends() === 2, 5_000);
      page.detach();
    });
  }, 30_000);

  it('a Chat id is refused', async () => {
    const fake = fakeService('chat-basic');
    try {
      const chat = await fake.service.create({ mode: 'chat', text: 'hello' });
      for (const mode of ['ask', 'auto', 'bypass'] as const) {
        for (const up of [true, false]) {
          const refused = await apiError(fake.service.setPermissionMode(chat.conversation.id, mode, up));
          expect([mode, up, refused.code, refused.message]).toEqual([
            mode,
            up,
            'E_BAD_REQUEST',
            'Permission mode is for Work only.',
          ]);
        }
      }
      expect((await fake.service.state()).permissionRaised).toEqual([]);
    } finally {
      await fake.close();
    }
  }, 20_000);
});

describe('the move to Bypass with open cards (D6)', () => {
  it('Ask → Bypass with an open card: it is allowed at once, after its line, with a Bypass mark', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: 'ls', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.cards().length === 1, 5_000);
      const card = page.cards()[0] as PermissionInteraction;
      expect(await svc.setPermissionMode(id, 'bypass', true)).toEqual({ mode: 'bypass' });
      await until(() => page.ends() === 1, 5_000);
      expect([page.closed(card.id), page.lastOutcome()]).toEqual(['answered', 'completed']);
      const reply = fake
        .records()
        .findIndex(
          (r) =>
            r.kind === 'in' && (r.msg as Json).result !== undefined && JSON.stringify(r.msg).includes('allow-once'),
        );
      expect(reply).toBeGreaterThan(-1);
      expect(logLines(fake, id).map((l) => [l.kind, l.mode])).toEqual([
        ['mode', 'bypass'],
        ['allowed', 'bypass'],
      ]);
      expect(page.marks().at(-1)).toMatchObject({ kind: 'allowed', mode: 'bypass', toolCallId: card.toolCallId });
      page.detach();
    });
  }, 30_000);

  it('Auto → Bypass with two open cards: both are allowed, one line each', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { id, page } = await started(fake, project);
      await svc.setPermissionMode(id, 'auto', true);
      await svc.prompt(id, 'two');
      // `cat a.txt` names a file the folder does not hold: both run in Auto only when both pass; here they wait.
      await until(() => page.open().length + page.marks().filter((m) => m.kind === 'allowed').length === 2, 5_000);
      const open = page.open().length;
      expect(await svc.setPermissionMode(id, 'bypass', true)).toEqual({ mode: 'bypass' });
      await until(() => page.ends() === 2, 5_000);
      expect(page.marks().filter((m) => m.kind === 'allowed' && m.mode === 'bypass')).toHaveLength(open);
      page.detach();
    });
  }, 30_000);

  it('a card with Deny only stays, a card with neither answer is withdrawn, a plan card stays', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { id, page } = await started(fake, project);
      await svc.prompt(id, 'deny-only');
      await until(() => page.open().length === 1, 5_000);
      expect(await svc.setPermissionMode(id, 'bypass', true)).toEqual({ mode: 'bypass' });
      expect(page.open()).toHaveLength(1);
      await svc.answer(id, page.open()[0]!.id, { optionId: 'reject-once' });
      await until(() => page.ends() === 2, 5_000);
      await svc.setPermissionMode(id, 'ask', false);
      await svc.prompt(id, 'neither');
      await until(() => page.open().length === 1, 5_000);
      const neither = page.open()[0]!;
      await svc.setPermissionMode(id, 'bypass', true);
      await until(() => page.ends() === 3, 5_000);
      expect([page.closed(neither.id), page.lastOutcome()]).toEqual(['cancelled', 'denied']);
      expect(page.marks().filter((m) => m.kind === 'allowed')).toEqual([]);
      page.detach();
    });
    await withWork(
      async (fake, project) => {
        const svc = fake.service;
        const { conversation } = await svc.create({ mode: 'work', text: 'plan', cwd: project });
        const page = new Page(svc, conversation.id);
        await until(() => page.open().length === 1, 5_000);
        expect(await svc.setPermissionMode(conversation.id, 'bypass', true)).toEqual({ mode: 'bypass' });
        expect(page.open().map((i) => i.kind)).toEqual(['plan']);
        await svc.answer(conversation.id, page.open()[0]!.id, { decision: 'approved' });
        await until(() => page.ends() === 1, 5_000);
        page.detach();
      },
      { mutate: () => {} },
      'work-plan',
    );
  }, 40_000);

  it('a failed allowed line: the card stays, the mode is Ask with record-failed, and the route replies ask', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { conversation } = await svc.create({ mode: 'work', text: 'ls', cwd: project });
      const id = conversation.id;
      const page = new Page(svc, id);
      await until(() => page.cards().length === 1, 5_000);
      await patchAppend(
        async (line, original) => {
          if (line.kind === 'allowed') throw new Error('EIO');
          return original();
        },
        async () => {
          expect(await svc.setPermissionMode(id, 'bypass', true)).toEqual({ mode: 'ask' });
        },
      );
      expect([page.mode(), page.open().length]).toEqual(['ask', 1]);
      await until(() => modeMarks(page).length === 2, 5_000);
      expect(modeMarks(page)).toEqual(['ask>bypass:user', 'bypass>ask:record-failed']);
      await svc.answer(id, page.open()[0]!.id, { optionId: 'reject-once' });
      await until(() => page.ends() === 1, 5_000);
      page.detach();
    });
  }, 30_000);
});

describe('the epoch: a move down wins over a move up sent before it', () => {
  it('an up that waits on its checks, then a down: the up is 409 mode-changed and nothing of it is written', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { id, page } = await started(fake, project);
      await svc.setPermissionMode(id, 'auto', true);
      const held = holdFolderCheck(svc);
      const up = apiError(svc.setPermissionMode(id, 'bypass', true));
      await until(() => held.entered(), 5_000);
      const down = svc.setPermissionMode(id, 'ask', false);
      expect(page.mode()).toBe('ask');
      expect(await down).toEqual({ mode: 'ask' });
      held.release();
      const refused = await up;
      expect([refused.code, refused.reason]).toEqual(['E_CONFLICT', 'mode-changed']);
      await until(() => modeLines(fake, id).length === 2, 5_000);
      expect(modeLines(fake, id)).toEqual(['ask>auto:user', 'auto>ask:user']);
      expect(svc.modeOf(id)).toBe('ask');
      page.detach();
    });
  }, 30_000);

  it('an up that waits on its mode line, then a down: 409; the up line is kept and the down line follows from it', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { id, page } = await started(fake, project);
      await svc.setPermissionMode(id, 'auto', true);
      const g = gate();
      let entered = false;
      await patchAppend(
        async (line, original) => {
          if (line.kind === 'mode' && line.mode === 'bypass') {
            entered = true;
            await g.wait;
          }
          return original();
        },
        async () => {
          const up = apiError(svc.setPermissionMode(id, 'bypass', true));
          await until(() => entered, 5_000);
          expect(await svc.setPermissionMode(id, 'ask', false)).toEqual({ mode: 'ask' });
          g.open();
          const refused = await up;
          expect([refused.code, refused.reason]).toEqual(['E_CONFLICT', 'mode-changed']);
        },
      );
      await until(() => modeLines(fake, id).length === 3, 5_000);
      expect(modeLines(fake, id)).toEqual(['ask>auto:user', 'auto>bypass:user', 'bypass>ask:user']);
      expect([svc.modeOf(id), page.mode()]).toEqual(['ask', 'ask']);
      page.detach();
    });
  }, 30_000);

  it("the app's own move to Ask (record-failed) while an up waits: the up is refused, the mode is Ask", async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { id, page } = await started(fake, project);
      await svc.setPermissionMode(id, 'auto', true);
      const held = holdFolderCheck(svc);
      await patchAppend(
        async (line, original) => {
          if (line.kind === 'allowed') throw new Error('EIO');
          return original();
        },
        async () => {
          const up = apiError(svc.setPermissionMode(id, 'bypass', true));
          await until(() => held.entered(), 5_000);
          await svc.prompt(id, 'again');
          await until(() => page.cards().length === 2, 5_000);
          expect(svc.modeOf(id)).toBe('ask');
          held.release();
          const refused = await up;
          expect([refused.code, refused.reason]).toEqual(['E_CONFLICT', 'mode-changed']);
        },
      );
      expect(svc.modeOf(id)).toBe('ask');
      await svc.answer(id, page.cards()[1]!.id, { optionId: 'reject-once' });
      await until(() => page.ends() === 2, 5_000);
      page.detach();
    });
  }, 30_000);

  it('a down during the D6 loop: the cards not yet answered stay, a written line is withdrawn, the reply is Ask', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { id, page } = await started(fake, project);
      await svc.prompt(id, 'two');
      await until(() => page.open().length === 2, 5_000);
      const g = gate();
      let entered = false;
      await patchAppend(
        async (line, original) => {
          if (line.kind === 'allowed' && !entered) {
            entered = true;
            await original();
            await g.wait;
            return;
          }
          return original();
        },
        async () => {
          const up = svc.setPermissionMode(id, 'bypass', true);
          await until(() => entered, 5_000);
          expect(await svc.setPermissionMode(id, 'ask', false)).toEqual({ mode: 'ask' });
          g.open();
          expect(await up).toEqual({ mode: 'ask' });
        },
      );
      expect(page.open()).toHaveLength(2);
      await until(() => logLines(fake, id).some((l) => l.kind === 'withdrawn'), 5_000);
      const lines = logLines(fake, id);
      const allowed = lines.find((l) => l.kind === 'allowed')!;
      expect(lines.find((l) => l.kind === 'withdrawn')).toMatchObject({
        of: allowed.at,
        toolCallId: allowed.toolCallId,
      });
      for (const card of page.open()) await svc.answer(id, card.id, { optionId: 'reject-once' });
      await until(() => page.ends() === 2, 5_000);
      page.detach();
    });
  }, 30_000);

  it('a down answers before a pending up settles', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { id, page } = await started(fake, project);
      await svc.setPermissionMode(id, 'auto', true);
      const held = holdFolderCheck(svc);
      let upSettled = false;
      const up = apiError(svc.setPermissionMode(id, 'bypass', true)).finally(() => {
        upSettled = true;
      });
      await until(() => held.entered(), 5_000);
      expect(await svc.setPermissionMode(id, 'ask', false)).toEqual({ mode: 'ask' });
      expect(upSettled).toBe(false);
      held.release();
      await up;
      page.detach();
    });
  }, 30_000);

  it('a pick of the mode it is in while an up waits calls the up back: it answers at once, the up is 409', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { id, page } = await started(fake, project);
      const held = holdFolderCheck(svc);
      const up = apiError(svc.setPermissionMode(id, 'bypass', true));
      await until(() => held.entered(), 5_000);
      let stayed = false;
      const stay = svc.setPermissionMode(id, 'ask', false).then((reply) => {
        stayed = true;
        return reply;
      });
      await until(() => stayed, 5_000);
      expect(await stay).toEqual({ mode: 'ask' });
      held.release();
      const refused = await up;
      expect([refused.code, refused.reason]).toEqual(['E_CONFLICT', 'mode-changed']);
      expect([svc.modeOf(id), page.mode(), modeLines(fake, id)]).toEqual(['ask', 'ask', []]);
      page.detach();
    });
  }, 30_000);
});

describe('the direction of a request: only a request sent as up moves the mode up', () => {
  it('a down picked on a view that still shows Bypass after the move to Ask: 409 mode-changed, the mode stays Ask', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { id, page } = await started(fake, project);
      expect(await svc.setPermissionMode(id, 'bypass', true)).toEqual({ mode: 'bypass' });
      expect(await svc.setPermissionMode(id, 'ask', false)).toEqual({ mode: 'ask' });
      const refused = await apiError(svc.setPermissionMode(id, 'auto', false));
      expect([refused.code, refused.reason]).toEqual(['E_CONFLICT', 'mode-changed']);
      expect([svc.modeOf(id), page.mode(), (await svc.state()).permissionRaised]).toEqual(['ask', 'ask', []]);
      await until(() => modeLines(fake, id).length === 2, 5_000);
      await svc.prompt(id, 'again');
      await until(() => page.cards().length === 2, 5_000);
      expect(modeLines(fake, id)).toEqual(['ask>bypass:user', 'bypass>ask:user']);
      await svc.answer(id, page.cards()[1]!.id, { optionId: 'reject-once' });
      await until(() => page.ends() === 2, 5_000);
      page.detach();
    });
  }, 30_000);

  it('a call-back sent as down above the mode while an up waits: the up loses, nothing moves up', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { id, page } = await started(fake, project);
      const held = holdFolderCheck(svc);
      const up = apiError(svc.setPermissionMode(id, 'bypass', true));
      await until(() => held.entered(), 5_000);
      let answered = false;
      const back = apiError(svc.setPermissionMode(id, 'auto', false)).finally(() => {
        answered = true;
      });
      await until(() => answered, 5_000).finally(held.release);
      expect([(await back).code, (await back).reason]).toEqual(['E_CONFLICT', 'mode-changed']);
      const refused = await up;
      expect([refused.code, refused.reason]).toEqual(['E_CONFLICT', 'mode-changed']);
      expect([svc.modeOf(id), page.mode(), modeLines(fake, id)]).toEqual(['ask', 'ask', []]);
      page.detach();
    });
  }, 30_000);

  it('another window moved down while an up waits; the call-back of the first window (its old mode) moves nothing up', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { id, page } = await started(fake, project);
      await svc.setPermissionMode(id, 'auto', true);
      const held = holdFolderCheck(svc);
      const up = apiError(svc.setPermissionMode(id, 'bypass', true));
      await until(() => held.entered(), 5_000);
      expect(await svc.setPermissionMode(id, 'ask', false)).toEqual({ mode: 'ask' });
      const back = svc.setPermissionMode(id, 'auto', false).then(
        () => 'moved',
        (error: unknown) => routeError('chat', error).reason,
      );
      held.release();
      expect((await up).reason).toBe('mode-changed');
      expect(await back).toBe('mode-changed');
      await until(() => modeLines(fake, id).length === 2, 5_000);
      expect([svc.modeOf(id), modeLines(fake, id)]).toEqual(['ask', ['ask>auto:user', 'auto>ask:user']]);
      page.detach();
    });
  }, 30_000);

  it('after a Control Panel restart, a down from a page that still shows Bypass: 409, the mode stays Ask', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { id, page } = await started(fake, project);
      await svc.setPermissionMode(id, 'bypass', true);
      page.detach();
      await svc.close();
      const next = fake.restart();
      const refused = await apiError(next.setPermissionMode(id, 'auto', false));
      expect([refused.code, refused.reason]).toEqual(['E_CONFLICT', 'mode-changed']);
      expect([next.modeOf(id), (await next.state()).permissionRaised]).toEqual(['ask', []]);
      expect(modeLines(fake, id)).toEqual(['ask>bypass:user', 'bypass>ask:restart']);
    });
  }, 30_000);

  it('a request sent as up whose target is not above the mode: at the mode it answers the mode; below it moves down', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { id, page } = await started(fake, project);
      await svc.setPermissionMode(id, 'bypass', true);
      expect(await svc.setPermissionMode(id, 'bypass', true)).toEqual({ mode: 'bypass' });
      expect(await svc.setPermissionMode(id, 'auto', true)).toEqual({ mode: 'auto' });
      expect([svc.modeOf(id), page.mode()]).toEqual(['auto', 'auto']);
      await until(() => modeLines(fake, id).length === 2, 5_000);
      expect(modeLines(fake, id)).toEqual(['ask>bypass:user', 'bypass>auto:user']);
      page.detach();
    });
  }, 30_000);

  it('a down above the mode for an unknown id is 404, for a Chat id 400, as for any other request', async () => {
    const fake = fakeService('chat-basic');
    try {
      const svc = fake.service;
      expect((await apiError(svc.setPermissionMode('no-such-id', 'bypass', false))).code).toBe('E_NOT_FOUND');
      const chat = await svc.create({ mode: 'chat', text: 'hello' });
      const refused = await apiError(svc.setPermissionMode(chat.conversation.id, 'auto', false));
      expect([refused.code, refused.message]).toEqual(['E_BAD_REQUEST', 'Permission mode is for Work only.']);
    } finally {
      await fake.close();
    }
  }, 20_000);
});

describe('the record', () => {
  it('moved down while the allowed line is written: the plain card, no allowed mark, and the line is withdrawn', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { id, page } = await started(fake, project);
      await svc.setPermissionMode(id, 'bypass', true);
      const g = gate();
      let entered = false;
      await patchAppend(
        async (line, original) => {
          if (line.kind === 'allowed') {
            entered = true;
            await g.wait;
          }
          return original();
        },
        async () => {
          await svc.prompt(id, 'again');
          await until(() => entered, 5_000);
          expect(await svc.setPermissionMode(id, 'ask', false)).toEqual({ mode: 'ask' });
          g.open();
          await until(() => page.cards().length === 2, 5_000);
        },
      );
      const card = page.cards()[1] as PermissionInteraction;
      expect([card.autoAsk, card.options.map((o) => o.optionId)]).toEqual([
        undefined,
        ['always-allow', 'allow-once', 'reject-once'],
      ]);
      expect(page.marks().some((m) => m.kind === 'allowed')).toBe(false);
      await until(() => logLines(fake, id).some((l) => l.kind === 'withdrawn'), 5_000);
      await svc.answer(id, card.id, { optionId: 'reject-once' });
      await until(() => page.ends() === 2, 5_000);
      page.detach();
    });
  }, 30_000);

  it('a history line that cannot be written: the call is a card and the mode is Ask with a record-failed line', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const host = hostState(svc);
      const { id, page } = await started(fake, project);
      await svc.setPermissionMode(id, 'bypass', true);
      chmodSync(join(fake.paths.permissionLog, `${id}.jsonl`), 0o400);
      try {
        await svc.prompt(id, 'again');
        await until(() => page.cards().length === 2, 5_000);
        expect((page.cards()[1] as PermissionInteraction).autoAsk).toBeUndefined();
        expect(page.mode()).toBe('ask');
        await until(() => modeMarks(page).length === 2, 5_000);
        expect(modeMarks(page).at(-1)).toBe('bypass>ask:record-failed');
        await svc.answer(id, page.cards()[1]!.id, { optionId: 'reject-once' });
        await until(() => page.ends() === 2, 5_000);
      } finally {
        chmodSync(join(fake.paths.permissionLog, `${id}.jsonl`), 0o600);
      }
      page.detach();
      host.off();
    });
  }, 30_000);

  it('delete removes the entry, the epoch and the history file', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { id, page } = await started(fake, project);
      await svc.setPermissionMode(id, 'bypass', true);
      const file = join(fake.paths.permissionLog, `${id}.jsonl`);
      expect(existsSync(file)).toBe(true);
      await svc.delete(id);
      expect(existsSync(file)).toBe(false);
      expect([(await svc.state()).permissionRaised, svc.modeOf(id)]).toEqual([[], 'ask']);
      page.detach();
    });
  }, 30_000);

  it('a move down whose line is queued behind a held up, then delete: the file is not written again', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { id, page } = await started(fake, project);
      await svc.setPermissionMode(id, 'auto', true);
      const file = join(fake.paths.permissionLog, `${id}.jsonl`);
      const held = holdFolderCheck(svc);
      const up = apiError(svc.setPermissionMode(id, 'bypass', true));
      await until(() => held.entered(), 5_000);
      expect(await svc.setPermissionMode(id, 'ask', false)).toEqual({ mode: 'ask' });
      page.detach();
      await svc.delete(id);
      expect(existsSync(file)).toBe(false);
      held.release();
      expect((await up).status).toBe(404);
      await (svc as unknown as { permissionChain: Promise<unknown> }).permissionChain;
      expect(existsSync(file)).toBe(false);
    });
  }, 30_000);

  it('a tool hook of the user: the info and the up lines say so (Auto and Bypass), and ls still runs', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      mkdirSync(join(fake.root, 'home', '.claude'), { recursive: true });
      writeFileSync(
        join(fake.root, 'home', '.claude', 'settings.json'),
        JSON.stringify({ hooks: { PreToolUse: [{ matcher: '*', hooks: [{ type: 'command', command: 'true' }] }] } }),
      );
      const { id, page } = await started(fake, project);
      await until(() => page.state?.conversation.userHooks === true, 5_000);
      await svc.setPermissionMode(id, 'auto', true);
      await svc.setPermissionMode(id, 'bypass', true);
      expect(page.marks().filter((m) => m.kind === 'mode')).toMatchObject([
        { mode: 'auto', userHooks: true },
        { mode: 'bypass', userHooks: true },
      ]);
      await svc.prompt(id, 'again');
      await until(() => page.ends() === 2, 5_000);
      expect(page.cards()).toHaveLength(1);
      expect(JSON.stringify(logLines(fake, id))).not.toContain('PreToolUse');
      page.detach();
    });
  }, 30_000);

  it('a settings file that changed: a settings mark at a mode change, at the next session load and after a restart', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const settings = join(fake.root, 'agent-home', 'config.toml');
      const { id, page } = await started(fake, project);
      await svc.setPermissionMode(id, 'auto', true);
      writeFileSync(settings, '[permission]\nrules = []\n');
      await svc.setPermissionMode(id, 'ask', false);
      await until(() => page.marks().length === 3, 5_000);
      expect(page.marks().map((m) => m.kind)).toEqual(['mode', 'settings', 'mode']);

      await svc.setPermissionMode(id, 'bypass', true);
      writeFileSync(settings, '[permission]\nrules = [1]\n');
      await svc.agentAction({ action: 'restart', mode: 'work' });
      await svc.prompt(id, 'again');
      await until(() => page.ends() === 2, 5_000);
      await until(() => page.marks().filter((m) => m.kind === 'settings').length === 2, 5_000);

      writeFileSync(settings, '[permission]\nrules = [2]\n');
      page.detach();
      await svc.close();
      const next = fake.restart();
      const after = new Page(next, id);
      await until(() => after.state !== null, 5_000);
      const kinds = after.marks().map((m) => (m.kind === 'mode' ? `${m.from}>${m.mode}:${m.by}` : m.kind));
      expect(kinds.slice(-2)).toEqual(['settings', 'bypass>ask:restart']);
      after.detach();
    });
  }, 40_000);
});

describe('a Control Panel restart', () => {
  it.each<'auto' | 'bypass'>(['auto', 'bypass'])(
    'after %s: Ask with one restart line; the next request is a card',
    async (mode) => {
      await withWork(async (fake, project) => {
        const svc = fake.service;
        const { id, page } = await started(fake, project);
        await svc.setPermissionMode(id, mode, true);
        await svc.prompt(id, 'again');
        await until(() => page.ends() === 2, 5_000);
        expect(page.cards()).toHaveLength(1);
        page.detach();
        await svc.close();
        const next = fake.restart();
        expect((await next.state()).permissionRaised).toEqual([]);
        const after = new Page(next, id);
        await until(() => after.state !== null, 5_000);
        expect(after.mode()).toBe('ask');
        expect(after.marks().map((m) => m.kind)).toEqual(['mode', 'allowed', 'mode']);
        expect(after.marks()[1]).toMatchObject({ kind: 'allowed', mode, card: null });
        expect(modeMarks(after)).toEqual([`ask>${mode}:user`, `${mode}>ask:restart`]);
        expect(logLines(fake, id).at(-1)).toMatchObject({ kind: 'mode', mode: 'ask', from: mode, by: 'restart' });
        await next.prompt(id, 'third');
        await until(() => after.cards().length === 1, 5_000);
        expect((after.cards()[0] as PermissionInteraction).autoAsk).toBeUndefined();
        await next.answer(id, after.cards()[0]!.id, { optionId: 'reject-once' });
        await until(() => after.ends() >= 1, 5_000);
        after.detach();
      });
    },
    40_000,
  );

  it('an allowed line with its withdrawn line on disk: no mark after the reload', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const { id, page } = await started(fake, project);
      await svc.setPermissionMode(id, 'bypass', true);
      page.detach();
      await svc.close();
      const file = join(fake.paths.permissionLog, `${id}.jsonl`);
      writeFileSync(
        file,
        `${readFileSync(file, 'utf8')}${JSON.stringify({ kind: 'allowed', at: 7, mode: 'bypass', toolCallId: 'c1', access: 'bash' })}\n${JSON.stringify({ kind: 'withdrawn', at: 8, of: 7, toolCallId: 'c1' })}\n`,
      );
      const next = fake.restart();
      const after = new Page(next, id);
      await until(() => after.state !== null, 5_000);
      expect(after.marks().map((m) => m.kind)).toEqual(['mode', 'mode']);
      after.detach();
    });
  }, 30_000);
});

describe('the marks of a conversation', () => {
  it(`sync sends every mark, card text for the newest ${MARK_CARDS_KEPT}; marks are redacted`, () => {
    const conversation = new Conversation(
      {
        id: 'c1',
        mode: 'work',
        title: '',
        cwd: '/w',
        modelId: 'm',
        createdAt: 1,
        updatedAt: 1,
        hot: true,
        permissionMode: 'bypass',
      },
      { needles: () => [INFERENCE_TOKEN], flushMs: 5 },
    );
    const card = (n: number): PermissionInteraction => ({
      id: `i${n}`,
      kind: 'permission',
      toolCallId: `call-${n}`,
      toolName: 'run_terminal_command',
      access: 'bash',
      title: 'Execute',
      input: '{}',
      command: `echo ${INFERENCE_TOKEN} ${n}`,
      path: null,
      resolvedPath: null,
      options: [],
      flags: [],
    });
    for (let n = 0; n < 2_500; n++) {
      conversation.permissionMark({
        kind: 'allowed',
        at: n,
        mode: 'bypass',
        toolCallId: `call-${n}`,
        access: 'bash',
        card: card(n),
      });
    }
    const events: ConversationEvent[] = [];
    conversation.attach((e) => events.push(e), { turn: null, interactions: [] });
    const sync = events[0];
    if (sync?.type !== 'sync') throw new Error('no sync');
    expect(sync.permissionMarks).toHaveLength(2_500);
    const withCard = sync.permissionMarks.filter((m) => m.kind === 'allowed' && m.card !== null);
    expect(withCard).toHaveLength(MARK_CARDS_KEPT);
    expect(withCard[0]).toMatchObject({ toolCallId: `call-${2_500 - MARK_CARDS_KEPT}` });
    expect(JSON.stringify(sync.permissionMarks)).not.toContain(INFERENCE_TOKEN);
    conversation.permissionMark({ kind: 'settings', at: 3_000, changed: ['~/.claude.json'] });
    expect(events.at(-1)).toMatchObject({ type: 'permission-mark', mark: { kind: 'settings' } });
    conversation.permissionMark({
      kind: 'allowed',
      at: 3_001,
      mode: 'auto',
      toolCallId: 'call-live',
      access: 'bash',
      card: card(3_001),
    });
    expect(events.at(-1)).toMatchObject({ type: 'permission-mark', mark: { toolCallId: 'call-live' } });
    expect(JSON.stringify(events.at(-1))).not.toContain(INFERENCE_TOKEN);
    expect(JSON.stringify(conversation.permissionMarks().at(-1))).not.toContain(INFERENCE_TOKEN);
  });
});

describe('the settings check names the files that changed', () => {
  function claudeJson(fake: FakeService, value: Json): void {
    writeFileSync(join(fake.root, 'home', '.claude.json'), JSON.stringify(value));
  }

  function settingsMarks(page: Page): Array<Extract<PermissionMark, { kind: 'settings' }>> {
    return page.marks().filter((m): m is Extract<PermissionMark, { kind: 'settings' }> => m.kind === 'settings');
  }

  /** Auto, `change`, then Ask: the move down compares the settings with the move up. */
  async function acrossAuto(fake: FakeService, id: string, page: Page, change: () => void): Promise<void> {
    const modes = modeMarks(page).length;
    await fake.service.setPermissionMode(id, 'auto', true);
    change();
    await fake.service.setPermissionMode(id, 'ask', false);
    await until(() => modeMarks(page).length === modes + 2, 5_000);
    await (fake.service as unknown as { permissionChain: Promise<unknown> }).permissionChain;
  }

  /** The `changed` list of each settings mark, oldest first. */
  function changedLists(page: Page): string[][] {
    return settingsMarks(page).map((m) => m.changed);
  }

  /** A path as the thread shows it, for this fake home. */
  function shown(fake: FakeService, path: string): string {
    const home = join(fake.root, 'home');
    return path.startsWith(`${home}/`) ? `~${path.slice(home.length)}` : path;
  }

  function cwdOf(page: Page): string {
    const cwd = page.state?.conversation.cwd;
    if (typeof cwd !== 'string') throw new Error('no folder');
    return cwd;
  }

  it('a usage counter in ~/.claude.json is no change', async () => {
    await withWork(async (fake, project) => {
      claudeJson(fake, { numStartups: 1, pluginUsage: { a: 1 }, mcpServers: {} });
      const { id, page } = await started(fake, project);
      await acrossAuto(fake, id, page, () => {
        claudeJson(fake, { numStartups: 2, pluginUsage: { a: 2, b: 1 }, mcpServers: {} });
      });
      expect(page.marks().map((m) => m.kind)).toEqual(['mode', 'mode']);
      page.detach();
    });
  }, 30_000);

  it('~/.claude.json: an MCP server, or the MCP keys of the folder or its git root, is a change; another folder is not', async () => {
    await withWork(async (fake, project) => {
      mkdirSync(join(fake.root, 'home', 'code', '.git'), { recursive: true });
      claudeJson(fake, { mcpServers: {}, projects: {} });
      const { id, page } = await started(fake, project);
      const cwd = cwdOf(page);
      const root = join(cwd, '..');
      await acrossAuto(fake, id, page, () => claudeJson(fake, { mcpServers: { x: { command: 'x' } }, projects: {} }));
      await acrossAuto(fake, id, page, () =>
        claudeJson(fake, { mcpServers: { x: { command: 'x' } }, projects: { [cwd]: { mcpServers: { y: {} } } } }),
      );
      await acrossAuto(fake, id, page, () =>
        claudeJson(fake, {
          mcpServers: { x: { command: 'x' } },
          projects: { [cwd]: { mcpServers: { y: {} } }, [root]: { enabledMcpjsonServers: ['z'] } },
        }),
      );
      await acrossAuto(fake, id, page, () =>
        claudeJson(fake, {
          projects: {
            [cwd]: { mcpServers: { y: {} }, history: ['a'] },
            [root]: { enabledMcpjsonServers: ['z'], allowedTools: [] },
            '/elsewhere': { mcpServers: { w: {} } },
          },
          mcpServers: { x: { command: 'x' } },
          pluginUsage: { a: 9 },
        }),
      );
      expect(changedLists(page)).toEqual([['~/.claude.json'], ['~/.claude.json'], ['~/.claude.json']]);
      page.detach();
    });
  }, 40_000);

  it('a folder settings file, two files, a file that appears and disappears: each mark names its files', async () => {
    await withWork(async (fake, project) => {
      const { id, page } = await started(fake, project);
      const cwd = cwdOf(page);
      const local = join(cwd, '.claude', 'settings.local.json');
      const agentConfig = join(fake.root, 'agent-home', 'config.toml');
      const mcp = join(cwd, '.mcp.json');
      await acrossAuto(fake, id, page, () => {
        mkdirSync(join(cwd, '.claude'), { recursive: true });
        writeFileSync(local, '{"permissions":{"allow":["Bash(ls)"]}}');
      });
      await acrossAuto(fake, id, page, () => {
        writeFileSync(agentConfig, '[permission]\nrules = []\n');
        claudeJson(fake, { mcpServers: { x: {} } });
      });
      await acrossAuto(fake, id, page, () => writeFileSync(mcp, '{"mcpServers":{}}'));
      await acrossAuto(fake, id, page, () => rmSync(mcp));
      expect(changedLists(page)).toEqual([
        [shown(fake, local)],
        [shown(fake, agentConfig), '~/.claude.json'],
        [shown(fake, mcp)],
        [shown(fake, mcp)],
      ]);
      const lines = logLines(fake, id).filter((l) => l.kind === 'settings');
      expect(lines.map((l) => l.changed)).toEqual([
        ['folder:.claude/settings.local.json'],
        ['agent:config.toml', 'home:.claude.json'],
        ['folder:.mcp.json'],
        ['folder:.mcp.json'],
      ]);
      expect(JSON.stringify(logLines(fake, id))).not.toContain(cwd);
      page.detach();
    });
  }, 40_000);

  it('~/.claude.json that is not JSON is one change; a second bad text is not another', async () => {
    await withWork(async (fake, project) => {
      claudeJson(fake, { mcpServers: {} });
      const { id, page } = await started(fake, project);
      await acrossAuto(fake, id, page, () => writeFileSync(join(fake.root, 'home', '.claude.json'), '{"mcpServers":'));
      await acrossAuto(fake, id, page, () => writeFileSync(join(fake.root, 'home', '.claude.json'), 'not json'));
      expect(changedLists(page)).toEqual([['~/.claude.json']]);
      page.detach();
    });
  }, 30_000);

  it('a record in the old form (one print per line) loads with no settings mark and no restart line', async () => {
    await withWork(async (fake, project) => {
      const { id, page } = await started(fake, project);
      page.detach();
      await fake.service.close();
      const print = 'b'.repeat(64);
      mkdirSync(fake.paths.permissionLog, { recursive: true });
      writeFileSync(
        join(fake.paths.permissionLog, `${id}.jsonl`),
        [
          { kind: 'mode', at: 1, mode: 'auto', from: 'ask', by: 'user', print },
          { kind: 'allowed', at: 2, mode: 'auto', toolCallId: 'call-1', access: 'bash' },
          { kind: 'settings', at: 3, print: 'c'.repeat(64) },
          { kind: 'mode', at: 4, mode: 'bypass', from: 'auto', by: 'user', print },
        ]
          .map((l) => JSON.stringify(l))
          .join('\n') + '\n',
      );
      writeFileSync(join(fake.root, 'agent-home', 'config.toml'), '[permission]\nrules = [9]\n');
      const next = fake.restart();
      const after = new Page(next, id);
      await until(() => after.state !== null, 5_000);
      await (next as unknown as { permissionChain: Promise<unknown> }).permissionChain;
      expect(after.marks().map((m) => m.kind)).toEqual(['allowed']);
      expect(after.mode()).toBe('ask');
      expect(logLines(fake, id)).toHaveLength(4);
      after.detach();
    });
  }, 30_000);
});

describe('the temp folder Auto covers', () => {
  /** A throwaway temp root spelled `/tmp` (the gate never reads it here), inside `holder`. */
  function tempRoot(): { holder: string; roots: AllowedRoot[] } {
    const holder = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-temp-holder-')));
    const top = join(holder, 't');
    mkdirSync(top);
    return { holder, roots: [{ top, spellings: [top, '/tmp'], kind: 'temp' }] };
  }

  it('a move up to Auto must name it; that line and its mark carry temp, also a move from Bypass down to Auto', async () => {
    const { holder, roots } = tempRoot();
    try {
      await withWork(
        async (fake, project) => {
          const svc = fake.service;
          const host = hostState(svc);
          await until(() => host.last()?.state.tempRoot === '/tmp');
          expect(host.last()?.state.tempTop).toBe(roots[0]!.top);
          const { id, page } = await started(fake, project);
          for (const temp of [null, roots[0]!.top, '/private/tmp']) {
            const refused = await apiError(svc.setPermissionMode(id, 'auto', true, temp));
            expect([temp, refused.code, refused.reason]).toEqual([temp, 'E_CONFLICT', 'temp-unnamed']);
          }
          expect(page.mode()).toBe('ask');
          expect(logLines(fake, id)).toEqual([]);
          expect(await svc.setPermissionMode(id, 'auto', true, '/tmp')).toEqual({ mode: 'auto' });
          expect(logLines(fake, id).at(-1)).toMatchObject({ kind: 'mode', mode: 'auto', from: 'ask', temp: true });
          await until(() => modeMarks(page).length === 1);
          expect(page.marks()[0]).toMatchObject({ kind: 'mode', mode: 'auto', temp: true });
          expect(await svc.setPermissionMode(id, 'bypass', true, '/tmp')).toEqual({ mode: 'bypass' });
          expect(Object.hasOwn(logLines(fake, id).at(-1)!, 'temp')).toBe(false);
          expect(await svc.setPermissionMode(id, 'auto', false)).toEqual({ mode: 'auto' });
          await until(() => modeLines(fake, id).length === 3, 5_000);
          expect(logLines(fake, id).at(-1)).toMatchObject({ mode: 'auto', from: 'bypass', temp: true });
          expect(await svc.setPermissionMode(id, 'ask', false)).toEqual({ mode: 'ask' });
          await until(() => modeLines(fake, id).length === 4, 5_000);
          expect(Object.hasOwn(logLines(fake, id).at(-1)!, 'temp')).toBe(false);
          page.detach();
          host.off();
        },
        { tempRoots: roots },
      );
    } finally {
      rmSync(holder, { recursive: true, force: true });
    }
  }, 30_000);

  it('a Work folder that holds the temp folder gets none: Auto needs no name and its line has no temp', async () => {
    const { holder, roots } = tempRoot();
    try {
      await withWork(
        async (fake) => {
          const { id } = await started(fake, holder);
          expect(await fake.service.setPermissionMode(id, 'auto', true)).toEqual({ mode: 'auto' });
          expect(logLines(fake, id).at(-1)).toMatchObject({ kind: 'mode', mode: 'auto' });
          expect(Object.hasOwn(logLines(fake, id).at(-1)!, 'temp')).toBe(false);
        },
        { tempRoots: roots },
      );
    } finally {
      rmSync(holder, { recursive: true, force: true });
    }
  }, 30_000);

  it('with no root given, the service covers /tmp exactly when the resolver accepts it on this machine', async () => {
    const fake = fakeService('work-permission', { mutate: SCRIPTS, tempRoots: 'default' });
    const host = hostState(fake.service);
    try {
      const home = join(fake.root, 'home');
      const temps = [tmpdir(), process.env.TMPDIR ?? ''].filter((t) => t !== '');
      const want = resolveTempRoot('/tmp', home, temps).length === 0 ? null : '/tmp';
      await until(() => host.last() !== undefined);
      expect(host.last()!.state.tempRoot).toBe(want);
    } finally {
      host.off();
      await fake.close();
    }
  });
});

describe('the mode picked on the New Work screen', () => {
  type Hooked = { startTurn: (conversation: Conversation, text: string, cwd?: string) => { turnId: string } };

  /** Runs `spy` at each start of a turn while `run` runs. */
  async function onStart(svc: ChatService, spy: (conversation: Conversation) => void, run: () => Promise<void>) {
    const target = svc as unknown as Hooked;
    const original = target.startTurn;
    target.startTurn = function (this: ChatService, conversation, text, cwd) {
      spy(conversation);
      return Reflect.apply(original, this, [conversation, text, cwd]) as { turnId: string };
    };
    try {
      await run();
    } finally {
      target.startTurn = original;
    }
  }

  async function settledTurn(svc: ChatService, id: string): Promise<Page> {
    const page = new Page(svc, id);
    await until(() => page.state !== null && page.state.turn === null && (page.state.entries.length ?? 0) > 1, 8_000);
    return page;
  }

  it('Bypass, then the first message: the mode line is on disk before the turn starts; its call runs with no card', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const host = hostState(svc);
      expect(await svc.setDraftPermissionMode('d1', 'bypass', true, project, null)).toEqual({ mode: 'bypass' });
      await until(() => host.last()?.state.permissionDraft?.draft === 'd1');
      expect(host.last()?.state.permissionDraft).toEqual({ draft: 'd1', mode: 'bypass' });
      let atStart: Json[] | null = null;
      let made: Awaited<ReturnType<ChatService['create']>> | null = null;
      await onStart(
        svc,
        (c) => {
          atStart = logLines(fake, c.info.id);
          expect(svc.modeOf(c.info.id)).toBe('bypass');
        },
        async () => {
          made = await svc.create({ mode: 'work', text: 'printf', cwd: project, draft: 'd1' });
        },
      );
      const id = made!.conversation.id;
      expect(made!.conversation.permissionMode).toBe('bypass');
      expect(atStart).toEqual([expect.objectContaining({ kind: 'mode', mode: 'bypass', from: 'ask', by: 'user' })]);
      await until(() => host.last()?.state.permissionRaised.some((r) => r.id === id && r.mode === 'bypass') === true);
      expect(host.last()?.state.permissionDraft).toBeNull();
      const page = await settledTurn(svc, id);
      expect(page.cards()).toEqual([]);
      expect(page.marks().map((m) => m.kind)).toEqual(['mode', 'allowed']);
      expect(page.marks()[1]).toMatchObject({ kind: 'allowed', mode: 'bypass', card: { command: "printf '%d\\n' 5" } });
      page.detach();
      host.off();
    });
  }, 30_000);

  it('the first message in another folder: Ask, and the pick is gone', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      const other = join(fake.root, 'home', 'code', 'other');
      mkdirSync(other, { recursive: true });
      await svc.setDraftPermissionMode('d1', 'bypass', true, project, null);
      const { conversation } = await svc.create({ mode: 'work', text: 'ls', cwd: other, draft: 'd1' });
      expect([conversation.permissionMode, svc.modeOf(conversation.id)]).toEqual(['ask', 'ask']);
      expect((await svc.state()).permissionDraft).toBeNull();
      const page = new Page(svc, conversation.id);
      await until(() => page.cards().length === 1, 5_000);
      await svc.answer(conversation.id, page.cards()[0]!.id, { optionId: 'reject-once' });
      page.detach();
      expect(logLines(fake, conversation.id)).toEqual([]);
    });
  }, 30_000);

  it('one pick, one conversation: a second first message with the same draft is Ask', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      await svc.setDraftPermissionMode('d1', 'auto', true, project, null);
      const a = await svc.create({ mode: 'work', text: 'ls', cwd: project, draft: 'd1' });
      const b = await svc.create({ mode: 'work', text: 'ls', cwd: project, draft: 'd1' });
      expect([a.conversation.permissionMode, b.conversation.permissionMode]).toEqual(['auto', 'ask']);
      const up = await apiError(svc.setDraftPermissionMode('d1', 'bypass', true, project, null));
      expect([up.code, up.reason]).toEqual(['E_CONFLICT', 'mode-changed']);
      (await settledTurn(svc, a.conversation.id)).detach();
      const page = new Page(svc, b.conversation.id);
      await until(() => page.cards().length === 1, 5_000);
      await svc.answer(b.conversation.id, page.cards()[0]!.id, { optionId: 'reject-once' });
      page.detach();
    });
  }, 30_000);

  it('a move down on the draft: to Ask the first message is Ask; Bypass to Auto starts in Auto with the temp folder', async () => {
    const holder = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-temp-holder-')));
    const top = join(holder, 't');
    mkdirSync(top);
    try {
      await withWork(
        async (fake, project) => {
          const svc = fake.service;
          await svc.setDraftPermissionMode('d1', 'bypass', true, project, null);
          expect(await svc.setDraftPermissionMode('d1', 'ask', false, null, null)).toEqual({ mode: 'ask' });
          expect((await svc.state()).permissionDraft).toBeNull();
          const a = await svc.create({ mode: 'work', text: 'printf', cwd: project, draft: 'd1' });
          expect(a.conversation.permissionMode).toBe('ask');
          await svc.cancel(a.conversation.id);
          await svc.setDraftPermissionMode('d2', 'bypass', true, project, null);
          expect(await svc.setDraftPermissionMode('d2', 'auto', false, null, null)).toEqual({ mode: 'auto' });
          expect((await svc.state()).permissionDraft).toEqual({ draft: 'd2', mode: 'auto' });
          const b = await svc.create({ mode: 'work', text: 'ls', cwd: project, draft: 'd2' });
          expect(b.conversation.permissionMode).toBe('auto');
          expect(logLines(fake, b.conversation.id)[0]).toMatchObject({
            kind: 'mode',
            mode: 'auto',
            from: 'ask',
            temp: true,
          });
          (await settledTurn(svc, b.conversation.id)).detach();
        },
        { tempRoots: [{ top, spellings: [top, '/tmp'], kind: 'temp' }] },
      );
    } finally {
      rmSync(holder, { recursive: true, force: true });
    }
  }, 30_000);

  it('a first message that fails before its conversation exists keeps the pick; the next one takes it', async () => {
    await withWork(
      async (fake, project) => {
        const svc = fake.service;
        await svc.setDraftPermissionMode('d1', 'bypass', true, project, null);
        const host = (svc as unknown as { hosts: { work: { createSession: (o: unknown) => Promise<unknown> } } }).hosts
          .work;
        const original = host.createSession;
        host.createSession = () => new Promise(() => {});
        const slow = await apiError(svc.create({ mode: 'work', text: 'printf', cwd: project, draft: 'd1' }));
        host.createSession = original;
        expect(slow.reason).toBe('agent-slow');
        expect((await svc.state()).permissionDraft).toEqual({ draft: 'd1', mode: 'bypass' });
        const made = await svc.create({ mode: 'work', text: 'printf', cwd: project, draft: 'd1' });
        expect(made.conversation.permissionMode).toBe('bypass');
        (await settledTurn(svc, made.conversation.id)).detach();
      },
      { createBudgetMs: 1_500 },
    );
  }, 30_000);

  it('a mode line that cannot be written at the first message: Ask, the turn runs, and its call is a card', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      await svc.setDraftPermissionMode('d1', 'bypass', true, project, null);
      let made: Awaited<ReturnType<ChatService['create']>> | null = null;
      await patchAppend(
        async (line, original) => {
          if (line.kind === 'mode') throw new Error('disk full');
          await original();
        },
        async () => {
          made = await svc.create({ mode: 'work', text: 'ls', cwd: project, draft: 'd1' });
        },
      );
      const id = made!.conversation.id;
      expect([made!.conversation.permissionMode, svc.modeOf(id)]).toEqual(['ask', 'ask']);
      const page = new Page(svc, id);
      await until(() => page.cards().length === 1, 5_000);
      expect((page.cards()[0] as PermissionInteraction).autoAsk).toBeUndefined();
      await svc.answer(id, page.cards()[0]!.id, { optionId: 'reject-once' });
      page.detach();
    });
  }, 30_000);

  it('an up that waits on the folder check, then a down: 409 mode-changed and no pick', async () => {
    await withWork(async (_fake, project) => {
      const svc = _fake.service;
      const up = apiError(svc.setDraftPermissionMode('d1', 'bypass', true, project, null));
      expect(await svc.setDraftPermissionMode('d1', 'ask', false, null, null)).toEqual({ mode: 'ask' });
      const refused = await up;
      expect([refused.code, refused.reason]).toEqual(['E_CONFLICT', 'mode-changed']);
      expect((await svc.state()).permissionDraft).toBeNull();
    });
  }, 30_000);

  it('a move down above the draft mode (a stale view) is 409 and moves nothing up', async () => {
    await withWork(async (_fake, project) => {
      const svc = _fake.service;
      for (const mode of ['auto', 'bypass'] as const) {
        const stale = await apiError(svc.setDraftPermissionMode('d1', mode, false, null, null));
        expect([mode, stale.code, stale.reason]).toEqual([mode, 'E_CONFLICT', 'mode-changed']);
      }
      await svc.setDraftPermissionMode('d1', 'auto', true, project, null);
      const stale = await apiError(svc.setDraftPermissionMode('d1', 'bypass', false, null, null));
      expect([stale.code, stale.reason]).toEqual(['E_CONFLICT', 'mode-changed']);
      expect((await svc.state()).permissionDraft).toEqual({ draft: 'd1', mode: 'auto' });
    });
  }, 30_000);

  it('Auto on a draft must name the temp folder Auto covers', async () => {
    const holder = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-temp-holder-')));
    const top = join(holder, 't');
    mkdirSync(top);
    try {
      await withWork(
        async (fake, project) => {
          const svc = fake.service;
          for (const temp of [null, top, '/private/tmp']) {
            const refused = await apiError(svc.setDraftPermissionMode('d1', 'auto', true, project, temp));
            expect([temp, refused.code, refused.reason]).toEqual([temp, 'E_CONFLICT', 'temp-unnamed']);
          }
          expect((await svc.state()).permissionDraft).toBeNull();
          expect(await svc.setDraftPermissionMode('d1', 'auto', true, project, '/tmp')).toEqual({ mode: 'auto' });
          expect(await svc.setDraftPermissionMode('d1', 'bypass', true, project, null)).toEqual({ mode: 'bypass' });
        },
        { tempRoots: [{ top, spellings: [top, '/tmp'], kind: 'temp' }] },
      );
    } finally {
      rmSync(holder, { recursive: true, force: true });
    }
  }, 30_000);

  it('a down on the draft while the first message raises the mode: Ask; a down after it moves the conversation', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      await svc.setDraftPermissionMode('d1', 'bypass', true, project, null);
      const held = holdFolderCheck(svc);
      const making = svc.create({ mode: 'work', text: 'ls', cwd: project, draft: 'd1' });
      await until(() => held.entered(), 5_000);
      expect(await svc.setDraftPermissionMode('d1', 'auto', false, null, null)).toEqual({ mode: 'ask' });
      held.release();
      const a = await making;
      expect([a.conversation.permissionMode, svc.modeOf(a.conversation.id)]).toEqual(['ask', 'ask']);
      const page = new Page(svc, a.conversation.id);
      await until(() => page.cards().length === 1, 5_000);
      await svc.answer(a.conversation.id, page.cards()[0]!.id, { optionId: 'reject-once' });
      page.detach();

      await svc.setDraftPermissionMode('d2', 'bypass', true, project, null);
      const b = await svc.create({ mode: 'work', text: 'printf', cwd: project, draft: 'd2' });
      expect(b.conversation.permissionMode).toBe('bypass');
      expect(await svc.setDraftPermissionMode('d2', 'auto', false, null, null)).toEqual({ mode: 'auto' });
      expect(svc.modeOf(b.conversation.id)).toBe('auto');
      expect(await svc.setDraftPermissionMode('d2', 'ask', false, null, null)).toEqual({ mode: 'ask' });
      expect(svc.modeOf(b.conversation.id)).toBe('ask');
      await until(() => modeLines(fake, b.conversation.id).length === 3, 5_000);
      expect(modeLines(fake, b.conversation.id)).toEqual(['ask>bypass:user', 'bypass>auto:user', 'auto>ask:user']);
      await svc.cancel(b.conversation.id);
    });
  }, 30_000);

  it('a down on the sent draft after another pick of a draft still reaches its conversation', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      await svc.setDraftPermissionMode('d1', 'bypass', true, project, null);
      const held = holdFolderCheck(svc);
      const making = svc.create({ mode: 'work', text: 'ls', cwd: project, draft: 'd1' });
      await until(() => held.entered(), 5_000);
      await svc.setDraftPermissionMode('d2', 'auto', true, project, null);
      expect(await svc.setDraftPermissionMode('d1', 'ask', false, null, null)).toEqual({ mode: 'ask' });
      held.release();
      const made = await making;
      const id = made.conversation.id;
      expect([made.conversation.permissionMode, svc.modeOf(id)]).toEqual(['ask', 'ask']);
      expect((await svc.state()).permissionDraft).toEqual({ draft: 'd2', mode: 'auto' });
      const page = new Page(svc, id);
      await until(() => page.cards().length === 1, 5_000);
      await svc.answer(id, page.cards()[0]!.id, { optionId: 'reject-once' });
      page.detach();
    });
  }, 30_000);

  it('a down on the sent draft after the last window went still reaches its conversation', async () => {
    const fake = fakeService('work-permission', { mutate: SCRIPTS });
    try {
      const svc = fake.service;
      const project = join(fake.root, 'home', 'code', 'app');
      mkdirSync(project, { recursive: true });
      const window = hostState(svc);
      await svc.setDraftPermissionMode('d1', 'bypass', true, project, null);
      const held = holdFolderCheck(svc);
      const making = svc.create({ mode: 'work', text: 'ls', cwd: project, draft: 'd1' });
      await until(() => held.entered(), 5_000);
      window.off();
      expect(await svc.setDraftPermissionMode('d1', 'ask', false, null, null)).toEqual({ mode: 'ask' });
      held.release();
      const made = await making;
      const id = made.conversation.id;
      expect([made.conversation.permissionMode, svc.modeOf(id)]).toEqual(['ask', 'ask']);
      await svc.cancel(id);
    } finally {
      await fake.close();
    }
  }, 30_000);

  it('a turn that cannot start after the raise leaves the conversation in Ask', async () => {
    await withWork(async (fake, project) => {
      const svc = fake.service;
      await svc.setDraftPermissionMode('d1', 'bypass', true, project, null);
      let id = '';
      const target = svc as unknown as Hooked;
      const original = target.startTurn;
      target.startTurn = (conversation) => {
        id = conversation.info.id;
        throw routeError('chat', new Error('no turn'));
      };
      try {
        await apiError(svc.create({ mode: 'work', text: 'printf', cwd: project, draft: 'd1' }));
      } finally {
        target.startTurn = original;
      }
      expect(svc.modeOf(id)).toBe('ask');
      await until(() => modeLines(fake, id).length === 2, 5_000);
      expect(modeLines(fake, id)).toEqual(['ask>bypass:user', 'bypass>ask:user']);
    });
  }, 30_000);

  it('the last window closes: the pick goes; after a Control Panel restart nothing is picked', async () => {
    const fake = fakeService('work-permission', { mutate: SCRIPTS });
    try {
      const project = join(fake.root, 'home', 'code', 'app');
      mkdirSync(project, { recursive: true });
      const a = hostState(fake.service);
      const b = hostState(fake.service);
      await fake.service.setDraftPermissionMode('d1', 'bypass', true, project, null);
      a.off();
      expect((await fake.service.state()).permissionDraft).toEqual({ draft: 'd1', mode: 'bypass' });
      b.off();
      expect((await fake.service.state()).permissionDraft).toBeNull();
      const c = hostState(fake.service);
      await fake.service.setDraftPermissionMode('d2', 'bypass', true, project, null);
      await fake.service.close();
      const next = fake.restart();
      expect((await next.state()).permissionDraft).toBeNull();
      const made = await next.create({ mode: 'work', text: 'ls', cwd: project, draft: 'd2' });
      expect(made.conversation.permissionMode).toBe('ask');
      await next.cancel(made.conversation.id);
      c.off();
    } finally {
      await fake.close();
    }
  }, 30_000);

  it('one pick at a time: a second draft replaces the first, whose first message is Ask', async () => {
    await withWork(async (_fake, project) => {
      const svc = _fake.service;
      await svc.setDraftPermissionMode('d1', 'bypass', true, project, null);
      await svc.setDraftPermissionMode('d2', 'auto', true, project, null);
      expect((await svc.state()).permissionDraft).toEqual({ draft: 'd2', mode: 'auto' });
      const made = await svc.create({ mode: 'work', text: 'ls', cwd: project, draft: 'd1' });
      expect(made.conversation.permissionMode).toBe('ask');
      const page = new Page(svc, made.conversation.id);
      await until(() => page.cards().length === 1, 5_000);
      await svc.answer(made.conversation.id, page.cards()[0]!.id, { optionId: 'reject-once' });
      page.detach();
    });
  }, 30_000);
});

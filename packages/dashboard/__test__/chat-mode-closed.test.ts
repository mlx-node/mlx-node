/**
 * Nothing but the permission-mode route moves a Work conversation's mode up: not a field on another route, not prompt
 * text, not a file on disk (the Work index, the history file, an old-format file, the agent's own session files), not
 * anything the agent sends. Every case ends with a request that shows a card: the mode did not move. Through the real
 * runtime (worker thread included) on the scripted fake agent. Also: the old names are gone from the dashboard.
 */

import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import type { ApiResponse } from '../src/api/errors.js';
import { MemorySecretStore } from '../src/chat/providers/secrets.js';
import type {
  ChatHostState,
  ConversationEvent,
  ConversationInfo,
  Interaction,
  PermissionInteraction,
  PermissionMark,
  PermissionMode,
} from '../src/chat/types.js';
import { createDashboardRuntime, type DashboardRuntime } from '../src/runtime.js';
import { FAKE_AGENT, INFERENCE_TOKEN, readScenario, until, type Json } from './helpers/chat-fake.js';
import { startFakeEngine, type FakeEngine } from './helpers/fake-engine.js';

const PACKAGE = fileURLToPath(new URL('../', import.meta.url));
const SRC = join(PACKAGE, 'src');
const SELF = fileURLToPath(import.meta.url);

let base: string;
let engine: FakeEngine;
let runtime: DashboardRuntime | null = null;
let scenarioPath: string;

function scenario(steps: Json[]): void {
  const s: Json = { ...readScenario('work-permission'), recordTo: join(base, 'record.jsonl') };
  (s.prompts as Json[])[0]!.steps = [...steps, { text: 'Done.', chunk: 5 }, { end: { stopReason: 'end_turn' } }];
  scenarioPath = join(base, 'scenario.json');
  writeFileSync(scenarioPath, JSON.stringify(s));
}

function boot(): void {
  const home = join(base, 'home');
  runtime = createDashboardRuntime({
    dbPath: ':memory:',
    tracesDir: join(base, 'traces'),
    modelsDir: join(base, 'models'),
    runtimeDir: join(base, 'runtime'),
    supervisorPath: join(base, 'supervisor.json'),
    agentHome: join(base, 'agent-home'),
    chat: {
      root: join(base, 'chat'),
      homeDir: home,
      baseEnv: { PATH: process.env.PATH ?? '/usr/bin:/bin', HOME: home },
      agentCommand: { command: process.execPath, args: [FAKE_AGENT, scenarioPath] },
      secrets: new MemorySecretStore(),
      readEndpoint: async () => ({
        version: 1,
        pid: process.pid,
        url: engine.url,
        token: INFERENCE_TOKEN,
        models: [{ name: 'qwen3.5-4b', path: '/models/qwen3.5-4b' }],
      }),
      timings: { stopEofMs: 300, stopTermMs: 300 },
      flushMs: 5,
      stateCoalesceMs: 5,
    },
  });
}

const call = (method: string, path: string, body?: unknown): Promise<ApiResponse> =>
  runtime!.call({ method, path, ...(body === undefined ? {} : { body }) });

function code(response: ApiResponse): string {
  return response.ok ? 'ok' : response.code;
}

interface Watch {
  events: ConversationEvent[];
  off(): void;
}

function watch(id: string): Watch {
  const events: ConversationEvent[] = [];
  const off = runtime!.watchChat({ scope: 'conversation', conversationId: id }, (e) =>
    events.push(e as ConversationEvent),
  );
  return { events, off };
}

function watchHost(): { states: ChatHostState[]; off: () => void } {
  const states: ChatHostState[] = [];
  const off = runtime!.watchChat({ scope: 'host' }, (e) => {
    const event = e as { type: string; state?: ChatHostState };
    if (event.type === 'state' && event.state !== undefined) states.push(event.state);
  });
  return { states, off };
}

function cards(w: Watch): Interaction[] {
  const seen = new Map<string, Interaction>();
  for (const e of w.events) {
    const list = e.type === 'interaction' ? [e.interaction] : e.type === 'sync' ? e.interactions : [];
    for (const card of list) if (!seen.has(card.id)) seen.set(card.id, card);
  }
  return [...seen.values()];
}

function ends(w: Watch): number {
  return w.events.filter((e) => e.type === 'turn' && e.turn === null).length;
}

function modeOf(w: Watch): PermissionMode | undefined {
  let mode: PermissionMode | undefined;
  for (const e of w.events) if (e.type === 'sync' || e.type === 'conversation') mode = e.conversation.permissionMode;
  return mode;
}

function marks(w: Watch): PermissionMark[] {
  let out: PermissionMark[] = [];
  for (const e of w.events) {
    if (e.type === 'sync') out = [...e.permissionMarks];
    else if (e.type === 'permission-mark') out.push(e.mark);
  }
  return out;
}

async function state(): Promise<ChatHostState> {
  const response = await call('GET', '/api/chat/state');
  if (!response.ok) throw new Error(response.message);
  return response.body as ChatHostState;
}

const LS = {
  permission: {
    toolName: 'run_terminal_command',
    kind: 'execute',
    title: 'Execute `ls`',
    rawInput: { variant: 'Bash', command: 'ls', description: 'list', is_background: false },
    output: 'a.txt',
  },
};

/** A Work conversation whose first `ls` card the user allowed; the host stream is held, as by the window. */
async function workConversation(project: string): Promise<{ id: string; w: Watch }> {
  const created = await call('POST', '/api/chat/conversations', { mode: 'work', text: 'ls', cwd: project });
  if (!created.ok) throw new Error(created.message);
  const id = (created.body as { conversation: ConversationInfo }).conversation.id;
  const w = watch(id);
  await until(() => cards(w).length === 1, 5_000);
  const answered = await call('POST', `/api/chat/conversations/${id}/interactions/${cards(w)[0]!.id}`, {
    optionId: 'allow-once',
  });
  expect(answered.ok).toBe(true);
  await until(() => ends(w) === 1, 5_000);
  return { id, w };
}

function project(): string {
  const path = join(base, 'home', 'code', 'app');
  mkdirSync(path, { recursive: true });
  return path;
}

beforeEach(async () => {
  base = mkdtempSync(join(tmpdir(), 'dash-mode-closed-'));
  engine = await startFakeEngine();
});

afterEach(async () => {
  await runtime?.close();
  runtime = null;
  await engine.close();
  rmSync(base, { recursive: true, force: true });
});

/** The next `ls` of the conversation is a card with no Auto reason, and the mode is Ask: nothing moved it. */
async function stillAsks(id: string, w: Watch): Promise<void> {
  const before = cards(w).length;
  const turns = ends(w);
  const prompt = await call('POST', `/api/chat/conversations/${id}/prompt`, { text: 'ls' });
  expect(prompt.ok).toBe(true);
  await until(() => cards(w).length === before + 1, 5_000);
  const card = cards(w)[before] as PermissionInteraction;
  expect([card.autoAsk, modeOf(w), (await state()).permissionRaised]).toEqual([undefined, 'ask', []]);
  await call('POST', `/api/chat/conversations/${id}/interactions/${card.id}`, { optionId: 'reject-once' });
  await until(() => ends(w) === turns + 1, 5_000);
}

describe('the permission-mode route', () => {
  it('takes exactly {mode} for a known Work conversation and answers {mode}; the old route is gone', async () => {
    scenario([LS]);
    boot();
    const host = watchHost();
    const { id, w } = await workConversation(project());
    const path = `/api/chat/conversations/${id}/permission-mode`;
    for (const body of [
      {},
      { mode: 'Bypass' },
      { mode: 'yolo' },
      { mode: true },
      { mode: ['bypass'] },
      { mode: 'bypass', on: true },
      { on: true },
      ['bypass'],
      'bypass',
      null,
    ]) {
      expect([body, code(await call('POST', path, body))]).toEqual([body, 'E_BAD_REQUEST']);
    }
    expect(code(await call('POST', `/api/chat/conversations/${id}/bypass`, { on: true }))).toBe('E_NOT_FOUND');
    expect(code(await call('POST', '/api/chat/conversations/no-such-id/permission-mode', { mode: 'bypass' }))).toBe(
      'E_NOT_FOUND',
    );
    expect(code(await call('POST', '/api/chat/conversations/bad%20id/permission-mode', { mode: 'auto' }))).toBe(
      'E_NOT_FOUND',
    );
    await stillAsks(id, w);
    const up = await call('POST', path, { mode: 'bypass' });
    expect(up.ok && up.body).toEqual({ mode: 'bypass' });
    expect((await state()).permissionRaised).toEqual([{ id, mode: 'bypass' }]);
    await until(() => modeOf(w) === 'bypass', 5_000);
    const down = await call('POST', path, { mode: 'ask' });
    expect(down.ok && down.body).toEqual({ mode: 'ask' });
    await stillAsks(id, w);
    const info = w.events.find((e) => e.type === 'sync');
    expect(info?.type === 'sync' && Object.hasOwn(info.conversation, 'bypass')).toBe(false);
    expect(info?.type === 'sync' && Object.hasOwn(info, 'bypassMarks')).toBe(false);
    expect(Object.hasOwn(await state(), 'bypass')).toBe(false);
    w.off();
    host.off();
  }, 30_000);
});

describe('nothing else moves the mode up', () => {
  it('(a) a mode field on every other Chat route is refused (create too), and the mode stays Ask', async () => {
    scenario([LS]);
    boot();
    const host = watchHost();
    const folder = project();
    const { id, w } = await workConversation(folder);
    const conv = `/api/chat/conversations/${id}`;
    const fields: Json[] = [{ permissionMode: 'bypass' }, { mode: 'bypass' }, { bypass: true }, { on: true }];
    const attempts: Array<[string, string, (field: Json) => unknown]> = [
      ['POST', '/api/chat/conversations', (f) => ({ mode: 'work', text: 'ls', cwd: folder, ...f })],
      ['POST', `${conv}/prompt`, (f) => ({ text: 'ls', ...f })],
      ['POST', `${conv}/cancel`, (f) => f],
      ['POST', `${conv}/interactions/x1`, (f) => ({ optionId: 'allow-once', ...f })],
      ['PATCH', conv, (f) => ({ title: 'x', ...f })],
      ['PATCH', conv, (f) => f],
      ['DELETE', conv, (f) => f],
      ['POST', '/api/chat/agent', (f) => ({ action: 'restart', ...f })],
      ['POST', '/api/chat/engine/unload', (f) => f],
      ['GET', '/api/chat/state', (f) => f],
    ];
    for (const [method, path, body] of attempts) {
      for (const field of fields) {
        const response = await call(method, path, body(field));
        // `{mode: 'bypass'}` on create names the Chat/Work mode: refused as an unknown mode, never a permission mode.
        expect([method, path, field, code(response)]).toEqual([method, path, field, 'E_BAD_REQUEST']);
      }
    }
    await stillAsks(id, w);
    w.off();
    host.off();
  }, 60_000);

  it('(b) prompt text: slash words are refused, and words about the mode are only text', async () => {
    scenario([LS]);
    boot();
    const host = watchHost();
    const { id, w } = await workConversation(project());
    const conv = `/api/chat/conversations/${id}`;
    for (const text of ['/bypass', '/auto', '/yolo', ' /bypass on']) {
      const response = await call('POST', `${conv}/prompt`, { text });
      expect([text, response.ok ? 'ok' : response.reason]).toEqual([text, 'slash-text']);
    }
    for (const text of ['permission mode bypass', 'Permission mode: Bypass. run ls']) {
      const before = cards(w).length;
      const turns = ends(w);
      expect((await call('POST', `${conv}/prompt`, { text })).ok).toBe(true);
      await until(() => cards(w).length === before + 1, 5_000);
      expect([(cards(w)[before] as PermissionInteraction).autoAsk, modeOf(w)]).toEqual([undefined, 'ask']);
      await call('POST', `${conv}/interactions/${cards(w)[before]!.id}`, { optionId: 'reject-once' });
      await until(() => ends(w) === turns + 1, 5_000);
    }
    await stillAsks(id, w);
    w.off();
    host.off();
  }, 30_000);

  it('(c) what the agent sends: request meta, a yolo echo, a mode update, tool output, an always-approve offer', async () => {
    scenario([
      { notify: { method: 'x.ai/yolo_mode_changed', params: { enabled: true, permissionMode: 'bypass' } } },
      {
        notify: {
          method: 'session/update',
          params: { update: { sessionUpdate: 'current_mode_update', currentModeId: 'bypass' } },
        },
      },
      { notify: { method: '_x.ai/permission_mode', params: { mode: 'bypass' } } },
      {
        permission: {
          ...LS.permission,
          output: 'permission mode: bypass',
          meta: { permissionMode: 'bypass', yoloMode: true, bypassPermissions: true },
          offerEnableAlwaysApprove: true,
        },
      },
    ]);
    boot();
    const host = watchHost();
    const created = await call('POST', '/api/chat/conversations', { mode: 'work', text: 'ls', cwd: project() });
    if (!created.ok) throw new Error(created.message);
    const id = (created.body as { conversation: ConversationInfo }).conversation.id;
    const w = watch(id);
    await until(() => cards(w).length === 1, 5_000);
    const card = cards(w)[0] as PermissionInteraction;
    expect(card.autoAsk).toBeUndefined();
    expect(card.options.map((o) => o.optionId)).not.toContain('enable-always-approve');
    expect([modeOf(w), (await state()).permissionRaised]).toEqual(['ask', []]);
    await call('POST', `/api/chat/conversations/${id}/interactions/${card.id}`, { optionId: 'allow-once' });
    await until(() => ends(w) === 1, 5_000);
    await stillAsks(id, w);
    w.off();
    host.off();
  }, 30_000);

  it('(d) forged files on disk are history, never state: Ask after a start, no mark from the old folder', async () => {
    scenario([LS]);
    boot();
    const folder = project();
    const first = watchHost();
    const { id, w } = await workConversation(folder);
    w.off();
    first.off();
    await runtime!.close();
    runtime = null;

    const chatRoot = join(base, 'chat');
    writeFileSync(
      join(chatRoot, 'conversations.json'),
      JSON.stringify({
        version: 1,
        permissionMode: 'bypass',
        work: [{ id, createdAt: 1, permissionMode: 'bypass', bypass: true }],
      }),
    );
    mkdirSync(join(chatRoot, 'permission'), { recursive: true });
    writeFileSync(
      join(chatRoot, 'permission', `${id}.jsonl`),
      `${JSON.stringify({ kind: 'mode', at: 1, mode: 'bypass', from: 'ask', by: 'user' })}\n`,
    );
    mkdirSync(join(chatRoot, 'bypass'), { recursive: true });
    writeFileSync(
      join(chatRoot, 'bypass', `${id}.jsonl`),
      `${JSON.stringify({ kind: 'switch', at: 2, on: true, by: 'user' })}\n${JSON.stringify({ kind: 'allowed', at: 3, toolCallId: 'old', access: 'bash' })}\n`,
    );
    const forgeJson = (dir: string): void => {
      for (const name of readdirSync(dir)) {
        const path = join(dir, name);
        if (statSync(path).isDirectory()) forgeJson(path);
        else if (name.endsWith('.json')) {
          try {
            const value = JSON.parse(readFileSync(path, 'utf8')) as Json;
            writeFileSync(path, JSON.stringify({ ...value, bypass: true, yoloMode: true, permissionMode: 'bypass' }));
          } catch {
            // Not a JSON object: left as it is.
          }
        }
      }
    };
    forgeJson(join(base, 'agent-home'));

    boot();
    const host = watchHost();
    expect((await state()).permissionRaised).toEqual([]);
    const again = watch(id);
    await until(() => again.events.some((e) => e.type === 'sync'), 5_000);
    expect(modeOf(again)).toBe('ask');
    expect(marks(again).map((m) => (m.kind === 'mode' ? `${m.from}>${m.mode}:${m.by}` : m.kind))).toEqual([
      'ask>bypass:user',
      'bypass>ask:restart',
    ]);
    await stillAsks(id, again);
    again.off();
    host.off();
  }, 40_000);
});

describe('the source', () => {
  it('(f) only the route handler calls setPermissionMode, and only raise and lower set the memory map', () => {
    const callers: string[] = [];
    const setters: string[] = [];
    const walk = (dir: string): void => {
      for (const name of readdirSync(dir)) {
        const path = join(dir, name);
        if (statSync(path).isDirectory()) walk(path);
        else if (name.endsWith('.ts')) {
          const text = readFileSync(path, 'utf8');
          const file = relative(SRC, path);
          for (
            let at = text.indexOf('setPermissionMode(');
            at !== -1;
            at = text.indexOf('setPermissionMode(', at + 1)
          ) {
            const lineStart = text.lastIndexOf('\n', at) + 1;
            const line = text.slice(lineStart, text.indexOf('\n', at)).trim();
            if (!line.startsWith('setPermissionMode(id: string')) callers.push(file);
          }
          for (
            let at = text.indexOf('permissionModes.set(');
            at !== -1;
            at = text.indexOf('permissionModes.set(', at + 1)
          ) {
            const method = text.lastIndexOf('\n  private ', at);
            const head = text.slice(method + 11, text.indexOf('(', method));
            setters.push(`${file} ${head}`);
          }
        }
      }
    };
    walk(SRC);
    expect(callers).toEqual(['api/handlers/chat.ts']);
    expect(setters).toEqual(['chat/service.ts async raise', 'chat/service.ts lower']);
  });
});

const isUpper = (c: string): boolean => c >= 'A' && c <= 'Z';
const isLower = (c: string): boolean => c >= 'a' && c <= 'z';
const isLetter = (c: string): boolean => isUpper(c) || isLower(c);

/** Old spellings that are exact text. */
const OLD_TEXT = [
  'BYPASS_',
  'styles.bypass',
  'info.bypass',
  'host.bypass',
  'host?.bypass',
  'item.bypass',
  'bypass: false',
  'bypass: true',
  'bypass: []',
  'bypass={',
  'bypass?:',
  '.bypass {',
  'chat-bypass',
  'chat:bypass',
  'mlx-bypass',
  'dash-bypass',
];
/** Old phrases, each ending at a non-letter. */
const OLD_PHRASES = ['Bypass permissions', 'Bypass on', 'Bypass off', 'Bypass marks'];

/** The old names in a text, by hand-written rules (no regex). */
function oldNames(text: string): string[] {
  const hits: string[] = [];
  for (let i = text.indexOf('ypass'); i !== -1; i = text.indexOf('ypass', i + 1)) {
    const start = i - 1;
    const head = text[start];
    if (head !== 'b' && head !== 'B') continue;
    const before = text[start - 1] ?? '';
    const after = text[i + 5] ?? '';
    const near = text.slice(Math.max(0, start - 12), i + 12);
    if (isUpper(after) && text.slice(start, start + 9) !== 'bypassCSP') hits.push(near);
    else if (head === 'B' && isLower(before)) hits.push(near);
    else if (after === '-' && isLetter(text[i + 6] ?? '')) hits.push(near);
    else if (before === '/' && ["'", '"', '`', '/', ' '].includes(after)) hits.push(near);
    else if (head === 'b' && after === ' ' && !isLetter(before) && text.slice(start - 11, start) !== 'Allowed by ') {
      hits.push(near);
    }
  }
  for (const old of OLD_TEXT) {
    for (let at = text.indexOf(old); at !== -1; at = text.indexOf(old, at + 1)) hits.push(old);
  }
  for (const old of OLD_PHRASES) {
    for (let at = text.indexOf(old); at !== -1; at = text.indexOf(old, at + 1)) {
      if (!isLetter(text[at + old.length] ?? '')) hits.push(old);
    }
  }
  return hits;
}

const SCANNED = ['.ts', '.tsx', '.css', '.json', '.mjs'];
const ROOTS = [SRC, join(PACKAGE, 'ui', 'src'), join(PACKAGE, 'scripts'), join(PACKAGE, '__test__')];
const SKIP = new Set([SELF]);
const WEB = join(PACKAGE, 'web');

function scan(dir: string, out: Array<[string, string[]]>): void {
  for (const name of readdirSync(dir)) {
    if (name === 'node_modules' || name === 'dist' || name === 'web' || name.endsWith('.tsbuildinfo')) continue;
    const path = join(dir, name);
    if (statSync(path).isDirectory()) scan(path, out);
    else if (!SKIP.has(path) && SCANNED.some((ext) => name.endsWith(ext))) {
      const hits = oldNames(readFileSync(path, 'utf8'));
      if (hits.length > 0) out.push([relative(PACKAGE, path), hits]);
    }
  }
}

describe('(g) the old names are gone', () => {
  it('the rules catch each old spelling and pass the new ones', () => {
    const old = [
      'bypassAsk',
      'bypassAskText',
      'BypassAsk',
      'BypassMark',
      'bypassMarks',
      'decideBypass(',
      'bypassFacts',
      'BypassScan',
      'BypassLog',
      'BypassLine',
      'setBypass(',
      'switchBypass(',
      'bypassInput',
      'handleChatBypass,',
      'onBypassAllowed',
      'onBypassMarked',
      'bypassOn',
      'BypassChip',
      'BypassChipProps',
      'BypassDialog',
      'BypassDialogProps',
      'useBypassSwitch',
      'bypassShown',
      'bypassBlocked',
      'bypassOf',
      'bypassPermissions',
      './bypass-chip',
      './bypass-scan.js',
      './bypass-log.js',
      '`bypass-end-${n}`',
      "'./chat-bypass'",
      'mlx-bypass-',
      'dash-bypass-',
      "'chat:bypass-off'",
      "'chat:bypass'",
      'data-bypass-chip',
      'fixtures/bypass/x.json',
      "'/bypass'",
      '/conversations/:id/bypass ',
      'with bypass on',
      'Bypass permissions',
      'Bypass on.',
      'Bypass off',
      'Bypass marks',
      'BYPASS_PER_TURN_MAX',
      'info.bypass',
      'bypass: false,',
      '.bypass {',
    ];
    for (const name of old) expect([name, oldNames(` ${name} `).length > 0]).toEqual([name, true]);
    const fresh = [
      "mode: 'bypass'",
      'Allowed by bypass · exit 0',
      "'chat:permission-bypass'",
      'data-permission-mode="bypass"',
      'bypassCSP',
      'Permission mode: Bypass. Commands',
      'Bypass only answers',
      '`ask`, `auto`, `bypass`.',
      'disable_bypass_permissions_mode',
    ];
    for (const text of fresh) expect([text, oldNames(text)]).toEqual([text, []]);
  });

  it('no old name in the dashboard server, scripts, tests and docs; the old files are gone', () => {
    const out: Array<[string, string[]]> = [];
    for (const root of ROOTS) scan(root, out);
    const docs = oldNames(readFileSync(join(PACKAGE, '..', '..', 'docs', 'dashboard.md'), 'utf8'));
    if (docs.length > 0) out.push(['docs/dashboard.md', docs]);
    expect(out).toEqual([]);
    for (const gone of [
      'src/chat/agent/bypass-scan.ts',
      'src/chat/agent/bypass.ts',
      'src/chat/bypass-log.ts',
      '__test__/fixtures/bypass',
      'ui/src/pages/chat/work/bypass-chip.tsx',
      'ui/src/pages/chat/work/bypass-chip.module.css',
      'ui/src/components/chat-bypass.tsx',
      'ui/src/components/chat-bypass.module.css',
    ]) {
      expect([gone, existsSync(join(PACKAGE, gone))]).toEqual([gone, false]);
    }
  });

  // `web/` is built (not in git): `pack.test.ts` builds it; on a checkout with no build there is nothing to read.
  it.skipIf(!existsSync(join(WEB, 'index.html')))(
    'the built page holds the new route and texts, and no old one',
    () => {
      const scripts = readdirSync(join(WEB, 'assets')).filter(
        (name) => name.startsWith('index-') && name.endsWith('.js'),
      );
      expect(scripts.length).toBeGreaterThan(0);
      const text = scripts.map((name) => readFileSync(join(WEB, 'assets', name), 'utf8')).join('\n');
      expect(text).toContain('/permission-mode');
      expect(text).toContain('Switch to Bypass in this conversation?');
      expect(text).toContain('Allowed by bypass');
      for (const old of ['/bypass`', "/bypass'", '/bypass"', 'Bypass permissions', 'Bypass on', 'Bypass off']) {
        expect([old, text.includes(old)]).toEqual([old, false]);
      }
    },
  );
});

/**
 * The Sessions API end to end: a real runtime (worker thread included) over a
 * temp copy of the grok-session fixtures. Never points at the real ~/.mlx-agent.
 */

import {
  cpSync,
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  rmSync,
  symlinkSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { MessageChannel } from 'node:worker_threads';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { containedSessionDir, SessionWriteError, validateRenameTitle } from '../src/agent-sessions/writes.js';
import { createRpcClient } from '../src/rpc/client.js';
import { serveRuntimeOverPort } from '../src/rpc/host.js';
import { bindEventTargetPort } from '../src/rpc/port.js';
import { createDashboardRuntime, type DashboardRuntime } from '../src/runtime.js';
import { createTestClient, type TestClient } from './helpers/api-client.js';

const FIXTURE_HOME = fileURLToPath(new URL('./fixtures/grok-sessions/home', import.meta.url));
const DAY = 24 * 60 * 60 * 1000;

const id = (n: number) => `01a20000-0000-7000-8000-${n.toString(16).padStart(12, '0')}`;
const P = id(0x0a);
const C1 = id(0x0b);
const C2 = id(0x0c);
const G = id(0x0d);
const F = id(0x10);
const M = id(0x11);
const D = id(0x12);
const GONE = id(0x19);
const PROJ = '%2FUsers%2Fdev%2Fproj';
const WORKTREE = '%2FUsers%2Fdev%2F.mlx-agent%2Fworktrees%2Fproj%2Fsubagent-01a20000-0000-7000-8000-00000000000c';

let base: string;
let home: string;
let runtime: DashboardRuntime;
let api: TestClient;

function setMtimes(dir: string, when: number): void {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    const st = lstatSync(path);
    if (st.isSymbolicLink()) continue;
    if (st.isDirectory()) setMtimes(path, when);
    utimesSync(path, new Date(when), new Date(when));
  }
}

function sessionDir(cwdFolder: string, sid: string): string {
  return join(home, 'sessions', cwdFolder, sid);
}

function readSummary(dir: string): Record<string, unknown> {
  return JSON.parse(readFileSync(join(dir, 'summary.json'), 'utf8')) as Record<string, unknown>;
}

function register(sessionId: string, pid = process.pid): void {
  writeFileSync(
    join(home, 'active_sessions.json'),
    JSON.stringify([{ session_id: sessionId, pid, cwd: '/Users/dev/proj', opened_at: '2026-10-29T03:30:16.660505Z' }]),
  );
}

function touchNow(path: string): void {
  const now = new Date();
  utimesSync(path, now, now);
}

async function json<T>(path: string, init?: Parameters<TestClient['fetch']>[1]): Promise<{ status: number; body: T }> {
  const res = await api.fetch(path, init);
  return { status: res.status, body: (await res.json()) as T };
}

const rename = (sid: string, title: unknown) =>
  json<{ id: string; title: string; error?: string }>(`/api/sessions/${encodeURIComponent(sid)}`, {
    method: 'PATCH',
    body: JSON.stringify({ title }),
  });

beforeEach(() => {
  base = mkdtempSync(join(tmpdir(), 'dash-sessions-'));
  home = join(base, 'agent-home');
  cpSync(FIXTURE_HOME, home, { recursive: true });
  setMtimes(home, Date.now() - DAY);
  runtime = createDashboardRuntime({
    dbPath: ':memory:',
    tracesDir: join(base, 'traces'),
    modelsDir: join(base, 'models'),
    agentHome: home,
  });
  api = createTestClient(runtime);
});

afterEach(async () => {
  await runtime.close();
  rmSync(base, { recursive: true, force: true });
});

describe('GET /api/sessions', () => {
  it('lists top-level sessions newest first with the cwd facet', async () => {
    const { status, body } = await json<{ sessions: { id: string; live: boolean }[]; total: number; cwds: string[] }>(
      '/api/sessions',
    );
    expect(status).toBe(200);
    expect(body.sessions.map((s) => s.id)).toEqual([M, P, F]);
    expect(body.sessions.every((s) => !s.live)).toBe(true);
    expect(body.total).toBe(3);
    expect(body.cwds).toEqual(['/Users/dev/other', '/Users/dev/proj']);
  });

  it('filters by q and exact cwd, and pages with limit/offset', async () => {
    const ids = async (query: string) =>
      (await json<{ sessions: { id: string }[]; total: number }>(`/api/sessions?${query}`)).body;
    expect((await ids('q=RENAMED')).sessions.map((s) => s.id)).toEqual([M]);
    expect((await ids('q=dev%2Fother')).sessions.map((s) => s.id)).toEqual([M]);
    expect((await ids(`cwd=${encodeURIComponent('/Users/dev/proj')}`)).sessions.map((s) => s.id)).toEqual([P, F]);
    expect((await ids('cwd=%2FUsers%2Fdev')).total).toBe(0);
    const page = await ids('limit=1&offset=1');
    expect(page.sessions.map((s) => s.id)).toEqual([P]);
    expect(page.total).toBe(3);
    expect((await ids('limit=nope&offset=-4')).sessions.map((s) => s.id)).toEqual([M, P, F]);
  });

  it('shows a session whose TUI pid is alive as live', async () => {
    register(P);
    const { body } = await json<{ sessions: { id: string; live: boolean }[] }>('/api/sessions');
    expect(body.sessions.find((s) => s.id === P)!.live).toBe(true);
    expect(body.sessions.find((s) => s.id === M)!.live).toBe(false);
  });
});

describe('GET /api/sessions/:id', () => {
  it('returns the parent with its subagent children and a resume command', async () => {
    const { status, body } = await json<{
      session: { id: string; subagents: { id: string; exists: boolean }[]; parent: unknown };
      transcript: unknown[];
      turns: unknown[];
      resumeCommand: string;
    }>(`/api/sessions/${P}`);
    expect(status).toBe(200);
    expect(body.session.subagents.map((s) => [s.id, s.exists])).toEqual([
      [C1, true],
      [C2, true],
      [GONE, false],
    ]);
    expect(body.session.parent).toBeNull();
    expect(body.transcript.length).toBeGreaterThan(0);
    expect(body.turns).toHaveLength(2);
    // The temp home is not the default one, so the command names it.
    expect(body.resumeCommand).toBe(`cd '/Users/dev/proj' && GROK_HOME='${home}' mlx agent --resume ${P}`);
  });

  it('serves hidden subagent children with a link back to the parent', async () => {
    const child = await json<{ session: { kind: string; parent: { id: string; title: string } } }>(
      `/api/sessions/${C1}`,
    );
    expect(child.status).toBe(200);
    expect(child.body.session.kind).toBe('subagent');
    expect(child.body.session.parent).toEqual({ id: P, title: 'Research the project layout' });
    const worktreeChild = await json<{ session: { parent: { id: string } } }>(`/api/sessions/${C2}`);
    expect(worktreeChild.body.session.parent.id).toBe(P);
    const grandchild = await json<{ session: { parent: { id: string } } }>(`/api/sessions/${G}`);
    expect(grandchild.body.session.parent.id).toBe(C1);
  });

  it('404s unknown, ambiguous and traversal ids', async () => {
    for (const path of [`/api/sessions/${id(0x99)}`, `/api/sessions/${D}`, '/api/sessions/..', '/api/sessions/a%2Fb']) {
      expect((await api.fetch(path)).status).toBe(404);
    }
    expect((await api.fetch('/api/sessions/%zz')).status).toBe(400);
  });
});

describe('PATCH /api/sessions/:id', () => {
  it('sets the title like grok /rename and keeps every other field', async () => {
    const dir = sessionDir(PROJ, P);
    const before = readSummary(dir);
    const res = await rename(P, '  New\u0007 name‮  ');
    expect(res).toEqual({ status: 200, body: { id: P, title: 'New name' } });

    const after = readSummary(dir);
    expect(after.generated_title).toBe('New name');
    expect(after.title_is_manual).toBe(true);
    // Not empty before, so grok leaves it.
    expect(after.session_summary).toBe('Research the project');
    expect(after.updated_at).not.toBe(before.updated_at);
    expect(Date.parse(after.updated_at as string)).toBeGreaterThan(Date.now() - 60_000);
    const strip = (s: Record<string, unknown>) => {
      const { generated_title: _g, title_is_manual: _t, updated_at: _u, ...rest } = s;
      return rest;
    };
    expect(strip(after)).toEqual(strip(before));
    // serde `to_vec_pretty` layout, and no temp file left behind.
    expect(readFileSync(join(dir, 'summary.json'), 'utf8')).toBe(JSON.stringify(after, null, 2));
    expect(readFileSync(join(dir, 'title_refresh_idx'), 'utf8')).toBe('2');
    expect(readdirSync(dir).filter((n) => n.endsWith('.tmp'))).toEqual([]);

    const list = await json<{ sessions: { id: string; title: string; live: boolean }[] }>('/api/sessions');
    // Our own write does not make the session look busy.
    expect(list.body.sessions.find((s) => s.id === P)).toMatchObject({ title: 'New name', live: false });
  });

  it('mirrors the title into an empty session_summary', async () => {
    expect((await rename(F, 'Forked work')).status).toBe(200);
    expect(readSummary(sessionDir(PROJ, F))).toMatchObject({
      generated_title: 'Forked work',
      session_summary: 'Forked work',
      title_is_manual: true,
    });
  });

  it('allows a second rename and a delete right after our own rename', async () => {
    expect((await rename(P, 'First')).status).toBe(200);
    expect((await rename(P, 'Second')).status).toBe(200);
    expect(readSummary(sessionDir(PROJ, P)).generated_title).toBe('Second');
    expect((await api.fetch(`/api/sessions/${P}`, { method: 'DELETE' })).status).toBe(200);
  });

  it('409s while the session or an ancestor is open in a live TUI', async () => {
    register(P);
    const original = readFileSync(join(sessionDir(PROJ, P), 'summary.json'), 'utf8');
    const res = await rename(P, 'Nope');
    expect(res.status).toBe(409);
    expect(res.body.error).toMatch(/open in a running mlx agent/);
    expect(readFileSync(join(sessionDir(PROJ, P), 'summary.json'), 'utf8')).toBe(original);
    const child = await rename(C1, 'Nope');
    expect(child.status).toBe(409);
    expect(child.body.error).toContain(`via parent ${P}`);
  });

  it('ignores a registry entry whose pid is dead', async () => {
    register(P, 2 ** 31 - 2);
    expect((await rename(P, 'Fine')).status).toBe(200);
  });

  it('409s when any session file was written in the last 30 s', async () => {
    touchNow(join(sessionDir(PROJ, P), 'updates.jsonl'));
    const res = await rename(P, 'Nope');
    expect(res.status).toBe(409);
    expect(res.body.error).toMatch(/last 30 s/);
    expect(readSummary(sessionDir(PROJ, P)).generated_title).toBe('Research the project layout');
  });

  it('409s when active_sessions.json cannot be parsed', async () => {
    writeFileSync(join(home, 'active_sessions.json'), '{not json');
    expect((await rename(P, 'Nope')).status).toBe(409);
  });

  it('400s a bad body and 404s an unknown id first', async () => {
    const bad: [string, unknown][] = [
      ['missing title', {}],
      ['non-string', { title: 42 }],
      ['blank after stripping controls', { title: ' \u0007‎ ' }],
      ['101 scalars', { title: '😀'.repeat(101) }],
      ['over the byte guard', { title: 'a'.repeat(465) }],
    ];
    for (const [, body] of bad) {
      const res = await api.fetch(`/api/sessions/${P}`, { method: 'PATCH', body: JSON.stringify(body) });
      expect(res.status).toBe(400);
    }
    expect((await api.fetch(`/api/sessions/${P}`, { method: 'PATCH', rawBody: '{oops' })).status).toBe(400);
    expect((await api.fetch(`/api/sessions/${P}`, { method: 'PATCH' })).status).toBe(400);
    expect((await api.fetch(`/api/sessions/${id(0x99)}`, { method: 'PATCH', rawBody: '{oops' })).status).toBe(404);
    expect((await rename('..', 'x')).status).toBe(404);
    expect((await rename('a/b', 'x')).status).toBe(400);
    expect((await rename(P, '😀'.repeat(100))).status).toBe(200);
    expect(readSummary(sessionDir(PROJ, P)).generated_title).toBe('😀'.repeat(100));
  });
});

describe('DELETE /api/sessions/:id', () => {
  it('removes the session and its subagent children, across cwd folders, and nothing else', async () => {
    const res = await json<{ deleted: string[] }>(`/api/sessions/${P}`, { method: 'DELETE' });
    expect(res.status).toBe(200);
    expect(res.body.deleted[0]).toBe(P);
    expect([...res.body.deleted].sort()).toEqual([P, C1, C2, G].sort());
    for (const [folder, sid] of [
      [PROJ, P],
      [PROJ, C1],
      [WORKTREE, C2],
      [PROJ, G],
    ] as const) {
      expect(existsSync(sessionDir(folder, sid))).toBe(false);
    }
    // The fork keeps pointing at a parent that is gone, as grok leaves it.
    expect(existsSync(sessionDir(PROJ, F))).toBe(true);
    expect(existsSync(sessionDir('%2FUsers%2Fdev%2Fother', M))).toBe(true);
    expect(existsSync(join(home, 'sessions', PROJ, 'prompt_history.jsonl'))).toBe(true);
    expect(existsSync(join(home, 'sessions', 'session_search.sqlite'))).toBe(true);
    expect(existsSync(join(home, 'sessions', WORKTREE))).toBe(true);

    expect((await api.fetch(`/api/sessions/${P}`)).status).toBe(404);
    expect((await json<{ sessions: { id: string }[] }>('/api/sessions')).body.sessions.map((s) => s.id)).toEqual([
      M,
      F,
    ]);
    const fork = await json<{ session: { parent: unknown } }>(`/api/sessions/${F}`);
    expect(fork.body.session.parent).toEqual({ id: P, title: null });
    expect((await api.fetch(`/api/sessions/${P}`, { method: 'DELETE' })).status).toBe(404);
  });

  it('409s and removes nothing while the session or an ancestor is live', async () => {
    register(P);
    expect((await api.fetch(`/api/sessions/${P}`, { method: 'DELETE' })).status).toBe(409);
    expect((await api.fetch(`/api/sessions/${G}`, { method: 'DELETE' })).status).toBe(409);
    expect(existsSync(sessionDir(PROJ, P))).toBe(true);
    expect(existsSync(sessionDir(PROJ, G))).toBe(true);
  });

  it('409s the whole delete when one child was written in the last 30 s', async () => {
    touchNow(join(sessionDir(PROJ, G), 'summary.json'));
    const res = await json<{ error: string }>(`/api/sessions/${P}`, { method: 'DELETE' });
    expect(res.status).toBe(409);
    expect(res.body.error).toContain(G);
    for (const sid of [P, C1, G]) expect(existsSync(sessionDir(PROJ, sid))).toBe(true);
    expect(existsSync(sessionDir(WORKTREE, C2))).toBe(true);
  });

  it('rejects traversal ids and never follows symlinks out of the sessions folder', async () => {
    const outside = join(base, 'outside');
    mkdirSync(outside);
    writeFileSync(join(outside, 'keep.txt'), 'keep');

    // URL parsing folds `..` and `%2E%2E` away before routing; what reaches the
    // handler as one segment is refused by its shape.
    for (const path of ['/api/sessions/..', '/api/sessions/%2E%2E']) {
      expect((await api.fetch(path, { method: 'DELETE' })).status).toBe(404);
    }
    for (const path of ['/api/sessions/a%2Fb', `/api/sessions/.${P}`, `/api/sessions/${PROJ}%2F${P}`]) {
      expect((await api.fetch(path, { method: 'DELETE' })).status).toBe(400);
    }
    expect(existsSync(sessionDir(PROJ, P))).toBe(true);

    // A symlinked session folder is never indexed, so it cannot be named.
    const linkedId = id(0x40);
    const linkedTarget = join(base, 'linked-session');
    mkdirSync(linkedTarget);
    writeFileSync(
      join(linkedTarget, 'summary.json'),
      readFileSync(join(sessionDir(PROJ, P), 'summary.json'), 'utf8').replace(P, linkedId),
    );
    symlinkSync(linkedTarget, sessionDir(PROJ, linkedId));
    expect((await api.fetch(`/api/sessions/${linkedId}`, { method: 'DELETE' })).status).toBe(404);
    expect(existsSync(join(linkedTarget, 'summary.json'))).toBe(true);

    // A child folder that is a symlink is skipped; a symlink inside a session is
    // unlinked, not followed.
    const c2Real = join(base, 'c2-real');
    cpSync(sessionDir(WORKTREE, C2), c2Real, { recursive: true });
    rmSync(sessionDir(WORKTREE, C2), { recursive: true });
    symlinkSync(c2Real, sessionDir(WORKTREE, C2));
    symlinkSync(outside, join(sessionDir(PROJ, P), 'terminal'));
    setMtimes(home, Date.now() - DAY);

    const res = await json<{ deleted: string[] }>(`/api/sessions/${P}`, { method: 'DELETE' });
    expect(res.status).toBe(200);
    expect([...res.body.deleted].sort()).toEqual([P, C1, G].sort());
    expect(readFileSync(join(outside, 'keep.txt'), 'utf8')).toBe('keep');
    expect(existsSync(join(c2Real, 'summary.json'))).toBe(true);
    expect(lstatSync(sessionDir(WORKTREE, C2)).isSymbolicLink()).toBe(true);
  });
});

describe('Sessions API over the RPC port', () => {
  it('lists, renames and refuses over a real MessageChannel', async () => {
    const { port1, port2 } = new MessageChannel();
    const dispose = serveRuntimeOverPort(runtime, bindEventTargetPort(port2));
    const client = createRpcClient(bindEventTargetPort(port1), { onUnresponsive: () => port2.close() });
    const rpc = createTestClient(client);
    try {
      const list = await rpc.fetch('/api/sessions?limit=2');
      expect(list.status).toBe(200);
      expect(((await list.json()) as { sessions: { id: string }[] }).sessions.map((s) => s.id)).toEqual([M, P]);
      const renamed = await rpc.fetch(`/api/sessions/${M}`, {
        method: 'PATCH',
        body: JSON.stringify({ title: 'Via port' }),
      });
      expect(renamed.status).toBe(200);
      expect(await renamed.json()).toEqual({ id: M, title: 'Via port' });
      register(P);
      const refused = await rpc.fetch(`/api/sessions/${P}`, { method: 'DELETE' });
      expect(refused.status).toBe(409);
      expect(((await refused.json()) as { error: string }).error).toMatch(/running mlx agent/);
    } finally {
      client.close();
      dispose();
    }
  });
});

describe('write guards', () => {
  it('containedSessionDir refuses a session reached through a symlinked cwd folder', () => {
    const sessionsRoot = join(home, 'sessions');
    symlinkSync(join(sessionsRoot, PROJ), join(sessionsRoot, '%2Flinked'));
    expect(() => containedSessionDir(sessionsRoot, { id: P, dir: join(sessionsRoot, '%2Flinked', P) })).toThrow(
      SessionWriteError,
    );
    expect(() => containedSessionDir(sessionsRoot, { id: P, dir: `${sessionsRoot}/${PROJ}/../${PROJ}/${P}` })).toThrow(
      /direct <cwd>\/<id>/,
    );
    expect(() => containedSessionDir(join(sessionsRoot, PROJ), { id: P, dir: sessionDir(PROJ, P) })).toThrow(
      SessionWriteError,
    );
    expect(containedSessionDir(sessionsRoot, { id: P, dir: sessionDir(PROJ, P) })).toBe(sessionDir(PROJ, P));
  });

  it('validateRenameTitle strips like grok and trims Rust whitespace only', () => {
    expect(validateRenameTitle(' A\u0085 b‏　')).toBe('A b');
    // U+FEFF is not Rust whitespace, so it stays.
    expect(validateRenameTitle('﻿x')).toBe('﻿x');
    expect(() => validateRenameTitle('\uD800x')).toThrow(/Unicode/);
  });
});

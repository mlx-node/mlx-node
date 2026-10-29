import {
  appendFileSync,
  cpSync,
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

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { isPidAlive, newestFileMtimeMs, readActiveRegistry } from '../src/agent-sessions/liveness.js';
import { AgentSessionsReader, type AgentSessionsReaderOptions, resumeCommand } from '../src/agent-sessions/reader.js';

const FIXTURE_HOME = fileURLToPath(new URL('./fixtures/grok-sessions/home', import.meta.url));
const NOW = Date.parse('2026-10-29T12:00:00Z');
const DAY = 24 * 60 * 60 * 1000;
const LIVE_PID = 424242;

const id = (n: number) => `01a20000-0000-7000-8000-${n.toString(16).padStart(12, '0')}`;
const P = id(0x0a);
const C1 = id(0x0b);
const C2 = id(0x0c);
const G = id(0x0d);
const H = id(0x0e);
const K = id(0x0f);
const F = id(0x10);
const M = id(0x11);
const D = id(0x12);
const B = id(0x13);
const X = id(0x14);
const O = id(0x16);
const R = id(0x17);
const GONE = id(0x19);
const PROJ = '%2FUsers%2Fdev%2Fproj';

let home: string;

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

function reader(extra: Partial<AgentSessionsReaderOptions> = {}): AgentSessionsReader {
  return new AgentSessionsReader({
    agentHome: home,
    defaultAgentHome: home,
    now: () => NOW,
    isPidAlive: (pid) => pid === LIVE_PID,
    ...extra,
  });
}

function register(entries: { session_id: string; pid: number }[]): void {
  writeFileSync(
    join(home, 'active_sessions.json'),
    JSON.stringify(
      entries.map((e) => ({ ...e, cwd: '/Users/dev/proj', opened_at: '2026-10-29T03:30:16.660505Z' })),
      null,
      2,
    ),
  );
}

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), 'grok-sessions-'));
  cpSync(FIXTURE_HOME, home, { recursive: true });
  setMtimes(home, NOW - DAY);
});

afterEach(() => {
  rmSync(home, { recursive: true, force: true });
});

describe('AgentSessionsReader.list', () => {
  it('lists only top-level sessions, newest first with microsecond tiebreak', () => {
    const result = reader().list();
    expect(result.sessions.map((s) => s.id)).toEqual([M, P, F]);
    expect(result.total).toBe(3);
    expect(result.cwds).toEqual(['/Users/dev/other', '/Users/dev/proj']);
  });

  it('excludes hidden subagents, headless runs, husks, ambiguous ids, bad and mismatched summaries', () => {
    const r = reader();
    const listed = new Set(r.list({ limit: 500 }).sessions.map((s) => s.id));
    for (const hidden of [C1, C2, G, H, K, D, B, X, O, R]) expect(listed.has(hidden)).toBe(false);
    const index = r.scan();
    expect([...index.ambiguous]).toEqual([D]);
    expect(index.sessions.has(D)).toBe(false);
    expect(index.sessions.has(B)).toBe(false);
    expect(index.sessions.has(X)).toBe(false);
    expect(index.sessions.has(id(0x15))).toBe(false);
    expect(index.sessions.has(id(0x18))).toBe(false);
    // Hidden and headless sessions are still indexed for detail views.
    for (const present of [C1, C2, G, H, K, O, R]) expect(index.sessions.has(present)).toBe(true);
  });

  it('builds rows from summary fields and transcript token totals', () => {
    const rows = reader().list().sessions;
    const p = rows.find((s) => s.id === P)!;
    expect(p).toEqual({
      id: P,
      title: 'Research the project layout',
      cwd: '/Users/dev/proj',
      model: 'Qwen3.8-27B-UD-Q4_K_M',
      branch: 'main',
      createdAt: '2026-10-10T14:41:17.006303Z',
      lastActiveAt: '2026-10-20T10:00:00.000001Z',
      messageCount: 12,
      live: false,
      kind: null,
      parentId: null,
      subagentCount: 3,
      tokens: { input: 38723, output: 104, cached: 28182 },
    });
    const m = rows.find((s) => s.id === M)!;
    expect(m.title).toBe('Renamed session');
    expect(m.tokens).toBeNull();
    const f = rows.find((s) => s.id === F)!;
    // A fork with no messages and no title is not a husk; it points at its source.
    expect(f).toMatchObject({ kind: 'fork', parentId: P, title: '', lastActiveAt: '2026-10-19T00:00:00Z' });
  });

  it('filters by q over title and cwd, by exact cwd, and paginates', () => {
    const r = reader();
    expect(r.list({ q: 'RENAMED' }).sessions.map((s) => s.id)).toEqual([M]);
    expect(r.list({ q: 'dev/other' }).sessions.map((s) => s.id)).toEqual([M]);
    expect(r.list({ cwd: '/Users/dev/proj' }).sessions.map((s) => s.id)).toEqual([P, F]);
    expect(r.list({ cwd: '/Users/dev' }).total).toBe(0);
    const page = r.list({ limit: 1, offset: 1 });
    expect(page.sessions.map((s) => s.id)).toEqual([P]);
    expect(page.total).toBe(3);
    expect(r.list({ limit: 0 }).sessions).toHaveLength(1);
    expect(r.list({ limit: 10_000 }).sessions).toHaveLength(3);
  });

  it('skips symlinked cwd folders, symlinked session dirs, symlinked summaries and dot dirs', () => {
    const sessions = join(home, 'sessions');
    symlinkSync(join(sessions, PROJ), join(sessions, '%2Flinked'));
    symlinkSync(sessionDir(PROJ, P), join(sessions, '%2Ftmp%2Fscratch', id(0x30)));
    const viaLinkedSummary = sessionDir('%2Ftmp%2Fscratch', id(0x31));
    mkdirSync(viaLinkedSummary);
    symlinkSync(join(sessionDir(PROJ, P), 'summary.json'), join(viaLinkedSummary, 'summary.json'));
    const dot = join(sessions, PROJ, `.${id(0x32)}`);
    mkdirSync(dot);
    writeFileSync(
      join(dot, 'summary.json'),
      readFileSync(join(sessionDir(PROJ, P), 'summary.json'), 'utf8').replace(P, `.${id(0x32)}`),
    );

    const r = reader();
    const index = r.scan();
    // Following the cwd link would have made every proj session ambiguous.
    expect([...index.ambiguous]).toEqual([D]);
    for (const skipped of [id(0x30), id(0x31), `.${id(0x32)}`]) expect(index.sessions.has(skipped)).toBe(false);
    expect(r.list().sessions.map((s) => s.id)).toEqual([M, P, F]);
  });

  it('reuses parsed files while (mtime, size) is unchanged and re-reads when either moves', () => {
    const r = reader();
    expect(r.list().sessions.find((s) => s.id === P)!.title).toBe('Research the project layout');
    const summaryPath = join(sessionDir(PROJ, P), 'summary.json');
    const original = readFileSync(summaryPath, 'utf8');
    // Same size, same mtime: the cached parse must be served.
    writeFileSync(summaryPath, original.replace('Research the project layout', 'Research the project LAYOUT'));
    utimesSync(summaryPath, new Date(NOW - DAY), new Date(NOW - DAY));
    expect(r.list().sessions.find((s) => s.id === P)!.title).toBe('Research the project layout');
    utimesSync(summaryPath, new Date(NOW - DAY + 1000), new Date(NOW - DAY + 1000));
    expect(r.list().sessions.find((s) => s.id === P)!.title).toBe('Research the project LAYOUT');

    const updates = join(sessionDir(PROJ, P), 'updates.jsonl');
    appendFileSync(
      updates,
      `${JSON.stringify({
        timestamp: 1791644100,
        method: '_x.ai/session/update',
        params: {
          sessionId: P,
          update: {
            sessionUpdate: 'turn_completed',
            prompt_id: 'p3',
            stop_reason: 'end_turn',
            usage: { inputTokens: 7, outputTokens: 3, cachedReadTokens: 1 },
          },
          _meta: { eventId: `${P}-99`, agentTimestampMs: 1791644100000 },
        },
      })}\n`,
    );
    setMtimes(sessionDir(PROJ, P), NOW - DAY);
    expect(r.list().sessions.find((s) => s.id === P)!.tokens).toEqual({ input: 38730, output: 107, cached: 28183 });
  });

  it('returns an empty list when the sessions root does not exist', () => {
    rmSync(join(home, 'sessions'), { recursive: true });
    expect(reader().list()).toEqual({ sessions: [], total: 0, cwds: [] });
  });
});

describe('AgentSessionsReader.detail', () => {
  it('links subagent children from meta.json, across cwd folders', () => {
    const detail = reader().detail(P)!;
    expect(detail.session.subagents).toEqual([
      {
        id: C1,
        title: 'Trace the decode loop',
        type: 'general-purpose',
        description: 'Trace the decode loop',
        status: 'completed',
        model: 'Qwen3.8-27B-UD-Q4_K_M',
        exists: true,
      },
      {
        id: C2,
        title: 'Trace paged attention',
        type: 'codex:codex-rescue',
        description: 'Trace paged attention',
        status: 'running',
        model: 'Qwen3.8-27B-UD-Q4_K_M',
        exists: true,
      },
      {
        id: GONE,
        title: 'Deleted child',
        type: 'general-purpose',
        description: 'Deleted child',
        status: 'failed',
        model: 'Qwen3.8-27B-UD-Q4_K_M',
        exists: false,
      },
    ]);
    expect(detail.session).toMatchObject({
      gitRoot: '/Users/dev/proj/',
      headCommit: 'e5282a8dccf1636e62b5df12e79a7b18f63d5af1',
      remotes: ['git@example.com:dev/proj.git'],
      agentName: 'grok-build-plan',
      reasoningEffort: null,
      lastTurnSummary: 'Planned the work',
      parent: null,
    });
    expect(detail.transcript.length).toBeGreaterThan(0);
    expect(detail.turns).toHaveLength(2);
    expect(detail.transcriptError).toBeUndefined();
  });

  it('serves hidden children with a parent link; resumed subagents do not claim their source as parent', () => {
    const r = reader();
    expect(r.detail(C1)!.session).toMatchObject({
      kind: 'subagent',
      parentId: P,
      parent: { id: P, title: 'Research the project layout' },
    });
    expect(r.detail(C1)!.session.subagents.map((s) => s.id)).toEqual([G]);
    expect(r.detail(G)!.session.parent).toEqual({ id: C1, title: 'Trace the decode loop' });
    expect(r.detail(C2)!.session.parentId).toBe(P);
    expect(r.detail(O)!.session.parent).toBeNull();
    expect(r.detail(R)!.session.parent).toBeNull();
    expect(r.detail(F)!.session.parent).toEqual({ id: P, title: 'Research the project layout' });
  });

  it('marks a missing fork parent with a null title', () => {
    rmSync(sessionDir(PROJ, P), { recursive: true });
    expect(reader().detail(F)!.session.parent).toEqual({ id: P, title: null });
  });

  it('returns null for unknown and ambiguous ids', () => {
    const r = reader();
    expect(r.detail('nope')).toBeNull();
    expect(r.detail(D)).toBeNull();
  });

  it('reports an unreadable transcript without failing the detail', () => {
    const updates = join(sessionDir(PROJ, P), 'updates.jsonl');
    rmSync(updates);
    mkdirSync(updates);
    const detail = reader().detail(P)!;
    expect(detail.transcript).toEqual([]);
    expect(detail.transcriptError).toBe('updates.jsonl is not a regular file');
    expect(detail.session.tokens).toBeNull();
  });

  it('a session without updates.jsonl has an empty transcript', () => {
    const detail = reader().detail(F)!;
    expect(detail.transcript).toEqual([]);
    expect(detail.turns).toEqual([]);
    expect(detail.transcriptError).toBeUndefined();
  });

  it('builds the resume command', () => {
    expect(reader().detail(P)!.resumeCommand).toBe(`cd '/Users/dev/proj' && mlx agent --resume ${P}`);
    expect(reader({ defaultAgentHome: '/nonexistent/.mlx-agent' }).detail(P)!.resumeCommand).toBe(
      `cd '/Users/dev/proj' && GROK_HOME='${home}' mlx agent --resume ${P}`,
    );
  });
});

describe('liveness', () => {
  it('a registered live pid makes the session and every subagent descendant live', () => {
    register([{ session_id: P, pid: LIVE_PID }]);
    const r = reader();
    expect(r.list().sessions.find((s) => s.id === P)!.live).toBe(true);
    expect(r.liveness(P, true)).toEqual({ live: true, reason: 'registered', sessionId: P });
    expect(r.liveness(C2, true)).toEqual({ live: true, reason: 'registered', sessionId: P });
    expect(r.liveness(G, true)).toEqual({ live: true, reason: 'registered', sessionId: P });
    expect(r.detail(G)!.session.live).toBe(true);
    expect(r.liveness(M, true).live).toBe(false);
  });

  it('ignores registry entries whose pid is dead', () => {
    register([{ session_id: P, pid: LIVE_PID + 1 }]);
    expect(reader().liveness(P, true).live).toBe(false);
  });

  it('a corrupt registry fails closed only for write guards', () => {
    writeFileSync(join(home, 'active_sessions.json'), '[{"session_id": ');
    const r = reader();
    expect(r.liveness(M, true)).toEqual({ live: true, reason: 'registry-unreadable', sessionId: M });
    expect(r.liveness(M, false).live).toBe(false);
    expect(r.list().sessions.every((s) => !s.live)).toBe(true);
  });

  it('any file written in the last 30 s counts, not only summary/updates', () => {
    const events = join(sessionDir('%2FUsers%2Fdev%2Fother', M), 'events.jsonl');
    writeFileSync(events, '{}\n');
    utimesSync(events, new Date(NOW - 10_000), new Date(NOW - 10_000));
    expect(reader().liveness(M, true)).toEqual({ live: true, reason: 'recent-write', sessionId: M });
    utimesSync(events, new Date(NOW - 31_000), new Date(NOW - 31_000));
    expect(reader().liveness(M, true).live).toBe(false);
  });

  it('an unterminated turn written in the last 10 minutes counts', () => {
    const dir = sessionDir('%2Ftmp%2Fscratch', H);
    setMtimes(dir, NOW - 5 * 60_000);
    expect(reader().liveness(H, true)).toEqual({ live: true, reason: 'open-turn', sessionId: H });
    setMtimes(dir, NOW - 11 * 60_000);
    expect(reader().liveness(H, true).live).toBe(false);
    // A terminated turn this recent is not live.
    setMtimes(sessionDir(PROJ, P), NOW - 5 * 60_000);
    expect(reader().liveness(P, true).live).toBe(false);
  });

  it('readActiveRegistry: missing or empty file is an empty registry', () => {
    expect(readActiveRegistry(home)).toEqual({ ok: true, ids: new Set() });
    writeFileSync(join(home, 'active_sessions.json'), '');
    expect(readActiveRegistry(home)).toEqual({ ok: true, ids: new Set() });
    writeFileSync(join(home, 'active_sessions.json'), '{}');
    expect(readActiveRegistry(home).ok).toBe(false);
  });

  it('isPidAlive probes with signal 0', () => {
    expect(isPidAlive(process.pid)).toBe(true);
    expect(isPidAlive(0)).toBe(false);
    expect(isPidAlive(-1)).toBe(false);
    expect(isPidAlive(1.5)).toBe(false);
  });

  it('newestFileMtimeMs ignores subdirectories and symlinks', () => {
    const dir = sessionDir(PROJ, P);
    utimesSync(join(dir, 'subagents'), new Date(NOW), new Date(NOW));
    symlinkSync(join(home, 'active_sessions.json'), join(dir, 'link'));
    expect(newestFileMtimeMs(dir)).toBe(NOW - DAY);
    expect(newestFileMtimeMs(join(dir, 'missing'))).toBeNull();
  });
});

describe('resumeCommand', () => {
  it('single-quotes the cwd and odd ids', () => {
    expect(resumeCommand("/tmp/it's here", 'abc', '/h', '/h')).toBe(
      `cd '/tmp/it'\\''s here' && mlx agent --resume abc`,
    );
    expect(resumeCommand('/tmp', 'a b', '/h', '/h')).toBe(`cd '/tmp' && mlx agent --resume 'a b'`);
    expect(resumeCommand('/tmp', 'abc', "/x/'h", '/h')).toBe(
      `cd '/tmp' && GROK_HOME='/x/'\\''h' mlx agent --resume abc`,
    );
  });
});

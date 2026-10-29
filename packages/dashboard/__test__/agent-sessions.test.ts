import {
  appendFileSync,
  chmodSync,
  cpSync,
  existsSync,
  lstatSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  renameSync,
  rmSync,
  symlinkSync,
  truncateSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { isPidAlive, newestFileMtimeMs, readActiveRegistry } from '../src/agent-sessions/liveness.js';
import {
  AgentSessionsReader,
  type AgentSessionsReaderOptions,
  MAX_UPDATES_BYTES,
  resumeCommand,
} from '../src/agent-sessions/reader.js';
import { deleteSession, renameSession, sweepTombstones, TOMBSTONE_PREFIX } from '../src/agent-sessions/writes.js';

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

  it('subagentCount is every nested child delete would remove, not the direct meta links', () => {
    rmSync(sessionDir(PROJ, G), { recursive: true });
    const r = reader();
    // Direct links are C1, C2 and the missing GONE; G under C1 is now gone too.
    expect(r.list().sessions.find((s) => s.id === P)!.subagentCount).toBe(2);
    expect(r.detail(C1)!.session.subagentCount).toBe(0);
    expect(deleteSession(r, P).deleted).toEqual([P, C1, C2]);
  });

  it('marks each transcript subagent row with whether its child session exists', () => {
    rmSync(sessionDir(PROJ, C1), { recursive: true });
    const rows = reader()
      .detail(P)!
      .transcript.filter((e) => e.kind === 'subagent')
      .map((e) => [e.childId, e.exists]);
    expect(rows).toEqual([
      [C1, false],
      [C2, true],
    ]);
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

  it('an unterminated turn behind an unparsable transcript counts for write guards only', () => {
    const dir = sessionDir('%2Ftmp%2Fscratch', H);
    // Sparse: stat says too large to parse, nothing is written.
    truncateSync(join(dir, 'updates.jsonl'), MAX_UPDATES_BYTES + 1);
    setMtimes(dir, NOW - 2 * 60_000);
    expect(reader().liveness(H, true)).toEqual({ live: true, reason: 'open-turn', sessionId: H });
    expect(reader().liveness(H, false).live).toBe(false);
    expect(() => deleteSession(reader(), H)).toThrow(/turn in progress/);
    expect(existsSync(dir)).toBe(true);
    // Idle past the open-turn window, it can be deleted like a parsed one.
    setMtimes(dir, NOW - 11 * 60_000);
    expect(reader().liveness(H, true).live).toBe(false);
  });

  it('a parent whose summary does not parse still guards its subagent chain', () => {
    writeFileSync(join(sessionDir(PROJ, P), 'summary.json'), '{"info": ');
    register([{ session_id: P, pid: LIVE_PID }]);
    const r = reader();
    expect(r.scan().sessions.has(P)).toBe(false);
    expect(r.liveness(C1, true)).toEqual({ live: true, reason: 'registered', sessionId: P });
    expect(r.liveness(G, true)).toEqual({ live: true, reason: 'registered', sessionId: P });
    expect(r.liveness(G, false)).toEqual({ live: true, reason: 'registered', sessionId: P });
    expect(() => renameSession(r, G, 'Renamed')).toThrow(/open in a running mlx agent \(via parent/);
    expect(() => deleteSession(r, C1)).toThrow(/open in a running mlx agent \(via parent/);
    expect(existsSync(sessionDir(PROJ, C1))).toBe(true);
    expect(existsSync(sessionDir(PROJ, G))).toBe(true);

    // Unregistered, the parent's state is unknowable: guards refuse, display does not claim live.
    rmSync(join(home, 'active_sessions.json'));
    expect(r.liveness(G, true)).toEqual({ live: true, reason: 'ancestry-unknown', sessionId: P });
    expect(r.liveness(G, false).live).toBe(false);
    expect(() => deleteSession(r, C1)).toThrow(/spawned by .*summary cannot be read/);
    expect(existsSync(sessionDir(PROJ, G))).toBe(true);
  });

  it('a session folder that cannot be listed fails every write guard closed', () => {
    const subagents = join(sessionDir(PROJ, P), 'subagents');
    chmodSync(subagents, 0o000);
    try {
      const r = reader();
      expect(r.scan().unreadable).toEqual([subagents]);
      expect(r.liveness(M, true)).toEqual({ live: true, reason: 'ancestry-unknown', sessionId: M });
      expect(r.liveness(M, false).live).toBe(false);
      expect(() => deleteSession(r, G)).toThrow(/cannot be listed/);
      expect(existsSync(sessionDir(PROJ, G))).toBe(true);
    } finally {
      chmodSync(subagents, 0o755);
    }
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

describe('deleteSession', () => {
  const OTHER = '%2FUsers%2Fdev%2Fother';
  const WORKTREE = '%2FUsers%2Fdev%2F.mlx-agent%2Fworktrees%2Fproj%2Fsubagent-01a20000-0000-7000-8000-00000000000c';

  function tombstones(): string[] {
    const root = join(home, 'sessions');
    return readdirSync(root).flatMap((cwd) => {
      try {
        return readdirSync(join(root, cwd))
          .filter((n) => n.startsWith(TOMBSTONE_PREFIX))
          .map((n) => join(root, cwd, n));
      } catch {
        return [];
      }
    });
  }

  it('a folder that cannot be fully removed is already gone from every scan, and a later delete finishes it', () => {
    const assets = join(sessionDir(OTHER, M), 'assets');
    mkdirSync(assets);
    writeFileSync(join(assets, 'image.png'), 'x');
    setMtimes(home, NOW - DAY);
    chmodSync(assets, 0o000);
    const r = reader();
    try {
      expect(deleteSession(r, M)).toEqual({ deleted: [M] });
      expect(existsSync(sessionDir(OTHER, M))).toBe(false);
      const left = tombstones();
      expect(left).toHaveLength(1);
      expect(existsSync(join(left[0]!, 'summary.json'))).toBe(false);
      expect(r.scan().sessions.has(M)).toBe(false);
      expect(r.scan().unreadable).toEqual([]);
      expect(r.list().sessions.map((s) => s.id)).not.toContain(M);
      expect(sweepTombstones(r.sessionsRoot)).toEqual(left);
    } finally {
      for (const t of tombstones()) chmodSync(join(t, 'assets'), 0o755);
      if (existsSync(assets)) chmodSync(assets, 0o755);
    }
    // Any later delete retries the leftover.
    expect([...deleteSession(r, P).deleted].sort()).toEqual([P, C1, C2, G].sort());
    expect(tombstones()).toEqual([]);
  });

  it('a rename that fails part way keeps the requested session and the failed folder whole', () => {
    const worktree = join(home, 'sessions', WORKTREE);
    chmodSync(worktree, 0o555);
    const r = reader();
    let error: unknown;
    try {
      // Deepest first: G goes, C2 cannot be renamed, C1 and P are never touched.
      deleteSession(r, P);
    } catch (err) {
      error = err;
    } finally {
      chmodSync(worktree, 0o755);
    }
    expect(existsSync(sessionDir(PROJ, G))).toBe(false);
    expect(existsSync(join(sessionDir(WORKTREE, C2), 'summary.json'))).toBe(true);
    expect((error as Error).message).toMatch(new RegExp(`Could not delete session ${C2}.*already deleted: ${G}`));
    expect(tombstones()).toEqual([]);
    const index = r.scan();
    for (const sid of [P, C1, C2]) expect(index.sessions.has(sid)).toBe(true);
    expect(r.list().sessions.map((s) => s.id)).toContain(P);
    expect(r.detail(C1)!.session.subagents.map((s) => [s.id, s.exists])).toEqual([[G, false]]);

    expect([...deleteSession(r, P).deleted].sort()).toEqual([P, C1, C2].sort());
    for (const [folder, sid] of [
      [PROJ, P],
      [PROJ, C1],
      [WORKTREE, C2],
    ] as const) {
      expect(existsSync(sessionDir(folder, sid))).toBe(false);
    }
  });

  it('a crash after the child renames leaves the parent indexed and the tombstones to the next delete', () => {
    // What a delete of P that died after renaming G and C2 leaves on disk.
    const projDir = join(home, 'sessions', PROJ);
    const gTomb = join(projDir, `${TOMBSTONE_PREFIX}g`);
    const c2Tomb = join(home, 'sessions', WORKTREE, `${TOMBSTONE_PREFIX}c2`);
    renameSync(sessionDir(PROJ, G), gTomb);
    renameSync(sessionDir(WORKTREE, C2), c2Tomb);
    // Look-alikes the sweep must leave alone.
    const outside = mkdtempSync(join(tmpdir(), 'grok-outside-'));
    writeFileSync(join(outside, 'keep.txt'), 'keep');
    symlinkSync(outside, join(projDir, `${TOMBSTONE_PREFIX}link`));
    writeFileSync(join(projDir, `${TOMBSTONE_PREFIX}file`), 'x');
    try {
      const r = reader();
      const index = r.scan();
      expect(index.sessions.has(G)).toBe(false);
      expect(index.sessions.has(C2)).toBe(false);
      expect(index.unreadable).toEqual([]);
      expect(r.list().sessions.map((s) => s.id)).toContain(P);
      expect(r.liveness(P, true).live).toBe(false);

      expect([...deleteSession(r, P).deleted].sort()).toEqual([P, C1].sort());
      expect(existsSync(gTomb)).toBe(false);
      expect(existsSync(c2Tomb)).toBe(false);
      expect(readFileSync(join(outside, 'keep.txt'), 'utf8')).toBe('keep');
      expect(lstatSync(join(projDir, `${TOMBSTONE_PREFIX}link`)).isSymbolicLink()).toBe(true);
      expect(existsSync(join(projDir, `${TOMBSTONE_PREFIX}file`))).toBe(true);
    } finally {
      rmSync(outside, { recursive: true, force: true });
    }
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

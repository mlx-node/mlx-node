import {
  appendFileSync,
  chmodSync,
  cpSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  symlinkSync,
  unlinkSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { eq } from 'drizzle-orm';
import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { openDashboardDb, type DashboardDb } from '../src/db/open.js';
import { sessions, turns } from '../src/db/schema.js';
import {
  classifySessionFile,
  ingestSessions,
  isValidSessionTopology,
  readSessionEntries,
  verifySessionFileId,
} from '../src/ingest/sessions.js';

/** Write JSONL lines (objects) to a fresh `--w--` session dir; returns the file path. */
function writeSessionFile(base: string, fileName: string, lines: object[]): string {
  const dir = join(base, 'sessions', '--w--');
  mkdirSync(dir, { recursive: true });
  const file = join(dir, fileName);
  writeFileSync(file, `${lines.map((l) => JSON.stringify(l)).join('\n')}\n`);
  return file;
}

const FIXTURE_SESSIONS = fileURLToPath(new URL('./fixtures/sessions', import.meta.url));

let dash: DashboardDb;
let workRoot: string;

beforeEach(() => {
  dash = openDashboardDb(':memory:');
  const base = mkdtempSync(join(tmpdir(), 'dash-sessions-'));
  workRoot = join(base, 'sessions');
  cpSync(FIXTURE_SESSIONS, workRoot, { recursive: true });
});

afterEach(() => {
  dash.close();
  rmSync(join(workRoot, '..'), { recursive: true, force: true });
});

describe('ingestSessions', () => {
  it('indexes fixture sessions with derived fields', async () => {
    const res = await ingestSessions(dash, workRoot);
    expect(res.scanned).toBe(3);
    expect(res.updated).toBe(2);
    expect(res.removed).toBe(0);
    expect(res.warnings.length).toBeGreaterThanOrEqual(1);

    const rows = dash.db.select().from(sessions).all();
    expect(rows).toHaveLength(2);

    const one = dash.db.select().from(sessions).where(eq(sessions.id, 'fix-1')).all()[0];
    expect(one.name).toBe('First session');
    expect(one.messageCount).toBe(4);
    expect(one.firstMessage).toBe('Hello, world');
    expect(one.cwd).toBe('/w');
    expect(one.created).toBe(Date.parse('2026-07-01T10:00:00.000Z'));

    const allTurns = dash.db.select().from(turns).all();
    expect(allTurns).toHaveLength(3);

    const traced = dash.db.select().from(turns).where(eq(turns.traceId, 'trace-aaa')).all();
    expect(traced).toHaveLength(1);
    expect(traced[0].sessionId).toBe('fix-1');
    expect(traced[0].entryId).toBe('m2');
    expect(traced[0].inputTokens).toBe(100);
    expect(traced[0].outputTokens).toBe(50);
    expect(traced[0].cachedTokens).toBe(10);
    expect(traced[0].reasoningTokens).toBe(5);
  });

  // chmod 000 is a no-op for root, so the unreadable dir would still list — skip.
  const canTestUnreadable = (process.getuid?.() ?? 0) !== 0;
  (canTestUnreadable ? it : it.skip)(
    'isolates an unreadable project subdir so sibling sessions still ingest',
    async () => {
      // A sibling `--bad--/` whose inner `readdir` throws (chmod 000, or a TOCTOU
      // vanish between the root listing and the call) must NOT abort the whole
      // scan — the valid `--w--` fixture sessions must still be reconciled.
      const bad = join(workRoot, '--bad--');
      mkdirSync(bad, { recursive: true });
      writeFileSync(join(bad, 'x.jsonl'), '{"type":"session","id":"unreadable"}\n');
      chmodSync(bad, 0o000);
      try {
        const res = await ingestSessions(dash, workRoot);
        expect(res.updated).toBe(2);
        expect(dash.db.select().from(sessions).all()).toHaveLength(2);
      } finally {
        chmodSync(bad, 0o755); // restore so afterEach cleanup can remove it
      }
    },
  );

  (canTestUnreadable ? it : it.skip)('keeps indexed rows when the session ROOT itself is unreadable', async () => {
    // An unreadable root is UNKNOWN, not empty. Reconciliation retains a row only
    // while its path is in the discoverable set, so listing the root as empty
    // would delete every session on a single chmod. The pass must be skipped —
    // warned, rows untouched — and must not throw, or the caller's combined
    // try/catch would skip the trace ingest that follows it.
    await ingestSessions(dash, workRoot);
    expect(dash.db.select().from(sessions).all()).toHaveLength(2);

    chmodSync(workRoot, 0o000);
    try {
      const res = await ingestSessions(dash, workRoot);
      expect(res.scanned).toBe(0);
      expect(res.updated).toBe(0);
      expect(res.removed).toBe(0);
      expect(res.warnings.some((w) => w.includes('scan skipped'))).toBe(true);
      expect(dash.db.select().from(sessions).all()).toHaveLength(2);
      expect(dash.db.select().from(turns).all()).toHaveLength(3);
    } finally {
      chmodSync(workRoot, 0o755); // restore so afterEach cleanup can remove it
    }
  });

  it('keeps indexed rows when the session root is a regular file, not a directory', async () => {
    // `--session-dir` is never validated, so pointing it at a transcript instead
    // of its directory makes the root listing throw ENOTDIR. Same unknown state
    // as an unreadable root: warn and skip, never reconcile.
    await ingestSessions(dash, workRoot);
    expect(dash.db.select().from(sessions).all()).toHaveLength(2);

    const notADir = join(workRoot, '..', 'sessions.jsonl');
    writeFileSync(notADir, '{"type":"session","id":"typo"}\n');
    const res = await ingestSessions(dash, notADir);
    expect(res.scanned).toBe(0);
    expect(res.removed).toBe(0);
    expect(res.warnings.some((w) => w.includes('scan skipped'))).toBe(true);
    expect(dash.db.select().from(sessions).all()).toHaveLength(2);
  });

  it('still reconciles indexed rows away when the session root is absent', async () => {
    // The counterpart to the two above: an ABSENT root is provably empty, so the
    // existing wipe stays. Skipping here instead would strand rows for a deleted
    // root forever.
    await ingestSessions(dash, workRoot);
    expect(dash.db.select().from(sessions).all()).toHaveLength(2);

    const res = await ingestSessions(dash, join(workRoot, '..', 'gone'));
    expect(res.removed).toBe(2);
    expect(res.warnings).toEqual([]);
    expect(dash.db.select().from(sessions).all()).toHaveLength(0);
    expect(dash.db.select().from(turns).all()).toHaveLength(0);
  });

  it('skips unchanged files on re-ingest', async () => {
    await ingestSessions(dash, workRoot);
    const second = await ingestSessions(dash, workRoot);
    expect(second.scanned).toBe(3);
    expect(second.updated).toBe(0);
    expect(second.removed).toBe(0);
    expect(dash.db.select().from(sessions).all()).toHaveLength(2);
  });

  it('re-ingests only the file that changed', async () => {
    await ingestSessions(dash, workRoot);
    const file = join(workRoot, '--w--', '2026-07-01T10-00-00_fix-1.jsonl');
    appendFileSync(
      file,
      `${JSON.stringify({
        type: 'message',
        id: 'm5',
        parentId: 'm4',
        timestamp: '2026-07-01T10:00:06.000Z',
        message: { role: 'user', content: 'One more', timestamp: 1782036006000 },
      })}\n`,
    );

    const second = await ingestSessions(dash, workRoot);
    expect(second.updated).toBe(1);
    expect(second.removed).toBe(0);

    const one = dash.db.select().from(sessions).where(eq(sessions.id, 'fix-1')).all()[0];
    expect(one.messageCount).toBe(5);
  });

  it('removes rows when a session file is deleted', async () => {
    await ingestSessions(dash, workRoot);
    unlinkSync(join(workRoot, '--w--', '2026-07-02T10-00-00_fix-2.jsonl'));

    const second = await ingestSessions(dash, workRoot);
    expect(second.removed).toBe(1);

    expect(dash.db.select().from(sessions).all()).toHaveLength(1);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'fix-2')).all()).toHaveLength(0);
  });

  it('skips and reports a malformed file without throwing', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-bad-'));
    const soloRoot = join(soloBase, 'sessions');
    mkdirSync(join(soloRoot, '--x--'), { recursive: true });
    writeFileSync(join(soloRoot, '--x--', 'bad.jsonl'), '{"type":"session","id":"trunc","cw');

    const res = await ingestSessions(dash, soloRoot);
    expect(res.scanned).toBe(1);
    expect(res.updated).toBe(0);
    expect(res.warnings).toHaveLength(1);
    expect(dash.db.select().from(sessions).all()).toHaveLength(0);

    rmSync(soloBase, { recursive: true, force: true });
  });

  // G4: the shared DB is not keyed by session root. After switching --session-dir
  // A→B (both roots' files still on disk), A's out-of-root rows must be reconciled
  // away — reconciliation retains only paths discoverable under the CURRENT root.
  it('drops the old root sessions after switching --session-dir', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-switchroot-'));
    const rootA = join(soloBase, 'A', 'sessions');
    const rootB = join(soloBase, 'B', 'sessions');
    const dirA = join(rootA, '--w--');
    const dirB = join(rootB, '--w--');
    mkdirSync(dirA, { recursive: true });
    mkdirSync(dirB, { recursive: true });
    writeFileSync(
      join(dirA, 'a.jsonl'),
      `${[
        { type: 'session', version: 3, id: 'root-A', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/a' },
        {
          type: 'message',
          id: 'u1',
          parentId: null,
          timestamp: '2026-07-01T10:00:01.000Z',
          message: { role: 'user', content: 'from A' },
        },
        {
          type: 'message',
          id: 'a1',
          parentId: 'u1',
          timestamp: '2026-07-01T10:00:02.000Z',
          message: {
            role: 'assistant',
            content: [{ type: 'text', text: 'yo' }],
            model: 'qwen3_5',
            usage: { input: 5, output: 6 },
          },
        },
      ]
        .map((l) => JSON.stringify(l))
        .join('\n')}\n`,
    );
    writeFileSync(
      join(dirB, 'b.jsonl'),
      `${[
        { type: 'session', version: 3, id: 'root-B', timestamp: '2026-07-02T10:00:00.000Z', cwd: '/b' },
        {
          type: 'message',
          id: 'u1',
          parentId: null,
          timestamp: '2026-07-02T10:00:01.000Z',
          message: { role: 'user', content: 'from B' },
        },
      ]
        .map((l) => JSON.stringify(l))
        .join('\n')}\n`,
    );

    await ingestSessions(dash, rootA);
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'root-A')).all()).toHaveLength(1);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'root-A')).all()).toHaveLength(1);

    const res = await ingestSessions(dash, rootB);
    // A's session + turn rows are reconciled away; B is indexed.
    expect(res.removed).toBe(1);
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'root-A')).all()).toHaveLength(0);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'root-A')).all()).toHaveLength(0);
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'root-B')).all()).toHaveLength(1);

    rmSync(soloBase, { recursive: true, force: true });
  });

  // G4: reconciling against the current-root discoverable set must NOT drop an
  // in-root session skipped by the mtime/size watermark (unchanged files are
  // still present in listSessionFiles output).
  it('preserves an in-root unchanged session across re-ingest', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-inroot-'));
    const root = join(soloBase, 'sessions');
    writeSessionFile(soloBase, 's.jsonl', [
      { type: 'session', version: 3, id: 'keep-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' },
      {
        type: 'message',
        id: 'u1',
        parentId: null,
        timestamp: '2026-07-01T10:00:01.000Z',
        message: { role: 'user', content: 'hi' },
      },
      {
        type: 'message',
        id: 'a1',
        parentId: 'u1',
        timestamp: '2026-07-01T10:00:02.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'yo' }],
          model: 'qwen3_5',
          usage: { input: 5, output: 6 },
        },
      },
    ]);
    await ingestSessions(dash, root);
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'keep-1')).all()).toHaveLength(1);

    // Re-ingest the same root with the file unchanged (watermark skip) → the row
    // must survive reconciliation, not be dropped as "not rescanned this pass".
    const res = await ingestSessions(dash, root);
    expect(res.updated).toBe(0);
    expect(res.removed).toBe(0);
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'keep-1')).all()).toHaveLength(1);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'keep-1')).all()).toHaveLength(1);

    rmSync(soloBase, { recursive: true, force: true });
  });

  // Finding 12: turns must follow the active branch, not abandoned tree branches.
  it('indexes only the active-branch turn, not an abandoned one', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-branch-'));
    const root = join(soloBase, 'sessions');
    writeSessionFile(soloBase, 'branched.jsonl', [
      { type: 'session', version: 3, id: 'br-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' },
      {
        type: 'message',
        id: 'u1',
        parentId: null,
        timestamp: '2026-07-01T10:00:01.000Z',
        message: { role: 'user', content: 'q' },
      },
      // Abandoned assistant branch off u1 (huge tokens — must NOT be counted).
      {
        type: 'message',
        id: 'a1',
        parentId: 'u1',
        timestamp: '2026-07-01T10:00:02.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'abandoned' }],
          model: 'gemma4',
          usage: { input: 999, output: 999 },
        },
      },
      // Active replacement off u1, appended last → the leaf.
      {
        type: 'message',
        id: 'a2',
        parentId: 'u1',
        timestamp: '2026-07-01T10:00:03.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'active' }],
          model: 'qwen3_5',
          usage: { input: 1, output: 2 },
        },
      },
    ]);

    await ingestSessions(dash, root);

    const turnRows = dash.db.select().from(turns).where(eq(turns.sessionId, 'br-1')).all();
    expect(turnRows).toHaveLength(1);
    expect(turnRows[0].entryId).toBe('a2');
    expect(turnRows[0].model).toBe('qwen3_5');
    expect(turnRows[0].inputTokens).toBe(1);

    const row = dash.db.select().from(sessions).where(eq(sessions.id, 'br-1')).all()[0];
    // Active branch is [u1, a2]; abandoned a1 is excluded from the count.
    expect(row.messageCount).toBe(2);

    rmSync(soloBase, { recursive: true, force: true });
  });

  it('rejects a message entry whose payload is not an object (would throw downstream)', () => {
    const header = { type: 'session', version: 3, id: 's1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' };
    const good = { type: 'message', id: 'm1', parentId: null, message: { role: 'user', content: 'q' } };
    expect(isValidSessionTopology([header, good] as never)).toBe(true);
    // `message: null` (or a scalar) passes the entry-object check but throws on the
    // downstream `msg.role` deref — reject it so ingest quarantines the file.
    expect(isValidSessionTopology([header, { ...good, message: null }] as never)).toBe(false);
    expect(isValidSessionTopology([header, { ...good, message: 42 }] as never)).toBe(false);
  });

  it('quarantines a session corrupted to a non-object message after it was indexed', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-nullmsg-'));
    const root = join(soloBase, 'sessions');
    const good = [
      { type: 'session', version: 3, id: 'nm-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' },
      { type: 'message', id: 'u1', parentId: null, message: { role: 'user', content: 'q' } },
      {
        type: 'message',
        id: 'a1',
        parentId: 'u1',
        timestamp: '2026-07-01T10:00:02.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'a' }],
          model: 'm',
          usage: { input: 1, output: 2 },
        },
      },
    ];
    const file = writeSessionFile(soloBase, 'nm.jsonl', good);
    await ingestSessions(dash, root);
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'nm-1')).all()).toHaveLength(1);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'nm-1')).all()).toHaveLength(1);

    // Corrupt the assistant message's payload to `null`; a bytewise-different file
    // re-ingests. The old code threw in deriveSession → generic outer catch → stale
    // rows lingered. Now it is quarantined like any other unprojectable file.
    const corrupt = good.map((l) => (l.id === 'a1' ? { ...l, message: null } : l));
    writeFileSync(file, `${corrupt.map((l) => JSON.stringify(l)).join('\n')}\n`);
    const result = await ingestSessions(dash, root);

    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'nm-1')).all()).toHaveLength(0);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'nm-1')).all()).toHaveLength(0);
    expect(result.removed).toBeGreaterThanOrEqual(1);
    expect(result.warnings.some((w) => w.includes('non-object message'))).toBe(true);

    rmSync(soloBase, { recursive: true, force: true });
  });

  // Finding 9: a genuine legacy v1 session (message entries carry no id/parentId)
  // must be migrated in memory BEFORE the topology guard, so its history lands in
  // the index instead of being dropped as an invalid tree.
  it('migrates a legacy v1 session and indexes it instead of dropping it', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-v1-'));
    const root = join(soloBase, 'sessions');
    // parseSessionEntries does not migrate: v1 message entries have no id here.
    writeSessionFile(soloBase, 'legacy.jsonl', [
      { type: 'session', version: 1, id: 'v1-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' },
      { type: 'message', timestamp: '2026-07-01T10:00:01.000Z', message: { role: 'user', content: 'legacy hi' } },
      {
        type: 'message',
        timestamp: '2026-07-01T10:00:02.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'legacy yo' }],
          model: 'qwen3_5',
          usage: { input: 5, output: 6 },
        },
      },
    ]);

    const res = await ingestSessions(dash, root);
    expect(res.updated).toBe(1);
    expect(res.warnings).toHaveLength(0);

    const row = dash.db.select().from(sessions).where(eq(sessions.id, 'v1-1')).all()[0];
    expect(row).toBeDefined();
    expect(row.messageCount).toBe(2);
    expect(row.firstMessage).toBe('legacy hi');

    const turnRows = dash.db.select().from(turns).where(eq(turns.sessionId, 'v1-1')).all();
    expect(turnRows).toHaveLength(1);
    expect(turnRows[0].model).toBe('qwen3_5');
    expect(turnRows[0].inputTokens).toBe(5);

    rmSync(soloBase, { recursive: true, force: true });
  });

  // Finding 3: a reused path must not leave a stale row that can delete the new file.
  it('reconciles a stale row when a path is reused by a new session', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-reuse-'));
    const root = join(soloBase, 'sessions');
    const file = writeSessionFile(soloBase, 's.jsonl', [
      { type: 'session', version: 3, id: 'sess-A', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' },
      {
        type: 'message',
        id: 'a1',
        parentId: null,
        timestamp: '2026-07-01T10:00:01.000Z',
        message: { role: 'user', content: 'from A' },
      },
    ]);
    await ingestSessions(dash, root);
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'sess-A')).all()).toHaveLength(1);

    // Replace the path with session B (different id) and bump mtime forward.
    writeFileSync(
      file,
      `${JSON.stringify({ type: 'session', version: 3, id: 'sess-B', timestamp: '2026-07-02T10:00:00.000Z', cwd: '/w' })}\n${JSON.stringify({ type: 'message', id: 'b1', parentId: null, timestamp: '2026-07-02T10:00:01.000Z', message: { role: 'user', content: 'from B is longer' } })}\n`,
    );
    const later = Date.now() / 1000 + 5;
    utimesSync(file, later, later);
    await ingestSessions(dash, root);

    // Only B remains; the stale A row (which still pointed at this path) is gone.
    const rows = dash.db.select().from(sessions).all();
    expect(rows).toHaveLength(1);
    expect(rows[0].id).toBe('sess-B');
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'sess-A')).all()).toHaveLength(0);

    rmSync(soloBase, { recursive: true, force: true });
  });

  // #4: pi with an explicit `--session-dir X` writes sessions FLAT as
  // `X/<ts>_<id>.jsonl` (no `--<cwd>--` subdir), and `mlx dashboard --session-dir`
  // is documented to point at exactly that dir. Both the flat root-level layout AND
  // the `--*--/` project-subdir layout must be indexed, with no file double-counted.
  it('indexes flat root-level session files AND --cwd-- subdir files together', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-flat-'));
    const root = join(soloBase, 'sessions');
    mkdirSync(root, { recursive: true });

    // Flat pi explicit-session-dir file, sitting directly under the root.
    writeFileSync(
      join(root, '2026-07-01T10-00-00_flat-1.jsonl'),
      `${[
        { type: 'session', version: 3, id: 'flat-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/flat' },
        {
          type: 'message',
          id: 'f1',
          parentId: null,
          timestamp: '2026-07-01T10:00:01.000Z',
          message: { role: 'user', content: 'flat hi' },
        },
      ]
        .map((l) => JSON.stringify(l))
        .join('\n')}\n`,
    );

    // A classic `--<cwd>--/` project-subdir file in the same root.
    const subDir = join(root, '--sub--');
    mkdirSync(subDir, { recursive: true });
    writeFileSync(
      join(subDir, '2026-07-02T10-00-00_sub-1.jsonl'),
      `${[
        { type: 'session', version: 3, id: 'sub-1', timestamp: '2026-07-02T10:00:00.000Z', cwd: '/sub' },
        {
          type: 'message',
          id: 's1',
          parentId: null,
          timestamp: '2026-07-02T10:00:01.000Z',
          message: { role: 'user', content: 'sub hi' },
        },
      ]
        .map((l) => JSON.stringify(l))
        .join('\n')}\n`,
    );

    const res = await ingestSessions(dash, root);
    // Exactly two files scanned — the flat one and the subdir one, no double-count.
    expect(res.scanned).toBe(2);
    expect(res.updated).toBe(2);

    const flat = dash.db.select().from(sessions).where(eq(sessions.id, 'flat-1')).all();
    expect(flat).toHaveLength(1);
    expect(flat[0].cwd).toBe('/flat');
    const sub = dash.db.select().from(sessions).where(eq(sessions.id, 'sub-1')).all();
    expect(sub).toHaveLength(1);
    expect(sub[0].cwd).toBe('/sub');

    rmSync(soloBase, { recursive: true, force: true });
  });

  // #4 defense: a symlinked root-level `.jsonl` must be skipped too (same no-follow
  // guard as the subdir scan), so it cannot surface an external transcript.
  it('skips a symlinked root-level transcript (no-follow), still indexes a real one', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-flat-link-'));
    const root = join(soloBase, 'sessions');
    mkdirSync(root, { recursive: true });

    const externalFile = join(soloBase, 'external.jsonl');
    writeFileSync(
      externalFile,
      `${[
        { type: 'session', version: 3, id: 'flat-external', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/secret' },
        {
          type: 'message',
          id: 'e1',
          parentId: null,
          timestamp: '2026-07-01T10:00:01.000Z',
          message: { role: 'user', content: 'secret' },
        },
      ]
        .map((l) => JSON.stringify(l))
        .join('\n')}\n`,
    );
    symlinkSync(externalFile, join(root, '2026-07-01T10-00-00_flat-external.jsonl'));

    writeFileSync(
      join(root, '2026-07-02T10-00-00_flat-real.jsonl'),
      `${[
        { type: 'session', version: 3, id: 'flat-real', timestamp: '2026-07-02T10:00:00.000Z', cwd: '/flat' },
        {
          type: 'message',
          id: 'r1',
          parentId: null,
          timestamp: '2026-07-02T10:00:01.000Z',
          message: { role: 'user', content: 'local' },
        },
      ]
        .map((l) => JSON.stringify(l))
        .join('\n')}\n`,
    );

    const res = await ingestSessions(dash, root);
    expect(res.scanned).toBe(1);
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'flat-external')).all()).toHaveLength(0);
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'flat-real')).all()).toHaveLength(1);

    rmSync(soloBase, { recursive: true, force: true });
  });

  // Finding 4: a symlinked .jsonl must never be indexed (it could point at an
  // external Pi transcript), while a genuine sibling file still indexes.
  it('skips a symlinked transcript but still indexes a real sibling', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-symlink-'));
    const root = join(soloBase, 'sessions');
    const dir = join(root, '--w--');
    mkdirSync(dir, { recursive: true });

    // A valid external Pi session living OUTSIDE the managed sessions root.
    const externalDir = join(soloBase, 'external');
    mkdirSync(externalDir, { recursive: true });
    const externalFile = join(externalDir, 'external.jsonl');
    writeFileSync(
      externalFile,
      `${[
        { type: 'session', version: 3, id: 'external-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/secret' },
        {
          type: 'message',
          id: 'm1',
          parentId: null,
          timestamp: '2026-07-01T10:00:01.000Z',
          message: { role: 'user', content: 'secret transcript' },
        },
      ]
        .map((l) => JSON.stringify(l))
        .join('\n')}\n`,
    );
    // Symlink the external file into the managed project dir.
    symlinkSync(externalFile, join(dir, 'evil.jsonl'));

    // A genuine local session in the same dir must still be indexed.
    writeFileSync(
      join(dir, 'real.jsonl'),
      `${[
        { type: 'session', version: 3, id: 'real-1', timestamp: '2026-07-02T10:00:00.000Z', cwd: '/w' },
        {
          type: 'message',
          id: 'r1',
          parentId: null,
          timestamp: '2026-07-02T10:00:01.000Z',
          message: { role: 'user', content: 'local hi' },
        },
      ]
        .map((l) => JSON.stringify(l))
        .join('\n')}\n`,
    );

    const res = await ingestSessions(dash, root);
    // Only the real file was scanned; the symlink was skipped at listing time.
    expect(res.scanned).toBe(1);
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'external-1')).all()).toHaveLength(0);
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'real-1')).all()).toHaveLength(1);

    rmSync(soloBase, { recursive: true, force: true });
  });

  // Finding 5: reconciliation must drop a row whose indexed file was swapped for a
  // symlink. `existsSync` follows the link (keeping the stale row), so the removal
  // loop uses the same lstat-is-regular predicate the scanner does.
  it('reconciles a row whose file was replaced by an in-root symlink', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-relink-'));
    const root = join(soloBase, 'sessions');
    const dir = join(root, '--w--');
    mkdirSync(dir, { recursive: true });

    const aFile = join(dir, 'a.jsonl');
    writeFileSync(
      aFile,
      `${[
        { type: 'session', version: 3, id: 'link-A', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' },
        {
          type: 'message',
          id: 'a1',
          parentId: null,
          timestamp: '2026-07-01T10:00:01.000Z',
          message: { role: 'user', content: 'A' },
        },
      ]
        .map((l) => JSON.stringify(l))
        .join('\n')}\n`,
    );
    // A genuine sibling B (the symlink target — a regular file elsewhere in root).
    const bFile = join(dir, 'b.jsonl');
    writeFileSync(
      bFile,
      `${[
        { type: 'session', version: 3, id: 'link-B', timestamp: '2026-07-02T10:00:00.000Z', cwd: '/w' },
        {
          type: 'message',
          id: 'b1',
          parentId: null,
          timestamp: '2026-07-02T10:00:01.000Z',
          message: { role: 'user', content: 'B' },
        },
      ]
        .map((l) => JSON.stringify(l))
        .join('\n')}\n`,
    );

    await ingestSessions(dash, root);
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'link-A')).all()).toHaveLength(1);

    // Swap A's real file for a symlink to B. `existsSync(aFile)` still resolves
    // true (it follows the link), but lstat sees a non-regular file → drop the row.
    unlinkSync(aFile);
    symlinkSync(bFile, aFile);

    const res = await ingestSessions(dash, root);
    expect(res.removed).toBe(1);
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'link-A')).all()).toHaveLength(0);
    // B (a real file) stays indexed.
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'link-B')).all()).toHaveLength(1);

    rmSync(soloBase, { recursive: true, force: true });
  });

  // Finding 3: the delete-time guard used by the API before rmSync.
  it('verifySessionFileId matches only the file current header id', () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-verify-'));
    const file = writeSessionFile(soloBase, 's.jsonl', [
      { type: 'session', version: 3, id: 'sess-B', timestamp: '2026-07-02T10:00:00.000Z', cwd: '/w' },
    ]);
    expect(verifySessionFileId(file, 'sess-B')).toBe(true);
    expect(verifySessionFileId(file, 'sess-A')).toBe(false);
    expect(verifySessionFileId(join(soloBase, 'missing.jsonl'), 'sess-B')).toBe(false);
    rmSync(soloBase, { recursive: true, force: true });
  });

  // The four-way verdict the delete handler branches on: `verifySessionFileId`'s
  // single `false` cannot tell a file that PROVABLY belongs to another session
  // (`different` — row-only cleanup is safe) from one we simply cannot verify
  // (`unverifiable` — deleting the row while the file lingers would desync the DB
  // from disk), so the delete guard needs the reason, not just a boolean.
  it('classifySessionFile distinguishes missing / matches / different / unverifiable', () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-classify-'));

    const match = writeSessionFile(soloBase, 'match.jsonl', [
      { type: 'session', version: 3, id: 'sess-B', timestamp: '2026-07-02T10:00:00.000Z', cwd: '/w' },
    ]);
    expect(classifySessionFile(match, 'sess-B')).toBe('matches');
    // Same readable file, header names a DIFFERENT session (path reused).
    expect(classifySessionFile(match, 'sess-A')).toBe('different');

    // No file on disk → nothing to orphan.
    expect(classifySessionFile(join(soloBase, 'gone.jsonl'), 'sess-B')).toBe('missing');

    // Exists + parseable, but carries no `type:'session'` header → unverifiable.
    const headerless = writeSessionFile(soloBase, 'headerless.jsonl', [
      {
        type: 'message',
        id: 'm1',
        parentId: null,
        timestamp: '2026-07-02T10:00:01.000Z',
        message: { role: 'user', content: 'orphaned, no header' },
      },
    ]);
    expect(classifySessionFile(headerless, 'sess-B')).toBe('unverifiable');

    // Exists but unreadable → unverifiable (skip as root, where chmod 000 is a no-op).
    if ((process.getuid?.() ?? 0) !== 0) {
      chmodSync(headerless, 0o000);
      try {
        expect(classifySessionFile(headerless, 'sess-B')).toBe('unverifiable');
      } finally {
        chmodSync(headerless, 0o644);
      }
    }

    rmSync(soloBase, { recursive: true, force: true });
  });

  // Finding 5: the GET detail read path (readSessionEntries) must be READ-ONLY.
  // A v1 session with a malformed trailing line — the case where SessionManager's
  // open-for-write would persist the v1→v3 migration AND drop the malformed line —
  // must leave the file byte-for-byte unchanged while still projecting the valid
  // messages (migrated in memory so the v1 entries carry ids).
  it('readSessionEntries reads a v1 + malformed-trailing session without touching disk', () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-readonly-'));
    const dir = join(soloBase, 'sessions', '--w--');
    mkdirSync(dir, { recursive: true });
    const file = join(dir, 'v1-partial.jsonl');

    // Genuine v1: message entries carry no id/parentId. Trailing line is truncated.
    const header = JSON.stringify({
      type: 'session',
      version: 1,
      id: 'ro-1',
      timestamp: '2026-07-01T10:00:00.000Z',
      cwd: '/w',
    });
    const user = JSON.stringify({
      type: 'message',
      timestamp: '2026-07-01T10:00:01.000Z',
      message: { role: 'user', content: 'hello v1' },
    });
    const asst = JSON.stringify({
      type: 'message',
      timestamp: '2026-07-01T10:00:02.000Z',
      message: {
        role: 'assistant',
        content: [{ type: 'text', text: 'hi v1' }],
        model: 'qwen3_5',
        usage: { input: 5, output: 6 },
      },
    });
    const truncated = '{"type":"message","message":{"role":"asst';
    const original = `${header}\n${user}\n${asst}\n${truncated}`;
    writeFileSync(file, original);
    const before = readFileSync(file);

    const entries = readSessionEntries(file);
    // Valid messages projected; migration assigned ids in memory.
    const messages = entries.filter((e) => e.type === 'message');
    expect(messages).toHaveLength(2);
    for (const m of messages) expect(typeof m.id).toBe('string');

    // The file on disk is byte-for-byte unchanged: no migration persisted, no
    // malformed line dropped.
    const after = readFileSync(file);
    expect(after.length).toBe(before.length);
    expect(after.equals(before)).toBe(true);
    expect(after.toString('utf-8')).toBe(original);

    rmSync(soloBase, { recursive: true, force: true });
  });

  // Finding 13: a truncated trailing line must not advance the ingest watermark.
  it('does not mark a truncated file fully-ingested and re-ingests once completed', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-trunc-'));
    const root = join(soloBase, 'sessions');
    const dir = join(root, '--w--');
    mkdirSync(dir, { recursive: true });
    const file = join(dir, 'partial.jsonl');

    const header = JSON.stringify({
      type: 'session',
      version: 3,
      id: 'trunc-1',
      timestamp: '2026-07-01T10:00:00.000Z',
      cwd: '/w',
    });
    const user = JSON.stringify({
      type: 'message',
      id: 'm1',
      parentId: null,
      timestamp: '2026-07-01T10:00:01.000Z',
      message: { role: 'user', content: 'hi' },
    });
    const asst = JSON.stringify({
      type: 'message',
      id: 'm2',
      parentId: 'm1',
      timestamp: '2026-07-01T10:00:02.000Z',
      message: {
        role: 'assistant',
        content: [{ type: 'text', text: 'yo' }],
        model: 'qwen3_5',
        usage: { input: 5, output: 6 },
      },
    });
    const truncated =
      '{"type":"message","id":"m3","parentId":"m2","timestamp":"2026-07-01T10:00:03.000Z","message":{"role":"asst';

    writeFileSync(file, `${header}\n${user}\n${asst}\n${truncated}`);
    const res1 = await ingestSessions(dash, root);
    expect(res1.warnings.some((w) => /trailing|malformed/i.test(w))).toBe(true);

    const row1 = dash.db.select().from(sessions).where(eq(sessions.id, 'trunc-1')).all()[0];
    // Partial data indexed, but the watermark is NOT the current mtime/size.
    expect(row1.lastIngestedMtime).toBe(0);
    expect(row1.lastIngestedSize).toBe(0);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'trunc-1')).all()).toHaveLength(1);

    // Complete the trailing write and bump mtime.
    const asst3 = JSON.stringify({
      type: 'message',
      id: 'm3',
      parentId: 'm2',
      timestamp: '2026-07-01T10:00:03.000Z',
      message: {
        role: 'assistant',
        content: [{ type: 'text', text: 'done' }],
        model: 'qwen3_5',
        usage: { input: 7, output: 8 },
      },
    });
    writeFileSync(file, `${header}\n${user}\n${asst}\n${asst3}\n`);
    const later = Date.now() / 1000 + 5;
    utimesSync(file, later, later);

    const res2 = await ingestSessions(dash, root);
    // The stale (0,0) watermark forced a re-ingest that now sees the completed line.
    expect(res2.updated).toBe(1);
    expect(res2.warnings.some((w) => /trailing|malformed/i.test(w))).toBe(false);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'trunc-1')).all()).toHaveLength(2);
    const row2 = dash.db.select().from(sessions).where(eq(sessions.id, 'trunc-1')).all()[0];
    expect(row2.lastIngestedMtime).toBeGreaterThan(0);

    rmSync(soloBase, { recursive: true, force: true });
  });

  // F3: a valid header + a malformed INTERIOR line (a valid line after it) must
  // NOT be indexed as a real transcript. pi's parser silently drops the interior
  // line and keeps going; the orphaned child (its parent was the dropped line) is
  // accepted by the topology guard as a false root, so the derived transcript is
  // a corrupt subset (here: missing u1). Since this is not a single-trailing-line
  // live-append, it must be quarantined/skipped, not indexed.
  it('does not index a session with a malformed interior line (false transcript)', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-interior-'));
    const root = join(soloBase, 'sessions');
    const dir = join(root, '--w--');
    mkdirSync(dir, { recursive: true });
    const file = join(dir, 'interior.jsonl');

    const header = JSON.stringify({
      type: 'session',
      version: 3,
      id: 'int-1',
      timestamp: '2026-07-01T10:00:00.000Z',
      cwd: '/w',
    });
    const user = JSON.stringify({
      type: 'message',
      id: 'u1',
      parentId: null,
      timestamp: '2026-07-01T10:00:01.000Z',
      message: { role: 'user', content: 'q' },
    });
    // A truncated INTERIOR record — pi drops it and continues parsing.
    const brokenInterior =
      '{"type":"message","id":"m2","parentId":"u1","timestamp":"2026-07-01T10:00:02.000Z","message":{"role":"asst';
    // Its child references the DROPPED m2 → orphaned → accepted as a false root.
    const orphan = JSON.stringify({
      type: 'message',
      id: 'a3',
      parentId: 'm2',
      timestamp: '2026-07-01T10:00:03.000Z',
      message: {
        role: 'assistant',
        content: [{ type: 'text', text: 'orphan' }],
        model: 'qwen3_5',
        usage: { input: 7, output: 8 },
      },
    });
    // 4 non-blank lines; the LAST line (orphan) parses → this is an interior drop,
    // not a trailing incomplete-prefix.
    writeFileSync(file, `${header}\n${user}\n${brokenInterior}\n${orphan}\n`);

    const res = await ingestSessions(dash, root);
    expect(res.warnings.some((w) => /malformed|skipped/i.test(w))).toBe(true);
    // The corrupt session is absent — never indexed as a false transcript.
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'int-1')).all()).toHaveLength(0);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'int-1')).all()).toHaveLength(0);

    rmSync(soloBase, { recursive: true, force: true });
  });

  // F3: when a previously-indexed valid transcript is swapped (same path) to one
  // with an interior drop, its stale rows must be QUARANTINED (removed), not left
  // feeding the overview — mirroring the topology/no-header quarantine branches.
  it('quarantines stale rows when a regular file is swapped to an interior-drop transcript', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-interior-swap-'));
    const root = join(soloBase, 'sessions');
    const file = writeSessionFile(soloBase, 's.jsonl', [
      { type: 'session', version: 3, id: 'sw-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' },
      {
        type: 'message',
        id: 'u1',
        parentId: null,
        timestamp: '2026-07-01T10:00:01.000Z',
        message: { role: 'user', content: 'hi' },
      },
      {
        type: 'message',
        id: 'a1',
        parentId: 'u1',
        timestamp: '2026-07-01T10:00:02.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'yo' }],
          model: 'qwen3_5',
          usage: { input: 5, output: 6 },
        },
      },
    ]);
    await ingestSessions(dash, root);
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'sw-1')).all()).toHaveLength(1);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'sw-1')).all()).toHaveLength(1);

    // Swap the SAME path to a valid-header file with a dropped interior record.
    const header = JSON.stringify({
      type: 'session',
      version: 3,
      id: 'sw-1',
      timestamp: '2026-07-01T10:00:00.000Z',
      cwd: '/w',
    });
    const user = JSON.stringify({
      type: 'message',
      id: 'u1',
      parentId: null,
      timestamp: '2026-07-02T10:00:01.000Z',
      message: { role: 'user', content: 'hi' },
    });
    const brokenInterior =
      '{"type":"message","id":"m2","parentId":"u1","timestamp":"2026-07-02T10:00:02.000Z","message":{"role":"asst';
    const orphan = JSON.stringify({
      type: 'message',
      id: 'a3',
      parentId: 'm2',
      timestamp: '2026-07-02T10:00:03.000Z',
      message: {
        role: 'assistant',
        content: [{ type: 'text', text: 'orphan' }],
        model: 'qwen3_5',
        usage: { input: 7, output: 8 },
      },
    });
    writeFileSync(file, `${header}\n${user}\n${brokenInterior}\n${orphan}\n`);
    const later = Date.now() / 1000 + 5;
    utimesSync(file, later, later);

    const res = await ingestSessions(dash, root);
    expect(res.warnings.some((w) => /malformed|skipped/i.test(w))).toBe(true);
    expect(res.removed).toBe(1);
    // The stale rows are GONE, not merely skipped.
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'sw-1')).all()).toHaveLength(0);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'sw-1')).all()).toHaveLength(0);

    rmSync(soloBase, { recursive: true, force: true });
  });

  // Finding G: a cyclic entry tree must be quarantined, not walked unbounded
  // (pi's branch walker has no visited-set → self-parent/cycle hangs/OOMs).
  it('quarantines a cyclic session tree without hanging and still ingests valid siblings', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-cycle-'));
    const root = join(soloBase, 'sessions');
    // Self-parented leaf: `x.parentId === 'x'` — the walker would loop forever.
    writeSessionFile(soloBase, 'cyclic.jsonl', [
      { type: 'session', version: 3, id: 'cyc-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' },
      {
        type: 'message',
        id: 'x',
        parentId: 'x',
        timestamp: '2026-07-01T10:00:01.000Z',
        message: { role: 'user', content: 'loop' },
      },
    ]);
    writeSessionFile(soloBase, 'valid.jsonl', [
      { type: 'session', version: 3, id: 'ok-1', timestamp: '2026-07-02T10:00:00.000Z', cwd: '/w' },
      {
        type: 'message',
        id: 'u1',
        parentId: null,
        timestamp: '2026-07-02T10:00:01.000Z',
        message: { role: 'user', content: 'hi' },
      },
      {
        type: 'message',
        id: 'a1',
        parentId: 'u1',
        timestamp: '2026-07-02T10:00:02.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'ok' }],
          model: 'qwen3_5',
          usage: { input: 3, output: 4 },
        },
      },
    ]);

    const res = await ingestSessions(dash, root);
    expect(res.warnings.some((w) => /cycle|duplicate|invalid session tree/i.test(w))).toBe(true);

    // The cyclic file is skipped; the valid sibling in the same run still lands.
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'cyc-1')).all()).toHaveLength(0);
    const ok = dash.db.select().from(sessions).where(eq(sessions.id, 'ok-1')).all();
    expect(ok).toHaveLength(1);
    expect(ok[0].messageCount).toBe(2);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'ok-1')).all()).toHaveLength(1);

    rmSync(soloBase, { recursive: true, force: true });
  });

  // Finding H: when the leaf is detached metadata (a `session_info` from a
  // rename), only the active message-bearing branch is indexed — never a flat
  // union of every abandoned branch.
  it('indexes only the active branch when the leaf is detached metadata', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-detached-'));
    const root = join(soloBase, 'sessions');
    writeSessionFile(soloBase, 'forked.jsonl', [
      { type: 'session', version: 3, id: 'fk-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' },
      {
        type: 'message',
        id: 'u1',
        parentId: null,
        timestamp: '2026-07-01T10:00:01.000Z',
        message: { role: 'user', content: 'q' },
      },
      // Abandoned 999-token assistant off u1 (must NOT be indexed).
      {
        type: 'message',
        id: 'a1',
        parentId: 'u1',
        timestamp: '2026-07-01T10:00:02.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'abandoned' }],
          model: 'gemma4',
          usage: { input: 999, output: 999 },
        },
      },
      // Active 1-token replacement off u1.
      {
        type: 'message',
        id: 'a2',
        parentId: 'u1',
        timestamp: '2026-07-01T10:00:03.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'active' }],
          model: 'qwen3_5',
          usage: { input: 1, output: 2 },
        },
      },
      // Detached metadata leaf appended last (a rename) — carries no message.
      { type: 'session_info', id: 'si1', parentId: null, timestamp: '2026-07-01T10:00:04.000Z', name: 'Forked' },
    ]);

    await ingestSessions(dash, root);

    const turnRows = dash.db.select().from(turns).where(eq(turns.sessionId, 'fk-1')).all();
    expect(turnRows).toHaveLength(1);
    expect(turnRows[0].entryId).toBe('a2');
    expect(turnRows[0].model).toBe('qwen3_5');
    expect(turnRows[0].inputTokens).toBe(1);

    const row = dash.db.select().from(sessions).where(eq(sessions.id, 'fk-1')).all()[0];
    // Active branch is [u1, a2]; abandoned a1 and the detached info leaf excluded.
    expect(row.messageCount).toBe(2);
    // Session-level name still derives from the whole file, separate from turns.
    expect(row.name).toBe('Forked');

    rmSync(soloBase, { recursive: true, force: true });
  });

  // Finding 6: a valid transcript swapped to a headerless (but REGULAR) JSONL must
  // have its previously-indexed rows QUARANTINED (removed), not merely skipped —
  // reconciliation only removes rows whose path is no longer a regular file, so a
  // swapped-to-garbage regular file would otherwise feed the overview forever.
  it('quarantines stale rows when a regular file is swapped to a headerless transcript', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-quarantine-'));
    const root = join(soloBase, 'sessions');
    const file = writeSessionFile(soloBase, 's.jsonl', [
      { type: 'session', version: 3, id: 'q-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' },
      {
        type: 'message',
        id: 'u1',
        parentId: null,
        timestamp: '2026-07-01T10:00:01.000Z',
        message: { role: 'user', content: 'hi' },
      },
      {
        type: 'message',
        id: 'a1',
        parentId: 'u1',
        timestamp: '2026-07-01T10:00:02.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'yo' }],
          model: 'qwen3_5',
          usage: { input: 5, output: 6 },
        },
      },
    ]);
    await ingestSessions(dash, root);
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'q-1')).all()).toHaveLength(1);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'q-1')).all()).toHaveLength(1);

    // Swap the SAME path for a headerless (regular) JSONL — no `type:session`
    // line, so deriveSession returns null. Bump mtime so the watermark re-reads it.
    writeFileSync(
      file,
      `${JSON.stringify({ type: 'message', id: 'm1', parentId: null, timestamp: '2026-07-02T10:00:00.000Z', message: { role: 'user', content: 'headerless' } })}\n`,
    );
    const later = Date.now() / 1000 + 5;
    utimesSync(file, later, later);

    const res = await ingestSessions(dash, root);
    expect(res.warnings.some((w) => /no valid session header/i.test(w))).toBe(true);
    expect(res.removed).toBe(1);
    // The stale rows are GONE, not just skipped.
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'q-1')).all()).toHaveLength(0);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'q-1')).all()).toHaveLength(0);
    expect(dash.db.select().from(sessions).all()).toHaveLength(0);

    rmSync(soloBase, { recursive: true, force: true });
  });

  // Finding 6: the same quarantine applies when a valid transcript is swapped to a
  // cyclic (topologically invalid) but REGULAR JSONL — caught at the topology guard
  // before deriveSession, its stale rows must still be removed.
  it('quarantines stale rows when a regular file is swapped to a cyclic transcript', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-quarantine-cyc-'));
    const root = join(soloBase, 'sessions');
    const file = writeSessionFile(soloBase, 's.jsonl', [
      { type: 'session', version: 3, id: 'qc-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' },
      {
        type: 'message',
        id: 'u1',
        parentId: null,
        timestamp: '2026-07-01T10:00:01.000Z',
        message: { role: 'user', content: 'hi' },
      },
      {
        type: 'message',
        id: 'a1',
        parentId: 'u1',
        timestamp: '2026-07-01T10:00:02.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'yo' }],
          model: 'qwen3_5',
          usage: { input: 5, output: 6 },
        },
      },
    ]);
    await ingestSessions(dash, root);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'qc-1')).all()).toHaveLength(1);

    // Swap for a self-parented (cyclic) tree: valid header, invalid topology.
    writeFileSync(
      file,
      `${JSON.stringify({ type: 'session', version: 3, id: 'qc-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' })}\n${JSON.stringify({ type: 'message', id: 'x', parentId: 'x', timestamp: '2026-07-02T10:00:01.000Z', message: { role: 'user', content: 'loop' } })}\n`,
    );
    const later = Date.now() / 1000 + 5;
    utimesSync(file, later, later);

    const res = await ingestSessions(dash, root);
    expect(res.warnings.some((w) => /cycle|duplicate|invalid session tree/i.test(w))).toBe(true);
    expect(res.removed).toBe(1);
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'qc-1')).all()).toHaveLength(0);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'qc-1')).all()).toHaveLength(0);

    rmSync(soloBase, { recursive: true, force: true });
  });

  // Finding 8: a file swapped to `<header>\nnull\n` is valid JSONL (parses to
  // [header, null]) but a top-level `null` entry has no `.type`/`.id` to walk.
  // `isValidSessionTopology` must reject it (return false) instead of throwing a
  // TypeError that escapes every quarantine branch — otherwise the stale session +
  // turn rows survive every rescan forever and keep feeding overview metrics. The
  // rejection routes the file into the topology-quarantine branch, removing them.
  it('quarantines stale rows when a regular file is swapped to a top-level null transcript', async () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-null-topology-'));
    const root = join(soloBase, 'sessions');
    const file = writeSessionFile(soloBase, 's.jsonl', [
      { type: 'session', version: 3, id: 'nl-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' },
      {
        type: 'message',
        id: 'u1',
        parentId: null,
        timestamp: '2026-07-01T10:00:01.000Z',
        message: { role: 'user', content: 'hi' },
      },
      {
        type: 'message',
        id: 'a1',
        parentId: 'u1',
        timestamp: '2026-07-01T10:00:02.000Z',
        message: {
          role: 'assistant',
          content: [{ type: 'text', text: 'yo' }],
          model: 'qwen3_5',
          usage: { input: 5, output: 6 },
        },
      },
    ]);
    await ingestSessions(dash, root);
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'nl-1')).all()).toHaveLength(1);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'nl-1')).all()).toHaveLength(1);

    // Swap the SAME path to a valid header followed by a literal `null` line.
    const header = JSON.stringify({
      type: 'session',
      version: 3,
      id: 'nl-1',
      timestamp: '2026-07-01T10:00:00.000Z',
      cwd: '/w',
    });
    writeFileSync(file, `${header}\nnull\n`);
    const later = Date.now() / 1000 + 5;
    utimesSync(file, later, later);

    const res = await ingestSessions(dash, root);
    // Quarantined via the topology branch (not thrown past every branch).
    expect(res.warnings.some((w) => /cycle|duplicate|invalid session tree/i.test(w))).toBe(true);
    expect(res.removed).toBe(1);
    // The stale session AND its token turns are REMOVED, not retained forever.
    expect(dash.db.select().from(sessions).where(eq(sessions.id, 'nl-1')).all()).toHaveLength(0);
    expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'nl-1')).all()).toHaveLength(0);

    rmSync(soloBase, { recursive: true, force: true });
  });

  // The test above pins only the case where the header is the FIRST record and is
  // already at the current version: pi's migrator locates the header with
  // `entries.find((e) => e.type === 'session')`, which short-circuits at index 0 and
  // never dereferences a later `null`. That short-circuit does NOT generalize. A
  // `null` positioned BEFORE the header is dereferenced by that same `find`, and a
  // legacy (v1) header makes `migrateV1ToV2` re-walk EVERY record — so a `null`
  // anywhere throws. Both throw a TypeError from `migrateSessionEntries` BEFORE
  // `isValidSessionTopology` can reject the file, landing in the per-file catch that
  // only warns — leaving the stale session and turn rows to feed the overview on
  // every later scan. Each shape must reach the quarantine branch instead.
  const preHeaderNullShapes: { name: string; build: (id: string) => string }[] = [
    {
      name: 'a null record before the header',
      build: (id) =>
        `null\n${JSON.stringify({ type: 'session', version: 3, id, timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' })}\n`,
    },
    {
      name: 'a legacy v1 header followed by a null record',
      build: (id) =>
        `${JSON.stringify({ type: 'session', id, timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' })}\nnull\n`,
    },
  ];

  for (const shape of preHeaderNullShapes) {
    it(`quarantines stale rows when a file is swapped to ${shape.name}`, async () => {
      const soloBase = mkdtempSync(join(tmpdir(), 'dash-null-premigrate-'));
      const root = join(soloBase, 'sessions');
      const file = writeSessionFile(soloBase, 's.jsonl', [
        { type: 'session', version: 3, id: 'pm-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' },
        {
          type: 'message',
          id: 'u1',
          parentId: null,
          timestamp: '2026-07-01T10:00:01.000Z',
          message: { role: 'user', content: 'hi' },
        },
        {
          type: 'message',
          id: 'a1',
          parentId: 'u1',
          timestamp: '2026-07-01T10:00:02.000Z',
          message: {
            role: 'assistant',
            content: [{ type: 'text', text: 'yo' }],
            model: 'qwen3_5',
            usage: { input: 5, output: 6 },
          },
        },
      ]);
      await ingestSessions(dash, root);
      expect(dash.db.select().from(sessions).where(eq(sessions.id, 'pm-1')).all()).toHaveLength(1);
      expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'pm-1')).all()).toHaveLength(1);

      writeFileSync(file, shape.build('pm-1'));
      const later = Date.now() / 1000 + 5;
      utimesSync(file, later, later);

      const res = await ingestSessions(dash, root);
      // The stale session AND its token turns are REMOVED, not retained forever.
      expect(dash.db.select().from(sessions).where(eq(sessions.id, 'pm-1')).all()).toHaveLength(0);
      expect(dash.db.select().from(turns).where(eq(turns.sessionId, 'pm-1')).all()).toHaveLength(0);
      expect(res.removed).toBe(1);
      // Routed through the topology quarantine, NOT past every branch into the
      // per-file catch (whose warning would carry the raw TypeError text).
      expect(res.warnings.some((w) => /cycle|duplicate|invalid session tree/i.test(w))).toBe(true);
      expect(res.warnings.some((w) => /Cannot read properties of null/i.test(w))).toBe(false);

      rmSync(soloBase, { recursive: true, force: true });
    });
  }

  // The delete/rename guards find the header with the same unguarded `.type` deref,
  // which a record before the header throws on — escaping both handlers. The header
  // is still readable, parseable and present, so the documented verdict is the
  // ownership answer ('matches'/'different'), NOT 'unverifiable': a session with one
  // corrupt line must stay deletable rather than 409 forever.
  it('classifySessionFile reads past a null record before the header', () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-null-classify-'));
    const dir = join(soloBase, 'sessions', '--w--');
    mkdirSync(dir, { recursive: true });
    const file = join(dir, 's.jsonl');
    writeFileSync(
      file,
      `null\n${JSON.stringify({ type: 'session', version: 3, id: 'cf-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' })}\n`,
    );

    expect(classifySessionFile(file, 'cf-1')).toBe('matches');
    expect(classifySessionFile(file, 'cf-other')).toBe('different');
    expect(verifySessionFileId(file, 'cf-1')).toBe(true);
    expect(verifySessionFileId(file, 'cf-other')).toBe(false);

    rmSync(soloBase, { recursive: true, force: true });
  });

  // The read-only detail path migrates too. A non-object record must not surface as
  // a raw TypeError; the entries come back unmigrated so the caller's topology guard
  // produces the same "invalid session tree" verdict ingest reports.
  it('readSessionEntries returns non-object records instead of throwing from the migrator', () => {
    const soloBase = mkdtempSync(join(tmpdir(), 'dash-null-read-'));
    const dir = join(soloBase, 'sessions', '--w--');
    mkdirSync(dir, { recursive: true });
    const file = join(dir, 's.jsonl');
    writeFileSync(
      file,
      `null\n${JSON.stringify({ type: 'session', version: 3, id: 'rd-1', timestamp: '2026-07-01T10:00:00.000Z', cwd: '/w' })}\n`,
    );

    const entries = readSessionEntries(file);
    expect(entries).toHaveLength(2);
    expect(entries[0]).toBeNull();
    expect(isValidSessionTopology(entries)).toBe(false);

    rmSync(soloBase, { recursive: true, force: true });
  });
});

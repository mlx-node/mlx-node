import { readFileSync, realpathSync, rmSync, statSync } from 'node:fs';
import { basename, dirname, join, sep } from 'node:path';

import { SessionManager, parseSessionEntries, type FileEntry } from '@earendil-works/pi-coding-agent';
import { eq } from 'drizzle-orm';

import { sessions, turns } from '../../db/schema.js';
import {
  activeBranchEntries,
  classifySessionFile,
  countJsonlLines,
  findSessionHeader,
  isValidSessionTopology,
  lastLineParses,
  readSessionEntries,
  verifySessionFileId,
} from '../../ingest/sessions.js';
import type { ApiPaths, ApiRequest, WorkerApiContext } from '../context.js';
import { queryInt, requireBody, toInt } from '../context.js';
import { ApiError } from '../errors.js';
import { collectCallArgs, mapTranscriptEntry, type TranscriptEntry } from '../transcript.js';

/**
 * A session file modified within this window may be actively written by a live
 * agent turn (a SEPARATE process — pi has no cross-process lock). The rename writes
 * the durable name into the pi session JSONL: `SessionManager.open` snapshots the
 * current leaf and `appendSessionInfo` parents the new entry to it, so a turn
 * appended by a concurrent agent between our snapshot and append becomes a sibling
 * of the rename — on the next resume one of them is orphaned (the turn is lost, or
 * the rename silently vanishes). We refuse the rename while the file looks active
 * (mtime inside this window) and only proceed for an idle session.
 *
 * This is a best-effort PRODUCT rule, not a hard guarantee: a session idle past this
 * window that then goes live, or one that goes live within the stat→append window,
 * can still race. It removes the realistic reachability, not the theoretical race.
 * Storing the name index-only is not an alternative — it would break the disposable-
 * index invariant (the rename would be lost when `dashboard.db` is rebuilt from the
 * JSONL source of truth), so the durable JSONL write is the only correct home.
 */
const LIVE_SESSION_WINDOW_MS = 30_000;

export function handleSessionsList(ctx: WorkerApiContext, req: ApiRequest): unknown {
  const q = req.query.get('q');
  const cwd = req.query.get('cwd');
  const model = req.query.get('model');
  const from = queryInt(req.query, 'from');
  const to = queryInt(req.query, 'to');
  // Page the (possibly large) list. `limit` defaults to 500 and is clamped to
  // [1, 500] so an unbounded/absurd value can't be forced; `offset` clamps to >= 0.
  const limit = Math.min(Math.max(queryInt(req.query, 'limit') ?? 500, 1), 500);
  const offset = Math.max(queryInt(req.query, 'offset') ?? 0, 0);

  const where: string[] = [];
  const args: Array<string | number> = [];
  if (q !== null && q !== '') {
    where.push('(s.name LIKE ? OR s.first_message LIKE ?)');
    args.push(`%${q}%`, `%${q}%`);
  }
  if (cwd !== null && cwd !== '') {
    where.push('s.cwd = ?');
    args.push(cwd);
  }
  if (model !== null && model !== '') {
    where.push('EXISTS (SELECT 1 FROM turns t WHERE t.session_id = s.id AND t.model = ?)');
    args.push(model);
  }
  if (from !== null) {
    where.push('s.modified >= ?');
    args.push(from);
  }
  if (to !== null) {
    where.push('s.modified <= ?');
    args.push(to);
  }

  const whereSql = where.length > 0 ? `WHERE ${where.join(' AND ')}` : '';
  const matchedIds = `SELECT s.id FROM sessions s ${whereSql}`;
  const sql = `
    SELECT s.id, s.path, s.cwd, s.name, s.created, s.modified,
           s.message_count AS messageCount, s.first_message AS firstMessage,
           (SELECT group_concat(DISTINCT t.model) FROM turns t
              WHERE t.session_id = s.id AND t.model IS NOT NULL) AS models,
           (SELECT COALESCE(SUM(t.input_tokens), 0) FROM turns t WHERE t.session_id = s.id) AS inputTokens,
           (SELECT COALESCE(SUM(t.output_tokens), 0) FROM turns t WHERE t.session_id = s.id) AS outputTokens
    FROM sessions s
    ${whereSql}
    ORDER BY s.modified DESC
    LIMIT ? OFFSET ?`;

  // A separate COUNT over the SAME filter so the client (e.g. the Overview tile)
  // reports the true match total rather than the capped page length.
  const total = toInt(
    (ctx.dash.sqlite.prepare(`SELECT COUNT(*) AS n FROM sessions s ${whereSql}`).get(...args) as { n: unknown }).n,
  );

  // Tokens over THOSE SAME sessions, from the SAME filter.
  //
  // This exists so a caller that shows both numbers cannot pair them with a
  // different scope. `sessions`/`turns` hold only what lives under the CURRENT
  // `--session-dir` (ingest deletes rows whose file left the root), while
  // `traces` is a machine-wide log from `~/.mlx-node/metrics/traces` and
  // `/metrics/overview` sums it whole. Reading a count from here and a token
  // total from there produces a tile whose halves describe different session
  // sets — the same defect the Cache page's root scoping fixed, where a
  // filesystem scan of one cache root sat beside a hit rate summed over every
  // root the machine ever used.
  //
  // This counts each DISTINCT inference in the matched set once — the same rule
  // `/metrics/overview` applies — and is deliberately NOT the sum of the
  // per-session `/api/sessions/:id/metrics` totals. The first arm collapses
  // `turns` on their canonical identity across the WHOLE matched set, so a
  // forked session (which copies inherited turns VERBATIM, same `trace_id` and
  // `entry_id`) contributes its inherited history once, whereas each per-session
  // call reports every copy to the session holding it. Adding those per-session
  // totals up therefore exceeds this number by the shared history, and so does
  // adding up the per-row `inputTokens`/`outputTokens` columns below, which stay
  // plain per-session sums with no dedup and no trace arm precisely because a
  // session row must show what THAT session spent. Both readings are wanted; the
  // headline total is the one that must not bill a fork twice.
  //
  // The second arm adds subagent work, which runs on an in-memory session manager
  // and therefore writes a `traces` row with no `turns` row at all. It is
  // admitted only via `root_session_id` pointing INTO the matched set, and only
  // when no `turns` row already carries that `trace_id`; the inner
  // `trace_id IS NOT NULL` keeps `NOT IN` from going NULL and dropping the whole
  // set. That dedup subquery is unscoped by session on purpose (the per-session
  // handler scopes its own to `session_id = ?`): a trace some session already
  // accounts for must never be added again to a set-wide total. Trace rows store
  // GROSS `prompt_tokens`, so the input is clamped to the producer's net
  // `MAX(prompt - cached, 0)` rather than double-counting cache reads against the
  // already-net turns side.
  const tokenTotals = ctx.dash.sqlite
    .prepare(
      `SELECT COALESCE(SUM(inputTokens), 0) AS inputTokens, COALESCE(SUM(outputTokens), 0) AS outputTokens
       FROM (SELECT COALESCE(input_tokens, 0) AS inputTokens, COALESCE(output_tokens, 0) AS outputTokens
             FROM turns WHERE session_id IN (${matchedIds})
             GROUP BY COALESCE(trace_id, entry_id, CAST(id AS TEXT))
             UNION ALL
             SELECT MAX(COALESCE(prompt_tokens, 0) - COALESCE(cached_tokens, 0), 0),
                    COALESCE(output_tokens, 0)
             FROM traces
             WHERE root_session_id IN (${matchedIds}) AND session_id != root_session_id
               AND trace_id NOT IN (SELECT trace_id FROM turns WHERE trace_id IS NOT NULL))`,
    )
    .get(...args, ...args) as { inputTokens: unknown; outputTokens: unknown } | undefined;
  const tokens = toInt(tokenTotals?.inputTokens) + toInt(tokenTotals?.outputTokens);

  // Every directory the index holds — deliberately UNFILTERED, and deliberately
  // served rather than left to the client.
  //
  // The Sessions page builds its directory dropdown from this. Deriving it from
  // the rows instead cannot work: the page above is capped at `limit`, so a
  // directory whose sessions are all older than the cap has no row to be read
  // off, and it silently disappears from the filter — at exactly the moment the
  // page's own footnote ("narrow with the directory filter to reach older ones")
  // sends the user there, since that footnote only appears once `total` exceeds
  // the page. The same defect `tokens` above exists to prevent: a number the
  // client cannot compute from what it was handed must come from the server.
  //
  // Unfiltered because the dropdown is the way OUT of a filter as well as in.
  // Scoping it to the current `where` would leave a chosen directory as the only
  // option it lists, so the user could never switch to another one.
  const cwds = (
    ctx.dash.sqlite.prepare('SELECT DISTINCT cwd FROM sessions ORDER BY cwd').all() as Array<{
      cwd: unknown;
    }>
  ).map((row) => String(row.cwd));

  const rows = ctx.dash.sqlite.prepare(sql).all(...args, limit, offset);
  const list = rows.map((row) => ({
    id: String(row.id),
    path: String(row.path),
    cwd: String(row.cwd),
    name: row.name === null ? null : String(row.name),
    created: toInt(row.created),
    modified: toInt(row.modified),
    messageCount: toInt(row.messageCount),
    firstMessage: row.firstMessage === null ? null : String(row.firstMessage),
    models: typeof row.models === 'string' && row.models !== '' ? row.models.split(',') : [],
    inputTokens: toInt(row.inputTokens),
    outputTokens: toInt(row.outputTokens),
  }));
  return { sessions: list, total, tokens, cwds };
}

export function lookupSession(
  ctx: WorkerApiContext,
  id: string,
): { path: string; row: typeof sessions.$inferSelect } | null {
  const rows = ctx.dash.db.select().from(sessions).where(eq(sessions.id, id)).all();
  if (rows.length === 0) return null;
  return { path: rows[0].path, row: rows[0] };
}

/** The row for `id`, or the 404 every session route raises for an unknown id. */
export function requireSession(ctx: WorkerApiContext, id: string): { path: string; row: typeof sessions.$inferSelect } {
  const found = lookupSession(ctx, id);
  if (found === null) throw ApiError.notFound(`Session "${id}" not found`);
  return found;
}

/**
 * Guard: the CANONICAL session file must stay inside the canonical sessions
 * root. Both sides are resolved through `realpathSync` so a symlink at any
 * component can't escape a purely lexical containment check. When the target
 * itself does not exist (e.g. its file was already removed — the delete path
 * still cleans up its stale rows), a missing path has no symlink to follow, so
 * its existing parent is canonicalized and the final segment re-attached
 * lexically. A root that cannot be canonicalized, or a target whose parent is
 * also gone, is treated as outside (fail closed).
 */
export function insideSessionsRoot(sessionsRoot: string, path: string): boolean {
  const contained = (real: string, root: string): boolean => real === root || real.startsWith(root + sep);
  let root: string;
  try {
    root = realpathSync(sessionsRoot);
  } catch {
    return false;
  }
  try {
    return contained(realpathSync(path), root);
  } catch {
    try {
      return contained(join(realpathSync(dirname(path)), basename(path)), root);
    } catch {
      return false;
    }
  }
}

/** The 403 raised when an indexed path resolves outside the managed root. */
function requireInsideRoot(ctx: ApiPaths, path: string): void {
  if (!insideSessionsRoot(ctx.sessionsRoot, path)) {
    throw ApiError.forbidden('Session file resolves outside the managed sessions root');
  }
}

export async function handleSessionDetail(ctx: WorkerApiContext, req: ApiRequest): Promise<unknown> {
  const found = requireSession(ctx, req.params.id);
  // A GET must not disclose a transcript whose file resolves outside the managed
  // root (a stale/symlinked row). Ingestion no longer indexes symlinked files,
  // but a row predating that fix, or a path swapped to a symlink after indexing,
  // is caught here — the same guard PATCH/DELETE already apply.
  requireInsideRoot(ctx, found.path);
  const { row } = found;
  let entries: FileEntry[] | null = null;
  let transcript: TranscriptEntry[] = [];
  let transcriptError: string | undefined;
  try {
    // Read-only, byte-for-byte the way ingest reads a session (parse + in-memory
    // v1→v3 migrate, never a rewrite). `SessionManager.open` opens the file for
    // write and migrates on construction, so a plain GET of a v1 or partially
    // corrupt session would persist the migration and permanently drop malformed
    // lines — a read must never mutate the source of truth.
    entries = readSessionEntries(row.path);
  } catch (err) {
    transcriptError = err instanceof Error ? err.message : String(err);
  }
  if (entries !== null) {
    // The indexed path may resolve (in-root) to a DIFFERENT session than this row
    // — its file swapped for another transcript, or for an in-root symlink to one.
    // Containment alone can't catch that (the target is still in-root), so require
    // the parsed header id to still be THIS row before serving its metadata with
    // that file's transcript. On mismatch, reconcile the stale row and refuse.
    //
    // Through `findSessionHeader`, never a raw `.type` filter: `readSessionEntries`
    // returns the array UNMIGRATED once any record is a non-object, and a bare
    // `null` line is a valid JSONL record that `parseSessionEntries` keeps
    // verbatim. Dereferencing one throws here, ahead of the topology gate below
    // whose whole contract is to run "BEFORE any `.type`/`.id` access" — turning a
    // malformed file into a 500 carrying a raw TypeError, where every other
    // non-object record already reports the transcript invalid and still serves
    // the row. The lookup stays FIRST, though: it is the identity guard, and
    // running the topology check ahead of it would serve this row's metadata for a
    // file that had been swapped for a different, also-corrupt session.
    const header = findSessionHeader(entries) as { id?: unknown } | undefined;
    if (header === undefined || header.id !== row.id) {
      await ctx.runIngest();
      throw ApiError.conflict(`Session "${req.params.id}" no longer matches its indexed file`);
    }
    // A file mutated into a cycle/self-parent since it was indexed (ingest
    // warns+skips but leaves the stale row) would send pi's visited-set-free
    // branch walker into a non-terminating loop that no try/catch can intercept.
    // Gate the projection on the same topology guard ingest uses, surfacing the
    // failure through `transcriptError` like every other detail error here.
    if (!isValidSessionTopology(entries)) {
      transcriptError = 'Session tree is invalid (cycle, duplicate id, or non-object message); transcript unavailable';
    } else {
      const isMessage = (entry: TranscriptEntry | null): entry is TranscriptEntry => entry !== null;
      // Project the SAME active, message-bearing branch the index derives its turns
      // from, so the transcript never disagrees with the indexed turn set. When the
      // natural leaf is a detached `session_info` (e.g. after a rename), this
      // re-projects from the latest message-bearing leaf — never a flat union of
      // every abandoned branch, which would resurrect superseded turns.
      //
      // The branch walk returns a root-to-leaf parent chain, which IS the order to
      // render: a tool result follows its call because it is that call's child.
      // Re-sorting by wall clock would let a clock step backwards, or one message
      // with no parseable timestamp (`ts = 0`, which the topology guard accepts),
      // hoist a result above its call — and would then disagree with `firstMessage`,
      // which the index derives from this same chain without sorting. `ts` is for
      // display only.
      const branch = activeBranchEntries(entries);
      const callArgs = collectCallArgs(branch);
      transcript = branch.map((entry) => mapTranscriptEntry(entry, callArgs)).filter(isMessage);
    }
  }
  return {
    session: {
      id: row.id,
      path: row.path,
      cwd: row.cwd,
      name: row.name,
      created: row.created,
      modified: row.modified,
      messageCount: row.messageCount,
      firstMessage: row.firstMessage,
    },
    transcript,
    ...(transcriptError !== undefined ? { transcriptError } : {}),
  };
}

export async function handleSessionRename(ctx: WorkerApiContext, req: ApiRequest): Promise<unknown> {
  const found = requireSession(ctx, req.params.id);
  const body = requireBody(req);
  const name = (body as { name?: unknown } | null)?.name;
  if (typeof name !== 'string' || name === '') {
    throw ApiError.badRequest('Body must include a non-empty "name" string');
  }
  requireInsideRoot(ctx, found.path);
  // The indexed path may have been reused by a newer session since it was
  // indexed. Verify the file header still identifies THIS session before writing
  // its name — otherwise `appendSessionInfo` would stamp the name into a foreign
  // session's file. On mismatch, reconcile the index and refuse rather than mutate.
  if (!verifySessionFileId(found.path, req.params.id)) {
    await ctx.runIngest();
    throw ApiError.conflict(`Session "${req.params.id}" no longer matches its indexed file`);
  }
  // `SessionManager.open` migrates and rewrites the file on construction,
  // persisting only the successfully-parsed in-memory entries — so an unparseable
  // line or an incomplete trailing write would be permanently truncated from disk
  // (the GET detail handler avoids this exact call by reading read-only). There is
  // no non-destructive rename in the pi SDK, so refuse rather than lose records:
  // an unparseable line (parsed count < non-blank line count) or a malformed
  // trailing line means opening for write would drop data. A complete v1 file is
  // still safe — its migration preserves every record — so only true data loss
  // is blocked here.
  let wouldDropRecords: boolean;
  try {
    const raw = readFileSync(found.path, 'utf8');
    wouldDropRecords = countJsonlLines(raw) !== parseSessionEntries(raw).length || !lastLineParses(raw);
  } catch {
    // The file changed under us since the header check; refuse rather than open
    // for write, and reconcile the index to reflect reality.
    await ctx.runIngest();
    throw ApiError.conflict(`Session "${req.params.id}" no longer matches its indexed file`);
  }
  if (wouldDropRecords) {
    throw ApiError.conflict('Session file has incomplete/malformed records; cannot rename without data loss');
  }
  // Liveness pre-check: a session whose file was modified within LIVE_SESSION_WINDOW_MS
  // may be actively written by a concurrent agent turn (see the constant's note).
  // Renaming it would race that turn with no cross-process lock to protect us, so
  // refuse while it looks active and only append for an idle session.
  let mtimeMs: number;
  try {
    mtimeMs = statSync(found.path).mtimeMs;
  } catch {
    // The file changed under us since the checks above; reconcile and refuse rather
    // than open a file that may have been swapped out.
    await ctx.runIngest();
    throw ApiError.conflict(`Session "${req.params.id}" no longer matches its indexed file`);
  }
  if (Date.now() - mtimeMs < LIVE_SESSION_WINDOW_MS) {
    throw ApiError.conflict('Cannot rename a session that is currently active; try again once the agent is idle.');
  }
  try {
    const manager = SessionManager.open(found.path);
    manager.appendSessionInfo(name);
  } catch (err) {
    throw ApiError.internal(err instanceof Error ? err.message : 'Failed to rename session');
  }
  // Re-index so the stored name reflects the freshly appended session_info line.
  await ctx.runIngest();
  return { id: req.params.id, name };
}

export async function handleSessionDelete(ctx: WorkerApiContext, req: ApiRequest): Promise<unknown> {
  const found = requireSession(ctx, req.params.id);
  requireInsideRoot(ctx, found.path);
  // Decide what to do with the on-disk transcript BEFORE touching any rows. A bare
  // "does it still belong to this id" boolean cannot tell a file that provably
  // belongs to a DIFFERENT session (its path reused by a newer one) apart from one we
  // simply cannot verify (unreadable / corrupt / headerless):
  //   - 'matches'      → unlink the file, then drop the rows.
  //   - 'different'    → a newer session's file; must NOT unlink it. Drop our stale
  //                      rows only and let re-ingest reconcile.
  //   - 'missing'      → nothing to unlink; drop the rows.
  //   - 'unverifiable' → deleting the rows while leaving the file on disk desyncs the
  //                      DB from disk: the transcript could re-index on a later
  //                      successful ingest (reappearing after we said `deleted`), or
  //                      linger as an orphan. Reconcile the index and refuse, mirroring
  //                      the rename handler, rather than half-delete.
  // Unlike the rename above, delete deliberately carries NO liveness pre-check.
  // Rename is a read-modify-write, so a concurrent turn silently orphans a record
  // INSIDE a file that still looks intact — corruption the user never asked for and
  // cannot see. Unlinking is neither: it is exactly the outcome the user asked for,
  // behind a confirm dialog that names this hazard verbatim ("If an agent is
  // currently using this session, deleting it may orphan in-progress turns"). A live
  // agent's next append — pi's `SessionManager._persist` writes by PATH, not through
  // a held fd — recreates the path holding that entry alone; ingest reports `no
  // valid session header` and indexes nothing, and pi's own discovery drops
  // headerless files from the resume list, so the leftover is inert. Borrowing
  // LIVE_SESSION_WINDOW_MS here would not even prevent it: an agent idle at its
  // prompt (where a session spends most of its life) has a stale mtime and still
  // appends on the next message. It would only make a just-finished or just-crashed
  // session undeletable until the window passes.
  const verdict = classifySessionFile(found.path, req.params.id);
  if (verdict === 'unverifiable') {
    await ctx.runIngest();
    throw ApiError.conflict(
      `Session "${req.params.id}" file exists but could not be verified (unreadable or corrupt); delete refused to avoid orphaning it`,
    );
  }
  if (verdict === 'matches') {
    rmSync(found.path, { force: true });
  }
  const { db, sqlite } = ctx.dash;
  sqlite.exec('BEGIN');
  try {
    db.delete(turns).where(eq(turns.sessionId, req.params.id)).run();
    db.delete(sessions).where(eq(sessions.id, req.params.id)).run();
    sqlite.exec('COMMIT');
  } catch (err) {
    sqlite.exec('ROLLBACK');
    throw ApiError.internal(err instanceof Error ? err.message : 'Failed to delete session rows');
  }
  return { deleted: true, id: req.params.id };
}

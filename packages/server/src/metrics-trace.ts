/**
 * `MetricsTrace` — always-on per-turn inference telemetry, appended as JSON
 * Lines to `$HOME/.mlx-node/metrics/traces/<YYYY-MM-DD>-<pid>.jsonl`.
 *
 * This is the durable sink behind the dashboard's `traces` table
 * (`packages/dashboard/src/ingest/traces.ts`). One record is written per
 * dispatched inference turn (ok, error or cancelled) by the inference process —
 * via the {@link ./turn-trace.ts} model wrapper, so every client of the
 * `/v1/responses` and `/v1/messages` endpoints lands in the same stream.
 *
 * Contract:
 *   - Default-on; the `MLX_AGENT_METRICS` env var set to `0` / `false` / `off`
 *     (case-insensitive) is the only kill switch. The name predates the
 *     serve-path writer (it gated the retired in-agent producer); it is kept
 *     so existing operator configs keep disabling telemetry.
 *   - `record()` NEVER throws: every field is allowlisted (no free text ever
 *     lands on disk) and all fs work is wrapped — telemetry must never break
 *     an inference turn.
 */

import { appendFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';

import { resolveMlxNodeHome } from './host/paths.js';

/** Directory holding this process's `<YYYY-MM-DD>-<pid>.jsonl` trace files. */
export function metricsTraceDir(): string {
  return join(resolveMlxNodeHome(), 'metrics', 'traces');
}

/**
 * Per-turn delta of every COUNTER on the native `ColdCacheStats` — the paged
 * K/V block traffic. One entry per native counter field, named
 * `cold` + PascalCase(nativeKey); the three non-counter fields (`enabled`,
 * `root`, `quotaBytes`) describe the tier's identity rather than a turn and are
 * carried separately as {@link MetricsTraceRecord.coldEnabled} /
 * {@link MetricsTraceRecord.coldRoot}.
 *
 * This is the TS half of a cross-language invariant.
 * `__test__/server/cold-counter-fields.test.ts` derives the same names from
 * `coldCacheStats()` at runtime and demands an exact match, so a counter added
 * natively but not here — or, the failure this list exists for, one quietly
 * dropped from here — is a red test rather than an empty dashboard column.
 */
export const COLD_COUNTER_FIELDS = [
  'coldHits',
  'coldMisses',
  'coldEnqueued',
  'coldQueueDrops',
  'coldBytesWritten',
  'coldBytesRestored',
  'coldEvictions',
  'coldCorruptions',
  'coldWriteErrors',
  'coldRestoreDeclines',
] as const;

/**
 * Per-turn delta of every counter on the native `ColdSidecarStats` — the
 * recurrent / sliding-window state that lives OUTSIDE the paged pool. Named
 * `coldSidecar` + PascalCase(nativeKey).
 *
 * Deliberately a second, differently-prefixed list rather than a merge: both
 * native structs carry `enqueued` and `queueDrops`, and they count different
 * objects (blocks vs sidecars). Flattening them into one namespace would make
 * two unrelated numbers collide on one column.
 */
export const COLD_SIDECAR_FIELDS = [
  'coldSidecarCaptureReached',
  'coldSidecarChainEmpty',
  'coldSidecarBoundarySkips',
  'coldSidecarAlreadyPersisted',
  'coldSidecarEnqueued',
  'coldSidecarQueueDrops',
  'coldSidecarInstalled',
] as const;

/** Every cold-tier per-turn delta field, in native-struct order. */
export const COLD_DELTA_FIELDS = [...COLD_COUNTER_FIELDS, ...COLD_SIDECAR_FIELDS] as const;

/** A key of {@link COLD_DELTA_FIELDS}; every one is an optional `number`. */
export type ColdDeltaField = (typeof COLD_DELTA_FIELDS)[number];

export type TurnOutcome = 'ok' | 'error' | 'cancelled';
export type TurnErrorKind = 'finish_error' | 'exception';
export type TurnFailPhase = 'queued' | 'slot' | 'prepare' | 'prefill' | 'decode';
export type TurnEndpoint = 'messages' | 'responses' | 'other';
export type TurnDispatch = 'start' | 'continue' | 'continue_tool';
/** Where a turn's thinking budget came from: the request, or the server's option. */
export type TurnBudgetFrom = 'request' | 'server';
/** How a thought ended (native `reasoningClose`). */
export type TurnReasoningClose = 'model' | 'budget' | 'cycle' | 'open';

export const TURN_OUTCOMES: readonly TurnOutcome[] = ['ok', 'error', 'cancelled'];
export const TURN_ERROR_KINDS: readonly TurnErrorKind[] = ['finish_error', 'exception'];
/** The phases of a running native turn, in order; `done` is left out. */
export const TURN_FAIL_PHASES: readonly TurnFailPhase[] = ['queued', 'slot', 'prepare', 'prefill', 'decode'];
export const TURN_ENDPOINTS: readonly TurnEndpoint[] = ['messages', 'responses', 'other'];
export const TURN_DISPATCHES: readonly TurnDispatch[] = ['start', 'continue', 'continue_tool'];
export const TURN_BUDGET_FROM: readonly TurnBudgetFrom[] = ['request', 'server'];
export const TURN_REASONING_CLOSES: readonly TurnReasoningClose[] = ['model', 'budget', 'cycle', 'open'];

/** Most MTP draft positions a record carries. */
export const MTP_ACCEPT_BY_POS_MAX = 16;

/**
 * One dispatched turn's telemetry. Every field is a number, a closed enum, or
 * an identifier — never model output text, tool arguments, or prompt content.
 */
export interface MetricsTraceRecord extends Partial<Record<ColdDeltaField, number>> {
  v: 2;
  traceId: string;
  /** Write time = end of the dispatch (ms since epoch). */
  ts: number;
  pid: number;
  /** The turn's `ChatSession` cache-owner id (`ChatConfig.cacheOwnerId`). */
  sessionId?: string;

  /** Name the client asked for (may be an alias). */
  model: string;
  /** Checkpoint name the binding was first registered under. */
  modelId?: string;
  /** Family id from discovery, e.g. `qwen3_5`. */
  modelType?: string;

  outcome: TurnOutcome;
  /** Only when outcome is `error`. */
  errorKind?: TurnErrorKind;
  /** Only when outcome is not `ok` and a progress snapshot of the stream was readable. */
  failPhase?: TurnFailPhase;
  endpoint: TurnEndpoint;
  stream: boolean;
  dispatch: TurnDispatch;
  /** Sanitized native reason; `cancelled` for cancelled turns, `error` for errors. */
  finishReason: string;

  /** The wrapped dispatch call. */
  durationMs: number;
  /** Admission lane wait plus the coordinator's read wait behind a running load. */
  queueMs?: number;
  /** Time parked behind another request's model load. */
  loadWaitMs?: number;
  /** Time this request spent loading the model itself (it owned the load). */
  resolveMs?: number;
  /** Handler start → dispatch. Includes `queueMs`, `loadWaitMs` and `resolveMs`. */
  preDispatchMs?: number;
  /** Native: creation → the model thread picked the turn up. Outside `ttftMs`. Streamed turns of a native that reports it. */
  waitMs?: number;
  /** Native: time the scheduler could not start the turn (lanes or memory). Inside `ttftMs`. */
  slotMs?: number;
  /** Native: render, prefix match, cache restore and GDN replay before the prefill. Inside `ttftMs`. */
  prepareMs?: number;

  /** Always present when outcome is `ok`. Best effort otherwise. */
  promptTokens?: number;
  cachedTokens?: number;
  outputTokens?: number;
  reasoningTokens?: number;
  /** Native: how the thought ended; absent when the turn did not start in thinking or the addon does not report it. */
  reasoningClose?: TurnReasoningClose;
  /** Native: thinking tokens when a cut began. */
  reasoningCutAt?: number;
  /** Native: period in tokens of the cycle the check saw. */
  reasoningCyclePeriod?: number;
  /** Native: close tokens after the thought ended. */
  reasoningStrayCloses?: number;
  /** The thinking budget sent to native, after the window scaled it. */
  thinkingBudget?: number;
  thinkingBudgetFrom?: TurnBudgetFrom;
  /** `true` when the cycle guard was sent; else absent. */
  thinkingGuard?: true;
  /** `maxNewTokens` sent to native. */
  outputCap?: number;
  /** Non-ok streams: prefill tokens materialized before the turn ended. */
  prefillDoneTokens?: number;
  /** Usable context of the loaded model, in tokens. */
  ctxWindow?: number;

  ttftMs?: number;
  /** Prefill window: `begin_prefill` to the first token sampled. Ok streamed turns that native measured only. */
  prefillMs?: number;
  /** Tokens that prefill window processed (cached prefix excluded); written with `prefillMs`, never alone. */
  prefillTokens?: number;
  /** Written only when > 0: native reports 0.0 for a rate it could not measure. */
  decodeTps?: number;

  mtpCycles?: number;
  /** Committed tokens per cycle (drafts + 1). */
  mtpMeanAccepted?: number;
  /** Accepted draft tokens per cycle. */
  mtpMeanDrafts?: number;
  mtpMeanDepth?: number;
  /** Acceptance rate per draft position, at most {@link MTP_ACCEPT_BY_POS_MAX} entries. */
  mtpAcceptByPos?: number[];

  toolCount: number;
  toolCallCount?: number;
  imageCount: number;

  /** Turns of this model in native flight when this one began, itself included. */
  concurrency: number;
  /** Highest value of the same count while this turn was live. */
  concurrencyPeak: number;

  /** MLX active bytes at turn end. */
  memActiveBytes?: number;
  /** MLX process high-water mark at turn end. Not a per-turn peak. */
  memPeakBytes?: number;

  /**
   * GPU pipelines the process created while serving, between this turn's begin and end. Process-wide, so
   * overlapping turns share them. Absent when the native counter is off.
   */
  pipelinesCreated?: number;
  /** Time spent creating those pipelines. Written with `pipelinesCreated`. */
  pipelineCreateMs?: number;

  /** Cold-tier restore hits accrued this turn (synchronous counter — exact). */
  coldHits?: number;
  /** Cold-tier lookup misses accrued this turn (synchronous counter — exact). */
  coldMisses?: number;
  /**
   * Cold-tier bytes that LANDED this turn — credited natively only after the
   * payload sync, the commit rename and the directory fsync all succeeded, so
   * a write that failed contributes nothing here and one {@link
   * coldWriteErrors} instead.
   *
   * Advances on an async writer thread, so this delta is APPROXIMATE — it may
   * attribute a prior turn's flush to the turn that observes it, and a write
   * still in flight at the end of the turn lands in a later one. That is the
   * only sense in which it lags; it is never an enqueue-time estimate.
   */
  coldBytesWritten?: number;
  /** Cold-tier bytes read back on validated hits this turn (synchronous — exact). */
  coldBytesRestored?: number;
  /**
   * Canonical cold-cache root this turn's tier was operating in, as produced by
   * `canonicalCacheRoot` — the JOIN KEY the dashboard scopes its hit-rate and
   * trend to. Written ONLY when the tier was actually open; a turn that ran
   * with the tier off carries `coldEnabled: false` and NO root, and a record
   * written by a build that predates this field carries neither. Those two
   * states are deliberately distinguishable: the dashboard reports "recorded
   * before cache attribution existed" separately from "the tier was off".
   */
  coldRoot?: string;
  /** Whether the cold tier was open for this turn (`ColdCacheStats.enabled`). */
  coldEnabled?: boolean;
  /**
   * Cold-tier write-queue submissions accrued this turn. Incremented on the
   * calling thread, so this delta is EXACT.
   */
  coldEnqueued?: number;
  /**
   * Cold-tier writes REFUSED at admission this turn because the bounded queue
   * was full. Incremented on the calling thread, so this delta is EXACT.
   *
   * Now admission refusals only: the commit-rename-failure arm used to be
   * counted here too, which put one event under a name describing a different
   * cause. It is a {@link coldWriteErrors} instead, and the two are disjoint —
   * never sum them.
   */
  coldQueueDrops?: number;
  /**
   * Cold-tier objects evicted by quota enforcement this turn. Advances on the
   * background writer thread — APPROXIMATE, same caveat as `coldBytesWritten`.
   */
  coldEvictions?: number;
  /**
   * Cold-tier objects that failed validation on restore this turn. Incremented
   * on the calling thread, so this delta is EXACT. Always accompanied by a
   * miss, i.e. corruptions are a SUBSET of `coldMisses` — never add the two.
   */
  coldCorruptions?: number;
  /**
   * CUMULATIVE corruptions since the tier opened in this process, not a delta.
   * Carried because the acceptance bar is "corruptions must be 0" and a delta
   * alone cannot prove it: a turn that aborts or errors carries no cold
   * fields, so a corruption during it lands in no delta at all. The next
   * successful turn's absolute total still includes it, so `MAX(total) > 0`
   * over any window is the sound "did this ever happen" latch.
   */
  coldCorruptionsTotal?: number;
  /**
   * CUMULATIVE queue drops since the tier opened in this process, for the same
   * reason as {@link coldCorruptionsTotal}: an errored turn's dropped writes
   * would otherwise be invisible.
   */
  coldQueueDropsTotal?: number;
  /**
   * Cold-tier writes this turn that the queue ACCEPTED and that never reached
   * disk — a read-only, full or unmounted cache root, a quota the object
   * cannot fit, a failed rename, a failed fsync.
   *
   * The native writer is deliberately fail-open (a broken cache root must not
   * change a single emitted token) and it returns its error to nobody, so
   * before this counter existed the entire class was invisible: a turn against
   * an unwritable root reported `coldQueueDrops 0`, `coldCorruptions 0`, an
   * empty stderr and a clean exit. Advances on the background writer thread,
   * so APPROXIMATE per turn in the same way {@link coldBytesWritten} is —
   * non-zero over a window is the signal, not the exact attribution.
   */
  coldWriteErrors?: number;
  /**
   * Restores REFUSED this turn: the tier held candidate blocks and the walk
   * handed back none of them.
   *
   * Neither a hit nor a miss, because both of those count per-block lookups
   * and a refusal happens instead of one — which is why a refused restore
   * reported `coldHits 0, coldMisses 0`, exactly the row a turn that never
   * consulted the tier produces. Incremented on the calling thread, so EXACT.
   * The reason (`no_backed_boundary`, `parent_chain_unavailable`,
   * `restore_short_of_boundary`) and the block geometry go to the
   * `[MLX_TRACE] paged cold_restore_declined` line.
   */
  coldRestoreDeclines?: number;
  /**
   * CUMULATIVE write errors since the tier opened in this process, for the
   * same reason as {@link coldCorruptionsTotal} — and more sharply, since this
   * counter advances on the background writer: the error covering the last
   * turn before a crash reaches no delta at all.
   *
   * No such total exists for {@link coldRestoreDeclines} on purpose. A "did it
   * ever happen" latch is only meaningful for a counter whose healthy value is
   * zero; the first turn of any new prompt legitimately declines, so a latch
   * there would be pinned on from the first minute and mean nothing.
   */
  coldWriteErrorsTotal?: number;
  /**
   * Turns this turn's process spent reaching a family's sidecar capture. Every
   * other `coldSidecar*` counter is a sub-count of this one, so `0` here while
   * blocks were written means the finalize path never called the capture at
   * all — a different bug from a capture that ran and declined.
   */
  coldSidecarCaptureReached?: number;
  /**
   * Sidecar captures that found no whole persisted block to anchor recurrent
   * state under.
   */
  coldSidecarChainEmpty?: number;
  /**
   * Sidecar captures whose chain covered blocks but where no retained
   * checkpoint sat at or below its reach — the "ladder collapsed to its
   * deepest rung" signature.
   */
  coldSidecarBoundarySkips?: number;
  /**
   * Sidecar captures that selected a boundary already on disk. The STEADY
   * STATE of a repeated prompt, not a fault: without it a healthy run
   * (`coldSidecarEnqueued == 0` because the first turn already wrote it) is
   * indistinguishable from a broken one.
   */
  coldSidecarAlreadyPersisted?: number;
  /**
   * Sidecars handed to the writer queue this turn. A SUBSET of
   * {@link coldEnqueued}, which counts every object that took a queue slot —
   * so summing them double-counts each sidecar.
   */
  coldSidecarEnqueued?: number;
  /**
   * Sidecars the writer queue refused because it was full. Likewise a subset
   * of {@link coldQueueDrops}, not a separate population.
   */
  coldSidecarQueueDrops?: number;
  /**
   * Restored sidecars a family INSTALLED as its live per-turn state. The one
   * read-side sidecar counter, and the only thing that separates "restored and
   * used" from "restored, then silently re-derived by a full O(prefix) replay":
   * the replay produces correct state, so `cachedTokens`, `coldHits`,
   * `coldCorruptions` and the output text are all identical either way.
   */
  coldSidecarInstalled?: number;
}

export type MetricsTraceInput = Omit<MetricsTraceRecord, 'v' | 'pid'>;

export function metricsEnvDisabled(): boolean {
  const raw = process.env.MLX_AGENT_METRICS;
  if (raw === undefined) return false;
  const normalized = raw.trim().toLowerCase();
  return normalized === '0' || normalized === 'false' || normalized === 'off';
}

/** A finite number, or `undefined` if the value is absent / non-finite. */
function finite(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

/** A finite number ≥ 0, or `undefined`. */
function nonNeg(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : undefined;
}

/** A finite number > 0, or `undefined`. */
function positive(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : undefined;
}

export function nonNegInt(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : undefined;
}

export function enumOf<T extends string>(value: unknown, allowed: readonly T[]): T | undefined {
  if (typeof value !== 'string') return undefined;
  for (const candidate of allowed) {
    if (candidate === value) return candidate;
  }
  return undefined;
}

const SAFE_REASON_MAX = 32;

/** `value` when it is 1–32 chars of `a-z`, `0-9`, `_`; otherwise `'other'`. */
export function safeReason(value: unknown): string {
  if (typeof value !== 'string' || value.length === 0 || value.length > SAFE_REASON_MAX) return 'other';
  for (let i = 0; i < value.length; i++) {
    const c = value.charCodeAt(i);
    const lower = c >= 97 && c <= 122;
    const digit = c >= 48 && c <= 57;
    if (!lower && !digit && c !== 95) return 'other';
  }
  return value;
}

/** First {@link MTP_ACCEPT_BY_POS_MAX} finite entries, each clamped to [0, 1] and rounded to 4 decimals. */
export function rateList(value: unknown): number[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const out: number[] = [];
  for (const entry of value as unknown[]) {
    if (out.length >= MTP_ACCEPT_BY_POS_MAX) break;
    if (typeof entry !== 'number' || !Number.isFinite(entry)) continue;
    const clamped = entry < 0 ? 0 : entry > 1 ? 1 : entry;
    out.push(Math.round(clamped * 1e4) / 1e4);
  }
  return out;
}

function nonEmptyString(value: unknown): string | undefined {
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

export class MetricsTrace {
  readonly enabled: boolean;
  private readonly dir: string;
  private readonly now: () => number;
  /** The day file whose directory is known to exist. */
  private readyFile: string | null = null;

  constructor(opts?: { dir?: string; now?: () => number }) {
    this.enabled = !metricsEnvDisabled();
    this.dir = opts?.dir ?? metricsTraceDir();
    this.now = opts?.now ?? Date.now;
  }

  /** `<dir>/<YYYY-MM-DD>-<pid>.jsonl` — UTC date so rotation is timezone-stable. */
  currentFile(): string {
    const date = new Date(this.now()).toISOString().slice(0, 10);
    return join(this.dir, `${date}-${process.pid}.jsonl`);
  }

  /**
   * Append one allowlisted JSON line. Never throws: a broken sink must not
   * surface into the inference path. Excess input properties are dropped — the
   * record is rebuilt field by field so free text can never reach disk. A
   * record whose `outcome`, `endpoint` or `dispatch` is not a known value is
   * dropped.
   */
  record(rec: MetricsTraceInput): void {
    if (!this.enabled) return;
    try {
      const outcome = enumOf(rec.outcome, TURN_OUTCOMES);
      const endpoint = enumOf(rec.endpoint, TURN_ENDPOINTS);
      const dispatch = enumOf(rec.dispatch, TURN_DISPATCHES);
      if (outcome === undefined || endpoint === undefined || dispatch === undefined) return;
      if (typeof rec.traceId !== 'string' || typeof rec.model !== 'string') return;
      const out: MetricsTraceRecord = {
        v: 2,
        traceId: rec.traceId,
        ts: finite(rec.ts) ?? this.now(),
        pid: process.pid,
        model: rec.model,
        outcome,
        endpoint,
        stream: rec.stream === true,
        dispatch,
        finishReason: safeReason(rec.finishReason),
        durationMs: nonNeg(rec.durationMs) ?? 0,
        toolCount: nonNegInt(rec.toolCount) ?? 0,
        imageCount: nonNegInt(rec.imageCount) ?? 0,
        concurrency: nonNegInt(rec.concurrency) ?? 0,
        concurrencyPeak: nonNegInt(rec.concurrencyPeak) ?? 0,
      };
      const sessionId = nonEmptyString(rec.sessionId);
      if (sessionId !== undefined) out.sessionId = sessionId;
      const modelId = nonEmptyString(rec.modelId);
      if (modelId !== undefined) out.modelId = modelId;
      const modelType = nonEmptyString(rec.modelType);
      if (modelType !== undefined) out.modelType = modelType;
      if (outcome === 'error') {
        const errorKind = enumOf(rec.errorKind, TURN_ERROR_KINDS);
        if (errorKind !== undefined) out.errorKind = errorKind;
      }
      if (outcome !== 'ok') {
        const failPhase = enumOf(rec.failPhase, TURN_FAIL_PHASES);
        if (failPhase !== undefined) out.failPhase = failPhase;
      }
      for (const key of [
        'queueMs',
        'loadWaitMs',
        'resolveMs',
        'preDispatchMs',
        'ttftMs',
        'waitMs',
        'slotMs',
        'prepareMs',
      ] as const) {
        const value = nonNeg(rec[key]);
        if (value !== undefined) out[key] = value;
      }
      for (const key of [
        'promptTokens',
        'cachedTokens',
        'outputTokens',
        'reasoningTokens',
        'reasoningCutAt',
        'reasoningStrayCloses',
        'thinkingBudget',
        'outputCap',
        'prefillDoneTokens',
        'ctxWindow',
        'mtpCycles',
        'toolCallCount',
      ] as const) {
        const value = nonNegInt(rec[key]);
        if (value !== undefined) out[key] = value;
      }
      const reasoningClose = enumOf(rec.reasoningClose, TURN_REASONING_CLOSES);
      if (reasoningClose !== undefined) out.reasoningClose = reasoningClose;
      const cyclePeriod = nonNegInt(rec.reasoningCyclePeriod);
      if (cyclePeriod !== undefined && cyclePeriod > 0) out.reasoningCyclePeriod = cyclePeriod;
      const budgetFrom = enumOf(rec.thinkingBudgetFrom, TURN_BUDGET_FROM);
      if (budgetFrom !== undefined) out.thinkingBudgetFrom = budgetFrom;
      if (rec.thinkingGuard === true) out.thinkingGuard = true;
      const prefillMs = positive(rec.prefillMs);
      const prefillTokens = nonNegInt(rec.prefillTokens);
      if (outcome === 'ok' && prefillMs !== undefined && prefillTokens !== undefined && prefillTokens > 0) {
        out.prefillMs = prefillMs;
        out.prefillTokens = prefillTokens;
      }
      const decodeTps = positive(rec.decodeTps);
      if (decodeTps !== undefined) out.decodeTps = decodeTps;
      for (const key of [
        'mtpMeanAccepted',
        'mtpMeanDrafts',
        'mtpMeanDepth',
        'memActiveBytes',
        'memPeakBytes',
      ] as const) {
        const value = nonNeg(rec[key]);
        if (value !== undefined) out[key] = value;
      }
      const pipelinesCreated = nonNegInt(rec.pipelinesCreated);
      const pipelineCreateMs = nonNeg(rec.pipelineCreateMs);
      if (pipelinesCreated !== undefined && pipelineCreateMs !== undefined) {
        out.pipelinesCreated = pipelinesCreated;
        out.pipelineCreateMs = pipelineCreateMs;
      }
      const mtpAcceptByPos = rateList(rec.mtpAcceptByPos);
      if (mtpAcceptByPos !== undefined) out.mtpAcceptByPos = mtpAcceptByPos;
      // Every cold-tier delta comes off ONE list, so a counter can only be
      // dropped from the JSONL by being dropped from `COLD_DELTA_FIELDS` —
      // which `__test__/server/cold-counter-fields.test.ts` pins to the native structs.
      for (const key of COLD_DELTA_FIELDS) {
        const value = finite(rec[key]);
        if (value !== undefined) out[key] = value;
      }
      // A cache identity is only meaningful for a tier that was actually open;
      // an empty root would create a bucket no dashboard root can ever match.
      if (typeof rec.coldRoot === 'string' && rec.coldRoot.length > 0) out.coldRoot = rec.coldRoot;
      if (typeof rec.coldEnabled === 'boolean') out.coldEnabled = rec.coldEnabled;
      const coldCorruptionsTotal = finite(rec.coldCorruptionsTotal);
      if (coldCorruptionsTotal !== undefined) out.coldCorruptionsTotal = coldCorruptionsTotal;
      const coldQueueDropsTotal = finite(rec.coldQueueDropsTotal);
      if (coldQueueDropsTotal !== undefined) out.coldQueueDropsTotal = coldQueueDropsTotal;
      const coldWriteErrorsTotal = finite(rec.coldWriteErrorsTotal);
      if (coldWriteErrorsTotal !== undefined) out.coldWriteErrorsTotal = coldWriteErrorsTotal;

      this.append(`${JSON.stringify(out)}\n`);
    } catch {
      // Telemetry is best-effort; a broken sink must never break inference.
    }
  }

  private append(line: string): void {
    const file = this.currentFile();
    if (this.readyFile !== file) {
      mkdirSync(dirname(file), { recursive: true });
      this.readyFile = file;
    }
    try {
      appendFileSync(file, line);
    } catch {
      // The directory may have been removed under a cached flag: heal once.
      this.readyFile = null;
      mkdirSync(dirname(file), { recursive: true });
      appendFileSync(file, line);
      this.readyFile = file;
    }
  }
}

/**
 * `MetricsTrace` — always-on per-turn inference telemetry, appended as JSON
 * Lines to `$HOME/.mlx-node/metrics/traces/<YYYY-MM-DD>-<pid>.jsonl`.
 *
 * This is a durable sink that complements the transient in-memory
 * {@link ./performance-status.ts} WeakMap (which only feeds the live TUI
 * footer). One record is written per successful inference turn so the
 * dashboard can correlate throughput, cache reuse, and cold-tier deltas back
 * to the pi session that produced them via `mlxTraceId`.
 *
 * Contract:
 *   - Default-on; the `MLX_AGENT_METRICS` env var set to `0` / `false` / `off`
 *     (case-insensitive) is the only kill switch.
 *   - `record()` NEVER throws: every field is allowlisted (no free text ever
 *     lands on disk) and all fs work is wrapped — telemetry must never break
 *     an inference turn.
 */

import { appendFileSync, mkdirSync } from 'node:fs';
import { dirname, join } from 'node:path';

import { metricsTraceDir } from '../paths.js';

/**
 * Per-turn delta of every COUNTER on the native `ColdCacheStats` — the paged
 * K/V block traffic. One entry per native counter field, named
 * `cold` + PascalCase(nativeKey); the three non-counter fields (`enabled`,
 * `root`, `quotaBytes`) describe the tier's identity rather than a turn and are
 * carried separately as {@link MetricsTraceRecord.coldEnabled} /
 * {@link MetricsTraceRecord.coldRoot}.
 *
 * This is the TS half of a cross-language invariant.
 * `__test__/cold-counter-fields.test.ts` derives the same names from
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
  'coldSidecarRestoreSuppressed',
] as const;

/** Every cold-tier per-turn delta field, in native-struct order. */
export const COLD_DELTA_FIELDS = [...COLD_COUNTER_FIELDS, ...COLD_SIDECAR_FIELDS] as const;

/** A key of {@link COLD_DELTA_FIELDS}; every one is an optional `number`. */
export type ColdDeltaField = (typeof COLD_DELTA_FIELDS)[number];

/**
 * One inference turn's telemetry. Every field is a number, a small
 * enumerated string (`finishReason`), or an identifier — never model output
 * text, tool arguments, or prompt content.
 */
export interface MetricsTraceRecord extends Partial<Record<ColdDeltaField, number>> {
  /** Schema version. */
  v: 1;
  /** Join key minted by `TurnEmitter`; also stamped on the pi message as `mlxTraceId`. */
  traceId: string;
  /** Wall-clock write time (ms since epoch). */
  ts: number;
  /** pi per-request session id (parent vs each subagent differ). */
  sessionId?: string;
  /** Root pi session id the turn was SUBMITTED under (snapshotted at submit). */
  rootSessionId?: string;
  /** Root pi session JSONL file path the turn was SUBMITTED under (snapshotted at submit). */
  rootSessionFile?: string;
  /** Model id served this turn (`mlx/<dir-name>`'s `<dir-name>`). */
  model: string;
  /** Turn duration (ms) bracketing resident selection + prefill + decode. */
  durationMs: number;
  /**
   * Queue + cold-load wait (ms) before native work began this turn — the gap
   * between turn submission and the serialized inference callback firing.
   * Subtract from `durationMs` to isolate execution-only latency.
   */
  queueMs?: number;
  /** `true` when the model was already warm/resident, `false` on a cold load/swap. */
  resident?: boolean;
  /** Native finish reason (`stop` / `length` / `tool_calls` / …). */
  finishReason: string;
  promptTokens: number;
  cachedTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  ttftMs?: number;
  prefillTps?: number;
  decodeTps?: number;
  mtpCycles?: number;
  mtpMeanAccepted?: number;
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
   * alone cannot prove it: a turn that aborts or errors never reaches
   * `record()`, so a corruption during it lands in no delta at all. The next
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
  /**
   * Restored prefixes a family THREW AWAY this turn, releasing the request and
   * restarting the turn cold (gemma4's large-sliding-restore suppression).
   *
   * A different event from {@link coldRestoreDeclines}: there the walk refused
   * to serve anything, here it served and the family discarded it. This one is
   * therefore preceded by real `coldHits` and `coldBytesRestored`, so the turn
   * reads as successful reuse right up to the point where it recomputes the
   * whole prompt anyway.
   */
  coldSidecarRestoreSuppressed?: number;
}

type MetricsTraceInput = Omit<MetricsTraceRecord, 'v'>;

function envDisabled(): boolean {
  const raw = process.env.MLX_AGENT_METRICS;
  if (raw === undefined) return false;
  const normalized = raw.trim().toLowerCase();
  return normalized === '0' || normalized === 'false' || normalized === 'off';
}

/** A finite number, or `undefined` if the value is absent / non-finite. */
function finite(value: number | undefined): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

export class MetricsTrace {
  readonly enabled: boolean;
  private readonly dir: string;
  private readonly now: () => number;

  constructor(opts?: { dir?: string; now?: () => number }) {
    this.enabled = !envDisabled();
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
   * record is rebuilt field by field so free text can never reach disk.
   */
  record(rec: MetricsTraceInput): void {
    if (!this.enabled) return;
    try {
      const out: MetricsTraceRecord = {
        v: 1,
        traceId: rec.traceId,
        ts: rec.ts,
        model: rec.model,
        durationMs: rec.durationMs,
        finishReason: rec.finishReason,
        promptTokens: rec.promptTokens,
        cachedTokens: rec.cachedTokens,
        outputTokens: rec.outputTokens,
        reasoningTokens: rec.reasoningTokens,
      };
      if (rec.sessionId !== undefined) out.sessionId = rec.sessionId;
      if (rec.rootSessionId !== undefined) out.rootSessionId = rec.rootSessionId;
      if (rec.rootSessionFile !== undefined) out.rootSessionFile = rec.rootSessionFile;
      const queueMs = finite(rec.queueMs);
      if (queueMs !== undefined) out.queueMs = queueMs;
      if (typeof rec.resident === 'boolean') out.resident = rec.resident;
      const ttftMs = finite(rec.ttftMs);
      if (ttftMs !== undefined) out.ttftMs = ttftMs;
      const prefillTps = finite(rec.prefillTps);
      if (prefillTps !== undefined) out.prefillTps = prefillTps;
      const decodeTps = finite(rec.decodeTps);
      if (decodeTps !== undefined) out.decodeTps = decodeTps;
      const mtpCycles = finite(rec.mtpCycles);
      if (mtpCycles !== undefined) out.mtpCycles = mtpCycles;
      const mtpMeanAccepted = finite(rec.mtpMeanAccepted);
      if (mtpMeanAccepted !== undefined) out.mtpMeanAccepted = mtpMeanAccepted;
      // Every cold-tier delta comes off ONE list, so a counter can only be
      // dropped from the JSONL by being dropped from `COLD_DELTA_FIELDS` —
      // which `__test__/cold-counter-fields.test.ts` pins to the native structs.
      // Spelling them out here is what let four `coldCacheStats()` counters sit
      // unwritten for the life of the feature.
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

      const file = this.currentFile();
      mkdirSync(dirname(file), { recursive: true });
      appendFileSync(file, `${JSON.stringify(out)}\n`);
    } catch {
      // Telemetry is best-effort; a broken sink must never break inference.
    }
  }
}

/**
 * `TurnTraceModel` — the `SessionCapableModel` wrapper that produces one
 * {@link MetricsTraceRecord} per completed inference turn.
 *
 * Installed once per `SessionRegistry` — the single allocation point of every
 * `ChatSession` the serve process hands out — so it sits on ALL of the
 * registry's native dispatch surface at once: `chatSessionStart` /
 * `chatSessionContinue` / `chatSessionContinueTool` and the three streaming
 * variants, for both `/v1/responses` and `/v1/messages`, streaming or not,
 * warm hit or cold replay. There is deliberately no endpoint-level hook:
 * the endpoints hand work to the session and only learn the wire shape
 * afterwards; the model call is the one place the final `ChatResult` /
 * terminal `ChatStreamEvent` exists before the turn is committed.
 *
 * Turn boundaries: the cold-tier counters are process-wide and cumulative, so
 * the baseline is snapshotted INSIDE the wrapped call (the actual dispatch
 * moment) and diffed when the turn completes successfully. A turn that
 * throws, aborts, or ends on `finishReason: 'error'` never records — but it
 * still ADVANCES the shared watermarks ({@link TurnTraceModel.discardTurn}),
 * so its counter increments are consumed rather than inherited by the next
 * successful turn's delta. That advance is floored at the oldest LIVE turn's
 * baseline: a failure finishing mid-overlap consumes only up to what a
 * survivor will still see, never the survivor's own increments. On
 * continuously batched models
 * (`maxConcurrentSequences() > 1`) two turns can overlap in native flight:
 * each turn then reports the counter advance since the PREVIOUS record's
 * high-water mark ({@link TurnWatermarks}), so attribution between
 * overlapping turns is approximate while the deltas still sum to the
 * counter's real advance — the dashboard SUMs these columns. `durationMs`
 * therefore measures the wrapped dispatch call itself; admission-lane queue
 * waits happen upstream and are not included.
 */

import { randomUUID } from 'node:crypto';

import {
  coldCacheStats,
  coldSidecarStats,
  type ChatConfig,
  type ChatMessage,
  type ChatResult,
  type ColdCacheStats,
  type ColdSidecarStats,
  type PerformanceMetrics,
} from '@mlx-node/core';
import { mtpAutoDefaultAllowed, type ChatStreamEvent, type SessionCapableModel } from '@mlx-node/lm';
import { canonicalCacheRoot } from '@mlx-node/lm/cold-tier';

import { MetricsTrace, type MetricsTraceInput } from './metrics-trace.js';

/** Read the process-wide cold-tier snapshot; the native addon may be absent (unit tests). */
function safeColdStats(): ColdCacheStats | undefined {
  try {
    return coldCacheStats();
  } catch {
    return undefined;
  }
}

/**
 * Read the process-wide sidecar counters. Separate from {@link safeColdStats}
 * because it reads a different native struct that has no `enabled` gate: the
 * counters live in a plain static, not in the tier, so they are valid even
 * when the tier never opened — which is exactly the case a run with zero
 * sidecars needs distinguished.
 */
function safeSidecarStats(): ColdSidecarStats | undefined {
  try {
    return coldSidecarStats();
  } catch {
    return undefined;
  }
}

/**
 * `ChatConfig` plus the trace-only field the serve endpoints stamp per
 * request: the registry name the request resolved through. The napi object
 * boundary drops unknown props, so carrying it on the config costs nothing
 * — and it is the only per-request handle a shared model wrapper has:
 * aliased registrations reuse one `SessionRegistry`, so the registration
 * name alone cannot attribute a turn to the alias the client asked for.
 * Never written by model code; read only here.
 */
export interface TracedChatConfig extends ChatConfig {
  modelAlias?: string;
}

/** Injectable seams for {@link TurnTraceModel} (unit tests). */
export interface TurnTraceDeps {
  /**
   * Fallback model name for turns the request did not stamp: a static
   * registration name, or a resolver consulted per turn so an unregistered
   * first alias does not leave traces attributed to a dead name. Either
   * side may resolve to `'unknown'`.
   */
  modelName?: string | (() => string | undefined);
  /** Durable telemetry sink; defaults to a fresh {@link MetricsTrace}. */
  trace?: Pick<MetricsTrace, 'record'>;
  /** Process-wide cold-tier reader; defaults to the native addon. */
  coldStats?: () => ColdCacheStats | undefined;
  /** Process-wide sidecar-counter reader; defaults to the native addon. */
  sidecarStats?: () => ColdSidecarStats | undefined;
  /** Clock for `ts`, `durationMs`, and the trace filename. Default: `Date.now`. */
  now?: () => number;
  /**
   * High-water marks for the process-wide counters. Default: a module-level
   * store SHARED by every wrapper in the process — see {@link TurnWatermarks}.
   * Tests inject a fresh one so records stay deterministic.
   */
  watermarks?: TurnWatermarks;
}

/**
 * Per-counter high-water marks, one per cold/sidecar field. The native
 * counters are process-global and cumulative, so diffing each turn against
 * its OWN begin-snapshot double-counts whatever another turn increments
 * during the overlap — two turns around one hit would record 1 + 2 for 2
 * total, and the dashboard `SUM`s these columns. Recording each turn's delta
 * since the PREVIOUS record's mark instead makes intervals sequential:
 * attribution between overlapping turns stays approximate, but the sum of
 * deltas equals the counter's real advance at every point. Per-field marks
 * (not per-struct) so a snapshot missing one struct cannot lose the other's
 * attribution. A field that DECREASES below its mark means the counter
 * reset; the new cumulative value is the delta, not a negative.
 */
export interface TurnWatermarks {
  /** Field-name → last recorded counter value for the cold tier. */
  cold: Record<string, number>;
  /** Field-name → last recorded counter value for the sidecar counters. */
  sidecar: Record<string, number>;
  /**
   * Begin snapshots of turns still in flight. {@link TurnTraceModel.discardTurn}
   * floors its mark advance at the oldest live baseline so a failed turn
   * cannot consume counter increments a surviving turn will still diff — a
   * failure finishing mid-overlap would otherwise eat a concurrent success's
   * hits/misses and the dashboard would undercount real cache activity.
   * Lazily created when absent, so injected literals keep working.
   */
  live?: Set<{ cold: ColdCacheStats | undefined; sidecar: ColdSidecarStats | undefined }>;
}

/** Process-wide default: every `TurnTraceModel` in the serve process shares it. */
const SHARED_WATERMARKS: TurnWatermarks = { cold: {}, sidecar: {} };

/**
 * Per-turn recording context. `null` when the trace sink is disabled — the
 * wrapper then costs one check per call, no snapshots at all.
 */
interface TurnContext {
  traceId: string;
  sessionId: string | undefined;
  /** Registry name this request resolved through; wins over the binding name. */
  modelAlias: string | undefined;
  startMs: number;
  cold: ColdCacheStats | undefined;
  sidecar: ColdSidecarStats | undefined;
  done: boolean;
}

/**
 * Members of {@link SessionCapableModel} whose presence is optional and
 * whose calls are pure queries — forwarded verbatim so the wrapper is
 * invisible to `ChatSession` feature detection (`hasBlockPagedCache`,
 * `applyChatTemplate`, …). The wrapper MUST NOT define one the inner model
 * lacks: `hasBlockPagedCache` absent means "not paged". The single exception
 * is `mtpAutoEnabled`: when the inner lacks it, the constructor synthesizes a
 * forwarder to {@link mtpAutoDefaultAllowed} so `ChatSession`'s proto-walk
 * fallback (suppressed for `NemotronHModel`) still sees the INNER chain —
 * the wrapper's own `TurnTraceModel` proto would silently re-enable MTP.
 */
/** Cold-tier counters diffed per turn — the keys {@link TurnWatermarks.cold} tracks. */
const COLD_DELTA_KEYS = [
  'hits',
  'misses',
  'bytesWritten',
  'bytesRestored',
  'enqueued',
  'queueDrops',
  'evictions',
  'corruptions',
  'writeErrors',
  'restoreDeclines',
] as const;

/** Sidecar counters diffed per turn — the keys {@link TurnWatermarks.sidecar} tracks. */
const SIDECAR_DELTA_KEYS = [
  'captureReached',
  'chainEmpty',
  'boundarySkips',
  'alreadyPersisted',
  'enqueued',
  'queueDrops',
  'installed',
] as const;

const OPTIONAL_MEMBER_NAMES = [
  'applyChatTemplate',
  'expandedPromptTokenCount',
  'contextLimits',
  'supportsImages',
  'supportsReplayReasoningCapture',
  'replaysAssistantRawText',
  'hasBlockPagedCache',
  'normalizeReasoningEffort',
  'maxConcurrentSequences',
  'hasMtpWeights',
  'mtpAutoEnabled',
  'releaseCacheOwner',
] as const;

/**
 * The terminal outcome of one dispatch. `ChatResult` (non-streaming) and the
 * `done: true` `ChatStreamEvent` (streaming) both satisfy it — the union's
 * shared telemetry fields only; free-text members never reach the record.
 */
interface TurnFinal {
  finishReason: string;
  promptTokens: number;
  numTokens: number;
  reasoningTokens: number;
  cachedTokens?: number;
  performance?: PerformanceMetrics;
}

export class TurnTraceModel implements SessionCapableModel {
  private readonly inner: SessionCapableModel;
  private readonly deps: TurnTraceDeps;
  private readonly trace: Pick<MetricsTrace, 'record'>;

  // Optional members installed verbatim on this instance in the constructor
  // when — and only when — the inner model defines them. `declare` so the
  // `implements` check sees the optional surface while emitting nothing.
  declare applyChatTemplate: SessionCapableModel['applyChatTemplate'];
  declare expandedPromptTokenCount: SessionCapableModel['expandedPromptTokenCount'];
  declare contextLimits: SessionCapableModel['contextLimits'];
  declare supportsImages: SessionCapableModel['supportsImages'];
  declare supportsReplayReasoningCapture: SessionCapableModel['supportsReplayReasoningCapture'];
  declare replaysAssistantRawText: SessionCapableModel['replaysAssistantRawText'];
  declare hasBlockPagedCache: SessionCapableModel['hasBlockPagedCache'];
  declare normalizeReasoningEffort: SessionCapableModel['normalizeReasoningEffort'];
  declare maxConcurrentSequences: SessionCapableModel['maxConcurrentSequences'];
  declare hasMtpWeights: SessionCapableModel['hasMtpWeights'];
  declare mtpAutoEnabled: SessionCapableModel['mtpAutoEnabled'];
  declare releaseCacheOwner: SessionCapableModel['releaseCacheOwner'];

  constructor(inner: SessionCapableModel, deps: TurnTraceDeps = {}) {
    this.inner = inner;
    this.deps = deps;
    // Resolved per instance — not a module-scope shared sink — so a later
    // `MLX_AGENT_METRICS` change still reaches `new MetricsTrace()`'s
    // construction-time env read for the NEXT registry, matching how the
    // retired producer constructed its sink at extension creation.
    this.trace = deps.trace ?? new MetricsTrace();
    for (const name of OPTIONAL_MEMBER_NAMES) {
      const member = (inner as unknown as Record<string, unknown>)[name];
      if (typeof member === 'function') {
        Object.defineProperty(this, name, { value: member.bind(inner) });
      }
    }
    // Synthesized, not mirrored: resolves the same verdict a bare ChatSession
    // would reach by walking the inner's prototype chain.
    if (typeof inner.mtpAutoEnabled !== 'function') {
      Object.defineProperty(this, 'mtpAutoEnabled', { value: () => mtpAutoDefaultAllowed(inner) });
    }
  }

  private get now(): () => number {
    return this.deps.now ?? Date.now;
  }

  /**
   * Open a turn context: mint the trace id, stamp the start clock, and
   * snapshot the process-wide cold counters BEFORE native work begins. The
   * snapshot ALSO seeds any unmarked counter field — the first turn a field
   * is measured still gets its own delta, after which marks chain
   * intervals so recorded deltas always sum to the counter's real advance.
   */
  private beginTurn(config: ChatConfig | null | undefined): TurnContext | null {
    const trace = this.trace;
    if (trace instanceof MetricsTrace && !trace.enabled) return null;
    const marks = this.deps.watermarks ?? SHARED_WATERMARKS;
    const cold = (this.deps.coldStats ?? safeColdStats)();
    if (cold) {
      for (const key of COLD_DELTA_KEYS) {
        const value = cold[key];
        if (typeof value === 'number') marks.cold[key] ??= value;
      }
    }
    const sidecar = (this.deps.sidecarStats ?? safeSidecarStats)();
    if (sidecar) {
      for (const key of SIDECAR_DELTA_KEYS) {
        const value = sidecar[key];
        if (typeof value === 'number') marks.sidecar[key] ??= value;
      }
    }
    const ctx: TurnContext = {
      traceId: randomUUID(),
      sessionId: typeof config?.cacheOwnerId === 'string' ? config.cacheOwnerId : undefined,
      // The requesting alias outranks the binding name: one SessionRegistry
      // (and one wrapper) serves every alias of a model.
      modelAlias: (config as TracedChatConfig | null | undefined)?.modelAlias,
      startMs: this.now(),
      cold,
      sidecar,
      done: false,
    };
    (marks.live ??= new Set()).add(ctx);
    return ctx;
  }

  /**
   * Diff the end-of-turn counters against the per-field high-water marks and
   * append the record. Rebuilt field by field from the terminal result —
   * never the free-text members — then handed to the allowlisted sink.
   *
   * The SIDECAR diff is its own guard, deliberately not folded into the cold
   * block: the sidecar reader never consults the tier, so it keeps reporting
   * when the tier failed to open — a run with `coldHits: 0` and no tier is
   * exactly when "did the capture even run?" is the question.
   */
  private finishTurn(ctx: TurnContext, final: TurnFinal): void {
    if (ctx.done) return;
    ctx.done = true;
    (this.deps.watermarks ?? SHARED_WATERMARKS).live?.delete(ctx);
    try {
      const coldStats = this.deps.coldStats ?? safeColdStats;
      const sidecarStats = this.deps.sidecarStats ?? safeSidecarStats;
      const dep = this.deps.modelName;
      const fallback = typeof dep === 'function' ? dep() : dep;
      const rec: MetricsTraceInput = {
        traceId: ctx.traceId,
        ts: this.now(),
        sessionId: ctx.sessionId,
        model: ctx.modelAlias ?? fallback ?? 'unknown',
        durationMs: Math.max(0, this.now() - ctx.startMs),
        finishReason: final.finishReason,
        promptTokens: final.promptTokens,
        cachedTokens: final.cachedTokens ?? 0,
        outputTokens: final.numTokens,
        reasoningTokens: final.reasoningTokens,
      };
      const perf = final.performance;
      if (perf) {
        rec.ttftMs = perf.ttftMs;
        rec.prefillTps = perf.prefillTokensPerSecond;
        rec.decodeTps = perf.decodeTokensPerSecond;
        rec.mtpCycles = perf.mtpCycles;
        // mlx-vlm-comparable headline accept rate (committed tokens per cycle).
        rec.mtpMeanAccepted = perf.mtpMeanAcceptedTokensTotal;
      }
      // One delta per counter field: current minus the LAST record's mark
      // (beginTurn seeds an unmarked field from that turn's own baseline, so
      // the first record reports a real per-turn delta too), then advance
      // the mark. Marks keep overlapping turns from reporting the same
      // increment twice — deltas always chain, never nest — and a counter
      // that RESETS yields its new cumulative value, not a negative.
      const marks = this.deps.watermarks ?? SHARED_WATERMARKS;
      const delta = (mark: Record<string, number>, key: string, current: number | undefined): number | undefined => {
        if (current === undefined) return undefined;
        const since = mark[key];
        // current < mark means the counter reset: the whole new cumulative
        // value is the advance since the reset, not zero and not negative.
        const value = since === undefined || current < since ? current : current - since;
        mark[key] = current;
        return value;
      };
      const cold = coldStats();
      if (cold && ctx.cold) {
        rec.coldHits = delta(marks.cold, 'hits', cold.hits);
        rec.coldMisses = delta(marks.cold, 'misses', cold.misses);
        rec.coldBytesWritten = delta(marks.cold, 'bytesWritten', cold.bytesWritten);
        rec.coldBytesRestored = delta(marks.cold, 'bytesRestored', cold.bytesRestored);
        rec.coldEnqueued = delta(marks.cold, 'enqueued', cold.enqueued);
        rec.coldQueueDrops = delta(marks.cold, 'queueDrops', cold.queueDrops);
        rec.coldEvictions = delta(marks.cold, 'evictions', cold.evictions);
        rec.coldCorruptions = delta(marks.cold, 'corruptions', cold.corruptions);
        rec.coldWriteErrors = delta(marks.cold, 'writeErrors', cold.writeErrors);
        rec.coldRestoreDeclines = delta(marks.cold, 'restoreDeclines', cold.restoreDeclines);
        // Absolutes, not deltas: a dropped write inside an aborted turn (or
        // in the last turn before a crash) can land in a delta attributed to
        // the NEXT record, and one at process end lands in none at all. The
        // cumulative counter observed by each successful turn is what makes
        // "corruptions must be 0" checkable at all (`MAX(total) > 0` over any
        // window) — and "is my cache root broken?" is a question about ever,
        // not about this turn.
        rec.coldCorruptionsTotal = cold.corruptions;
        rec.coldQueueDropsTotal = cold.queueDrops;
        rec.coldWriteErrorsTotal = cold.writeErrors;
        // Cache IDENTITY comes from the END-of-turn snapshot, never the
        // baseline: the tier opens LAZILY on first use, so on a process's
        // first turn the baseline is the all-zero default with
        // `enabled: false` and an empty root. Canonicalized here — in the
        // writer — so the dashboard never has to match whatever spelling Rust
        // happened to construct.
        rec.coldEnabled = cold.enabled;
        // ONE emptiness test, applied to the CANONICAL value. Gating on the
        // raw native string used a DIFFERENT test from the one
        // `canonicalCacheRoot` applies (it trims), so a whitespace-only root
        // passed `length > 0` here and canonicalized to `''` — which
        // `MetricsTrace.record` then dropped, leaving a row that says the
        // tier was ON while carrying no root at all.
        if (cold.enabled) {
          const canonicalRoot = canonicalCacheRoot(cold.root);
          if (canonicalRoot.length > 0) rec.coldRoot = canonicalRoot;
        }
      }
      const sidecar = sidecarStats();
      if (sidecar && ctx.sidecar) {
        rec.coldSidecarCaptureReached = delta(marks.sidecar, 'captureReached', sidecar.captureReached);
        rec.coldSidecarChainEmpty = delta(marks.sidecar, 'chainEmpty', sidecar.chainEmpty);
        rec.coldSidecarBoundarySkips = delta(marks.sidecar, 'boundarySkips', sidecar.boundarySkips);
        rec.coldSidecarAlreadyPersisted = delta(marks.sidecar, 'alreadyPersisted', sidecar.alreadyPersisted);
        rec.coldSidecarEnqueued = delta(marks.sidecar, 'enqueued', sidecar.enqueued);
        rec.coldSidecarQueueDrops = delta(marks.sidecar, 'queueDrops', sidecar.queueDrops);
        rec.coldSidecarInstalled = delta(marks.sidecar, 'installed', sidecar.installed);
      }
      this.trace.record(rec);
    } catch {
      // Telemetry is best-effort; a broken recorder must never break inference.
    }
  }

  /**
   * Close a turn that will never record: thrown, aborted, errored, or ended
   * on a stream that yielded no successful terminal event. Its cold/sidecar
   * increments are real process advances — leaving the marks stale would
   * fold them into the NEXT recorded turn's delta, attributing a failed
   * turn's cache activity to whoever records next. Advancing the marks
   * without recording consumes the increment: uncommitted turns stay
   * unattributed on purpose.
   *
   * The advance is floored per field at the oldest OTHER live turn's begin
   * snapshot: increments that could still belong to a survivor's window are
   * left for it. A dead-owner rule keyed on the counter's current value alone
   * would eat a concurrent success's own increments and undercount.
   */
  private discardTurn(ctx: TurnContext): void {
    if (ctx.done) return;
    ctx.done = true;
    try {
      const marks = this.deps.watermarks ?? SHARED_WATERMARKS;
      marks.live?.delete(ctx);
      // Field-wise floor: the oldest surviving turn's begin baseline for the
      // SAME counter struct (cold and sidecar share field names like
      // `enqueued`), so that turn's whole window stays reportable.
      const advance = (
        mark: Record<string, number>,
        key: string,
        which: 'cold' | 'sidecar',
        current: number | undefined,
      ): void => {
        if (current === undefined) return;
        let floor = current;
        if (marks.live) {
          for (const live of marks.live) {
            const begin = (live[which] as Record<string, unknown> | undefined)?.[key];
            if (typeof begin === 'number' && begin < floor) floor = begin;
          }
        }
        mark[key] = Math.max(mark[key] ?? floor, floor);
      };
      const cold = (this.deps.coldStats ?? safeColdStats)();
      if (cold) {
        for (const key of COLD_DELTA_KEYS) advance(marks.cold, key, 'cold', cold[key]);
      }
      const sidecar = (this.deps.sidecarStats ?? safeSidecarStats)();
      if (sidecar) {
        for (const key of SIDECAR_DELTA_KEYS) advance(marks.sidecar, key, 'sidecar', sidecar[key]);
      }
    } catch {
      // Best-effort; a broken stat reader must never break inference.
    }
  }

  /** The shared body of the three non-streaming dispatch entry points. */
  private async dispatch(
    kind: 'chatSessionStart' | 'chatSessionContinue' | 'chatSessionContinueTool',
    messages: ChatMessage[],
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): Promise<ChatResult> {
    const ctx = this.beginTurn(config);
    try {
      const result = await this.inner[kind](messages, config, signal);
      // Record only committed turns: `finishReason: 'error'` results are
      // returned, not thrown, and mirror the stream-side non-record rule.
      if (ctx !== null && result.finishReason !== 'error') {
        this.finishTurn(ctx, result);
      }
      return result;
    } finally {
      // A turn that threw or returned 'error' never reached finishTurn —
      // discard consumes its marks so its increments leak nowhere.
      if (ctx !== null) this.discardTurn(ctx);
    }
  }

  /** The shared body of the three streaming dispatch entry points. */
  private async *streamDispatch(
    kind: 'chatStreamSessionStart' | 'chatStreamSessionContinue' | 'chatStreamSessionContinueTool',
    messages: ChatMessage[],
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    const ctx = this.beginTurn(config);
    try {
      for await (const event of this.inner[kind](messages, config, signal)) {
        if (ctx !== null && event.done && event.finishReason !== 'error') {
          this.finishTurn(ctx, event);
        }
        yield event;
      }
    } finally {
      // Mid-stream throws, consumer abandonment, and error-terminal events
      // all exit through here — the mark advance keeps the sums honest.
      if (ctx !== null) this.discardTurn(ctx);
    }
  }

  chatSessionStart(messages: ChatMessage[], config?: ChatConfig | null, signal?: AbortSignal): Promise<ChatResult> {
    return this.dispatch('chatSessionStart', messages, config, signal);
  }

  chatSessionContinue(messages: ChatMessage[], config?: ChatConfig | null, signal?: AbortSignal): Promise<ChatResult> {
    return this.dispatch('chatSessionContinue', messages, config, signal);
  }

  chatSessionContinueTool(
    messages: ChatMessage[],
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): Promise<ChatResult> {
    return this.dispatch('chatSessionContinueTool', messages, config, signal);
  }

  chatStreamSessionStart(
    messages: ChatMessage[],
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    return this.streamDispatch('chatStreamSessionStart', messages, config, signal);
  }

  chatStreamSessionContinue(
    messages: ChatMessage[],
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    return this.streamDispatch('chatStreamSessionContinue', messages, config, signal);
  }

  chatStreamSessionContinueTool(
    messages: ChatMessage[],
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    return this.streamDispatch('chatStreamSessionContinueTool', messages, config, signal);
  }

  resetCaches(): void | Promise<void> {
    return this.inner.resetCaches();
  }
}

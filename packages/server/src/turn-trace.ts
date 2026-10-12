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
 * moment) and diffed when the turn completes successfully. The per-turn
 * context — not shared state — holds the baseline, so a turn that throws,
 * aborts, or ends on `finishReason: 'error'` simply never diffs and cannot
 * leak its restores into the next successful turn's delta. On continuously
 * batched models (`maxConcurrentSequences() > 1`) two turns can overlap in
 * native flight, which makes every synchronous-counter delta a best-effort
 * attribution — the same caveat the `*_Total` absolute columns exist to
 * cover. `durationMs` therefore measures the wrapped dispatch call itself;
 * admission-lane queue waits happen upstream and are not included.
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
import type { ChatStreamEvent, SessionCapableModel } from '@mlx-node/lm';
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

/** Injectable seams for {@link TurnTraceModel} (unit tests). */
export interface TurnTraceDeps {
  /** Model name recorded on every trace line. Default: `'unknown'`. */
  modelName?: string;
  /** Durable telemetry sink; defaults to a fresh {@link MetricsTrace}. */
  trace?: Pick<MetricsTrace, 'record'>;
  /** Process-wide cold-tier reader; defaults to the native addon. */
  coldStats?: () => ColdCacheStats | undefined;
  /** Process-wide sidecar-counter reader; defaults to the native addon. */
  sidecarStats?: () => ColdSidecarStats | undefined;
  /** Clock for `ts`, `durationMs`, and the trace filename. Default: `Date.now`. */
  now?: () => number;
}

/**
 * Per-turn recording context. `null` when the trace sink is disabled — the
 * wrapper then costs one check per call, no snapshots at all.
 */
interface TurnContext {
  traceId: string;
  sessionId: string | undefined;
  startMs: number;
  cold: ColdCacheStats | undefined;
  sidecar: ColdSidecarStats | undefined;
  done: boolean;
}

/**
 * Members of {@link SessionCapableModel} whose presence is optional and
 * whose calls are pure queries — forwarded verbatim so the wrapper is
 * invisible to `ChatSession` feature detection (`hasBlockPagedCache`,
 * `mtpAutoEnabled`, `applyChatTemplate`, …). The wrapper MUST NOT define
 * one the inner model lacks: `ChatSession.mtpAutoDefaultAllowed` treats a
 * missing `mtpAutoEnabled` as a proto-walk fallback, and `hasBlockPagedCache`
 * absent means "not paged". Mirroring presence exactly preserves both.
 */
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
  }

  private get now(): () => number {
    return this.deps.now ?? Date.now;
  }

  /**
   * Open a turn context: mint the trace id, stamp the start clock, and
   * snapshot the process-wide cold counters BEFORE native work begins so the
   * record below can emit this turn's delta rather than the process total.
   */
  private beginTurn(config: ChatConfig | null | undefined): TurnContext | null {
    const trace = this.trace;
    if (trace instanceof MetricsTrace && !trace.enabled) return null;
    const cold = (this.deps.coldStats ?? safeColdStats)();
    const sidecar = (this.deps.sidecarStats ?? safeSidecarStats)();
    return {
      traceId: randomUUID(),
      sessionId: typeof config?.cacheOwnerId === 'string' ? config.cacheOwnerId : undefined,
      startMs: this.now(),
      cold,
      sidecar,
      done: false,
    };
  }

  /**
   * Diff the end-of-turn counters against the turn's own baseline and append
   * the record. Rebuilt field by field from the terminal result — never the
   * free-text members — then handed to the allowlisted sink.
   *
   * The SIDECAR diff is its own guard, deliberately not folded into the cold
   * block: the sidecar reader never consults the tier, so it keeps reporting
   * when the tier failed to open — a run with `coldHits: 0` and no tier is
   * exactly when "did the capture even run?" is the question.
   */
  private finishTurn(ctx: TurnContext, final: TurnFinal): void {
    if (ctx.done) return;
    ctx.done = true;
    try {
      const coldStats = this.deps.coldStats ?? safeColdStats;
      const sidecarStats = this.deps.sidecarStats ?? safeSidecarStats;
      const rec: MetricsTraceInput = {
        traceId: ctx.traceId,
        ts: this.now(),
        sessionId: ctx.sessionId,
        model: this.deps.modelName ?? 'unknown',
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
      const cold = coldStats();
      if (cold && ctx.cold) {
        rec.coldHits = cold.hits - ctx.cold.hits;
        rec.coldMisses = cold.misses - ctx.cold.misses;
        rec.coldBytesWritten = cold.bytesWritten - ctx.cold.bytesWritten;
        rec.coldBytesRestored = cold.bytesRestored - ctx.cold.bytesRestored;
        rec.coldEnqueued = cold.enqueued - ctx.cold.enqueued;
        rec.coldQueueDrops = cold.queueDrops - ctx.cold.queueDrops;
        rec.coldEvictions = cold.evictions - ctx.cold.evictions;
        rec.coldCorruptions = cold.corruptions - ctx.cold.corruptions;
        rec.coldWriteErrors = cold.writeErrors - ctx.cold.writeErrors;
        rec.coldRestoreDeclines = cold.restoreDeclines - ctx.cold.restoreDeclines;
        // Absolutes, not deltas: an aborted/errored turn never reaches this
        // recorder, so a corruption or a dropped write during one lands in NO
        // delta. The cumulative counter observed by the next successful turn
        // still carries it, which is what makes "corruptions must be 0"
        // checkable at all (`MAX(total) > 0` over any window).
        rec.coldCorruptionsTotal = cold.corruptions;
        rec.coldQueueDropsTotal = cold.queueDrops;
        // Same latch, and the counter that needs it most: a write error is
        // raised on the background writer, so the one covering the LAST turn
        // before a crash lands in no delta at all — and "is my cache root
        // broken?" is a question about ever, not about this turn.
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
        rec.coldSidecarCaptureReached = sidecar.captureReached - ctx.sidecar.captureReached;
        rec.coldSidecarChainEmpty = sidecar.chainEmpty - ctx.sidecar.chainEmpty;
        rec.coldSidecarBoundarySkips = sidecar.boundarySkips - ctx.sidecar.boundarySkips;
        rec.coldSidecarAlreadyPersisted = sidecar.alreadyPersisted - ctx.sidecar.alreadyPersisted;
        rec.coldSidecarEnqueued = sidecar.enqueued - ctx.sidecar.enqueued;
        rec.coldSidecarQueueDrops = sidecar.queueDrops - ctx.sidecar.queueDrops;
        rec.coldSidecarInstalled = sidecar.installed - ctx.sidecar.installed;
        rec.coldSidecarRestoreSuppressed = sidecar.restoreSuppressed - ctx.sidecar.restoreSuppressed;
      }
      this.trace.record(rec);
    } catch {
      // Telemetry is best-effort; a broken recorder must never break inference.
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
    const result = await this.inner[kind](messages, config, signal);
    // Record only committed turns: `finishReason: 'error'` results are
    // returned, not thrown, and mirror the stream-side non-record rule.
    if (ctx !== null && result.finishReason !== 'error') {
      this.finishTurn(ctx, result);
    }
    return result;
  }

  /** The shared body of the three streaming dispatch entry points. */
  private async *streamDispatch(
    kind: 'chatStreamSessionStart' | 'chatStreamSessionContinue' | 'chatStreamSessionContinueTool',
    messages: ChatMessage[],
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): AsyncGenerator<ChatStreamEvent> {
    const ctx = this.beginTurn(config);
    for await (const event of this.inner[kind](messages, config, signal)) {
      if (ctx !== null && event.done && event.finishReason !== 'error') {
        this.finishTurn(ctx, event);
      }
      yield event;
    }
  }

  chatSessionStart(messages: ChatMessage[], config?: ChatConfig | null, signal?: AbortSignal): Promise<ChatResult> {
    return this.dispatch('chatSessionStart', messages, config, signal);
  }

  chatSessionContinue(
    messages: ChatMessage[],
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): Promise<ChatResult> {
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

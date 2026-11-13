/**
 * `TurnTraceModel` — the `SessionCapableModel` wrapper that produces one
 * {@link MetricsTraceRecord} per dispatched inference turn: `ok`, `error` or
 * `cancelled`.
 *
 * Installed once per `SessionRegistry` — the single allocation point of every
 * `ChatSession` the serve process hands out — so it sits on ALL of the
 * registry's native dispatch surface at once: `chatSessionStart` /
 * `chatSessionContinue` / `chatSessionContinueTool` and the three streaming
 * variants, for both `/v1/responses` and `/v1/messages`, streaming or not,
 * warm hit or cold replay. There is deliberately no endpoint-level hook:
 * the endpoints hand work to the session and only learn the wire shape
 * afterwards; the model call is the one place the final `ChatResult` /
 * terminal `ChatStreamEvent` exists before the turn is committed. The
 * endpoints stamp request timing on the config (`turnMeta`).
 *
 * Turn boundaries: the cold-tier counters are process-wide and cumulative, so
 * the baseline is snapshotted INSIDE the wrapped call (the actual dispatch
 * moment) and diffed when the turn completes successfully. A turn that
 * throws, aborts, or ends on `finishReason: 'error'` records no cold fields —
 * but it still ADVANCES the shared watermarks ({@link TurnTraceModel.discardTurn}),
 * so its counter increments are consumed rather than inherited by the next
 * successful turn's delta. That advance is floored at the oldest LIVE turn's
 * baseline: a failure finishing mid-overlap consumes only up to what a
 * survivor will still see, never the survivor's own increments. On
 * continuously batched models
 * (`maxConcurrentSequences() > 1`) two turns can overlap in native flight:
 * each turn then reports the counter advance since the PREVIOUS record's
 * high-water mark ({@link TurnWatermarks}), so attribution between
 * overlapping turns is approximate while the deltas still sum to the
 * counter's real advance — the dashboard SUMs these columns.
 */

import { randomUUID } from 'node:crypto';

import {
  coldCacheStats,
  coldSidecarStats,
  getMemorySnapshot,
  type ChatConfig,
  type ChatMessage,
  type ChatResult,
  type ChatStreamProgress,
  type ColdCacheStats,
  type ColdSidecarStats,
  type PerformanceMetrics,
} from '@mlx-node/core';
import {
  isMediaHeldRestartError,
  mtpAutoDefaultAllowed,
  type ChatStreamEvent,
  type ChatStreamControl,
  type SessionCapableModel,
} from '@mlx-node/lm';
import { canonicalCacheRoot } from '@mlx-node/lm/cold-tier';

import {
  MetricsTrace,
  type MetricsTraceInput,
  type TurnBudgetFrom,
  type TurnDispatch,
  type TurnEndpoint,
  type TurnErrorKind,
  type TurnFailPhase,
} from './metrics-trace.js';
import { readPipelineStats, type PipelineStats, type TurnProgressSnapshot } from './native-progress.js';
import { reasoningOutcomeOf, type ThinkingChatConfig } from './native-thinking.js';
import type { MeasuredPrefill } from './timing.js';

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

/** Process-wide memory reader; the native addon may be absent (unit tests). */
function safeMemory(): { activeBytes: number; peakBytes: number } | undefined {
  try {
    const snapshot = getMemorySnapshot();
    return { activeBytes: snapshot.activeBytes, peakBytes: snapshot.peakBytes };
  } catch {
    return undefined;
  }
}

/** Request timing the serve endpoints stamp on the dispatched config. */
export interface TurnRequestMeta {
  endpoint: TurnEndpoint;
  queueMs?: number;
  loadWaitMs?: number;
  resolveMs?: number;
  preDispatchMs?: number;
  /** Set when the turn has a thinking budget: the request's own, or the server's option. */
  thinkingBudgetFrom?: TurnBudgetFrom;
}

/** Identity of the model a binding was created for. */
export interface ModelIdentity {
  modelId: string;
  modelType: string;
  contextWindow: number;
}

/** Process-wide, monotonic. Read by the runtime sampler. */
export interface TurnCounters {
  ok: number;
  error: number;
  cancelled: number;
  outputTokens: number;
  /**
   * Tokens of streaming turns only. A non-stream turn credits all its tokens at
   * settle, so a live rate must read this counter, never `outputTokens`.
   */
  streamedTokens: number;
}

const COUNTERS: TurnCounters = { ok: 0, error: 0, cancelled: 0, outputTokens: 0, streamedTokens: 0 };

export function turnCounters(): Readonly<TurnCounters> {
  return { ...COUNTERS };
}

let edgeListener: (() => void) | null = null;

/** One listener (the sampler). Called on every turn begin and settle. */
export function setTurnEdgeListener(listener: (() => void) | null): void {
  edgeListener = listener;
}

function notifyEdge(): void {
  const listener = edgeListener;
  if (listener === null) return;
  try {
    listener();
  } catch {
    // A broken listener must never break inference.
  }
}

/** One turn of this model in native flight. */
export interface LiveTurn {
  traceId: string;
  /** Alias the client asked for. */
  model: string;
  /** Checkpoint name from the binding's identity; null without one. */
  modelId: string | null;
  /** The wrapper's resolved window; null when unknown. */
  ctxWindow: number | null;
  startedAt: number;
  stream: boolean;
  dispatch: TurnDispatch;
  progress: TurnProgressSnapshot | null;
}

/**
 * Where a failed stream stopped. Native sets the phase to `done` BEFORE it
 * sends an error or the terminal chunk, so on `done` the surviving timers say
 * how far the turn got. Before a prefill, time in `slot` wins over time in
 * `prepare`: only a lane or memory wait puts a turn in `slot`, and the
 * scheduler's own refusal comes from there, while every scheduled turn spends
 * some time in `prepare`.
 */
export function failPhaseOf(snap: TurnProgressSnapshot | null): TurnFailPhase | undefined {
  if (snap === null) return undefined;
  if (snap.phase !== 'done') return snap.phase;
  if (snap.decodeElapsedMs !== undefined) return 'decode';
  if (snap.prefillElapsedMs !== undefined) return 'prefill';
  if (snap.slotMs !== undefined && snap.slotMs > 0) return 'slot';
  if (snap.prepareMs !== undefined && snap.prepareMs > 0) return 'prepare';
  return 'queued';
}

/** The frozen prefill window of a finished stream, or undefined when native did not measure one. */
export function measuredPrefillOf(snap: TurnProgressSnapshot | null | undefined): MeasuredPrefill | undefined {
  if (snap === null || snap === undefined) return undefined;
  const tokens = snap.prefillTotal;
  const ms = snap.prefillElapsedMs;
  // Without `prefillTotal` the clock counts from `begin_turn`, and without
  // `decodeElapsedMs` the window is still running: neither is a prefill window.
  if (tokens === undefined || ms === undefined || snap.decodeElapsedMs === undefined) return undefined;
  if (!Number.isInteger(tokens) || tokens <= 0 || !Number.isFinite(ms) || ms <= 0) return undefined;
  return { prefillMs: ms, prefillTokens: tokens };
}

/**
 * Keeps the last native progress snapshot readable after the stream bridge
 * clears its reader (`packages/lm/src/stream.ts` sets `read = null` in its
 * `finally`, while the native handle is still alive), and forwards every
 * reader and stop to the caller's control: `ChatSession.stop` calls the
 * caller's `stop`, so a stop that is not forwarded never reaches native.
 * `stopped` is set when native accepts one: `ChatSession` then reports the
 * turn with no tool calls, whatever native decoded.
 */
class TraceControl implements ChatStreamControl {
  private reader: (() => ChatStreamProgress | null) | null = null;
  private stopper: (() => boolean) | null = null;
  last: ChatStreamProgress | null = null;
  stopped = false;
  constructor(private readonly outer: ChatStreamControl | undefined) {}
  get read(): (() => ChatStreamProgress | null) | null {
    return this.reader;
  }
  set read(next: (() => ChatStreamProgress | null) | null) {
    if (next === null && this.reader !== null) {
      try {
        this.last = this.reader();
      } catch {
        // The handle may already be gone; keep the previous capture.
      }
    }
    this.reader = next;
    if (this.outer !== undefined) this.outer.read = next;
  }
  get stop(): (() => boolean) | null {
    return this.stopper;
  }
  set stop(next: (() => boolean) | null) {
    this.stopper = next;
    if (this.outer === undefined) return;
    this.outer.stop =
      next === null
        ? null
        : (): boolean => {
            const sent = next();
            if (sent) this.stopped = true;
            return sent;
          };
  }
}

/**
 * `ChatConfig` plus the trace-only fields the serve endpoints stamp per
 * request: the registry name the request resolved through, and the request
 * timing (`turnMeta`). The napi object
 * boundary drops unknown props, so carrying it on the config costs nothing
 * — and it is the only per-request handle a shared model wrapper has:
 * aliased registrations reuse one `SessionRegistry`, so the registration
 * name alone cannot attribute a turn to the alias the client asked for.
 * Never written by model code; read only here.
 */
export interface TracedChatConfig extends ChatConfig {
  modelAlias?: string;
  turnMeta?: TurnRequestMeta;
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
  /** Identity of the model this binding was created for. */
  identity?: ModelIdentity;
  /** The served window, read once by the model registry; the identity's trained window when absent. */
  ctxWindow?: number;
  /** Memory reader for the turn-end snapshot; defaults to the native addon. */
  memory?: () => { activeBytes: number; peakBytes: number } | undefined;
  /** Process-wide pipeline counter; defaults to the native addon. Null while the counter is off. */
  pipelineStats?: () => PipelineStats | null;
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
  stream: boolean;
  dispatch: TurnDispatch;
  meta: TurnRequestMeta | undefined;
  toolCount: number;
  imageCount: number;
  concurrency: number;
  concurrencyPeak: number;
  signal: AbortSignal | undefined;
  progress: TraceControl | null;
  /** Generated tokens already added to the process counter. */
  creditedTokens: number;
  /** The pipeline counter at dispatch; null while it is off. */
  pipelines: PipelineStats | null;
  /** The thinking budget, the guard and `maxNewTokens` of the config sent to native. */
  thinkingBudget: number | undefined;
  thinkingGuard: boolean;
  outputCap: number | undefined;
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
  /** Only the length is read. */
  toolCalls?: readonly unknown[];
}

type DispatchKind = 'chatSessionStart' | 'chatSessionContinue' | 'chatSessionContinueTool';
type StreamDispatchKind = 'chatStreamSessionStart' | 'chatStreamSessionContinue' | 'chatStreamSessionContinueTool';

function dispatchOf(kind: DispatchKind | StreamDispatchKind): TurnDispatch {
  switch (kind) {
    case 'chatSessionStart':
    case 'chatStreamSessionStart':
      return 'start';
    case 'chatSessionContinue':
    case 'chatStreamSessionContinue':
      return 'continue';
    default:
      return 'continue_tool';
  }
}

function imageCountOf(messages: readonly ChatMessage[] | null | undefined): number {
  if (!Array.isArray(messages)) return 0;
  let count = 0;
  for (const message of messages) count += message?.images?.length ?? 0;
  return count;
}

export class TurnTraceModel implements SessionCapableModel {
  private readonly inner: SessionCapableModel;
  private readonly deps: TurnTraceDeps;
  private readonly trace: Pick<MetricsTrace, 'record'>;
  private readonly liveCtx = new Set<TurnContext>();
  private begun = 0;
  private lastSettleAt = 0;
  private readonly ctxWindow: number | undefined;

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
    const served = deps.ctxWindow;
    const identityWindow = deps.identity?.contextWindow;
    this.ctxWindow =
      typeof served === 'number' && served > 0
        ? served
        : typeof identityWindow === 'number' && identityWindow > 0
          ? identityWindow
          : undefined;
  }

  /** Live turns of this model. Also credits streamed tokens to the process counter. */
  liveTurns(): LiveTurn[] {
    const out: LiveTurn[] = [];
    for (const ctx of this.liveCtx) {
      let progress: TurnProgressSnapshot | null = null;
      try {
        progress = ctx.progress?.read?.() ?? null;
      } catch {
        progress = null;
      }
      const generated = progress?.generatedTokens;
      if (typeof generated === 'number' && generated > ctx.creditedTokens) {
        COUNTERS.outputTokens += generated - ctx.creditedTokens;
        COUNTERS.streamedTokens += generated - ctx.creditedTokens;
        ctx.creditedTokens = generated;
      }
      out.push({
        traceId: ctx.traceId,
        model: this.modelNameOf(ctx),
        modelId: this.deps.identity?.modelId ?? null,
        ctxWindow: this.ctxWindow ?? null,
        startedAt: ctx.startMs,
        stream: ctx.stream,
        dispatch: ctx.dispatch,
        progress,
      });
    }
    return out;
  }

  /** `begun` rises by one per dispatch; the sampler uses it to detect a burst between ticks. */
  turnStats(): { live: number; begun: number; lastSettleAt: number } {
    return { live: this.liveCtx.size, begun: this.begun, lastSettleAt: this.lastSettleAt };
  }

  private modelNameOf(ctx: TurnContext): string {
    const dep = this.deps.modelName;
    const fallback = typeof dep === 'function' ? dep() : dep;
    return ctx.modelAlias ?? fallback ?? 'unknown';
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
  private beginTurn(
    config: ChatConfig | null | undefined,
    messages: readonly ChatMessage[] | null | undefined,
    kind: DispatchKind | StreamDispatchKind,
    stream: boolean,
    signal: AbortSignal | undefined,
    outerControl: ChatStreamControl | undefined,
  ): TurnContext | null {
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
    const traced = config as TracedChatConfig | null | undefined;
    const ctx: TurnContext = {
      traceId: randomUUID(),
      sessionId: typeof config?.cacheOwnerId === 'string' ? config.cacheOwnerId : undefined,
      // The requesting alias outranks the binding name: one SessionRegistry
      // (and one wrapper) serves every alias of a model.
      modelAlias: traced?.modelAlias,
      startMs: this.now(),
      cold,
      sidecar,
      done: false,
      stream,
      dispatch: dispatchOf(kind),
      meta: traced?.turnMeta,
      toolCount: config?.tools?.length ?? 0,
      imageCount: imageCountOf(messages),
      concurrency: 0,
      concurrencyPeak: 0,
      signal,
      progress: stream ? new TraceControl(outerControl) : null,
      creditedTokens: 0,
      pipelines: this.pipelineStats(),
      thinkingBudget: config?.thinkingTokenBudget ?? undefined,
      thinkingGuard: (config as ThinkingChatConfig | null | undefined)?.thinkingCycleGuard === true,
      outputCap: config?.maxNewTokens ?? undefined,
    };
    (marks.live ??= new Set()).add(ctx);
    this.liveCtx.add(ctx);
    const live = this.liveCtx.size;
    ctx.concurrency = live;
    ctx.concurrencyPeak = live;
    for (const other of this.liveCtx) {
      if (other.concurrencyPeak < live) other.concurrencyPeak = live;
    }
    this.begun += 1;
    notifyEdge();
    return ctx;
  }

  private pipelineStats(): PipelineStats | null {
    try {
      return (this.deps.pipelineStats ?? readPipelineStats)();
    } catch {
      return null;
    }
  }

  /** The fields every outcome shares: identity, timing, counts, overlap, memory, phase timers, pipelines. */
  private baseRecord(ctx: TurnContext, snap: TurnProgressSnapshot | null): MetricsTraceInput {
    const now = this.now();
    const identity = this.deps.identity;
    const meta = ctx.meta;
    const rec: MetricsTraceInput = {
      traceId: ctx.traceId,
      ts: now,
      sessionId: ctx.sessionId,
      model: this.modelNameOf(ctx),
      modelId: identity?.modelId,
      modelType: identity?.modelType,
      outcome: 'ok',
      endpoint: meta?.endpoint ?? 'other',
      stream: ctx.stream,
      dispatch: ctx.dispatch,
      finishReason: 'other',
      durationMs: Math.max(0, now - ctx.startMs),
      queueMs: meta?.queueMs,
      loadWaitMs: meta?.loadWaitMs,
      resolveMs: meta?.resolveMs,
      preDispatchMs: meta?.preDispatchMs,
      thinkingBudget: ctx.thinkingBudget,
      thinkingBudgetFrom: meta?.thinkingBudgetFrom,
      ...(ctx.thinkingGuard ? { thinkingGuard: true as const } : {}),
      outputCap: ctx.outputCap,
      ctxWindow: this.ctxWindow,
      toolCount: ctx.toolCount,
      imageCount: ctx.imageCount,
      concurrency: ctx.concurrency,
      concurrencyPeak: ctx.concurrencyPeak,
    };
    let memory: { activeBytes: number; peakBytes: number } | undefined;
    try {
      memory = (this.deps.memory ?? safeMemory)();
    } catch {
      memory = undefined;
    }
    if (memory) {
      rec.memActiveBytes = memory.activeBytes;
      rec.memPeakBytes = memory.peakBytes;
    }
    if (snap !== null) {
      rec.waitMs = snap.waitMs;
      rec.slotMs = snap.slotMs;
      rec.prepareMs = snap.prepareMs;
    }
    const begin = ctx.pipelines;
    const end = begin === null ? null : this.pipelineStats();
    if (begin !== null && end !== null && end.createdServing >= begin.createdServing) {
      rec.pipelinesCreated = end.createdServing - begin.createdServing;
      rec.pipelineCreateMs = Math.max(0, end.servingMs - begin.servingMs);
    }
    return rec;
  }

  /** Remove a turn from the live set and credit its last tokens. */
  private settle(ctx: TurnContext, totalTokens: number | undefined): void {
    this.liveCtx.delete(ctx);
    if (typeof totalTokens === 'number' && totalTokens > ctx.creditedTokens) {
      COUNTERS.outputTokens += totalTokens - ctx.creditedTokens;
      if (ctx.stream) COUNTERS.streamedTokens += totalTokens - ctx.creditedTokens;
      ctx.creditedTokens = totalTokens;
    }
    this.lastSettleAt = this.now();
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
    this.settle(ctx, final.numTokens);
    COUNTERS.ok += 1;
    try {
      const coldStats = this.deps.coldStats ?? safeColdStats;
      const sidecarStats = this.deps.sidecarStats ?? safeSidecarStats;
      const snap = this.readProgress(ctx);
      const rec = this.baseRecord(ctx, snap);
      rec.outcome = 'ok';
      rec.finishReason = final.finishReason;
      rec.promptTokens = final.promptTokens;
      rec.cachedTokens = final.cachedTokens ?? 0;
      rec.outputTokens = final.numTokens;
      rec.reasoningTokens = final.reasoningTokens;
      Object.assign(rec, reasoningOutcomeOf(final));
      if (ctx.progress?.stopped === true) rec.toolCallCount = 0;
      else if (Array.isArray(final.toolCalls)) rec.toolCallCount = final.toolCalls.length;
      const perf = final.performance;
      if (perf) {
        rec.ttftMs = perf.ttftMs;
        rec.decodeTps = perf.decodeTokensPerSecond;
        rec.mtpCycles = perf.mtpCycles;
        // mlx-vlm-comparable headline accept rate (committed tokens per cycle).
        rec.mtpMeanAccepted = perf.mtpMeanAcceptedTokensTotal;
        rec.mtpMeanDrafts = perf.mtpMeanAcceptedTokens;
        rec.mtpMeanDepth = perf.mtpMeanDepth;
        rec.mtpAcceptByPos = perf.mtpAcceptanceByPosition;
      }
      const measured = measuredPrefillOf(snap);
      if (measured !== undefined) {
        rec.prefillMs = measured.prefillMs;
        rec.prefillTokens = measured.prefillTokens;
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
    notifyEdge();
  }

  /**
   * Record an `error` or `cancelled` turn. Marks advance exactly as
   * {@link discardTurn} does; the record carries no cold fields.
   *
   * The live progress read comes first: a `finish_error` turn is recorded
   * inside the stream loop, before the bridge clears the reader, so the
   * captured snapshot is still null there.
   */
  private recordFailure(
    ctx: TurnContext,
    outcome: 'error' | 'cancelled',
    errorKind?: TurnErrorKind,
    final?: TurnFinal,
  ): void {
    if (ctx.done) return;
    const snap = this.readProgress(ctx);
    this.discardTurn(ctx);
    const total = final !== undefined ? final.numTokens : snap?.generatedTokens;
    this.settle(ctx, total);
    COUNTERS[outcome] += 1;
    try {
      const rec = this.baseRecord(ctx, snap);
      rec.outcome = outcome;
      rec.finishReason = outcome === 'cancelled' ? 'cancelled' : 'error';
      if (outcome === 'error') rec.errorKind = errorKind;
      rec.failPhase = failPhaseOf(snap);
      if (final !== undefined) {
        rec.promptTokens = final.promptTokens;
        rec.cachedTokens = final.cachedTokens;
        rec.outputTokens = final.numTokens;
        rec.reasoningTokens = final.reasoningTokens;
        Object.assign(rec, reasoningOutcomeOf(final));
        if (Array.isArray(final.toolCalls)) rec.toolCallCount = final.toolCalls.length;
      } else if (snap !== null) {
        rec.outputTokens = snap.generatedTokens;
        rec.cachedTokens = snap.cachedTokens;
        if (snap.cachedTokens !== undefined && snap.prefillTotal !== undefined) {
          rec.promptTokens = snap.cachedTokens + snap.prefillTotal;
        }
      }
      if (snap !== null) rec.prefillDoneTokens = snap.prefillDone;
      this.trace.record(rec);
    } catch {
      // Telemetry is best-effort; a broken recorder must never break inference.
    }
    notifyEdge();
  }

  private readProgress(ctx: TurnContext): TurnProgressSnapshot | null {
    try {
      return ctx.progress?.read?.() ?? ctx.progress?.last ?? null;
    } catch {
      return ctx.progress?.last ?? null;
    }
  }

  /** Close a turn that is retried elsewhere and must not be recorded. */
  private rerouteTurn(ctx: TurnContext): void {
    if (ctx.done) return;
    this.discardTurn(ctx);
    this.settle(ctx, undefined);
    notifyEdge();
  }

  /**
   * Close the cold accounting of a turn that carries no cold fields: thrown,
   * aborted, errored, rerouted, or ended on a stream that yielded no
   * successful terminal event. Its cold/sidecar
   * increments are real process advances — leaving the marks stale would
   * fold them into the NEXT recorded turn's delta, attributing a failed
   * turn's cache activity to whoever records next. Advancing the marks
   * without attributing consumes the increment: uncommitted turns stay
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
    kind: DispatchKind,
    messages: ChatMessage[],
    config?: ChatConfig | null,
    signal?: AbortSignal,
  ): Promise<ChatResult> {
    const ctx = this.beginTurn(config, messages, kind, false, signal, undefined);
    try {
      const result = await this.inner[kind](messages, config, signal);
      if (ctx !== null) {
        if (result.finishReason !== 'error') this.finishTurn(ctx, result);
        else this.recordFailure(ctx, 'error', 'finish_error', result);
      }
      return result;
    } catch (err) {
      if (ctx !== null) this.settleThrown(ctx, err, signal);
      throw err;
    }
  }

  /**
   * A media-held restart is retried by `ChatSession` through
   * `chatSessionStart`; recording it would invent a failure.
   */
  private settleThrown(ctx: TurnContext, err: unknown, signal: AbortSignal | undefined): void {
    if (isMediaHeldRestartError(err)) this.rerouteTurn(ctx);
    else if (signal?.aborted === true) this.recordFailure(ctx, 'cancelled');
    else this.recordFailure(ctx, 'error', 'exception');
  }

  /** The shared body of the three streaming dispatch entry points. */
  private async *streamDispatch(
    kind: StreamDispatchKind,
    messages: ChatMessage[],
    config?: ChatConfig | null,
    signal?: AbortSignal,
    control?: ChatStreamControl,
  ): AsyncGenerator<ChatStreamEvent> {
    const ctx = this.beginTurn(config, messages, kind, true, signal, control);
    let thrown: { err: unknown } | null = null;
    try {
      for await (const event of this.inner[kind](messages, config, signal, ctx?.progress ?? control)) {
        if (ctx !== null && event.done) {
          if (event.finishReason !== 'error') this.finishTurn(ctx, event);
          else this.recordFailure(ctx, 'error', 'finish_error', event);
        }
        yield event;
      }
    } catch (err) {
      thrown = { err };
      throw err;
    } finally {
      if (ctx !== null && !ctx.done) {
        // A stream that ends with no terminal event and no throw was
        // abandoned by its consumer or aborted.
        if (thrown !== null) this.settleThrown(ctx, thrown.err, signal);
        else this.recordFailure(ctx, 'cancelled');
      }
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
    control?: ChatStreamControl,
  ): AsyncGenerator<ChatStreamEvent> {
    return this.streamDispatch('chatStreamSessionStart', messages, config, signal, control);
  }

  chatStreamSessionContinue(
    messages: ChatMessage[],
    config?: ChatConfig | null,
    signal?: AbortSignal,
    control?: ChatStreamControl,
  ): AsyncGenerator<ChatStreamEvent> {
    return this.streamDispatch('chatStreamSessionContinue', messages, config, signal, control);
  }

  chatStreamSessionContinueTool(
    messages: ChatMessage[],
    config?: ChatConfig | null,
    signal?: AbortSignal,
    control?: ChatStreamControl,
  ): AsyncGenerator<ChatStreamEvent> {
    return this.streamDispatch('chatStreamSessionContinueTool', messages, config, signal, control);
  }

  resetCaches(): void | Promise<void> {
    return this.inner.resetCaches();
  }
}

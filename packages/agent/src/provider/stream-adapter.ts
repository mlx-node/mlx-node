/**
 * `makeMlxStreamSimple` — the provider bridge's pi `streamSimple` seam.
 *
 * Every pi LLM call becomes one warm replay against the host's resident
 * `ChatSession` (spike-proven pattern):
 *
 *   resetPreservingNativeCacheForWarmReuse(session)  // JS-state-only wipe
 *   session.primeHistory(contextToChatMessages(ctx)) // pi's full history
 *   session.startFromHistoryStream(config, signal)   // cold replay, warm KV
 *
 * The whole per-call body — resident selection INCLUDED — runs inside one
 * `MlxModelHost.runWithResident` closure, so concurrent pi calls (and
 * model swaps) execute strictly sequentially and the session can never be
 * swapped out mid-turn. Do not split this into `ensureResident` + a
 * separate serialization step; that pattern has a stale-resident race.
 *
 * Contract (absolute): the returned function NEVER throws and its stream
 * always terminates — with exactly ONE terminal event. Enforced in layers:
 *
 *   - A turn-wide `terminated` flag: every ending routes through
 *     `terminalize` (or the native-final branch), so the first terminal
 *     wins and all later work — including a resident closure that was
 *     queued behind stalled inference/loading when the abort landed —
 *     observes the flag and skips ALL session work.
 *   - Abort coverage has no queued gap: an already-aborted signal
 *     terminates before the host is even engaged, and an abort listener
 *     spans the whole queued/running window so the stream terminates
 *     promptly even when `runWithResident` never yields.
 *   - Failures become stream events via `TurnEmitter` (`onError` /
 *     `onAborted`). Hostile error values are contained inside the emitter
 *     itself (`onError` shares the hardened `coerceErrorMessage`), so the
 *     TurnEmitter-independent failsafe below is defense in depth: if the
 *     emitter still fails — a synchronous setup throw from a hostile
 *     `Model` getter, or any residual defect — it pushes a minimal
 *     terminal directly onto the stream. A push/end failure at that last
 *     layer is swallowed: there is no further recovery surface.
 */

import type {
  Api,
  AssistantMessage,
  AssistantMessageEventStream,
  Context,
  Model,
  SimpleStreamOptions,
} from '@earendil-works/pi-ai';
import { createAssistantMessageEventStream } from '@earendil-works/pi-ai';
import type { ChatSession, ChatStreamFinal, PerformanceMetrics } from '@mlx-node/lm';

import type { DiscoveredModelLike } from '../types.js';
import { buildChatConfig, resolveReasoningMode, type ResolvedReasoningMode } from './chat-config.js';
import { contextToChatMessages, toolsToDefinitions } from './convert-messages.js';
import { coerceErrorMessage } from './error-coercion.js';
import { emptyUsage, TurnEmitter } from './events.js';
import { resetPreservingNativeCacheForWarmReuse } from './warm-reuse.js';

/**
 * The exact `MlxModelHost` surface the adapter consumes, kept structural
 * so tests can drive the adapter with a scripted fake host. `MlxModelHost`
 * satisfies this interface as-is.
 */
export interface StreamSimpleHost {
  /** Discovery record for `modelId` (source of the `ModelType` → launch preset). */
  modelInfo(modelId: string): DiscoveredModelLike | undefined;
  /**
   * Atomic resident selection + serialized inference closure (see `MlxModelHost`).
   * `fn` receives a `resident` boolean: `true` when the model was already warm
   * (reused), `false` when this turn had to load/swap it.
   */
  runWithResident<T>(modelId: string, fn: (session: ChatSession, resident: boolean) => Promise<T>): Promise<T>;
  /** Flag the resident as post-error so the next turn does a full reset (see `MlxModelHost`). */
  markResidentDirty(modelId: string): void;
  /** Read-and-clear the resident's post-error flag; `true` ⇒ full-reset this turn. */
  consumeResidentDirty(modelId: string): boolean;
  /** Drop the resident so the next turn reloads it (post-error reset failure). */
  invalidateResident(modelId: string): void;
}

export type PerformanceRecorder = (message: AssistantMessage, performance: PerformanceMetrics) => void;
export type RootCacheOwnerResolver = () => string | undefined;
/** Resolves the root session's JSONL path for the metrics-trace root correlation. */
export type RootSessionFileResolver = () => string | undefined;

/**
 * Durable per-turn telemetry hook. Fires exactly once, only on a SUCCESSFUL
 * native final (never on abort / error / load failure), from the one seam
 * that sees the raw `ChatStreamFinal` alongside the per-request
 * `options.sessionId` and the turn's minted `traceId`. Best-effort: the
 * adapter guards the call so a throwing recorder can never break inference.
 */
export type TurnRecorder = (rec: {
  traceId: string;
  sessionId?: string;
  /** Root session id/file snapshotted when this turn was submitted (see below). */
  rootSessionId?: string;
  rootSessionFile?: string;
  model: string;
  final: ChatStreamFinal;
  /** Whole-turn wall-clock (ms): queue wait + resident selection + prefill + decode. */
  durationMs: number;
  /**
   * Queue + cold-load wait (ms) BEFORE native work began this turn: the gap
   * between turn submission and the serialized `runWithResident` callback
   * firing (behind earlier inference and/or a model load/swap). Subtract from
   * `durationMs` to recover execution-only time.
   */
  queueMs: number;
  /** `true` when the model was already warm/resident, `false` on a cold load/swap this turn. */
  resident: boolean;
}) => void;

/** Property read that must not throw (poisoned getters on a hostile `Model`). */
function safeString(read: () => string, fallback: string): string {
  try {
    const value = read();
    return typeof value === 'string' ? value : fallback;
  } catch {
    return fallback;
  }
}

/** Read numeric model metadata without trusting a user-configurable getter. */
function readPositiveSafeInteger(read: () => unknown): number | undefined {
  try {
    const value = read();
    return typeof value === 'number' && Number.isSafeInteger(value) && value > 0 ? value : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Publish the native model's load-time physical context limit onto pi's shared
 * model object. The parent session and every in-process subagent resolve this
 * same object from one `ModelRegistry`, so the first completed model load gives
 * all later turns the correct auto-compaction window without another channel.
 *
 * This is advisory only: the ChatSession preflight remains the correctness
 * backstop for the first turn and for hostile/invalid getters. Never expand a
 * discovery-time limit here, and never let metadata synchronization break an
 * otherwise valid inference turn.
 */
function publishEffectiveContextWindow(
  model: Model<Api>,
  session: ChatSession,
  configuredModelMaxTokens: number | undefined,
): void {
  try {
    const effective = Math.floor(session.contextLimits()?.effectiveWindowTokens ?? 0);
    if (!Number.isSafeInteger(effective) || effective <= 0) return;
    model.contextWindow = Math.min(model.contextWindow, effective);
    // Use the already-validated snapshot. Reading the shared metadata here
    // would let Math.min coerce invalid user configuration into a valid value
    // that a later turn would then trust.
    if (configuredModelMaxTokens !== undefined) {
      model.maxTokens = Math.min(configuredModelMaxTokens, model.contextWindow);
    }
  } catch {
    // Exact native preflight still protects capacity; keep serving the turn.
  }
}

/**
 * Publish the loaded model's authoritative image capability onto Pi's shared
 * model object. Discovery stays conservatively text-only; the first resident
 * load upgrades the same object before Pi executes any tool call emitted by
 * that inference turn. Return the native truth separately so a hostile/frozen
 * Pi model object cannot prevent image bytes from reaching the provider.
 */
function publishImageCapability(model: Model<Api>, session: ChatSession): boolean {
  let supportsImages = false;
  try {
    supportsImages = session.supportsImages();
  } catch {
    return false;
  }

  try {
    const advertisesImages = model.input.includes('image');
    if (supportsImages && !advertisesImages) {
      model.input = [...model.input, 'image'];
    } else if (!supportsImages && advertisesImages) {
      // Pi reuses model objects across resident swaps. Reconcile a stale
      // positive capability without disturbing any other inputs or their
      // order, so its tools do not return image blocks to a text-only model.
      model.input = model.input.filter((input) => input !== 'image');
    }
  } catch {
    // Native capability remains authoritative for this turn's conversion.
  }
  return supportsImages;
}

/**
 * Minimal terminal `AssistantMessage` for the TurnEmitter-independent
 * failsafe path. Every field read is guarded — this must stay
 * constructible even when the `Model` object itself is hostile (it may be
 * the very reason `TurnEmitter` construction failed).
 */
function failsafeMessage(model: Model<Api>, reason: 'aborted' | 'error', message: string): AssistantMessage {
  return {
    role: 'assistant',
    content: [],
    api: safeString(() => model.api, 'unknown'),
    provider: safeString(() => model.provider, 'unknown'),
    model: safeString(() => model.id, 'unknown'),
    usage: emptyUsage(),
    stopReason: reason,
    errorMessage: message,
    timestamp: Date.now(),
  };
}

export function makeMlxStreamSimple(
  host: StreamSimpleHost,
  onPerformance?: PerformanceRecorder,
  resolveRootCacheOwner?: RootCacheOwnerResolver,
  onTurnRecord?: TurnRecorder,
  onTurnStart?: () => void,
  resolveRootSessionFile?: RootSessionFileResolver,
): (model: Model<Api>, context: Context, options?: SimpleStreamOptions) => AssistantMessageEventStream {
  return (model, context, options) => {
    const stream = createAssistantMessageEventStream();

    /**
     * Exactly-one-terminal guard for the WHOLE turn. `TurnEmitter` has its
     * own `finished` flag, but it cannot cover pre-emitter failures or the
     * failsafe path. Once set, late work — including a resident closure
     * that finally runs after an abort-while-queued — must do nothing.
     */
    let terminated = false;
    let emitter: TurnEmitter | undefined;
    let signal: AbortSignal | undefined;
    let rootCacheOwnerId: string | undefined;
    let rootSessionFile: string | undefined;
    let resolvedReasoning: ResolvedReasoningMode;
    let detachAbort: (() => void) | undefined;

    /**
     * Last-resort terminal, independent of `TurnEmitter` (which may be
     * broken or never constructed). Push/end failures are swallowed — the
     * StreamFn contract forbids throwing into pi and there is no further
     * recovery surface.
     */
    const pushFailsafeTerminal = (reason: 'aborted' | 'error', message: string): void => {
      try {
        stream.push({ type: 'error', reason, error: failsafeMessage(model, reason, message) });
        stream.end();
      } catch {
        // No recovery surface left.
      }
    };

    /**
     * Idempotent terminal: the first caller wins, later callers no-op.
     * Routes through `TurnEmitter` when possible; falls back to the
     * direct-push failsafe when the emitter is missing or throws
     * (defense in depth — `onError` shares the hardened coercion and is
     * not expected to throw).
     */
    const terminalize = (kind: 'aborted' | 'error', err?: unknown): void => {
      if (terminated) return;
      terminated = true;
      detachAbort?.();
      detachAbort = undefined;
      if (kind === 'error') {
        // A native error mid-decode can leave the physical KV ahead of the
        // committed history; flag the resident so the NEXT turn does a full
        // reset (cold prefill) instead of a misaligned warm reuse. Only the
        // error terminal marks dirty — abort / stop / length keep the cache
        // consistent and preserve warm reuse. Guarded: a hostile `model.id`
        // getter must not derail the terminal.
        try {
          host.markResidentDirty(model.id);
        } catch {
          // Nothing to mark — fail safe.
        }
      }
      if (emitter) {
        try {
          if (kind === 'aborted') {
            emitter.onAborted();
          } else {
            emitter.onError(err);
          }
          return;
        } catch {
          // TurnEmitter itself failed — fall through to the failsafe.
        }
      }
      pushFailsafeTerminal(kind, kind === 'aborted' ? 'Request was aborted' : coerceErrorMessage(err));
    };

    const onAbort = (): void => {
      terminalize('aborted');
    };

    try {
      // Synchronous setup is inside the containment too: a hostile
      // `options`/`Model` getter or a TurnEmitter constructor failure must
      // become a stream terminal, never a synchronous throw into pi.
      signal = options?.signal;
      // Snapshot the top-level owner before this request can queue behind
      // another inference. A later /new or /resume must not relabel an older
      // request that was already submitted under the previous root. The metrics
      // root id reuses this same session_start value; only the JSONL path is a
      // separate snapshot, taken here so it can never drift from the id.
      rootCacheOwnerId = resolveRootCacheOwner?.();
      rootSessionFile = resolveRootSessionFile?.();
      // Snapshot once: the native config and the replay provenance must describe
      // the same resolved template mode. Presence alone is wrong for Pi's
      // minimal/low levels, both of which resolve to disabled thinking.
      resolvedReasoning = resolveReasoningMode(options?.reasoning);
      emitter = new TurnEmitter(stream, model, onPerformance, resolvedReasoning.thinkingEnabled);
    } catch (err) {
      terminalize('error', err);
      return stream;
    }
    const turn = emitter;

    // Abort coverage from here has no gap: pre-check a signal that is
    // already aborted (never engage the host at all), then keep a listener
    // installed across the whole queued/running window so a request parked
    // behind stalled inference/loading still terminates promptly.
    if (signal?.aborted) {
      terminalize('aborted');
      return stream;
    }
    if (signal) {
      const s = signal;
      s.addEventListener('abort', onAbort, { once: true });
      detachAbort = () => {
        s.removeEventListener('abort', onAbort);
      };
    }

    void (async () => {
      let sawNativeFinal = false;
      // Bracket the whole serialized turn (queue wait + resident selection +
      // prefill + decode) so a MetricsTrace record can report wall-clock
      // duration. Read at the terminal only on the success path below.
      const turnStartedAt = Date.now();
      await host.runWithResident(model.id, async (session, resident) => {
        // Callback entry: the queue wait + any cold model load/swap have now
        // resolved, but no native work (prime/prefill/decode) has started. This
        // is the seam that separates queue+load latency from execution time —
        // `durationMs` (read at the terminal) still brackets the whole turn.
        const execStartedAt = Date.now();
        const queueMs = Math.max(0, execStartedAt - turnStartedAt);
        // Terminated while queued behind earlier inference/loading (or
        // between stages below): the terminal already went out — skip ALL
        // session work (no warm-reset, no prime, no stream).
        if (terminated) return;
        // Serialized-turn start: fire before any native work so a metrics
        // consumer can snapshot process-wide cold-tier counters here and diff
        // them at the success terminal. Because turns are serialized, a prior
        // turn that aborted/errored has already fully drained by now, so its
        // activity is baked into this snapshot and never attributed to this
        // turn's delta. Best-effort — a throwing hook must not break inference.
        if (onTurnStart) {
          try {
            onTurnStart();
          } catch {
            // Telemetry snapshot is best-effort; never break inference.
          }
        }
        // Snapshot the composed budget before publishing native context limits.
        // The advisory publisher must not read this raw value itself:
        // Math.min can coerce hostile metadata (for example "512" or Infinity)
        // into a finite number that a later turn would then trust.
        const configuredModelMaxTokens = readPositiveSafeInteger(() => model.maxTokens);
        publishEffectiveContextWindow(model, session, configuredModelMaxTokens);
        const supportsImages = publishImageCapability(model, session);
        const discovered = host.modelInfo(model.id);
        if (!discovered) {
          throw new Error(`mlx streamSimple: no discovery record for model "${model.id}"`);
        }
        if (host.consumeResidentDirty(model.id)) {
          // Previous turn errored mid-decode: the physical KV may be ahead of
          // the committed history, so a warm reuse would misalign this
          // replay's prefix. Full-reset (clears native caches + history →
          // hit=0 → cold prefill). If the reset itself fails the session is
          // untrustworthy — drop the resident so the next call reloads it.
          try {
            await session.reset();
          } catch (err) {
            host.invalidateResident(model.id);
            throw err;
          }
        } else {
          await resetPreservingNativeCacheForWarmReuse(session);
        }
        if (terminated) return;
        try {
          session.primeHistory(contextToChatMessages(context, supportsImages));
          const config = buildChatConfig(
            discovered.modelType,
            options,
            toolsToDefinitions(context.tools),
            rootCacheOwnerId,
            resolvedReasoning,
            configuredModelMaxTokens,
          );
          for await (const event of session.startFromHistoryStream(config, signal)) {
            if (event.done) {
              if (event.finishReason === 'error') {
                // In-band native error terminal: chat-session yields a `done`
                // event with `finishReason: 'error'` WITHOUT committing a final
                // (no `sawFinal`), so the physical KV may be ahead of the
                // committed history. Mark the resident dirty (synchronously,
                // inside the callback, so a queued turn observes it) and route
                // to onError — sending it to onFinal treats it as success and
                // skips the dirty flag.
                if (!terminated) {
                  terminated = true;
                  detachAbort?.();
                  detachAbort = undefined;
                  try {
                    host.markResidentDirty(model.id);
                  } catch {
                    // Nothing to mark — fail safe.
                  }
                  try {
                    turn.onError(new Error('native stream reported finishReason=error'));
                  } catch (err) {
                    pushFailsafeTerminal('error', coerceErrorMessage(err));
                  }
                }
              } else {
                sawNativeFinal = true;
                if (!terminated) {
                  terminated = true;
                  detachAbort?.();
                  detachAbort = undefined;
                  try {
                    turn.onFinal(event);
                    // Durable per-turn telemetry: only a successful final gets
                    // here. Guarded independently of onFinal so a throwing
                    // recorder cannot derail the (already-emitted) terminal.
                    if (onTurnRecord) {
                      try {
                        onTurnRecord({
                          traceId: turn.traceId,
                          sessionId: options?.sessionId,
                          // Root correlation is the SUBMIT-time snapshot, not the
                          // live root, so a /new or /resume that landed while this
                          // turn ran can't reattribute it (id == cacheOwner root).
                          rootSessionId: rootCacheOwnerId,
                          rootSessionFile,
                          model: model.id,
                          final: event,
                          durationMs: Math.max(0, Date.now() - turnStartedAt),
                          // Queue+load wait captured at callback entry, and the
                          // warm/cold distinction from `runWithResident` — so the
                          // record can attribute wait vs execution latency.
                          queueMs,
                          resident,
                        });
                      } catch {
                        // Telemetry is best-effort; never break inference.
                      }
                    }
                  } catch (err) {
                    pushFailsafeTerminal('error', coerceErrorMessage(err));
                  }
                }
              }
            } else if (!terminated) {
              turn.onDelta(event);
            }
          }
        } catch (err) {
          // A native decode fault thrown mid-stream can leave the physical KV
          // ahead of the committed history. Flag the resident dirty so the NEXT
          // turn full-resets — SYNCHRONOUSLY here, before this callback rejects
          // and `runSerialized` releases the chain, so a queued turn observes
          // dirty === true (the detached `.catch` terminalize runs too late for
          // that). Abort is excluded: a clean cancel realigns the cache (warm
          // reuse stays valid) and its terminal has already fired. Re-throw so
          // the detached `.catch` still terminalizes the stream.
          if (!signal?.aborted) {
            try {
              host.markResidentDirty(model.id);
            } catch {
              // Nothing to mark — fail safe.
            }
          }
          throw err;
        }
      });
      if (!terminated && !sawNativeFinal) {
        // An aborted native stream ends cleanly with NO final event; any
        // other final-less ending is a native-protocol violation.
        if (signal?.aborted) {
          terminalize('aborted');
        } else {
          terminalize('error', new Error('stream ended without final event'));
        }
      }
    })().catch((err: unknown) => {
      terminalize('error', err);
    });

    return stream;
  };
}

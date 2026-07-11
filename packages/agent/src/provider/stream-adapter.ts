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
import type { ChatSession } from '@mlx-node/lm';

import type { DiscoveredModelLike } from '../types.js';
import { buildChatConfig } from './chat-config.js';
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
  /** Atomic resident selection + serialized inference closure (see `MlxModelHost`). */
  runWithResident<T>(modelId: string, fn: (session: ChatSession) => Promise<T>): Promise<T>;
  /** Flag the resident as post-error so the next turn does a full reset (see `MlxModelHost`). */
  markResidentDirty(modelId: string): void;
  /** Read-and-clear the resident's post-error flag; `true` ⇒ full-reset this turn. */
  consumeResidentDirty(modelId: string): boolean;
  /** Drop the resident so the next turn reloads it (post-error reset failure). */
  invalidateResident(modelId: string): void;
}

/** Property read that must not throw (poisoned getters on a hostile `Model`). */
function safeString(read: () => string, fallback: string): string {
  try {
    const value = read();
    return typeof value === 'string' ? value : fallback;
  } catch {
    return fallback;
  }
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
      emitter = new TurnEmitter(stream, model);
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
      await host.runWithResident(model.id, async (session) => {
        // Terminated while queued behind earlier inference/loading (or
        // between stages below): the terminal already went out — skip ALL
        // session work (no warm-reset, no prime, no stream).
        if (terminated) return;
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
          session.primeHistory(contextToChatMessages(context));
          const config = buildChatConfig(discovered.modelType, options, toolsToDefinitions(context.tools));
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

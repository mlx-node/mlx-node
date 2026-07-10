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
 *     `onAborted`). If the emitter itself fails — a synchronous setup
 *     throw, or `onError` choking while coercing a hostile error value —
 *     a TurnEmitter-independent failsafe pushes a minimal terminal
 *     directly onto the stream. A push/end failure at that last layer is
 *     swallowed: there is no further recovery surface.
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
}

/**
 * Coerce an arbitrary thrown value to a message string without trusting
 * it: an `Error` whose `message` getter throws, an object with a poisoned
 * `toString` / `Symbol.toPrimitive`, a null-prototype object (where
 * `String(err)` itself throws), and a revoked Proxy — where even
 * `err instanceof Error` throws, because `instanceof` walks the prototype
 * chain through the (revoked or throwing) `getPrototypeOf` trap — all
 * land on the constant fallback instead of escaping. Circular objects are
 * fine — `String` never serializes deeply.
 */
export function coerceErrorMessage(err: unknown): string {
  try {
    // The `instanceof` check MUST live inside the guard: on a revoked
    // Proxy (or any Proxy with a throwing `getPrototypeOf` trap) the
    // check itself throws a TypeError before any property is read.
    if (err instanceof Error) {
      const { message } = err;
      if (typeof message === 'string' && message.length > 0) return message;
    }
  } catch {
    // hostile prototype walk or poisoned `message` getter — fall through
  }
  try {
    return String(err);
  } catch {
    return 'unserializable error';
  }
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
     * direct-push failsafe when the emitter is missing or throws (e.g.
     * while coercing a hostile error value).
     */
    const terminalize = (kind: 'aborted' | 'error', err?: unknown): void => {
      if (terminated) return;
      terminated = true;
      detachAbort?.();
      detachAbort = undefined;
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
        await resetPreservingNativeCacheForWarmReuse(session);
        if (terminated) return;
        session.primeHistory(contextToChatMessages(context));
        const config = buildChatConfig(discovered.modelType, options, toolsToDefinitions(context.tools));
        for await (const event of session.startFromHistoryStream(config, signal)) {
          if (event.done) {
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
          } else if (!terminated) {
            turn.onDelta(event);
          }
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

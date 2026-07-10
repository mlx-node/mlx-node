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
 * always terminates. All failures — unknown model, load failure, prime
 * failure, iteration throw — become stream events via `TurnEmitter`
 * (`onError`), and an aborted native stream (generator ends with no final
 * event) becomes a synthesized 'aborted' terminal via `onAborted`.
 */

import type { Api, AssistantMessageEventStream, Context, Model, SimpleStreamOptions } from '@earendil-works/pi-ai';
import { createAssistantMessageEventStream } from '@earendil-works/pi-ai';
import type { ChatSession } from '@mlx-node/lm';

import type { DiscoveredModelLike } from '../types.js';
import { buildChatConfig } from './chat-config.js';
import { contextToChatMessages, toolsToDefinitions } from './convert-messages.js';
import { TurnEmitter } from './events.js';
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

export function makeMlxStreamSimple(
  host: StreamSimpleHost,
): (model: Model<Api>, context: Context, options?: SimpleStreamOptions) => AssistantMessageEventStream {
  return (model, context, options) => {
    const stream = createAssistantMessageEventStream();
    const emitter = new TurnEmitter(stream, model);

    void (async () => {
      let sawNativeFinal = false;
      await host.runWithResident(model.id, async (session) => {
        const discovered = host.modelInfo(model.id);
        if (!discovered) {
          throw new Error(`mlx streamSimple: no discovery record for model "${model.id}"`);
        }
        await resetPreservingNativeCacheForWarmReuse(session);
        session.primeHistory(contextToChatMessages(context));
        const config = buildChatConfig(discovered.modelType, options, toolsToDefinitions(context.tools));
        for await (const event of session.startFromHistoryStream(config, options?.signal)) {
          if (event.done) {
            sawNativeFinal = true;
            emitter.onFinal(event);
          } else {
            emitter.onDelta(event);
          }
        }
      });
      if (!sawNativeFinal) {
        // An aborted native stream ends cleanly with NO final event; any
        // other final-less ending is a native-protocol violation.
        if (options?.signal?.aborted) {
          emitter.onAborted();
        } else {
          emitter.onError(new Error('stream ended without final event'));
        }
      }
    })().catch((err: unknown) => {
      emitter.onError(err);
    });

    return stream;
  };
}

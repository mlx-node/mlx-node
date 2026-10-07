import { randomUUID } from 'node:crypto';
import { resolve } from 'node:path';

import {
  createAssistantMessageEventStream,
  type AssistantMessage,
  type TranscriptContext,
} from '@earendil-works/pi-ai';
import { stream as streamAnthropic } from '@earendil-works/pi-ai/api/anthropic-messages';
import { readDesktopEndpoint } from '@mlx-node/server/host/desktop-endpoint';

import { buildChatConfig } from './chat-config.js';
import { emptyUsage } from './events.js';
import type { makeMlxStreamSimple } from './stream-adapter.js';

/** Once attached, connection errors never authorize another engine or a replay. */
export function desktopStreamFactory(readEndpoint = readDesktopEndpoint): typeof makeMlxStreamSimple {
  const clientId = randomUUID();
  return (host, _onPerformance, rootOwner, _onRecord, _onStart, _rootFile, thinkingBudget) =>
    (model, context, options) => {
      const output = createAssistantMessageEventStream();
      // Capture mutable Pi session/flag state before discovery yields.
      const owner = rootOwner?.();
      void (async () => {
        const budget = thinkingBudget?.();
        options?.signal?.throwIfAborted();
        const endpoint = await readEndpoint();
        if (!endpoint) throw new Error('The mlx-node app inference engine stopped. Restart it and retry.');
        const discovered = host.modelInfo(model.id);
        if (!discovered) throw new Error(`Unknown local model: ${model.id}`);
        const appModel = endpoint.models.find((entry) => entry.name === model.id);
        if (!appModel) throw new Error(`Model ${model.id} is not available in the mlx-node app.`);
        if (resolve(discovered.path) !== resolve(appModel.path)) {
          throw new Error(`Model ${model.id} uses a different model directory in the mlx-node app.`);
        }
        const signal = options?.signal
          ? AbortSignal.any([options.signal, AbortSignal.timeout(3000)])
          : AbortSignal.timeout(3000);
        const headers = { authorization: `Bearer ${endpoint.token}` };
        const health = await fetch(`${endpoint.url}/health`, { headers, signal, redirect: 'error' });
        const state = (await health.json()) as { pid?: number; models?: unknown };
        if (!health.ok || state.pid !== endpoint.pid || !state.models) {
          throw new Error('The mlx-node app inference connection changed. Restart the app and retry.');
        }
        const inventory = await fetch(`${endpoint.url}/v1/models`, { headers, signal, redirect: 'error' });
        const models = (await inventory.json()) as { data?: { id: string }[] };
        if (!inventory.ok || !models.data?.some((entry) => entry.id === model.id)) {
          throw new Error(
            `Model ${model.id} is not available in the mlx-node app. Select a model available in the app.`,
          );
        }
        const config = buildChatConfig(
          discovered.modelType,
          options,
          undefined,
          owner,
          undefined,
          model.maxTokens,
          budget,
        );
        // Pi's Anthropic codec preserves tool calls, images, reasoning, and usage.
        // Keep persisted messages under our mlx identity across local/desktop turns.
        const wireContext = {
          ...context,
          messages: context.messages.map((message) =>
            message.role === 'assistant' && message.provider === model.provider
              ? { ...message, api: 'anthropic-messages' }
              : message,
          ),
        } as TranscriptContext;
        const stream = streamAnthropic(
          { ...model, api: 'anthropic-messages', baseUrl: endpoint.url, headers: undefined, compat: undefined },
          wireContext,
          {
            signal: options?.signal,
            apiKey: endpoint.token,
            maxRetries: 0,
            maxTokens: config.maxNewTokens,
            temperature: config.temperature,
            cacheRetention: 'none',
            // Refuse redirects and any attempt to leave this exact local origin.
            fetch: (input, init) => {
              const url = new URL(input instanceof Request ? input.url : String(input));
              if (url.origin !== endpoint.url) throw new Error('Invalid desktop inference request origin.');
              return fetch(input, { ...init, redirect: 'error' });
            },
            onPayload: (payload) => ({
              ...(payload as object),
              top_p: config.topP,
              top_k: config.topK,
              cache_salt: `${clientId}:${options?.sessionId ?? owner ?? 'default'}`,
              extra_body: {
                reasoning_effort: config.reasoningEffort,
                thinking_budget: config.thinkingTokenBudget,
              },
            }),
          },
        );
        for await (const event of stream) {
          const message = 'partial' in event ? event.partial : event.type === 'done' ? event.message : event.error;
          message.api = model.api;
          output.push(event);
        }
        output.end();
      })().catch((error: unknown) => {
        const reason = options?.signal?.aborted ? 'aborted' : 'error';
        const message: AssistantMessage = {
          role: 'assistant',
          content: [],
          api: model.api,
          provider: model.provider,
          model: model.id,
          usage: emptyUsage(),
          timestamp: Date.now(),
          stopReason: reason,
          errorMessage:
            reason === 'aborted' ? 'Request was aborted' : error instanceof Error ? error.message : String(error),
        };
        output.push({ type: 'error', reason, error: message });
        output.end();
      });
      return output;
    };
}

/** POST /v1/messages/count_tokens — Anthropic Messages token-count endpoint. */

import type { ServerResponse } from 'node:http';

import type { ChatMessage, ToolDefinition } from '@mlx-node/core';

import {
  sendAnthropicBadRequest,
  sendAnthropicInternalError,
  sendAnthropicNotFound,
  sendAnthropicNotImplemented,
} from '../errors.js';
import type { IdleSweeper } from '../idle-sweeper.js';
import { mapAnthropicRequest } from '../mappers/anthropic-request.js';
import type { ModelRegistry, ServableModel } from '../registry.js';
import type { AnthropicCountTokensRequest, AnthropicCountTokensResponse } from '../types-anthropic.js';

interface ChatTemplateTokenCounter {
  applyChatTemplate(
    messages: ChatMessage[],
    addGenerationPrompt?: boolean | null,
    tools?: ToolDefinition[] | null,
    enableThinking?: boolean | null,
  ): Promise<Uint32Array> | Uint32Array;
}

function getChatTemplateTokenCounter(model: ServableModel): ChatTemplateTokenCounter | null {
  const candidate = model as ServableModel & Partial<ChatTemplateTokenCounter>;
  return typeof candidate.applyChatTemplate === 'function' ? (candidate as ChatTemplateTokenCounter) : null;
}

function endJson(res: ServerResponse, body: AnthropicCountTokensResponse): void {
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(body));
}

export async function handleCountMessageTokens(
  res: ServerResponse,
  body: AnthropicCountTokensRequest,
  registry: ModelRegistry,
  idleSweeper?: IdleSweeper | null,
  resolveModel?: (name: string) => Promise<void>,
): Promise<void> {
  if (body == null || typeof body !== 'object') {
    sendAnthropicBadRequest(res, 'Request body must be a JSON object');
    return;
  }
  if (!body.model) {
    sendAnthropicBadRequest(res, 'Missing required field: model');
    return;
  }
  if (!body.messages || !Array.isArray(body.messages) || body.messages.length === 0) {
    sendAnthropicBadRequest(res, 'Missing required field: messages');
    return;
  }

  for (const msg of body.messages) {
    if (msg == null || typeof msg !== 'object') {
      sendAnthropicBadRequest(res, 'Each message must be a non-null object');
      return;
    }
  }

  let mapped: ReturnType<typeof mapAnthropicRequest>;
  try {
    mapped = mapAnthropicRequest(body);
  } catch (err) {
    sendAnthropicBadRequest(res, err instanceof Error ? err.message : 'Invalid request');
    return;
  }

  if (resolveModel) {
    try {
      if (idleSweeper) {
        await idleSweeper.withSuspendedDrains(() => resolveModel(body.model));
      } else {
        await resolveModel(body.model);
      }
    } catch (err) {
      sendAnthropicInternalError(res, err instanceof Error ? err.message : 'Failed to resolve model');
      return;
    }
  }

  const model = registry.get(body.model);
  if (!model) {
    sendAnthropicNotFound(res, `Model "${body.model}" not found`);
    return;
  }

  const counter = getChatTemplateTokenCounter(model);
  if (!counter) {
    sendAnthropicNotImplemented(
      res,
      `Model "${body.model}" does not expose applyChatTemplate(); token counting requires a ` +
        `non-generating chat-template tokenizer API on the registered model.`,
    );
    return;
  }

  try {
    const tokens = await counter.applyChatTemplate(mapped.messages, true, mapped.config.tools ?? null);
    endJson(res, { input_tokens: tokens.length });
  } catch (err) {
    sendAnthropicInternalError(res, err instanceof Error ? err.message : 'Failed to count tokens');
  }
}

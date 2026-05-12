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
import type { ModelWorkCoordinator } from '../model-work-coordinator.js';
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
  modelWorkCoordinator?: ModelWorkCoordinator,
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
      const runResolve = () =>
        idleSweeper ? idleSweeper.withSuspendedDrains(() => resolveModel(body.model)) : resolveModel(body.model);
      if (modelWorkCoordinator) await modelWorkCoordinator.withModelLoad(runResolve);
      else await runResolve();
    } catch (err) {
      sendAnthropicInternalError(res, err instanceof Error ? err.message : 'Failed to resolve model');
      return;
    }
  }

  const lease = registry.acquireDispatchLease(body.model);
  if (!lease) {
    if (registry.get(body.model) != null) {
      sendAnthropicInternalError(res, 'session registry missing for registered model');
      return;
    }
    sendAnthropicNotFound(res, `Model "${body.model}" not found`);
    return;
  }
  const leaseModel = lease.model;
  const sessionReg = lease.registry;
  const preLockInstanceId = lease.instanceId;

  try {
    // Token counting is a pure-CPU tokenize-and-template operation; it must NOT
    // queue behind the per-model generation FIFO (`sessionReg.withExclusive`)
    // because that serializes against multi-minute decode passes and turns a
    // millisecond call into a multi-hundred-second wait. The dispatch lease
    // already pins the model object + binding for the duration of this call,
    // and `withInference` (a shared reader lock against `withModelLoad`) is
    // enough to keep the model from being swapped out mid-tokenize.
    const bindingStillMatchesLease = () => {
      const lockedSessionReg = registry.getSessionRegistry(body.model);
      const lockedInstanceId = registry.getInstanceId(body.model);
      return (
        lockedSessionReg !== undefined &&
        lockedInstanceId !== undefined &&
        lockedSessionReg === sessionReg &&
        lockedInstanceId === preLockInstanceId
      );
    };

    const rejectChangedBinding = (phase: string) => {
      sendAnthropicBadRequest(
        res,
        `Model "${body.model}" binding changed while the token-count request was ${phase}. ` +
          `A concurrent register() re-pointed the name at a different model instance ` +
          `(or released it entirely), so counting against the leased model would use ` +
          `a stale model object. Retry the request — if the swap was intentional, the ` +
          `new binding will service the retry cleanly.`,
      );
    };

    const runCountWithModelRead = async () => {
      if (!bindingStillMatchesLease()) {
        rejectChangedBinding('waiting for the model-load reader gate');
        return;
      }

      const counter = getChatTemplateTokenCounter(leaseModel);
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
        if (!bindingStillMatchesLease()) {
          rejectChangedBinding('running');
          return;
        }
        endJson(res, { input_tokens: tokens.length });
      } catch (err) {
        sendAnthropicInternalError(res, err instanceof Error ? err.message : 'Failed to count tokens');
      }
    };

    if (modelWorkCoordinator) await modelWorkCoordinator.withInference(runCountWithModelRead);
    else await runCountWithModelRead();
  } catch (err) {
    sendAnthropicInternalError(res, err instanceof Error ? err.message : 'Failed to count tokens');
  } finally {
    registry.releaseDispatchLease(leaseModel);
  }
}

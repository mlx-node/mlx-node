/** POST /v1/messages/count_tokens — Anthropic Messages token-count endpoint. */

import type { IncomingMessage, ServerResponse } from 'node:http';

import { countPromptTokens, isImageProcessingError, validateRequestImagesAsync } from '@mlx-node/lm/chat-session';

import {
  sendAnthropicBadRequest,
  sendAnthropicInternalError,
  sendAnthropicNotFound,
  sendAnthropicNotImplemented,
} from '../errors.js';
import type { IdleSweeper } from '../idle-sweeper.js';
import { residentSessionRegistry } from '../inference-admission.js';
import { mapAnthropicRequest } from '../mappers/anthropic-request.js';
import { ModelLoadCancelledError, type ModelWorkCoordinator } from '../model-work-coordinator.js';
import type { ModelRegistry } from '../registry.js';
import { stringifyStrictJson } from '../strict-json.js';
import type { AnthropicCountTokensRequest, AnthropicCountTokensResponse } from '../types-anthropic.js';

function endJson(res: ServerResponse, body: AnthropicCountTokensResponse): void {
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(stringifyStrictJson(body));
}

/** `levelsFamily`: as `ServerThinking.levelsFamily` of `/v1/messages`, so the count renders the prompt it renders. */
export async function handleCountMessageTokens(
  res: ServerResponse,
  body: AnthropicCountTokensRequest,
  registry: ModelRegistry,
  idleSweeper?: IdleSweeper | null,
  resolveModel?: (name: string) => Promise<void>,
  modelWorkCoordinator?: ModelWorkCoordinator,
  bindLoadedModel?: (name: string) => boolean,
  levelsFamily?: (model: string) => string | undefined,
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
    mapped = mapAnthropicRequest(body, levelsFamily?.(body.model));
  } catch (err) {
    sendAnthropicBadRequest(res, err instanceof Error ? err.message : 'Invalid request');
    return;
  }

  const abortController = new AbortController();
  const abortReq = (res as { req?: IncomingMessage }).req;
  const abortSocket = res.socket ?? abortReq?.socket ?? null;
  const onAbortClose = (): void => {
    abortController.abort();
  };
  const onAbortReqClose = (): void => {
    if (abortReq?.complete !== true) {
      abortController.abort();
    }
  };
  const onAbortError = (_err: unknown): void => {
    abortController.abort();
  };
  const isTransportDisconnected = (): boolean =>
    abortController.signal.aborted ||
    res.destroyed ||
    res.writableEnded ||
    abortSocket?.destroyed === true ||
    abortReq?.aborted === true;
  let abortListenersAttached = false;
  const armAbortListeners = (): void => {
    if (abortListenersAttached) return;
    res.once('close', onAbortClose);
    res.once('error', onAbortError);
    if (abortSocket != null) {
      abortSocket.once('close', onAbortClose);
    }
    if (abortReq) {
      abortReq.once('aborted', onAbortClose);
      abortReq.once('close', onAbortReqClose);
      abortReq.once('error', onAbortError);
    }
    if (isTransportDisconnected()) {
      abortController.abort();
    }
    abortListenersAttached = true;
  };

  let acquiredLeaseModel: NonNullable<ReturnType<ModelRegistry['acquireDispatchLease']>>['model'] | undefined;

  try {
    // The writer bracket waits for every running answer; a resident name must not take it, as in `/v1/messages`.
    const residentRegistry = residentSessionRegistry(registry, body.model, bindLoadedModel, modelWorkCoordinator);
    if (resolveModel && !residentRegistry) {
      armAbortListeners();
      if (isTransportDisconnected()) {
        abortController.abort();
        return;
      }
      try {
        const runResolve = () => {
          if (isTransportDisconnected()) {
            abortController.abort();
            throw new ModelLoadCancelledError();
          }
          return idleSweeper
            ? idleSweeper.withSuspendedDrains(() => resolveModel(body.model))
            : resolveModel(body.model);
        };
        if (modelWorkCoordinator) {
          await modelWorkCoordinator.withModelLoad(runResolve, undefined, abortController.signal);
        } else {
          await runResolve();
        }
      } catch (err) {
        if (err instanceof ModelLoadCancelledError || isTransportDisconnected()) {
          return;
        }
        sendAnthropicInternalError(res, err instanceof Error ? err.message : 'Failed to resolve model');
        return;
      }
      if (isTransportDisconnected()) {
        abortController.abort();
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
    acquiredLeaseModel = leaseModel;
    const sessionReg = lease.registry;
    const preLockInstanceId = lease.instanceId;

    const hasImages = mapped.messages.some((message) => (message.images?.length ?? 0) > 0);
    if (hasImages && leaseModel.supportsImages?.() !== true) {
      sendAnthropicBadRequest(res, `Could not process image: the model "${body.model}" does not take images.`);
      return;
    }
    if (hasImages) {
      armAbortListeners();
      try {
        await validateRequestImagesAsync(mapped.messages, { signal: abortController.signal });
      } catch {
        if (abortController.signal.aborted) {
          return;
        }
        sendAnthropicBadRequest(res, 'Could not process image');
        return;
      }
    }

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
      if (abortController.signal.aborted) {
        return;
      }
      if (!bindingStillMatchesLease()) {
        rejectChangedBinding('waiting for the model-load reader gate');
        return;
      }

      if (typeof leaseModel.applyChatTemplate !== 'function') {
        sendAnthropicNotImplemented(
          res,
          `Model "${body.model}" does not expose applyChatTemplate(); token counting requires a ` +
            `non-generating chat-template tokenizer API on the registered model.`,
        );
        return;
      }

      try {
        // The same count the context clamp uses, with the effort the session would normalize.
        const effort = leaseModel.normalizeReasoningEffort
          ? leaseModel.normalizeReasoningEffort(mapped.config.reasoningEffort)
          : mapped.config.reasoningEffort;
        const inputTokens = await countPromptTokens(leaseModel, mapped.messages, {
          tools: mapped.config.tools,
          reasoningEffort: effort,
        });
        if (abortController.signal.aborted) {
          return;
        }
        if (!bindingStillMatchesLease()) {
          rejectChangedBinding('running');
          return;
        }
        endJson(res, { input_tokens: inputTokens });
      } catch (err) {
        if (abortController.signal.aborted) {
          return;
        }
        if (isImageProcessingError(err)) {
          sendAnthropicBadRequest(res, 'Could not process image');
        } else {
          sendAnthropicInternalError(res, err instanceof Error ? err.message : 'Failed to count tokens');
        }
      }
    };

    armAbortListeners();
    if (abortController.signal.aborted) {
      return;
    }
    if (modelWorkCoordinator) {
      await modelWorkCoordinator.withInference(runCountWithModelRead, abortController.signal);
    } else {
      await runCountWithModelRead();
    }
  } catch (err) {
    if (abortController.signal.aborted) {
      return;
    }
    sendAnthropicInternalError(res, err instanceof Error ? err.message : 'Failed to count tokens');
  } finally {
    if (abortListenersAttached) {
      res.removeListener('close', onAbortClose);
      res.removeListener('error', onAbortError);
      if (abortSocket != null) {
        abortSocket.removeListener('close', onAbortClose);
      }
      if (abortReq) {
        abortReq.removeListener('aborted', onAbortClose);
        abortReq.removeListener('close', onAbortReqClose);
        abortReq.removeListener('error', onAbortError);
      }
      abortListenersAttached = false;
    }
    if (acquiredLeaseModel !== undefined) {
      registry.releaseDispatchLease(acquiredLeaseModel);
    }
  }
}

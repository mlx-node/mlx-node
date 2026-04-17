/** Path-based router for /v1/* endpoints. */

import type { IncomingMessage, ServerResponse } from 'node:http';

import type { ResponseStore } from '@mlx-node/core';

import { handleCreateMessage } from './endpoints/messages.js';
import { handleListModels } from './endpoints/models.js';
import { handleCreateResponse } from './endpoints/responses.js';
import {
  sendAnthropicBadRequest,
  sendAnthropicMethodNotAllowed,
  sendBadRequest,
  sendMethodNotAllowed,
  sendNotFound,
} from './errors.js';
import type { ModelRegistry } from './registry.js';
import type { AnthropicMessagesRequest } from './types-anthropic.js';
import type { ResponsesAPIRequest } from './types.js';

/** Max request body size (10 MB). */
const MAX_BODY_BYTES = 10 * 1024 * 1024;

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let totalBytes = 0;
    req.on('data', (chunk: Buffer) => {
      totalBytes += chunk.length;
      if (totalBytes > MAX_BODY_BYTES) {
        reject(new Error('Request body too large'));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => resolve(Buffer.concat(chunks).toString('utf-8')));
    req.on('error', reject);
  });
}

export async function routeRequest(
  req: IncomingMessage,
  res: ServerResponse,
  registry: ModelRegistry,
  store: ResponseStore | null,
  responseRetentionSec?: number,
): Promise<void> {
  const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
  const path = url.pathname;

  if (path === '/v1/models') {
    if (req.method !== 'GET') {
      sendMethodNotAllowed(res, 'GET');
      return;
    }
    handleListModels(res, registry);
    return;
  }

  if (path === '/v1/responses') {
    if (req.method !== 'POST') {
      sendMethodNotAllowed(res, 'POST');
      return;
    }

    let body: ResponsesAPIRequest;
    try {
      const raw = await readBody(req);
      body = JSON.parse(raw) as ResponsesAPIRequest;
    } catch (err) {
      const msg =
        err instanceof Error && err.message === 'Request body too large' ? err.message : 'Invalid JSON in request body';
      sendBadRequest(res, msg);
      return;
    }

    await handleCreateResponse(res, body, registry, store, req, responseRetentionSec);
    return;
  }

  if (path === '/v1/messages') {
    if (req.method !== 'POST') {
      sendAnthropicMethodNotAllowed(res, 'POST');
      return;
    }

    let body: AnthropicMessagesRequest;
    try {
      const raw = await readBody(req);
      body = JSON.parse(raw) as AnthropicMessagesRequest;
    } catch (err) {
      const msg =
        err instanceof Error && err.message === 'Request body too large' ? err.message : 'Invalid JSON in request body';
      sendAnthropicBadRequest(res, msg);
      return;
    }

    await handleCreateMessage(res, body, registry, req);
    return;
  }

  if (path === '/health' || path === '/v1/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok' }));
    return;
  }

  sendNotFound(res, `No route matches ${req.method} ${path}`);
}

import { spawn, type ChildProcess } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { open, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { fileURLToPath } from 'node:url';

import {
  createAssistantMessageEventStream,
  type AssistantMessage,
  type AssistantMessageEvent,
} from '@earendil-works/pi-ai';

import { emptyUsage } from './events.js';
import { SharedEventDecoder } from './shared-events.js';
import {
  SHARED_PROTOCOL,
  sharedLocation,
  type SharedEndpoint,
  type SharedFrame,
  type SharedRequest,
} from './shared-protocol.js';
import type { makeMlxStreamSimple } from './stream-adapter.js';

/** A failed shared service is an error, never permission to load another model locally. */
export async function ensureSharedWorker(signal?: AbortSignal): Promise<SharedEndpoint> {
  const { directory, port } = sharedLocation();
  const probe = async (): Promise<SharedEndpoint | undefined> => {
    signal?.throwIfAborted();
    let endpoint: SharedEndpoint;
    try {
      endpoint = JSON.parse(await readFile(join(directory, 'endpoint.json'), 'utf8'));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined;
      throw error;
    }
    if (endpoint.port !== port || !endpoint.token) throw new Error('Invalid shared delegate endpoint.');
    let response: Response;
    try {
      response = await fetch(`http://127.0.0.1:${port}/health`, {
        headers: { authorization: `Bearer ${endpoint.token}` },
        signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(1000)]) : AbortSignal.timeout(1000),
        redirect: 'error',
      });
    } catch {
      signal?.throwIfAborted();
      return undefined;
    }
    if (!response.ok) {
      await response.body?.cancel();
      return undefined;
    }
    const health = (await response.json()) as { protocol?: string; pid?: number };
    if (health.protocol !== SHARED_PROTOCOL || endpoint.protocol !== SHARED_PROTOCOL) {
      throw new Error('The shared delegate service uses a different protocol. Stop the old service and retry.');
    }
    return health.pid === endpoint.pid ? endpoint : undefined;
  };
  const running = await probe();
  if (running) return running;
  mkdirSync(directory, { recursive: true, mode: 0o700 });
  const deadline = Date.now() + 15_000;
  let child: ChildProcess | undefined;
  let spawnError: Error | undefined;
  while (Date.now() < deadline) {
    signal?.throwIfAborted();
    if (spawnError) throw spawnError;
    if (!child || child.exitCode === 0) {
      // A contender exits successfully if the retiring worker still holds the
      // election port. Try again after it exits, once cleanup can release it.
      const log = await open(join(directory, 'service.log'), 'a', 0o600);
      try {
        const entry = new URL(
          import.meta.url.endsWith('.ts') ? './shared-worker.ts' : './shared-worker.js',
          import.meta.url,
        );
        child = spawn(
          process.execPath,
          [...process.execArgv.filter((arg) => !arg.startsWith('--inspect')), fileURLToPath(entry)],
          { detached: true, stdio: ['ignore', 'ignore', log.fd] },
        );
        child.on('error', (error) => {
          spawnError = error;
        });
        child.unref();
      } finally {
        await log.close();
      }
    }
    await delay(100, undefined, { signal });
    const endpoint = await probe();
    if (endpoint) return endpoint;
    if (child.exitCode !== null && child.exitCode !== 0) break;
    if (child.signalCode !== null) break;
  }
  throw new Error(
    `The shared delegate service could not start. See ${join(directory, 'service.log')}. No local model was loaded.`,
  );
}

async function openSharedStream(
  body: string,
  connect: typeof ensureSharedWorker,
  signal?: AbortSignal,
): Promise<ReadableStream<Uint8Array>> {
  for (let attempt = 0; ; attempt++) {
    signal?.throwIfAborted();
    const endpoint = await connect(signal);
    let response: Response;
    try {
      response = await fetch(`http://127.0.0.1:${endpoint.port}/stream`, {
        method: 'POST',
        redirect: 'error',
        signal,
        headers: { 'content-type': 'application/json', authorization: `Bearer ${endpoint.token}` },
        body,
      });
    } catch (error) {
      signal?.throwIfAborted();
      // Connection refusal proves that inference never started. A reset or a
      // truncated response does not, so those failures must not replay a turn.
      if (attempt < 2 && error instanceof Error && (error.cause as NodeJS.ErrnoException)?.code === 'ECONNREFUSED')
        continue;
      throw error;
    }
    if (response.ok && response.body) return response.body;
    await response.body?.cancel();
    // Retirement and stale credentials both reject before backend admission.
    // Queue saturation (429) is deliberately not a worker rotation signal.
    if (attempt < 2 && (response.status === 503 || response.status === 401)) continue;
    throw new Error(`Shared delegate inference failed (HTTP ${response.status}).`);
  }
}

export function sharedStreamFactory(
  profile: Pick<SharedRequest['profile'], 'persistPagedCache' | 'preserveEmbeddedGemmaDraft'>,
  connect = ensureSharedWorker,
): typeof makeMlxStreamSimple {
  return (host, onPerformance, rootOwner, onRecord, _onStart, rootFile, thinkingBudget) =>
    (model, context, options) => {
      const stream = createAssistantMessageEventStream();
      let terminal: Extract<AssistantMessageEvent, { type: 'done' | 'error' }> | undefined;
      const events = new SharedEventDecoder();
      let settled = false;
      const fail = (error: unknown): void => {
        if (settled) return;
        settled = true;
        const reason = options?.signal?.aborted ? 'aborted' : 'error';
        const message: AssistantMessage = {
          ...(events.partial ?? {
            role: 'assistant',
            content: [],
            api: model.api,
            provider: model.provider,
            model: model.id,
            usage: emptyUsage(),
            timestamp: Date.now(),
          }),
          stopReason: reason,
          errorMessage:
            reason === 'aborted' ? 'Request was aborted' : error instanceof Error ? error.message : String(error),
        };
        for (const event of events.finishOpenBlocks()) stream.push(event);
        stream.push({ type: 'error', reason, error: message });
        stream.end();
      };
      // Capture mutable session/flag state now, before waiting for service startup.
      let request: SharedRequest;
      try {
        const discovered = host.modelInfo(model.id);
        if (!discovered) throw new Error(`Unknown local model: ${model.id}`);
        request = {
          profile: { ...profile, discovered },
          model,
          context,
          options: {
            sessionId: options?.sessionId,
            maxTokens: options?.maxTokens,
            temperature: options?.temperature,
            reasoning: options?.reasoning,
            thinkingBudgets: options?.thinkingBudgets,
          },
          rootSessionId: rootOwner?.(),
          rootSessionFile: rootFile?.(),
          thinkingBudget: thinkingBudget?.(),
        };
      } catch (error) {
        fail(error);
        return stream;
      }
      const abort = () => fail(new Error('Request was aborted'));
      options?.signal?.addEventListener('abort', abort, { once: true });
      void (async () => {
        const body = await openSharedStream(JSON.stringify(request), connect, options?.signal);
        let buffered = '';
        const decoder = new TextDecoder();
        const reader = body.getReader();
        try {
          while (!settled) {
            const { done, value } = await reader.read();
            if (done) break;
            buffered += decoder.decode(value, { stream: true });
            let newline: number;
            while ((newline = buffered.indexOf('\n')) >= 0) {
              const frame = JSON.parse(buffered.slice(0, newline)) as SharedFrame;
              buffered = buffered.slice(newline + 1);
              if ('error' in frame) throw new Error(frame.error);
              if ('event' in frame) {
                const event = events.decode(frame.event);
                if (event.type === 'done' || event.type === 'error') terminal = event;
                else {
                  stream.push(event);
                }
              } else {
                // Metadata is advisory and must never corrupt a successful completion.
                try {
                  if ('performance' in frame)
                    onPerformance?.(terminal?.type === 'done' ? terminal.message : frame.message, frame.performance);
                  else if ('record' in frame) onRecord?.(frame.record);
                  else Object.assign(model, frame.model);
                } catch {
                  /* best effort */
                }
              }
            }
            if (buffered.length > 64 * 1024 * 1024) throw new Error('Shared inference frame is too large.');
          }
          if (settled) return;
          if (!terminal || buffered.length) throw new Error('Shared inference connection ended before completion.');
          settled = true;
          stream.push(terminal);
          stream.end();
        } finally {
          await reader.cancel().catch(() => {});
          reader.releaseLock();
        }
      })()
        .catch(fail)
        .finally(() => options?.signal?.removeEventListener('abort', abort));
      return stream;
    };
}

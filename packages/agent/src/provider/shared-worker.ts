import { encodeSharedEvent } from './shared-events.js';
import { sharedCacheOwners, sharedLocation } from './shared-protocol.js';
import { startSharedService, type SharedBackend } from './shared-service.js';

/** Imports native code only in the elected process, on its first inference request. */
async function loadBackend(): Promise<SharedBackend> {
  const [{ SharedInferenceHost }, { makeMlxStreamSimple }, { coldCacheDrain }] = await Promise.all([
    import('./shared-host.js'),
    import('./stream-adapter.js'),
    import('@mlx-node/core'),
  ]);
  const host = new SharedInferenceHost();
  return {
    busy: () => host.pending > 0,
    close: async () => {
      await host.close();
      coldCacheDrain(5000);
    },
    async *stream(request, signal) {
      const { model, options } = request;
      let performance: import('./shared-protocol.js').SharedFrame | undefined;
      let record: import('./shared-protocol.js').SharedFrame | undefined;
      const owners = sharedCacheOwners(request);
      const stream = makeMlxStreamSimple(
        host.forRequest(request.profile, signal),
        (message, metrics) => {
          performance = { message, performance: metrics };
        },
        () => owners.root,
        (value) => {
          record = { record: { ...value, sessionId: options.sessionId, rootSessionId: request.rootSessionId } };
        },
        undefined,
        () => request.rootSessionFile,
        () => request.thinkingBudget,
      )(model, request.context, { ...options, sessionId: owners.owner, signal });
      for await (const event of stream) yield { event: encodeSharedEvent(event) };
      if (performance) yield performance;
      if (record) yield record;
      yield { model: { contextWindow: model.contextWindow, maxTokens: model.maxTokens, input: model.input } };
    },
  };
}

process.env.MLX_PAGED_PREFILL_CHUNK_SIZE ??= '2048';
try {
  const service = await startSharedService({ ...sharedLocation(), loadBackend });
  for (const signal of ['SIGINT', 'SIGTERM'] as const) {
    process.once(signal, () => {
      void service.close().catch((error: unknown) => {
        console.error(error);
        process.exitCode = 1;
      });
    });
  }
} catch (error) {
  // Simultaneous first callers may launch contenders; only the bound worker survives.
  if ((error as NodeJS.ErrnoException).code !== 'EADDRINUSE') {
    console.error(error);
    process.exitCode = 1;
  }
}

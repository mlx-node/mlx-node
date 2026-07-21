/**
 * `createMlxProviderExtension` — the pi inline extension that registers
 * the in-process `mlx` provider.
 *
 * Task 8's `runAgent` passes the returned extension into pi's `main()`
 * via `extensionFactories`; pi calls the factory during extension load,
 * and `registerProvider` makes every discovered local model resolvable
 * as `mlx/<dir-name>` with no /login (the literal apiKey marks the
 * models available).
 *
 * Import discipline (load-bearing): pi is import-order sensitive to its
 * config env vars, so this module — which the CLI imports BEFORE those
 * env vars are set — must not runtime-import `@earendil-works/pi-coding-agent`
 * at module top level. Only type-only pi imports appear here; the
 * `ExtensionAPI` value arrives as the factory argument.
 */

import type { ExtensionAPI, InlineExtension } from '@earendil-works/pi-coding-agent';
import { coldCacheStats, type ColdCacheStats } from '@mlx-node/core';

import { MetricsTrace, type MetricsTraceRecord } from './metrics-trace.js';
import { MlxModelHost } from './model-host.js';
import type { MlxModelInfo } from './models.js';
import { PerformanceStatus } from './performance-status.js';
import { makeMlxStreamSimple, type TurnRecorder } from './stream-adapter.js';

/** Read the process-wide cold-tier snapshot; the native addon may be absent (unit tests). */
function safeColdStats(): ColdCacheStats | undefined {
  try {
    return coldCacheStats();
  } catch {
    return undefined;
  }
}

/**
 * Build the `mlx-provider` inline extension serving `models`. The host
 * (one per process — it owns the single GPU-resident model) is created
 * eagerly so repeated factory invocations can never spawn a second
 * host, but stays lazy about weights: nothing loads until the first
 * `streamSimple` call.
 */
export function createMlxProviderExtension(models: MlxModelInfo[], host?: MlxModelHost): InlineExtension {
  const resolvedHost = host ?? new MlxModelHost(models.map((m) => m.discovered));
  const performanceStatus = new PerformanceStatus();
  const metricsTrace = new MetricsTrace();
  // This closure outlives Pi runtime replacement. Pi creates a replacement
  // runtime for /new and /resume and reruns inline extension factories; each
  // new factory's session_start updates the root while child sessions keep
  // using this registered stream.
  let rootCacheOwnerId: string | undefined;

  // Cold-tier counters are cumulative since the tier opened. Keep the prior
  // snapshot so each turn records its own delta. Inference is serialized per
  // process (host promise chain), so the SYNCHRONOUS counters (hits/misses/
  // bytesRestored) are exact per-turn; bytesWritten advances on the async
  // writer thread, so its delta is approximate (documented on the record).
  let prevCold = safeColdStats();
  const onTurnRecord: TurnRecorder = ({ traceId, sessionId, model, final, durationMs }) => {
    const rec: Omit<MetricsTraceRecord, 'v' | 'rootSessionId' | 'rootSessionFile'> = {
      traceId,
      ts: Date.now(),
      sessionId,
      model,
      durationMs,
      finishReason: final.finishReason,
      promptTokens: final.promptTokens,
      cachedTokens: final.cachedTokens ?? 0,
      outputTokens: final.numTokens,
      reasoningTokens: final.reasoningTokens,
    };
    const perf = final.performance;
    if (perf) {
      rec.ttftMs = perf.ttftMs;
      rec.prefillTps = perf.prefillTokensPerSecond;
      rec.decodeTps = perf.decodeTokensPerSecond;
      rec.mtpCycles = perf.mtpCycles;
      // mlx-vlm-comparable headline accept rate (committed tokens per cycle).
      rec.mtpMeanAccepted = perf.mtpMeanAcceptedTokensTotal;
    }
    const cold = safeColdStats();
    if (cold && prevCold) {
      rec.coldHits = cold.hits - prevCold.hits;
      rec.coldMisses = cold.misses - prevCold.misses;
      rec.coldBytesWritten = cold.bytesWritten - prevCold.bytesWritten;
      rec.coldBytesRestored = cold.bytesRestored - prevCold.bytesRestored;
    }
    if (cold) prevCold = cold;
    metricsTrace.record(rec);
  };

  const streamSimple = makeMlxStreamSimple(
    resolvedHost,
    performanceStatus.record,
    () => rootCacheOwnerId,
    onTurnRecord,
  );
  return {
    name: 'mlx-provider',
    factory: (pi: ExtensionAPI) => {
      pi.registerProvider('mlx', {
        api: 'mlx',
        baseUrl: 'mlx://local',
        apiKey: 'mlx-local',
        streamSimple,
        models: models.map((m) => m.piModel),
      });
      pi.on('session_start', (_event, ctx) => {
        rootCacheOwnerId = ctx.sessionManager.getSessionId();
        // Correlate every subsequent turn's MetricsTrace record with the
        // active root session. `getSessionFile` is optional-chained so a
        // minimal test/mock session manager without it still works.
        metricsTrace.setRootSession(ctx.sessionManager.getSessionId(), ctx.sessionManager.getSessionFile?.());
      });
      pi.on('message_end', (event, ctx) => {
        performanceStatus.showMessage(event, ctx);
      });
      // Do not clear on turn_start. Pi emits a fresh turn after every tool
      // result, before the next inference has terminal metrics to replace the
      // completed sample; clearing here makes the footer disappear precisely
      // while a long tool-follow-up prefill is running.
      pi.on('model_select', (_event, ctx) => {
        performanceStatus.clear(ctx);
      });
      pi.on('session_shutdown', (_event, ctx) => {
        performanceStatus.clear(ctx);
      });
    },
  };
}

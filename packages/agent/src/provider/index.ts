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
import { coldCacheStats, coldSidecarStats, type ColdCacheStats, type ColdSidecarStats } from '@mlx-node/core';

import { canonicalCacheRoot } from '../cold-tier.js';
import { MetricsTrace, type MetricsTraceRecord } from './metrics-trace.js';
import { MLX_API, MLX_API_KEY, MLX_BASE_URL, MLX_PROVIDER_ID } from './mlx-identity.js';
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
 * Read the process-wide sidecar counters. Separate from {@link safeColdStats}
 * because it reads a different native struct that has no `enabled` gate: the
 * counters live in a plain static, not in the tier, so they are valid even
 * when the tier never opened — which is exactly the case a run with zero
 * sidecars needs distinguished.
 */
function safeSidecarStats(): ColdSidecarStats | undefined {
  try {
    return coldSidecarStats();
  } catch {
    return undefined;
  }
}

/** Injectable seams for {@link createMlxProviderExtension} (unit tests). */
export interface MlxProviderExtensionDeps {
  /** Process-wide cold-tier reader; defaults to the native addon (absent in unit tests). */
  coldStats?: () => ColdCacheStats | undefined;
  /** Process-wide sidecar-counter reader; defaults to the native addon. */
  sidecarStats?: () => ColdSidecarStats | undefined;
  /** Durable per-turn telemetry sink; defaults to a fresh {@link MetricsTrace}. */
  metricsTrace?: MetricsTrace;
}

/**
 * Build the `mlx-provider` inline extension serving `models`. The host
 * (one per process — it owns the single GPU-resident model) is created
 * eagerly so repeated factory invocations can never spawn a second
 * host, but stays lazy about weights: nothing loads until the first
 * `streamSimple` call.
 */
export function createMlxProviderExtension(
  models: MlxModelInfo[],
  host?: MlxModelHost,
  deps: MlxProviderExtensionDeps = {},
): InlineExtension {
  const resolvedHost = host ?? new MlxModelHost(models.map((m) => m.discovered));
  const performanceStatus = new PerformanceStatus();
  const metricsTrace = deps.metricsTrace ?? new MetricsTrace();
  const readColdStats = deps.coldStats ?? safeColdStats;
  const readSidecarStats = deps.sidecarStats ?? safeSidecarStats;
  // This closure outlives Pi runtime replacement. Pi creates a replacement
  // runtime for /new and /resume and reruns inline extension factories; each
  // new factory's session_start updates the root while child sessions keep
  // using this registered stream. The root id doubles as the cache owner and
  // the metrics-trace root; the JSONL path travels alongside it for metrics.
  let rootCacheOwnerId: string | undefined;
  let rootSessionFile: string | undefined;

  // Cold-tier counters are cumulative since the tier opened. Snapshot them at
  // each turn's native start (`onTurnStart`, fired inside the serialized host
  // closure) and diff at the success terminal, so a turn that aborted or
  // errored — which never reaches `onTurnRecord` — can't leak its restores into
  // the next successful turn's delta. Inference is serialized per process (host
  // promise chain), so by the time a turn snapshots, any prior turn has fully
  // drained. The SYNCHRONOUS counters (hits/misses/bytesRestored/enqueued/
  // corruptions/queueDrops/restoreDeclines) are exact per-turn; bytesWritten,
  // evictions and writeErrors advance on the async writer thread, so those
  // deltas are approximate (documented per field on the record).
  //
  // The SIDECAR counters are snapshotted the same way but are exact without
  // exception: every one of them is recorded inside the native turn finalize,
  // on the calling thread, before the turn returns. They also have no `enabled`
  // gate — they live in a plain process static rather than in the tier — so
  // they read honestly on a run where the tier never opened, which is the run
  // that most needs them.
  let turnStartCold = readColdStats();
  let turnStartSidecar = readSidecarStats();
  const onTurnStart = (): void => {
    turnStartCold = readColdStats();
    turnStartSidecar = readSidecarStats();
  };
  const onTurnRecord: TurnRecorder = ({
    traceId,
    sessionId,
    rootSessionId,
    rootSessionFile,
    model,
    final,
    durationMs,
    queueMs,
    resident,
  }) => {
    const rec: Omit<MetricsTraceRecord, 'v'> = {
      traceId,
      ts: Date.now(),
      sessionId,
      rootSessionId,
      rootSessionFile,
      model,
      durationMs,
      queueMs,
      resident,
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
    const cold = readColdStats();
    if (cold && turnStartCold) {
      rec.coldHits = cold.hits - turnStartCold.hits;
      rec.coldMisses = cold.misses - turnStartCold.misses;
      rec.coldBytesWritten = cold.bytesWritten - turnStartCold.bytesWritten;
      rec.coldBytesRestored = cold.bytesRestored - turnStartCold.bytesRestored;
      rec.coldEnqueued = cold.enqueued - turnStartCold.enqueued;
      rec.coldQueueDrops = cold.queueDrops - turnStartCold.queueDrops;
      rec.coldEvictions = cold.evictions - turnStartCold.evictions;
      rec.coldCorruptions = cold.corruptions - turnStartCold.corruptions;
      rec.coldWriteErrors = cold.writeErrors - turnStartCold.writeErrors;
      rec.coldRestoreDeclines = cold.restoreDeclines - turnStartCold.restoreDeclines;
      // Absolutes, not deltas: an aborted/errored turn never reaches this
      // recorder, so a corruption or a dropped write during one lands in NO
      // delta. The cumulative counter observed by the next successful turn
      // still carries it, which is what makes "corruptions must be 0"
      // checkable at all (`MAX(total) > 0` over any window).
      rec.coldCorruptionsTotal = cold.corruptions;
      rec.coldQueueDropsTotal = cold.queueDrops;
      // Same latch, and the counter that needs it most: a write error is
      // raised on the background writer, so the one covering the LAST turn
      // before a crash lands in no delta at all — and "is my cache root
      // broken?" is a question about ever, not about this turn.
      rec.coldWriteErrorsTotal = cold.writeErrors;
      // Cache IDENTITY comes from the END-of-turn snapshot, never the baseline:
      // the tier opens LAZILY on first use, so on a process's first turn
      // `turnStartCold` is the all-zero default with `enabled: false` and an
      // empty root. Canonicalized here — in the writer — so the dashboard
      // never has to match whatever spelling Rust happened to construct.
      rec.coldEnabled = cold.enabled;
      // ONE emptiness test, applied to the CANONICAL value. Gating on the raw
      // native string used a DIFFERENT test from the one `canonicalCacheRoot`
      // applies (it trims), so a whitespace-only root passed `length > 0` here
      // and canonicalized to `''` — which `MetricsTrace.record` then dropped,
      // leaving a row that says the tier was ON while carrying no root at all.
      if (cold.enabled) {
        const canonicalRoot = canonicalCacheRoot(cold.root);
        if (canonicalRoot.length > 0) rec.coldRoot = canonicalRoot;
      }
    }
    // Its own guard, deliberately not folded into the block above: the sidecar
    // reader never consults the tier, so it keeps reporting when `cold` is
    // undefined or the tier failed to open. A run with `coldHits: 0` and no
    // tier is exactly when "did the capture even run?" is the question, and
    // gating these on `cold` would blank them precisely then.
    const sidecar = readSidecarStats();
    if (sidecar && turnStartSidecar) {
      rec.coldSidecarCaptureReached = sidecar.captureReached - turnStartSidecar.captureReached;
      rec.coldSidecarChainEmpty = sidecar.chainEmpty - turnStartSidecar.chainEmpty;
      rec.coldSidecarBoundarySkips = sidecar.boundarySkips - turnStartSidecar.boundarySkips;
      rec.coldSidecarAlreadyPersisted = sidecar.alreadyPersisted - turnStartSidecar.alreadyPersisted;
      rec.coldSidecarEnqueued = sidecar.enqueued - turnStartSidecar.enqueued;
      rec.coldSidecarQueueDrops = sidecar.queueDrops - turnStartSidecar.queueDrops;
      rec.coldSidecarInstalled = sidecar.installed - turnStartSidecar.installed;
      rec.coldSidecarRestoreSuppressed = sidecar.restoreSuppressed - turnStartSidecar.restoreSuppressed;
    }
    metricsTrace.record(rec);
  };

  const streamSimple = makeMlxStreamSimple(
    resolvedHost,
    performanceStatus.record,
    () => rootCacheOwnerId,
    onTurnRecord,
    onTurnStart,
    () => rootSessionFile,
  );
  return {
    name: 'mlx-provider',
    factory: (pi: ExtensionAPI) => {
      pi.registerProvider(MLX_PROVIDER_ID, {
        api: MLX_API,
        baseUrl: MLX_BASE_URL,
        apiKey: MLX_API_KEY,
        streamSimple,
        models: models.map((m) => m.piModel),
      });
      pi.on('session_start', (_event, ctx) => {
        rootCacheOwnerId = ctx.sessionManager.getSessionId();
        // Snapshot the root JSONL path so a turn submitted under this root is
        // correlated to it at completion — even for subagent turns, which have
        // no session file of their own. `getSessionFile` is optional-chained so
        // a minimal test/mock session manager without it still works.
        rootSessionFile = ctx.sessionManager.getSessionFile?.();
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

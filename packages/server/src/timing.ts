/** Wire-safe server timing extensions derived from native performance metrics. */

export interface PerformanceMetricsForUsage {
  ttftMs?: number;
  prefillTokensPerSecond?: number;
  decodeTokensPerSecond?: number;
}

interface TimingUsageExtensions {
  /** Server-extension: native time-to-first-token in milliseconds. */
  time_to_first_token_ms?: number;
  /** Server-extension: prompt-token throughput for the tokens actually prefetched this turn. */
  prefill_tokens_per_second?: number;
  /** Server-extension: generated-token throughput during decode. */
  decode_tokens_per_second?: number;
  /** Server-extension: native/server inference elapsed, excluding HTTP transport and logging overhead. */
  server_inference_elapsed_ms?: number;
  /** Server-extension alias for disambiguating native TTFT from request/HTTP elapsed time. */
  server_time_to_first_token_ms?: number;
  /** Server-extension: handler-start to first native token, including model resolve/load and queue wait. */
  server_total_time_to_first_token_ms?: number;
  /** Server-extension alias for native prefill throughput. */
  server_prefill_tokens_per_second?: number;
  /** Server-extension alias for native decode throughput. */
  server_decode_tokens_per_second?: number;
  /** Server-extension: prompt tokens actually prefetched this turn after cached-prefix reuse. */
  prefill_input_tokens?: number;
  /** Server-extension: prompt tokens skipped because a cached prefix was reused. */
  cached_prefix_tokens?: number;
  /**
   * Server-extension: time spent resolving/loading/aliasing the requested model
   * before registry lookup. Includes both the synchronous lookup AND any time
   * spent driving the load. Excludes time spent blocked behind a peer
   * request's in-flight load — that wait is reported separately via
   * `server_load_wait_ms` so a fast follower request is not mis-attributed
   * a long resolve when it merely inherited a cold-load wait.
   */
  server_model_resolve_ms?: number;
  /**
   * Server-extension: wall-clock time this request spent blocked on the
   * process-wide model-load writer lock. Set when the load was already
   * in flight when this request arrived (a peer drove the load and we
   * waited for it to finish) AND when this request itself drove the
   * load. Inspect `server_load_owner` to disambiguate. When the writer
   * lock was free and the resolve was a no-op (model already loaded),
   * this field is elided.
   */
  server_load_wait_ms?: number;
  /**
   * Server-extension: `true` when this request acquired the model-load
   * writer lock with no contention (i.e. either the model was already
   * loaded and the call was a no-op, or this request itself drove the
   * load). `false` when the request was parked behind a peer's in-flight
   * load. Elided when no load coordinator is wired (single-process
   * tests, embedded callers).
   */
  server_load_owner?: boolean;
  /** Server-extension: time spent waiting behind the per-model execution mutex. */
  server_queue_ms?: number;
  /** Server-extension: handler time before native inference begins, including resolve and queue wait. */
  server_pre_inference_ms?: number;
  /** Server-extension: effective process-level paged-prefill chunk size. */
  server_paged_prefill_chunk_size?: number;
  /** Server-extension: effective process-level paged-prefill eval/clear cadence. */
  server_paged_prefill_eval_interval?: number;
  /** Server-extension: effective process-level paged-decode cache-clear cadence. */
  server_paged_decode_cache_clear_interval?: number;
}

function finitePositive(value: number | undefined): number | undefined {
  return value != null && Number.isFinite(value) && value > 0 ? value : undefined;
}

function finiteNonNegativeInteger(value: number | undefined): number | undefined {
  return value != null && Number.isFinite(value) && value >= 0 ? Math.max(0, Math.floor(value)) : undefined;
}

function finiteNonNegative(value: number | undefined): number | undefined {
  return value != null && Number.isFinite(value) && value >= 0 ? value : undefined;
}

export interface ServerTimingForUsage {
  server_model_resolve_ms?: number;
  server_load_wait_ms?: number;
  server_load_owner?: boolean;
  server_queue_ms?: number;
  server_pre_inference_ms?: number;
  server_paged_prefill_chunk_size?: number;
  server_paged_prefill_eval_interval?: number;
  server_paged_decode_cache_clear_interval?: number;
}

const I32_MAX = 0x7fff_ffff;

function parseI32(value: string | undefined): number | undefined {
  if (value == null) return undefined;
  const trimmed = value.trim();
  if (!/^[+-]?\d+$/.test(trimmed)) return undefined;
  const parsed = Number.parseInt(trimmed, 10);
  return Number.isSafeInteger(parsed) && parsed >= -0x8000_0000 && parsed <= I32_MAX ? parsed : undefined;
}

function parseNonNegativeI32(value: string | undefined, fallback: number): number {
  const parsed = parseI32(value);
  return parsed != null && parsed >= 0 ? parsed : fallback;
}

function parsePositiveI32(value: string | undefined, fallback: number): number {
  const parsed = parseI32(value);
  return parsed != null && parsed > 0 ? parsed : fallback;
}

export function resolveServerTuningForUsage(
  env: Record<string, string | undefined> = process.env,
): Pick<
  ServerTimingForUsage,
  'server_paged_prefill_chunk_size' | 'server_paged_prefill_eval_interval' | 'server_paged_decode_cache_clear_interval'
> {
  return {
    server_paged_prefill_chunk_size: parseNonNegativeI32(env.MLX_PAGED_PREFILL_CHUNK_SIZE, 0),
    server_paged_prefill_eval_interval: parsePositiveI32(env.MLX_PAGED_PREFILL_EVAL_INTERVAL, 8),
    server_paged_decode_cache_clear_interval: parsePositiveI32(env.MLX_PAGED_DECODE_CACHE_CLEAR_INTERVAL, 64),
  };
}

function computeServerInferenceElapsedMs(
  ttftMs: number | undefined,
  decodeTokensPerSecond: number | undefined,
  outputTokens: number | undefined,
): number | undefined {
  if (ttftMs == null) return undefined;

  const generatedTokens = finiteNonNegativeInteger(outputTokens);
  if (generatedTokens == null) {
    return ttftMs;
  }
  if (generatedTokens <= 1) {
    return ttftMs;
  }
  if (decodeTokensPerSecond == null) {
    return undefined;
  }
  return ttftMs + ((generatedTokens - 1) / decodeTokensPerSecond) * 1000;
}

function buildTimingUsageExtensions(
  performance: PerformanceMetricsForUsage | undefined,
  promptTokens: number | undefined,
  outputTokens: number | undefined,
  cachedTokens: number | undefined,
  serverTiming?: ServerTimingForUsage,
): TimingUsageExtensions {
  const ttftMs = finitePositive(performance?.ttftMs);
  const prefillTokensPerSecond = finitePositive(performance?.prefillTokensPerSecond);
  const decodeTokensPerSecond = finitePositive(performance?.decodeTokensPerSecond);
  const serverInferenceElapsedMs = computeServerInferenceElapsedMs(ttftMs, decodeTokensPerSecond, outputTokens);
  const preInferenceMs = finiteNonNegative(serverTiming?.server_pre_inference_ms);

  const extensions: TimingUsageExtensions = {};
  if (ttftMs != null) {
    extensions.time_to_first_token_ms = ttftMs;
    extensions.server_time_to_first_token_ms = ttftMs;
    if (preInferenceMs != null) {
      extensions.server_total_time_to_first_token_ms = preInferenceMs + ttftMs;
    }
  }
  if (prefillTokensPerSecond != null) {
    extensions.prefill_tokens_per_second = prefillTokensPerSecond;
    extensions.server_prefill_tokens_per_second = prefillTokensPerSecond;
  }
  if (decodeTokensPerSecond != null) {
    extensions.decode_tokens_per_second = decodeTokensPerSecond;
    extensions.server_decode_tokens_per_second = decodeTokensPerSecond;
  }
  if (serverInferenceElapsedMs != null && Number.isFinite(serverInferenceElapsedMs) && serverInferenceElapsedMs > 0) {
    extensions.server_inference_elapsed_ms = serverInferenceElapsedMs;
  }

  if (performance != null) {
    const prompt = finiteNonNegativeInteger(promptTokens);
    const cached = finiteNonNegativeInteger(cachedTokens);
    if (prompt != null) {
      const cachedPrefix = cached == null ? 0 : Math.min(cached, prompt);
      extensions.prefill_input_tokens = prompt - cachedPrefix;
      if (cachedPrefix > 0) {
        extensions.cached_prefix_tokens = cachedPrefix;
      }
    }
  }

  const modelResolveMs = finiteNonNegative(serverTiming?.server_model_resolve_ms);
  if (modelResolveMs != null) {
    extensions.server_model_resolve_ms = modelResolveMs;
  }
  const loadWaitMs = finiteNonNegative(serverTiming?.server_load_wait_ms);
  if (loadWaitMs != null) {
    extensions.server_load_wait_ms = loadWaitMs;
  }
  if (typeof serverTiming?.server_load_owner === 'boolean') {
    extensions.server_load_owner = serverTiming.server_load_owner;
  }
  const queueMs = finiteNonNegative(serverTiming?.server_queue_ms);
  if (queueMs != null) {
    extensions.server_queue_ms = queueMs;
  }
  if (preInferenceMs != null) {
    extensions.server_pre_inference_ms = preInferenceMs;
  }
  const pagedPrefillChunkSize = finiteNonNegativeInteger(serverTiming?.server_paged_prefill_chunk_size);
  if (pagedPrefillChunkSize != null) {
    extensions.server_paged_prefill_chunk_size = pagedPrefillChunkSize;
  }
  const pagedPrefillEvalInterval = finiteNonNegativeInteger(serverTiming?.server_paged_prefill_eval_interval);
  if (pagedPrefillEvalInterval != null) {
    extensions.server_paged_prefill_eval_interval = pagedPrefillEvalInterval;
  }
  const pagedDecodeCacheClearInterval = finiteNonNegativeInteger(
    serverTiming?.server_paged_decode_cache_clear_interval,
  );
  if (pagedDecodeCacheClearInterval != null) {
    extensions.server_paged_decode_cache_clear_interval = pagedDecodeCacheClearInterval;
  }

  return extensions;
}

export function mergeTimingUsageExtensions<T extends TimingUsageExtensions>(
  usage: T,
  performance: PerformanceMetricsForUsage | undefined,
  promptTokens: number | undefined,
  outputTokens: number | undefined,
  cachedTokens: number | undefined,
  serverTiming?: ServerTimingForUsage,
): void {
  Object.assign(usage, buildTimingUsageExtensions(performance, promptTokens, outputTokens, cachedTokens, serverTiming));
}

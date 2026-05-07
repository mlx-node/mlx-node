/** Wire-safe server timing extensions derived from native performance metrics. */

export interface PerformanceMetricsForUsage {
  ttftMs?: number;
  prefillTokensPerSecond?: number;
  decodeTokensPerSecond?: number;
}

export interface TimingUsageExtensions {
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
  /** Server-extension alias for native prefill throughput. */
  server_prefill_tokens_per_second?: number;
  /** Server-extension alias for native decode throughput. */
  server_decode_tokens_per_second?: number;
  /** Server-extension: prompt tokens actually prefetched this turn after cached-prefix reuse. */
  prefill_input_tokens?: number;
  /** Server-extension: prompt tokens skipped because a cached prefix was reused. */
  cached_prefix_tokens?: number;
  /** Server-extension: time spent resolving/loading/aliasing the requested model before registry lookup. */
  server_model_resolve_ms?: number;
  /** Server-extension: time spent waiting behind the per-model execution mutex. */
  server_queue_ms?: number;
  /** Server-extension: handler time before native inference begins, including resolve and queue wait. */
  server_pre_inference_ms?: number;
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
  server_queue_ms?: number;
  server_pre_inference_ms?: number;
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

export function buildTimingUsageExtensions(
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

  const extensions: TimingUsageExtensions = {};
  if (ttftMs != null) {
    extensions.time_to_first_token_ms = ttftMs;
    extensions.server_time_to_first_token_ms = ttftMs;
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
  const queueMs = finiteNonNegative(serverTiming?.server_queue_ms);
  if (queueMs != null) {
    extensions.server_queue_ms = queueMs;
  }
  const preInferenceMs = finiteNonNegative(serverTiming?.server_pre_inference_ms);
  if (preInferenceMs != null) {
    extensions.server_pre_inference_ms = preInferenceMs;
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

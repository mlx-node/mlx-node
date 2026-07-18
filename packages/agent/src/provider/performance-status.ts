/**
 * Transient per-message inference telemetry for the interactive agent footer.
 *
 * Pi's `Usage` object drives token accounting and compaction, so native
 * throughput must not be smuggled into it. The exact AssistantMessage object
 * is delivered to extension `message_end` handlers before persistence; a
 * provider-scoped WeakMap therefore carries the metrics to the TUI without
 * changing the conversation schema or retaining completed messages.
 */

import type { AssistantMessage } from '@earendil-works/pi-ai';
import type { ExtensionContext } from '@earendil-works/pi-coding-agent';
import type { PerformanceMetrics } from '@mlx-node/lm';

const STATUS_KEY = 'mlx-performance';

interface ThroughputSample {
  ttftMs?: number;
  prefillTokensPerSecond: number;
  decodeTokensPerSecond: number;
  inputTokens?: number;
  cachedTokens?: number;
}

interface MessageEndLike {
  message: { role: string };
}

function finiteRate(value: number): number | undefined {
  return Number.isFinite(value) && value >= 0 ? value : undefined;
}

function finiteTokenCount(value: number | undefined): number | undefined {
  return value !== undefined && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

function formatRate(value: number): string {
  return value.toLocaleString('en-US', {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
}

function formatDuration(ttftMs: number): string {
  if (ttftMs < 1000) return `${formatRate(ttftMs)} ms`;
  return `${(ttftMs / 1000).toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })} s`;
}

function formatTokenCount(value: number): string {
  if (value < 1000) return value.toLocaleString('en-US');
  const [divisor, suffix] = value < 1_000_000 ? [1000, 'k'] : [1_000_000, 'm'];
  return `${(value / divisor).toLocaleString('en-US', { maximumFractionDigits: 1 })}${suffix}`;
}

function formatContext(sample: ThroughputSample): string {
  let context = '';
  if (sample.ttftMs !== undefined) context += ` · TTFT ${formatDuration(sample.ttftMs)}`;
  if (sample.inputTokens !== undefined && sample.cachedTokens !== undefined) {
    if (sample.cachedTokens > 0) {
      context +=
        ` · input ${formatTokenCount(sample.inputTokens)} new` + ` + ${formatTokenCount(sample.cachedTokens)} cached`;
    } else {
      context += ` · input ${formatTokenCount(sample.inputTokens)}`;
    }
  }
  return context;
}

function formatSample(sample: ThroughputSample): string {
  return (
    `mlx${formatContext(sample)}` +
    ` · prefill ${formatRate(sample.prefillTokensPerSecond)} tok/s` +
    ` · decode ${formatRate(sample.decodeTokensPerSecond)} tok/s`
  );
}

export class PerformanceStatus {
  private readonly byMessage = new WeakMap<AssistantMessage, ThroughputSample>();

  /** Record only complete, displayable samples; malformed native metrics are ignored. */
  readonly record = (message: AssistantMessage, performance: PerformanceMetrics): void => {
    const prefillTokensPerSecond = finiteRate(performance.prefillTokensPerSecond);
    const decodeTokensPerSecond = finiteRate(performance.decodeTokensPerSecond);
    if (prefillTokensPerSecond === undefined || decodeTokensPerSecond === undefined) return;
    const sample: ThroughputSample = { prefillTokensPerSecond, decodeTokensPerSecond };
    const ttftMs = finiteRate(performance.ttftMs);
    if (ttftMs !== undefined) sample.ttftMs = ttftMs;
    const inputTokens = finiteTokenCount(message.usage?.input);
    const cachedTokens = finiteTokenCount(message.usage?.cacheRead);
    if (inputTokens !== undefined && cachedTokens !== undefined) {
      sample.inputTokens = inputTokens;
      sample.cachedTokens = cachedTokens;
    }
    this.byMessage.set(message, sample);
  };

  /** Render the successful mlx inference associated with this exact Pi message. */
  showMessage(event: MessageEndLike, ctx: ExtensionContext): void {
    if (ctx.mode !== 'tui' || event.message.role !== 'assistant') return;
    const sample = this.byMessage.get(event.message as AssistantMessage);
    if (!sample) return;
    ctx.ui.setStatus(STATUS_KEY, ctx.ui.theme.fg('dim', formatSample(sample)));
  }

  /** Prevent the selected model's completed sample lingering after its lifecycle ends. */
  clear(ctx: ExtensionContext): void {
    if (ctx.mode === 'tui') ctx.ui.setStatus(STATUS_KEY, undefined);
  }
}

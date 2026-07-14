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
  prefillTokensPerSecond: number;
  decodeTokensPerSecond: number;
}

interface MessageEndLike {
  message: { role: string };
}

function finiteRate(value: number): number | undefined {
  return Number.isFinite(value) && value >= 0 ? value : undefined;
}

function formatRate(value: number): string {
  return value.toLocaleString('en-US', {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
}

function formatSample(sample: ThroughputSample): string {
  return (
    `mlx · prefill ${formatRate(sample.prefillTokensPerSecond)} tok/s` +
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
    this.byMessage.set(message, { prefillTokensPerSecond, decodeTokensPerSecond });
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

import type { AssistantMessage } from '@earendil-works/pi-ai';
import type { ExtensionContext } from '@earendil-works/pi-coding-agent';
import type { PerformanceMetrics } from '@mlx-node/lm';
import { describe, expect, it } from 'vite-plus/test';

import { PerformanceStatus } from '../src/provider/performance-status.js';

const MESSAGE = { role: 'assistant' } as AssistantMessage;
const METRICS: PerformanceMetrics = {
  ttftMs: 125,
  prefillTokensPerSecond: 1234.56,
  decodeTokensPerSecond: 42.34,
};

function messageEnd(message: { role: string } = MESSAGE): { type: 'message_end'; message: { role: string } } {
  return { type: 'message_end', message };
}

function makeContext(mode: ExtensionContext['mode'] = 'tui'): {
  ctx: ExtensionContext;
  statuses: Array<[string, string | undefined]>;
} {
  const statuses: Array<[string, string | undefined]> = [];
  const ctx = {
    mode,
    ui: {
      theme: {
        fg(color: string, text: string): string {
          return `[${color}]${text}`;
        },
      },
      setStatus(key: string, text: string | undefined): void {
        statuses.push([key, text]);
      },
    },
  } as unknown as ExtensionContext;
  return { ctx, statuses };
}

describe('PerformanceStatus', () => {
  it('renders prefill and decode speed for the exact completed assistant message', () => {
    const status = new PerformanceStatus();
    const { ctx, statuses } = makeContext();
    status.record(MESSAGE, METRICS);

    status.showMessage(messageEnd(), ctx);

    expect(statuses).toEqual([
      ['mlx-performance', '[dim]mlx · prefill 1,234.6 tok/s · decode 42.3 tok/s'],
    ]);
  });

  it('does not show another message\'s metrics or write outside TUI mode', () => {
    const status = new PerformanceStatus();
    status.record(MESSAGE, METRICS);
    const other = { role: 'assistant' } as AssistantMessage;
    const tui = makeContext();
    const print = makeContext('print');

    status.showMessage(messageEnd(other), tui.ctx);
    status.showMessage(messageEnd(), print.ctx);

    expect(tui.statuses).toEqual([]);
    expect(print.statuses).toEqual([]);
  });

  it('keeps a completed assistant sample visible while its tool result starts the next turn', () => {
    const status = new PerformanceStatus();
    const { ctx, statuses } = makeContext();
    status.record(MESSAGE, METRICS);

    status.showMessage(messageEnd(), ctx);
    status.showMessage(messageEnd({ role: 'toolResult' }), ctx);

    expect(statuses).toEqual([
      ['mlx-performance', '[dim]mlx · prefill 1,234.6 tok/s · decode 42.3 tok/s'],
    ]);
  });

  it.each([
    { prefillTokensPerSecond: Number.NaN, decodeTokensPerSecond: 10 },
    { prefillTokensPerSecond: 10, decodeTokensPerSecond: Number.POSITIVE_INFINITY },
    { prefillTokensPerSecond: -1, decodeTokensPerSecond: 10 },
  ])('ignores malformed throughput metrics: %j', (rates) => {
    const status = new PerformanceStatus();
    const { ctx, statuses } = makeContext();
    status.record(MESSAGE, { ...METRICS, ...rates });

    status.showMessage(messageEnd(), ctx);

    expect(statuses).toEqual([]);
  });

  it('clears stale status on model change or shutdown', () => {
    const status = new PerformanceStatus();
    const { ctx, statuses } = makeContext();

    status.clear(ctx);

    expect(statuses).toEqual([['mlx-performance', undefined]]);
  });
});

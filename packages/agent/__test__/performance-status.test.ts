import type { AssistantMessage } from '@earendil-works/pi-ai';
import type { ExtensionContext } from '@earendil-works/pi-coding-agent';
import type { PerformanceMetrics } from '@mlx-node/lm';
import { describe, expect, it } from 'vite-plus/test';

import { PerformanceStatus } from '../src/provider/performance-status.js';

const MESSAGE = {
  role: 'assistant',
  usage: { input: 1234, cacheRead: 0 },
} as AssistantMessage;
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
      ['mlx-performance', '[dim]mlx · TTFT 125.0 ms · input 1.2k · prefill 1,234.6 tok/s · decode 42.3 tok/s'],
    ]);
  });

  it('makes a tiny cached suffix explicit instead of presenting it like bulk prefill', () => {
    const status = new PerformanceStatus();
    const message = {
      role: 'assistant',
      usage: { input: 42, cacheRead: 114556 },
    } as AssistantMessage;
    const { ctx, statuses } = makeContext();
    status.record(message, {
      ttftMs: 561.677,
      prefillTokensPerSecond: 74.776,
      decodeTokensPerSecond: 42.514,
    });

    status.showMessage(messageEnd(message), ctx);

    expect(statuses).toEqual([
      [
        'mlx-performance',
        '[dim]mlx · TTFT 561.7 ms · input 42 new + 114.6k cached · prefill 74.8 tok/s · decode 42.5 tok/s',
      ],
    ]);
  });

  it('keeps bulk cached-prefill throughput in context', () => {
    const status = new PerformanceStatus();
    const message = {
      role: 'assistant',
      usage: { input: 4460, cacheRead: 114640 },
    } as AssistantMessage;
    const { ctx, statuses } = makeContext();
    status.record(message, {
      ttftMs: 5330.159,
      prefillTokensPerSecond: 836.748,
      decodeTokensPerSecond: 42.728,
    });

    status.showMessage(messageEnd(message), ctx);

    expect(statuses).toEqual([
      [
        'mlx-performance',
        '[dim]mlx · TTFT 5.33 s · input 4.5k new + 114.6k cached · prefill 836.7 tok/s · decode 42.7 tok/s',
      ],
    ]);
  });

  it("does not show another message's metrics or write outside TUI mode", () => {
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
      ['mlx-performance', '[dim]mlx · TTFT 125.0 ms · input 1.2k · prefill 1,234.6 tok/s · decode 42.3 tok/s'],
    ]);
  });

  it('falls back to valid throughput when usage or TTFT metadata is malformed', () => {
    const status = new PerformanceStatus();
    const message = {
      role: 'assistant',
      usage: { input: Number.NaN, cacheRead: -1 },
    } as AssistantMessage;
    const { ctx, statuses } = makeContext();
    status.record(message, { ...METRICS, ttftMs: Number.POSITIVE_INFINITY });

    status.showMessage(messageEnd(message), ctx);

    expect(statuses).toEqual([['mlx-performance', '[dim]mlx · prefill 1,234.6 tok/s · decode 42.3 tok/s']]);
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

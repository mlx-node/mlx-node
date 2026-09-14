/** @vitest-environment happy-dom */

import type { DelegationSummary, SessionRow } from '@/lib/types';
import SessionDetail from '@/pages/session-detail';
import Sessions from '@/pages/sessions';
import { act, createElement } from 'react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { renderPage, stubApi, type RenderedPage } from './render.js';

const positive: DelegationSummary = {
  status: 'complete',
  tokenizer: 'o200k_base',
  evidenceTokens: 4507,
  handoffTokens: 875,
  savedTokens: 3632,
  savingsRatio: 3632 / 4507,
};
const negative: DelegationSummary = {
  status: 'complete',
  tokenizer: 'o200k_base',
  evidenceTokens: 116,
  handoffTokens: 256,
  savedTokens: -140,
  savingsRatio: -140 / 116,
};
const row = (id: string, delegation?: DelegationSummary): SessionRow => ({
  id,
  path: `/sessions/${id}.jsonl`,
  cwd: '/work',
  name: `Task ${id}`,
  created: Date.now(),
  modified: Date.now(),
  messageCount: 4,
  firstMessage: null,
  models: ['local-model'],
  inputTokens: 1000,
  outputTokens: 1000,
  delegation,
});
let page: RenderedPage | undefined;
let dispose: (() => void) | undefined;
afterEach(() => {
  page?.unmount();
  dispose?.();
  vi.restoreAllMocks();
});

describe('delegate session UI', () => {
  it('shows badges and positive, negative, zero and unavailable estimates without labeling ordinary sessions', async () => {
    dispose = stubApi({
      '/sessions': {
        sessions: [
          row('review', positive),
          row('verify', negative),
          row('zero', { ...positive, handoffTokens: 4507, savedTokens: 0, savingsRatio: 0 }),
          row('pending', { status: 'incomplete', reason: 'no-final-handoff' }),
          row('ordinary'),
        ],
        total: 5,
        tokens: 10000,
        cwds: ['/work'],
      },
    });
    page = await renderPage(createElement(MemoryRouter, null, createElement(Sessions)), (text) =>
      text.includes('Task ordinary'),
    );
    const rows = [...page.container.querySelectorAll('tbody tr')];
    expect(rows[0]!.textContent).toContain('Delegate');
    expect(rows[0]!.textContent).toContain('3,632 tokens saved');
    expect(rows[1]!.textContent).toContain('140 extra tokens');
    expect(rows[2]!.textContent).toContain('0 tokens saved');
    expect(rows[3]!.textContent).toContain('Savings unavailable');
    expect(rows[4]!.textContent).not.toContain('Delegate');
    expect(rows[4]!.textContent).not.toContain('tokens saved');

    const writeText = vi.spyOn(navigator.clipboard, 'writeText').mockResolvedValue();
    const copy = rows[0]!.querySelector('button[aria-label="Copy resume command"]')!;
    await act(async () => {
      (copy as HTMLButtonElement).click();
    });
    expect(writeText).toHaveBeenCalledWith("mlx agent --session '/sessions/review.jsonl'");
  });

  it('shows the calculation and measurement limits on session detail, including negative ratios', async () => {
    dispose = stubApi({
      '/sessions/verify': { session: row('verify', negative), transcript: [] },
      '/sessions/verify/metrics': { sessionId: 'verify', turns: [], traces: [] },
    });
    page = await renderPage(
      createElement(
        MemoryRouter,
        { initialEntries: ['/sessions/verify'] },
        createElement(
          Routes,
          null,
          createElement(Route, { path: '/sessions/:id', element: createElement(SessionDetail) }),
        ),
      ),
      (text) => text.includes('Extra tokens returned'),
    );
    expect(page.text()).toContain('Estimated');
    expect(page.text()).toContain('120.7% larger summary');
    expect(page.text()).toContain('Evidence read locally116');
    expect(page.text()).toContain('Summary returned256');
    expect(page.text()).toContain('transcript rereads are not measured');
  });
});

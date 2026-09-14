/** @vitest-environment happy-dom */
import CodingAgents from '@/pages/coding-agents';
import { createElement, act } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, expect, it, vi } from 'vite-plus/test';

import type { CodingAgentsState } from '../src/coding-agents.js';
import { renderPage, stubApi, type RenderedPage } from './render.js';

let page: RenderedPage | undefined;
let dispose: (() => void) | undefined;
afterEach(() => {
  page?.unmount();
  dispose?.();
  page = undefined;
  dispose = undefined;
});

function fixture(available: boolean): CodingAgentsState {
  return {
    available,
    command: available ? '/test/.mlx-node/bin/mlx' : null,
    model: available ? 'qwen3-local' : null,
    unavailableReason: available ? null : 'Install a local model first to check and set up coding agents.',
    agents: [
      {
        id: 'claude',
        name: 'Claude Code',
        path: '/test/.claude/CLAUDE.md',
        status: available ? 'installed' : 'unchecked',
        detail: null,
        checkedAt: null,
      },
      {
        id: 'codex',
        name: 'Codex',
        path: '/test/.codex/AGENTS.md',
        status: available ? 'not-installed' : 'unchecked',
        detail: null,
        checkedAt: null,
      },
      {
        id: 'grok',
        name: 'Grok',
        path: '/test/.grok/AGENTS.md',
        status: available ? 'error' : 'unchecked',
        detail: available ? 'The local model returned an unclear result.' : null,
        checkedAt: null,
      },
    ],
  };
}

it('disables every setup action and links to models when no local model is installed', async () => {
  dispose = stubApi({ '/coding-agents': fixture(false) });
  page = await renderPage(createElement(MemoryRouter, null, createElement(CodingAgents)), (text) =>
    text.includes('A local model is required'),
  );
  expect(page.text()).toContain('Install a local model first');
  expect([...page.container.querySelectorAll('button')].every((button) => button.disabled)).toBe(true);
  expect(page.container.querySelector('a')?.getAttribute('href')).toBe('/welcome');
  expect(page.text()).not.toContain('Installed');
});

it('renders installed, installable and failed checks as distinct states', async () => {
  dispose = stubApi({ '/coding-agents': fixture(true) });
  page = await renderPage(createElement(MemoryRouter, null, createElement(CodingAgents)), (text) =>
    text.includes('Check status uses'),
  );
  expect(page.text()).toContain('Installed');
  expect(page.text()).toContain('Install…');
  expect(page.text()).toContain('Check again');
  expect(page.container.querySelector('[role="alert"]')?.textContent).toContain('unclear result');
  const install = [...page.container.querySelectorAll('button')].find((button) => button.textContent === 'Install…')!;
  expect(install.disabled).toBe(false);
});

it('offers command repair rather than a model download when the launcher is unavailable', async () => {
  const state = fixture(false);
  state.model = 'qwen3-local';
  state.unavailableReason = 'The app command is unavailable.';
  dispose = stubApi({ '/coding-agents': state });
  page = await renderPage(createElement(MemoryRouter, null, createElement(CodingAgents)), (text) =>
    text.includes('Retry setup'),
  );
  expect(page.text()).toContain('Command setup needs attention');
  expect(page.text()).not.toContain('Install a model');
  expect(page.text()).not.toContain('Installed');
  const enabled = [...page.container.querySelectorAll('button')].filter((button) => !button.disabled);
  expect(enabled.map((button) => button.textContent)).toEqual(['Retry setup']);
});

it('updates an older prompt through the install action', async () => {
  const state = fixture(true);
  state.agents[0].status = 'needs-update';
  const call = vi.fn();
  dispose = stubApi({ '/coding-agents': state }, { onCall: call });
  page = await renderPage(createElement(MemoryRouter, null, createElement(CodingAgents)), (text) =>
    text.includes('Update…'),
  );
  const update = [...page.container.querySelectorAll('button')].find((button) => button.textContent === 'Update…')!;
  await act(async () => {
    update.click();
  });
  expect(
    call.mock.calls.some(([request]) => request.body?.action === 'install' && request.body?.agent === 'claude'),
  ).toBe(true);
});

it('does not automatically retry a failed model check on each render', async () => {
  const state = fixture(true);
  state.agents.forEach((row) => {
    row.status = 'error';
  });
  const call = vi.fn();
  dispose = stubApi({ '/coding-agents': state }, { onCall: call });
  page = await renderPage(createElement(MemoryRouter, null, createElement(CodingAgents)), (text) =>
    text.includes('Check status uses'),
  );
  await act(async () => {});
  expect(call.mock.calls.every(([request]) => request.method === 'GET' || request.body?.action === 'refresh')).toBe(
    true,
  );
});

it('never starts model checks when opening a page with unchecked files', async () => {
  const state = fixture(true);
  state.agents.forEach((row) => {
    row.status = 'unchecked';
  });
  const call = vi.fn();
  dispose = stubApi({ '/coding-agents': state }, { onCall: call });
  page = await renderPage(createElement(MemoryRouter, null, createElement(CodingAgents)), (text) =>
    text.includes('Opening this page uses cached results'),
  );
  await act(async () => {});
  expect(call.mock.calls.some(([request]) => request.body?.action === 'refresh')).toBe(true);
  expect(call.mock.calls.some(([request]) => request.body?.action === 'detect')).toBe(false);
});

it('distinguishes the active check from waiting checks', async () => {
  const state = fixture(true);
  state.agents[0].status = 'checking';
  state.agents[1].status = 'waiting';
  state.agents[2].status = 'waiting';
  dispose = stubApi({ '/coding-agents': state });
  page = await renderPage(createElement(MemoryRouter, null, createElement(CodingAgents)), (text) =>
    text.includes('Waiting…'),
  );
  expect(page.text().match(/Checking…/g)).toHaveLength(1);
  expect(page.text().match(/Waiting…/g)).toHaveLength(2);
});

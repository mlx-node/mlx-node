import { readdirSync, readFileSync } from 'node:fs';
import { join, relative } from 'node:path';

/** @vitest-environment happy-dom */
import CodingAgents from '@/pages/coding-agents';
import { createElement, act } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, expect, it, onTestFinished, vi } from 'vite-plus/test';

import type { CodingAgentsState } from '../src/coding-agents.js';
import { byRole, deferred, renderPage, stubApi, type RenderedPage, waitFor } from './render.js';

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
    block: available ? '<!-- mlx-node:begin -->\nUse /test/.mlx-node/bin/mlx\n<!-- mlx-node:end -->' : null,
    unavailableReason: available ? null : 'Install a local model first to set up coding agents.',
    agents: [
      {
        id: 'claude',
        name: 'Claude Code',
        path: '/test/.claude/CLAUDE.md',
        status: available ? 'installed' : 'unchecked',
        detail: null,
      },
      {
        id: 'codex',
        name: 'Codex',
        path: '/test/.codex/AGENTS.md',
        status: available ? 'not-installed' : 'unchecked',
        detail: null,
      },
      {
        id: 'grok',
        name: 'Grok',
        path: '/test/.grok/AGENTS.md',
        status: available ? 'error' : 'unchecked',
        detail: available ? 'The portato managed instructions markers in this file are incomplete or repeated.' : null,
      },
    ],
  };
}

const agentButtons = (page: RenderedPage) => [...page.container.querySelectorAll('[aria-live] button')];
const buttonText = (page: RenderedPage, text: string) =>
  [...page.container.querySelectorAll('button')].find((button) => button.textContent?.trim() === text)!;
const ready = (text: string) => text.includes('Update replaces only that block');

it('disables agent actions and links to models when no local model is installed', async () => {
  dispose = stubApi({ '/coding-agents': fixture(false) });
  page = await renderPage(createElement(MemoryRouter, null, createElement(CodingAgents)), (text) =>
    text.includes('A local model is required'),
  );
  expect(page.text()).toContain('Install a local model first');
  expect(agentButtons(page).every((button) => (button as HTMLButtonElement).disabled)).toBe(true);
  expect(page.container.querySelector('a')?.getAttribute('href')).toBe('/welcome');
  expect(page.text()).not.toContain('Installed');
});

it('renders installed, installable and failed files as distinct states', async () => {
  dispose = stubApi({ '/coding-agents': fixture(true) });
  page = await renderPage(createElement(MemoryRouter, null, createElement(CodingAgents)), ready);
  expect(page.text()).toContain('Installed');
  expect(agentButtons(page).map((button) => button.textContent)).toEqual(['Install', 'Check again']);
  expect(page.container.querySelector('[role="alert"]')?.textContent).toContain('markers');
  expect(buttonText(page, 'Install').disabled).toBe(false);
  expect(page.text()).not.toContain('Recheck');
});

it('shows each instruction file under the home folder as ~/…, the full path on hover', async () => {
  dispose = stubApi({
    '/coding-agents': fixture(true),
    '/health': { status: 'ok', modelsDir: '', tracesDir: '', runtimeDir: '', homeDir: '/test' },
  });
  page = await renderPage(createElement(MemoryRouter, null, createElement(CodingAgents)), ready);
  await waitFor(() => page!.text().includes('Instructions in ~/.claude/CLAUDE.md.'));
  expect(page.container.querySelector('[title="/test/.codex/AGENTS.md"]')?.textContent).toBe('~/.codex/AGENTS.md');
  expect(page.text()).not.toContain('/test/.claude/CLAUDE.md');
});

it('shows the Setup command under the home folder as ~/…, copying the full path', async () => {
  const original = Object.getOwnPropertyDescriptor(navigator, 'clipboard');
  const writeText = vi.fn(() => Promise.resolve());
  Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });
  onTestFinished(() => {
    if (original === undefined) Reflect.deleteProperty(navigator, 'clipboard');
    else Object.defineProperty(navigator, 'clipboard', original);
  });
  dispose = stubApi({
    '/coding-agents': fixture(true),
    '/health': { status: 'ok', modelsDir: '', tracesDir: '', runtimeDir: '', homeDir: '/test' },
  });
  page = await renderPage(createElement(MemoryRouter, null, createElement(CodingAgents)), ready);
  const setup = byRole(page.container, 'region', 'Setup')!;
  await waitFor(() => setup.querySelector('[title="/test/.mlx-node/bin/mlx"]')?.textContent === '~/.mlx-node/bin/mlx');
  expect(setup.textContent).not.toContain('/test/.mlx-node/bin/mlx');
  await act(async () => byRole(setup, 'button', 'Copy Command')?.click());
  expect(writeText).toHaveBeenCalledWith('/test/.mlx-node/bin/mlx');
});

it('offers command repair rather than a model download when the launcher is unavailable', async () => {
  const state = fixture(false);
  state.model = 'qwen3-local';
  state.unavailableReason = 'The app command is unavailable.';
  const call = vi.fn();
  dispose = stubApi({ '/coding-agents': state }, { onCall: call });
  page = await renderPage(createElement(MemoryRouter, null, createElement(CodingAgents)), (text) =>
    text.includes('Retry setup'),
  );
  expect(page.text()).toContain('Command setup needs attention');
  expect(page.text()).not.toContain('Install a model');
  expect(agentButtons(page).every((button) => (button as HTMLButtonElement).disabled)).toBe(true);
  await act(async () => buttonText(page!, 'Retry setup').click());
  expect(call.mock.calls.some(([request]) => request.body?.action === 'refresh')).toBe(true);
});

it('updates an outdated block through the install action', async () => {
  const state = fixture(true);
  state.agents[0].status = 'needs-update';
  const call = vi.fn();
  dispose = stubApi({ '/coding-agents': state }, { onCall: call });
  page = await renderPage(createElement(MemoryRouter, null, createElement(CodingAgents)), (text) =>
    text.includes('Update'),
  );
  await act(async () => buttonText(page!, 'Update').click());
  expect(
    call.mock.calls.some(([request]) => request.body?.action === 'install' && request.body?.agent === 'claude'),
  ).toBe(true);
});

it('only reads state on open and refreshes when Check again or Check status is pressed', async () => {
  const call = vi.fn();
  dispose = stubApi({ '/coding-agents': fixture(true) }, { onCall: call });
  page = await renderPage(createElement(MemoryRouter, null, createElement(CodingAgents)), ready);
  await act(async () => {});
  expect(call.mock.calls.every(([request]) => request.method === 'GET')).toBe(true);
  for (const label of ['Check again', 'Check status']) {
    await vi.waitFor(() => expect(buttonText(page!, label).disabled).toBe(false));
    call.mockClear();
    await act(async () => buttonText(page!, label).click());
    await vi.waitFor(() =>
      expect(call.mock.calls.map(([request]) => request.body?.action).filter(Boolean)).toEqual(['refresh']),
    );
  }
});

it('reads state again when the window regains focus', async () => {
  const call = vi.fn();
  dispose = stubApi({ '/coding-agents': fixture(true) }, { onCall: call });
  page = await renderPage(createElement(MemoryRouter, null, createElement(CodingAgents)), ready);
  call.mockClear();
  await act(async () => {
    window.dispatchEvent(new Event('focus'));
  });
  await vi.waitFor(() => expect(call.mock.calls.some(([request]) => request.method === 'GET')).toBe(true));
});

it('shows the model and command the agents will use', async () => {
  dispose = stubApi({ '/coding-agents': fixture(true) });
  page = await renderPage(createElement(MemoryRouter, null, createElement(CodingAgents)), ready);
  const setup = byRole(page.container, 'region', 'Setup');
  expect(setup?.textContent).toContain('qwen3-local');
  expect(setup?.textContent).toContain('/test/.mlx-node/bin/mlx');
});

it('only the clicked row shows its working state', async () => {
  const routes: Record<string, unknown> = { '/coding-agents': fixture(true) };
  dispose = stubApi(routes);
  page = await renderPage(createElement(MemoryRouter, null, createElement(CodingAgents)), ready);
  const gate = deferred(fixture(true));
  routes['/coding-agents'] = gate.body;
  await act(async () => buttonText(page!, 'Install').click());
  await vi.waitFor(() => expect(page!.text()).toContain('Installing…'));
  const cells = [...page.container.querySelectorAll('[aria-live]')];
  const working = cells.find((cell) => cell.textContent?.includes('Installing…'));
  expect(working?.querySelector('button')).toBeNull();
  const others = cells.filter((cell) => cell !== working).flatMap((cell) => [...cell.querySelectorAll('button')]);
  expect(others.map((button) => button.textContent)).toEqual(['Check again']);
  expect(others.every((button) => button.disabled)).toBe(true);
  await act(async () => gate.release());
});

it('its status re-read is the one timer in the SPA, and the docs say so', () => {
  const ui = join(import.meta.dirname, '../ui/src');
  const vendored = join(ui, 'components/arc');
  const timers: string[] = [];
  const walk = (dir: string): void => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name);
      if (entry.isDirectory()) {
        if (path !== vendored) walk(path);
      } else if (path.endsWith('.ts') || path.endsWith('.tsx')) {
        if (readFileSync(path, 'utf8').includes('setInterval(')) timers.push(relative(ui, path));
      }
    }
  };
  walk(ui);
  expect(timers).toEqual(['components/arc-x/load-card/load-card.tsx', 'pages/coding-agents.tsx']);
  const card = readFileSync(join(ui, 'components/arc-x/load-card/load-card.tsx'), 'utf8');
  for (const reader of ['lib/api', 'lib/use-api', 'fetch(']) expect(card).not.toContain(reader);
  for (const doc of ['../../../docs/dashboard.md', '../README.md']) {
    const text = readFileSync(join(import.meta.dirname, doc), 'utf8');
    expect(text).not.toContain('never polls');
    expect(text).toContain('Coding Agents');
    expect(text).toContain('30 s');
  }
});

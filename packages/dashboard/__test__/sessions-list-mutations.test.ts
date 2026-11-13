/**
 * @vitest-environment happy-dom
 */

/**
 * A rename or delete on the Sessions list is applied to the rows the moment the runtime confirms it, not when the
 * list refetch lands: a stale row would keep a live Delete whose second click answers "not found".
 */

import type { SessionListResponse, SessionRow } from '@/lib/types';
import Sessions, { withoutSessions, withTitle } from '@/pages/sessions';
import { act, createElement } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, expect, it } from 'vite-plus/test';

import { deferred, renderPage, type RenderedPage, sequence, stubApi } from './render.js';

const PARENT_ID = '019a0000-0000-7000-8000-000000000001';
const OTHER_ID = '019a0000-0000-7000-8000-000000000002';

function row(overrides: Partial<SessionRow>): SessionRow {
  return {
    id: PARENT_ID,
    origin: 'agent',
    title: 'Refactor the planner',
    cwd: '/work/mlx-node',
    model: 'qwen3.8-27b',
    branch: 'main',
    createdAt: '2026-10-20T09:30:00.000Z',
    lastActiveAt: '2026-10-20T11:00:00.000Z',
    messageCount: 42,
    live: false,
    kind: null,
    parentId: null,
    subagentCount: 0,
    tokens: { input: 1_000, output: 2_000, cached: 500 },
    ...overrides,
  };
}

function listBody(): SessionListResponse {
  return {
    sessions: [row({}), row({ id: OTHER_ID, title: 'Second one' })],
    total: 2,
    cwds: ['/work/mlx-node'],
  };
}

let mounted: RenderedPage | undefined;
let restore: (() => void) | undefined;

afterEach(() => {
  mounted?.unmount();
  mounted = undefined;
  restore?.();
  restore = undefined;
});

async function mountList(routes: Record<string, unknown>): Promise<void> {
  restore = stubApi(routes);
  mounted = await renderPage(
    createElement(MemoryRouter, { initialEntries: ['/sessions'] }, createElement(Sessions)),
    (text) => text.includes('Second one'),
  );
}

async function flush(): Promise<void> {
  for (let i = 0; i < 10; i++) {
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }
}

function rowOf(id: string): HTMLElement | null {
  return mounted!.container.querySelector<HTMLElement>(`tr[data-session-id="${id}"]`);
}

function dialogButton(text: string): HTMLButtonElement {
  const dialog = document.querySelector<HTMLElement>('[role="dialog"]')!;
  const button = [...dialog.querySelectorAll('button')].find((b) => b.textContent?.trim() === text);
  expect(button, text).toBeTruthy();
  return button!;
}

it('drops a deleted row at once, while the list refetch is still loading', async () => {
  const held = deferred({ ...listBody(), sessions: [listBody().sessions[1]], total: 1 });
  await mountList({
    '/sessions': sequence(listBody(), held.body),
    [`/sessions/${PARENT_ID}`]: { deleted: [PARENT_ID] },
  });
  await act(async () => rowOf(PARENT_ID)!.querySelector<HTMLButtonElement>('button[aria-label="Delete"]')!.click());
  await act(async () => dialogButton('Delete').click());
  await flush();
  expect(rowOf(PARENT_ID)).toBeNull();
  expect(rowOf(OTHER_ID)).not.toBeNull();
  held.release();
  await flush();
  expect(rowOf(PARENT_ID)).toBeNull();
});

it('shows a saved title at once, while the list refetch is still loading', async () => {
  const held = deferred({ ...listBody(), sessions: [row({ title: 'New name' }), listBody().sessions[1]] });
  await mountList({
    '/sessions': sequence(listBody(), held.body),
    [`/sessions/${PARENT_ID}`]: { id: PARENT_ID, title: 'New name' },
  });
  await act(async () => rowOf(PARENT_ID)!.querySelector<HTMLButtonElement>('button[aria-label="Rename"]')!.click());
  const input = document.querySelector<HTMLInputElement>('[role="dialog"] input')!;
  await act(async () => {
    Reflect.set(HTMLInputElement.prototype, 'value', 'New name', input);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await act(async () => dialogButton('Save').click());
  await flush();
  expect(rowOf(PARENT_ID)?.textContent).toContain('New name');
  held.release();
});

it('counts only the listed rows a delete removed', () => {
  expect(withoutSessions(listBody(), [PARENT_ID, 'not-listed'])).toMatchObject({
    total: 1,
    sessions: [{ id: OTHER_ID }],
  });
  expect(withTitle(listBody(), OTHER_ID, 'x').sessions.map((entry) => entry.title)).toEqual([
    'Refactor the planner',
    'x',
  ]);
});

/** @vitest-environment happy-dom */

import { act, createElement } from 'react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { CHAT } from '../ui/src/lib/chat-strings.js';
import Chat from '../ui/src/pages/chat.js';
import { forgetPlace } from '../ui/src/pages/chat/drafts.js';
import { chatHostSnapshot } from '../ui/src/pages/chat/use-chat-host.js';
import { answer, end, user } from './helpers/chat-thread-fixtures.js';
import { byRole, READY_CHAT_HOST, renderPage, type RenderedPage, stubApi, type StubbedApi, waitFor } from './render.js';

// Happy DOM lays nothing out, so the scroll state is held here: the thread is scrolled away until Jump is pressed.
vi.mock('../ui/src/lib/stick-to-bottom.js', async (importOriginal) => {
  const real = await importOriginal<typeof import('../ui/src/lib/stick-to-bottom.js')>();
  const { useState } = await import('react');
  return {
    ...real,
    useStickToBottom: () => {
      const [away, setAway] = useState(true);
      return {
        contentRef: () => undefined,
        pinned: !away,
        fresh: false,
        showJump: away,
        jump: () => setAway(false),
        pin: () => setAway(false),
      };
    },
  };
});

let page: RenderedPage | undefined;
let api: StubbedApi | undefined;

afterEach(() => {
  page?.unmount();
  api?.();
  page = undefined;
  api = undefined;
  forgetPlace('p-jump');
});

describe('Chat page: jump to latest', () => {
  it('pressing Jump to latest from the keyboard leaves focus in the composer, not on the page', async () => {
    const now = Date.now();
    api = stubApi(
      { '/chat/conversations': { conversations: [] } },
      {
        chat: {
          host: READY_CHAT_HOST,
          conversations: {
            'p-jump': {
              type: 'sync',
              seq: 0,
              conversation: {
                id: 'p-jump',
                mode: 'chat',
                title: 'Jump',
                cwd: null,
                modelId: 'qwen3.5-4b',
                createdAt: now,
                updatedAt: now,
                hot: false,
                permissionMode: 'ask',
              },
              entries: [user('Hello'), answer('Hi there'), end()],
              turns: [],
              turn: null,
              interactions: [],
              notices: [],
              transcriptError: null,
              permissionMarks: [],
            },
          },
        },
      },
    );
    page = await renderPage(
      createElement(
        MemoryRouter,
        { initialEntries: ['/chat?c=p-jump'] },
        createElement(Routes, null, createElement(Route, { path: '/chat', element: createElement(Chat) })),
      ),
      (text) => text.includes('Hi there'),
    );
    const want = JSON.stringify(READY_CHAT_HOST);
    await waitFor(() => JSON.stringify(chatHostSnapshot().state) === want, 2_000);
    const composer = byRole(page.container, 'textbox', CHAT.composerLabel)!;
    await waitFor(() => document.activeElement === composer);
    const jump = byRole(document.body, 'button', CHAT.jump)!;
    act(() => jump.focus());
    expect(document.activeElement).toBe(jump);
    act(() => jump.click());
    expect(byRole(document.body, 'button', CHAT.jump)).toBeNull();
    await waitFor(() => document.activeElement === composer);
  });
});

/**
 * @vitest-environment happy-dom
 */

/**
 * The tiles of a picture read: nothing is asked for while the row is closed; the tiles are named; a 404 is asked again
 * once after 1 s, then "no longer on disk" with Try again; 413 and 415 have their own sentences; the blob URLs live as
 * long as the conversation's scope, not the row, and are revoked when it closes.
 */

import { act, createElement, type ReactElement } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vite-plus/test';

import type { ApiCall } from '../src/runtime.js';
import {
  PICTURE_RETRY_MS,
  PictureScope,
  PictureStrip,
  picturesDetail,
  type ToolPicture,
} from '../ui/src/components/arc-x/picture-strip/picture-strip.js';
import { ToolCall } from '../ui/src/components/arc-x/tool-call/tool-call.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import { mount, rerender, unmountAll } from './helpers/mount.js';
import { accessibleName, allByRole, byRole, sequence, stubApi, stubError, waitFor, type StubbedApi } from './render.js';

const P = CHAT.pictures;
const SESSION = 'sess-1';
const PNG = Buffer.from('png').toString('base64');

const PDF: ToolPicture[] = [4, 5, 6].map((page, n) => ({ n, mediaType: 'image/jpeg', bytes: 400_000, page }));
const ONE: ToolPicture[] = [{ n: 0, mediaType: 'image/png', bytes: 412 * 1024 }];

const path = (call: string, n: number) => `/sessions/${SESSION}/pictures/${call}/${n}`;

let api: StubbedApi | null = null;
let calls: string[] = [];
let made = 0;
let revoked: string[] = [];

beforeEach(() => {
  calls = [];
  made = 0;
  revoked = [];
  vi.spyOn(URL, 'createObjectURL').mockImplementation(() => `blob:picture-${++made}`);
  vi.spyOn(URL, 'revokeObjectURL').mockImplementation((url: string) => {
    revoked.push(url);
  });
});

afterEach(async () => {
  unmountAll();
  vi.restoreAllMocks();
  if (api !== null) {
    api();
    await api.closed;
    api = null;
  }
});

function serve(routes: Record<string, unknown>): void {
  api = stubApi(routes, {
    onCall: (call: ApiCall) => {
      if (call.path.includes('/pictures/')) calls.push(call.path);
    },
  });
}

/** A read row: closed until clicked, the strip its body, under one conversation's scope. */
function row(call: string, pictures: ToolPicture[], session: string | null = SESSION): ReactElement {
  return createElement(PictureScope, {
    session,
    children: createElement(ToolCall, {
      name: 'Read',
      title: 'docs/spec.pdf',
      status: 'done',
      detail: picturesDetail(pictures, pictures[0]?.page === undefined ? undefined : 12),
      body: createElement(PictureStrip, { call, pictures, path: 'docs/spec.pdf' }),
    }),
  });
}

function openRow(container: HTMLElement): void {
  const button = container.querySelector<HTMLButtonElement>('button[aria-expanded]')!;
  act(() => button.click());
}

describe('picturesDetail', () => {
  it('counts a picture, pictures and the pages of a PDF, with their size', () => {
    expect(picturesDetail(ONE, undefined)).toBe(`${P.one} · 412 KB`);
    expect(picturesDetail([...ONE, { ...ONE[0]!, n: 1 }], undefined)).toBe(`${fill(P.many, { n: 2 })} · 824 KB`);
    expect(picturesDetail(PDF, 12)).toBe(`${fill(P.pages, { n: 3, total: 12 })} · 1.1 MB`);
  });
});

describe('PictureStrip', () => {
  it('asks for nothing while the row is closed; opened, reads each picture once and names the tiles', async () => {
    serve({
      [path('call_pdf', 0)]: { mediaType: 'image/jpeg', data: PNG },
      [path('call_pdf', 1)]: { mediaType: 'image/jpeg', data: PNG },
      [path('call_pdf', 2)]: { mediaType: 'image/jpeg', data: PNG },
    });
    const container = mount(row('call_pdf', PDF));
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 20));
    });
    expect(calls).toEqual([]);
    openRow(container);
    const list = byRole(container, 'list', P.label)!;
    expect(list).not.toBeNull();
    await waitFor(() => allByRole(container, 'button').filter((b) => b.closest('ul') !== null).length === 3);
    expect(allByRole(list, 'button').map((b) => accessibleName(b))).toEqual(
      [4, 5, 6].map((page) => fill(P.openPage, { page })),
    );
    expect(list.textContent).toBe([4, 5, 6].map((page) => fill(P.page, { page })).join(''));
    expect(calls).toEqual([0, 1, 2].map((n) => `/api${path('call_pdf', n)}`));
    expect(container.querySelector('img')!.getAttribute('src')).toBe('blob:picture-1');
  });

  it('a single picture: "Open picture 1 of 1"; while loading a skeleton that says so', async () => {
    serve({ [path('call_png', 0)]: { mediaType: 'image/png', data: PNG } });
    const container = mount(row('call_png', ONE));
    openRow(container);
    expect(container.querySelector('[data-state="loading"]')!.textContent).toBe(P.loading);
    await waitFor(() => byRole(container, 'button', fill(P.open, { n: 1, total: 1 })) !== null);
  });

  it(`a 404 is asked again once after ${PICTURE_RETRY_MS} ms, then "no longer on disk"; Try again reads it`, async () => {
    serve({
      [path('call_png', 0)]: sequence(
        stubError('E_NOT_FOUND', 'Picture not found'),
        stubError('E_NOT_FOUND', 'Picture not found'),
        { mediaType: 'image/png', data: PNG },
      ),
    });
    const container = mount(row('call_png', ONE));
    openRow(container);
    await waitFor(() => calls.length === 1);
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, PICTURE_RETRY_MS / 2));
    });
    expect(calls).toHaveLength(1);
    await waitFor(() => container.textContent!.includes(P.gone), PICTURE_RETRY_MS * 2);
    expect(calls).toHaveLength(2);
    expect(container.querySelector('[data-state="failed"]')).not.toBeNull();
    act(() => byRole(container, 'button', P.again)!.click());
    await waitFor(() => byRole(container, 'button', fill(P.open, { n: 1, total: 1 })) !== null);
    expect(calls).toHaveLength(3);
    expect(container.textContent).not.toContain(P.gone);
  });

  it('413 says the picture is too large, 415 that its kind is not shown; neither is asked again', async () => {
    serve({
      [path('call_two', 0)]: stubError('E_PAYLOAD_TOO_LARGE', 'too large'),
      [path('call_two', 1)]: stubError('E_UNSUPPORTED_MEDIA_TYPE', 'not served'),
    });
    const two: ToolPicture[] = [
      { n: 0, mediaType: 'image/png', bytes: 5_000_000 },
      { n: 1, mediaType: 'image/svg+xml', bytes: 10 },
    ];
    const container = mount(row('call_two', two));
    openRow(container);
    await waitFor(() => container.textContent!.includes(P.notShown));
    expect(container.textContent).toContain(P.tooBig);
    expect(container.textContent).not.toContain(P.gone);
    expect(byRole(container, 'button', P.again)).toBeNull();
    expect(calls).toHaveLength(2);
  });

  it('keeps the URLs while the conversation is open (a closed row reads nothing again) and revokes them when it closes', async () => {
    serve({
      [path('call_png', 0)]: { mediaType: 'image/png', data: PNG },
      '/sessions/sess-2/pictures/call_png/0': { mediaType: 'image/png', data: PNG },
    });
    const container = mount(row('call_png', ONE));
    openRow(container);
    await waitFor(() => container.querySelector('img') !== null);
    openRow(container);
    await waitFor(() => container.querySelector('img') === null);
    openRow(container);
    await waitFor(() => container.querySelector('img') !== null);
    expect(calls).toHaveLength(1);
    expect(revoked).toEqual([]);
    rerender(row('call_png', ONE, 'sess-2'));
    expect(revoked).toEqual(['blob:picture-1']);
    await waitFor(() => container.querySelector('img')?.getAttribute('src') === 'blob:picture-2');
    unmountAll();
    expect(revoked).toEqual(['blob:picture-1', 'blob:picture-2']);
  });

  it('revokes a URL whose read ends after the conversation closed', async () => {
    serve({ [path('call_png', 0)]: { mediaType: 'image/png', data: PNG } });
    const container = mount(row('call_png', ONE));
    openRow(container);
    unmountAll();
    await waitFor(() => made === 1);
    await waitFor(() => revoked.length === 1);
    expect(revoked).toEqual(['blob:picture-1']);
    expect(container.isConnected).toBe(false);
  });

  it('shows nothing without a scope', () => {
    const container = mount(row('call_png', ONE, null));
    openRow(container);
    expect(byRole(container, 'list', P.label)).toBeNull();
  });
});

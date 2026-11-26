/**
 * @vitest-environment happy-dom
 */

/**
 * The picture viewer: a modal dialog named "Picture viewer" whose focus starts on Close; arrows and ← → move with a
 * polite "2 of 3"; the picture's alt names its source; + − 0 zoom; Esc closes and focus goes back to the tile of the
 * picture shown. Also opened from a strip's tile.
 */

import { act, createElement, useState, type ReactElement } from 'react';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  PictureScope,
  PictureStrip,
  type ToolPicture,
} from '../ui/src/components/arc-x/picture-strip/picture-strip.js';
import { PictureViewer, type ViewerPicture } from '../ui/src/components/arc-x/picture-viewer/picture-viewer.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import { mount, unmountAll } from './helpers/mount.js';
import { accessibleName, allByRole, byRole, stubApi, waitFor, type StubbedApi } from './render.js';

const P = CHAT.pictures;

const PICTURES: ViewerPicture[] = [4, 5, 6].map((page, i) => ({
  key: i,
  url: `blob:page-${page}`,
  state: 'ready',
  alt: fill(P.altPage, { page, path: 'docs/spec.pdf' }),
  title: `docs/spec.pdf · ${fill(P.page, { page })}`,
  open: fill(P.openPage, { page }),
}));

let api: StubbedApi | null = null;

afterEach(async () => {
  unmountAll();
  vi.restoreAllMocks();
  if (api !== null) {
    api();
    await api.closed;
    api = null;
  }
});

function Harness(props: { pictures: ViewerPicture[]; onClose?: () => void }): ReactElement {
  const [open, setOpen] = useState(true);
  const [index, setIndex] = useState(1);
  return createElement(PictureViewer, {
    open,
    pictures: props.pictures,
    index,
    onIndex: setIndex,
    onClose: () => {
      setOpen(false);
      props.onClose?.();
    },
  });
}

function dialog(): HTMLElement {
  const found = byRole(document.body, 'dialog', P.viewer);
  if (found === null) throw new Error('no viewer');
  return found;
}

function key(name: string): void {
  act(() => {
    dialog().dispatchEvent(new KeyboardEvent('keydown', { key: name, bubbles: true }));
  });
}

function counter(): string {
  return dialog().querySelector('[aria-live="polite"]')!.textContent ?? '';
}

describe('PictureViewer', () => {
  it('is a modal dialog; focus starts on Close; the picture is named by its source and the counter says where it is', async () => {
    mount(createElement(Harness, { pictures: PICTURES }));
    const view = dialog();
    expect(view.getAttribute('aria-modal')).toBe('true');
    await waitFor(() => document.activeElement === byRole(view, 'button', P.close));
    expect(counter()).toBe(fill(P.count, { n: 2, total: 3 }));
    expect(view.textContent).toContain(PICTURES[1]!.title);
    const img = view.querySelector<HTMLImageElement>('img:not([alt=""])')!;
    expect(img.getAttribute('alt')).toBe(fill(P.altPage, { page: 5, path: 'docs/spec.pdf' }));
    expect(img.getAttribute('src')).toBe('blob:page-5');
  });

  it('← → and the arrow buttons move, stopping at the ends; the thumbnails mark the one shown', () => {
    mount(createElement(Harness, { pictures: PICTURES }));
    key('ArrowRight');
    expect(counter()).toBe(fill(P.count, { n: 3, total: 3 }));
    expect((byRole(dialog(), 'button', P.next) as HTMLButtonElement).disabled).toBe(true);
    key('ArrowRight');
    expect(counter()).toBe(fill(P.count, { n: 3, total: 3 }));
    act(() => byRole(dialog(), 'button', P.prev)!.click());
    key('ArrowLeft');
    expect(counter()).toBe(fill(P.count, { n: 1, total: 3 }));
    expect((byRole(dialog(), 'button', P.prev) as HTMLButtonElement).disabled).toBe(true);
    const thumbs = allByRole(dialog(), 'button').filter((b) => b.hasAttribute('aria-current') || b.closest('[class]'));
    const named = thumbs.filter((b) => PICTURES.some((p) => p.open === accessibleName(b)));
    expect(named.map((b) => b.getAttribute('aria-current'))).toEqual(['true', null, null]);
    act(() => named[2]!.click());
    expect(counter()).toBe(fill(P.count, { n: 3, total: 3 }));
  });

  it('+ − 0 zoom: Zoom out and Actual fit turn on once zoomed and off at fit', () => {
    mount(createElement(Harness, { pictures: PICTURES }));
    const out = () => byRole(dialog(), 'button', P.zoomOut) as HTMLButtonElement;
    const fit = () => byRole(dialog(), 'button', P.zoomReset) as HTMLButtonElement;
    expect(out().disabled).toBe(true);
    key('+');
    expect(out().disabled).toBe(false);
    expect(fit().disabled).toBe(false);
    key('0');
    expect(fit().disabled).toBe(true);
    key('=');
    key('-');
    expect(out().disabled).toBe(true);
  });

  it('a picture not read yet or gone says so in place of the picture', () => {
    const pictures: ViewerPicture[] = [
      { ...PICTURES[0]!, url: null, state: 'gone' },
      { ...PICTURES[1]!, url: null, state: 'loading' },
      PICTURES[2]!,
    ];
    mount(createElement(Harness, { pictures }));
    expect(dialog().textContent).toContain(P.loading);
    key('ArrowLeft');
    expect(dialog().textContent).toContain(P.gone);
    expect(dialog().querySelector('img:not([alt=""])')).toBeNull();
  });

  it('Esc closes it', () => {
    const onClose = vi.fn();
    mount(createElement(Harness, { pictures: PICTURES, onClose }));
    key('Escape');
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(byRole(document.body, 'dialog', P.viewer)).toBeNull();
  });

  it('from a strip: a tile opens it on its picture, and Esc gives focus back to the tile of the picture shown', async () => {
    let made = 0;
    vi.spyOn(URL, 'createObjectURL').mockImplementation(() => `blob:p-${++made}`);
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    const data = Buffer.from('jpg').toString('base64');
    api = stubApi({
      '/sessions/s1/pictures/call_pdf/0': { mediaType: 'image/jpeg', data },
      '/sessions/s1/pictures/call_pdf/1': { mediaType: 'image/jpeg', data },
    });
    const pictures: ToolPicture[] = [4, 5].map((page, n) => ({ n, mediaType: 'image/jpeg', bytes: 3, page }));
    const container = mount(
      createElement(PictureScope, {
        session: 's1',
        children: createElement(PictureStrip, { call: 'call_pdf', pictures, path: 'docs/spec.pdf' }),
      }),
    );
    await waitFor(() => byRole(container, 'button', fill(P.openPage, { page: 5 })) !== null);
    const first = byRole(container, 'button', fill(P.openPage, { page: 4 }))!;
    act(() => first.click());
    expect(counter()).toBe(fill(P.count, { n: 1, total: 2 }));
    expect(dialog().querySelector('img:not([alt=""])')!.getAttribute('alt')).toBe(
      fill(P.altPage, { page: 4, path: 'docs/spec.pdf' }),
    );
    key('ArrowRight');
    key('Escape');
    await waitFor(() => byRole(document.body, 'dialog', P.viewer) === null);
    const shown = byRole(container, 'button', fill(P.openPage, { page: 5 }));
    await waitFor(() => document.activeElement === shown);
  });
});

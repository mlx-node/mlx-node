/**
 * @vitest-environment happy-dom
 */

import { act, createElement, Fragment } from 'react';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import {
  type AttachmentChipItem,
  AttachmentStrip,
  type AttachmentStripProps,
} from '../ui/src/components/arc-x/attachment-strip/attachment-strip.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { mount, rerender, unmountAll } from './helpers/mount.js';
import { accessibleName, allByRole, byRole } from './render.js';

afterEach(unmountAll);

const FILE: AttachmentChipItem = { id: 'f1', kind: 'file', name: 'notes.md', bytes: 2_048 };
const PICTURE: AttachmentChipItem = {
  id: 'i1',
  kind: 'image',
  name: 'photo.jpg',
  mediaType: 'image/jpeg',
  bytes: 319_488,
  thumbnail: 'data:image/jpeg;base64,/9j/',
};
const OTHER: AttachmentChipItem = { id: 'f2', kind: 'file', name: 'b.ts', bytes: 10 };

function strip(props: AttachmentStripProps): HTMLElement {
  return mount(createElement(AttachmentStrip, props));
}

function chips(container: HTMLElement): HTMLElement[] {
  return allByRole(byRole(container, 'list', CHAT.attach.list)!, 'listitem');
}

describe('AttachmentStrip', () => {
  it('a chip per item with its name and size; a composer picture has a thumbnail and its type', () => {
    const container = strip({ items: [FILE, PICTURE], onRemove: () => undefined });
    const [file, picture] = chips(container);
    expect(file!.textContent).toContain('notes.md');
    expect(file!.textContent).toContain('2.0 KB');
    expect(file!.querySelector('img')).toBeNull();
    expect(picture!.textContent).toContain('photo.jpg');
    expect(picture!.textContent).toContain('JPEG · 312 KB');
    const thumb = picture!.querySelector('img')!;
    expect([thumb.getAttribute('src'), thumb.getAttribute('alt')]).toEqual([PICTURE.thumbnail, '']);
  });

  it('remove calls back with the id; focus moves to the next chip, else to the message field', () => {
    const onRemove = vi.fn();
    const container = mount(
      createElement(
        'form',
        null,
        createElement(AttachmentStrip, { items: [FILE, OTHER], onRemove }),
        createElement('textarea', { 'aria-label': 'Message' }),
      ),
    );
    const first = byRole(container, 'button', fill(CHAT.attach.remove, { name: 'notes.md' }))!;
    act(() => first.click());
    expect(onRemove).toHaveBeenCalledWith('f1');
    expect(document.activeElement).toBe(byRole(container, 'button', fill(CHAT.attach.remove, { name: 'b.ts' })));
    act(() => byRole(container, 'button', fill(CHAT.attach.remove, { name: 'b.ts' }))!.click());
    expect(onRemove).toHaveBeenLastCalledWith('f2');
    expect(document.activeElement).toBe(byRole(container, 'textbox', 'Message'));
  });

  it('readOnly: no remove button and no thumbnail; a thread picture reads Image with its type and size', () => {
    const thread: AttachmentChipItem = { id: 'u0', kind: 'image', mediaType: 'image/png', bytes: 1_024 };
    const container = strip({ items: [FILE, PICTURE, thread], readOnly: true });
    expect(allByRole(container, 'button')).toEqual([]);
    expect(container.querySelector('img')).toBeNull();
    const last = chips(container)[2]!;
    expect(last.textContent).toContain(CHAT.attach.image);
    expect(last.textContent).toContain('PNG · 1.0 KB');
  });

  it('a problem chip shows its text in place of the size', () => {
    const container = strip({ items: [{ ...PICTURE, problem: CHAT.attach.chipModel }], onRemove: () => undefined });
    const chip = chips(container)[0]!;
    expect(chip.textContent).toContain(CHAT.attach.chipModel);
    expect(chip.textContent).not.toContain('312 KB');
    expect(chip.hasAttribute('data-problem')).toBe(true);
  });

  it('the note is a status line, also with no chips; nothing at all with neither', () => {
    const container = strip({ items: [], note: fill(CHAT.attach.errors.pdfLocked, { name: 'a.pdf' }) });
    expect(byRole(container, 'status')?.textContent).toBe(fill(CHAT.attach.errors.pdfLocked, { name: 'a.pdf' }));
    expect(byRole(container, 'list')).toBeNull();
    rerender(createElement(AttachmentStrip, { items: [] }));
    expect(container.innerHTML).toBe('');
    rerender(createElement(Fragment, null, createElement(AttachmentStrip, { items: [FILE], note: 'x' })));
    expect(byRole(container, 'status')?.textContent).toBe('x');
    expect(chips(container)).toHaveLength(1);
  });

  it('the caption follows the chips, in the warning tone when near full', () => {
    const text = fill(CHAT.attach.tokensNearFull, { n: '41,200' });
    const container = strip({ items: [FILE], caption: { text, tone: 'warning' } });
    const caption = [...container.querySelectorAll('[data-tone]')].find((node) => node.textContent === text);
    expect(caption?.getAttribute('data-tone')).toBe('warning');
    expect(byRole(container, 'list')!.contains(caption!)).toBe(false);
    expect(accessibleName(byRole(container, 'list')!)).toBe(CHAT.attach.list);
  });

  it('CSS: chips from tokens, the problem border in danger, the note and the warning caption in warning', () => {
    const rules = cssRules('components/arc-x/attachment-strip/attachment-strip.module.css');
    expect(cssValue(rules, '.chip', 'min-height')).toBe('var(--control-height-md)');
    expect(cssValue(rules, '.chip', 'border')).toBe('1px solid var(--border-subtle)');
    expect(cssValue(rules, '.chip[data-problem]', 'border-color')).toBe('var(--danger)');
    expect(cssValue(rules, '.chip[data-problem] .meta', 'color')).toBe('var(--danger)');
    expect(cssValue(rules, '.note', 'color')).toBe('var(--warning)');
    expect(cssValue(rules, ".caption[data-tone='warning']", 'color')).toBe('var(--warning)');
    expect(cssValue(rules, '.name', 'max-width')).toBe('12rem');
  });
});

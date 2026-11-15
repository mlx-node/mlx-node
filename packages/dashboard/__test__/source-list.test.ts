/**
 * @vitest-environment happy-dom
 */

import { act, createElement } from 'react';
import { afterEach, describe, expect, it, vi } from 'vite-plus/test';

import { CitationChip, SourceList, type SourceItem } from '../ui/src/components/arc-x/source-list/source-list.js';
import { mount, rerender, unmountAll } from './helpers/mount.js';
import { byRole, waitFor } from './render.js';

afterEach(unmountAll);

const SOURCES: SourceItem[] = [
  { url: 'https://a.example/1', host: 'a.example', title: 'A one', date: '2026-11-01' },
  { url: 'https://b.example/2', host: 'b.example', title: 'B two', cited: true, n: 2 },
  { url: 'https://c.example/3', host: 'c.example', title: 'C three' },
  { url: 'https://d.example/4', host: 'd.example', title: 'D four' },
  { url: 'https://e.example/5', host: 'e.example', title: 'E five' },
  { url: 'javascript:alert(1)', host: 'x', title: 'Bad' },
];

function hrefs(container: Element): string[] {
  return [...container.querySelectorAll('a')].map((a) => a.getAttribute('href') ?? '');
}

describe('SourceList', () => {
  it('block: cited chips first, a count of the rest, never an image, only http(s) links', () => {
    const container = mount(createElement(SourceList, { sources: SOURCES }));
    expect(container.textContent).toContain('Sources');
    expect(hrefs(container)).toEqual(['https://b.example/2']);
    const more = byRole(container, 'button', 'Show all 6 sources')!;
    expect(more.textContent).toBe('+5');
    expect(container.querySelector('img')).toBeNull();
    act(() => more.click());
    expect(more.getAttribute('aria-expanded')).toBe('true');
    const text = container.textContent ?? '';
    expect(text.indexOf('B two')).toBeLessThan(text.indexOf('Also found'));
    expect(text.indexOf('Also found')).toBeLessThan(text.indexOf('A one'));
    expect(hrefs(container).some((href) => href.startsWith('javascript'))).toBe(false);
    expect(text).toContain('Bad');
    expect(text).toContain('a.example · 2026-11-01');
  });

  it('no cited source: the first four are chips', () => {
    const container = mount(
      createElement(SourceList, { sources: SOURCES.map((s) => ({ ...s, cited: false })), label: 'Found' }),
    );
    expect(container.textContent).toContain('Found');
    expect(hrefs(container)).toEqual([
      'https://a.example/1',
      'https://b.example/2',
      'https://c.example/3',
      'https://d.example/4',
    ]);
  });

  it('controlled: focusUrl focuses that row', async () => {
    const onOpenChange = vi.fn();
    const container = mount(createElement(SourceList, { sources: SOURCES, open: false, onOpenChange }));
    act(() => byRole(container, 'button', 'Show all 6 sources')!.click());
    expect(onOpenChange).toHaveBeenCalledWith(true);
    rerender(
      createElement(SourceList, { sources: SOURCES, open: true, onOpenChange, focusUrl: 'https://c.example/3' }),
    );
    await waitFor(() => document.activeElement?.getAttribute('href') === 'https://c.example/3');
  });

  it('list layout: rows only, with letter marks', () => {
    const container = mount(createElement(SourceList, { sources: SOURCES.slice(0, 2), layout: 'list' }));
    expect(byRole(container, 'button')).toBeNull();
    expect(container.querySelectorAll('li')).toHaveLength(2);
    expect(container.querySelector('img')).toBeNull();
  });

  it('nothing to list: nothing drawn', () => {
    const container = mount(createElement(SourceList, { sources: [] }));
    expect(container.innerHTML).toBe('');
  });
});

describe('CitationChip', () => {
  it('names its source, shows it on focus and calls back on click', async () => {
    const onSelect = vi.fn();
    const container = mount(createElement(CitationChip, { n: 2, title: 'B two', host: 'b.example', onSelect }));
    const chip = byRole(container, 'button', 'Source 2: B two, b.example')!;
    expect(chip.textContent).toBe('2');
    act(() => chip.focus());
    await waitFor(() => document.body.textContent?.includes('B two') === true);
    act(() => chip.click());
    expect(onSelect).toHaveBeenCalledTimes(1);
  });
});

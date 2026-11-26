/**
 * @vitest-environment happy-dom
 */

import { act, createElement } from 'react';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import {
  DiffView,
  type DiffViewProps,
  diffRows,
  EditDiffs,
  type ToolEdit,
  unifiedText,
} from '../ui/src/components/arc-x/diff-view/diff-view.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { mount, unmountAll } from './helpers/mount.js';
import { accessibleName, allByRole, byRole } from './render.js';

afterEach(unmountAll);

const TWO_HUNKS: DiffViewProps['hunks'] = [
  { start: [118, 118], body: '   case x:\n-  const a = 1;\n+  const a = 2;\n   return a;' },
  { start: [134, 134], body: '   call();\n+  done();' },
];

function view(props: Partial<DiffViewProps> = {}): HTMLElement {
  return mount(
    createElement(DiffView, {
      path: 'ui/src/lib/work-tools.ts',
      hunks: TWO_HUNKS,
      added: 2,
      removed: 1,
      state: 'applied',
      ...props,
    }),
  );
}

describe('DiffView', () => {
  it('is a figure named by the path and the counts', () => {
    const container = view();
    const figure = byRole(container, 'figure')!;
    expect(accessibleName(figure)).toBe(
      `${fill(CHAT.diff.label, { path: 'ui/src/lib/work-tools.ts' })}: ${fill(CHAT.diff.countsSr, { added: 2, removed: 1 })}`,
    );
  });

  it('starts a changed row with hidden "Removed:" or "Added:" words; the tint is not the only sign', () => {
    const container = view();
    const removed = container.querySelector('[data-mark="removed"]')!;
    const added = container.querySelector('[data-mark="added"]')!;
    expect(removed.textContent).toContain(`${CHAT.diff.removed} `);
    expect(added.textContent).toContain(`${CHAT.diff.added} `);
    expect(removed.querySelector('[aria-hidden="true"]:nth-child(3)')!.textContent).toBe('−');
    const numbers = [...removed.querySelectorAll('[aria-hidden="true"]')].slice(0, 2).map((n) => n.textContent);
    expect(numbers).toEqual(['119', '']);
  });

  it('numbers both columns from each hunk and folds the lines between hunks into one gap row', () => {
    const shown = diffRows(TWO_HUNKS);
    expect(
      shown.map((row) => (row.kind === 'gap' ? `gap ${row.n}` : `${row.old ?? '·'} ${row.new ?? '·'} ${row.mark}`)),
    ).toEqual(['118 118  ', '119 · -', '· 119 +', '120 120  ', 'gap 13', '134 134  ', '· 135 +']);
    const container = view();
    expect(container.textContent).toContain(fill(CHAT.diff.gap, { n: 13 }));
    expect(view({ hunks: [TWO_HUNKS[0]!, { start: [122, 122], body: '+x' }] }).textContent).toContain(CHAT.diff.gapOne);
  });

  it('shows 14 rows, then "Show n more lines" that opens a named scroller and toggles aria-expanded', () => {
    const body = Array.from({ length: 20 }, (_, i) => `+line ${i}`).join('\n');
    const container = view({ hunks: [{ start: [1, 1], body }], added: 20, removed: 0 });
    expect(container.querySelectorAll('[data-mark]')).toHaveLength(14);
    const more = byRole(container, 'button', fill(CHAT.diff.more, { n: 6 }))!;
    expect(more.getAttribute('aria-expanded')).toBe('false');
    expect(byRole(container, 'region')).toBeNull();
    act(() => more.click());
    expect(container.querySelectorAll('[data-mark]')).toHaveLength(20);
    expect(more.getAttribute('aria-expanded')).toBe('true');
    expect(more.textContent).toBe(CHAT.diff.less);
    const scroller = byRole(container, 'region')!;
    expect(scroller.getAttribute('tabindex')).toBe('0');
    expect(accessibleName(scroller)).toBe(accessibleName(byRole(container, 'figure')!));
    expect(scroller.id).toBe(more.getAttribute('aria-controls'));
  });

  it('heads a change not made yet with "Not applied yet", a failed one with "Not applied", a made one with nothing', () => {
    expect(view({ state: 'pending' }).querySelector('figcaption')!.textContent).toBe(CHAT.diff.pending);
    expect(view({ state: 'not-applied' }).querySelector('figcaption')!.textContent).toBe(CHAT.diff.notApplied);
    expect(view({ state: 'applied' }).querySelector('figcaption')).toBeNull();
  });

  it('a new file has one number column', () => {
    const container = view({ create: true, hunks: [{ start: [0, 1], body: '+a\n+b' }], added: 2, removed: 0 });
    expect(container.querySelector('figure')!.hasAttribute('data-create')).toBe(true);
    const rules = cssRules('components/arc-x/diff-view/diff-view.module.css');
    expect(cssValue(rules, '.root[data-create] .old', 'display')).toBe('none');
    expect(cssValue(rules, '.old', 'display')).toBeUndefined();
  });

  it('says what the caps left out and when only line endings changed', () => {
    expect(view({ cut: 1_240 }).textContent).toContain(fill(CHAT.diff.cut, { n: '1,240' }));
    const ends = view({ hunks: [], added: 0, removed: 0, endsOnly: true });
    expect(ends.textContent).toContain(CHAT.diff.endsOnly);
    expect(ends.querySelectorAll('[data-mark]')).toHaveLength(0);
  });

  it('draws a row as plain text with hidden characters shown', () => {
    const container = view({ hunks: [{ start: [1, 1], body: '+<b>x</b>\u202e' }], added: 1, removed: 0 });
    expect(container.querySelector('b')).toBeNull();
    expect(container.querySelector('[data-mark="added"]')!.textContent).toContain('<b>x</b>\\u{202E}');
  });

  it('copies the change as unified diff text', () => {
    expect(byRole(view(), 'button', CHAT.diff.copy)).not.toBeNull();
    expect(unifiedText('a.ts', TWO_HUNKS, false)).toBe(
      [
        '--- a/a.ts',
        '+++ b/a.ts',
        '@@ -118,3 +118,3 @@',
        '   case x:',
        '-  const a = 1;',
        '+  const a = 2;',
        '   return a;',
        '@@ -134,1 +134,2 @@',
        '   call();',
        '+  done();',
        '',
      ].join('\n'),
    );
    expect(unifiedText('n.md', [{ start: [0, 1], body: '+a' }], true)).toBe(
      '--- /dev/null\n+++ b/n.md\n@@ -0,0 +1,1 @@\n+a\n',
    );
  });
});

describe('EditDiffs', () => {
  const edit = (path: string): ToolEdit => ({ path, added: 1, removed: 0, hunks: [{ start: [1, 1], body: '+x' }] });

  it('stacks the files of one call, names each, and folds the 4th on behind "n more files"', () => {
    const container = mount(
      createElement(EditDiffs, {
        edits: ['/w/a.ts', '/w/b.ts', '/w/c.ts', '/w/d.ts', '/w/e.ts'].map(edit),
        state: 'applied',
        folder: '/w',
      }),
    );
    const figures = allByRole(container, 'figure');
    expect(figures.map((f) => f.querySelector('figcaption')?.textContent)).toEqual(['a.ts', 'b.ts', 'c.ts']);
    const more = byRole(container, 'button', fill(CHAT.diff.moreFiles, { n: 2 }))!;
    act(() => more.click());
    expect(allByRole(container, 'figure')).toHaveLength(5);
    const four = mount(
      createElement(EditDiffs, {
        edits: ['/w/a.ts', '/w/b.ts', '/w/c.ts', '/w/d.ts'].map(edit),
        state: 'applied',
        folder: '/w',
      }),
    );
    expect(byRole(four, 'button', CHAT.diff.moreFilesOne)).not.toBeNull();
  });

  it('names no file above a call’s only diff', () => {
    const container = mount(createElement(EditDiffs, { edits: [edit('/w/a.ts')], state: 'applied', folder: '/w' }));
    expect(container.querySelector('figcaption')).toBeNull();
  });
});

/**
 * @vitest-environment happy-dom
 */

import { act, createElement, type ReactElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, describe, expect, it } from 'vite-plus/test';

import {
  DataTable,
  freeColumn,
  sortRows,
  visibleColumns,
  type DataColumn,
  type DataTableProps,
} from '../ui/src/components/arc-x/data-table/data-table.js';
import { useRichMotion } from '../ui/src/components/arc-x/motion/use-entrance.js';
import { TURN_COLUMNS } from '../ui/src/pages/models/model-drawer.js';
import { cssRules, cssValue } from './helpers/css-rules.js';
import { byRole, skipAnimations, waitFor } from './render.js';

interface Row {
  id: string;
  name: string;
  turns: number | null;
}

const ROWS: readonly Row[] = [
  { id: 'a', name: 'alpha', turns: 3 },
  { id: 'b', name: 'bravo', turns: null },
  { id: 'c', name: 'charlie', turns: 12 },
];

const COLUMNS: readonly DataColumn<Row>[] = [
  { id: 'name', header: 'Name', cell: (row) => row.name, sortValue: (row) => row.name },
  {
    id: 'turns',
    header: 'Turns',
    cell: (row) => (row.turns === null ? '—' : String(row.turns)),
    sortValue: (row) => row.turns,
    align: 'end',
    minTableWidth: 10_000,
  },
];

let root: Root | null = null;

function mount(element: ReactElement): HTMLElement {
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const container = document.createElement('div');
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => root?.render(element));
  return container;
}

function table(overrides: Partial<DataTableProps<Row>> = {}): ReactElement {
  return createElement(DataTable<Row>, {
    caption: 'Sessions',
    columns: COLUMNS,
    rows: ROWS,
    rowKey: (row: Row) => row.id,
    ...overrides,
  });
}

function bodyRows(container: HTMLElement): HTMLTableRowElement[] {
  return [...container.querySelectorAll<HTMLTableRowElement>('tbody tr')];
}

function names(container: HTMLElement): string[] {
  return bodyRows(container).map((row) => row.cells[0]?.textContent ?? '');
}

function keyDown(target: Element, key: string): void {
  act(() => {
    target.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
  });
}

afterEach(() => {
  act(() => root?.unmount());
  root = null;
  document.body.innerHTML = '';
});

describe('column rule', () => {
  it('every cell is one line: no wrap inside a fixed row, a cut value ends in an ellipsis', () => {
    const rules = cssRules('components/arc-x/data-table/data-table.module.css');
    expect(cssValue(rules, '.cell', 'white-space')).toBe('nowrap');
    expect(cssValue(rules, '.cell', 'text-overflow')).toBe('ellipsis');
  });

  it('the Models drawer recent turns leave Outcome the room of its longest word', () => {
    const drawerTable = 429;
    const cancelledWithDot = 71;
    const fixed = TURN_COLUMNS.reduce((sum, column) => sum + (typeof column.width === 'number' ? column.width : 0), 0);
    expect(TURN_COLUMNS.find((column) => column.id === 'outcome')?.flex).toBe(true);
    expect(drawerTable - fixed).toBeGreaterThanOrEqual(cancelledWithDot + 24);
  });

  it('the measured box takes its width from its container: a table wider than the card cannot widen it', () => {
    const rules = cssRules('components/arc-x/data-table/data-table.module.css');
    expect(cssValue(rules, '.root', 'contain')).toBe('inline-size');
    expect(cssValue(rules, '.root', 'min-width')).toBe('0');
  });

  it('the flex column takes the free width: its col has no width, the others keep theirs', () => {
    const columns: DataColumn<Row>[] = [
      { id: 'when', header: 'When', width: 112, cell: (row) => row.name },
      { id: 'name', header: 'Name', flex: true, truncate: true, cell: (row) => row.name },
      { id: 'turns', header: 'Turns', width: 80, cell: (row) => String(row.turns) },
    ];
    const container = mount(table({ columns }));
    const cols = [...container.querySelectorAll('colgroup col')].map((col) => col.getAttribute('style'));
    expect(cols).toEqual(['width: 112px;', null, 'width: 80px;']);
    expect(freeColumn(columns)).toBe(-1);
    expect(freeColumn([columns[0]!, columns[2]!])).toBe(0);
  });

  it('a table with no flex column and a width on each column gives the free width to its first column', () => {
    const columns: DataColumn<Row>[] = [
      { id: 'when', header: 'When', width: 112, cell: (row) => row.name },
      { id: 'turns', header: 'Turns', width: 80, cell: (row) => String(row.turns) },
    ];
    const container = mount(table({ columns }));
    const cols = [...container.querySelectorAll('colgroup col')].map((col) => col.getAttribute('style'));
    expect(cols).toEqual([null, 'width: 80px;']);
  });

  it('drops the column with the largest minTableWidth while the widths do not fit', () => {
    const columns = [
      { id: 'a', width: 200 },
      { id: 'b', flex: true as const, minWidth: 150 },
      { id: 'c', width: 100, minTableWidth: 300 },
      { id: 'd', width: 100, minTableWidth: 400 },
    ];
    expect(visibleColumns(columns, 0).map((c) => c.id)).toEqual(['a', 'b', 'c', 'd']);
    expect(visibleColumns(columns, 560).map((c) => c.id)).toEqual(['a', 'b', 'c', 'd']);
    expect(visibleColumns(columns, 500).map((c) => c.id)).toEqual(['a', 'b', 'c']);
    expect(visibleColumns(columns, 420).map((c) => c.id)).toEqual(['a', 'b']);
    expect(visibleColumns(columns, 340).map((c) => c.id)).toEqual(['a', 'b']);
  });
});

describe('sortRows', () => {
  it('puts null last in both directions and keeps ties in order', () => {
    const turns = COLUMNS[1];
    expect(sortRows(ROWS, turns, 'asc').map((row) => row.id)).toEqual(['a', 'c', 'b']);
    expect(sortRows(ROWS, turns, 'desc').map((row) => row.id)).toEqual(['c', 'a', 'b']);
    const tied = [...ROWS, { id: 'd', name: 'delta', turns: 3 }];
    expect(sortRows(tied, turns, 'asc').map((row) => row.id)).toEqual(['a', 'd', 'c', 'b']);
  });
});

describe('DataTable', () => {
  it('shows every column while its width is unmeasured, under a caption', () => {
    const container = mount(table());
    const table_ = byRole(container, 'table', 'Sessions');
    expect(table_).not.toBeNull();
    expect([...container.querySelectorAll('thead th')].map((th) => th.textContent)).toEqual(['Name', 'Turns']);
    expect(bodyRows(container)[0].cells).toHaveLength(2);
  });

  it('is one tab stop: ArrowDown, End and Home move real focus to a row with tabindex 0', () => {
    const container = mount(table());
    const rows = bodyRows(container);
    expect(rows.map((row) => row.getAttribute('tabindex'))).toEqual(['0', '-1', '-1']);

    act(() => rows[0].focus());
    keyDown(rows[0], 'ArrowDown');
    expect(document.activeElement).toBe(bodyRows(container)[1]);
    expect(bodyRows(container)[1].getAttribute('tabindex')).toBe('0');
    expect(bodyRows(container)[0].getAttribute('tabindex')).toBe('-1');

    keyDown(document.activeElement as Element, 'End');
    expect(document.activeElement).toBe(bodyRows(container)[2]);
    keyDown(document.activeElement as Element, 'Home');
    expect(document.activeElement).toBe(bodyRows(container)[0]);
    keyDown(document.activeElement as Element, 'ArrowUp');
    expect(document.activeElement).toBe(bodyRows(container)[0]);
  });

  it("keeps only the active row's controls in the tab order, and follows the row focus moves to", () => {
    const columns: DataColumn<Row>[] = [
      ...COLUMNS,
      {
        id: 'actions',
        header: '',
        cell: (row) =>
          createElement(
            'span',
            null,
            createElement('button', { type: 'button', tabIndex: 0 }, `Copy ${row.name}`),
            createElement('button', { type: 'button' }, `Delete ${row.name}`),
          ),
      },
    ];
    const container = mount(table({ columns }));
    const stops = (): string[] =>
      [...container.querySelectorAll<HTMLElement>('tbody tr, tbody button')]
        .filter((node) => node.tabIndex >= 0)
        .map((node) => (node instanceof HTMLTableRowElement ? `row ${node.cells[0]?.textContent}` : node.textContent!));
    expect(stops()).toEqual(['row alpha', 'Copy alpha', 'Delete alpha']);

    act(() => bodyRows(container)[0].focus());
    keyDown(bodyRows(container)[0], 'ArrowDown');
    expect(stops()).toEqual(['row bravo', 'Copy bravo', 'Delete bravo']);

    const far = [...container.querySelectorAll<HTMLButtonElement>('tbody button')].find(
      (button) => button.textContent === 'Delete charlie',
    )!;
    act(() => far.focus());
    expect(stops()).toEqual(['row charlie', 'Copy charlie', 'Delete charlie']);
    // An explicit tabindex (Arc Button sets 0) comes back as it was.
    expect(far.parentElement?.querySelector('button')?.getAttribute('tabindex')).toBe('0');
  });

  it('frame={false} draws no surface of its own, for a table inside a Panel', () => {
    const framed = mount(table());
    expect(framed.querySelector('table')?.closest('section')).not.toBeNull();
    act(() => root?.unmount());
    const bare = mount(table({ frame: false }));
    expect(bare.querySelector('table')?.closest('section')).toBeNull();
  });

  it('an arrow key at the edge does not pull focus back on a later render', () => {
    const outside = document.createElement('input');
    document.body.appendChild(outside);
    const container = mount(table());
    const rows = bodyRows(container);
    act(() => rows[0].focus());
    keyDown(rows[0], 'ArrowUp');
    keyDown(rows[0], 'Home');
    expect(document.activeElement).toBe(rows[0]);
    act(() => outside.focus());
    act(() => root?.render(table({ rows: [...ROWS] })));
    expect(document.activeElement).toBe(outside);
  });

  it('keeps the active row by key when a refetch prepends a row', () => {
    const four = [...ROWS, { id: 'd', name: 'delta', turns: 1 }];
    const container = mount(table({ rows: four }));
    act(() => bodyRows(container)[0].focus());
    keyDown(bodyRows(container)[0], 'ArrowDown');
    const bravo = bodyRows(container)[1];
    expect(document.activeElement).toBe(bravo);

    act(() => root?.render(table({ rows: [{ id: 'z', name: 'zulu', turns: 0 }, ...four] })));
    expect(document.activeElement).toBe(bravo);
    expect(bravo.getAttribute('tabindex')).toBe('0');
    expect(bodyRows(container).filter((row) => row.getAttribute('tabindex') === '0')).toEqual([bravo]);
    keyDown(bravo, 'ArrowDown');
    expect((document.activeElement as HTMLTableRowElement).cells[0]?.textContent).toBe('charlie');
  });

  it('moves focus to the row now at its place when a refetch removes the focused row', () => {
    const container = mount(table());
    act(() => bodyRows(container)[0].focus());
    keyDown(bodyRows(container)[0], 'ArrowDown');
    expect((document.activeElement as HTMLTableRowElement).cells[0]?.textContent).toBe('bravo');

    act(() => root?.render(table({ rows: ROWS.filter((row) => row.id !== 'b') })));
    expect((document.activeElement as HTMLTableRowElement).cells[0]?.textContent).toBe('charlie');
    expect((document.activeElement as HTMLElement).getAttribute('tabindex')).toBe('0');
  });

  it('puts rowAttrs on the rows, so a selected row is marked', () => {
    const container = mount(
      table({ rowAttrs: (row) => ({ 'data-session-id': row.id, 'data-selected': String(row.id === 'b') }) }),
    );
    const selected = container.querySelectorAll('tr[data-selected="true"]');
    expect(selected).toHaveLength(1);
    expect(selected[0].getAttribute('data-session-id')).toBe('b');
  });

  it('loading keeps the header and fills the same cells with one placeholder each', () => {
    const container = mount(table({ loading: true, loadingRows: 4 }));
    expect(byRole(container, 'table', 'Sessions')?.getAttribute('aria-busy')).toBe('true');
    expect([...container.querySelectorAll('thead th')].map((th) => th.textContent)).toEqual(['Name', 'Turns']);
    const rows = bodyRows(container);
    expect(rows).toHaveLength(4);
    for (const row of rows) {
      expect(row.cells).toHaveLength(2);
      for (const cell of row.cells) expect(cell.querySelectorAll('[data-placeholder]')).toHaveLength(1);
    }
  });

  it('empty and error replace the rows and keep the header', () => {
    const container = mount(table({ rows: [], empty: { title: 'No sessions', description: 'Start one.' } }));
    expect(container.textContent).toContain('No sessions');
    expect(container.querySelectorAll('thead th')).toHaveLength(2);
    act(() => root?.render(table({ error: { message: 'Could not load sessions' } })));
    expect(container.querySelector('[role="alert"]')?.textContent).toContain('Could not load sessions');
  });

  it('sorts from the header with aria-sort, null last, and never marks a re-sort under the harness', async () => {
    const container = mount(table({ defaultSort: { id: 'name', direction: 'asc' } }));
    expect(names(container)).toEqual(['alpha', 'bravo', 'charlie']);
    const turnsHeader = byRole(container, 'button', 'Turns');
    act(() => turnsHeader?.click());
    expect(names(container)).toEqual(['charlie', 'alpha', 'bravo']);
    expect(container.querySelector('th[aria-sort]')?.getAttribute('aria-sort')).toBe('descending');
    act(() => byRole(container, 'button', 'Turns')?.click());
    expect(names(container)).toEqual(['alpha', 'charlie', 'bravo']);
    expect(container.querySelector('th[aria-sort]')?.getAttribute('aria-sort')).toBe('ascending');
    await waitFor(() => true);
    expect(container.querySelector('tbody')?.hasAttribute('data-resorting')).toBe(false);
  });

  it('follows the row link on Enter', async () => {
    const container = mount(
      createElement(
        MemoryRouter,
        { initialEntries: ['/'] },
        createElement(
          Routes,
          null,
          createElement(Route, { path: '/', element: table({ rowTo: (row) => `/sessions/${row.id}` }) }),
          createElement(Route, { path: '/sessions/:id', element: createElement('h1', null, 'Session page') }),
        ),
      ),
    );
    expect(container.querySelectorAll('a[href="/sessions/a"]')).toHaveLength(1);
    const rows = bodyRows(container);
    act(() => rows[1].focus());
    keyDown(rows[1], 'Enter');
    await waitFor(() => container.textContent === 'Session page');
  });

  it('with motion on, rows 1-12 keep rich motion in their cells and row 13 on sits in a closed gate', () => {
    const previous = skipAnimations(false);
    try {
      const many: Row[] = [];
      for (let i = 0; i < 20; i += 1) many.push({ id: `r${i}`, name: `row ${i}`, turns: i });
      function Probe(): ReactElement {
        return createElement('span', { 'data-probe': '' }, String(useRichMotion()));
      }
      const probeColumns: readonly DataColumn<Row>[] = [
        COLUMNS[0],
        { id: 'probe', header: 'Probe', cell: () => createElement(Probe) },
      ];
      const container = mount(table({ rows: many, columns: probeColumns }));
      const probes = [...container.querySelectorAll('[data-probe]')].map((node) => node.textContent);
      expect(probes).toHaveLength(20);
      expect(probes[11]).toBe('true');
      expect(probes[12]).toBe('false');
      expect(probes.slice(0, 12).every((value) => value === 'true')).toBe(true);
      expect(probes.slice(12).every((value) => value === 'false')).toBe(true);
    } finally {
      skipAnimations(previous);
    }
  });

  it('a windowed row that finished its reveal does not replay it when it mounts again', () => {
    const previous = skipAnimations(false);
    try {
      const many: Row[] = [];
      for (let i = 0; i < 300; i += 1) many.push({ id: `r${i}`, name: `row ${i}`, turns: i });
      const container = mount(table({ rows: many }));
      const rowNamed = (name: string): HTMLTableRowElement | undefined =>
        bodyRows(container).find((row) => row.cells[0]?.textContent === name);
      const first = rowNamed('row 1');
      expect(first?.hasAttribute('data-reveal')).toBe(true);
      act(() => {
        first?.dispatchEvent(new Event('animationend', { bubbles: true }));
      });
      expect(rowNamed('row 1')?.hasAttribute('data-reveal')).toBe(false);

      act(() => root?.render(table({ rows: many.filter((row) => row.id !== 'r1') })));
      expect(rowNamed('row 1')).toBeUndefined();
      act(() => root?.render(table({ rows: [...many] })));
      const again = rowNamed('row 1');
      expect(again).toBeDefined();
      expect(again).not.toBe(first);
      expect(again?.hasAttribute('data-reveal')).toBe(false);
      expect(rowNamed('row 2')?.hasAttribute('data-reveal')).toBe(true);
    } finally {
      skipAnimations(previous);
    }
  });

  it('keeps a far-away active row mounted on its own: scrolled to the bottom of 3000 rows, only the window and row 0 render', async () => {
    const ROW = 36;
    const VIEW = 600;
    const many: Row[] = [];
    for (let i = 0; i < 3000; i += 1) many.push({ id: `r${i}`, name: `row ${i}`, turns: i });
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    const scroller = document.createElement('main');
    scroller.setAttribute('data-scroll-root', '');
    document.body.appendChild(scroller);
    Object.defineProperty(scroller, 'clientHeight', { configurable: true, value: VIEW });
    scroller.getBoundingClientRect = () => new DOMRect(0, 0, 800, VIEW);
    root = createRoot(scroller);
    act(() => root?.render(table({ rows: many })));
    const first = bodyRows(scroller)[0];
    act(() => first.focus());
    expect(first.getAttribute('tabindex')).toBe('0');

    const body = scroller.querySelector('tbody')!;
    body.getBoundingClientRect = () => new DOMRect(0, VIEW - 3000 * ROW, 800, 3000 * ROW);
    act(() => {
      scroller.dispatchEvent(new Event('scroll'));
    });
    const shown = (): HTMLTableRowElement[] =>
      bodyRows(scroller).filter((row) => row.getAttribute('aria-hidden') !== 'true');
    await waitFor(() => shown().some((row) => row.cells[0]?.textContent === 'row 2999'));

    const windowSize = Math.ceil(VIEW / ROW) + 2 * 8 + 1;
    expect(shown().length).toBeLessThanOrEqual(windowSize + 1);
    expect(shown()[0].cells[0]?.textContent).toBe('row 0');
    expect(shown()[0].getAttribute('tabindex')).toBe('0');
    expect(shown()[1].getAttribute('aria-rowindex')).not.toBe('3');
    const spacers = bodyRows(scroller).filter((row) => row.getAttribute('aria-hidden') === 'true');
    const spaced = spacers.reduce((sum, row) => sum + Number.parseFloat(row.style.height), 0);
    expect(spaced).toBe((3000 - shown().length) * ROW);
    expect(bodyRows(scroller).indexOf(shown()[0])).toBe(0);

    act(() => root?.render(table({ rows: [{ id: 'new', name: 'row new', turns: -1 }, ...many] })));
    expect(first.isConnected).toBe(true);
    expect(document.activeElement).toBe(first);
    expect(first.getAttribute('tabindex')).toBe('0');
  });

  it('windows a long table: fewer DOM rows, row count on the table, row index on rows', () => {
    const many: Row[] = [];
    for (let i = 0; i < 300; i += 1) many.push({ id: `r${i}`, name: `row ${i}`, turns: i });
    const container = mount(table({ rows: many }));
    const element = byRole(container, 'table', 'Sessions');
    expect(element?.getAttribute('aria-rowcount')).toBe('301');
    const rendered = bodyRows(container).filter((row) => row.getAttribute('aria-hidden') !== 'true');
    expect(rendered.length).toBeLessThan(150);
    expect(rendered[0].getAttribute('aria-rowindex')).toBe('2');
  });

  it('a linked row shows the title under the pointer, which its link overlay would hide, cut text included', () => {
    const columns: readonly DataColumn<Row>[] = [
      { id: 'name', header: 'Name', cell: (row) => row.name },
      {
        id: 'turns',
        header: 'Turns',
        cell: (row) =>
          createElement('span', { 'data-clip-title': `${row.name}: ${String(row.turns)} turns` }, String(row.turns)),
      },
      { id: 'note', header: 'Note', cell: () => createElement('span', { title: 'a note' }, 'note') },
    ];
    const container = mount(
      createElement(
        MemoryRouter,
        null,
        table({ columns, rows: ROWS.slice(0, 1), rowTo: (row) => `/sessions/${row.id}` }),
      ),
    );
    const link = container.querySelector('a[data-row-link]') as HTMLElement;
    const row = link.closest('tr') as HTMLElement;
    const clipped = row.querySelector('[data-clip-title]') as HTMLElement;
    const noted = row.querySelector('[title="a note"]') as HTMLElement;
    let under: Element[] = [];
    Object.defineProperty(document, 'elementsFromPoint', { configurable: true, value: () => under });
    const move = (): void => {
      act(() => {
        link.dispatchEvent(new PointerEvent('pointermove', { bubbles: true, clientX: 1, clientY: 1 }));
      });
    };
    try {
      under = [link, clipped, clipped.parentElement as Element, row];
      Object.defineProperty(clipped, 'scrollWidth', { configurable: true, value: 40 });
      Object.defineProperty(clipped, 'clientWidth', { configurable: true, value: 20 });
      move();
      expect(link.getAttribute('title')).toBe('alpha: 3 turns');
      Object.defineProperty(clipped, 'clientWidth', { configurable: true, value: 60 });
      move();
      expect(link.hasAttribute('title')).toBe(false);
      under = [link, noted, noted.parentElement as Element, row];
      move();
      expect(link.getAttribute('title')).toBe('a note');
    } finally {
      Reflect.deleteProperty(document, 'elementsFromPoint');
    }
  });
});

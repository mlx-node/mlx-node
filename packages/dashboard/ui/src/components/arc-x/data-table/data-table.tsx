import { ArrowUp } from 'lucide-react';
import { animate } from 'motion/react';
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type ReactElement,
  type ReactNode,
} from 'react';
import { Link } from 'react-router-dom';

import { forwardRowTitle } from '../../../lib/hover-title';
import { Alert } from '../../arc/alert/alert';
import { Button } from '../../arc/button/button';
import { EmptyState } from '../../arc/empty-state/empty-state';
import { motionX } from '../motion/motion';
import { MotionGate, useRichMotion } from '../motion/use-entrance';
import { Panel } from '../panel/panel';
import { Placeholder } from '../placeholder/placeholder';
import styles from './data-table.module.css';

export interface DataColumn<T> {
  id: string;
  header: string;
  cell: (row: T) => ReactNode;
  /** Enables sorting on this column. null sorts last in both directions. */
  sortValue?: (row: T) => number | string | null;
  /** Numeric columns: 'end' + tabular-nums. */
  align?: 'start' | 'end';
  /** Fixed: the widest real value plus the cell padding. Others share the rest. */
  width?: number | string;
  /** The one column that takes all free width: no `width`, and `truncate` set. */
  flex?: true;
  /** `flex` only: the narrowest it gets before the table drops a column. Default 120. */
  minWidth?: number;
  /** Hide the column while the table is narrower than this many px. */
  minTableWidth?: number;
  /** The value may be cut (every cell is one line); the cell supplies title. */
  truncate?: boolean;
  /** Hover text of the header. */
  headerTitle?: string;
  /** Width of the loading block in this column. Default '60%'. */
  placeholderWidth?: string;
}

export interface TableSort {
  id: string;
  direction: 'asc' | 'desc';
}

export interface DataTableProps<T> {
  /** Accessible name (visually hidden caption). */
  caption: string;
  columns: readonly DataColumn<T>[];
  rows: readonly T[];
  rowKey: (row: T) => string;
  /** Extra attributes on <tr>, e.g. { 'data-session-id': row.id }. */
  rowAttrs?: (row: T) => Record<`data-${string}`, string>;
  /** Makes the row a link target: the first cell's content is wrapped in a router link stretched over the row. */
  rowTo?: (row: T) => string | undefined;
  defaultSort?: TableSort;
  sort?: TableSort;
  onSortChange?: (sort: TableSort) => void;
  loading?: boolean;
  /** Default 6. */
  loadingRows?: number;
  empty?: { title: string; description: string; action?: ReactNode };
  error?: { message: string; retry?: () => void };
  /** Row 36 px | 30 px; header 32 px. */
  density?: 'default' | 'compact';
  /** Window the rows above this count. Default 150. */
  virtualizeAbove?: number;
  /** "Showing the newest 500 of 1,204 …" */
  footer?: ReactNode;
  /** false: no surface of its own, for a table that sits inside a Panel (one frame, not a card in a card). Default true. */
  frame?: boolean;
  className?: string;
}

/** Rows that run their first-load entrance and keep rich motion in their cells. */
export const RICH_ROWS = 12;
/** Re-sorts above this many rows swap in place. */
export const TRAVEL_ROWS = 60;
const WINDOW_MARGIN = 8;
const UNMEASURED_WINDOW = 40;
const ROW_PX = { default: 36, compact: 30 } as const;

/** Stable sort by the column's value; null last in both directions. */
export function sortRows<T>(rows: readonly T[], column: DataColumn<T> | undefined, direction: 'asc' | 'desc'): T[] {
  const sortValue = column?.sortValue;
  if (sortValue === undefined) return [...rows];
  const sign = direction === 'asc' ? 1 : -1;
  const keyed = rows.map((row, index) => ({ row, index, value: sortValue(row) }));
  keyed.sort((a, b) => {
    if (a.value === null || b.value === null) {
      if (a.value === b.value) return a.index - b.index;
      return a.value === null ? 1 : -1;
    }
    let order = 0;
    if (typeof a.value === 'number' && typeof b.value === 'number') order = a.value - b.value;
    else order = String(a.value).localeCompare(String(b.value));
    return order === 0 ? a.index - b.index : order * sign;
  });
  return keyed.map((entry) => entry.row);
}

interface Entrance {
  rows: readonly unknown[] | null;
  ready: boolean;
  /** Keys painted at least once. */
  seen: ReadonlySet<string>;
  /** Keys that enter with `reveal`, and their step; a key leaves when its reveal ends, so a remount never replays it. */
  reveal: ReadonlyMap<string, number>;
}

function nextEntrance(previous: Entrance, rows: readonly unknown[], keys: readonly string[], ready: boolean): Entrance {
  if (!ready || keys.length === 0) return { ...previous, rows, ready };
  const reveal = new Map(previous.reveal);
  if (previous.seen.size === 0) {
    for (let i = 0; i < keys.length && i < RICH_ROWS; i += 1)
      reveal.set(keys[i], Math.min(i, motionX.reveal.maxSteps - 1));
  } else {
    for (let i = 0; i < keys.length && !previous.seen.has(keys[i]); i += 1) {
      reveal.set(keys[i], Math.min(i, motionX.reveal.maxSteps - 1));
    }
  }
  const seen = new Set(previous.seen);
  for (const key of keys) seen.add(key);
  return { rows, ready, seen, reveal };
}

function columnStyle(column: { width?: number | string; flex?: true }): CSSProperties | undefined {
  if (column.width === undefined || column.flex === true) return undefined;
  return { width: typeof column.width === 'number' ? `${column.width}px` : column.width };
}

const FLEX_MIN_WIDTH = 120;

/**
 * The columns shown at `tableWidth`: each `minTableWidth` rule, then, while the fixed widths and the flex column's
 * minimum do not fit, the column with the largest `minTableWidth` goes too. 0: not measured yet, every column shows.
 */
export function visibleColumns<C extends Pick<DataColumn<unknown>, 'width' | 'flex' | 'minWidth' | 'minTableWidth'>>(
  columns: readonly C[],
  tableWidth: number,
): C[] {
  if (tableWidth <= 0) return [...columns];
  const shown = columns.filter((column) => column.minTableWidth === undefined || tableWidth >= column.minTableWidth);
  const need = (list: readonly C[]): number =>
    list.reduce((sum, column) => {
      if (column.flex === true) return sum + (column.minWidth ?? FLEX_MIN_WIDTH);
      return sum + (typeof column.width === 'number' ? column.width : 0);
    }, 0);
  while (need(shown) > tableWidth) {
    let drop = -1;
    shown.forEach((column, index) => {
      if (column.minTableWidth === undefined) return;
      if (drop === -1 || column.minTableWidth > shown[drop]!.minTableWidth!) drop = index;
    });
    if (drop === -1) break;
    shown.splice(drop, 1);
  }
  return shown;
}

/** With no flex column and a width on every column, the first column has none, so the free width goes to it. */
export function freeColumn(columns: ReadonlyArray<Pick<DataColumn<unknown>, 'width' | 'flex'>>): number {
  if (columns.some((column) => column.flex === true)) return -1;
  return columns.every((column) => column.width !== undefined) ? 0 : -1;
}

function scrollRootOf(node: Element | null): HTMLElement | null {
  return node?.closest<HTMLElement>('[data-scroll-root]') ?? null;
}

const CELL_FOCUSABLE = 'a[href], button, input, select, textarea, [tabindex]';
/** Holds the tabindex a control had before this table took it out of the tab order ('none': no attribute). */
const TAB_OFF = 'data-row-tab-off';

/** The table body is one tab stop: only the active row's own controls stay in the tab order. */
function applyRowTabStops(body: HTMLTableSectionElement, active: HTMLTableRowElement | undefined): void {
  for (const node of body.querySelectorAll<HTMLElement>(CELL_FOCUSABLE)) {
    if (node instanceof HTMLTableRowElement || node.hasAttribute('data-row-link')) continue;
    const saved = node.getAttribute(TAB_OFF);
    if (node.closest('tr') === active) {
      if (saved === null) continue;
      node.removeAttribute(TAB_OFF);
      if (saved === 'none') node.removeAttribute('tabindex');
      else node.setAttribute('tabindex', saved);
    } else if (saved === null && node.tabIndex >= 0) {
      node.setAttribute(TAB_OFF, node.getAttribute('tabindex') ?? 'none');
      node.setAttribute('tabindex', '-1');
    }
  }
}

/** A read-only table: sticky header, sortable columns, roving row focus, loading rows in the loaded shape, windowing. */
export function DataTable<T>(props: DataTableProps<T>): ReactElement {
  const rich = useRichMotion();
  const density = props.density ?? 'default';
  const loading = props.loading === true;
  const virtualizeAbove = props.virtualizeAbove ?? 150;

  const [ownSort, setOwnSort] = useState<TableSort | undefined>(props.defaultSort);
  const sort = props.sort ?? ownSort;
  const setSort = (next: TableSort): void => {
    if (props.sort === undefined) setOwnSort(next);
    props.onSortChange?.(next);
  };

  const [tableWidth, setTableWidth] = useState(0);
  const panelRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const node = panelRef.current;
    if (node === null || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width ?? 0;
      setTableWidth(width);
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const columns = visibleColumns(props.columns, tableWidth);
  const free = freeColumn(columns);
  const sortColumn = sort === undefined ? undefined : props.columns.find((column) => column.id === sort.id);
  const rows = useMemo(
    () => (sort === undefined ? [...props.rows] : sortRows(props.rows, sortColumn, sort.direction)),
    [props.rows, sortColumn, sort],
  );
  const { rowKey } = props;
  const keys = useMemo(() => rows.map((row) => rowKey(row)), [rows, rowKey]);

  const showRows = !loading && props.error === undefined && rows.length > 0;
  const windowed = showRows && rows.length > virtualizeAbove;

  const [entrance, setEntrance] = useState<Entrance>({ rows: null, ready: false, seen: new Set(), reveal: new Map() });
  const entranceReady = showRows && rich;
  if (entrance.rows !== props.rows || entrance.ready !== entranceReady) {
    setEntrance(nextEntrance(entrance, props.rows, keys, entranceReady));
  }

  // The active row is a record, not a position: a refetch that prepends or re-sorts rows must not move the tab stop.
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const keyIndex = useMemo(() => new Map(keys.map((key, index) => [key, index])), [keys]);
  const lastActiveIndex = useRef(0);
  const activeIndex =
    (activeKey === null ? undefined : keyIndex.get(activeKey)) ??
    Math.min(lastActiveIndex.current, Math.max(0, rows.length - 1));
  const focusPending = useRef(false);
  /** The focused row was detached in this commit; if focus fell to the page, it goes to the row now at its index. */
  const focusLost = useRef(false);
  const rowNodes = useRef(new Map<string, HTMLTableRowElement>());
  const tbodyRef = useRef<HTMLTableSectionElement | null>(null);

  const [range, setRange] = useState<{ start: number; end: number }>({ start: 0, end: UNMEASURED_WINDOW });
  const rowPx = useRef<number>(ROW_PX[density]);
  const measureWindow = useCallback(() => {
    const body = tbodyRef.current;
    if (body === null) return;
    const root = scrollRootOf(body);
    const viewTop = root?.getBoundingClientRect().top ?? 0;
    const viewHeight = root?.clientHeight ?? globalThis.innerHeight ?? 0;
    const bodyTop = body.getBoundingClientRect().top;
    const first = rowNodes.current.values().next().value;
    if (first !== undefined && first.offsetHeight > 0) rowPx.current = first.offsetHeight;
    if (viewHeight <= 0) return;
    const start = Math.max(0, Math.floor((viewTop - bodyTop) / rowPx.current) - WINDOW_MARGIN);
    const end = Math.ceil((viewTop + viewHeight - bodyTop) / rowPx.current) + WINDOW_MARGIN;
    setRange((current) => (current.start === start && current.end === end ? current : { start, end }));
  }, []);

  useEffect(() => {
    if (!windowed) return;
    const body = tbodyRef.current;
    const root = scrollRootOf(body);
    const target: HTMLElement | Window = root ?? globalThis.window;
    let frame = 0;
    const onScroll = (): void => {
      if (frame !== 0) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        measureWindow();
      });
    };
    measureWindow();
    target.addEventListener('scroll', onScroll, { passive: true });
    globalThis.addEventListener('resize', onScroll);
    return () => {
      if (frame !== 0) cancelAnimationFrame(frame);
      target.removeEventListener('scroll', onScroll);
      globalThis.removeEventListener('resize', onScroll);
    };
  }, [windowed, measureWindow]);

  // The active row (the one tab stop) stays mounted outside the window, between its own spacers.
  const shownIndices: number[] = [];
  if (windowed) {
    const windowStart = Math.min(range.start, rows.length);
    const windowEnd = Math.min(range.end, rows.length);
    if (activeIndex < windowStart) shownIndices.push(activeIndex);
    for (let index = windowStart; index < windowEnd; index += 1) shownIndices.push(index);
    if (activeIndex >= windowEnd && activeIndex < rows.length) shownIndices.push(activeIndex);
  } else {
    for (let index = 0; index < rows.length; index += 1) shownIndices.push(index);
  }

  useLayoutEffect(() => {
    lastActiveIndex.current = activeIndex;
    const lost = focusLost.current;
    focusLost.current = false;
    const dropped = lost && (document.activeElement === null || document.activeElement === document.body);
    if (!focusPending.current && !dropped) return;
    focusPending.current = false;
    const key = keys[activeIndex];
    if (key !== undefined) rowNodes.current.get(key)?.focus();
  });

  const activeRowKey = keys[activeIndex];
  useLayoutEffect(() => {
    const body = tbodyRef.current;
    if (body === null) return;
    const apply = (): void =>
      applyRowTabStops(body, activeRowKey === undefined ? undefined : rowNodes.current.get(activeRowKey));
    apply();
    if (typeof MutationObserver === 'undefined') return;
    // A cell can swap its controls without this table rendering (a confirm that resets its trigger).
    const observer = new MutationObserver(apply);
    observer.observe(body, { childList: true, subtree: true });
    return () => observer.disconnect();
  });

  const positions = useRef(new Map<string, number>());
  const lastSort = useRef<string>(sort === undefined ? '' : `${sort.id}:${sort.direction}`);
  useLayoutEffect(() => {
    const sortKey = sort === undefined ? '' : `${sort.id}:${sort.direction}`;
    const sorted = sortKey !== lastSort.current;
    lastSort.current = sortKey;
    const body = tbodyRef.current;
    const canTravel = rich && showRows && !windowed && rows.length <= TRAVEL_ROWS;
    if (!canTravel || body === null) {
      positions.current.clear();
      return;
    }
    const next = new Map<string, number>();
    for (const [key, node] of rowNodes.current) next.set(key, node.offsetTop);
    const before = positions.current;
    positions.current = next;
    if (!sorted || before.size === 0) return;
    const moves: Promise<unknown>[] = [];
    for (const [key, top] of next) {
      const old = before.get(key);
      const node = rowNodes.current.get(key);
      if (old === undefined || node === undefined || old === top) continue;
      moves.push(Promise.resolve(animate(node, { y: [old - top, 0] }, motionX.spring.travel)));
    }
    if (moves.length === 0) return;
    body.setAttribute('data-resorting', '');
    void Promise.all(moves).then(() => body.removeAttribute('data-resorting'));
  });

  const moveTo = (index: number): void => {
    if (rows.length === 0) return;
    const next = Math.max(0, Math.min(rows.length - 1, index));
    const key = keys[next];
    if (key === undefined) return;
    if (next === activeIndex) {
      // React skips a same-value update, so a pending flag would be consumed by an unrelated later render.
      rowNodes.current.get(key)?.focus();
      if (activeKey !== key) setActiveKey(key);
      return;
    }
    focusPending.current = true;
    setActiveKey(key);
  };

  const onBodyKeyDown = (event: KeyboardEvent<HTMLTableSectionElement>): void => {
    const target = event.target;
    if (!(target instanceof HTMLTableRowElement)) return;
    switch (event.key) {
      case 'ArrowDown':
        moveTo(activeIndex + 1);
        break;
      case 'ArrowUp':
        moveTo(activeIndex - 1);
        break;
      case 'Home':
        moveTo(0);
        break;
      case 'End':
        moveTo(rows.length - 1);
        break;
      case 'Enter': {
        const link = target.querySelector<HTMLAnchorElement>('a[data-row-link]');
        if (link === null) return;
        link.click();
        break;
      }
      default:
        return;
    }
    event.preventDefault();
  };

  const toggleSort = (column: DataColumn<T>): void => {
    if (sort !== undefined && sort.id === column.id) {
      setSort({ id: column.id, direction: sort.direction === 'asc' ? 'desc' : 'asc' });
    } else {
      setSort({ id: column.id, direction: column.align === 'end' ? 'desc' : 'asc' });
    }
  };

  const span = Math.max(1, columns.length);
  const loadingCount = props.loadingRows ?? 6;

  let body: ReactNode;
  if (loading) {
    const loadingRows: ReactElement[] = [];
    for (let i = 0; i < loadingCount; i += 1) {
      loadingRows.push(
        <tr key={`loading-${i}`} className={styles.row}>
          {columns.map((column) => (
            <td key={column.id} className={styles.cell} data-align={column.align ?? 'start'}>
              <Placeholder width={column.placeholderWidth ?? '60%'} inline />
            </td>
          ))}
        </tr>,
      );
    }
    body = loadingRows;
  } else if (props.error !== undefined) {
    const retry = props.error.retry;
    body = (
      <tr className={styles.stateRow}>
        <td colSpan={span} className={styles.stateCell}>
          <Alert tone="danger" title={props.error.message}>
            {retry !== undefined ? (
              <Button variant="secondary" size="sm" onClick={retry}>
                Try again
              </Button>
            ) : null}
          </Alert>
        </td>
      </tr>
    );
  } else if (rows.length === 0) {
    body =
      props.empty === undefined ? null : (
        <tr className={styles.stateRow}>
          <td colSpan={span} className={styles.stateCell}>
            <EmptyState title={props.empty.title} description={props.empty.description} action={props.empty.action} />
          </td>
        </tr>
      );
  } else {
    const rendered: ReactNode[] = [];
    const pushSpacer = (from: number, to: number): void => {
      rendered.push(
        <tr
          key={`space-${from}`}
          aria-hidden="true"
          className={styles.spacer}
          style={{ height: (to - from) * rowPx.current }}
        />,
      );
    };
    let cursor = 0;
    for (const index of shownIndices) {
      if (index > cursor) pushSpacer(cursor, index);
      cursor = index + 1;
      const row = rows[index];
      const key = keys[index];
      const to = props.rowTo?.(row);
      const step = entrance.reveal.get(key);
      const revealStyle = step === undefined ? undefined : ({ '--reveal-step': step } as CSSProperties);
      rendered.push(
        <MotionGate key={key} open={!windowed && index < RICH_ROWS}>
          <tr
            {...props.rowAttrs?.(row)}
            ref={(node) => {
              if (node !== null) {
                rowNodes.current.set(key, node);
                return;
              }
              const old = rowNodes.current.get(key);
              if (old !== undefined && old === document.activeElement) focusLost.current = true;
              rowNodes.current.delete(key);
            }}
            className={styles.row}
            data-linked={to !== undefined ? '' : undefined}
            data-reveal={rich && step !== undefined ? '' : undefined}
            style={revealStyle}
            tabIndex={index === activeIndex ? 0 : -1}
            aria-rowindex={windowed ? index + 2 : undefined}
            onFocus={() => {
              if (activeKey !== key) setActiveKey(key);
            }}
            onPointerMove={
              to === undefined
                ? undefined
                : (event) => {
                    const link = event.currentTarget.querySelector<HTMLElement>('[data-row-link]');
                    if (link !== null) forwardRowTitle(event.currentTarget, link, event.clientX, event.clientY);
                  }
            }
            onAnimationEnd={(event) => {
              if (event.target !== event.currentTarget || step === undefined) return;
              setEntrance((current) => {
                if (!current.reveal.has(key)) return current;
                const reveal = new Map(current.reveal);
                reveal.delete(key);
                return { ...current, reveal };
              });
            }}
          >
            {columns.map((column, columnIndex) => {
              const content = column.cell(row);
              return (
                <td
                  key={column.id}
                  className={styles.cell}
                  data-align={column.align ?? 'start'}
                  data-truncate={column.truncate === true ? '' : undefined}
                >
                  {columnIndex === 0 && to !== undefined ? (
                    <Link className={styles.rowLink} to={to} tabIndex={-1} data-row-link="">
                      {content}
                    </Link>
                  ) : (
                    content
                  )}
                </td>
              );
            })}
          </tr>
        </MotionGate>,
      );
    }
    if (cursor < rows.length) pushSpacer(cursor, rows.length);
    body = rendered;
  }

  const tableContent = (
    <>
      <table
        className={styles.table}
        data-density={density}
        aria-busy={loading ? true : undefined}
        aria-rowcount={windowed ? rows.length + 1 : undefined}
      >
        <caption className={styles.caption}>{props.caption}</caption>
        <colgroup>
          {columns.map((column, index) => (
            <col key={column.id} style={index === free ? undefined : columnStyle(column)} />
          ))}
        </colgroup>
        <thead>
          <tr aria-rowindex={windowed ? 1 : undefined}>
            {columns.map((column) => {
              const sorted = sort !== undefined && sort.id === column.id;
              const ariaSort = sorted ? (sort.direction === 'asc' ? 'ascending' : 'descending') : undefined;
              return (
                <th
                  key={column.id}
                  scope="col"
                  className={styles.head}
                  data-align={column.align ?? 'start'}
                  aria-sort={ariaSort}
                  title={column.headerTitle}
                >
                  {column.sortValue !== undefined ? (
                    <button
                      type="button"
                      className={styles.sortButton}
                      data-sorted={sorted ? sort.direction : undefined}
                      onClick={() => toggleSort(column)}
                    >
                      <span>{column.header}</span>
                      <ArrowUp className={styles.arrow} size={12} strokeWidth={1.75} aria-hidden="true" />
                    </button>
                  ) : (
                    column.header
                  )}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody ref={tbodyRef} className={styles.body} onKeyDown={onBodyKeyDown}>
          {body}
        </tbody>
      </table>
      {props.footer !== undefined ? <div className={styles.footer}>{props.footer}</div> : null}
    </>
  );
  return (
    <div
      ref={panelRef}
      className={props.className === undefined ? styles.root : `${styles.root} ${props.className}`}
      data-frame={props.frame === false ? 'none' : undefined}
    >
      {props.frame === false ? tableContent : <Panel padding="none">{tableContent}</Panel>}
    </div>
  );
}

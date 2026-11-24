import { Check, ChevronRight } from 'lucide-react';
import {
  type ButtonHTMLAttributes,
  createContext,
  forwardRef,
  type KeyboardEvent,
  type PointerEvent,
  type ReactElement,
  type ReactNode,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
} from 'react';

import { Popover, PopoverContent, PopoverTrigger } from '../../arc/popover/popover';
import styles from './menu.module.css';

/** How long the pointer rests on a submenu row before its panel opens. */
export const SUB_OPEN_MS = 100;
/** The menu panel's padding plus its border: the submenu's bottom edge lines up with the menu's. */
const PANEL_INSET = 5;
const SUB_GAP = 4;

export interface MenuProps {
  /** One element that takes a ref and button props. */
  trigger: ReactElement;
  /** Name of the menu. */
  label: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  side?: 'top' | 'bottom';
  align?: 'start' | 'end';
  children: ReactNode;
}

interface MenuRowProps {
  icon?: ReactNode;
  label: string;
  /** Right slot: a state or a shortcut. */
  end?: ReactNode;
  /** A 6 px dot before `end`, and `end` in the foreground colour. */
  endTone?: 'warning' | 'danger';
  /** A second line. */
  note?: string;
  /** Stays focusable (its note is read), does not act. */
  disabled?: boolean;
  /** Accessible name, default `label`. */
  name?: string;
  /** The row's shortcut as `aria-keyshortcuts`, e.g. `Meta+U`; `end` draws it. */
  keys?: string;
}

export interface MenuItemProps extends MenuRowProps {
  onSelect: () => void;
}

export interface MenuRadioProps extends MenuRowProps {
  /** Draws the check mark and sets `aria-checked`. */
  checked: boolean;
  onSelect: () => void;
}

export interface MenuSubProps extends MenuRowProps {
  /** Name of the panel, default `label`. */
  panelLabel?: string;
  children: ReactNode;
}

interface MenuContextValue {
  open: boolean;
  sub: string | null;
  setSub: (id: string | null) => void;
  close: () => void;
  /** False for a pointermove at the last seen coordinates (a scroll or a layout change, not the hand). */
  moved: (event: PointerEvent) => boolean;
  /** The menu node, to tell a row of this menu from anything else. */
  panel: () => HTMLElement | null;
}

const MenuContext = createContext<MenuContextValue | null>(null);

function useMenu(): MenuContextValue {
  const menu = useContext(MenuContext);
  if (menu === null) throw new Error('Menu rows render inside a Menu');
  return menu;
}

/** The submenu panel a row renders in, or null for a row of the menu itself. */
const SubContext = createContext<{ label: string } | null>(null);

function rowsOf(scope: HTMLElement): HTMLElement[] {
  return [...scope.querySelectorAll<HTMLElement>('[role="menuitem"], [role="menuitemradio"]')].filter(
    (row) => row.closest('[role="menu"]') === scope,
  );
}

/** The menu keys inside `scope`: the row to focus, or null. Tab is blocked. */
function rowKeys(event: KeyboardEvent<HTMLElement>, scope: HTMLElement): HTMLElement | null {
  if (event.key === 'Tab') {
    event.preventDefault();
    return null;
  }
  const rows = rowsOf(scope);
  const count = rows.length;
  if (count === 0) return null;
  const at = rows.indexOf(document.activeElement as HTMLElement);
  let next: number | null = null;
  if (event.key === 'ArrowDown') next = at < 0 ? 0 : (at + 1) % count;
  else if (event.key === 'ArrowUp') next = at < 0 ? count - 1 : (at - 1 + count) % count;
  else if (event.key === 'Home') next = 0;
  else if (event.key === 'End') next = count - 1;
  else if (event.key.length === 1 && event.key !== ' ' && !event.ctrlKey && !event.metaKey && !event.altKey) {
    const letter = event.key.toLowerCase();
    for (let step = 1; step <= count; step += 1) {
      const index = (at + step + count) % count;
      if ((rows[index]!.dataset.label ?? '').startsWith(letter)) {
        next = index;
        break;
      }
    }
  }
  if (next === null) return null;
  event.preventDefault();
  return rows[next]!;
}

/** A menu that opens from a trigger: `role="menu"` rows with real roving focus, and submenu panels beside them. */
export function Menu({
  trigger,
  label,
  open,
  onOpenChange,
  side = 'top',
  align = 'start',
  children,
}: MenuProps): ReactElement {
  const [sub, setSub] = useState<string | null>(null);
  const panel = useRef<HTMLDivElement>(null);
  const pointer = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    setSub(null);
    pointer.current = null;
  }, [open]);

  const context: MenuContextValue = {
    open,
    sub,
    setSub,
    close: () => onOpenChange(false),
    moved: (event) => {
      const last = pointer.current;
      if (last !== null && last.x === event.clientX && last.y === event.clientY) return false;
      pointer.current = { x: event.clientX, y: event.clientY };
      return true;
    },
    panel: () => panel.current,
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    // Keys from a submenu panel reach this node through the React tree, not the DOM tree.
    if (!event.currentTarget.contains(event.target as Node)) return;
    const row = rowKeys(event, event.currentTarget);
    if (row === null) return;
    row.focus();
    if (sub !== null && row.getAttribute('aria-expanded') !== 'true') setSub(null);
  };

  return (
    <MenuContext.Provider value={context}>
      <Popover open={open} onOpenChange={onOpenChange}>
        <PopoverTrigger asChild aria-haspopup="menu">
          {trigger}
        </PopoverTrigger>
        <PopoverContent
          ref={panel}
          role="menu"
          aria-label={label}
          side={side}
          align={align}
          className={styles.panel}
          onOpenAutoFocus={(event) => {
            event.preventDefault();
            const first = panel.current === null ? undefined : rowsOf(panel.current)[0];
            first?.focus();
          }}
          onKeyDown={onKeyDown}
        >
          {children}
        </PopoverContent>
      </Popover>
    </MenuContext.Provider>
  );
}

interface RowProps
  extends MenuRowProps, Omit<ButtonHTMLAttributes<HTMLButtonElement>, 'children' | 'disabled' | 'role'> {
  role?: 'menuitem' | 'menuitemradio';
  /** A radio row's state. */
  checked?: boolean;
  /** The submenu this row opens, for a row of MenuSub. */
  subId?: string;
  /** Runs once the pointer rests on the row for SUB_OPEN_MS. */
  onRest?: () => void;
}

const Row = forwardRef<HTMLButtonElement, RowProps>(function Row(
  {
    icon,
    label,
    end,
    endTone,
    note,
    disabled,
    name,
    keys,
    role = 'menuitem',
    checked,
    subId,
    onRest,
    onPointerMove,
    onPointerLeave,
    className,
    ...rest
  },
  ref,
) {
  const menu = useMenu();
  const inSub = useContext(SubContext) !== null;
  const noteId = useId();
  const timer = useRef<number | null>(null);
  const clear = (): void => {
    if (timer.current === null) return;
    window.clearTimeout(timer.current);
    timer.current = null;
  };
  useEffect(() => clear, []);
  useEffect(() => {
    if (!menu.open) clear();
  }, [menu.open]);

  return (
    <button
      type="button"
      role={role}
      tabIndex={-1}
      data-label={label.toLowerCase()}
      aria-label={name ?? label}
      aria-checked={role === 'menuitemradio' ? checked === true : undefined}
      aria-disabled={disabled === true ? true : undefined}
      aria-describedby={note !== undefined ? noteId : undefined}
      aria-keyshortcuts={keys}
      {...rest}
      // A popover trigger passes its own class: the row's class must stay beside it.
      className={className === undefined ? styles.row : `${styles.row} ${className}`}
      ref={ref}
      onPointerMove={(event) => {
        onPointerMove?.(event);
        if (event.pointerType !== 'mouse' || !menu.moved(event)) return;
        const row = event.currentTarget;
        if (document.activeElement !== row) row.focus({ preventScroll: true });
        // A row inside a panel must not close the panel it sits in.
        if (!inSub && menu.sub !== null && menu.sub !== subId) menu.setSub(null);
        if (onRest !== undefined && timer.current === null) {
          timer.current = window.setTimeout(() => {
            timer.current = null;
            onRest();
          }, SUB_OPEN_MS);
        }
      }}
      onPointerLeave={(event) => {
        onPointerLeave?.(event);
        clear();
      }}
    >
      {icon !== undefined ? (
        <span className={styles.icon} aria-hidden="true">
          {icon}
        </span>
      ) : null}
      <span className={styles.text}>
        <span className={styles.label}>{label}</span>
        {note !== undefined ? (
          <span className={styles.note} id={noteId}>
            {note}
          </span>
        ) : null}
      </span>
      {end !== undefined ? (
        <span className={styles.end} data-tone={endTone}>
          {endTone !== undefined ? <span className={styles.dot} aria-hidden="true" /> : null}
          {end}
        </span>
      ) : null}
      {checked === true ? <Check className={styles.check} size={14} strokeWidth={1.75} aria-hidden="true" /> : null}
      {subId !== undefined ? (
        <ChevronRight className={styles.chev} size={12} strokeWidth={1.75} aria-hidden="true" />
      ) : null}
    </button>
  );
});

/** A row that acts and closes the menu. */
export function MenuItem({ onSelect, ...row }: MenuItemProps): ReactElement {
  const menu = useMenu();
  return (
    <Row
      {...row}
      onClick={() => {
        if (row.disabled === true) return;
        onSelect();
        menu.close();
      }}
    />
  );
}

/** A choice of a MenuGroup: a click picks it and the menu stays open; the check mark follows `checked`. */
export function MenuRadio({ checked, onSelect, ...row }: MenuRadioProps): ReactElement {
  return (
    <Row
      {...row}
      role="menuitemradio"
      checked={checked}
      onClick={() => {
        if (row.disabled !== true) onSelect();
      }}
    />
  );
}

/** Rows inside a MenuSub panel with the menu's own keys; named by the panel. Other panel content sits outside it. */
export function MenuList({ children }: { children: ReactNode }): ReactElement {
  const sub = useContext(SubContext);
  if (sub === null) throw new Error('A MenuList renders inside a MenuSub panel');
  return (
    <div
      role="menu"
      aria-label={sub.label}
      className={styles.list}
      onKeyDown={(event) => rowKeys(event, event.currentTarget)?.focus()}
    >
      {children}
    </div>
  );
}

export function MenuGroup({ label, children }: { label: string; children: ReactNode }): ReactElement {
  return (
    <div role="group" aria-label={label} className={styles.group}>
      {children}
    </div>
  );
}

export function MenuSeparator(): ReactElement {
  return <div role="separator" className={styles.separator} />;
}

/** A row that opens a panel beside it: pointer rest, click, Enter, Space or the right arrow; the left arrow closes it. */
export function MenuSub({ panelLabel, children, ...row }: MenuSubProps): ReactElement {
  const menu = useMenu();
  const id = useId();
  const open = menu.open && menu.sub === id;
  const rowRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  /** The next open moves focus into the panel (keys, click); a pointer rest leaves it on the row. */
  const enter = useRef(false);

  /** The checked choice first. Radix's own open focus skips links, and a free panel's one control is often a link. */
  const focusPanel = (): void => {
    const node = panelRef.current;
    const target =
      node?.querySelector<HTMLElement>('[role="menuitemradio"][aria-checked="true"]') ??
      node?.querySelector<HTMLElement>('a[href], button:not(:disabled)') ??
      node;
    target?.focus();
  };
  const label = panelLabel ?? row.label;

  const enterSub = (): void => {
    if (open) {
      focusPanel();
      return;
    }
    enter.current = true;
    menu.setSub(id);
  };

  return (
    <Popover open={open} onOpenChange={(next) => menu.setSub(next ? id : null)}>
      <PopoverTrigger asChild>
        <Row
          {...row}
          ref={rowRef}
          subId={id}
          onRest={open || row.disabled === true ? undefined : () => menu.setSub(id)}
          onClick={(event) => {
            event.preventDefault();
            if (row.disabled !== true) enterSub();
          }}
          onKeyDown={(event) => {
            if (event.key !== 'ArrowRight') return;
            event.preventDefault();
            if (row.disabled !== true) enterSub();
          }}
        />
      </PopoverTrigger>
      <PopoverContent
        ref={panelRef}
        aria-label={label}
        side="right"
        align="end"
        sideOffset={PANEL_INSET + SUB_GAP}
        alignOffset={-PANEL_INSET}
        className={styles.sub}
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          if (enter.current) focusPanel();
          enter.current = false;
        }}
        onCloseAutoFocus={(event) => {
          // Focus already on another row of the menu (a pointer or an arrow moved it) stays there.
          const active = document.activeElement;
          const node = menu.panel();
          if (active !== null && node !== null && node.contains(active)) event.preventDefault();
        }}
        onKeyDown={(event) => {
          if (event.key !== 'ArrowLeft') return;
          event.preventDefault();
          rowRef.current?.focus();
          menu.setSub(null);
        }}
      >
        <SubContext.Provider value={{ label }}>{children}</SubContext.Provider>
      </PopoverContent>
    </Popover>
  );
}

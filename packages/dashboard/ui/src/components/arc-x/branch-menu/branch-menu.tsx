import { Check, FolderOpen, GitBranch, GitCommitHorizontal } from 'lucide-react';
import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
  type ReactElement,
} from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import { folderName } from '../../../lib/permission-text';
import { Popover, PopoverContent, PopoverTrigger } from '../../arc/popover/popover';
import { SearchField } from '../../arc/search-field/search-field';
import { Tooltip } from '../../arc/tooltip/tooltip';
import { MorphLoader } from '../morph-loader/morph-loader';
import { ToolChip } from '../tool-chip/tool-chip';
import styles from './branch-menu.module.css';

export interface BranchMenuItem {
  name: string;
  current: boolean;
  /** The other worktree that has it checked out: the row is off and says where. */
  elsewhere: string | null;
}

export interface BranchMenuProps {
  /** The chip's text: the branch, or the short hash of a detached HEAD. */
  label: string;
  detached: boolean;
  /** The chip's accessible name, and its tip while it opens. */
  name: string;
  /** Why the chip does not open: its tip, and the menu's note if it is open. Null: it opens. */
  off: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The work tree's top folder. */
  top: string;
  /** Current first, then newest. */
  branches: readonly BranchMenuItem[];
  /** Every local branch, `branches` being the newest of them. */
  total: number;
  /** Why every row is off (a Work conversation runs in the folder), or null. */
  busy: string | null;
  /** The branch being switched to: its row shows the loader, the others are off. */
  pending: string | null;
  /** Why the last pick did not switch. */
  alert: { lead: string; text: string } | null;
  onPick: (name: string) => void;
  /** `data-tray-part` of the chip. */
  part?: string;
}

const iconSize = { size: 14, strokeWidth: 1.75 } as const;

/**
 * The branch chip and its menu: a search field that keeps focus (↑ ↓ Home End move, Enter picks, Esc closes to the
 * chip), the work tree, the local branches as a listbox, then one muted line or an alert.
 */
export function BranchMenu(props: BranchMenuProps): ReactElement {
  const id = useId();
  const listId = `${id}-list`;
  const optionId = (index: number): string => `${id}-o${String(index)}`;
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const lastPointer = useRef({ x: -1, y: -1 });
  // A pick that switched leaves focus to the page; Radix's return to the chip would undo it.
  const picked = useRef(false);
  const folded = query.trim().toLowerCase();
  const rows = useMemo(
    () => (folded === '' ? props.branches : props.branches.filter((row) => row.name.toLowerCase().includes(folded))),
    [props.branches, folded],
  );
  const at = rows.length === 0 ? -1 : Math.min(active, rows.length - 1);
  const allOff = props.busy !== null || props.off !== null || props.pending !== null;
  const rowOff = (row: BranchMenuItem): boolean => allOff || row.elsewhere !== null;

  const note = props.off ?? props.busy;
  // A pick that did not switch keeps the menu open: Esc then returns to the chip.
  useEffect(() => {
    if (props.alert !== null || note !== null) picked.current = false;
  }, [props.alert, note]);

  useEffect(() => {
    if (!props.open || at < 0) return;
    document.getElementById(optionId(at))?.scrollIntoView({ block: 'nearest' });
  });

  const pick = (row: BranchMenuItem | undefined): void => {
    if (row === undefined || rowOff(row)) return;
    picked.current = true;
    if (row.current) {
      props.onOpenChange(false);
      return;
    }
    props.onPick(row.name);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>): void => {
    if (event.nativeEvent.isComposing) return;
    const last = rows.length - 1;
    let next: number | null = null;
    if (event.key === 'ArrowDown') next = Math.min(last, at + 1);
    else if (event.key === 'ArrowUp') next = Math.max(0, at - 1);
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = last;
    else if (event.key === 'Enter') {
      event.preventDefault();
      pick(rows[at]);
      return;
    }
    if (next === null) return;
    event.preventDefault();
    if (last >= 0) setActive(next);
  };

  const pointerMoved = (event: PointerEvent): boolean => {
    if (event.clientX === lastPointer.current.x && event.clientY === lastPointer.current.y) return false;
    lastPointer.current = { x: event.clientX, y: event.clientY };
    return true;
  };

  const line =
    folded !== '' && rows.length === 0
      ? fill(CHAT.branch.noMatch, { query: query.trim() })
      : folded === '' && props.branches.length <= 1
        ? CHAT.branch.noOther
        : folded === '' && props.total > props.branches.length
          ? fill(CHAT.branch.capped, { n: props.branches.length, total: props.total })
          : null;

  return (
    <Popover
      open={props.open}
      onOpenChange={(next) => {
        if (next && props.off !== null) return;
        if (next) {
          picked.current = false;
          setQuery('');
          setActive(0);
        }
        props.onOpenChange(next);
      }}
    >
      <Tooltip content={props.off ?? props.name}>
        <PopoverTrigger asChild>
          <ToolChip
            icon={props.detached ? <GitCommitHorizontal {...iconSize} /> : <GitBranch {...iconSize} />}
            label={props.label}
            aria-label={props.name}
            aria-disabled={props.off !== null ? true : undefined}
            aria-haspopup="listbox"
            data-tray-part={props.part}
            onClick={(event) => {
              if (props.off !== null) event.preventDefault();
            }}
          />
        </PopoverTrigger>
      </Tooltip>
      <PopoverContent
        side="top"
        align="start"
        className={styles.panel}
        aria-label={CHAT.branch.list}
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          inputRef.current?.focus();
        }}
        onCloseAutoFocus={(event) => {
          if (picked.current) event.preventDefault();
        }}
      >
        <div className={styles.search}>
          <SearchField
            ref={inputRef}
            label={CHAT.branch.search}
            placeholder={CHAT.branch.search}
            value={query}
            onValueChange={(value) => {
              setQuery(value);
              setActive(0);
            }}
            role="combobox"
            aria-expanded="true"
            aria-autocomplete="list"
            aria-controls={listId}
            aria-activedescendant={at >= 0 ? optionId(at) : undefined}
            autoComplete="off"
            spellCheck={false}
            onKeyDown={onKeyDown}
          />
        </div>
        <div className={styles.header} title={props.top}>
          <FolderOpen {...iconSize} aria-hidden="true" />
          <span className={styles.headerName}>{folderName(props.top)}</span>
        </div>
        {note !== null ? <p className={styles.note}>{note}</p> : null}
        <div id={listId} className={styles.list} role="listbox" aria-label={CHAT.branch.list}>
          {rows.map((row, index) => {
            const off = rowOff(row);
            const noteId = row.elsewhere !== null ? `${id}-n${String(index)}` : undefined;
            return (
              <div
                key={row.name}
                id={optionId(index)}
                className={styles.row}
                role="option"
                aria-selected={row.current}
                aria-disabled={off ? true : undefined}
                aria-describedby={noteId}
                data-active={index === at ? '' : undefined}
                title={row.name}
                onMouseDown={(event) => event.preventDefault()}
                onPointerMove={(event) => {
                  if (pointerMoved(event) && index !== at) setActive(index);
                }}
                onClick={() => pick(row)}
              >
                <span className={styles.rowText}>
                  <span className={styles.rowName}>{row.name}</span>
                  {row.elsewhere !== null ? (
                    <span id={noteId} className={styles.rowNote}>
                      {fill(CHAT.branch.elsewhere, { folder: folderName(row.elsewhere) })}
                    </span>
                  ) : null}
                </span>
                <span className={styles.rowEnd} aria-hidden="true">
                  {props.pending === row.name ? (
                    <MorphLoader size={14} status="loading" />
                  ) : row.current ? (
                    <Check {...iconSize} />
                  ) : null}
                </span>
              </div>
            );
          })}
        </div>
        {props.alert !== null ? (
          <div className={styles.alert} role="alert">
            <span>{props.alert.lead}</span>
            {props.alert.text !== '' ? <p className={styles.alertText}>{props.alert.text}</p> : null}
          </div>
        ) : line !== null ? (
          <p className={styles.line}>{line}</p>
        ) : null}
      </PopoverContent>
    </Popover>
  );
}

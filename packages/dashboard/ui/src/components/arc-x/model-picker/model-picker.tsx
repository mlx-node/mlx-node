import { Check, ChevronDown, TriangleAlert } from 'lucide-react';
import {
  useCallback,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactElement,
  type ReactNode,
} from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import { Popover, PopoverContent, PopoverTrigger } from '../../arc/popover/popover';
import { Tooltip } from '../../arc/tooltip/tooltip';
import { LiveLabel } from '../live-label/live-label';
import { MiddleText } from '../middle-text/middle-text';
import { ModelBadge } from '../model-badge/model-badge';
import { MorphLoader } from '../morph-loader/morph-loader';
import styles from './model-picker.module.css';

export interface ModelOption {
  id: string;
  state: 'loaded' | 'loading' | 'idle';
  /** "32K context", "Q4", "2.4 GB": joined with " · " after the state word. */
  facts: readonly string[];
  /** e.g. "small for Work"; drawn with a warning icon. */
  warning?: string;
}

export interface ModelPickerProps {
  value: string | null;
  options: readonly ModelOption[];
  onValueChange: (id: string) => void;
  disabled?: boolean;
  disabledReason?: string;
  /** A change is being saved. */
  working?: boolean;
  /** Under the list, when there are two or more options. */
  footer?: string;
  /** Shown instead of the footer while one or more rows are not loaded. */
  swapWarning?: string;
  /** The last row: what the engine holds now, and an action (Unload). Outside the listbox: a normal tab stop. */
  memory?: { text: string; action: ReactNode };
}

const STATE_WORD: Record<ModelOption['state'], string> = {
  loaded: CHAT.modelLoaded,
  loading: CHAT.modelLoading,
  idle: CHAT.modelIdle,
};

/** Scrolls `list` by the least amount that shows all of its `index`th row; rows are positioned against the list. */
function reveal(list: HTMLElement, index: number): void {
  const row = list.children.item(index);
  if (!(row instanceof HTMLElement)) return;
  const top = row.offsetTop;
  const bottom = top + row.offsetHeight;
  if (top < list.scrollTop) list.scrollTop = top;
  else if (bottom > list.scrollTop + list.clientHeight) list.scrollTop = bottom - list.clientHeight;
}

/** The model of the next message: a pill that opens a list with each model's state, context and size. */
export function ModelPicker(props: ModelPickerProps): ReactElement {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const activeRef = useRef(0);
  const pointer = useRef<{ x: number; y: number } | null>(null);
  const listId = useId();
  const rowId = useId();
  const listRef = useRef<HTMLDivElement | null>(null);
  const [list, setList] = useState<HTMLDivElement | null>(null);
  const attachList = useCallback((node: HTMLDivElement | null) => {
    listRef.current = node;
    setList(node);
  }, []);
  const empty = props.options.length === 0 || props.value === null;
  const disabled = props.disabled === true || empty;
  const selected = props.options.findIndex((option) => option.id === props.value);
  const highlighted = props.options[active];
  const selectedState = selected === -1 ? null : props.options[selected]!.state;
  // The footer must not follow the active row. The panel hangs from its bottom edge: a change of the footer height
  // moves the rows under a resting pointer, the active row changes, and the footer flips back.
  const swap =
    props.swapWarning !== undefined && props.options.some((option) => option.state !== 'loaded')
      ? props.swapWarning
      : null;
  const footer = swap ?? (props.options.length >= 2 ? (props.footer ?? null) : null);

  const activate = (index: number): void => {
    activeRef.current = index;
    setActive(index);
  };

  const openList = (): void => {
    activate(selected === -1 ? 0 : selected);
    setOpen(true);
  };

  useLayoutEffect(() => {
    if (list === null) return;
    reveal(list, activeRef.current);
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(() => reveal(list, activeRef.current));
    observer.observe(list);
    return () => observer.disconnect();
  }, [list]);

  const pick = (index: number): void => {
    const option = props.options[index];
    if (option === undefined) return;
    setOpen(false);
    if (option.id !== props.value) props.onValueChange(option.id);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    const last = props.options.length - 1;
    let next = active;
    if (event.key === 'ArrowDown') next = Math.min(last, active + 1);
    else if (event.key === 'ArrowUp') next = Math.max(0, active - 1);
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = last;
    else if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      pick(active);
      return;
    } else return;
    event.preventDefault();
    activate(next);
    if (list !== null) reveal(list, next);
  };

  const trigger = (
    <button
      type="button"
      className={styles.trigger}
      disabled={disabled}
      aria-haspopup="listbox"
      aria-expanded={open}
      data-state={selectedState ?? undefined}
      aria-label={
        empty
          ? CHAT.noModel
          : selectedState === null
            ? `${CHAT.model}: ${props.value}`
            : fill(CHAT.pillName, { model: props.value ?? '', state: CHAT.pillStates[selectedState] })
      }
      onKeyDown={(event) => {
        if (event.key === 'ArrowDown' && !open) {
          event.preventDefault();
          openList();
        }
      }}
    >
      {empty ? (
        <span className={styles.none}>{CHAT.noModel}</span>
      ) : (
        <span key={props.value} className={styles.name}>
          <ModelBadge name={props.value ?? ''} size="sm" showLogo maxChars={22} />
        </span>
      )}
      <span className={styles.chevron} aria-hidden="true">
        {props.working === true ? (
          <MorphLoader size={14} status="loading" />
        ) : (
          <ChevronDown size={12} strokeWidth={1.75} />
        )}
      </span>
    </button>
  );

  return (
    <Popover
      open={open && !disabled}
      onOpenChange={(next) => {
        if (next) openList();
        else setOpen(false);
      }}
    >
      {props.disabled === true && props.disabledReason !== undefined ? (
        <Tooltip content={props.disabledReason}>
          <span className={styles.tipTarget}>{trigger}</span>
        </Tooltip>
      ) : (
        <PopoverTrigger asChild>{trigger}</PopoverTrigger>
      )}
      <PopoverContent
        side="top"
        align="start"
        className={styles.panel}
        aria-label={CHAT.model}
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          listRef.current?.focus();
        }}
      >
        <p className={styles.heading} id={`${listId}-label`}>
          {CHAT.model}
        </p>
        <div
          ref={attachList}
          id={listId}
          className={styles.list}
          role="listbox"
          tabIndex={0}
          aria-labelledby={`${listId}-label`}
          aria-activedescendant={highlighted !== undefined ? `${rowId}-${active}` : undefined}
          onKeyDown={onKeyDown}
        >
          {props.options.map((option, index) => (
            <div
              key={option.id}
              id={`${rowId}-${index}`}
              className={styles.row}
              role="option"
              aria-selected={option.id === props.value}
              data-active={index === active ? '' : undefined}
              onPointerMove={(event) => {
                const last = pointer.current;
                if (last !== null && last.x === event.clientX && last.y === event.clientY) return;
                pointer.current = { x: event.clientX, y: event.clientY };
                activate(index);
              }}
              onClick={() => pick(index)}
            >
              <span className={styles.rowText}>
                <span className={styles.rowName}>
                  <MiddleText text={option.id} />
                </span>
                <span className={styles.rowFacts}>
                  {option.state === 'loading' ? (
                    <LiveLabel active>{STATE_WORD.loading}</LiveLabel>
                  ) : (
                    STATE_WORD[option.state]
                  )}
                  {option.facts.map((fact) => ` · ${fact}`).join('')}
                  {option.warning !== undefined ? (
                    <span className={styles.warning}>
                      {' · '}
                      <TriangleAlert size={12} strokeWidth={1.75} aria-hidden="true" />
                      {option.warning}
                    </span>
                  ) : null}
                </span>
              </span>
              {option.id === props.value ? (
                <Check className={styles.check} size={14} strokeWidth={1.75} aria-hidden="true" />
              ) : null}
            </div>
          ))}
        </div>
        {footer !== null ? (
          <p className={styles.footer} data-warning={swap !== null ? '' : undefined}>
            {swap !== null ? <TriangleAlert size={12} strokeWidth={1.75} aria-hidden="true" /> : null}
            <span>{footer}</span>
          </p>
        ) : null}
        {props.memory !== undefined ? (
          <div className={styles.memory}>
            <span className={styles.memoryText}>{props.memory.text}</span>
            {props.memory.action}
          </div>
        ) : null}
      </PopoverContent>
    </Popover>
  );
}

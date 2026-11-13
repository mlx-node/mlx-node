import { Check, ChevronDown, ChevronRight, TriangleAlert } from 'lucide-react';
import {
  type CSSProperties,
  type KeyboardEvent,
  type PointerEvent,
  type ReactElement,
  type ReactNode,
  useCallback,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import { cn } from '../../../lib/cn';
import { truncateMiddle } from '../../../lib/labels';
import { Badge } from '../../arc/badge/badge';
import { Popover, PopoverContent, PopoverTrigger } from '../../arc/popover/popover';
import { Tooltip } from '../../arc/tooltip/tooltip';
import { LiveLabel } from '../live-label/live-label';
import { MiddleText } from '../middle-text/middle-text';
import { ModelBadge } from '../model-badge/model-badge';
import { MorphLoader } from '../morph-loader/morph-loader';
import styles from './model-picker.module.css';

export interface ModelVariantOption {
  /** The served model id. The key of the row. */
  id: string;
  /** Unique inside its base. */
  label: string;
  state: 'loaded' | 'loading' | 'idle';
  /** "GGUF", "17.6 GB": joined with " · " after the state word. */
  facts: readonly string[];
  /** e.g. "small for Work"; drawn with a warning icon. */
  warning?: string;
}

export interface ModelBaseOption {
  /** The key of the row. */
  base: string;
  /** 'loading' when a variant is in load, else 'loaded' when one is in memory. */
  state: 'loaded' | 'loading' | 'idle';
  /** The label of the variant that `state` speaks of; left out on a base with one variant. */
  stateVariant?: string;
  /** After the state part: the count and the context, or the context, the format and the size. */
  facts: readonly string[];
  warning?: string;
  /** One or more, in display order. */
  variants: readonly ModelVariantOption[];
  /** The id of the row that is active when this base opens. */
  preselect: string;
}

export interface ModelPickerProps {
  /** The model id of the conversation. */
  value: string | null;
  bases: readonly ModelBaseOption[];
  /** The model whose base opens when `value` is not served: the model a new conversation starts on. */
  fallback?: string | null;
  onValueChange: (id: string) => void;
  disabled?: boolean;
  disabledReason?: string;
  /** A change is being saved. */
  working?: boolean;
  /** Under both lists. */
  footer?: string;
  /** Shown instead of the footer while one or more variants are not loaded. */
  swapWarning?: string;
  /** The last row: what the engine holds now, and an action (Unload). Outside the listboxes: a normal tab stop. */
  memory?: { text: string; action: ReactNode };
  /** The tools row holds more chips (Work): the quantization badge hides under a wider composer. */
  crowded?: boolean;
}

/** The left list asks for this many rows at least, and shows 8.5 at most. */
const MIN_ROWS = 4;
const MAX_ROWS = 8.5;

interface Keys {
  open: string;
  base: string;
  variant: string;
}

function holding(bases: readonly ModelBaseOption[], id: string | null | undefined): ModelBaseOption | undefined {
  if (id === null || id === undefined) return undefined;
  return bases.find((option) => option.variants.some((variant) => variant.id === id));
}

/** The rows the keys point at. A key that is gone falls back to the base of `value` (or `fallback`), then the first. */
function resolve(
  bases: readonly ModelBaseOption[],
  value: string | null,
  fallback: string | null | undefined,
  keys: Keys,
): { home: ModelBaseOption | undefined; open: ModelBaseOption | undefined; base: number; variant: number } {
  const home = holding(bases, value) ?? holding(bases, fallback) ?? bases[0];
  const open = bases.find((option) => option.base === keys.open) ?? home;
  let base = bases.findIndex((option) => option.base === keys.base);
  if (base === -1) base = home === undefined ? 0 : bases.indexOf(home);
  const variants = open?.variants ?? [];
  let variant = variants.findIndex((option) => option.id === keys.variant);
  if (variant === -1)
    variant = Math.max(
      0,
      variants.findIndex((option) => option.id === open?.preselect),
    );
  return { home, open, base, variant };
}

/** Scrolls `list` by the least amount that shows all of its `index`th row; rows are positioned against the list. */
function reveal(list: HTMLElement | null, index: number): void {
  const row = list?.children.item(index);
  if (list === null || !(row instanceof HTMLElement)) return;
  const top = row.offsetTop;
  const bottom = top + row.offsetHeight;
  if (top < list.scrollTop) list.scrollTop = top;
  else if (bottom > list.scrollTop + list.clientHeight) list.scrollTop = bottom - list.clientHeight;
}

/** One printable letter or digit with no modifier: a first-letter move. */
function letterOf(event: KeyboardEvent): string | null {
  if (event.key.length !== 1 || event.metaKey || event.ctrlKey || event.altKey) return null;
  const key = event.key.toLowerCase();
  return key !== key.toUpperCase() || (key >= '0' && key <= '9') ? key : null;
}

/** The next row after `from` whose text starts with `letter`, wrapping; `from` when none does. */
function byLetter(texts: readonly string[], from: number, letter: string): number {
  for (let step = 1; step <= texts.length; step += 1) {
    const index = (from + step) % texts.length;
    if (texts[index]!.toLowerCase().startsWith(letter)) return index;
  }
  return from;
}

/** The move of ArrowUp, ArrowDown, Home, End or a letter in a list of `texts`, or null for another key. */
function moveOf(event: KeyboardEvent, texts: readonly string[], from: number): number | null {
  const last = texts.length - 1;
  if (event.key === 'ArrowDown') return Math.min(last, from + 1);
  if (event.key === 'ArrowUp') return Math.max(0, from - 1);
  if (event.key === 'Home') return 0;
  if (event.key === 'End') return last;
  const letter = letterOf(event);
  return letter === null ? null : byLetter(texts, from, letter);
}

function Facts(props: {
  state: ModelBaseOption['state'];
  stateVariant?: string;
  facts: readonly string[];
  warning?: string;
}): ReactElement {
  const parts = [...(props.stateVariant !== undefined ? [props.stateVariant] : []), ...props.facts];
  const lead = props.state === 'idle' ? '' : ' · ';
  return (
    <span className={styles.rowFacts}>
      {props.state === 'loading' ? <LiveLabel active>{CHAT.modelLoading}</LiveLabel> : null}
      {props.state === 'loaded' ? CHAT.modelLoaded : null}
      {parts.length > 0 ? lead + parts.join(' · ') : null}
      {props.warning !== undefined ? (
        <span className={styles.warning}>
          {props.state === 'idle' && parts.length === 0 ? '' : ' · '}
          <TriangleAlert size={12} strokeWidth={1.75} aria-hidden="true" />
          {props.warning}
        </span>
      ) : null}
    </span>
  );
}

/**
 * The model of the next message: a pill with the base and the quantization, and a popover with two lists, the bases
 * on the left and the quantizations of the open base on the right.
 */
export function ModelPicker(props: ModelPickerProps): ReactElement {
  const [open, setOpen] = useState(false);
  const [keys, setKeysState] = useState<Keys>({ open: '', base: '', variant: '' });
  const keysRef = useRef(keys);
  const [rightLive, setRightLive] = useState({ focus: false, pointer: false });
  const pointer = useRef<{ x: number; y: number } | null>(null);
  const focusRight = useRef(false);
  const id = useId();
  const [left, setLeft] = useState<HTMLDivElement | null>(null);
  const [right, setRight] = useState<HTMLDivElement | null>(null);
  const leftRef = useRef<HTMLDivElement | null>(null);
  const rightRef = useRef<HTMLDivElement | null>(null);
  const attachLeft = useCallback((node: HTMLDivElement | null) => {
    leftRef.current = node;
    setLeft(node);
  }, []);
  const attachRight = useCallback((node: HTMLDivElement | null) => {
    rightRef.current = node;
    setRight(node);
  }, []);

  const bases = props.bases;
  const empty = bases.length === 0 || props.value === null;
  const disabled = props.disabled === true || empty;
  const valueBase = holding(bases, props.value);
  const selected = valueBase?.variants.find((variant) => variant.id === props.value);
  const at = resolve(bases, props.value, props.fallback, keys);
  const openOption = at.open;
  const variants = openOption?.variants ?? [];
  // The footer must not follow the active row. The panel hangs from its bottom edge: a change of the footer height
  // moves the rows under a resting pointer, the active row changes, and the footer flips back.
  const swap =
    props.swapWarning !== undefined &&
    bases.some((option) => option.variants.some((variant) => variant.state !== 'loaded'))
      ? props.swapWarning
      : null;
  const footer = swap ?? props.footer ?? null;
  const rows = Math.min(Math.max(bases.length, MIN_ROWS), MAX_ROWS);

  const setKeys = (next: Partial<Keys>): void => {
    const merged = { ...keysRef.current, ...next };
    keysRef.current = merged;
    setKeysState(merged);
  };

  const openList = (): void => {
    const home = resolve(bases, props.value, props.fallback, { open: '', base: '', variant: '' }).home;
    setKeys({ open: home?.base ?? '', base: home?.base ?? '', variant: home?.preselect ?? '' });
    setRightLive({ focus: false, pointer: false });
    setOpen(true);
  };

  // Read by the effects below through a ref: they run when a list mounts or the order changes, not on each state.
  const latest = useRef({ bases, value: props.value, fallback: props.fallback });
  useLayoutEffect(() => {
    latest.current = { bases, value: props.value, fallback: props.fallback };
  });
  const showActive = useCallback((): void => {
    const now = latest.current;
    const where = resolve(now.bases, now.value, now.fallback, keysRef.current);
    reveal(leftRef.current, where.base);
    reveal(rightRef.current, where.variant);
  }, []);

  useLayoutEffect(() => {
    const node = rightRef.current;
    if (!focusRight.current || node === null) return;
    focusRight.current = false;
    node.focus();
    showActive();
  });

  const order = bases
    .map((option) => [option.base, ...option.variants.map((variant) => variant.id)].join('\n'))
    .join('\n\n');
  useLayoutEffect(() => {
    showActive();
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(showActive);
    if (left !== null) observer.observe(left);
    if (right !== null) observer.observe(right);
    return () => observer.disconnect();
  }, [left, right, order, showActive]);

  const pick = (pickedId: string): void => {
    setOpen(false);
    if (pickedId !== props.value) props.onValueChange(pickedId);
  };

  /** Opens `option` in the right list with its preselected row active, and moves focus there. */
  const openBase = (option: ModelBaseOption): void => {
    setKeys({ open: option.base, base: option.base, variant: option.preselect });
    focusRight.current = true;
  };

  const enterBase = (option: ModelBaseOption): void => {
    if (option.variants.length === 1) pick(option.variants[0]!.id);
    else openBase(option);
  };

  const pointerMoved = (event: PointerEvent): boolean => {
    const last = pointer.current;
    if (last !== null && last.x === event.clientX && last.y === event.clientY) return false;
    pointer.current = { x: event.clientX, y: event.clientY };
    return true;
  };

  const onLeftKey = (event: KeyboardEvent<HTMLDivElement>): void => {
    const option = bases[at.base];
    if (option === undefined) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      enterBase(option);
      return;
    }
    if (event.key === 'ArrowRight') {
      event.preventDefault();
      openBase(option);
      return;
    }
    const next = moveOf(
      event,
      bases.map((row) => row.base),
      at.base,
    );
    if (next === null) return;
    event.preventDefault();
    setKeys({ base: bases[next]!.base });
    reveal(left, next);
  };

  const onRightKey = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.key === 'ArrowLeft' || (event.key === 'Tab' && event.shiftKey)) {
      setKeys({ base: openOption?.base ?? '' });
      if (event.key === 'Tab') return;
      event.preventDefault();
      left?.focus();
      return;
    }
    const variant = variants[at.variant];
    if (variant === undefined) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      pick(variant.id);
      return;
    }
    const next = moveOf(
      event,
      variants.map((row) => row.label),
      at.variant,
    );
    if (next === null) return;
    event.preventDefault();
    setKeys({ variant: variants[next]!.id });
    reveal(right, next);
  };

  const pillState = selected?.state ?? null;
  const trigger = (
    <button
      type="button"
      className={styles.trigger}
      disabled={disabled}
      aria-haspopup="listbox"
      aria-expanded={open}
      data-state={pillState ?? undefined}
      data-crowded={props.crowded === true ? '' : undefined}
      title={empty ? undefined : (props.value ?? undefined)}
      aria-label={
        empty
          ? CHAT.noModel
          : selected === undefined || valueBase === undefined
            ? `${CHAT.model}: ${props.value}`
            : fill(CHAT.pillName, {
                model: `${valueBase.base} ${selected.label}`,
                state: CHAT.pillStates[selected.state],
              })
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
          <ModelBadge
            name={valueBase?.base ?? props.value ?? ''}
            colorKey={props.value ?? undefined}
            size="sm"
            showLogo
            maxChars={22}
          />
          {selected !== undefined ? (
            <span data-chip-label="" className={styles.variant}>
              <Badge tone="neutral" size="sm" outline>
                {truncateMiddle(selected.label, 14)}
              </Badge>
            </span>
          ) : null}
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

  const showRightActive = rightLive.focus || rightLive.pointer;

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
          leftRef.current?.focus();
        }}
      >
        <div className={styles.columns} style={{ '--picker-rows': rows } as CSSProperties}>
          <div className={styles.column}>
            <p className={styles.heading} id={`${id}-bases`}>
              {CHAT.model}
            </p>
            <div
              ref={attachLeft}
              className={styles.list}
              role="listbox"
              tabIndex={0}
              aria-labelledby={`${id}-bases`}
              aria-controls={`${id}-quants`}
              aria-activedescendant={bases[at.base] !== undefined ? `${id}-b${String(at.base)}` : undefined}
              onKeyDown={onLeftKey}
            >
              {bases.map((option, index) => (
                <div
                  key={option.base}
                  id={`${id}-b${String(index)}`}
                  className={cn(styles.row, styles.baseRow)}
                  role="option"
                  aria-selected={option === valueBase}
                  data-active={index === at.base ? '' : undefined}
                  data-open={option === openOption ? '' : undefined}
                  onPointerMove={(event) => {
                    if (pointerMoved(event)) setKeys({ base: option.base });
                  }}
                  onClick={() => enterBase(option)}
                >
                  <span className={styles.rowText}>
                    <span className={styles.rowName}>
                      <MiddleText text={option.base} />
                    </span>
                    <Facts
                      state={option.state}
                      stateVariant={option.stateVariant}
                      facts={option.facts}
                      warning={option.warning}
                    />
                  </span>
                  {option.variants.length > 1 ? (
                    <ChevronRight className={styles.chev} size={12} strokeWidth={1.75} aria-hidden="true" />
                  ) : (
                    <span className={styles.end}>{option.variants[0]?.label}</span>
                  )}
                  {option === valueBase ? (
                    <Check className={styles.check} size={14} strokeWidth={1.75} aria-hidden="true" />
                  ) : null}
                </div>
              ))}
            </div>
          </div>
          <div
            className={cn(styles.column, styles.quants)}
            onPointerEnter={() => setRightLive((live) => ({ ...live, pointer: true }))}
            onPointerLeave={() => setRightLive((live) => ({ ...live, pointer: false }))}
          >
            <p className={styles.heading}>{CHAT.modelQuant}</p>
            <div className={styles.body}>
              <div
                key={openOption?.base ?? ''}
                ref={attachRight}
                id={`${id}-quants`}
                className={cn(styles.list, styles.inner, styles.swap)}
                role="listbox"
                tabIndex={0}
                aria-label={fill(CHAT.modelQuantList, { base: openOption?.base ?? '' })}
                aria-activedescendant={variants[at.variant] !== undefined ? `${id}-v${String(at.variant)}` : undefined}
                onKeyDown={onRightKey}
                onFocus={() => setRightLive((live) => ({ ...live, focus: true }))}
                onBlur={() => setRightLive((live) => ({ ...live, focus: false }))}
              >
                {variants.map((variant, index) => (
                  <div
                    key={variant.id}
                    id={`${id}-v${String(index)}`}
                    className={styles.row}
                    role="option"
                    aria-selected={variant.id === props.value}
                    data-active={showRightActive && index === at.variant ? '' : undefined}
                    onPointerMove={(event) => {
                      if (pointerMoved(event)) setKeys({ variant: variant.id });
                    }}
                    onClick={() => pick(variant.id)}
                  >
                    <span className={styles.rowText}>
                      <span className={styles.rowName}>
                        <MiddleText text={variant.label} />
                      </span>
                      <Facts state={variant.state} facts={variant.facts} warning={variant.warning} />
                    </span>
                    {variant.id === props.value ? (
                      <Check className={styles.check} size={14} strokeWidth={1.75} aria-hidden="true" />
                    ) : null}
                  </div>
                ))}
              </div>
            </div>
          </div>
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

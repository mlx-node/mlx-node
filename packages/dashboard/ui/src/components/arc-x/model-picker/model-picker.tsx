import { Check, ChevronDown, ChevronLeft, ChevronRight, Info, TriangleAlert } from 'lucide-react';
import {
  type CSSProperties,
  type KeyboardEvent,
  type PointerEvent,
  type ReactElement,
  type ReactNode,
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import { cn } from '../../../lib/cn';
import { Badge } from '../../arc/badge/badge';
import { Popover, PopoverContent, PopoverTrigger } from '../../arc/popover/popover';
import { Tooltip } from '../../arc/tooltip/tooltip';
import { LiveLabel } from '../live-label/live-label';
import { MiddleText } from '../middle-text/middle-text';
import { ModelBadge } from '../model-badge/model-badge';
import { MorphLoader } from '../morph-loader/morph-loader';
import styles from './model-picker.module.css';

/** A model's thinking levels and the one a pick of its row keeps. */
export interface ModelEffort {
  /** Least thinking first. */
  levels: ReadonlyArray<{ value: string; label: string }>;
  /** One of `levels`: the conversation's level when the row is of its base, else the model's default. */
  value: string;
  /** Behind the ⓘ of the Thinking effort column. */
  note?: string;
}

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
  /** Absent or no levels: no Thinking effort column, and a pick sends no level. */
  effort?: ModelEffort;
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
  /**
   * A pick: the model, and its level when the model has levels (the one picked in the Thinking effort column, else
   * the row's `effort.value`). Not called when neither differs from the conversation's.
   */
  onValueChange: (id: string, level?: string) => void;
  disabled?: boolean;
  disabledReason?: string;
  /**
   * Only the model rows are off (an answer runs): the chip still opens, for the Thinking effort column of the
   * conversation's model; the reason is the footer.
   */
  listDisabled?: boolean;
  listDisabledReason?: string;
  /** The agent did not take the level before the last answer: a warning dot in the chip's level badge. */
  levelWarning?: boolean;
  /** A change is being saved. */
  working?: boolean;
  /** Under the lists. */
  footer?: string;
  /** Shown instead of the footer while one or more variants are not loaded. */
  swapWarning?: string;
  /** The last row: what the engine holds now, and an action (Unload). Outside the listboxes: a normal tab stop. */
  memory?: { text: string; action: ReactNode };
  /** The tools row holds more chips (Work): the level badge hides under a wider composer. */
  crowded?: boolean;
}

/** The left list asks for this many rows at least, and shows 8.5 at most. */
const MIN_ROWS = 4;
const MAX_ROWS = 8.5;
/** The width of `.panel.panel[data-third]` in the CSS, and the gap the panel keeps from the window edge. */
const THIRD_WIDTH = 713;
const GUTTER = 16;

interface Keys {
  open: string;
  base: string;
  variant: string;
  level: string;
}

const NO_KEYS: Keys = { open: '', base: '', variant: '', level: '' };

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

function levelsOf(variant: ModelVariantOption | undefined): ModelEffort | undefined {
  const effort = variant?.effort;
  return effort !== undefined && effort.levels.length > 0 ? effort : undefined;
}

/** The label of the level a pick of `variant` keeps; null for a model with no levels. */
function keptLabel(variant: ModelVariantOption | undefined): string | null {
  const effort = levelsOf(variant);
  return effort?.levels.find((option) => option.value === effort.value)?.label ?? null;
}

/** The active level row: `key` when the list has it, else the level the row keeps, else the first. */
function levelIndex(effort: ModelEffort | undefined, key: string): number {
  if (effort === undefined) return 0;
  const at = effort.levels.findIndex((option) => option.value === key);
  if (at !== -1) return at;
  return Math.max(
    0,
    effort.levels.findIndex((option) => option.value === effort.value),
  );
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

/** The chip's name with the level of the model. */
function withLevel(name: string, level: string | null): string {
  return level === null ? name : fill(CHAT.thinkingLevel.chipName, { model: name, level });
}

/**
 * The model of the next message: a pill with the base and the level, and a popover of cascading lists: the bases, the
 * quantizations of the open base (a base with several), and the Thinking effort of the quantization in view (a model
 * with levels). A window too narrow for three columns shows the levels in place of the quantizations, with a back row.
 */
export function ModelPicker(props: ModelPickerProps): ReactElement {
  const [open, setOpen] = useState(false);
  const [keys, setKeysState] = useState<Keys>(NO_KEYS);
  const keysRef = useRef(keys);
  const [rightLive, setRightLive] = useState({ focus: false, pointer: false });
  const [levelsLive, setLevelsLive] = useState({ focus: false, pointer: false });
  /** Narrow window: the levels of the active quantization replace the quantization list. */
  const [deep, setDeep] = useState(false);
  /** Three columns fit between the chip's left edge and the window's right edge. */
  const [room, setRoom] = useState(true);
  const pointer = useRef<{ x: number; y: number } | null>(null);
  const focusTo = useRef<'quants' | 'levels' | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const id = useId();
  const [left, setLeft] = useState<HTMLDivElement | null>(null);
  const [right, setRight] = useState<HTMLDivElement | null>(null);
  const [levelsNode, setLevelsNode] = useState<HTMLDivElement | null>(null);
  const leftRef = useRef<HTMLDivElement | null>(null);
  const rightRef = useRef<HTMLDivElement | null>(null);
  const levelsRef = useRef<HTMLDivElement | null>(null);
  const attachLeft = useCallback((node: HTMLDivElement | null) => {
    leftRef.current = node;
    setLeft(node);
  }, []);
  const attachRight = useCallback((node: HTMLDivElement | null) => {
    rightRef.current = node;
    setRight(node);
  }, []);
  const attachLevels = useCallback((node: HTMLDivElement | null) => {
    levelsRef.current = node;
    setLevelsNode(node);
  }, []);

  const bases = props.bases;
  const empty = bases.length === 0 || props.value === null;
  const disabled = props.disabled === true || empty;
  const valueBase = holding(bases, props.value);
  const selected = valueBase?.variants.find((variant) => variant.id === props.value);
  const at = resolve(bases, props.value, props.fallback, keys);
  const openOption = at.open;
  const variants = openOption?.variants ?? [];
  const one = variants.length === 1;
  // The footer must not follow the active row. The panel hangs from its bottom edge: a change of the footer height
  // moves the rows under a resting pointer, the active row changes, and the footer flips back.
  const swap =
    props.swapWarning !== undefined &&
    bases.some((option) => option.variants.some((variant) => variant.state !== 'loaded'))
      ? props.swapWarning
      : null;
  const listOff = props.listDisabled === true;
  const footer = (listOff ? props.listDisabledReason : undefined) ?? swap ?? props.footer ?? null;
  const levelShown = keptLabel(selected);
  const rows = Math.min(Math.max(bases.length, MIN_ROWS), MAX_ROWS);

  /** The levels a row opens: while the model rows are off, only the conversation's model has them. */
  const nextOf = (variant: ModelVariantOption | undefined): ModelEffort | undefined =>
    listOff && variant?.id !== props.value ? undefined : levelsOf(variant);
  const activeVariant = variants[at.variant];
  const levelsVariant = nextOf(activeVariant) !== undefined && (one || room || deep) ? activeVariant : undefined;
  const effort = nextOf(levelsVariant);
  const inPlace = levelsVariant !== undefined && (one || !room);
  const third = levelsVariant !== undefined && !inPlace;
  const back = inPlace && !one;
  const levelAt = levelIndex(effort, keys.level);

  const setKeys = (next: Partial<Keys>): void => {
    const merged = { ...keysRef.current, ...next };
    keysRef.current = merged;
    setKeysState(merged);
  };

  const measure = useCallback((): void => {
    const node = triggerRef.current;
    if (node === null) return;
    setRoom(node.getBoundingClientRect().left + THIRD_WIDTH + GUTTER <= window.innerWidth);
  }, []);
  useEffect(() => {
    if (!open) return;
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, [open, measure]);

  const openList = (): void => {
    const home = resolve(bases, props.value, props.fallback, NO_KEYS).home;
    setKeys({ open: home?.base ?? '', base: home?.base ?? '', variant: home?.preselect ?? '', level: '' });
    setRightLive({ focus: false, pointer: false });
    setLevelsLive({ focus: false, pointer: false });
    setDeep(false);
    measure();
    setOpen(true);
  };

  // Read by the effects below through a ref: they run when a list mounts or the order changes, not on each state.
  const latest = useRef({ bases, value: props.value, fallback: props.fallback, level: levelAt });
  useLayoutEffect(() => {
    latest.current = { bases, value: props.value, fallback: props.fallback, level: levelAt };
  });
  const showActive = useCallback((): void => {
    const now = latest.current;
    const where = resolve(now.bases, now.value, now.fallback, keysRef.current);
    reveal(leftRef.current, where.base);
    reveal(rightRef.current, where.variant);
    reveal(levelsRef.current, now.level);
  }, []);

  useLayoutEffect(() => {
    const target = focusTo.current;
    const node = target === 'quants' ? rightRef.current : target === 'levels' ? levelsRef.current : null;
    if (node === null) return;
    focusTo.current = null;
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
    if (levelsNode !== null) observer.observe(levelsNode);
    return () => observer.disconnect();
  }, [left, right, levelsNode, order, showActive]);

  /** Model and level together; nothing is sent when both are the conversation's. */
  const commit = (variant: ModelVariantOption, level: string | undefined): void => {
    if (listOff && nextOf(variant) === undefined) return;
    setOpen(false);
    const current = levelsOf(selected)?.value;
    if (level === undefined) {
      if (variant.id !== props.value) props.onValueChange(variant.id);
    } else if (variant.id !== props.value || level !== current) props.onValueChange(variant.id, level);
  };

  /** A leaf row: a model with no levels (or, while the model rows are off, one that is not the conversation's). */
  const pick = (variant: ModelVariantOption): void => commit(variant, undefined);

  /** Opens `option` with its preselected row active, and moves focus to its next column. */
  const openBase = (option: ModelBaseOption): void => {
    setKeys({ open: option.base, base: option.base, variant: option.preselect, level: '' });
    setDeep(false);
    const only = option.variants.length === 1 ? option.variants[0] : undefined;
    focusTo.current = nextOf(only) !== undefined ? 'levels' : 'quants';
  };

  const enterBase = (option: ModelBaseOption): void => {
    const only = option.variants.length === 1 ? option.variants[0]! : undefined;
    if (only !== undefined && nextOf(only) === undefined) pick(only);
    else openBase(option);
  };

  /** The levels of `variant` with the level it keeps active, focus in them. */
  const openLevels = (variant: ModelVariantOption): void => {
    setKeys({ variant: variant.id, level: '' });
    if (!room) setDeep(true);
    focusTo.current = 'levels';
  };

  const levelsBack = (): void => {
    if (one) {
      setKeys({ base: openOption?.base ?? '' });
      left?.focus();
      return;
    }
    if (third) {
      right?.focus();
      return;
    }
    setDeep(false);
    focusTo.current = 'quants';
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
    if (event.key === 'Enter' || event.key === ' ' || event.key === 'ArrowRight') {
      event.preventDefault();
      if (nextOf(variant) !== undefined) openLevels(variant);
      else if (event.key !== 'ArrowRight') pick(variant);
      return;
    }
    const next = moveOf(
      event,
      variants.map((row) => row.label),
      at.variant,
    );
    if (next === null) return;
    event.preventDefault();
    setKeys({ variant: variants[next]!.id, level: '' });
    reveal(right, next);
  };

  const onLevelsKey = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.key === 'ArrowLeft') {
      event.preventDefault();
      levelsBack();
      return;
    }
    if (event.key === 'Tab' && event.shiftKey) {
      if (inPlace) setKeys({ base: openOption?.base ?? '' });
      return;
    }
    const level = effort?.levels[levelAt];
    if (levelsVariant === undefined || effort === undefined || level === undefined) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      commit(levelsVariant, level.value);
      return;
    }
    const next = moveOf(
      event,
      effort.levels.map((row) => row.label),
      levelAt,
    );
    if (next === null) return;
    event.preventDefault();
    setKeys({ level: effort.levels[next]!.value });
    reveal(levelsRef.current, next);
  };

  const trigger = (
    <button
      ref={triggerRef}
      type="button"
      className={styles.trigger}
      disabled={disabled}
      aria-haspopup="listbox"
      aria-expanded={open}
      data-state={selected?.state ?? undefined}
      data-crowded={props.crowded === true ? '' : undefined}
      title={
        empty
          ? undefined
          : levelShown === null
            ? (props.value ?? undefined)
            : fill(CHAT.thinkingLevel.chip, { model: props.value ?? '', level: levelShown })
      }
      aria-label={
        empty
          ? CHAT.noModel
          : withLevel(
              selected === undefined || valueBase === undefined
                ? `${CHAT.model}: ${props.value}`
                : fill(CHAT.pillName, {
                    model: `${valueBase.base} ${selected.label}`,
                    state: CHAT.pillStates[selected.state],
                  }),
              levelShown,
            )
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
          {levelShown !== null ? (
            <span
              data-chip-label=""
              data-chip-level=""
              data-fit-whole={props.crowded === true ? '' : undefined}
              className={styles.variant}
            >
              <Badge tone="neutral" size="sm" outline>
                {levelShown}
                {props.levelWarning === true ? <span className={styles.levelWarn} aria-hidden="true" /> : null}
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
  const showLevelsActive = levelsLive.focus || levelsLive.pointer;
  const levelsLiveOn = {
    onPointerEnter: () => setLevelsLive((live) => ({ ...live, pointer: true })),
    onPointerLeave: () => setLevelsLive((live) => ({ ...live, pointer: false })),
  };

  const levelsHeading =
    effort === undefined ? null : (
      <p className={styles.heading}>
        {CHAT.thinkingLevel.column}
        {effort.note !== undefined ? (
          <>
            <Tooltip content={effort.note}>
              <span className={styles.info}>
                <Info size={12} strokeWidth={1.75} aria-hidden="true" />
              </span>
            </Tooltip>
            <span id={`${id}-note`} hidden>
              {effort.note}
            </span>
          </>
        ) : null}
      </p>
    );

  const levelsList =
    levelsVariant === undefined || effort === undefined || openOption === undefined ? null : (
      <div
        key={inPlace ? `in:${openOption.base}` : 'third'}
        ref={attachLevels}
        id={`${id}-levels`}
        className={cn(styles.list, styles.inner, styles.swap)}
        role="listbox"
        tabIndex={0}
        aria-label={fill(CHAT.thinkingLevel.list, {
          model: one ? openOption.base : `${openOption.base} ${levelsVariant.label}`,
        })}
        aria-describedby={effort.note !== undefined ? `${id}-note` : undefined}
        aria-activedescendant={effort.levels[levelAt] !== undefined ? `${id}-l${String(levelAt)}` : undefined}
        onKeyDown={onLevelsKey}
        onFocus={() => setLevelsLive((live) => ({ ...live, focus: true }))}
        onBlur={() => setLevelsLive((live) => ({ ...live, focus: false }))}
      >
        {effort.levels.map((level, index) => (
          <div
            key={level.value}
            id={`${id}-l${String(index)}`}
            className={cn(styles.row, styles.levelRow)}
            role="option"
            aria-selected={level.value === effort.value}
            data-active={showLevelsActive && index === levelAt ? '' : undefined}
            onPointerMove={(event) => {
              if (pointerMoved(event)) setKeys({ level: level.value });
            }}
            onClick={() => commit(levelsVariant, level.value)}
          >
            <span className={styles.levelName}>{level.label}</span>
            {level.value === effort.value ? (
              <Check className={styles.check} size={14} strokeWidth={1.75} aria-hidden="true" />
            ) : null}
          </div>
        ))}
      </div>
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
        collisionPadding={GUTTER}
        className={styles.panel}
        data-third={third ? '' : undefined}
        aria-label={CHAT.model}
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          leftRef.current?.focus();
        }}
      >
        <div
          className={styles.columns}
          data-third={third ? '' : undefined}
          style={{ '--picker-rows': rows } as CSSProperties}
        >
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
              aria-controls={inPlace ? `${id}-levels` : `${id}-quants`}
              aria-activedescendant={bases[at.base] !== undefined ? `${id}-b${String(at.base)}` : undefined}
              onKeyDown={onLeftKey}
            >
              {bases.map((option, index) => {
                const only = option.variants.length === 1 ? option.variants[0] : undefined;
                const level = keptLabel(option.variants.find((variant) => variant.id === option.preselect));
                return (
                  <div
                    key={option.base}
                    id={`${id}-b${String(index)}`}
                    className={cn(styles.row, styles.baseRow)}
                    role="option"
                    aria-selected={option === valueBase}
                    aria-disabled={listOff && only !== undefined && nextOf(only) === undefined ? true : undefined}
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
                        {level !== null ? <span className={styles.rowLevel}>{level}</span> : null}
                      </span>
                      <Facts
                        state={option.state}
                        stateVariant={option.stateVariant}
                        facts={option.facts}
                        warning={option.warning}
                      />
                    </span>
                    {only === undefined ? (
                      <ChevronRight className={styles.chev} size={12} strokeWidth={1.75} aria-hidden="true" />
                    ) : nextOf(only) !== undefined ? (
                      <span className={styles.endNext}>
                        <span className={styles.end}>{only.label}</span>
                        <ChevronRight size={12} strokeWidth={1.75} aria-hidden="true" />
                      </span>
                    ) : (
                      <span className={styles.end}>{only.label}</span>
                    )}
                    {option === valueBase ? (
                      <Check className={styles.check} size={14} strokeWidth={1.75} aria-hidden="true" />
                    ) : null}
                  </div>
                );
              })}
            </div>
          </div>
          <div
            className={cn(styles.column, styles.quants)}
            data-back={back ? '' : undefined}
            onPointerEnter={
              inPlace ? levelsLiveOn.onPointerEnter : () => setRightLive((live) => ({ ...live, pointer: true }))
            }
            onPointerLeave={
              inPlace ? levelsLiveOn.onPointerLeave : () => setRightLive((live) => ({ ...live, pointer: false }))
            }
          >
            {back && levelsVariant !== undefined ? (
              <button
                type="button"
                tabIndex={-1}
                className={styles.back}
                aria-label={fill(CHAT.thinkingLevel.back, { quant: levelsVariant.label })}
                onClick={levelsBack}
              >
                <ChevronLeft size={12} strokeWidth={1.75} aria-hidden="true" />
                <MiddleText text={levelsVariant.label} />
              </button>
            ) : null}
            {inPlace ? levelsHeading : <p className={styles.heading}>{CHAT.modelQuant}</p>}
            <div className={styles.body}>
              {inPlace ? (
                levelsList
              ) : (
                <div
                  key={openOption?.base ?? ''}
                  ref={attachRight}
                  id={`${id}-quants`}
                  className={cn(styles.list, styles.inner, styles.swap)}
                  role="listbox"
                  tabIndex={0}
                  aria-label={fill(CHAT.modelQuantList, { base: openOption?.base ?? '' })}
                  aria-controls={third ? `${id}-levels` : undefined}
                  aria-activedescendant={
                    variants[at.variant] !== undefined ? `${id}-v${String(at.variant)}` : undefined
                  }
                  onKeyDown={onRightKey}
                  onFocus={() => setRightLive((live) => ({ ...live, focus: true }))}
                  onBlur={() => setRightLive((live) => ({ ...live, focus: false }))}
                >
                  {variants.map((variant, index) => {
                    const next = nextOf(variant) !== undefined;
                    return (
                      <div
                        key={variant.id}
                        id={`${id}-v${String(index)}`}
                        className={cn(styles.row, next && styles.variantNext)}
                        role="option"
                        aria-selected={variant.id === props.value}
                        aria-disabled={listOff && !next ? true : undefined}
                        data-active={showRightActive && index === at.variant ? '' : undefined}
                        data-open={third && variant === levelsVariant ? '' : undefined}
                        onPointerMove={(event) => {
                          if (pointerMoved(event)) setKeys({ variant: variant.id, level: '' });
                        }}
                        onClick={() => (next ? openLevels(variant) : pick(variant))}
                      >
                        <span className={styles.rowText}>
                          <span className={styles.rowName}>
                            <MiddleText text={variant.label} />
                          </span>
                          <Facts state={variant.state} facts={variant.facts} warning={variant.warning} />
                        </span>
                        {next ? (
                          <ChevronRight className={styles.chev} size={12} strokeWidth={1.75} aria-hidden="true" />
                        ) : null}
                        {variant.id === props.value ? (
                          <Check className={styles.check} size={14} strokeWidth={1.75} aria-hidden="true" />
                        ) : null}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
          {third ? (
            <div className={cn(styles.column, styles.quants)} {...levelsLiveOn}>
              {levelsHeading}
              <div className={styles.body}>{levelsList}</div>
            </div>
          ) : null}
        </div>
        {footer !== null ? (
          <p className={styles.footer} data-warning={swap !== null && !listOff ? '' : undefined}>
            {swap !== null && !listOff ? <TriangleAlert size={12} strokeWidth={1.75} aria-hidden="true" /> : null}
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

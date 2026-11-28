import * as Menu from '@radix-ui/react-dropdown-menu';
import { Check, ChevronDown, ChevronLeft, ChevronRight, Gauge, Layers, TriangleAlert } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import {
  Fragment,
  type FocusEvent,
  type KeyboardEvent,
  type MouseEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactElement,
  type ReactNode,
  type RefObject,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import { cn } from '../../../lib/cn';
import { motionTokens } from '../../arc/motion-tokens';
import { Tooltip } from '../../arc/tooltip/tooltip';
import { LiveLabel } from '../live-label/live-label';
import { classifyModel, type ModelFamily, ModelLogo } from '../model-badge/model-logos';
import { MorphLoader } from '../morph-loader/morph-loader';
import { motionX } from '../motion/motion';
import { useStaticMotion } from '../motion/use-static-motion';
import styles from './model-picker.module.css';

export interface ModelEffortLevel {
  value: string;
  label: string;
  /** Short one-line description shown under the level label (e.g. "Quickest answers"). */
  note?: string;
}

/** A model's thinking levels and the one a pick of its row keeps. */
export interface ModelEffort {
  /** Least thinking first. */
  levels: readonly ModelEffortLevel[];
  /** One of `levels`: the conversation's level when the row is of its base, else the model's default. */
  value: string;
  /** Model's default level value: renders Arc's quiet "Default" tag beside that row. */
  defaultValue?: string;
  /** Quiet note shown in the Effort submenu (e.g. "Applies to your next message."). */
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
  /** Absent or no levels: no Effort row, and a pick sends no level. */
  effort?: ModelEffort;
}

export interface ModelBaseOption {
  /** The key of the row. */
  base: string;
  /** One-line blurb shown under the model name (catalog description; falls back to facts). */
  note?: string;
  /** True when on our recommended list -> rendered outside at top level; others sit under "More models". */
  recommended?: boolean;
  /** Optional provider/family group label inside "More models" (inferred from classifyModel when omitted). */
  provider?: string;
  /** 'loading' when a variant is in load, else 'loaded' when one is in memory. */
  state: 'loaded' | 'loading' | 'idle';
  /** The label of the variant that `state` speaks of; left out on a base with one variant. */
  stateVariant?: string;
  /** After the state part: the count and the context, or the context, the format and the size. */
  facts: readonly string[];
  warning?: string;
  /** One or more, in display order. */
  variants: readonly ModelVariantOption[];
  /** The id of the row that is preselected when this base is clicked directly. */
  preselect: string;
}

export interface ModelPickerProps {
  /** The model id of the conversation. */
  value: string | null;
  bases: readonly ModelBaseOption[];
  /** The model whose base opens when `value` is not served: the model a new conversation starts on. */
  fallback?: string | null;
  /**
   * A pick: the model, and its level when the model has levels (the one picked in the Effort submenu, else
   * the row's `effort.value`). Not called when neither differs from the conversation's.
   */
  onValueChange: (id: string, level?: string) => void;
  disabled?: boolean;
  disabledReason?: string;
  /**
   * Only the model rows are off (an answer runs): the chip still opens, for the Effort submenu of the
   * conversation's model; the reason is the footer.
   */
  listDisabled?: boolean;
  listDisabledReason?: string;
  /** The agent did not take the level before the last answer: a warning dot beside the chip's level. */
  levelWarning?: boolean;
  /** A change is being saved. */
  working?: boolean;
  /** Under the lists. */
  footer?: string;
  /** Shown instead of the footer while one or more variants are not loaded. */
  swapWarning?: string;
  /** The last row: what the engine holds now, and an action (Unload). Outside the menu items: a normal tab stop. */
  memory?: { text: string; action: ReactNode };
  /** The tools row holds more chips (Work): the level badge hides under a wider composer. */
  crowded?: boolean;
  /** Preferred side of the trigger to open the menu on. Defaults to `'top'`. */
  side?: 'top' | 'bottom';
  /** Preferred alignment relative to the trigger. Defaults to `'start'`. */
  align?: 'start' | 'center' | 'end';
}

const { duration, ease } = motionTokens;
const settle = motionX.spring.expand;
const smooth = motionX.spring.reshape;
const standard = [...ease.standard] as [number, number, number, number];
const INSTANT = { duration: 0 } as const;

type Page =
  | { kind: 'main' }
  | { kind: 'effort' }
  | { kind: 'more' }
  | { kind: 'quant'; base: string; from: 'main' | 'more' };

const MAIN_PAGE: Page = { kind: 'main' };

function pageKey(page: Page): string {
  return page.kind === 'quant' ? `quant:${page.base}` : page.kind;
}

function pageDepth(page: Page): number {
  if (page.kind === 'main') return 0;
  if (page.kind === 'quant' && page.from === 'more') return 2;
  return 1;
}

function holding(bases: readonly ModelBaseOption[], id: string | null | undefined): ModelBaseOption | undefined {
  if (id === null || id === undefined) return undefined;
  return bases.find((option) => option.variants.some((variant) => variant.id === id));
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

/** The chip's name with the level of the model. */
function withLevel(name: string, level: string | null): string {
  return level === null ? name : fill(CHAT.thinkingLevel.chipName, { model: name, level });
}

function providerLabel(family: ModelFamily): string {
  switch (family) {
    case 'qwen':
      return 'Qwen';
    case 'gemma':
      return 'Gemma';
    case 'hy':
      return 'Tencent Hy';
    case 'liquid':
      return 'Liquid';
    case 'cloudflare':
      return 'Cloudflare';
    case 'internscience':
      return 'InternScience';
    case 'ornith':
      return 'Ornith';
    default:
      return CHAT.modelOtherProvider;
  }
}

function groupByProvider(bases: readonly ModelBaseOption[]): Array<{ provider: string; bases: ModelBaseOption[] }> {
  const buckets = new Map<string, ModelBaseOption[]>();
  const other: ModelBaseOption[] = [];
  for (const base of bases) {
    const provider = base.provider ?? providerLabel(classifyModel(base.base));
    if (provider === CHAT.modelOtherProvider) {
      other.push(base);
      continue;
    }
    const bucket = buckets.get(provider);
    if (bucket === undefined) buckets.set(provider, [base]);
    else bucket.push(base);
  }
  const groups = Array.from(buckets, ([provider, items]) => ({ provider, bases: items }));
  if (other.length > 0) groups.push({ provider: CHAT.modelOtherProvider, bases: other });
  return groups;
}

const LIST_MAX_HEIGHT = 420;
const LIST_VERTICAL_PADDING = 12;
const FOCUSABLE_IN_FOOT = 'button:not(:disabled), a[href], [tabindex]:not([tabindex="-1"]):not(:disabled)';

function firstFocusableMenuItem(list: HTMLElement | null): HTMLElement | null {
  if (list === null) return null;
  return (
    list.querySelector<HTMLElement>(
      '[role^="menuitem"][data-state="checked"]:not([data-disabled]), [role^="menuitem"][data-checked]:not([data-disabled])',
    ) ??
    list.querySelector<HTMLElement>('[role^="menuitem"]:not([data-back]):not([data-disabled])') ??
    list.querySelector<HTMLElement>('[role^="menuitem"]:not([data-disabled])')
  );
}

/* ---------------------------------------------------------------------------------------------------------------
 * Trigger: the mark crossfades and the width springs to the new name, so a switch never snaps the pill.
 * ------------------------------------------------------------------------------------------------------------- */

function TriggerLabel({
  modelId,
  displayName,
  level,
  levelWarning,
  crowded,
  reduced,
}: {
  modelId: string;
  displayName: string;
  level: string | null;
  levelWarning: boolean;
  crowded: boolean;
  reduced: boolean;
}): ReactElement {
  const measure = useRef<HTMLSpanElement>(null);
  const measuredKey = useRef<string | null>(null);
  const [size, setSize] = useState<{ width: number | 'auto'; animate: boolean }>({ width: 'auto', animate: false });
  const key = `${modelId}|${level ?? ''}`;
  if (crowded && (size.width !== 'auto' || size.animate)) {
    setSize({ width: 'auto', animate: false });
  }

  useLayoutEffect(() => {
    if (crowded) {
      measuredKey.current = null;
      return;
    }
    const node = measure.current;
    if (node === null || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(([entry]) => {
      if (entry === undefined) return;
      const measured = Math.ceil(entry.borderBoxSize?.[0]?.inlineSize ?? node.offsetWidth);
      if (measured <= 0) return;
      const current = node.dataset.key ?? '';
      // Only a change of model or effort morphs; the first measure and font swaps settle instantly.
      const animate = measuredKey.current !== null && measuredKey.current !== current;
      measuredKey.current = current;
      setSize({ width: measured, animate });
    });
    observer.observe(node);
    return () => {
      observer.disconnect();
      measuredKey.current = null;
    };
  }, [crowded]);

  const text = (hidden?: boolean) => (
    <>
      <span className={styles.triggerName} data-model={hidden ? undefined : modelId}>
        {displayName}
      </span>
      {level !== null ? (
        <span
          className={styles.triggerEffort}
          aria-hidden={hidden || undefined}
          data-chip-label=""
          data-chip-level={hidden ? undefined : ''}
          data-fit-whole={!hidden && crowded ? '' : undefined}
        >
          {level}
          {levelWarning ? <span className={styles.levelWarn} aria-hidden="true" /> : null}
        </span>
      ) : null}
    </>
  );

  return (
    <>
      <span className={styles.triggerMark} aria-hidden="true">
        <AnimatePresence initial={false} mode="popLayout">
          <motion.span
            key={displayName}
            className={styles.triggerMarkLayer}
            initial={reduced ? { opacity: 0 } : { opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={
              reduced
                ? { opacity: 0, transition: INSTANT }
                : { opacity: 0, transition: { duration: duration.instant, ease: standard } }
            }
            transition={reduced ? INSTANT : { duration: duration.fast, ease: standard }}
          >
            <ModelLogo model={displayName} className={styles.mark} />
          </motion.span>
        </AnimatePresence>
      </span>
      <motion.span
        className={styles.triggerText}
        initial={false}
        animate={{ width: crowded ? 'auto' : size.width }}
        transition={!crowded && size.animate && !reduced ? settle : INSTANT}
      >
        <AnimatePresence initial={false} mode="popLayout">
          <motion.span
            key={key}
            className={styles.triggerInner}
            initial={reduced ? { opacity: 0 } : { opacity: 0, y: 3 }}
            animate={{ opacity: 1, y: 0 }}
            exit={
              reduced
                ? { opacity: 0, transition: INSTANT }
                : { opacity: 0, transition: { duration: duration.instant, ease: standard } }
            }
            transition={reduced ? INSTANT : { duration: duration.fast, ease: standard }}
          >
            {text()}
          </motion.span>
        </AnimatePresence>
        {!crowded ? (
          <span ref={measure} data-key={key} className={cn(styles.triggerInner, styles.measure)} aria-hidden="true">
            {text(true)}
          </span>
        ) : null}
      </motion.span>
    </>
  );
}

/* ---------------------------------------------------------------------------------------------------------------
 * One highlight per menu surface. It glides between items for the pointer and jumps for the keyboard, so moving
 * through the list never flickers row by row.
 * ------------------------------------------------------------------------------------------------------------- */

function useGlide(reduced: boolean, resetKey?: string) {
  const [box, setBox] = useState<{ top: number; height: number; glide: boolean } | null>(null);
  const container = useRef<HTMLDivElement>(null);
  const pointer = useRef(false);
  const timer = useRef(0);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  // A new page of items starts without a highlight, so it never glides in from a row that is gone.
  const [shownKey, setShownKey] = useState(resetKey);
  if (shownKey !== resetKey) {
    setShownKey(resetKey);
    setBox(null);
  }

  const onFocus = (event: FocusEvent<HTMLDivElement>): void => {
    const target = event.target instanceof HTMLElement ? event.target : null;
    if (target !== null && !event.currentTarget.contains(target)) return;
    if (event.target instanceof Node && !event.currentTarget.contains(event.target)) return;
    const item = target !== null ? target.closest<HTMLElement>('[role^="menuitem"]') : null;
    const list = container.current;
    if (item !== null && (list === null || !list.contains(item))) return;
    window.clearTimeout(timer.current);
    if (item === null || list === null) {
      // A short grace period keeps the highlight gliding across separators and labels.
      timer.current = window.setTimeout(() => setBox(null), pointer.current ? 70 : 0);
      return;
    }
    let top = 0;
    let node: HTMLElement | null = item;
    let reachedList = false;
    while (node !== null && node !== list) {
      top += node.offsetTop;
      const parent: HTMLElement | null = node.offsetParent instanceof HTMLElement ? node.offsetParent : null;
      if (parent === list) {
        reachedList = true;
        break;
      }
      if (parent === null || !list.contains(parent)) break;
      node = parent;
    }
    const itemTop = reachedList ? top : item.offsetTop;
    const glide = pointer.current;
    setBox((current) => ({ top: itemTop, height: item.offsetHeight, glide: glide && current !== null }));
  };

  const highlight = (
    <motion.span
      className={styles.highlight}
      aria-hidden="true"
      initial={false}
      animate={box !== null ? { y: box.top, height: box.height, opacity: 1 } : { opacity: 0 }}
      transition={{
        default: box?.glide === true && !reduced ? settle : INSTANT,
        opacity: { duration: reduced ? 0 : duration.instant },
      }}
    />
  );

  const handlers = {
    onFocus,
    onPointerMoveCapture: () => {
      pointer.current = true;
    },
    onKeyDownCapture: () => {
      pointer.current = false;
    },
  };
  return { container, highlight, handlers };
}

/** Springs its height to the content, so a page change inside the menu never resizes it in one frame. */
function AutoHeight({
  children,
  reduced,
  listRef,
  listId,
  page,
  highlight,
}: {
  children: ReactNode;
  reduced: boolean;
  listRef: RefObject<HTMLDivElement | null>;
  listId: string;
  page: string;
  highlight: ReactNode;
}): ReactElement {
  const measureRef = useRef<HTMLDivElement>(null);
  const [height, setHeight] = useState<number | 'auto'>('auto');
  useLayoutEffect(() => {
    const node = measureRef.current;
    if (node === null || typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(() => {
      if (node.offsetHeight <= 0) return;
      const listEl = listRef.current;
      let pad = LIST_VERTICAL_PADDING;
      if (listEl !== null) {
        const computed = getComputedStyle(listEl);
        const top = Number.parseFloat(computed.paddingTop);
        const bottom = Number.parseFloat(computed.paddingBottom);
        if (Number.isFinite(top) && Number.isFinite(bottom) && top + bottom > 0) pad = top + bottom;
      }
      setHeight(Math.min(LIST_MAX_HEIGHT, Math.ceil(node.offsetHeight + pad)));
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [listRef]);
  return (
    <motion.div
      className={styles.autoHeight}
      initial={false}
      animate={{ height }}
      transition={reduced ? INSTANT : smooth}
    >
      <div ref={listRef} id={listId} className={styles.list} data-page={page}>
        {highlight}
        <div ref={measureRef} className={styles.listInner}>
          {children}
        </div>
      </div>
    </motion.div>
  );
}

/** Prevents Radix's 100ms hover-open timer from flashing a submenu open while the root menu is closing. */
function GuardedSub({ rootOpen, children }: { rootOpen: boolean; children: ReactNode }): ReactElement {
  const [subOpen, setSubOpen] = useState(false);
  if (!rootOpen && subOpen) {
    setSubOpen(false);
  }
  return (
    <Menu.Sub open={rootOpen && subOpen} onOpenChange={(next) => setSubOpen(rootOpen && next)}>
      {children}
    </Menu.Sub>
  );
}

function MenuSurface({
  children,
  footer,
  footRef,
  reduced,
  page,
  listId,
  side,
  align,
  onCloseAutoFocus,
  onKeyDown,
}: {
  children: ReactNode;
  footer?: ReactNode;
  footRef: RefObject<HTMLDivElement | null>;
  reduced: boolean;
  page: string;
  listId: string;
  side: 'top' | 'bottom';
  align: 'start' | 'center' | 'end';
  onCloseAutoFocus: (event: Event) => void;
  onKeyDown: (event: KeyboardEvent<HTMLDivElement>) => void;
}): ReactElement {
  const { container, highlight, handlers } = useGlide(reduced, page);

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.key === 'Tab') {
      const target = event.target instanceof Node ? event.target : null;
      if (target === null || event.currentTarget.contains(target)) {
        const foot = footRef.current;
        const controls = foot === null ? [] : [...foot.querySelectorAll<HTMLElement>(FOCUSABLE_IN_FOOT)];
        if (controls.length > 0 && foot !== null) {
          event.preventDefault();
          if (foot.contains(document.activeElement)) {
            const activeIdx = controls.indexOf(document.activeElement as HTMLElement);
            const nextIdx = event.shiftKey ? activeIdx - 1 : activeIdx + 1;
            if (activeIdx !== -1 && nextIdx >= 0 && nextIdx < controls.length) {
              controls[nextIdx]!.focus();
            } else {
              firstFocusableMenuItem(container.current)?.focus();
            }
          } else {
            const next = event.shiftKey ? controls[controls.length - 1] : controls[0];
            next?.focus();
          }
          return;
        }
      }
    }
    onKeyDown(event);
  };

  return (
    <Menu.Portal>
      <Menu.Content
        className={styles.menu}
        side={side}
        align={align}
        sideOffset={8}
        collisionPadding={12}
        loop
        aria-labelledby={undefined}
        aria-label={CHAT.model}
        onCloseAutoFocus={onCloseAutoFocus}
        onKeyDown={handleKeyDown}
        {...handlers}
      >
        <AutoHeight reduced={reduced} listRef={container} listId={listId} page={page} highlight={highlight}>
          {children}
        </AutoHeight>
        {footer}
      </Menu.Content>
    </Menu.Portal>
  );
}

function SubSurface({
  children,
  reduced,
  className,
  label,
}: {
  children: ReactNode;
  reduced: boolean;
  className?: string;
  label: string;
}): ReactElement {
  const { container, highlight, handlers } = useGlide(reduced);
  return (
    <Menu.Portal>
      <Menu.SubContent
        className={cn(styles.menu, className)}
        sideOffset={8}
        alignOffset={-7}
        collisionPadding={12}
        loop
        aria-labelledby={undefined}
        aria-label={label}
        {...handlers}
      >
        <div ref={container} className={styles.list}>
          {highlight}
          {children}
        </div>
      </Menu.SubContent>
    </Menu.Portal>
  );
}

/** Phones and narrow frames get in-menu pages instead of side submenus, which would leave the screen. */
const NARROW_QUERY = '(max-width: 640px)';
function useNarrow(): boolean {
  return useSyncExternalStore(
    (notify) => {
      if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') return () => {};
      const query = window.matchMedia(NARROW_QUERY);
      query.addEventListener('change', notify);
      return () => query.removeEventListener('change', notify);
    },
    () =>
      typeof window !== 'undefined' && typeof window.matchMedia === 'function'
        ? window.matchMedia(NARROW_QUERY).matches
        : false,
    () => false,
  );
}

function StateDot({ state }: { state: ModelBaseOption['state'] }): ReactElement | null {
  if (state !== 'loaded' && state !== 'loading') return null;
  return <span className={styles.stateDot} data-state={state} aria-hidden="true" />;
}

function VariantFacts({
  state,
  facts,
  warning,
}: {
  state: ModelVariantOption['state'];
  facts: readonly string[];
  warning?: string;
}): ReactElement | null {
  const hasState = state !== 'idle';
  const factsText = facts.join(' · ');
  if (!hasState && factsText === '' && warning === undefined) return null;
  return (
    <span className={styles.itemBlurb}>
      {state === 'loading' ? <LiveLabel active>{CHAT.modelLoading}</LiveLabel> : null}
      {state === 'loaded' ? CHAT.modelLoaded : null}
      {factsText !== '' ? (hasState ? ` · ${factsText}` : factsText) : null}
      {warning !== undefined ? (
        <span className={styles.warning}>
          {hasState || factsText !== '' ? ' · ' : ''}
          <TriangleAlert size={12} strokeWidth={1.75} aria-hidden="true" />
          {warning}
        </span>
      ) : null}
    </span>
  );
}

function BaseBlurb({
  note,
  facts,
  warning,
}: {
  note?: string;
  facts: readonly string[];
  warning?: string;
}): ReactElement | null {
  const lead = note !== undefined && note.trim() !== '' ? note : facts.length > 0 ? facts.join(' · ') : '';
  if (lead === '' && warning === undefined) return null;
  return (
    <span className={styles.itemBlurb}>
      {lead !== '' ? lead : null}
      {warning !== undefined ? (
        <span className={styles.warning}>
          {lead !== '' ? ' · ' : ''}
          <TriangleAlert size={12} strokeWidth={1.75} aria-hidden="true" />
          {warning}
        </span>
      ) : null}
    </span>
  );
}

/**
 * The model of the next message: one quiet pill that opens a short menu of recommended models with a one-line
 * note each, an Effort submenu for reasoning models, multi-build Quantization submenus, and every other model
 * under More models grouped by provider. On narrow viewports, submenus become in-menu pages with a Back row.
 */
export function ModelPicker(props: ModelPickerProps): ReactElement {
  const reduced = useStaticMotion();
  const [open, setOpen] = useState(false);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const footRef = useRef<HTMLDivElement>(null);
  const pointerToggled = useRef(false);
  const pointerCleanup = useRef<(() => void) | null>(null);
  const multiMouseDownBase = useRef<string | null>(null);
  const multiPointerCleanup = useRef<(() => void) | null>(null);
  const lastInput = useRef<'pointer' | 'keyboard'>('pointer');
  const narrow = useNarrow();
  const [page, setPage] = useState<Page>(MAIN_PAGE);
  /** Set once someone moves between pages, so opening the menu never plays a page transition. */
  const [paged, setPaged] = useState(false);
  const [slideDir, setSlideDir] = useState<1 | -1>(1);
  const previousPage = useRef<Page>(MAIN_PAGE);
  const listId = useId();

  const bases = props.bases;
  const empty = bases.length === 0 || props.value === null;
  const disabled = props.disabled === true || empty;
  const listDisabled = props.listDisabled === true;

  if (disabled && open) {
    setOpen(false);
    setPage(MAIN_PAGE);
    setPaged(false);
    setSlideDir(1);
  }
  const rootOpen = open && !disabled;

  const valueBase = holding(bases, props.value);
  const selected = valueBase?.variants.find((variant) => variant.id === props.value);
  const homeBase = valueBase ?? holding(bases, props.fallback) ?? bases[0];
  const homeVariant =
    selected ?? homeBase?.variants.find((variant) => variant.id === homeBase.preselect) ?? homeBase?.variants[0];
  const effortVariant = listDisabled ? selected : homeVariant;
  const effortBase = listDisabled ? valueBase : (valueBase ?? homeBase);
  const effort = levelsOf(effortVariant);
  const levelShown = keptLabel(selected);
  const effortShown = keptLabel(effortVariant);
  const currentLevel = levelsOf(selected)?.value;

  const hasRecommended = bases.some((base) => base.recommended === true);
  const outsideBases = useMemo(
    () => (hasRecommended ? bases.filter((base) => base.recommended === true) : bases),
    [bases, hasRecommended],
  );
  const moreBases = useMemo(
    () => (hasRecommended ? bases.filter((base) => base.recommended !== true) : []),
    [bases, hasRecommended],
  );
  const moreGroups = useMemo(() => groupByProvider(moreBases), [moreBases]);
  const hasMore = moreBases.length > 0;
  const inMore = valueBase !== undefined && moreBases.includes(valueBase);

  const variantById = useMemo(() => {
    const map = new Map<string, ModelVariantOption>();
    for (const base of bases) {
      for (const variant of base.variants) map.set(variant.id, variant);
    }
    return map;
  }, [bases]);

  const hasUnloaded = bases.some((option) => option.variants.some((variant) => variant.state !== 'loaded'));
  const isSwapWarning = !listDisabled && hasUnloaded && props.swapWarning !== undefined;
  const footerText =
    (listDisabled ? props.listDisabledReason : undefined) ??
    (hasUnloaded ? props.swapWarning : undefined) ??
    props.footer;

  const clearMultiMouseDown = (): void => {
    multiPointerCleanup.current?.();
    multiPointerCleanup.current = null;
    multiMouseDownBase.current = null;
  };

  useEffect(() => () => clearMultiMouseDown(), []);

  const changeOpen = (next: boolean): void => {
    if (next && disabled) return;
    if (next) {
      setPage(MAIN_PAGE);
      setPaged(false);
      setSlideDir(1);
      previousPage.current = MAIN_PAGE;
    } else {
      clearMultiMouseDown();
    }
    setOpen(next);
  };

  // Resolve the active narrow page, falling back cleanly if the backing data disappeared or list became disabled.
  const shown: Page = (() => {
    if (!narrow) return MAIN_PAGE;
    if (page.kind === 'effort') return effort !== undefined ? page : MAIN_PAGE;
    if (page.kind === 'more') return !listDisabled && hasMore ? page : MAIN_PAGE;
    if (page.kind === 'quant') {
      if (listDisabled) return MAIN_PAGE;
      const target = bases.find((base) => base.base === page.base);
      return target !== undefined && target.variants.length > 1 ? page : MAIN_PAGE;
    }
    return MAIN_PAGE;
  })();
  const shownKey = pageKey(shown);

  const moveTo = (next: Page): void => {
    setSlideDir(pageDepth(next) < pageDepth(shown) ? -1 : 1);
    setPaged(true);
    setPage(next);
  };

  // A new page focuses its checked or first item; going back focuses the row you came from.
  useEffect(() => {
    if (!rootOpen) {
      previousPage.current = MAIN_PAGE;
      return;
    }
    if (pageKey(shown) === pageKey(previousPage.current)) return;
    const from = previousPage.current;
    previousPage.current = shown;
    const frame = requestAnimationFrame(() => {
      const list = document.getElementById(listId);
      if (list === null) return;
      const goingBack = pageDepth(shown) < pageDepth(from);
      let target: HTMLElement | null = null;
      if (goingBack) {
        const wantRow = pageKey(from);
        for (const node of list.querySelectorAll<HTMLElement>('[data-row]:not([data-disabled])')) {
          if (node.dataset.row === wantRow) {
            target = node;
            break;
          }
        }
      }
      if (target === null) {
        target = firstFocusableMenuItem(list);
      }
      target?.focus({ preventScroll: false });
      target?.scrollIntoView({ block: 'nearest' });
    });
    return () => cancelAnimationFrame(frame);
  }, [shown, rootOpen, listId]);

  // Radix returns focus to the pill on close. After a pointer choice it should not draw the keyboard ring.
  useEffect(() => {
    const onPointer = (): void => {
      lastInput.current = 'pointer';
    };
    const onKey = (): void => {
      lastInput.current = 'keyboard';
    };
    window.addEventListener('pointerdown', onPointer, true);
    window.addEventListener('keydown', onKey, true);
    return () => {
      window.removeEventListener('pointerdown', onPointer, true);
      window.removeEventListener('keydown', onKey, true);
    };
  }, []);

  const onCloseAutoFocus = (event: Event): void => {
    if (lastInput.current !== 'pointer') return;
    event.preventDefault();
    triggerRef.current?.focus({ preventScroll: true, focusVisible: false } as FocusOptions);
  };

  /** Model and level together; nothing is sent when neither differs from the conversation's. */
  const commit = (variant: ModelVariantOption, explicitLevel?: string): void => {
    if (listDisabled && variant.id !== props.value) return;
    const variantEffort = levelsOf(variant);
    const nextLevel = variantEffort !== undefined ? (explicitLevel ?? variantEffort.value) : undefined;
    changeOpen(false);
    if (nextLevel === undefined) {
      if (variant.id !== props.value) props.onValueChange(variant.id);
    } else if (variant.id !== props.value || nextLevel !== currentLevel) {
      props.onValueChange(variant.id, nextLevel);
    }
  };

  const chooseVariantById = (id: string): void => {
    const variant = variantById.get(id);
    if (variant !== undefined) commit(variant);
  };

  const preselectOf = (base: ModelBaseOption): ModelVariantOption =>
    base.variants.find((variant) => variant.id === base.preselect) ?? base.variants[0]!;

  const onMultiBasePointerDown =
    (base: ModelBaseOption) =>
    (event: ReactPointerEvent<HTMLDivElement>): void => {
      clearMultiMouseDown();
      if (listDisabled || event.pointerType !== 'mouse' || event.button !== 0 || event.ctrlKey) return;
      multiMouseDownBase.current = base.base;
      let timer = 0;
      const removeListeners = (): void => {
        window.removeEventListener('pointerup', scheduleReset, true);
        window.removeEventListener('pointercancel', scheduleReset, true);
      };
      const scheduleReset = (): void => {
        removeListeners();
        timer = window.setTimeout(() => {
          multiMouseDownBase.current = null;
          multiPointerCleanup.current = null;
        }, 0);
      };
      window.addEventListener('pointerup', scheduleReset, true);
      window.addEventListener('pointercancel', scheduleReset, true);
      multiPointerCleanup.current = () => {
        removeListeners();
        window.clearTimeout(timer);
      };
    };

  const onMultiBaseClick =
    (base: ModelBaseOption) =>
    (event: MouseEvent<HTMLDivElement>): void => {
      const fromMouseDown = multiMouseDownBase.current === base.base;
      clearMultiMouseDown();
      if (listDisabled) return;
      // Fast 1-click pick of preselect only for real primary mouse clicks on this row; touch taps,
      // keyboard activation, and VoiceOver/AT synthetic clicks open the quantization submenu instead.
      if (fromMouseDown) {
        event.preventDefault();
        commit(preselectOf(base));
      }
    };

  const goTo =
    (next: Page) =>
    (event: Event): void => {
      event.preventDefault();
      moveTo(next);
    };

  const renderQuantList = (base: ModelBaseOption): ReactElement => (
    <Menu.RadioGroup value={props.value ?? ''} onValueChange={chooseVariantById}>
      {base.variants.map((variant) => (
        <Menu.RadioItem
          key={variant.id}
          value={variant.id}
          disabled={listDisabled}
          textValue={variant.label}
          className={cn(styles.item, styles.quantItem)}
        >
          <span className={styles.itemText}>
            <span className={styles.itemName}>
              <span className={styles.itemNameText}>{variant.label}</span>
            </span>
            <VariantFacts state={variant.state} facts={variant.facts} warning={variant.warning} />
          </span>
          <Menu.ItemIndicator className={styles.check}>
            <Check aria-hidden="true" />
          </Menu.ItemIndicator>
        </Menu.RadioItem>
      ))}
    </Menu.RadioGroup>
  );

  const renderOutsideBase = (base: ModelBaseOption): ReactElement => {
    const single = base.variants.length === 1 ? base.variants[0] : undefined;
    const isSelected = base === valueBase;
    const content = (
      <>
        <ModelLogo model={base.base} className={styles.itemMark} />
        <span className={styles.itemText}>
          <span className={styles.itemName}>
            <span className={styles.itemNameText}>{base.base}</span>
            <StateDot state={base.state} />
          </span>
          <BaseBlurb note={base.note} facts={base.facts} warning={base.warning} />
        </span>
      </>
    );

    if (single !== undefined) {
      return (
        <Menu.RadioItem
          key={base.base}
          value={single.id}
          disabled={listDisabled}
          textValue={base.base}
          className={styles.item}
        >
          {content}
          <Menu.ItemIndicator className={styles.check}>
            <Check aria-hidden="true" />
          </Menu.ItemIndicator>
        </Menu.RadioItem>
      );
    }

    if (narrow) {
      return (
        <Menu.Item
          key={base.base}
          disabled={listDisabled}
          className={styles.item}
          data-row={`quant:${base.base}`}
          data-checked={isSelected ? '' : undefined}
          textValue={base.base}
          onSelect={goTo({ kind: 'quant', base: base.base, from: 'main' })}
        >
          {content}
          {isSelected ? (
            <span className={styles.check}>
              <Check aria-hidden="true" />
            </span>
          ) : null}
          <ChevronRight className={styles.rowChevron} aria-hidden="true" />
        </Menu.Item>
      );
    }

    return (
      <GuardedSub key={base.base} rootOpen={rootOpen}>
        <Menu.SubTrigger
          disabled={listDisabled}
          className={styles.item}
          textValue={base.base}
          data-checked={isSelected ? '' : undefined}
          onPointerDown={onMultiBasePointerDown(base)}
          onClick={onMultiBaseClick(base)}
        >
          {content}
          {isSelected ? (
            <span className={styles.check}>
              <Check aria-hidden="true" />
            </span>
          ) : null}
          <ChevronRight className={styles.rowChevron} aria-hidden="true" />
        </Menu.SubTrigger>
        <SubSurface
          reduced={reduced}
          className={styles.quantMenu}
          label={fill(CHAT.modelQuantList, { base: base.base })}
        >
          {renderQuantList(base)}
        </SubSurface>
      </GuardedSub>
    );
  };

  const renderMoreBase = (base: ModelBaseOption): ReactElement => {
    const single = base.variants.length === 1 ? base.variants[0] : undefined;
    const isSelected = base === valueBase;
    const content = (
      <>
        <ModelLogo model={base.base} className={styles.compactMark} />
        <span className={styles.compactName}>
          <span className={styles.compactNameText}>{base.base}</span>
          <StateDot state={base.state} />
        </span>
        {base.warning !== undefined ? (
          <span className={cn(styles.tag, styles.warning)}>
            <TriangleAlert size={11} strokeWidth={1.75} aria-hidden="true" />
            {base.warning}
          </span>
        ) : null}
      </>
    );

    if (single !== undefined) {
      return (
        <Menu.RadioItem
          key={base.base}
          value={single.id}
          disabled={listDisabled}
          textValue={base.base}
          className={cn(styles.item, styles.compact)}
        >
          {content}
          <Menu.ItemIndicator className={styles.check}>
            <Check aria-hidden="true" />
          </Menu.ItemIndicator>
        </Menu.RadioItem>
      );
    }

    if (narrow) {
      return (
        <Menu.Item
          key={base.base}
          disabled={listDisabled}
          className={cn(styles.item, styles.compact)}
          data-row={`quant:${base.base}`}
          data-checked={isSelected ? '' : undefined}
          textValue={base.base}
          onSelect={goTo({ kind: 'quant', base: base.base, from: 'more' })}
        >
          {content}
          {isSelected ? (
            <span className={styles.check}>
              <Check aria-hidden="true" />
            </span>
          ) : null}
          <ChevronRight className={styles.rowChevron} aria-hidden="true" />
        </Menu.Item>
      );
    }

    return (
      <GuardedSub key={base.base} rootOpen={rootOpen}>
        <Menu.SubTrigger
          disabled={listDisabled}
          className={cn(styles.item, styles.compact)}
          textValue={base.base}
          data-checked={isSelected ? '' : undefined}
          onPointerDown={onMultiBasePointerDown(base)}
          onClick={onMultiBaseClick(base)}
        >
          {content}
          {isSelected ? (
            <span className={styles.check}>
              <Check aria-hidden="true" />
            </span>
          ) : null}
          <ChevronRight className={styles.rowChevron} aria-hidden="true" />
        </Menu.SubTrigger>
        <SubSurface
          reduced={reduced}
          className={styles.quantMenu}
          label={fill(CHAT.modelQuantList, { base: base.base })}
        >
          {renderQuantList(base)}
        </SubSurface>
      </GuardedSub>
    );
  };

  const effortGroup =
    effort !== undefined && effortVariant !== undefined ? (
      <>
        <Menu.RadioGroup value={effort.value} onValueChange={(lvl) => commit(effortVariant, lvl)}>
          {effort.levels.map((item) => (
            <Menu.RadioItem
              key={item.value}
              value={item.value}
              textValue={item.label}
              className={cn(styles.item, styles.effortItem)}
            >
              <span className={styles.itemText}>
                <span className={styles.itemName}>
                  {item.label}
                  {item.value === effort.defaultValue ? (
                    <span className={styles.default}>{CHAT.thinkingLevel.defaultTag}</span>
                  ) : null}
                </span>
                {item.note !== undefined ? <span className={styles.itemBlurb}>{item.note}</span> : null}
              </span>
              <Menu.ItemIndicator className={styles.check}>
                <Check aria-hidden="true" />
              </Menu.ItemIndicator>
            </Menu.RadioItem>
          ))}
        </Menu.RadioGroup>
        {effort.note !== undefined ? <p className={styles.subNote}>{effort.note}</p> : null}
      </>
    ) : null;

  const moreGroup = (
    <Menu.RadioGroup value={props.value ?? ''} onValueChange={chooseVariantById}>
      {moreGroups.map((group) => (
        <Fragment key={group.provider}>
          <Menu.Label className={styles.groupLabel}>{group.provider}</Menu.Label>
          {group.bases.map(renderMoreBase)}
        </Fragment>
      ))}
    </Menu.RadioGroup>
  );

  const effortRow =
    effort !== undefined ? (
      <>
        <Gauge className={styles.rowIcon} aria-hidden="true" />
        <span className={styles.rowLabel}>{CHAT.thinkingLevel.effort}</span>
        {effortShown !== null ? (
          <span className={styles.rowValue}>
            {effortShown}
            {props.levelWarning === true ? <span className={styles.levelWarn} aria-hidden="true" /> : null}
          </span>
        ) : null}
        <ChevronRight className={styles.rowChevron} aria-hidden="true" />
      </>
    ) : null;

  const moreRow = (
    <>
      <Layers className={styles.rowIcon} aria-hidden="true" />
      <span className={styles.rowLabel}>{CHAT.modelMore}</span>
      {inMore && valueBase !== undefined ? <span className={styles.rowValue}>{valueBase.base}</span> : null}
      <ChevronRight className={styles.rowChevron} aria-hidden="true" />
    </>
  );

  const back = (title: string, target: Page): ReactElement => (
    <>
      <Menu.Item
        className={cn(styles.row, styles.back)}
        data-back=""
        textValue="Back"
        aria-label={`Back, ${title}`}
        onSelect={goTo(target)}
      >
        <ChevronLeft className={styles.rowIcon} aria-hidden="true" />
        <span className={styles.rowLabel}>{title}</span>
      </Menu.Item>
      <Menu.Separator className={styles.separator} />
    </>
  );

  const mainPage = (
    <>
      <Menu.RadioGroup value={props.value ?? ''} onValueChange={chooseVariantById}>
        {outsideBases.map(renderOutsideBase)}
      </Menu.RadioGroup>

      {effortGroup !== null || hasMore ? <Menu.Separator className={styles.separator} /> : null}

      {effortGroup !== null && effortRow !== null ? (
        narrow ? (
          <Menu.Item
            className={styles.row}
            data-row="effort"
            textValue={CHAT.thinkingLevel.effort}
            onSelect={goTo({ kind: 'effort' })}
          >
            {effortRow}
          </Menu.Item>
        ) : (
          <GuardedSub rootOpen={rootOpen}>
            <Menu.SubTrigger className={styles.row} textValue={CHAT.thinkingLevel.effort}>
              {effortRow}
            </Menu.SubTrigger>
            <SubSurface
              reduced={reduced}
              label={fill(CHAT.thinkingLevel.list, { model: effortBase?.base ?? '' })}
              className={styles.effortMenu}
            >
              {effortGroup}
            </SubSurface>
          </GuardedSub>
        )
      ) : null}

      {hasMore ? (
        narrow ? (
          <Menu.Item
            disabled={listDisabled}
            className={styles.row}
            data-row="more"
            textValue={CHAT.modelMore}
            onSelect={goTo({ kind: 'more' })}
          >
            {moreRow}
          </Menu.Item>
        ) : (
          <GuardedSub rootOpen={rootOpen}>
            <Menu.SubTrigger disabled={listDisabled} className={styles.row} textValue={CHAT.modelMore}>
              {moreRow}
            </Menu.SubTrigger>
            <SubSurface reduced={reduced} label={CHAT.modelMoreList} className={styles.moreMenu}>
              {moreGroup}
            </SubSurface>
          </GuardedSub>
        )
      ) : null}
    </>
  );

  const body = (() => {
    if (shown.kind === 'effort' && effortGroup !== null) {
      return (
        <>
          {back(CHAT.thinkingLevel.effort, MAIN_PAGE)}
          {effortGroup}
        </>
      );
    }
    if (shown.kind === 'more') {
      return (
        <>
          {back(CHAT.modelMore, MAIN_PAGE)}
          {moreGroup}
        </>
      );
    }
    if (shown.kind === 'quant') {
      const targetBase = bases.find((base) => base.base === shown.base);
      if (targetBase !== undefined) {
        return (
          <>
            {back(targetBase.base, shown.from === 'more' ? { kind: 'more' } : MAIN_PAGE)}
            {renderQuantList(targetBase)}
          </>
        );
      }
    }
    return mainPage;
  })();

  // Arrow keys move between pages on narrow viewports, the way they open and close submenus on desktop.
  const onMenuKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (!narrow) return;
    if (shown.kind !== 'main' && event.key === 'ArrowLeft') {
      event.preventDefault();
      moveTo(shown.kind === 'quant' && shown.from === 'more' && !listDisabled ? { kind: 'more' } : MAIN_PAGE);
      return;
    }
    if (event.key !== 'ArrowRight') return;
    const target = event.target instanceof HTMLElement ? event.target : null;
    if (target === null || target.hasAttribute('data-disabled')) return;
    const row = target.dataset.row;
    if (row === undefined) return;
    if (shown.kind === 'main' && row === 'effort') {
      event.preventDefault();
      moveTo({ kind: 'effort' });
    } else if (!listDisabled && shown.kind === 'main' && row === 'more') {
      event.preventDefault();
      moveTo({ kind: 'more' });
    } else if (!listDisabled && (shown.kind === 'main' || shown.kind === 'more') && row.startsWith('quant:')) {
      event.preventDefault();
      moveTo({ kind: 'quant', base: row.slice('quant:'.length), from: shown.kind });
    }
  };

  const onFootKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.key !== 'Escape' && event.key !== 'Tab') {
      event.stopPropagation();
    }
  };

  const footerSection =
    shown.kind === 'main' && (footerText !== undefined || props.memory !== undefined) ? (
      <div ref={footRef} className={styles.foot} onKeyDown={onFootKeyDown}>
        {footerText !== undefined ? (
          <p className={styles.footer} data-warning={isSwapWarning ? '' : undefined}>
            {isSwapWarning ? <TriangleAlert size={13} strokeWidth={1.75} aria-hidden="true" /> : null}
            <span>{footerText}</span>
          </p>
        ) : null}
        {props.memory !== undefined ? (
          <div className={styles.memory}>
            <span className={styles.memoryText}>{props.memory.text}</span>
            {props.memory.action}
          </div>
        ) : null}
      </div>
    ) : null;

  const displayName = valueBase?.base ?? props.value ?? '';
  const triggerTitle = empty
    ? undefined
    : levelShown === null
      ? (props.value ?? undefined)
      : fill(CHAT.thinkingLevel.chip, { model: props.value ?? '', level: levelShown });
  const triggerAriaLabel = empty
    ? CHAT.noModel
    : withLevel(
        selected === undefined || valueBase === undefined
          ? `${CHAT.model}: ${props.value}`
          : fill(CHAT.pillName, {
              model: `${valueBase.base} ${selected.label}`,
              state: CHAT.pillStates[selected.state],
            }),
        levelShown,
      );

  const clearPointerToggle = (): void => {
    pointerCleanup.current?.();
    pointerCleanup.current = null;
    pointerToggled.current = false;
  };

  useEffect(() => () => clearPointerToggle(), []);

  // Radix DropdownMenuTrigger only toggles on pointerdown / keydown (no onClick). Track pointerdown so
  // programmatic .click() and assistive tech toggle the menu without double-toggling real pointer clicks.
  const onTriggerPointerDown = (event: ReactPointerEvent<HTMLButtonElement>): void => {
    if (disabled || event.button !== 0 || event.ctrlKey) return;
    clearPointerToggle();
    pointerToggled.current = true;
    let timer = 0;
    const removeListeners = (): void => {
      window.removeEventListener('pointerup', scheduleReset, true);
      window.removeEventListener('pointercancel', scheduleReset, true);
    };
    const scheduleReset = (): void => {
      removeListeners();
      timer = window.setTimeout(() => {
        pointerToggled.current = false;
        pointerCleanup.current = null;
      }, 0);
    };
    window.addEventListener('pointerup', scheduleReset, true);
    window.addEventListener('pointercancel', scheduleReset, true);
    pointerCleanup.current = () => {
      removeListeners();
      window.clearTimeout(timer);
    };
  };

  const onTriggerClick = (event: MouseEvent<HTMLButtonElement>): void => {
    if (pointerToggled.current) {
      clearPointerToggle();
      return;
    }
    if (disabled || event.ctrlKey || event.detail !== 0) return;
    changeOpen(!open);
  };

  const triggerNode = (
    <Menu.Trigger
      ref={triggerRef}
      className={styles.trigger}
      disabled={disabled}
      data-state={selected?.state ?? undefined}
      data-crowded={props.crowded === true ? '' : undefined}
      title={triggerTitle}
      aria-label={triggerAriaLabel}
      onPointerDown={onTriggerPointerDown}
      onClick={onTriggerClick}
    >
      {empty ? (
        <span className={styles.none}>{CHAT.noModel}</span>
      ) : (
        <TriggerLabel
          modelId={props.value ?? ''}
          displayName={displayName}
          level={levelShown}
          levelWarning={props.levelWarning === true}
          crowded={props.crowded === true}
          reduced={reduced}
        />
      )}
      <span className={styles.chevronSlot} aria-hidden="true">
        {props.working === true ? (
          <MorphLoader size={14} status="loading" />
        ) : (
          <ChevronDown className={styles.chevron} aria-hidden="true" />
        )}
      </span>
    </Menu.Trigger>
  );

  return (
    <Menu.Root open={rootOpen} onOpenChange={changeOpen}>
      {props.disabled === true && props.disabledReason !== undefined ? (
        <Tooltip content={props.disabledReason}>
          <span className={styles.tipTarget}>{triggerNode}</span>
        </Tooltip>
      ) : (
        triggerNode
      )}

      <MenuSurface
        reduced={reduced}
        page={shownKey}
        listId={listId}
        side={props.side ?? 'top'}
        align={props.align ?? 'start'}
        footer={footerSection}
        footRef={footRef}
        onCloseAutoFocus={onCloseAutoFocus}
        onKeyDown={onMenuKeyDown}
      >
        <motion.div
          key={shownKey}
          className={styles.page}
          initial={reduced || !paged ? false : { opacity: 0, x: slideDir * 12 }}
          animate={{ opacity: 1, x: 0 }}
          transition={reduced ? INSTANT : { duration: duration.fast, ease: standard }}
        >
          {body}
        </motion.div>
      </MenuSurface>
    </Menu.Root>
  );
}

export default ModelPicker;

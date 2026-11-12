import { AnimatePresence, motion, useInView } from 'motion/react';
import {
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type CSSProperties,
  type FocusEvent,
  type KeyboardEvent,
  type ReactElement,
  type ReactNode,
} from 'react';

import { cn } from '../../../lib/cn';
import { NO_VALUE } from '../../../lib/format';
import { KeyValue } from '../key-value/key-value';
import { LiveDot } from '../live-dot/live-dot';
import { MiniSpark } from '../mini-spark/mini-spark';
import { motionX } from '../motion/motion';
import { useRichMotion } from '../motion/use-entrance';
import { NumberValue } from '../number-value/number-value';
import { Placeholder } from '../placeholder/placeholder';
import styles from './metric-ticker.module.css';

export interface TickerDelta {
  /** Signed, formatted change against the sample one minute earlier: "+4.2", "−1". */
  text: string;
  direction: 'up' | 'down' | 'flat';
  goodWhen: 'up' | 'down' | 'neutral';
}

export interface TickerItem {
  key: string;
  /** "Output rate" */
  label: string;
  /** Formatted; null -> "—". */
  value: string | null;
  unit?: string;
  /** null or absent: no chip. */
  delta?: TickerDelta | null;
  /** Hover / focus panel. */
  spark?: { data: readonly (number | null)[]; label: string };
  facts?: ReadonlyArray<{ label: string; value: string }>;
  /** e.g. <Meter/> */
  footer?: ReactNode;
}

export interface MetricTickerProps {
  /** "Now" */
  label: string;
  /** ≤ 8 */
  items: readonly TickerItem[];
  loading?: boolean;
  /** Set when the values are a last heartbeat, e.g. "12 s ago": shown first in the strip; nothing rolls or flashes. */
  stale?: string;
  className?: string;
}

const MAX_ITEMS = 8;
const LOADING_SLOTS = 6;
const PANEL_GAP_PX = 8;
const PANEL_WIDTH_PX = 248;
const GLYPH: Record<TickerDelta['direction'], string> = { up: '↗', down: '↘', flat: '→' };

/** Width of one unit glyph in `ch` of the value: `.unit` is --text-xs inside the --text-lg `.valueNode`; keep in step. */
const UNIT_CH = 0.7;
/** Chip text characters a slot holds before any chip came. */
const CHIP_SLOT_CHARS = 4;

/** The least width of a chip slot in `ch` of the chip font: glyph, gap, text and the `.chip` padding; keep in step. */
function chipReserve(chars: number): string {
  return `calc(${String(chars + 1)}ch + var(--space-0-5) + var(--space-1-5) + var(--space-1))`;
}

/** The reserve of a value and its unit together, so a narrower value leaves its free space after the unit. */
function groupReserve(chars: number, unit: string | undefined): string {
  const unitCh = unit === undefined || unit === '' ? 0 : 0.5 + unit.length * UNIT_CH;
  return `${String(chars + unitCh)}ch`;
}

function hasPanel(item: TickerItem): boolean {
  return item.spark !== undefined || (item.facts !== undefined && item.facts.length > 0) || item.footer !== undefined;
}

function unsigned(text: string): string {
  const first = text[0];
  return first === '+' || first === '-' || first === '−' ? text.slice(1) : text;
}

/** "up 4.2 in the last minute" */
export function tickerDeltaLabel(delta: TickerDelta): string {
  if (delta.direction === 'flat') return 'no change in the last minute';
  return `${delta.direction} ${unsigned(delta.text)} in the last minute`;
}

function deltaTone(delta: TickerDelta): 'good' | 'bad' | 'neutral' {
  if (delta.direction === 'flat' || delta.goodWhen === 'neutral') return 'neutral';
  return delta.direction === delta.goodWhen ? 'good' : 'bad';
}

/** One strip of live KPIs; drifts sideways when the items do not fit. */
export function MetricTicker(props: MetricTickerProps): ReactElement {
  const rich = useRichMotion();
  const id = useId();
  const panelId = `${id}-panel`;
  const rootRef = useRef<HTMLElement | null>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const groupRef = useRef<HTMLDivElement | null>(null);
  const trackRef = useRef<HTMLDivElement | null>(null);
  const inView = useInView(rootRef);
  const [overflow, setOverflow] = useState(false);
  const [hovered, setHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const [openKey, setOpenKey] = useState<string | null>(null);
  const [panelX, setPanelX] = useState(0);
  const hoverTimer = useRef<number | undefined>(undefined);

  const loading = props.loading === true;
  const stale = props.stale !== undefined;
  const items = props.items.slice(0, MAX_ITEMS);
  const slots: readonly TickerItem[] =
    loading && items.length === 0
      ? Array.from({ length: LOADING_SLOTS }, (_, index) => ({ key: `slot-${String(index)}`, label: '', value: null }))
      : items;
  const drift = overflow && rich && !stale;
  const moving = drift && inView && !hovered && !focused && openKey === null;
  const openItem = openKey === null ? undefined : items.find((item) => item.key === openKey);

  useLayoutEffect(() => {
    const viewport = viewportRef.current;
    const group = groupRef.current;
    if (viewport === null || group === null) return undefined;
    const measure = (): void => {
      const width = group.scrollWidth;
      setOverflow(width > viewport.clientWidth + 1);
      trackRef.current?.style.setProperty('--group-w', `${String(width)}px`);
      trackRef.current?.style.setProperty('--drift-time', `${String(width / motionX.drift)}s`);
    };
    measure();
    if (typeof ResizeObserver === 'undefined') return undefined;
    const observer = new ResizeObserver(measure);
    observer.observe(viewport);
    observer.observe(group);
    return () => observer.disconnect();
  }, [slots.length, stale]);

  useEffect(() => () => window.clearTimeout(hoverTimer.current), []);

  const placePanel = (button: HTMLElement): void => {
    const root = rootRef.current;
    if (root === null) return;
    const rootBox = root.getBoundingClientRect();
    const itemBox = button.getBoundingClientRect();
    const width = Math.min(PANEL_WIDTH_PX, rootBox.width);
    setPanelX(Math.max(0, Math.min(itemBox.left - rootBox.left, rootBox.width - width)));
  };

  const openFor = (item: TickerItem, button: HTMLElement): void => {
    if (!hasPanel(item)) return;
    placePanel(button);
    setOpenKey(item.key);
  };

  const onItemKey = (event: KeyboardEvent<HTMLButtonElement>, index: number): void => {
    if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
    const buttons = groupRef.current?.querySelectorAll<HTMLButtonElement>('button[data-ticker-item]');
    if (buttons === undefined) return;
    const next = buttons[index + (event.key === 'ArrowRight' ? 1 : -1)];
    if (next === undefined) return;
    event.preventDefault();
    next.focus();
    next.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  };

  const onBlurRoot = (event: FocusEvent<HTMLElement>): void => {
    const next = event.relatedTarget;
    if (next instanceof Node && rootRef.current?.contains(next) === true) return;
    setFocused(false);
    setOpenKey(null);
  };

  const renderItems = (copy: boolean): ReactNode =>
    slots.map((item, index) => (
      <TickerCell
        key={item.key}
        item={item}
        index={index}
        copy={copy}
        loading={loading}
        stale={stale}
        rich={rich}
        open={openKey === item.key}
        panelId={panelId}
        onOpen={(button) => openFor(item, button)}
        onHover={(button) => {
          window.clearTimeout(hoverTimer.current);
          if (!hasPanel(item)) return;
          hoverTimer.current = window.setTimeout(() => openFor(item, button), motionX.hoverHold * 1000);
        }}
        onKey={(event) => onItemKey(event, index)}
      />
    ));

  return (
    <section
      ref={rootRef}
      role="group"
      aria-label={props.label}
      aria-busy={loading ? true : undefined}
      className={cn(styles.root, props.className)}
      data-mode={overflow ? 'drift' : 'fit'}
      data-moving={moving ? '' : undefined}
      data-static={rich ? undefined : ''}
      data-stale={stale ? '' : undefined}
      onPointerEnter={() => setHovered(true)}
      onPointerLeave={() => {
        setHovered(false);
        window.clearTimeout(hoverTimer.current);
        setOpenKey(null);
      }}
      onFocus={() => setFocused(true)}
      onBlur={onBlurRoot}
      onKeyDown={(event) => {
        if (event.key !== 'Escape' || openKey === null) return;
        event.stopPropagation();
        setOpenKey(null);
      }}
    >
      <div ref={viewportRef} className={styles.viewport}>
        <div ref={trackRef} className={styles.track}>
          {stale ? (
            <div className={styles.stale}>
              <LiveDot tone="warning" size={8} label="Last heartbeat" showLabel />
              <span className={styles.staleText}>{props.stale}</span>
            </div>
          ) : null}
          <div ref={groupRef} className={styles.group}>
            {renderItems(false)}
          </div>
          {drift ? (
            <div className={styles.group} aria-hidden="true" inert>
              {renderItems(true)}
            </div>
          ) : null}
        </div>
      </div>
      <AnimatePresence>
        {openItem !== undefined ? (
          <motion.div
            key="panel"
            id={panelId}
            role="group"
            aria-label={`${openItem.label} details`}
            className={styles.panel}
            data-pop={rich ? '' : undefined}
            style={{ '--panel-gap': `${String(PANEL_GAP_PX)}px` } as CSSProperties}
            initial={{ x: panelX }}
            animate={{ x: panelX }}
            exit={{ opacity: 0, transition: rich ? motionX.swap.out : { duration: 0 } }}
            transition={rich ? motionX.spring.morph : { duration: 0 }}
            onPointerEnter={() => window.clearTimeout(hoverTimer.current)}
          >
            <div key={openItem.key} className={styles.panelBody}>
              {openItem.spark !== undefined ? (
                <div className={styles.spark}>
                  <MiniSpark data={openItem.spark.data} label={openItem.spark.label} width={220} height={56} />
                  <span className={styles.sparkCaption}>Last 10 minutes</span>
                </div>
              ) : null}
              {openItem.facts !== undefined && openItem.facts.length > 0 ? (
                <KeyValue
                  layout="rows"
                  ariaLabel={`${openItem.label} facts`}
                  items={openItem.facts.map((fact) => ({
                    key: fact.label,
                    label: fact.label,
                    value: fact.value,
                    numeric: true,
                  }))}
                />
              ) : null}
              {openItem.footer !== undefined ? <div className={styles.panelFooter}>{openItem.footer}</div> : null}
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>
    </section>
  );
}

interface TickerCellProps {
  item: TickerItem;
  index: number;
  copy: boolean;
  loading: boolean;
  stale: boolean;
  rich: boolean;
  open: boolean;
  panelId: string;
  onOpen: (button: HTMLElement) => void;
  onHover: (button: HTMLElement) => void;
  onKey: (event: KeyboardEvent<HTMLButtonElement>) => void;
}

function TickerCell(props: TickerCellProps): ReactElement {
  const { item } = props;
  const measured = item.value !== null && item.value !== NO_VALUE;
  const [widest, setWidest] = useState(item.value?.length ?? 1);
  if ((item.value?.length ?? 0) > widest) setWidest(item.value?.length ?? 0);
  const slotted = 'delta' in item;
  const chipChars = item.delta?.text.length ?? 0;
  const [widestChip, setWidestChip] = useState(Math.max(CHIP_SLOT_CHARS, chipChars));
  if (chipChars > widestChip) setWidestChip(chipChars);
  const [seen, setSeen] = useState(item.value);
  const [flashing, setFlashing] = useState(false);
  if (seen !== item.value) {
    setSeen(item.value);
    if (props.rich && !props.stale && !props.copy && seen !== null && measured) setFlashing(true);
  }
  useEffect(() => {
    if (!flashing) return undefined;
    const timer = window.setTimeout(() => setFlashing(false), motionX.flashHold * 1000);
    return () => window.clearTimeout(timer);
  }, [flashing, item.value]);

  const panel = hasPanel(item);
  const delta = measured && !props.stale && item.delta !== undefined && item.delta !== null ? item.delta : null;
  const reading = (
    <span className={styles.reading}>
      <span
        className={styles.valueNode}
        style={{ minWidth: groupReserve(widest, measured ? item.unit : undefined) }}
        data-stat-value={props.copy ? undefined : ''}
      >
        <NumberValue text={item.value} unit={item.unit} className={styles.value} />
        {measured && item.unit !== undefined && item.unit.length > 0 ? (
          <>
            {' '}
            <span className={styles.unit}>{item.unit}</span>
          </>
        ) : null}
      </span>
      {slotted ? (
        <span className={styles.chipSlot} style={{ minWidth: chipReserve(widestChip) }} data-chip-slot="">
          {delta !== null ? (
            <span
              className={styles.chip}
              data-tone={deltaTone(delta)}
              data-flash={flashing ? '' : undefined}
              aria-label={tickerDeltaLabel(delta)}
            >
              <span className={styles.glyph} aria-hidden="true">
                {GLYPH[delta.direction]}
              </span>
              <span aria-hidden="true">{delta.text}</span>
            </span>
          ) : null}
        </span>
      ) : null}
    </span>
  );

  return (
    <div
      role={props.copy ? undefined : 'group'}
      aria-label={props.copy || props.loading ? undefined : item.label}
      className={styles.item}
      data-open={props.open ? '' : undefined}
    >
      {props.loading ? null : (
        <button
          type="button"
          className={styles.button}
          tabIndex={props.copy ? -1 : undefined}
          data-ticker-item={props.copy ? undefined : ''}
          aria-label={`${item.label} details`}
          aria-expanded={panel ? props.open : undefined}
          aria-controls={panel && props.open ? props.panelId : undefined}
          onClick={(event) => props.onOpen(event.currentTarget)}
          onFocus={(event) => props.onOpen(event.currentTarget)}
          onPointerEnter={(event) => props.onHover(event.currentTarget)}
          onKeyDown={props.onKey}
        />
      )}
      <span className={styles.label}>
        <Placeholder loading={props.loading} width="9ch" step={props.index}>
          {item.label}
        </Placeholder>
      </span>
      <Placeholder loading={props.loading} width="7ch" step={props.index}>
        {reading}
      </Placeholder>
    </div>
  );
}

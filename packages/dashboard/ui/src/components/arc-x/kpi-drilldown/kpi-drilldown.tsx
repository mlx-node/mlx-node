import { Minimize2 } from 'lucide-react';
import { animate } from 'motion/react';
import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactElement,
  type ReactNode,
  type RefObject,
} from 'react';

import { cn } from '../../../lib/cn';
import { ScrollArea } from '../../arc/scroll-area/scroll-area';
import { IconButton } from '../icon-button/icon-button';
import { motionX } from '../motion/motion';
import { useStaticMotion } from '../motion/use-static-motion';
import { SampleChip, StatDeltaLine, StatReading, type StatDelta } from '../stat-tile/stat-tile';
import { DrilldownContext, type DrilldownContextValue } from './drilldown-context';
import styles from './kpi-drilldown.module.css';

export interface KpiDetail {
  /** "Decode speed" */
  label: string;
  /** Formatted, the same string as the tile. */
  value: string | null;
  unit?: string;
  delta?: StatDelta;
  sampleCount?: number;
  /** One sentence: what the number is and over which window. */
  description: string;
  /** Body: the trend with the previous period, the breakdown. */
  children: ReactNode;
  /** Links out, right-aligned: "Open in Metrics", "Show turns". */
  footer?: ReactNode;
}

export interface KpiDrilldownProps {
  /** Names the stage, e.g. "Last 24 hours". */
  label: string;
  /** `detailKey` of the open tile, or null. Controlled: the page keeps it in `?detail=`. */
  open: string | null;
  onOpenChange: (key: string | null) => void;
  /** Called for the open key only. null = unknown key: the panel does not open and the page drops the param. */
  detail: (key: string) => KpiDetail | null;
  /** The stage: one `StatGroup` whose tiles carry `detailKey`, then the block(s) the panel may cover. */
  children: ReactNode;
  className?: string;
}

interface Box {
  top: number;
  left: number;
  width: number;
  height: number;
}

const BOX_PROPS = ['top', 'left', 'width', 'height'] as const;

function relativeBox(element: Element, stage: Element): Box {
  const inner = element.getBoundingClientRect();
  const outer = stage.getBoundingClientRect();
  return { top: inner.top - outer.top, left: inner.left - outer.left, width: inner.width, height: inner.height };
}

function placeAt(panel: HTMLElement, box: Box): void {
  for (const prop of BOX_PROPS) panel.style.setProperty(prop, `${String(box[prop])}px`);
}

function clearPlace(panel: HTMLElement): void {
  for (const prop of BOX_PROPS) panel.style.removeProperty(prop);
}

/** A KPI tile that grows into a detail panel over its own row and the block under it. */
export function KpiDrilldown(props: KpiDrilldownProps): ReactElement {
  const { onOpenChange } = props;
  const staticMotion = useStaticMotion();
  const panelId = useId();
  const stageRef = useRef<HTMLDivElement | null>(null);
  const triggers = useRef(new Map<string, HTMLElement>());
  const tiles = useRef(new Map<string, HTMLElement>());
  /** The tile rectangle of the click that opened the panel; consumed by the panel's first layout. */
  const travelFrom = useRef<Box | null>(null);
  const focusOnOpen = useRef(false);

  const resolved = props.open !== null ? props.detail(props.open) : null;
  const target = resolved !== null ? props.open : null;
  const [shown, setShown] = useState<string | null>(target);
  const [closing, setClosing] = useState(false);
  if (target !== null) {
    if (shown !== target) setShown(target);
    if (closing) setClosing(false);
  } else if (shown !== null && !closing) {
    if (staticMotion) setShown(null);
    else setClosing(true);
  }

  const unknownKey = props.open !== null && resolved === null;
  useEffect(() => {
    if (unknownKey) onOpenChange(null);
  }, [unknownKey, onOpenChange]);

  const shownDetail = shown === null ? null : shown === target ? resolved : props.detail(shown);

  const tileFor = useCallback((key: string): HTMLElement | null => {
    const tile = tiles.current.get(key);
    if (tile !== undefined && tile.isConnected) return tile;
    return triggers.current.get(key)?.closest('section') ?? null;
  }, []);

  const context = useMemo<DrilldownContextValue>(
    () => ({
      open: shown,
      panelId,
      toggle: (key, tile) => {
        tiles.current.set(key, tile);
        const stage = stageRef.current;
        travelFrom.current = stage !== null && !staticMotion ? relativeBox(tile, stage) : null;
        focusOnOpen.current = true;
        onOpenChange(key);
      },
      registerTrigger: (key, trigger) => {
        if (trigger === null) triggers.current.delete(key);
        else triggers.current.set(key, trigger);
      },
    }),
    [shown, panelId, staticMotion, onOpenChange],
  );

  const finishClose = useCallback(() => {
    setClosing(false);
    setShown(null);
  }, []);

  const open = shown !== null && shownDetail !== null;
  return (
    <DrilldownContext.Provider value={context}>
      <div
        ref={stageRef}
        className={cn(styles.stage, props.className)}
        data-open={open ? '' : undefined}
        data-closing={open && closing ? '' : undefined}
      >
        <div className={styles.content} inert={open && !closing}>
          {props.children}
        </div>
        {open ? (
          <>
            <div className={styles.veil} aria-hidden="true" />
            <DrillPanel
              key="panel"
              id={panelId}
              detailKey={shown}
              detail={shownDetail}
              closing={closing}
              staticMotion={staticMotion}
              stageRef={stageRef}
              travelFrom={travelFrom}
              focusOnOpen={focusOnOpen}
              tileFor={tileFor}
              triggerFor={(key) => triggers.current.get(key) ?? null}
              onClose={() => onOpenChange(null)}
              onClosed={finishClose}
            />
          </>
        ) : null}
      </div>
    </DrilldownContext.Provider>
  );
}

interface DrillPanelProps {
  id: string;
  detailKey: string;
  detail: KpiDetail;
  closing: boolean;
  staticMotion: boolean;
  stageRef: RefObject<HTMLDivElement | null>;
  travelFrom: RefObject<Box | null>;
  focusOnOpen: RefObject<boolean>;
  tileFor: (key: string) => HTMLElement | null;
  triggerFor: (key: string) => HTMLElement | null;
  onClose: () => void;
  onClosed: () => void;
}

function DrillPanel(props: DrillPanelProps): ReactElement {
  const { closing, staticMotion, stageRef, travelFrom, focusOnOpen, tileFor, onClosed } = props;
  const panelRef = useRef<HTMLElement | null>(null);
  const headingRef = useRef<HTMLHeadingElement | null>(null);
  const headingId = useId();
  const keyRef = useRef(props.detailKey);
  const triggerForRef = useRef(props.triggerFor);
  const onClosedRef = useRef(onClosed);
  useLayoutEffect(() => {
    keyRef.current = props.detailKey;
    triggerForRef.current = props.triggerFor;
    onClosedRef.current = onClosed;
  });

  useLayoutEffect(() => {
    const panel = panelRef.current;
    const stage = stageRef.current;
    if (panel === null || stage === null) return undefined;
    const sizeInner = (): void => {
      panel.style.setProperty('--stage-w', `${String(stage.clientWidth)}px`);
      panel.style.setProperty('--stage-h', `${String(stage.clientHeight)}px`);
    };
    sizeInner();
    if (typeof ResizeObserver === 'undefined') return undefined;
    const observer = new ResizeObserver(sizeInner);
    observer.observe(stage);
    return () => observer.disconnect();
  }, [stageRef]);

  useLayoutEffect(() => {
    const panel = panelRef.current;
    return () => {
      // Runs before the panel leaves the DOM: focus is still inside if the user was in it.
      if (panel !== null && panel.contains(document.activeElement)) {
        const key = keyRef.current;
        queueMicrotask(() => triggerForRef.current(key)?.focus());
      }
    };
  }, []);

  useLayoutEffect(() => {
    if (!closing) {
      if (focusOnOpen.current) {
        focusOnOpen.current = false;
        headingRef.current?.focus({ preventScroll: true });
      }
      return;
    }
    const panel = panelRef.current;
    if (panel !== null && panel.contains(document.activeElement)) {
      triggerForRef.current(keyRef.current)?.focus({ preventScroll: true });
    }
  }, [closing, focusOnOpen]);

  useLayoutEffect(() => {
    const panel = panelRef.current;
    const stage = stageRef.current;
    if (panel === null || stage === null || staticMotion) {
      if (closing) onClosedRef.current();
      return undefined;
    }
    let cancelled = false;
    if (closing) {
      const tile = tileFor(keyRef.current);
      if (tile === null) {
        onClosedRef.current();
        return undefined;
      }
      const from = relativeBox(panel, stage);
      const to = relativeBox(tile, stage);
      placeAt(panel, from);
      const controls = animate(panel, { ...to }, motionX.spring.travel);
      void controls.then(() => {
        if (!cancelled) onClosedRef.current();
      });
      return () => {
        cancelled = true;
        controls.stop();
      };
    }
    const from = travelFrom.current ?? (panel.style.width !== '' ? relativeBox(panel, stage) : null);
    travelFrom.current = null;
    if (from === null) return undefined;
    placeAt(panel, from);
    const controls = animate(
      panel,
      { top: 0, left: 0, width: stage.clientWidth, height: stage.clientHeight },
      motionX.spring.travel,
    );
    void controls.then(() => {
      if (!cancelled) clearPlace(panel);
    });
    return () => {
      cancelled = true;
      controls.stop();
    };
  }, [closing, staticMotion, stageRef, travelFrom, tileFor]);

  const onKeyDown = (event: KeyboardEvent<HTMLElement>): void => {
    if (event.key !== 'Escape') return;
    event.stopPropagation();
    props.onClose();
  };

  const detail = props.detail;
  return (
    <section
      ref={panelRef}
      id={props.id}
      role="region"
      aria-labelledby={headingId}
      className={styles.panel}
      data-closing={closing ? '' : undefined}
      onKeyDown={onKeyDown}
    >
      <div className={styles.inner}>
        <div className={styles.head}>
          <div className={styles.headText}>
            <h3 ref={headingRef} id={headingId} tabIndex={-1} className={styles.title}>
              {detail.label}
            </h3>
            <div className={styles.reading}>
              <StatReading value={detail.value} unit={detail.unit} countUp={false} />
              <SampleChip sampleCount={detail.sampleCount} />
            </div>
            {detail.delta !== undefined ? <StatDeltaLine delta={detail.delta} /> : null}
            <p className={styles.description}>{detail.description}</p>
          </div>
          <IconButton
            label="Close details"
            variant="ghost"
            size="sm"
            icon={<Minimize2 size={14} aria-hidden="true" />}
            onClick={props.onClose}
          />
        </div>
        <ScrollArea className={styles.scroll} viewportClassName={styles.viewport}>
          <div key={props.detailKey} className={styles.body}>
            {detail.children}
          </div>
        </ScrollArea>
        {detail.footer !== undefined ? <div className={styles.footer}>{detail.footer}</div> : null}
      </div>
    </section>
  );
}

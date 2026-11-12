import * as Popover from '@radix-ui/react-popover';
import { AlertTriangle } from 'lucide-react';
import { motion, type Transition } from 'motion/react';
import {
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type MouseEvent,
  type ReactElement,
  type ReactNode,
} from 'react';

import { errMessage } from '../../../lib/errors';
import { Button } from '../../arc/button/button';
import { motionTokens } from '../../arc/motion-tokens';
import { Tooltip } from '../../arc/tooltip/tooltip';
import { IconButton } from '../icon-button/icon-button';
import { MorphLoader } from '../morph-loader/morph-loader';
import { motionX } from '../motion/motion';
import { useStaticMotion } from '../motion/use-static-motion';
import styles from './confirm-popover.module.css';

/** One confirmation, shown by ConfirmPopover at its trigger or by ConfirmDialog when the palette started it. */
export interface ConfirmContent {
  /** "Delete qwen3-8b?" */
  title: string;
  /** What is lost, in one or two sentences. */
  description: ReactNode;
  /** "Delete" */
  confirmLabel: string;
  /** "Deleting…" */
  workingLabel: string;
  /** "Deleted" */
  successLabel: string;
  /** Default 'danger'. */
  tone?: 'danger' | 'neutral';
  /** A warning line above the buttons, e.g. "This model is loaded." */
  notice?: ReactNode;
  /** Rejects with an Error whose message is shown in place. */
  onConfirm: () => Promise<void>;
  /** Printed above the error message, e.g. "Failed to update cache". Default "Could not finish". */
  errorTitle?: string;
  /** Runs after the success pill has gone (reload the list, toast, navigate). */
  onDone?: () => void;
}

export type ConfirmTrigger =
  | { kind: 'button'; label: string; icon?: ReactNode; size?: 'sm' | 'md'; variant?: 'secondary' | 'ghost' | 'danger' }
  | { kind: 'icon'; label: string; icon: ReactNode; size?: 'sm' | 'md' };

export interface ConfirmPopoverProps extends ConfirmContent {
  trigger: ConfirmTrigger;
  disabled?: boolean;
  /** Tooltip text while disabled. */
  disabledReason?: string;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  /** Which trigger corner the panel grows from. 'end' (default): top-right; 'start': top-left. */
  align?: 'start' | 'end';
}

export type ConfirmPhase = 'idle' | 'confirming' | 'working' | 'success';
type Shape = 'trigger' | 'panel' | 'success';

export const DEFAULT_ERROR_TITLE = 'Could not finish';

const PANEL_WIDTH = 300;
const EDGE_PX = 8;

/**
 * The corner the panel grows from. Flipped when the asked-for side has no room, because Radix's collision shift would
 * slide the panel off the trigger without changing `data-align`.
 */
export function fitAlign(
  align: 'start' | 'end',
  rect: { left: number; right: number },
  viewWidth: number,
): 'start' | 'end' {
  if (align === 'end' && rect.right - PANEL_WIDTH < EDGE_PX && rect.left + PANEL_WIDTH <= viewWidth - EDGE_PX)
    return 'start';
  if (align === 'start' && rect.left + PANEL_WIDTH > viewWidth - EDGE_PX && rect.right - PANEL_WIDTH >= EDGE_PX)
    return 'end';
  return align;
}
const INSTANT = { duration: 0 } as const;

interface Box {
  width: number;
  height: number;
}

function focusScrollRootIfLost(root: HTMLElement | null): void {
  const active = document.activeElement;
  if (active !== null && active !== document.body) return;
  (root?.isConnected === true ? root : document.querySelector<HTMLElement>('[data-scroll-root]'))?.focus();
}

function TriggerFace({ trigger }: { trigger: ConfirmTrigger }): ReactElement {
  if (trigger.kind === 'icon') return <span className={styles.faceIcon}>{trigger.icon}</span>;
  return (
    <span className={styles.faceLabel}>
      {trigger.icon}
      {trigger.label}
    </span>
  );
}

/**
 * A destructive button that grows into its own confirmation, runs the mutation in place, and shrinks into a
 * success pill. The surface is drawn in a portal over the trigger, so no clipping parent cuts it.
 */
export function ConfirmPopover(props: ConfirmPopoverProps): ReactElement {
  const staticMotion = useStaticMotion();
  const titleId = useId();
  const descriptionId = useId();
  const tone = props.tone ?? 'danger';

  const [phase, setPhaseState] = useState<ConfirmPhase>('idle');
  const [mounted, setMountedState] = useState(false);
  const [shape, setShape] = useState<Shape>('trigger');
  const [look, setLook] = useState<Shape>('trigger');
  const [fading, setFading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [triggerBox, setTriggerBox] = useState<Box>({ width: 0, height: 0 });
  const [align, setAlign] = useState<'start' | 'end'>(props.align ?? 'end');
  const [panelHeight, setPanelHeight] = useState(0);
  const [successWidth, setSuccessWidth] = useState(0);

  const phaseRef = useRef<ConfirmPhase>('idle');
  const mountedRef = useRef(false);
  const doneRef = useRef(false);
  const returnFocusRef = useRef(false);
  const aliveRef = useRef(true);
  const holdTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const anchorRef = useRef<HTMLSpanElement | null>(null);
  const scrollRootRef = useRef<HTMLElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const cancelRef = useRef<HTMLButtonElement | null>(null);
  const confirmRef = useRef<HTMLButtonElement | null>(null);
  // A state, not a ref: Radix Portal mounts its children one commit after `mounted` turns true.
  const [panel, setPanel] = useState<HTMLDivElement | null>(null);
  const successRef = useRef<HTMLSpanElement | null>(null);
  const latest = useRef(props);
  latest.current = props;

  const setPhase = (next: ConfirmPhase): void => {
    phaseRef.current = next;
    setPhaseState(next);
  };
  const setMounted = (next: boolean): void => {
    mountedRef.current = next;
    setMountedState(next);
  };

  const openNow = (): void => {
    if (phaseRef.current !== 'idle' || mountedRef.current || latest.current.disabled === true) return;
    const rect = anchorRef.current?.getBoundingClientRect();
    setTriggerBox({ width: rect?.width ?? 0, height: rect?.height ?? 0 });
    const asked = latest.current.align ?? 'end';
    setAlign(rect === undefined ? asked : fitAlign(asked, rect, globalThis.innerWidth));
    setError(null);
    setFading(false);
    setShape('panel');
    setLook(staticMotion ? 'panel' : 'trigger');
    setMounted(true);
    setPhase('confirming');
    latest.current.onOpenChange?.(true);
  };

  const cancel = (): void => {
    if (phaseRef.current !== 'confirming') return;
    returnFocusRef.current = true;
    setPhase('idle');
    setShape('trigger');
    latest.current.onOpenChange?.(false);
    if (staticMotion) setMounted(false);
  };

  const finish = (): void => {
    holdTimer.current = undefined;
    returnFocusRef.current = true;
    setPhase('idle');
    setFading(true);
    doneRef.current = true;
    latest.current.onOpenChange?.(false);
    if (staticMotion) setMounted(false);
  };

  const confirm = async (): Promise<void> => {
    if (phaseRef.current !== 'confirming') return;
    setError(null);
    setPhase('working');
    try {
      await latest.current.onConfirm();
    } catch (err) {
      if (!aliveRef.current || !mountedRef.current) return;
      setError(errMessage(err));
      setPhase('confirming');
      confirmRef.current?.focus();
      return;
    }
    if (!aliveRef.current) {
      latest.current.onDone?.();
      return;
    }
    if (!mountedRef.current) return;
    setPhase('success');
    setShape('success');
    holdTimer.current = setTimeout(finish, staticMotion ? 0 : motionX.successHold * 1000);
  };

  // The trigger can go while the hold runs (a live refetch drops its row): the mutation's follow-up must still run.
  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
      if (holdTimer.current !== undefined) clearTimeout(holdTimer.current);
      holdTimer.current = undefined;
      const left = phaseRef.current;
      phaseRef.current = 'idle';
      if (left !== 'idle') latest.current.onOpenChange?.(false);
      if (left === 'success' || doneRef.current) {
        doneRef.current = false;
        latest.current.onDone?.();
      }
    };
  }, []);

  // A controlled `open` opens from idle and cancels a confirmation; it never interrupts a running or finished one.
  const openProp = props.open;
  useEffect(() => {
    if (openProp === true) openNow();
    else if (openProp === false) cancel();
  }, [openProp]);

  // Passive, after Radix has released its focus trap: the inert surface would otherwise leave focus on <body> for the whole exit morph.
  useEffect(() => {
    if (phase !== 'idle' || !returnFocusRef.current) return;
    returnFocusRef.current = false;
    const trigger = triggerRef.current;
    if (trigger === null || !trigger.isConnected) return;
    const active = document.activeElement;
    if (active === null || active === document.body || panel?.contains(active) === true) {
      trigger.focus({ preventScroll: true });
    }
  }, [phase]);

  useEffect(() => {
    if (mounted || !doneRef.current) return;
    doneRef.current = false;
    latest.current.onDone?.();
    queueMicrotask(() => {
      const trigger = triggerRef.current;
      if (trigger !== null && trigger.isConnected) return;
      focusScrollRootIfLost(scrollRootRef.current);
    });
  }, [mounted]);

  // A layout cleanup, not a passive one: focus is still inside the anchor only before React removes it.
  useLayoutEffect(() => {
    const anchor = anchorRef.current;
    const root = anchor?.closest<HTMLElement>('[data-scroll-root]') ?? null;
    scrollRootRef.current = root;
    return () => {
      if (anchor === null || !anchor.contains(document.activeElement)) return;
      queueMicrotask(() => focusScrollRootIfLost(root));
    };
  }, []);

  useEffect(() => {
    if (staticMotion) {
      setLook(shape);
      return;
    }
    const frame = requestAnimationFrame(() => setLook(shape));
    return () => cancelAnimationFrame(frame);
  }, [shape, staticMotion]);

  useLayoutEffect(() => {
    if (panel === null) return;
    setPanelHeight(panel.offsetHeight);
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(() => setPanelHeight(panel.offsetHeight));
    observer.observe(panel);
    return () => observer.disconnect();
  }, [panel]);

  useLayoutEffect(() => {
    if (shape !== 'success') return;
    setSuccessWidth(successRef.current?.offsetWidth ?? 0);
  }, [shape]);

  const target: Box =
    shape === 'panel'
      ? { width: PANEL_WIDTH, height: panelHeight }
      : shape === 'success'
        ? { width: successWidth, height: triggerBox.height }
        : triggerBox;
  const sizeTransition = staticMotion ? INSTANT : motionX.spring.morph;
  const opacityTransition: Transition = staticMotion
    ? { duration: motionTokens.duration.instant }
    : { duration: motionTokens.duration.exit, ease: [...motionTokens.ease.standard] };

  const working = phase === 'working';
  const trigger = props.trigger;
  const blockWhenDisabled = (event: MouseEvent<HTMLButtonElement>): void => {
    if (props.disabled === true) event.preventDefault();
  };
  const triggerControl =
    trigger.kind === 'icon' ? (
      <IconButton
        ref={triggerRef}
        label={trigger.label}
        icon={trigger.icon}
        size={trigger.size ?? 'sm'}
        variant="ghost"
        className={styles.trigger}
        data-trigger="ghost"
        tooltip={props.disabled === true && props.disabledReason !== undefined ? props.disabledReason : true}
        aria-disabled={props.disabled === true ? true : undefined}
        onClick={blockWhenDisabled}
      />
    ) : (
      <Button
        ref={triggerRef}
        type="button"
        variant={trigger.variant ?? 'secondary'}
        size={trigger.size ?? 'md'}
        className={styles.trigger}
        data-trigger={(trigger.variant ?? 'secondary') === 'ghost' ? 'ghost' : 'solid'}
        aria-disabled={props.disabled === true ? true : undefined}
        onClick={blockWhenDisabled}
      >
        {trigger.icon}
        {trigger.label}
      </Button>
    );
  const wrappedTrigger = <Popover.Trigger asChild>{triggerControl}</Popover.Trigger>;

  return (
    <Popover.Root
      modal
      open={phase !== 'idle'}
      onOpenChange={(next) => {
        if (next) openNow();
        else cancel();
      }}
    >
      <Popover.Anchor asChild>
        <span ref={anchorRef} className={styles.anchor} data-phase={phase} data-covered={mounted ? '' : undefined}>
          {trigger.kind === 'button' && props.disabled === true && props.disabledReason !== undefined ? (
            <Tooltip content={props.disabledReason}>{wrappedTrigger}</Tooltip>
          ) : (
            wrappedTrigger
          )}
        </span>
      </Popover.Anchor>
      {mounted ? (
        <Popover.Portal forceMount>
          <Popover.Content
            forceMount
            className={styles.content}
            side="bottom"
            align={align}
            sideOffset={-triggerBox.height}
            collisionPadding={8}
            style={{ height: panelHeight }}
            aria-labelledby={titleId}
            aria-describedby={descriptionId}
            inert={phase === 'idle'}
            onOpenAutoFocus={(event) => {
              event.preventDefault();
              cancelRef.current?.focus();
            }}
            onCloseAutoFocus={(event) => {
              const active = document.activeElement;
              if (active !== null && active !== document.body && active !== triggerRef.current) event.preventDefault();
            }}
            onEscapeKeyDown={(event) => {
              if (phaseRef.current !== 'confirming') event.preventDefault();
            }}
            onInteractOutside={(event) => {
              if (phaseRef.current !== 'confirming') event.preventDefault();
            }}
          >
            <motion.div
              className={styles.surface}
              data-phase={phase}
              data-shape={shape}
              data-look={look}
              data-tone={tone}
              data-trigger={trigger.kind === 'icon' ? 'ghost' : (trigger.variant ?? 'secondary')}
              initial={{ width: triggerBox.width, height: triggerBox.height, opacity: 1 }}
              animate={{ width: target.width, height: target.height, opacity: fading ? 0 : 1 }}
              transition={{ width: sizeTransition, height: sizeTransition, opacity: opacityTransition }}
              onAnimationComplete={() => {
                if (phaseRef.current === 'idle' && mountedRef.current) setMounted(false);
              }}
            >
              <span
                className={styles.triggerFace}
                style={{ width: triggerBox.width, height: triggerBox.height }}
                aria-hidden="true"
              >
                <TriggerFace trigger={trigger} />
              </span>
              <div ref={setPanel} className={styles.panel}>
                <h2 id={titleId} className={styles.title}>
                  {props.title}
                </h2>
                <div id={descriptionId} className={styles.description}>
                  {props.description}
                </div>
                {props.notice !== undefined ? (
                  <div className={styles.notice}>
                    <AlertTriangle className={styles.noticeIcon} size={14} strokeWidth={1.75} aria-hidden="true" />
                    <span>{props.notice}</span>
                  </div>
                ) : null}
                {error !== null ? (
                  <div role="alert" className={styles.error}>
                    <p className={styles.errorTitle}>{props.errorTitle ?? DEFAULT_ERROR_TITLE}</p>
                    <p className={styles.errorMessage}>{error}</p>
                  </div>
                ) : null}
                <div className={styles.footer}>
                  <Button ref={cancelRef} type="button" variant="ghost" size="md" disabled={working} onClick={cancel}>
                    Cancel
                  </Button>
                  <Button
                    ref={confirmRef}
                    type="button"
                    variant={tone === 'danger' ? 'danger' : 'primary'}
                    size="md"
                    className={styles.confirm}
                    data-working={working ? '' : undefined}
                    data-static={staticMotion ? '' : undefined}
                    aria-busy={working ? true : undefined}
                    aria-disabled={working ? true : undefined}
                    onClick={() => {
                      void confirm();
                    }}
                  >
                    {working ? props.workingLabel : props.confirmLabel}
                  </Button>
                </div>
              </div>
              <span
                ref={successRef}
                className={styles.successFace}
                style={{ height: triggerBox.height }}
                aria-hidden="true"
              >
                <MorphLoader status={shape === 'success' ? 'success' : 'loading'} spinning={false} size={14} />
                {props.successLabel}
              </span>
            </motion.div>
            <span role="status" className={styles.srOnly}>
              {phase === 'success' ? props.successLabel : ''}
            </span>
          </Popover.Content>
        </Popover.Portal>
      ) : null}
    </Popover.Root>
  );
}

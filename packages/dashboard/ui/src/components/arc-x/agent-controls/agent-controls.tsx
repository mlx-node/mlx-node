import { Mail, Square } from 'lucide-react';
import { useId, useLayoutEffect, useRef, type ReactElement, type RefObject } from 'react';

import type { AgentButtons } from '../../../lib/agents';
import { CHAT, fill } from '../../../lib/chat-strings';
import { cn } from '../../../lib/cn';
import { Button } from '../../arc/button/button';
import { ConfirmPopover } from '../confirm-popover/confirm-popover';
import { IconButton } from '../icon-button/icon-button';
import styles from './agent-controls.module.css';

/** The agent a button acts on: its newest attempt's child session and its label. */
export interface AgentTarget {
  id: string;
  title: string;
}

/** Work: what the page does for an agent's buttons, on every surface. */
export interface AgentHandlers {
  /** The confirmed Stop; returns at once, a failure is the page's toast. */
  onStop: (target: AgentTarget) => void;
  /** Brings the request's card to the front. */
  onReview: (askId: string) => void;
  /** Opens its Agent view (`field`: with focus in its message field); focus comes back to `opener` when it closes. */
  onOpen: (target: AgentTarget, opener: HTMLElement | null, field: boolean) => void;
}

export interface AgentControlsProps {
  /** Its newest attempt's child session: the id every button acts on. */
  id: string;
  label: string;
  /** A workflow agent: its Stop says the workflow gets a cancelled result. */
  workflow: boolean;
  buttons: AgentButtons;
  /** The confirmed Stop; the press goes to the page and returns at once. */
  onStop?: () => void;
  onReview?: (askId: string) => void;
  /** Message: opens its Agent view with focus in the field. */
  onMessage?: (opener: HTMLElement) => void;
  /** 32 px targets (the drawer); touch gets them by itself. */
  large?: boolean;
  className?: string;
}

/** Message: drawn off with the reason as its tooltip and description, else it opens the view at the field. */
function MessageButton(props: {
  label: string;
  off: string | null;
  tooltip: string;
  keys: string;
  onMessage: (opener: HTMLElement) => void;
}): ReactElement {
  const offId = useId();
  const icon = <Mail size={12} strokeWidth={2} aria-hidden="true" />;
  if (props.off !== null) {
    return (
      <>
        <IconButton
          label={props.label}
          icon={icon}
          tooltip={props.off}
          aria-disabled={true}
          aria-describedby={offId}
          onClick={(event) => event.preventDefault()}
        />
        <span id={offId} hidden>
          {props.off}
        </span>
      </>
    );
  }
  return (
    <IconButton
      label={props.label}
      icon={icon}
      tooltip={props.tooltip}
      aria-keyshortcuts={props.keys}
      onClick={(event) => props.onMessage(event.currentTarget)}
    />
  );
}

/** The keys a row's buttons open from the row: tooltips and `aria-keyshortcuts` name them. */
export const CONTROL_KEYS = { message: 'M', stop: 'X' } as const;

function keyTip(name: string, key: string): string {
  return fill(CHAT.agent.keyTip, { name, key });
}

/**
 * A running agent's buttons, always in this order: Message (a Task subagent's) or Review (while it waits on a
 * request), then Stop. The Stop confirm opens on Keep running. Null when the row has none.
 */
export function AgentControls(props: AgentControlsProps): ReactElement | null {
  const { buttons, label } = props;
  const message = buttons.message !== null && props.onMessage !== undefined ? props.onMessage : null;
  const review = buttons.review !== null && props.onReview !== undefined ? buttons.review : null;
  const stop = buttons.stop && props.onStop !== undefined ? props.onStop : null;
  if (message === null && review === null && stop === null) return null;
  const stopName = fill(CHAT.agent.stop, { label });
  const messageName = fill(CHAT.agent.message, { label });
  return (
    <span
      className={cn(styles.controls, props.className)}
      data-agent-controls=""
      data-large={props.large === true ? '' : undefined}
    >
      {message !== null ? (
        <span className={styles.slot} data-agent-message="">
          <MessageButton
            label={messageName}
            off={buttons.message?.off ?? null}
            tooltip={keyTip(messageName, CONTROL_KEYS.message)}
            keys={CONTROL_KEYS.message}
            onMessage={message}
          />
        </span>
      ) : null}
      {review !== null ? (
        <Button
          type="button"
          size="sm"
          variant="secondary"
          className={styles.review}
          aria-label={fill(CHAT.agent.review, { label })}
          onClick={() => props.onReview?.(review)}
        >
          {CHAT.run.buttons.review}
        </Button>
      ) : null}
      {stop !== null ? (
        <span className={styles.slot} data-agent-stop="">
          <ConfirmPopover
            trigger={{
              kind: 'icon',
              label: stopName,
              icon: <Square size={12} strokeWidth={2} aria-hidden="true" />,
              tooltip: keyTip(stopName, CONTROL_KEYS.stop),
              keys: CONTROL_KEYS.stop,
            }}
            align="end"
            title={props.workflow ? CHAT.agent.stopTitle : CHAT.agent.stopTitleSubagent}
            description={props.workflow ? CHAT.agent.stopText : CHAT.agent.stopTextSubagent}
            confirmLabel={CHAT.run.buttons.stop}
            cancelLabel={CHAT.run.buttons.keepRunning}
            workingLabel={CHAT.agent.state.stopping}
            successLabel={CHAT.agent.state.stopping}
            onConfirm={async () => stop()}
          />
        </span>
      ) : null}
    </span>
  );
}

/** The part holds focus, or a popover one of its buttons opened does: it is drawn outside, named by `aria-controls`. */
export function holdsFocus(part: HTMLElement): boolean {
  const active = document.activeElement;
  if (active === null) return false;
  if (part.contains(active)) return true;
  const dialog = active.closest('[role="dialog"]');
  if (dialog === null || dialog.id === '') return false;
  for (const button of part.querySelectorAll('[aria-controls]'))
    if (button.getAttribute('aria-controls') === dialog.id) return true;
  return false;
}

/**
 * When the element at `ref` leaves while it holds focus (a Stop that went once its agent stops), the element `target`
 * names takes focus, unless something else took it first. `target` runs as it leaves, still in the page, and gives
 * what to read once it is gone.
 */
export function useFocusFallback(
  ref: RefObject<HTMLElement | null>,
  target: (leaving: HTMLElement) => () => HTMLElement | null,
): void {
  const latest = useRef(target);
  latest.current = target;
  // A layout cleanup: focus is still inside before React removes the node.
  useLayoutEffect(() => {
    const node = ref.current;
    return () => {
      if (node === null || !holdsFocus(node)) return;
      const next = latest.current(node);
      queueMicrotask(() => {
        const active = document.activeElement;
        if (active !== null && active !== document.body && !active.hasAttribute('data-scroll-root')) return;
        next()?.focus({ preventScroll: true });
      });
    };
  }, [ref]);
}

/** A key on a row presses the button it names: X opens the Stop confirm, M the Agent view at its field. True when it did. */
export function pressControl(row: HTMLElement, which: 'stop' | 'message'): boolean {
  const slot = row.querySelector<HTMLElement>(which === 'stop' ? '[data-agent-stop]' : '[data-agent-message]');
  const button = slot?.querySelector<HTMLButtonElement>('button') ?? null;
  if (button === null || button.getAttribute('aria-disabled') === 'true') return false;
  button.click();
  return true;
}

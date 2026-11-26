import * as PopoverPrimitive from '@radix-ui/react-popover';
import { useRef, type ReactElement, type ReactNode, type RefObject } from 'react';

import { CHAT } from '../../../lib/chat-strings';
import { Popover, PopoverContent } from '../../arc/popover/popover';
import styles from './context-panel.module.css';

export interface ContextPopoverProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  /** The ring that opens the panel: the anchor while it is drawn, and where focus goes back. */
  ring: RefObject<HTMLElement | null>;
  /** Where the panel hangs while the ring is not drawn (the end of the tools row). */
  fallback: RefObject<HTMLElement | null>;
  /** Takes focus on close when the ring is not drawn. */
  focusFallback: () => void;
  children: ReactNode;
}

/** The ring is in the page and not hidden by the narrow-box rule. */
function shown(node: HTMLElement | null): node is HTMLElement {
  return node !== null && node.isConnected && getComputedStyle(node).display !== 'none';
}

/**
 * The Context panel's popover, opening upward with its right edge on the ring; while the ring is hidden it hangs from
 * the fallback. Esc and a press outside close it (a press on the ring too: Radix closes on the click, after the ring's
 * own toggle); focus goes back to the ring (or the fallback's) unless the panel's own action moved it elsewhere.
 */
export function ContextPopover(props: ContextPopoverProps): ReactElement {
  const panel = useRef<HTMLDivElement | null>(null);
  const anchor = useRef({
    getBoundingClientRect: (): DOMRect => {
      const ring = props.ring.current;
      if (shown(ring)) return ring.getBoundingClientRect();
      return props.fallback.current?.getBoundingClientRect() ?? new DOMRect();
    },
  });
  return (
    <Popover open={props.open} onOpenChange={props.onOpenChange}>
      <PopoverPrimitive.Anchor virtualRef={anchor} />
      <PopoverContent
        ref={panel}
        side="top"
        align="end"
        aria-label={CHAT.history.context}
        className={styles.popover}
        onCloseAutoFocus={(event) => {
          event.preventDefault();
          const active = document.activeElement;
          if (active !== null && active !== document.body && panel.current?.contains(active) !== true) return;
          const ring = props.ring.current;
          if (shown(ring)) ring.focus();
          else props.focusFallback();
        }}
      >
        {props.children}
      </PopoverContent>
    </Popover>
  );
}

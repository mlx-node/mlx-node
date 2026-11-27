import { ArrowUp, Square } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import type { ReactElement } from 'react';

import { CHAT } from '../../../lib/chat-strings';
import { motionTokens } from '../../arc/motion-tokens';
import { Tooltip } from '../../arc/tooltip/tooltip';
import { IconButton } from '../icon-button/icon-button';
import { motionX } from '../motion/motion';
import { useStaticMotion } from '../motion/use-static-motion';
import styles from './composer.module.css';

/** `op`: a fork, a rewind or a recap of the conversation runs; it has no Stop, and Enter queues. */
export type ComposerPhase = 'idle' | 'creating' | 'starting' | 'running' | 'cancelling' | 'op';

/** While an answer starts or runs and the box has text: the round button queues the text. */
export interface Queueing {
  enabled: boolean;
  onQueue: () => void;
}

export interface SendButtonProps {
  phase: ComposerPhase;
  canSend: boolean;
  /** Why Send (or Queue) is off, for the tooltip. */
  reason?: string;
  /**
   * Set while an answer starts or runs (or an op runs) and the box has text: Queue, with a small Stop before it (none
   * for an op).
   */
  queueing?: Queueing;
  onSend: () => void;
  onStop: () => void;
}

interface Face {
  glyph: 'send' | 'stop' | 'ring';
  /** What a press does. */
  action: 'send' | 'queue' | 'stop' | 'none';
  enabled: boolean;
  name: string;
  tip: string;
  /** A hairline ring around the button; it turns while `spin`. */
  ring: boolean;
  spin: boolean;
  dim: boolean;
}

/**
 * What the one round button shows in each phase. It always does what Enter does: with text while an answer starts or
 * runs (`queueing` set) it is Queue, and Stop moves to the small square before it.
 */
export function sendFace(phase: ComposerPhase, canSend: boolean, reason?: string, queueing?: Queueing): Face {
  if (queueing !== undefined && (phase === 'starting' || phase === 'running' || phase === 'op')) {
    return {
      glyph: 'send',
      action: 'queue',
      enabled: queueing.enabled,
      name: CHAT.queue.add,
      tip: reason ?? (phase === 'op' ? CHAT.queue.addTipOp : CHAT.queue.addTip),
      ring: false,
      spin: false,
      dim: !queueing.enabled,
    };
  }
  switch (phase) {
    case 'op':
      return {
        glyph: 'ring',
        action: 'none',
        enabled: false,
        name: CHAT.opBusy,
        tip: CHAT.opBusyTip,
        ring: true,
        spin: true,
        dim: false,
      };
    case 'idle':
      return {
        glyph: 'send',
        action: 'send',
        enabled: canSend,
        name: CHAT.send,
        tip: reason ?? CHAT.sendTip,
        ring: false,
        spin: false,
        dim: !canSend,
      };
    case 'creating':
      return {
        glyph: 'ring',
        action: 'none',
        enabled: false,
        name: CHAT.starting,
        tip: CHAT.startingTip,
        ring: true,
        spin: true,
        dim: false,
      };
    case 'starting':
      return {
        glyph: 'stop',
        action: 'stop',
        enabled: true,
        name: CHAT.stop,
        tip: CHAT.stopTip,
        ring: true,
        spin: true,
        dim: false,
      };
    case 'running':
      return {
        glyph: 'stop',
        action: 'stop',
        enabled: true,
        name: CHAT.stop,
        tip: CHAT.stopTip,
        ring: false,
        spin: false,
        dim: false,
      };
    case 'cancelling':
      return {
        glyph: 'stop',
        action: 'none',
        enabled: false,
        name: CHAT.stopping,
        tip: CHAT.stoppingTip,
        ring: true,
        spin: true,
        dim: true,
      };
  }
}

const SWAP_IN = { duration: motionX.swap.in.duration, ease: motionX.swap.in.ease };
const SWAP_OUT = { duration: motionX.swap.out.duration, ease: motionX.swap.out.ease };

/**
 * The composer's one round button: Send, Queue or Stop. The glyph swaps in place; the button never changes size.
 * While it is Queue for an answer, Stop is a small ghost square just before it (Esc still stops); an op has no Stop.
 */
export function SendButton({ phase, canSend, reason, queueing, onSend, onStop }: SendButtonProps): ReactElement {
  const staticMotion = useStaticMotion();
  const face = sendFace(phase, canSend, reason, queueing);
  const glyph =
    face.glyph === 'send' ? (
      <ArrowUp size={16} strokeWidth={2} />
    ) : face.glyph === 'stop' ? (
      <Square size={10} strokeWidth={0} fill="currentColor" />
    ) : null;
  const button = (
    <button
      type="button"
      className={styles.send}
      data-phase={phase}
      data-dim={face.dim ? '' : undefined}
      aria-label={face.name}
      aria-keyshortcuts={face.action === 'queue' ? 'Enter' : undefined}
      disabled={!face.enabled}
      onClick={() => {
        if (!face.enabled) return;
        if (face.action === 'send') onSend();
        else if (face.action === 'queue') queueing?.onQueue();
        else if (face.action === 'stop') onStop();
      }}
    >
      {face.ring ? <span className={styles.ring} data-spin={face.spin ? '' : undefined} aria-hidden="true" /> : null}
      <span className={styles.glyphCell} aria-hidden="true">
        {staticMotion ? (
          <span key={face.glyph} className={styles.glyph}>
            {glyph}
          </span>
        ) : (
          <AnimatePresence initial={false}>
            <motion.span
              key={face.glyph}
              className={styles.glyph}
              initial={{ opacity: 0, scale: 0.6 }}
              animate={{ opacity: 1, scale: 1, filter: 'blur(0px)', transition: SWAP_IN }}
              exit={{ opacity: 0, scale: 0.6, filter: `blur(${motionTokens.blur.subtle}px)`, transition: SWAP_OUT }}
            >
              {glyph}
            </motion.span>
          </AnimatePresence>
        )}
      </span>
    </button>
  );
  const round = (
    <Tooltip content={face.tip}>
      <span className={styles.tipTarget}>{button}</span>
    </Tooltip>
  );
  if (face.action !== 'queue' || phase === 'op') return round;
  return (
    <>
      <IconButton
        className={styles.stopSmall}
        label={CHAT.stop}
        tooltip={CHAT.stopTip}
        icon={<Square size={10} strokeWidth={0} fill="currentColor" />}
        onClick={onStop}
      />
      {round}
    </>
  );
}

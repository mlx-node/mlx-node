import { ArrowUp, Square } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import type { ReactElement } from 'react';

import { CHAT } from '../../../lib/chat-strings';
import { motionTokens } from '../../arc/motion-tokens';
import { Tooltip } from '../../arc/tooltip/tooltip';
import { motionX } from '../motion/motion';
import { useStaticMotion } from '../motion/use-static-motion';
import styles from './composer.module.css';

export type ComposerPhase = 'idle' | 'creating' | 'starting' | 'running' | 'cancelling';

export interface SendButtonProps {
  phase: ComposerPhase;
  canSend: boolean;
  /** Why Send is off, for the tooltip. */
  reason?: string;
  onSend: () => void;
  onStop: () => void;
}

interface Face {
  glyph: 'send' | 'stop' | 'ring';
  enabled: boolean;
  name: string;
  tip: string;
  /** A hairline ring around the button; it turns while `spin`. */
  ring: boolean;
  spin: boolean;
  dim: boolean;
}

/** What the one round button shows in each phase. */
export function sendFace(phase: ComposerPhase, canSend: boolean, reason?: string): Face {
  switch (phase) {
    case 'idle':
      return {
        glyph: 'send',
        enabled: canSend,
        name: CHAT.send,
        tip: canSend ? CHAT.sendTip : (reason ?? CHAT.send),
        ring: false,
        spin: false,
        dim: !canSend,
      };
    case 'creating':
      return {
        glyph: 'ring',
        enabled: false,
        name: CHAT.starting,
        tip: CHAT.startingTip,
        ring: true,
        spin: true,
        dim: false,
      };
    case 'starting':
      return { glyph: 'stop', enabled: true, name: CHAT.stop, tip: CHAT.stopTip, ring: true, spin: true, dim: false };
    case 'running':
      return { glyph: 'stop', enabled: true, name: CHAT.stop, tip: CHAT.stopTip, ring: false, spin: false, dim: false };
    case 'cancelling':
      return {
        glyph: 'stop',
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

/** The composer's one round button: Send or Stop. The glyph swaps in place; the button never changes size. */
export function SendButton({ phase, canSend, reason, onSend, onStop }: SendButtonProps): ReactElement {
  const staticMotion = useStaticMotion();
  const face = sendFace(phase, canSend, reason);
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
      disabled={!face.enabled}
      onClick={() => {
        if (!face.enabled) return;
        if (face.glyph === 'send') onSend();
        else onStop();
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
  return (
    <Tooltip content={face.tip}>
      <span className={styles.tipTarget}>{button}</span>
    </Tooltip>
  );
}

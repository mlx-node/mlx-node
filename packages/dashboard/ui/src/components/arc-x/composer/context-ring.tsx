import type { ReactElement } from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import { formatNumber } from '../../../lib/format';
import { Tooltip } from '../../arc/tooltip/tooltip';
import styles from './composer.module.css';

export interface ContextUse {
  /** Tokens in the conversation's context now. */
  used: number;
  /** The model's context window in tokens. */
  window: number;
}

/** The arc is measured on a circle of path length 100. */
const MIN_ARC = 4;

/** The used share of the window, 0 to 1. */
export function contextShare({ used, window }: ContextUse): number {
  if (window <= 0 || used <= 0) return 0;
  return Math.min(used / window, 1);
}

/** The arc length out of 100; a used share above 0 is never shorter than the minimum, so it always shows. */
export function contextArc(context: ContextUse): number {
  const share = contextShare(context);
  return share === 0 ? 0 : Math.max(share * 100, MIN_ARC);
}

export function contextTone(context: ContextUse): 'warning' | 'danger' | undefined {
  const share = contextShare(context);
  return share >= 0.95 ? 'danger' : share >= 0.8 ? 'warning' : undefined;
}

/** "Context: 11,971 of 262,144 tokens (5%)". */
export function contextText(context: ContextUse): string {
  const share = contextShare(context);
  const percent =
    share > 0 && share < 0.01 ? CHAT.contextUnderOne : fill(CHAT.contextPercent, { n: Math.round(share * 100) });
  return fill(CHAT.contextUse, {
    used: formatNumber(context.used),
    window: formatNumber(context.window),
    percent,
  });
}

/** How full the context window is: a ring with no text; the numbers are in its tooltip and its name. */
export function ContextRing({ context }: { context: ContextUse }): ReactElement {
  const text = contextText(context);
  const arc = contextArc(context);
  return (
    <Tooltip content={text}>
      <span className={styles.context} role="img" aria-label={text} tabIndex={0} data-tone={contextTone(context)}>
        <svg viewBox="0 0 18 18" width={18} height={18} fill="none" strokeWidth={3} aria-hidden="true">
          <circle className={styles.contextTrack} cx={9} cy={9} r={7.5} />
          {arc > 0 ? (
            <circle
              className={styles.contextArc}
              cx={9}
              cy={9}
              r={7.5}
              pathLength={100}
              strokeDasharray={`${String(arc)} 100`}
              transform="rotate(-90 9 9)"
            />
          ) : null}
        </svg>
      </span>
    </Tooltip>
  );
}

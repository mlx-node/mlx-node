import type { ReactElement } from 'react';

import { CONTEXT_HIGH_PCT, quietFor } from '../../../lib/activity';
import { CHAT, fill } from '../../../lib/chat-strings';
import { formatElapsed } from '../../../lib/chat-turns';
import type { HelperProgress } from '../../../lib/chat-types';
import { formatCount } from '../../../lib/format';
import { errorWords, nowWords } from '../../../lib/helper-words';
import { SwapText } from '../swap-text/swap-text';
import styles from './activity-rail.module.css';

export type ProgressTone = 'normal' | 'muted' | 'warning' | 'danger';

/**
 * An agent's live line in the rail, ordered by need so a cut loses the least: what it does now (once its own lines
 * said) or how long it is quiet, errors, failed calls, calls, context, tokens. Ink: errors danger (a failed call is
 * not one), a high context warning, quiet muted.
 */
export function progressLine(progress: HelperProgress | null, now: number): { text: string; tone: ProgressTone } {
  if (progress === null) return { text: CHAT.activity.progressStarting, tone: 'muted' };
  const a = CHAT.activity;
  const parts: string[] = [];
  const quiet = quietFor(progress, now);
  if (quiet !== null) parts.push(fill(a.progressQuiet, { time: formatElapsed(quiet) }));
  else if (progress.now !== null) parts.push(nowWords(progress.now));
  parts.push(...errorWords(progress.errors, progress.failedCalls));
  parts.push(
    progress.toolCalls === 1 ? a.progressCallsOne : fill(a.progressCalls, { n: formatCount(progress.toolCalls) }),
  );
  parts.push(fill(a.progressContext, { pct: progress.contextPct }));
  parts.push(fill(a.progressTokens, { n: formatCount(progress.tokens) }));
  const tone: ProgressTone =
    progress.errors > 0
      ? 'danger'
      : progress.contextPct >= CONTEXT_HIGH_PCT
        ? 'warning'
        : quiet !== null
          ? 'muted'
          : 'normal';
  return { text: parts.join(' · '), tone };
}

/** The rail row's third line: text only, its numbers swapping per character. Not a live region. */
export function ProgressLine(props: { progress: HelperProgress | null; now: number; id?: string }): ReactElement {
  const { text, tone } = progressLine(props.progress, props.now);
  return (
    <span id={props.id} className={styles.progress} data-tone={tone} data-progress="" title={text}>
      <SwapText text={text} granularity="char" />
    </span>
  );
}

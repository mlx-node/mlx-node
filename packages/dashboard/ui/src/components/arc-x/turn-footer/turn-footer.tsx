import { Ban, Square, TriangleAlert } from 'lucide-react';
import type { ReactElement, ReactNode } from 'react';

import { Button } from '../../arc/button/button';
import { CopyButton } from '../../arc/copy-button/copy-button';
import { Tooltip } from '../../arc/tooltip/tooltip';
import styles from './turn-footer.module.css';

export interface TurnFooterProps {
  /** Formatted parts, joined with " · ". Empty: no stats. */
  stats: readonly string[];
  /** Tooltip of the stats. */
  statsTitle?: string;
  outcome?: {
    tone: 'neutral' | 'warning' | 'danger';
    text: string;
    detail?: string;
    /** Default: a square for neutral, a warning triangle otherwise. */
    glyph?: 'stop' | 'ban' | 'alert';
  };
  action?: { label: string; onSelect: () => void; disabled?: boolean; disabledReason?: string };
  copy?: { value: string; label: string };
  /** Line 2: what the engine does now, with an optional action (a link). */
  note?: { text: string; action?: ReactNode };
  /** Actions show only on hover or focus inside the nearest `[data-turn]`. */
  quiet?: boolean;
}

function Glyph({ glyph }: { glyph: 'stop' | 'ban' | 'alert' }): ReactElement {
  if (glyph === 'stop') return <Square size={10} strokeWidth={0} fill="currentColor" aria-hidden="true" />;
  if (glyph === 'ban') return <Ban size={12} strokeWidth={1.75} aria-hidden="true" />;
  return <TriangleAlert size={12} strokeWidth={1.75} aria-hidden="true" />;
}

/** One line under a turn: how it ended, its numbers, and its actions (copy, send again). */
export function TurnFooter(props: TurnFooterProps): ReactElement | null {
  const outcome = props.outcome;
  const hasStats = props.stats.length > 0;
  if (outcome === undefined && !hasStats && props.copy === undefined && props.action === undefined) return null;
  const glyph = outcome?.glyph ?? (outcome?.tone === 'neutral' ? 'stop' : 'alert');
  const action = props.action;
  const actionButton =
    action !== undefined ? (
      <Button type="button" variant="ghost" size="sm" disabled={action.disabled === true} onClick={action.onSelect}>
        {action.label}
      </Button>
    ) : null;
  return (
    <div className={styles.root} data-quiet={props.quiet === true ? '' : undefined}>
      <div className={styles.line}>
        {outcome !== undefined ? (
          <span className={styles.outcome} data-tone={outcome.tone}>
            <span className={styles.glyph}>
              <Glyph glyph={glyph} />
            </span>
            <span>{outcome.text}</span>
          </span>
        ) : null}
        {hasStats ? (
          <span className={styles.stats} title={props.statsTitle}>
            {props.stats.join(' · ')}
          </span>
        ) : null}
        <span className={styles.actions}>
          {actionButton !== null && action?.disabled === true && action.disabledReason !== undefined ? (
            <Tooltip content={action.disabledReason}>
              <span className={styles.tipTarget}>{actionButton}</span>
            </Tooltip>
          ) : (
            actionButton
          )}
          {props.copy !== undefined ? (
            <CopyButton value={props.copy.value} label={props.copy.label} iconOnly variant="plain" />
          ) : null}
        </span>
      </div>
      {props.note !== undefined ? (
        <p className={styles.note}>
          <span>{props.note.text}</span>
          {props.note.action}
        </p>
      ) : null}
      {outcome?.detail !== undefined ? (
        <p className={styles.detail} title={outcome.detail}>
          {outcome.detail}
        </p>
      ) : null}
    </div>
  );
}

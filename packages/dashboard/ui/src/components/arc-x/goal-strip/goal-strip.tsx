import { Target } from 'lucide-react';
import { useId, type ReactElement } from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import type { GoalAction } from '../../../lib/chat-types';
import { goalMeta, goalRowState, goalStateWords, type GoalEntry, type GoalView } from '../../../lib/goal';
import { visibleText } from '../../../lib/work-tools';
import { Button } from '../../arc/button/button';
import { ExpandableText } from '../../arc/expandable-text/expandable-text';
import { useNow } from '../../wait-clock';
import { EventRow } from '../event-row/event-row';
import { LiveLabel } from '../live-label/live-label';
import styles from './goal-strip.module.css';

export interface GoalStripProps {
  goal: GoalView;
  /** A press waits for its answer: every button is off until then. */
  pending: GoalAction | null;
  /** An answer, a wake or a history op runs: Resume and Clear are turns of their own, so they are off until it ends. */
  busy?: boolean;
  onAction: (action: GoalAction) => void;
}

/**
 * The goal above the composer: its objective, what it does now, its clock and tokens while it runs, and Pause (while it
 * runs) or Resume and Clear. Narrow, the tokens go first, then the word "Goal" and a running goal's state words. The
 * buttons keep their places, so after Pause the focus stays on the same button, now Resume.
 */
export function GoalStrip({ goal, pending, busy = false, onAction }: GoalStripProps): ReactElement {
  const now = useNow(goal.running);
  const waitId = useId();
  const waits = (action: GoalAction): boolean => busy && action !== 'pause';
  const words = goalStateWords(goal);
  const { clock, tokens } = goalMeta(goal, now);
  const objective = visibleText(goal.entry.objective);
  return (
    <div className={styles.frame} data-goal-strip="" data-state={goal.state}>
      <div
        role="group"
        aria-label={CHAT.goal.label}
        className={styles.box}
        data-running={goal.running ? '' : undefined}
      >
        <div className={styles.line}>
          <Target className={styles.icon} size={14} strokeWidth={1.75} aria-hidden="true" />
          <span className={styles.label} aria-hidden="true">
            {CHAT.goal.label}
          </span>
          <span className={styles.objective} title={objective}>
            {objective}
          </span>
          <span className={styles.state} title={words}>
            {goal.running ? <LiveLabel active>{words}</LiveLabel> : words}
          </span>
          {clock !== null ? (
            <span className={styles.clock} aria-hidden="true" data-goal-clock="">
              {clock}
            </span>
          ) : null}
          {tokens !== null ? (
            <span className={styles.tokens} data-goal-tokens="">
              {tokens}
            </span>
          ) : null}
          <span className={styles.actions}>
            {goal.actions.map((action, slot) => (
              <Button
                // By place, not by action: the button that had focus keeps it when Pause turns into Resume.
                key={slot}
                type="button"
                variant="ghost"
                size="sm"
                aria-disabled={pending !== null || waits(action) ? true : undefined}
                aria-describedby={waits(action) ? waitId : undefined}
                title={waits(action) ? CHAT.modelBusy : undefined}
                onClick={() => {
                  if (pending === null && !waits(action)) onAction(action);
                }}
              >
                {CHAT.goal[action]}
              </Button>
            ))}
            {goal.actions.some(waits) ? (
              <span id={waitId} hidden>
                {CHAT.modelBusy}
              </span>
            ) : null}
          </span>
        </div>
        {goal.state === 'paused' && goal.reason !== null ? (
          <ExpandableText lines={1} className={styles.reason}>
            {visibleText(goal.reason)}
          </ExpandableText>
        ) : null}
      </div>
    </div>
  );
}

/** The thread row where a goal began, updated in place: "Goal · <objective> · <what it did last>". */
export function GoalRow({ entry }: { entry: GoalEntry }): ReactElement {
  const state = goalRowState(entry);
  const objective = visibleText(entry.objective);
  return (
    <article aria-label={fill(CHAT.goal.row, { objective, state })} data-goal-row="">
      <EventRow
        icon={<Target size={14} strokeWidth={1.75} />}
        label={CHAT.goal.label}
        text={objective}
        facts={[state]}
        ts={entry.ts}
      />
    </article>
  );
}

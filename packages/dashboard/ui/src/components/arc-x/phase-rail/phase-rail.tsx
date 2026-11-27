import { Check, Circle, CircleDashed, Pause, Square, X } from 'lucide-react';
import { motion } from 'motion/react';
import { useId, type ReactElement, type ReactNode } from 'react';

import { cn } from '../../../lib/cn';
import type { ChipMark, StepChip } from '../../../lib/run-view';
import { motionTokens } from '../../arc/motion-tokens';
import { Tooltip } from '../../arc/tooltip/tooltip';
import { LiveDot } from '../live-dot/live-dot';
import { useStaticMotion } from '../motion/use-static-motion';
import styles from './phase-rail.module.css';

export interface PhaseRailProps {
  /** The rail's name for a screen reader. */
  label: string;
  chips: readonly StepChip[];
  /** The pressed chip's title; null: none, the list shows every agent. */
  selected: string | null;
  /** A press filters the list by the chip; a press on the pressed chip clears it. */
  onSelect: (title: string | null) => void;
  /** Where the run is (`Step 2 of 3: Verify`): shown beside the dots once the rail is too narrow for titles. */
  stepWords: string | null;
  /** A chip's count: `4`, or `1 of 2` while some of its agents are not done. */
  countOf: (chip: StepChip) => string | null;
}

const ICONS: Record<Exclude<ChipMark, 'active'>, ReactNode> = {
  done: <Check size={12} strokeWidth={2} />,
  pending: <Circle size={10} strokeWidth={1.75} />,
  failed: <X size={12} strokeWidth={2} />,
  stopped: <Square size={10} strokeWidth={1.75} />,
  paused: <Pause size={12} strokeWidth={1.75} />,
  unfinished: <CircleDashed size={12} strokeWidth={1.75} />,
};

const INSTANT = { duration: 0 } as const;

/**
 * One chip per step of a run, in order: its mark, its title and its agents' count; its detail as the tooltip. A chip
 * filters the list under it. Narrow, the chips scroll sideways; narrower still they are dots beside the step's words.
 */
export function PhaseRail(props: PhaseRailProps): ReactElement {
  const staticMotion = useStaticMotion();
  const highlight = useId();
  return (
    <div className={styles.root}>
      <div className={styles.chips} role="group" aria-label={props.label}>
        {props.chips.map((chip) => {
          const pressed = props.selected === chip.title;
          const count = props.countOf(chip);
          const button = (
            <button
              type="button"
              className={styles.chip}
              data-mark={chip.mark}
              aria-pressed={pressed}
              aria-label={count === null ? chip.title : `${chip.title}, ${count}`}
              onClick={() => props.onSelect(pressed ? null : chip.title)}
            >
              {pressed ? (
                <motion.span
                  layoutId={highlight}
                  className={styles.highlight}
                  transition={staticMotion ? INSTANT : motionTokens.spring.morph}
                  aria-hidden="true"
                />
              ) : null}
              <span className={styles.mark} aria-hidden="true">
                {chip.mark === 'active' ? <LiveDot tone="success" size={6} label="" /> : ICONS[chip.mark]}
              </span>
              <span className={styles.title}>{chip.title}</span>
              {count !== null ? <span className={styles.count}>{count}</span> : null}
            </button>
          );
          return chip.detail === null ? (
            <span key={chip.title} className={styles.slot}>
              {button}
            </span>
          ) : (
            <Tooltip key={chip.title} content={chip.detail}>
              {button}
            </Tooltip>
          );
        })}
      </div>
      {props.stepWords !== null ? (
        <span className={cn(styles.stepWords)} aria-hidden="true">
          {props.stepWords}
        </span>
      ) : null}
    </div>
  );
}

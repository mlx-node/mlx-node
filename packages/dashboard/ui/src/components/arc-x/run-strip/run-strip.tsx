import { ChevronRight, CircleHelp, Workflow } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import { useId, type ReactElement, type ReactNode } from 'react';

import type { EntryAnchor } from '../../../lib/activity';
import { CHAT, fill } from '../../../lib/chat-strings';
import type { StripLine } from '../../../lib/run-strip';
import { stopText } from '../../../lib/run-view';
import { Button } from '../../arc/button/button';
import { motionTokens } from '../../arc/motion-tokens';
import { WaitClock } from '../../wait-clock';
import { CONTROL_KEYS } from '../agent-controls/agent-controls';
import { ConfirmPopover } from '../confirm-popover/confirm-popover';
import { IconButton } from '../icon-button/icon-button';
import { LiveDot } from '../live-dot/live-dot';
import { useStaticMotion } from '../motion/use-static-motion';
import styles from './run-strip.module.css';

export interface RunStripProps {
  lines: readonly StripLine[];
  /** Scrolls the thread to the run's block and flashes it. */
  onJump: (anchor: EntryAnchor) => void;
  /** Opens the activity drawer. */
  onOpen: () => void;
  /** Brings the request's card to the front. */
  onReview: (askId: string) => void;
  /** Stop run; rejects with the reason to show. Absent: no Stop. */
  onStop?: (runId: string) => Promise<void>;
}

const INSTANT = { duration: 0 } as const;

function lineKey(line: StripLine): string {
  return line.kind === 'summary' ? 'summary' : `${line.kind}:${line.id}`;
}

function RunLine(props: {
  line: Extract<StripLine, { kind: 'run' }>;
  strip: RunStripProps;
  open: ReactNode;
}): ReactElement {
  const { line, strip } = props;
  const detailId = useId();
  const word = line.stopping ? CHAT.run.status.stopping : CHAT.run.status.running;
  return (
    <>
      <button
        type="button"
        className={styles.main}
        aria-label={fill(CHAT.activity.jump, { title: line.name })}
        aria-describedby={detailId}
        onClick={() => strip.onJump(line.anchor)}
      >
        <Workflow className={styles.icon} size={14} strokeWidth={1.75} aria-hidden="true" />
        <span className={styles.name} title={line.name}>
          {line.name}
        </span>
        <span id={detailId} className={styles.detail}>
          {line.step !== null ? <span className={styles.step}>{line.step}</span> : null}
          {line.counts.length > 0 ? <span className={styles.counts}>{line.counts.join(' · ')}</span> : null}
        </span>
      </button>
      <span className={styles.state}>
        <LiveDot tone={line.stopping ? 'neutral' : 'success'} pulse={!line.stopping} label={word} showLabel size={6} />
        <span className={styles.clock}>
          <WaitClock since={line.startTs} />
        </span>
      </span>
      {line.stoppable && strip.onStop !== undefined ? (
        <ConfirmPopover
          trigger={{
            kind: 'button',
            label: CHAT.run.buttons.stop,
            size: 'sm',
            variant: 'ghost',
            keys: CONTROL_KEYS.stop,
          }}
          title={CHAT.run.stopTitle}
          description={stopText(line.running)}
          confirmLabel={CHAT.run.buttons.stopWorkflow}
          cancelLabel={CHAT.run.buttons.keepRunning}
          workingLabel={CHAT.run.status.stopping}
          successLabel={CHAT.run.status.stopping}
          errorTitle={CHAT.run.stopError}
          onConfirm={() => strip.onStop!(line.id)}
        />
      ) : null}
      {props.open}
    </>
  );
}

/**
 * The live runs above the goal strip while the activity rail is a drawer: a line per run (name, step, counts, clock,
 * Stop, then › for the drawer) for up to two, one summary line for more, and the request an agent of one waits on.
 * Narrow, it drops the counts first, then the step; the state, the clock and the buttons stay.
 */
export function RunStrip(props: RunStripProps): ReactElement | null {
  const staticMotion = useStaticMotion();
  if (props.lines.length === 0) return null;
  const open = (
    <IconButton
      label={CHAT.run.strip.open}
      icon={<ChevronRight size={14} strokeWidth={1.75} aria-hidden="true" />}
      onClick={props.onOpen}
      className={styles.open}
    />
  );
  return (
    <section className={styles.frame} aria-label={CHAT.run.strip.label} data-run-strip="">
      <ul className={styles.box}>
        <AnimatePresence initial={false}>
          {props.lines.map((line) => (
            <motion.li
              key={lineKey(line)}
              className={styles.line}
              data-strip-line={line.kind}
              initial={staticMotion ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={staticMotion ? undefined : { opacity: 0, transition: { duration: motionTokens.duration.exit } }}
              transition={staticMotion ? INSTANT : motionTokens.spring.smooth}
            >
              {line.kind === 'run' ? (
                <RunLine line={line} strip={props} open={open} />
              ) : line.kind === 'ask' ? (
                <>
                  <CircleHelp className={styles.ask} size={14} strokeWidth={1.75} aria-hidden="true" />
                  <span className={styles.askText} title={`${line.name} · ${line.text}`}>
                    <span className={styles.name}>{line.name}</span>
                    {` · ${line.text}`}
                  </span>
                  <Button
                    type="button"
                    size="sm"
                    variant="secondary"
                    className={styles.review}
                    onClick={() => props.onReview(line.id)}
                  >
                    {CHAT.run.buttons.review}
                  </Button>
                </>
              ) : (
                <>
                  <Workflow className={styles.icon} size={14} strokeWidth={1.75} aria-hidden="true" />
                  <span className={styles.summary}>{line.text}</span>
                  {open}
                </>
              )}
            </motion.li>
          ))}
        </AnimatePresence>
      </ul>
    </section>
  );
}

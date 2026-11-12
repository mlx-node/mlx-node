import { Shield, ShieldAlert, ShieldCheck, ShieldOff, TriangleAlert } from 'lucide-react';
import type { ReactElement, ReactNode } from 'react';

import { CHAT, fill } from '../lib/chat-strings';
import type { PermissionInteraction, PermissionMode } from '../lib/chat-types';
import { type AllowedMark, allowedSummary, flagLines, type LineMark, markLine } from '../lib/permission-text';
import { EventRow } from './arc-x/event-row/event-row';
import { ExactText } from './arc-x/interaction-card/interaction-card';
import styles from './chat-permission-marks.module.css';

const GAP_WORDS = { spaces: CHAT.exactSpaces, emptyLines: CHAT.exactEmptyLines };

function exactNote(lines: number): string {
  return lines === 1 ? CHAT.exactLong : fill(CHAT.exactLines, { n: lines });
}

function Exact({ text, label }: { text: string; label: string }): ReactElement {
  return <ExactText text={text} label={label} overflowNote={exactNote} gapWords={GAP_WORDS} />;
}

/** The text a card would have shown for an allowed call: its exact wells, then its warnings. */
export function AllowedCard({ card }: { card: PermissionInteraction }): ReactElement {
  const wells: ReactNode[] = [];
  if (card.access === 'bash' && card.command !== null) {
    wells.push(<Exact key="command" text={card.command} label={CHAT.permCommand} />);
  } else if (card.access === 'edit' && card.path !== null) {
    wells.push(<Exact key="file" text={card.path} label={CHAT.permFile} />);
    if (card.resolvedPath !== null) wells.push(<Exact key="lands" text={card.resolvedPath} label={CHAT.permLands} />);
  } else {
    wells.push(<Exact key="input" text={card.input} label={CHAT.permInput} />);
  }
  const warnings = flagLines(card.flags);
  return (
    <>
      {wells}
      {warnings.length > 0 ? (
        <ul className={styles.warnings}>
          {warnings.map((line) => (
            <li key={line} className={styles.warning}>
              <span className={styles.warningIcon} aria-hidden="true">
                <TriangleAlert size={12} strokeWidth={1.75} />
              </span>
              <span>{line}</span>
            </li>
          ))}
        </ul>
      ) : null}
    </>
  );
}

const MODE_ICONS: Record<PermissionMode, ReactElement> = {
  ask: <Shield size={14} strokeWidth={1.75} />,
  auto: <ShieldCheck size={14} strokeWidth={1.75} />,
  bypass: <ShieldOff size={14} strokeWidth={1.75} />,
};

/** A mode or settings mark: one line of the thread, its icon by the mode it enters. */
export function ModeLine({ mark }: { mark: LineMark }): ReactElement {
  const line = markLine(mark);
  const mode = mark.kind === 'settings' ? null : mark.mode;
  return (
    <EventRow
      icon={mode === null ? <ShieldAlert size={14} strokeWidth={1.75} /> : MODE_ICONS[mode]}
      label={line.label}
      labelClassName={mode === 'bypass' ? styles.danger : undefined}
      text={line.text}
      tone={line.tone}
      ts={mark.at}
      wrap
      className={styles.line}
    />
  );
}

/** An allowed call whose tool row is not in this thread (a helper agent's call). */
export function AllowedLine({ mark }: { mark: AllowedMark }): ReactElement {
  return (
    <EventRow
      icon={MODE_ICONS[mark.mode]}
      label={CHAT.permission.allowed[mark.mode]}
      text={allowedSummary(mark)}
      ts={mark.at}
      className={styles.line}
    />
  );
}

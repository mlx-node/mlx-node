import { Shield, ShieldAlert, ShieldCheck, ShieldOff, TriangleAlert } from 'lucide-react';
import type { ReactElement, ReactNode } from 'react';

import { CHAT, fill } from '../lib/chat-strings';
import type { PermissionInteraction, PermissionMode } from '../lib/chat-types';
import {
  type AllowedMark,
  allowedSummary,
  allowedWord,
  changedPaths,
  flagLines,
  type LineMark,
  markLine,
  shortPath,
} from '../lib/permission-text';
import { EventRow } from './arc-x/event-row/event-row';
import { ExactText } from './arc-x/interaction-card/interaction-card';
import { ThreadMark } from './arc-x/thread-mark/thread-mark';
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

const MARK_ICONS: Record<PermissionMode | 'settings', ReactElement> = {
  ask: <Shield size={12} strokeWidth={1.75} />,
  auto: <ShieldCheck size={12} strokeWidth={1.75} />,
  bypass: <ShieldOff size={12} strokeWidth={1.75} />,
  settings: <ShieldAlert size={12} strokeWidth={1.75} />,
};

/**
 * "Changed: `a`, `b` and 2 more.": each path shortened, in code font, and kept on one line with the punctuation after
 * it. At least one path.
 */
function ChangedFiles({ changed }: { changed: readonly string[] }): ReactElement {
  const p = CHAT.permission;
  const { paths, more } = changedPaths(changed);
  const path = (text: string, after: string): ReactElement => (
    <span className={styles.path}>
      <code title={text}>{shortPath(text)}</code>
      {after}
    </span>
  );
  return (
    <>
      {`${p.settingsChanged} `}
      {more > 0 ? (
        <>
          {path(paths[0]!, ',')} {path(paths[1]!, '')}
          {` ${fill(p.settingsMore, { n: more })}.`}
        </>
      ) : paths.length === 2 ? (
        <>
          {path(paths[0]!, '')}
          {` ${p.settingsAnd} `}
          {path(paths[1]!, '.')}
        </>
      ) : (
        path(paths[0]!, '.')
      )}
    </>
  );
}

/** A mode or settings mark: a rule across the thread, its icon by the mode it enters. */
export function ModeLine({ mark, temp }: { mark: LineMark; temp: string | null }): ReactElement {
  const line = markLine(mark, temp);
  const detail =
    mark.kind === 'settings' ? (
      mark.changed.length > 0 ? (
        <ChangedFiles changed={mark.changed} />
      ) : undefined
    ) : line.detail.length > 0 ? (
      line.detail.join(' ')
    ) : undefined;
  return (
    <ThreadMark
      kind={mark.kind}
      icon={MARK_ICONS[mark.kind === 'settings' ? 'settings' : mark.mode]}
      label={line.label}
      fact={line.fact}
      detail={detail}
      tone={line.tone}
      ts={mark.at}
      data={mark.kind === 'settings' ? undefined : { 'data-mode': mark.mode }}
    />
  );
}

/** An allowed call whose tool row is not in this thread (a helper agent's call). */
export function AllowedLine({ mark, temp }: { mark: AllowedMark; temp: string | null }): ReactElement {
  return (
    <EventRow icon={MODE_ICONS[mark.mode]} label={allowedWord(mark, temp)} text={allowedSummary(mark)} ts={mark.at} />
  );
}

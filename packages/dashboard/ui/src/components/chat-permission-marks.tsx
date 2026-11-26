import { Plug, Shield, ShieldAlert, ShieldCheck, ShieldOff, ShieldX, TriangleAlert, Undo2 } from 'lucide-react';
import { useState, type ReactElement, type ReactNode } from 'react';

import { CHAT, fill } from '../lib/chat-strings';
import type { PermissionInteraction, PermissionMark, PermissionMode } from '../lib/chat-types';
import {
  type AllowedMark,
  type DeniedMark,
  allowedSummary,
  allowedWord,
  deniedWord,
  changedPaths,
  flagLines,
  type LineMark,
  markLine,
  type McpMark,
  mcpLine,
  type RewindMark,
  rewindRule,
  shortPath,
  subagentDeniedText,
} from '../lib/permission-text';
import { Disclosure } from './arc-x/disclosure/disclosure';
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

/** A call denied with a note whose tool row is not in this thread (a helper agent's call). */
export function DeniedLine({ mark }: { mark: DeniedMark }): ReactElement {
  return (
    <EventRow
      icon={<ShieldX size={14} strokeWidth={1.75} />}
      label={deniedWord(mark)}
      text={allowedSummary(mark)}
      ts={mark.at}
    />
  );
}

/** A helper's calls a mode allowed and those the user denied with a note, in closed disclosures under its row. */
export function HelperMarkLists(props: {
  allowed: readonly AllowedMark[] | null;
  denied: readonly DeniedMark[] | null;
  temp: string | null;
  /** Each new value past 0 opens both lists (the Agent view's counts asked for them). */
  openTick?: number;
}): ReactElement {
  const { allowed, denied } = props;
  const tick = props.openTick ?? 0;
  const [open, setOpen] = useState({ allowed: false, denied: false });
  const [seen, setSeen] = useState(tick);
  if (tick !== seen) {
    setSeen(tick);
    if (tick > 0) setOpen({ allowed: true, denied: true });
  }
  return (
    <>
      {allowed === null ? null : (
        <Disclosure
          title={CHAT.permission.subagentAllowedList}
          open={open.allowed}
          onOpenChange={(next) => setOpen((now) => ({ ...now, allowed: next }))}
        >
          <div className="grid min-w-0 gap-2">
            {allowed.map((mark, n) => (
              <AllowedLine key={`${mark.at}:${mark.toolCallId}:${n}`} mark={mark} temp={props.temp} />
            ))}
          </div>
        </Disclosure>
      )}
      {denied === null ? null : (
        <Disclosure
          title={subagentDeniedText(denied.length)}
          open={open.denied}
          onOpenChange={(next) => setOpen((now) => ({ ...now, denied: next }))}
        >
          <div className="grid min-w-0 gap-2">
            {denied.map((mark, n) => (
              <DeniedLine key={`${mark.at}:${mark.toolCallId}:${n}`} mark={mark} />
            ))}
          </div>
        </Disclosure>
      )}
    </>
  );
}

/** Work: the user rewound here; the rule stays in the thread where the removed part was. */
export function RewindLine({ mark }: { mark: RewindMark }): ReactElement {
  return (
    <ThreadMark
      kind="rewind"
      icon={<Undo2 size={12} strokeWidth={1.75} />}
      label={rewindRule(mark)}
      ts={mark.at}
      data={{ 'data-rewind-mode': mark.mode }}
    />
  );
}

/** Work: an MCP server asked for input; the line says how it ended, never what was typed. */
export function McpLine({ mark }: { mark: McpMark }): ReactElement {
  const line = mcpLine(mark);
  return (
    <EventRow
      icon={<Plug size={14} strokeWidth={1.75} />}
      label={line.label}
      text={line.outcome}
      ts={mark.at}
      tone={mark.outcome === 'cancel' || mark.outcome === 'expired' ? 'warning' : 'neutral'}
    />
  );
}

/** A mark drawn as its own line: an allowed or denied call with no row here, a mode or settings line, a rewind, an MCP request. */
export function MarkLine({ mark, temp }: { mark: PermissionMark; temp: string | null }): ReactElement {
  if (mark.kind === 'allowed') return <AllowedLine mark={mark} temp={temp} />;
  if (mark.kind === 'denied') return <DeniedLine mark={mark} />;
  if (mark.kind === 'mcp') return <McpLine mark={mark} />;
  if (mark.kind === 'rewind') return <RewindLine mark={mark} />;
  return <ModeLine mark={mark} temp={temp} />;
}

import { CircleHelp, ClipboardCheck, FilePen, Plug, Shield, SquareTerminal } from 'lucide-react';
import { useState, type ReactElement, type ReactNode } from 'react';

import { type AskFrom, type AskRow, waitWords } from '../../../lib/activity';
import { CHAT, fill } from '../../../lib/chat-strings';
import { WaitClock } from '../../wait-clock';
import { PathText } from '../path-text/path-text';
import styles from './activity-rail.module.css';

/** Characters of an edit's path in the row; the whole path is its hover title. */
const PATH_CHARS = 32;

function iconOf(row: AskRow): ReactNode {
  const props = { size: 14, strokeWidth: 1.75, 'aria-hidden': true } as const;
  if (row.kind === 'plan') return <ClipboardCheck {...props} />;
  if (row.kind === 'question') return <CircleHelp {...props} />;
  if (row.kind === 'mcp-input') return <Plug {...props} />;
  switch (row.access) {
    case 'bash':
      return <SquareTerminal {...props} />;
    case 'edit':
      return <FilePen {...props} />;
    case 'mcp':
      return <Plug {...props} />;
    default:
      return <Shield {...props} />;
  }
}

export function kindWord(row: AskRow): string {
  if (row.kind === 'plan') return CHAT.needs.kinds.plan;
  if (row.kind === 'question') return CHAT.needs.kinds.question;
  if (row.kind === 'mcp-input') return CHAT.needs.kinds.mcpInput;
  return CHAT.needs.kinds[row.access ?? 'other'];
}

/** Who asks, as the row's second line says it; null for the conversation's own agent. */
export function fromWords(from: AskFrom): string | null {
  switch (from.who) {
    case 'agent':
      return null;
    case 'subagent':
      return fill(CHAT.needs.fromSubagent, { label: from.label });
    case 'workflow':
      return fill(CHAT.needs.fromWorkflow, { run: from.run, label: from.label });
    case 'run':
      return fill(CHAT.needs.fromRun, { run: from.run });
  }
}

/** The second line: the kind, who asks when a helper does, and "Auto asks". */
export function askDetail(row: AskRow): string {
  const parts = [kindWord(row)];
  if (row.questions !== undefined && row.questions > 1) parts.push(fill(CHAT.needs.questions, { n: row.questions }));
  const who = fromWords(row.from);
  if (who !== null) parts.push(who);
  if (row.autoAsk) parts.push(CHAT.needs.autoAsks);
  return parts.join(' · ');
}

/** The row's name: the clock is hidden from assistive tech, so the words carry the wait. */
export function askName(row: AskRow, now: number): string {
  const time = waitWords(now - row.since);
  const who = fromWords(row.from);
  if (row.kind === 'mcp-input') {
    if (who === null) return fill(CHAT.needs.rowNameServer, { title: row.title, kind: kindWord(row), time });
    return fill(CHAT.needs.rowName, { title: row.title, kind: kindWord(row), who, time });
  }
  return fill(CHAT.needs.rowName, {
    title: row.title,
    kind: kindWord(row).toLowerCase(),
    who: who ?? CHAT.needs.fromAgent,
    time,
  });
}

/** One open request in the rail: a click brings its card to the front of the thread. */
export function AskRowView({ row, onAsk }: { row: AskRow; onAsk: (id: string) => void }): ReactElement {
  // No timer of its own (the SPA's timers are counted): the name's wait is read again when the row is reached.
  const [now, setNow] = useState(Date.now);
  const reread = (): void => setNow(Date.now());
  return (
    <li>
      <button
        type="button"
        className={styles.row}
        data-front={row.front ? '' : undefined}
        data-ask={row.kind}
        aria-current={row.front ? 'true' : undefined}
        aria-label={askName(row, now)}
        onFocus={reread}
        onPointerEnter={reread}
        onClick={() => onAsk(row.id)}
      >
        <span className={styles.icon}>{iconOf(row)}</span>
        <span className={styles.text}>
          {row.path ? (
            <PathText path={row.title} keepEnd maxChars={PATH_CHARS} className={styles.title} />
          ) : (
            <span className={styles.title} title={row.title}>
              {row.title}
            </span>
          )}
          <span className={styles.detail}>{askDetail(row)}</span>
        </span>
        <span className={styles.meta}>
          <WaitClock since={row.since} />
        </span>
      </button>
    </li>
  );
}

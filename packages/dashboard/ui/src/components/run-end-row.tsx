import { Check, CircleAlert, Square, X } from 'lucide-react';
import type { ReactElement, ReactNode } from 'react';

import { CHAT } from '../lib/chat-strings';
import { type EndRow, endRowOf } from '../lib/run-view';
import type { TranscriptEntry } from '../lib/types';
import { ChatMarkdown } from './arc-x/chat-markdown/chat-markdown';
import { Disclosure } from './arc-x/disclosure/disclosure';
import { EventRow } from './arc-x/event-row/event-row';
import { Button } from './arc/button/button';

type WakeEntry = Extract<TranscriptEntry, { kind: 'wake' }>;

export interface RunEndRowProps {
  entry: WakeEntry;
  /** The entry the run's block is drawn at: `Show run ↑`; null when it is not in the loaded transcript. */
  runAt: number | null;
  onJump?: (at: number) => void;
  /** The block's own name when it has one (a run of the PDF tool reads `Reading {file}`). */
  name?: string;
}

function glyphOf(row: EndRow, status: string | undefined): ReactNode {
  if (row.tone === 'danger') return <X size={14} strokeWidth={1.75} />;
  if (status === 'budget_limited') return <CircleAlert size={14} strokeWidth={1.75} />;
  if (row.tone === 'success' || row.tone === 'warning') return <Check size={14} strokeWidth={1.75} />;
  return <Square size={12} strokeWidth={1.75} />;
}

/**
 * The wake row of a run: how it ended, from the stored update its prompt id names (a resume empties the run's newest
 * report, so the row never reads it), its report in a closed fold, and the link back to its block. The main agent's
 * answer of the wake turn follows it.
 */
export function RunEndRow(props: RunEndRowProps): ReactElement | null {
  const row = endRowOf(props.entry, props.name);
  if (row === null) return null;
  const status = props.entry.workflow?.snapshot?.status;
  const runAt = props.runAt;
  const onJump = props.onJump;
  return (
    <div className="grid min-w-0 gap-1" data-run-end={props.entry.workflow?.runId}>
      <EventRow
        icon={glyphOf(row, status)}
        label={row.title}
        text=""
        facts={row.facts}
        tone={row.tone === 'danger' ? 'danger' : row.tone === 'warning' ? 'warning' : 'neutral'}
        action={
          runAt !== null && onJump !== undefined ? (
            <Button type="button" size="sm" variant="ghost" onClick={() => onJump(runAt)}>
              {CHAT.run.links.showRunUp}
            </Button>
          ) : undefined
        }
        ts={props.entry.ts}
      />
      {row.report !== null ? (
        <Disclosure title={CHAT.run.links.report}>
          <div className="grid min-w-0 gap-2">
            <ChatMarkdown text={row.report} links={{ mode: 'work' }} />
            {row.cut ? <p className="m-0 text-caption text-muted">{CHAT.run.reportCut.note}</p> : null}
          </div>
        </Disclosure>
      ) : null}
    </div>
  );
}

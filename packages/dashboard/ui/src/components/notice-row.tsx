import { Flag, ImageDown, ImageOff, RotateCw, TriangleAlert } from 'lucide-react';
import type { ReactElement } from 'react';

import { hookNoteRow, hookRow, pictureRow, retryRow } from '../lib/chat-turns';
import type { TranscriptEntry } from '../lib/types';
import { visibleText } from '../lib/work-tools';
import { EventRow } from './arc-x/event-row/event-row';

/**
 * A stored notice of the agent as one quiet event row, the same in the thread and on the Sessions page. `running`: its
 * turn runs now. The agent's own words are plain text with hidden characters shown as their codes.
 */
export function NoticeRow({
  entry,
  running,
}: {
  entry: Extract<TranscriptEntry, { kind: 'notice' }>;
  running: boolean;
}): ReactElement {
  const view =
    entry.code === 'picture'
      ? pictureRow(entry)
      : entry.code === 'hook'
        ? hookRow(entry)
        : entry.code === 'hook-note'
          ? hookNoteRow(entry)
          : retryRow(entry, running);
  const icon =
    entry.code === 'retry' ? (
      <RotateCw size={14} strokeWidth={1.75} />
    ) : entry.code === 'hook' ? (
      <TriangleAlert size={14} strokeWidth={1.75} />
    ) : entry.code === 'hook-note' ? (
      <Flag size={14} strokeWidth={1.75} />
    ) : entry.what === 'dropped' ? (
      <ImageOff size={14} strokeWidth={1.75} />
    ) : (
      <ImageDown size={14} strokeWidth={1.75} />
    );
  const counted = entry.code === 'hook' && entry.times > 1;
  return (
    <div data-notice-row={entry.code}>
      <EventRow
        icon={icon}
        label={view.label}
        text={view.text}
        facts={view.facts}
        tone={view.tone}
        detail={view.detail === undefined ? undefined : visibleText(view.detail)}
        ts={counted ? undefined : entry.ts}
      />
    </div>
  );
}

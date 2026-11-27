import { PencilLine } from 'lucide-react';
import type { ReactElement } from 'react';

import { CHAT } from '../lib/chat-strings';
import { oneLine } from '../lib/web-tools';
import { visibleText } from '../lib/work-tools';
import { EventRow } from './arc-x/event-row/event-row';

/** Characters of a note the row shows and its hover text holds. */
const NOTE_SHOWN = 500;

/**
 * The user's note on a call denied with one: an event row inside the answer, read as "Your note: …". Plain text on one
 * line, hidden characters shown as their codes.
 */
export function NoteRow({ text, ts }: { text: string; ts?: number }): ReactElement {
  const line = oneLine(visibleText(text)).trim();
  const shown = line.length > NOTE_SHOWN ? `${line.slice(0, NOTE_SHOWN)}…` : line;
  return (
    <div data-note-row="">
      <EventRow icon={<PencilLine size={14} strokeWidth={1.75} />} label={CHAT.noteRow} text={shown} ts={ts} />
    </div>
  );
}

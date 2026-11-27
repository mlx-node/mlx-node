import { CircleAlert } from 'lucide-react';
import { type ReactElement, useState } from 'react';

import { CHAT, fill } from '../lib/chat-strings';
import type { UnreadPdf } from '../lib/pdf-run';
import { toast } from '../lib/toast';
import { EventRow } from './arc-x/event-row/event-row';
import { Button } from './arc/button/button';

/**
 * Under a turn that sent a PDF to the PDF tool when no run read it: one line, and Read it now while the conversation's
 * model sees images (the agent starts the run itself). The Sessions page draws the line alone.
 */
export function PdfUnreadRow(props: { pdf: UnreadPdf; onRead?: (id: string) => Promise<void> }): ReactElement {
  const { pdf, onRead } = props;
  const [pressed, setPressed] = useState(false);
  const read = (): void => {
    if (onRead === undefined || pressed) return;
    setPressed(true);
    onRead(pdf.id).then(
      () => setPressed(false),
      () => {
        setPressed(false);
        toast.error(CHAT.pdf.readNowFailed);
      },
    );
  };
  return (
    <EventRow
      icon={<CircleAlert size={14} strokeWidth={1.75} />}
      label={fill(CHAT.pdf.unread, { name: pdf.name })}
      text=""
      tone="warning"
      action={
        onRead === undefined ? undefined : (
          <Button type="button" size="sm" variant="secondary" disabled={pressed} onClick={read}>
            {CHAT.pdf.readNow}
          </Button>
        )
      }
    />
  );
}

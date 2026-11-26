import { Ban } from 'lucide-react';
import type { ReactElement } from 'react';

import { CHAT, fill } from '../lib/chat-strings';
import { oneLine } from '../lib/web-tools';
import { visibleText } from '../lib/work-tools';
import { EventRow } from './arc-x/event-row/event-row';
import { Button } from './arc/button/button';

/**
 * The note the app sent for a call a never-allow rule denied, in place of a "Your note" row: "Your rule · never allow
 * git push", with Remove rule while the rule is kept. Plain text on one line, hidden characters shown as their codes.
 */
export function RuleRow(props: {
  words: string;
  ts?: number;
  /** Removes the rule; none: the rule is gone, or this view cannot remove it. */
  onRemove?: () => void;
  removing?: boolean;
}): ReactElement {
  const words = oneLine(visibleText(props.words)).trim();
  const remove = props.onRemove;
  return (
    <div data-rule-row="">
      <EventRow
        icon={<Ban size={14} strokeWidth={1.75} />}
        label={CHAT.never.ruleLine}
        text={fill(CHAT.never.ruleText, { words })}
        ts={props.ts}
        action={
          remove === undefined ? undefined : (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              loading={props.removing === true}
              aria-label={fill(CHAT.never.removeLabel, { words })}
              onClick={remove}
            >
              {CHAT.never.remove}
            </Button>
          )
        }
      />
    </div>
  );
}

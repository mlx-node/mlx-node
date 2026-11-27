import type { CSSProperties, ReactElement } from 'react';

import { parseSlash, slashToken } from '../lib/chat-input';
import { Badge } from './arc/badge/badge';
import { ExpandableText } from './arc/expandable-text/expandable-text';

/** ExpandableText sets its own size and tracking; sent text keeps the bubble's reading size and tracking. */
export const SENT_TEXT_STYLE: CSSProperties = { fontSize: 'inherit', letterSpacing: 'inherit' };

/** The command a sent message starts with, as typed, and the words after it; null for any other text. */
export function commandHead(text: string): { name: string; rest: string } | null {
  const token = slashToken(text);
  if (token === null || parseSlash(text) === null) return null;
  return { name: text.slice(token.start + 1, token.end), rest: text.slice(token.end).trim() };
}

/** A sent message's text; one that is a slash command shows the command as a chip before its words. */
export function MessageText({ text }: { text: string }): ReactElement {
  const head = commandHead(text);
  if (head === null)
    return (
      <ExpandableText lines={12} style={SENT_TEXT_STYLE}>
        {text}
      </ExpandableText>
    );
  return (
    <div className="flex min-w-0 flex-wrap items-baseline gap-1.5" data-command-text="">
      <Badge size="sm">/{head.name}</Badge>
      {head.rest !== '' ? (
        <ExpandableText lines={12} style={SENT_TEXT_STYLE}>
          {head.rest}
        </ExpandableText>
      ) : null}
    </div>
  );
}

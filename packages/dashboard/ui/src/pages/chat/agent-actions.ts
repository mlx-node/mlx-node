/**
 * Work: Stop and Message for one agent of a conversation, the same on the Chat page and on the agent's Sessions page.
 * A Stop's failure is a toast with Try again; a message's outcome is a toast or a reason the field shows.
 */

import { transcriptPath } from '../../lib/activity';
import type { MessageEnd, MessageTarget } from '../../lib/agent-message';
import { tooLongWords } from '../../lib/agent-message';
import { DashboardApiError, mutate } from '../../lib/api';
import { CHAT, fill } from '../../lib/chat-strings';
import type { SubagentMessageResult } from '../../lib/chat-types';
import { toast } from '../../lib/toast';
import { conversationPath } from './conversation-actions';

function agentPath(conversationId: string, childId: string, action: string): string {
  return `${conversationPath(conversationId)}/subagents/${encodeURIComponent(childId)}/${action}`;
}

/** The confirm already asked; a failed press is a toast whose Try again sends the Stop once more. */
export function stopAgent(
  conversationId: string,
  target: MessageTarget,
  drafts: Map<string, string>,
  announce: (sentence: string) => void,
): void {
  drafts.delete(target.id);
  const send = (): void => {
    mutate('POST', agentPath(conversationId, target.id, 'stop')).then(
      () => announce(CHAT.status.taskStopped),
      (err: unknown) => {
        if (err instanceof DashboardApiError && err.code === 'E_NOT_FOUND') {
          toast.info(CHAT.activity.stopGone);
          return;
        }
        toast.error(fill(CHAT.activity.stopError, { title: target.title }), {
          action: { label: CHAT.activity.tryAgain, onClick: send },
        });
      },
    );
  };
  send();
}

/**
 * A message to a running subagent. Sent and unsure are toasts with Open (its transcript); a reason the field can show
 * stays in it with the text; an agent that ended drops the draft.
 */
export async function messageAgent(
  conversationId: string,
  target: MessageTarget,
  text: string,
  deps: { drafts: Map<string, string>; announce: (sentence: string) => void; open: (path: string) => void },
): Promise<MessageEnd> {
  const open = { label: CHAT.activity.messageOpenAction, onClick: () => deps.open(transcriptPath(target.id)) };
  let result: SubagentMessageResult;
  try {
    result = (
      await mutate<{ result: SubagentMessageResult }>('POST', agentPath(conversationId, target.id, 'message'), {
        text,
      })
    ).result;
  } catch (err) {
    if (!(err instanceof DashboardApiError && err.code === 'E_NOT_FOUND'))
      return { kind: 'stay', reason: CHAT.activity.messageError };
    deps.drafts.delete(target.id);
    toast.info(CHAT.activity.messageGone);
    return { kind: 'close', clear: true };
  }
  switch (result) {
    case 'sent':
      deps.drafts.delete(target.id);
      toast.success(fill(CHAT.activity.messageSent, { title: target.title }), { action: open });
      deps.announce(CHAT.status.messageSent);
      return { kind: 'close', clear: true };
    case 'unsure':
      toast.warning(fill(CHAT.activity.messageUnsure, { title: target.title }), { action: open });
      return { kind: 'close', clear: false };
    case 'off':
      toast.error(CHAT.activity.messageOff);
      return { kind: 'close', clear: false };
    case 'not-ready':
      return { kind: 'stay', reason: CHAT.activity.messageNotReady };
    case 'busy':
      return { kind: 'stay', reason: CHAT.activity.messageBusy };
    case 'late':
      return { kind: 'stay', reason: CHAT.activity.messageLate };
    case 'too-long':
      return { kind: 'stay', reason: tooLongWords(text) };
    case 'error':
      return { kind: 'stay', reason: CHAT.activity.messageError };
  }
}

/** The Agent view's watch: its start, and its end when the view closes (answered the same when none was open). */
export function watchPath(conversationId: string, childId: string): string {
  return agentPath(conversationId, childId, 'watch');
}

/**
 * Work: the head controls and message field of an agent's own Sessions page, while the agent process holds the
 * conversation that started it. Else one line says the page reads from disk, with Open in Work. Nothing for an agent
 * whose parent is no Work conversation of this app.
 */

import { type ReactElement, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import {
  AgentMessageField,
  type AgentMessaging,
  AgentStateLine,
  agentFacts,
  type PendingMessage,
} from '../../components/agent-view';
import { ButtonLink } from '../../components/arc-x/button-link/button-link';
import { useNow } from '../../components/wait-clock';
import type { MessageTarget } from '../../lib/agent-message';
import { type AgentAsk, agentRowOf, runAgents, subagentRows } from '../../lib/agents';
import type { ConversationSnapshot } from '../../lib/chat-store';
import { CHAT } from '../../lib/chat-strings';
import { messageAgent, stopAgent } from '../chat/agent-actions';
import { watchConversation } from '../chat/use-conversation';
import styles from './child-controls.module.css';

/** The Chat page of `parentId` with the agent's view open. */
export function workViewUrl(parentId: string, childId: string): string {
  return `/chat?c=${encodeURIComponent(parentId)}&agent=${encodeURIComponent(childId)}`;
}

export function ChildControls({ childId, parentId }: { childId: string; parentId: string }): ReactElement | null {
  const navigate = useNavigate();
  const [snapshot, setSnapshot] = useState<ConversationSnapshot>({ state: null, gone: null });
  useEffect(() => watchConversation(parentId, setSnapshot), [parentId]);
  const state = snapshot.state;
  const entries = state?.entries;
  const row = useMemo(() => {
    if (entries === undefined) return null;
    const rows = [...subagentRows(entries).values(), ...[...runAgents(entries).values()].flat()];
    return agentRowOf(rows, childId);
  }, [entries, childId]);
  const interactions = state?.interactions;
  const asks = useMemo(() => {
    const list: AgentAsk[] = [];
    for (const item of interactions ?? []) {
      if (item.kind !== 'permission' || item.from === undefined) continue;
      const from = item.from;
      list.push({
        id: item.id,
        child: from.child,
        run: from.run,
        label: from.label,
        title: item.title,
        access: item.access,
        openedAt: item.openedAt,
      });
    }
    return list;
  }, [interactions]);
  const drafts = useRef(new Map<string, string>()).current;
  const pending = useRef(new Map<string, PendingMessage>()).current;
  const [said, setSaid] = useState({ n: 0, text: '' });
  const announce = useCallback((text: string) => setSaid((now) => ({ n: now.n + 1, text })), []);
  const messaging = useMemo<AgentMessaging>(
    () => ({
      drafts,
      pending,
      onSend: (target: MessageTarget, text: string) =>
        messageAgent(parentId, target, text, { drafts, announce, open: (path) => void navigate(path) }),
    }),
    [drafts, pending, parentId, announce, navigate],
  );
  const hot = state?.conversation.hot === true;
  const now = useNow(hot && row !== null);
  if (state === null || state.conversation.mode !== 'work' || row === null) return null;
  if (!hot) {
    return (
      <div className={styles.root} data-child-controls="disk">
        <p className={styles.disk}>{CHAT.agent.fromDisk}</p>
        <ButtonLink to={workViewUrl(parentId, row.childId)} variant="secondary" size="sm">
          {CHAT.agent.view.openInWork}
        </ButtonLink>
      </div>
    );
  }
  const facts = agentFacts(row, { agentLive: true, live: state.live, asks, now });
  return (
    <div className={styles.root} data-child-controls="live">
      <AgentStateLine
        row={row}
        facts={facts}
        handlers={{
          onStop: (target) => stopAgent(parentId, target, drafts, announce),
          onReview: () => void navigate(workViewUrl(parentId, row.childId)),
        }}
      />
      <AgentMessageField
        target={{ id: row.childId, title: row.label }}
        field={facts.field}
        messaging={messaging}
        focusTick={0}
      />
      <div role="status" aria-live="polite" className="sr-only">
        <span key={said.n}>{said.text}</span>
      </div>
    </div>
  );
}

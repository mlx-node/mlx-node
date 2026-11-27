/** Work: the open Agent view of the Chat page, with its watch and the attempt it shows. */

import { type ReactElement, useState } from 'react';

import { type AgentViewProps, AgentView } from '../../components/agent-view';
import type { AgentRow as AgentRowData } from '../../lib/agents';
import { useAgentFeed } from './use-agent-feed';

export type AgentPaneProps = Omit<AgentViewProps, 'attempt' | 'feed' | 'onRetry' | 'onAttempt'> & {
  conversationId: string;
};

/** The attempt shown: the one picked while the row's attempts stay the same, else the newest (a retry opens on it). */
function useAttempt(row: AgentRowData): [string, (childId: string) => void] {
  const key = row.attempts.join('\n');
  const [pick, setPick] = useState<{ key: string; attempt: string } | null>(null);
  const attempt = pick !== null && pick.key === key ? pick.attempt : row.childId;
  return [attempt, (childId) => setPick({ key, attempt: childId })];
}

export function AgentPane(props: AgentPaneProps): ReactElement {
  const { conversationId, ...view } = props;
  const [attempt, setAttempt] = useAttempt(props.row);
  const { feed, retry } = useAgentFeed(conversationId, attempt);
  return <AgentView {...view} attempt={attempt} feed={feed} onRetry={retry} onAttempt={setAttempt} />;
}

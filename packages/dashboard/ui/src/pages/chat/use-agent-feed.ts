/**
 * Work: the list one attempt's Agent view shows. It watches the attempt for this window, applies the watch's ops from
 * the conversation stream in order, watches again after every `sync` (the host drops a window's watches when the
 * conversation has no page, as while the window is hidden), and ends the watch when the view closes.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import { type AgentFeed, feedAnswer, feedEvent, feedFailed, feedStart } from '../../lib/agent-view';
import { mutate } from '../../lib/api';
import type { AgentWatchAnswer } from '../../lib/chat-types';
import { WINDOW_ID } from '../../lib/window-id';
import { watchPath } from './agent-actions';
import { listenConversation } from './use-conversation';

/**
 * The watch calls of one agent go one after another: the host reads the file before it holds a watch, so an end sent
 * before a start's answer would reach it first and leave that watch held, taking one of the window's two.
 */
const chains = new Map<string, Promise<unknown>>();

function inTurn<T>(key: string, call: () => Promise<T>): Promise<T> {
  const before = chains.get(key) ?? Promise.resolve();
  const next = before.then(call, call);
  const settled = next.catch(() => undefined);
  chains.set(key, settled);
  void settled.then(() => {
    if (chains.get(key) === settled) chains.delete(key);
  });
  return next;
}

/**
 * Mounted views of each agent. The host holds one watch per window and agent, and a view that mounts takes it over, as
 * the column does while the drawer slides out: only the last view to close ends it.
 */
const holders = new Map<string, number>();

export interface AgentFeedHandle {
  feed: AgentFeed | null;
  /** Watch again (after a failed watch). */
  retry: () => void;
}

export function useAgentFeed(conversationId: string, childId: string): AgentFeedHandle {
  const [feed, setFeed] = useState<AgentFeed | null>(null);
  const requests = useRef(0);
  const start = useRef<() => void>(() => undefined);

  useEffect(() => {
    let closed = false;
    const path = watchPath(conversationId, childId);
    const key = `${conversationId}\u0000${childId}`;
    const apply = (next: (feed: AgentFeed) => AgentFeed): void => {
      if (!closed) setFeed((current) => (current === null ? current : next(current)));
    };
    const watch = (): void => {
      const request = ++requests.current;
      setFeed((current) => feedStart(current, childId, request));
      inTurn(key, () => mutate<AgentWatchAnswer>('POST', path, { window: WINDOW_ID })).then(
        (answer) => apply((current) => feedAnswer(current, request, answer)),
        () => apply((current) => feedFailed(current, request)),
      );
    };
    start.current = watch;
    holders.set(key, (holders.get(key) ?? 0) + 1);
    // The answer of the watch and the stream's events come over one port in the order the host sent them.
    const stop = listenConversation(conversationId, (event) => {
      if (event.type === 'sync') watch();
      else if (event.type === 'agent-entries') apply((current) => feedEvent(current, event, WINDOW_ID));
    });
    watch();
    return () => {
      closed = true;
      stop();
      start.current = () => undefined;
      const left = (holders.get(key) ?? 1) - 1;
      if (left > 0) {
        holders.set(key, left);
        return;
      }
      holders.delete(key);
      const query = `?window=${encodeURIComponent(WINDOW_ID)}`;
      inTurn(key, () => mutate('DELETE', `${path}${query}`)).catch(() => undefined);
    };
  }, [conversationId, childId]);

  const retry = useCallback(() => start.current(), []);
  return { feed: feed !== null && feed.childId === childId ? feed : null, retry };
}

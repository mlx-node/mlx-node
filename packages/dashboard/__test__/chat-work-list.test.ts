/** The Work list names every session of the Work index, however many terminal sessions are newer. */

import { describe, expect, it } from 'vite-plus/test';

import type { SessionListQuery, SessionListResponse, SessionRow } from '../src/agent-sessions/types.js';
import type { WorkerApiContext } from '../src/api/context.js';
import { handleChatConversations } from '../src/api/handlers/chat.js';

function row(id: string): SessionRow {
  return {
    id,
    origin: 'agent',
    title: id,
    cwd: '/Users/me/code',
    model: null,
    branch: null,
    createdAt: '2026-11-01T00:00:00.000Z',
    lastActiveAt: '2026-11-01T00:00:00.000Z',
    messageCount: 1,
    live: false,
    kind: null,
    parentId: null,
    subagentCount: 0,
    tokens: null,
  };
}

describe('GET /api/chat/conversations?mode=work', () => {
  it('finds a Work session behind more than a page of newer terminal sessions', async () => {
    const all = Array.from({ length: 1_200 }, (_, i) => row(i === 1_100 ? 'work-old' : i === 3 ? 'work-new' : `t${i}`));
    const ctx = {
      chatIndex: () => ({
        version: 1,
        work: [
          { id: 'work-new', createdAt: 2 },
          { id: 'work-old', createdAt: 1 },
        ],
      }),
      agentSessions: {
        list: (query: SessionListQuery): SessionListResponse => {
          const offset = query.offset ?? 0;
          return { sessions: all.slice(offset, offset + (query.limit ?? 50)), total: all.length, cwds: [] };
        },
      },
    } as unknown as WorkerApiContext;
    const req = { method: 'GET', path: '', params: {}, body: null, query: new URLSearchParams('mode=work') };
    const body = (await handleChatConversations(ctx, req)) as { conversations: Array<{ id: string }> };
    expect(body.conversations.map((c) => c.id)).toEqual(['work-new', 'work-old']);
  });
});

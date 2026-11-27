/** The Work list names every session of the Work index, however many terminal sessions are newer. */

import { mkdirSync, mkdtempSync, realpathSync, renameSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { describe, expect, it } from 'vite-plus/test';

import type { SessionListQuery, SessionListResponse, SessionRow } from '../src/agent-sessions/types.js';
import type { WorkerApiContext } from '../src/api/context.js';
import { handleChatConversations } from '../src/api/handlers/chat.js';
import { ScratchFolders } from '../src/chat/work-scratch.js';

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

describe('GET /api/chat/conversations: forks', () => {
  const fork = (id: string, parentId: string): SessionRow => ({ ...row(id), kind: 'fork', parentId });
  const req = (mode: string) => ({
    method: 'GET',
    path: '',
    params: {},
    body: null,
    query: new URLSearchParams(`mode=${mode}`),
  });

  // Mutation: drop the `kind === 'fork'` check → a subagent's parent link reads as a fork.
  it('a fork row names its origin, with the title while the origin is listed; other rows have none', async () => {
    const sessions = [
      fork('copy', 'orig'),
      fork('orphan', 'gone'),
      { ...row('child'), kind: 'subagent_resume', parentId: 'orig' },
      row('orig'),
    ];
    const ctx = {
      chatSessions: {
        list: (): SessionListResponse => ({ sessions, total: sessions.length, cwds: [] }),
      },
      chatIndex: () => ({ version: 1, work: sessions.map((s, n) => ({ id: s.id, createdAt: n })) }),
      agentSessions: {
        list: (): SessionListResponse => ({ sessions, total: sessions.length, cwds: [] }),
      },
    } as unknown as WorkerApiContext;
    for (const mode of ['chat', 'work']) {
      const body = (await handleChatConversations(ctx, req(mode))) as {
        conversations: Array<{ id: string; forkOf?: unknown }>;
      };
      expect(body.conversations.map((c) => [c.id, c.forkOf])).toEqual([
        ['copy', { id: 'orig', title: 'orig' }],
        ['orphan', { id: 'gone', title: null }],
        ['child', undefined],
        ['orig', undefined],
      ]);
    }
  });
});

describe('GET /api/chat/conversations: Work without a folder', () => {
  const req = { method: 'GET', path: '', params: {}, body: null, query: new URLSearchParams('mode=work') };

  // Mutation: the parent folder read in place of the record → a folder picked in ~/Portato reads as No folder; the
  // record read by path alone → a new folder at a recorded path does too.
  it('marks a row only when its folder has a record and is still the folder recorded', async () => {
    const base = mkdtempSync(join(tmpdir(), 'mlx-work-list-'));
    try {
      const root = join(base, 'home', 'Portato');
      const folder = (name: string): string => {
        const path = join(root, name);
        mkdirSync(path, { recursive: true });
        return realpathSync(path);
      };
      const own = folder('2026-11-26-2057');
      const replaced = folder('2026-11-26-2058');
      const picked = folder('my-project');
      const file = join(base, 'chat', 'scratch-folders.json');
      const records = new ScratchFolders(file);
      await records.add(own);
      await records.add(replaced);
      renameSync(replaced, `${replaced}-old`);
      mkdirSync(replaced);
      const cwds: Array<[string, string]> = [
        ['own', own],
        ['replaced', replaced],
        ['picked', picked],
        ['elsewhere', '/Users/me/code'],
      ];
      const sessions = cwds.map(([id, cwd]) => ({ ...row(id), cwd }));
      const ctx = (scratchFolders: string) =>
        ({
          chatIndex: () => ({ version: 1, work: sessions.map((s, n) => ({ id: s.id, createdAt: n })) }),
          agentSessions: { list: (): SessionListResponse => ({ sessions, total: sessions.length, cwds: [] }) },
          scratchFolders,
        }) as unknown as WorkerApiContext;
      const marks = async (scratchFolders: string): Promise<Array<[string, unknown]>> => {
        const body = (await handleChatConversations(ctx(scratchFolders), req)) as {
          conversations: Array<{ id: string; noFolder?: true }>;
        };
        return body.conversations.map((c) => [c.id, c.noFolder]);
      };
      expect(await marks(file)).toEqual([
        ['own', true],
        ['replaced', undefined],
        ['picked', undefined],
        ['elsewhere', undefined],
      ]);
      expect((await marks(join(base, 'missing.json'))).every(([, mark]) => mark === undefined)).toBe(true);
    } finally {
      rmSync(base, { recursive: true, force: true });
    }
  });
});

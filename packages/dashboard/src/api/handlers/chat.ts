/**
 * The Chat routes. Every body is checked against its exact field list (`route-input.ts`) and every handler runs under
 * the error text law (`route-errors.ts`): a failure is a typed code with fixed text, never a raw error message.
 */

import { MAX_LIST_LIMIT } from '../../agent-sessions/reader.js';
import type { SessionRow } from '../../agent-sessions/types.js';
import { workIds } from '../../chat/index-file.js';
import { guarded } from '../../chat/route-errors.js';
import {
  agentActionInput,
  createConversationInput,
  idParam,
  interactionInput,
  modeQuery,
  noBody,
  patchConversationInput,
  permissionModeInput,
  promptInput,
} from '../../chat/route-input.js';
import type { ChatService } from '../../chat/service.js';
import { agentTitle } from '../../chat/titles.js';
import type { ChatMode, ConversationRow } from '../../chat/types.js';
import type { ApiRequest, MainApiContext, WorkerApiContext } from '../context.js';
import { requireBody } from '../context.js';
import { ApiError } from '../errors.js';

function service(ctx: MainApiContext): ChatService {
  if (ctx.chat === undefined) throw new ApiError('E_UNAVAILABLE', 'The agent could not be reached.');
  return ctx.chat;
}

function epochOf(iso: string): number {
  const ms = Date.parse(iso);
  return Number.isFinite(ms) ? ms : 0;
}

function conversationRow(row: SessionRow, mode: ChatMode): ConversationRow {
  return {
    id: row.id,
    mode,
    title: agentTitle(row.title),
    cwd: mode === 'work' ? row.cwd : null,
    modelId: row.model ?? '',
    createdAt: epochOf(row.createdAt),
    updatedAt: epochOf(row.lastActiveAt),
    hot: false,
    permissionMode: 'ask',
    turns: row.messageCount,
  };
}

/** `GET /api/chat/conversations?mode=chat|work`, newest first. Answered by the database worker. */
export function handleChatConversations(ctx: WorkerApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const mode = modeQuery(req.query);
    if (mode === 'chat') {
      const rows = ctx.chatSessions.list({ limit: MAX_LIST_LIMIT }).sessions;
      return { conversations: rows.map((row) => conversationRow(row, 'chat')) };
    }
    const work = workIds(ctx.chatIndex());
    if (work.size === 0) return { conversations: [] };
    const rows: SessionRow[] = [];
    for (let offset = 0; rows.length < work.size; offset += MAX_LIST_LIMIT) {
      const page = ctx.agentSessions.list({ limit: MAX_LIST_LIMIT, offset });
      for (const row of page.sessions) if (work.has(row.id)) rows.push(row);
      if (offset + MAX_LIST_LIMIT >= page.total) break;
    }
    return { conversations: rows.slice(0, MAX_LIST_LIMIT).map((row) => conversationRow(row, 'work')) };
  });
}

export function handleChatState(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    noBody(req.body);
    return service(ctx).state();
  });
}

export function handleChatCreate(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => service(ctx).create(createConversationInput(requireBody(req))));
}

export function handleChatPrompt(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    return service(ctx).prompt(id, promptInput(requireBody(req)).text);
  });
}

export function handleChatCancel(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    noBody(req.body);
    return service(ctx).cancel(id);
  });
}

export function handleChatInteraction(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    const interactionId = idParam(req.params.iid);
    return service(ctx).answer(id, interactionId, interactionInput(requireBody(req)));
  });
}

/**
 * `POST /api/chat/conversations/:id/permission-mode {mode, up, temp?}`: the only way to move a permission mode up, and only
 * with `up: true`. A request with `up: false` to a mode above the conversation's is 409 `mode-changed`.
 */
export function handleChatPermissionMode(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    const input = permissionModeInput(requireBody(req));
    return service(ctx).setPermissionMode(id, input.mode, input.up, input.temp);
  });
}

export function handleChatPatch(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    return service(ctx).patch(id, patchConversationInput(requireBody(req)));
  });
}

export function handleChatDelete(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    noBody(req.body);
    return service(ctx).delete(id);
  });
}

export function handleChatEngineUnload(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    noBody(req.body);
    return service(ctx).unloadModels();
  });
}

export function handleChatAgent(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => service(ctx).agentAction(agentActionInput(requireBody(req))));
}

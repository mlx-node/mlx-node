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
  agentWatchInput,
  agentWatchQuery,
  createConversationInput,
  draftPermissionModeInput,
  forkInput,
  goalInput,
  idParam,
  interactionInput,
  modeQuery,
  noBody,
  patchConversationInput,
  pdfReadInput,
  pdfStageInput,
  permissionModeInput,
  promptInput,
  queuePatchInput,
  queueRowParam,
  queueSendInput,
  rewindInput,
  rewindPlanInput,
  ruleParam,
  runParam,
  scheduleParam,
  stageParam,
  subagentMessageInput,
  subagentParam,
  taskParam,
  workGitQuery,
  workHooksQuery,
  workSettingsInput,
  workSwitchInput,
} from '../../chat/route-input.js';
import type { ChatService } from '../../chat/service.js';
import { agentTitle } from '../../chat/titles.js';
import type { ChatMode, ConversationRow } from '../../chat/types.js';
import { isOwned, readScratchRecordsSync, type ScratchRecord } from '../../chat/work-scratch.js';
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

/**
 * A list row. `listed`: the titles of the rows of this list, by id; a fork's origin not among them reads as gone (a
 * fork's origin is a conversation of the same list). `scratch`: the folders Work without a folder made, by path.
 */
function conversationRow(
  row: SessionRow,
  mode: ChatMode,
  listed: ReadonlyMap<string, string>,
  scratch: ReadonlyMap<string, ScratchRecord>,
): ConversationRow {
  return {
    id: row.id,
    mode,
    title: agentTitle(row.title),
    cwd: mode === 'work' ? row.cwd : null,
    ...(mode === 'work' && isOwned(row.cwd, scratch) ? { noFolder: true as const } : {}),
    modelId: row.model ?? '',
    createdAt: epochOf(row.createdAt),
    updatedAt: epochOf(row.lastActiveAt),
    hot: false,
    permissionMode: 'ask',
    turns: row.messageCount,
    ...(row.kind === 'fork' && row.parentId !== null
      ? { forkOf: { id: row.parentId, title: listed.get(row.parentId) ?? null } }
      : {}),
  };
}

function rowsOf(
  rows: readonly SessionRow[],
  mode: ChatMode,
  scratch: ReadonlyMap<string, ScratchRecord> = new Map(),
): ConversationRow[] {
  const listed = new Map(rows.map((row) => [row.id, agentTitle(row.title)]));
  return rows.map((row) => conversationRow(row, mode, listed, scratch));
}

/** `GET /api/chat/conversations?mode=chat|work`, newest first. Answered by the database worker. */
export function handleChatConversations(ctx: WorkerApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const mode = modeQuery(req.query);
    if (mode === 'chat') {
      const rows = ctx.chatSessions.list({ limit: MAX_LIST_LIMIT }).sessions;
      return { conversations: rowsOf(rows, 'chat') };
    }
    const work = workIds(ctx.chatIndex());
    if (work.size === 0) return { conversations: [] };
    const rows: SessionRow[] = [];
    for (let offset = 0; rows.length < work.size; offset += MAX_LIST_LIMIT) {
      const page = ctx.agentSessions.list({ limit: MAX_LIST_LIMIT, offset });
      for (const row of page.sessions) if (work.has(row.id)) rows.push(row);
      if (offset + MAX_LIST_LIMIT >= page.total) break;
    }
    return { conversations: rowsOf(rows.slice(0, MAX_LIST_LIMIT), 'work', readScratchRecordsSync(ctx.scratchFolders)) };
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
    const input = promptInput(requireBody(req));
    const chat = service(ctx);
    if (input.whenBusy === undefined) return chat.prompt(id, input.text, input.attachments, input.plan);
    return chat.prompt(id, input.text, input.attachments, input.plan, input.whenBusy);
  });
}

/** `PATCH …/:id/queue/:qid {text} | {editing}`: a queued row's new text, or a page opening or closing it for edit. */
export function handleChatQueueEdit(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    const rowId = queueRowParam(req.params.qid);
    return service(ctx).queueEdit(id, rowId, queuePatchInput(requireBody(req)));
  });
}

export function handleChatQueueRemove(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    const rowId = queueRowParam(req.params.qid);
    noBody(req.body);
    return service(ctx).queueRemove(id, rowId);
  });
}

/** `POST …/:id/queue/:qid/send {how: "now"}`: Work's Send now of a queued row. */
export function handleChatQueueSend(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    const rowId = queueRowParam(req.params.qid);
    queueSendInput(requireBody(req));
    return service(ctx).queueSend(id, rowId);
  });
}

/** `DELETE …/:id/queue`: Clear. */
export function handleChatQueueClear(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    noBody(req.body);
    return service(ctx).queueClear(id);
  });
}

/** `POST …/:id/queue/release`: Send next. The hold ends; the first row goes when the conversation is free. */
export function handleChatQueueRelease(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    noBody(req.body);
    return service(ctx).queueRelease(id);
  });
}

export function handleChatCancel(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    noBody(req.body);
    return service(ctx).cancel(id);
  });
}

/**
 * `POST /api/chat/conversations/:id/compact`: Compact now. 202 once the run started; it ends on the conversation stream
 * (`compaction` with `end`), and `…/cancel` stops it. 409 `conversation-busy` while the conversation works.
 */
export function handleChatCompact(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    noBody(req.body);
    return service(ctx).compact(id);
  });
}

/**
 * `POST /api/chat/conversations/:id/goal {action}`: a press on the goal strip (Work). 202 with `{turnId}` once the
 * agent's own `/goal pause|resume|clear` started, or `{stopped: true}` when Pause stopped the running answer. 400 in
 * Chat; 409 `conversation-busy` for Resume or Clear while the conversation works.
 */
export function handleChatGoal(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    return service(ctx).goal(id, goalInput(requireBody(req)).action);
  });
}

/**
 * `POST /api/chat/pdf-stage {stage, page, pages, data}`: one page of a PDF for the PDF tool, in order; `stage` null
 * opens a stage with page 1. Answers `{stage}`; `attachment-refused` for a page out of order or past a limit.
 */
export function handleChatPdfStage(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => service(ctx).stagePdf(pdfStageInput(requireBody(req))));
}

/** `DELETE /api/chat/pdf-stage/:stage`: the PDF's chip was removed before Send; its pages go. */
export function handleChatPdfStageDelete(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const stage = stageParam(req.params.stage);
    noBody(req.body);
    return service(ctx).dropPdfStage(stage);
  });
}

/**
 * `POST /api/chat/conversations/:id/pdf-read {pdf}`: Read it now (Work). 202 with `{turnId}` once the agent's own
 * `/workflow read-pdf` started; 404 for a PDF the conversation did not send to the tool, 400 in Chat.
 */
export function handleChatPdfRead(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    return service(ctx).readPdf(id, pdfReadInput(requireBody(req)).pdf);
  });
}

/**
 * `POST /api/chat/conversations/:id/recap`: the agent writes where things stand. 202 once the wait started; the wait and
 * how it ended ride the conversation stream (`recap`), and the recap itself is a stored row. 409 `conversation-busy`
 * while an answer, a wake, a history op or another recap runs.
 */
export function handleChatRecap(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    noBody(req.body);
    return service(ctx).recap(id);
  });
}

/**
 * `POST /api/chat/conversations/:id/rewind/plan {entry, ts}`: what Rewind to here would remove, and the token its
 * commit takes. 409 `rewind-changed` when the entry at `entry` is not the page's (another time), `rewind-not-point` for
 * a message the agent never ran, `conversation-busy` while the conversation works.
 */
export function handleChatRewindPlan(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    const input = rewindPlanInput(requireBody(req));
    return service(ctx).rewindPlan(id, input.entry, input.ts);
  });
}

/**
 * `POST /api/chat/conversations/:id/rewind {token, mode}`: the plan's rewind. 202 once it started; it ends on the
 * conversation stream (`rewind`). 409 `rewind-changed` when the conversation moved since the plan.
 */
export function handleChatRewind(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    return service(ctx).rewind(id, rewindInput(requireBody(req)));
  });
}

/**
 * `POST /api/chat/conversations/:id/fork {} | {entry, ts}`: a copy of the conversation, whole or up to and with the
 * answer of the message at `entry` (`ts`: its time); answers `{id}` of the copy once it is on disk. 409 `fork-changed`
 * when the entry at `entry` is not the page's, `fork-not-point`, `fork-running`, `recap-running`, `conversation-busy`.
 */
export function handleChatFork(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    const input = forkInput(requireBody(req));
    return service(ctx).fork(id, input.at);
  });
}

/**
 * `GET /api/chat/conversations/:id/context`: the Context panel. The agent's parts while the conversation is loaded in
 * the running agent, else `parts: null` and the stored use; never starts the agent.
 */
export function handleChatContext(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    noBody(req.body);
    return service(ctx).context(id);
  });
}

/**
 * `POST /api/chat/conversations/:id/runs/:runId/stop`: Stop run. 404 `No such run.` when the run is not live, 409 when
 * the agent rejects the stop, 503 when its reply does not come.
 */
export function handleChatStopRun(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    const runId = runParam(req.params.runId);
    noBody(req.body);
    return service(ctx).stopRun(id, runId);
  });
}

/**
 * `POST /api/chat/conversations/:id/tasks/:taskId/stop`: Stop a background command or monitor. 404 `No such task.`
 * when it is not live, 503 when the agent's reply does not come. Its end rides the conversation stream.
 */
export function handleChatStopTask(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    const taskId = taskParam(req.params.taskId);
    noBody(req.body);
    return service(ctx).stopTask(id, taskId);
  });
}

/**
 * `POST /api/chat/conversations/:id/subagents/:childId/stop`: Stop one of the conversation's own subagents. 404 `No
 * such subagent.` when it is not running here, 503 when the agent's reply does not come. Its end rides the conversation
 * stream.
 */
export function handleChatStopSubagent(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    const childId = subagentParam(req.params.childId);
    noBody(req.body);
    return service(ctx).stopSubagent(id, childId);
  });
}

/**
 * `POST /api/chat/conversations/:id/subagents/:childId/message` `{text}`: a message to one of the conversation's own
 * running subagents, steered into its turn. `{result}` names how it ended; 404 `No such subagent.` when it is not
 * running here or the agent refused it as gone.
 */
export function handleChatMessageSubagent(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    const childId = subagentParam(req.params.childId);
    const { text } = subagentMessageInput(requireBody(req));
    return service(ctx).messageSubagent(id, childId, text);
  });
}

/**
 * `POST /api/chat/conversations/:id/subagents/:childId/watch` `{window}`: the Agent view of one agent the conversation
 * started (`childId`: the attempt it shows), as `{childId, attempts, entries, live}`. While `live`, the conversation
 * stream's `agent-entries` events for `window` carry its later ops and, at its end, the stored list; a `sync` of the
 * stream means the window watches again. 404 `No such subagent.` for an agent that is not the conversation's own, 409
 * for a third live view in one window.
 */
export function handleChatWatchSubagent(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    const childId = subagentParam(req.params.childId);
    const { window } = agentWatchInput(requireBody(req));
    return service(ctx).watchSubagent(id, childId, window);
  });
}

/** `DELETE /api/chat/conversations/:id/subagents/:childId/watch?window=<id>`: the view closed. Always `{}`. */
export function handleChatUnwatchSubagent(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    const childId = subagentParam(req.params.childId);
    const { window } = agentWatchQuery(req.query);
    noBody(req.body);
    return service(ctx).unwatchSubagent(id, childId, window);
  });
}

/**
 * `POST /api/chat/conversations/:id/schedules/:scheduleId/delete`: Delete a schedule, loading the conversation first
 * when the running agent does not hold it. 404 `No such schedule.` when it is not listed or the agent no longer holds
 * it, 409 while the engine restarted under live commands, 503 when the agent's reply does not come. Its row leaves on
 * the conversation stream.
 */
export function handleChatDeleteSchedule(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    const scheduleId = scheduleParam(req.params.scheduleId);
    noBody(req.body);
    return service(ctx).deleteSchedule(id, scheduleId);
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
 * `GET /api/chat/conversations/:id/never`: the never-allow rules of a Work conversation's folder. 404 for a Chat
 * conversation.
 */
export function handleChatNeverRules(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    noBody(req.body);
    return service(ctx).neverRules(id);
  });
}

/**
 * `DELETE /api/chat/conversations/:id/never/:ruleId`: removes a never-allow rule of the conversation's folder. 404 `No
 * such rule.` when the folder has no such rule, 500 when the rule file was not written (the rule stays).
 */
export function handleChatNeverRemove(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const id = idParam(req.params.id);
    const ruleId = ruleParam(req.params.ruleId);
    noBody(req.body);
    return service(ctx).removeNeverRule(id, ruleId);
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

/**
 * `POST /api/chat/drafts/:draft/permission-mode {mode, up, cwd?, temp?}`: the mode picked on the New Work screen,
 * kept by the Control Panel for that draft and folder until the first message makes the conversation.
 */
export function handleChatDraftPermissionMode(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    const draft = idParam(req.params.draft);
    const input = draftPermissionModeInput(requireBody(req));
    return service(ctx).setDraftPermissionMode(draft, input.mode, input.up, input.cwd, input.temp);
  });
}

/** `GET /api/chat/work/settings`: the app-wide Work settings. */
export function handleChatWorkSettings(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    noBody(req.body);
    return service(ctx).getWorkSettings();
  });
}

/** `PUT /api/chat/work/settings`: saves them; the Work agent gets them at its next start. */
export function handleChatWorkSettingsPut(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => service(ctx).setWorkSettings(workSettingsInput(requireBody(req))));
}

/** `GET /api/chat/work/hooks?cwd=`: whether the user has tool hooks for that Work folder. */
export function handleChatWorkHooks(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    noBody(req.body);
    return service(ctx).workHooks(workHooksQuery(req.query));
  });
}

/** `GET /api/chat/work/git?cwd=`: the work tree of that Work folder, its local branches and uncommitted changes. */
export function handleChatWorkGit(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => {
    noBody(req.body);
    return service(ctx).workGit(workGitQuery(req.query));
  });
}

/** `POST /api/chat/work/git/switch`: `git switch` to a local branch the read lists. */
export function handleChatWorkGitSwitch(ctx: MainApiContext, req: ApiRequest): Promise<unknown> {
  return guarded('chat', () => service(ctx).switchWorkBranch(workSwitchInput(requireBody(req))));
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

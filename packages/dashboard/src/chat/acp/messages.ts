/** The ACP shapes the adapter reads. Every inbound value goes through one of these readers; nothing is cast unchecked. */

import type { CommandRow } from '../commands.js';
import type { RewindFileState } from '../types.js';

export type Json = Record<string, unknown>;

export function isRecord(value: unknown): value is Json {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function str(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

/** `_meta` of a record, or an empty object. */
function metaOf(value: Json): Json {
  return isRecord(value._meta) ? value._meta : {};
}

export interface InitializeFacts {
  protocolVersion: unknown;
  loadSession: boolean;
  embeddedContext: boolean;
  pluginDirs: boolean;
  /** `x.ai/workflows/stop` and the requester on child asks are in this build. */
  workflowControl: boolean;
  agentVersion: string | null;
  /** null when `modelState.availableModels` is not an array of `{modelId: string}`. */
  models: Array<{ modelId: string; contextTokens: number | null; efforts: EffortOption[] | null }> | null;
}

/**
 * One reasoning-effort option of a model, as the agent lists it in `_meta.reasoningEfforts` (its serialized
 * `ReasoningEffortOption`, the engine's `/v1/models` row): `id` is what `session/set_config_option` names, `value` the
 * level it sends.
 */
export interface EffortOption {
  id: string;
  value: string;
  default: boolean;
}

/** The options of `_meta.reasoningEfforts`; null when the key is missing or lists none that read. */
function readEffortOptions(value: unknown): EffortOption[] | null {
  if (!Array.isArray(value)) return null;
  const out: EffortOption[] = [];
  for (const item of value) {
    if (!isRecord(item) || typeof item.id !== 'string' || typeof item.value !== 'string') continue;
    out.push({ id: item.id, value: item.value, default: item.default === true });
  }
  return out.length === 0 ? null : out;
}

export function readInitialize(result: unknown): InitializeFacts {
  const r = isRecord(result) ? result : {};
  const caps = isRecord(r.agentCapabilities) ? r.agentCapabilities : {};
  const prompt = isRecord(caps.promptCapabilities) ? caps.promptCapabilities : {};
  const meta = metaOf(r);
  const state = isRecord(meta.modelState) ? meta.modelState : {};
  let models: InitializeFacts['models'] = null;
  if (Array.isArray(state.availableModels)) {
    models = [];
    for (const m of state.availableModels) {
      if (!isRecord(m) || typeof m.modelId !== 'string') {
        models = null;
        break;
      }
      const mm = metaOf(m);
      const ctx = mm.totalContextTokens;
      models.push({
        modelId: m.modelId,
        contextTokens: typeof ctx === 'number' && Number.isSafeInteger(ctx) && ctx > 0 ? ctx : null,
        efforts: readEffortOptions(mm.reasoningEfforts),
      });
    }
  }
  return {
    protocolVersion: r.protocolVersion,
    loadSession: caps.loadSession === true,
    embeddedContext: prompt.embeddedContext === true,
    pluginDirs: meta['x.ai/pluginDirs'] === true,
    workflowControl: meta['x.ai/workflowControl'] === true,
    agentVersion: str(meta.agentVersion),
    models,
  };
}

export interface SessionReplyFacts {
  sessionId: string | null;
  currentModelId: string | null;
  /** `_meta.currentWorkingDirectory` (new), else `_meta["x.ai/sessionDetail"].cwd` (load). */
  cwd: string | null;
  /** The option ids of the session's `reasoning_effort` config option; null when the reply has no such option. */
  efforts: string[] | null;
  /** That option's `currentValue`; null with no option. */
  effort: string | null;
}

export function readSessionReply(result: unknown): SessionReplyFacts {
  const r = isRecord(result) ? result : {};
  const meta = metaOf(r);
  const models = isRecord(r.models) ? r.models : {};
  const detail = isRecord(meta['x.ai/sessionDetail']) ? meta['x.ai/sessionDetail'] : {};
  const effort = readEffortConfig(r.configOptions);
  return {
    sessionId: str(r.sessionId) ?? str(meta.sessionId),
    currentModelId: str(models.currentModelId),
    cwd: str(meta.currentWorkingDirectory) ?? str(detail.cwd),
    efforts: effort?.ids ?? null,
    effort: effort?.current ?? null,
  };
}

/**
 * The `reasoning_effort` select of a `configOptions` list (a session reply, the `session/set_config_option` reply, a
 * `config_option_update` push): its flat option ids and `currentValue`. Null when the list has no such select.
 */
export function readEffortConfig(value: unknown): { ids: string[]; current: string | null } | null {
  if (!Array.isArray(value)) return null;
  for (const option of value) {
    if (!isRecord(option) || option.id !== 'reasoning_effort' || option.type !== 'select') continue;
    if (!Array.isArray(option.options)) return null;
    const ids: string[] = [];
    for (const item of option.options) if (isRecord(item) && typeof item.value === 'string') ids.push(item.value);
    return { ids, current: str(option.currentValue) };
  }
  return null;
}

export interface SessionInfoFacts {
  agentName: string | null;
  model: string | null;
  apiBackend: string | null;
  toolDefinitionsCount: number | null;
  systemPromptTokens: number | null;
  toolDefinitionsTokens: number | null;
  usageCategories: Array<{ label: string; detail: string }>;
  /** The agent's own count of the context now, and its window. */
  used: number | null;
  total: number | null;
  /** The agent's estimate of the conversation's items (bytes / 4); with the two above it need not add up to `used`. */
  messageTokens: number | null;
  freeTokens: number | null;
  compactionCount: number | null;
  /** Where the agent compacts by itself, 0–100 of the window. */
  autoCompactThresholdPercent: number | null;
  /** The current prompt index (0-based), as the agent's rewind names a prompt. */
  turnIndex: number | null;
}

/**
 * `_x.ai/session/info` nests its payload once more: `reply.result.result`. Any other shape is null. A reply with no
 * `context` (the session is not loaded) gives null numbers, never zeros.
 */
export function readSessionInfo(result: unknown): SessionInfoFacts | null {
  if (!isRecord(result) || !isRecord(result.result)) return null;
  const r = result.result;
  const ctx = isRecord(r.context) ? r.context : {};
  const count = (v: unknown): number | null => (typeof v === 'number' && Number.isSafeInteger(v) && v >= 0 ? v : null);
  const categories: SessionInfoFacts['usageCategories'] = [];
  if (Array.isArray(ctx.usageCategories)) {
    for (const c of ctx.usageCategories) {
      if (isRecord(c) && typeof c.label === 'string') categories.push({ label: c.label, detail: str(c.detail) ?? '' });
    }
  }
  const percent = count(ctx.autoCompactThresholdPercent);
  return {
    agentName: str(r.agentName),
    model: str(r.model),
    apiBackend: str(r.apiBackend),
    toolDefinitionsCount: count(ctx.toolDefinitionsCount),
    systemPromptTokens: count(ctx.systemPromptTokens),
    toolDefinitionsTokens: count(ctx.toolDefinitionsTokens),
    usageCategories: categories,
    used: count(ctx.used),
    total: count(ctx.total),
    messageTokens: count(ctx.messageTokens),
    freeTokens: count(ctx.freeTokens),
    compactionCount: count(ctx.compactionCount),
    autoCompactThresholdPercent: percent !== null && percent <= 100 ? percent : null,
    turnIndex: count(r.turnIndex),
  };
}

export interface ServerStatus {
  sessionId: string;
  name: string;
  status: string;
}

export function readServerStatus(params: unknown): ServerStatus | null {
  if (!isRecord(params)) return null;
  const sessionId = str(params.sessionId);
  const name = str(params.name);
  const status = str(params.status);
  return sessionId !== null && name !== null && status !== null ? { sessionId, name, status } : null;
}

export interface PermissionOption {
  optionId: string;
  name: string;
  kind: string;
}

export interface PermissionRequest {
  sessionId: string;
  toolCallId: string;
  /** `toolCall.kind` (`execute`, `edit`, `other`, …). */
  kind: string | null;
  title: string;
  /** `toolCall._meta["x.ai/tool"]`, when it is an object. */
  tool: { name: string | null; namespace: string | null } | null;
  rawInput: unknown;
  options: PermissionOption[];
  /** The request's `_meta.hookAsk`: set when a `PreToolUse` hook forced this request. */
  hookAsk?: { hookName: string; reason: string | null };
  /** `_meta["x.ai/requester"]`: the child session that asked (`sessionId` stays the parent's); null for the parent's own ask. */
  requester: Requester | null;
}

export interface Requester {
  sessionId: string;
  label: string;
  /** The workflow run of the child; null for a plain `spawn_subagent` child. */
  runId: string | null;
  cwd: string;
}

/** A present `x.ai/requester` of another shape is `undefined`: the request is unreadable. */
function readRequester(value: unknown): Requester | null | undefined {
  if (value === undefined) return null;
  if (!isRecord(value)) return undefined;
  const sessionId = str(value.sessionId);
  const label = str(value.label);
  const cwd = str(value.cwd);
  const runId = value.runId === null ? null : str(value.runId);
  if (sessionId === null || sessionId === '' || label === null || cwd === null || cwd === '') return undefined;
  if (runId === '' || (runId === null && value.runId !== null)) return undefined;
  return { sessionId, label, runId, cwd };
}

export function readPermissionRequest(params: unknown): PermissionRequest | null {
  if (!isRecord(params) || !isRecord(params.toolCall) || !Array.isArray(params.options)) return null;
  const sessionId = str(params.sessionId);
  const call = params.toolCall;
  const toolCallId = str(call.toolCallId);
  if (sessionId === null || toolCallId === null) return null;
  const options: PermissionOption[] = [];
  for (const o of params.options) {
    if (!isRecord(o) || typeof o.optionId !== 'string') return null;
    options.push({ optionId: o.optionId, name: str(o.name) ?? '', kind: str(o.kind) ?? '' });
  }
  const meta = metaOf(call);
  const tool = isRecord(meta['x.ai/tool']) ? meta['x.ai/tool'] : null;
  const requestMeta = metaOf(params);
  const requester = readRequester(requestMeta['x.ai/requester']);
  if (requester === undefined) return null;
  const ask = requestMeta.hookAsk;
  const hookName = isRecord(ask) ? str(ask.hookName) : null;
  return {
    sessionId,
    toolCallId,
    kind: str(call.kind),
    title: str(call.title) ?? '',
    tool: tool === null ? null : { name: str(tool.name), namespace: str(tool.namespace) },
    rawInput: call.rawInput,
    options,
    ...(isRecord(ask) && hookName !== null ? { hookAsk: { hookName, reason: str(ask.reason) } } : {}),
    requester,
  };
}

/**
 * The methods that carry session updates. The agent sends its background task and schedule updates live under methods
 * of their own and stores them as `_x.ai/session/update` lines, all but `scheduled_task_fired` and `monitor_event`,
 * which are live only.
 */
export const UPDATE_METHODS = [
  'session/update',
  '_x.ai/session_notification',
  '_x.ai/session/update',
  '_x.ai/task_backgrounded',
  '_x.ai/task_completed',
  '_x.ai/scheduled_task_created',
  '_x.ai/scheduled_task_fired',
  '_x.ai/scheduled_task_deleted',
  '_x.ai/monitor_event',
] as const;

export interface SessionUpdateMessage {
  sessionId: string;
  /** True for every method but `session/update`. */
  xai: boolean;
  update: Json;
  meta: Json | undefined;
}

export function readSessionUpdate(method: string, params: unknown): SessionUpdateMessage | null {
  if (!(UPDATE_METHODS as readonly string[]).includes(method)) return null;
  if (!isRecord(params) || !isRecord(params.update) || typeof params.update.sessionUpdate !== 'string') return null;
  const sessionId = str(params.sessionId);
  if (sessionId === null) return null;
  return {
    sessionId,
    xai: method !== 'session/update',
    update: params.update,
    meta: isRecord(params._meta) ? params._meta : undefined,
  };
}

/**
 * The agent's live-only words on its own requests, never stored (`S/session/pending_interaction.rs`): `opened` for
 * `pending_interaction`, `resolved` for `interaction_resolved` (its wait ended for any reason). The id is the session's
 * `tool_call_id`, null when it is not a string. Null for every other update.
 */
export function readAskSignal(update: Json): { kind: 'opened' | 'resolved'; toolCallId: string | null } | null {
  if (update.sessionUpdate === 'pending_interaction') return { kind: 'opened', toolCallId: str(update.tool_call_id) };
  if (update.sessionUpdate === 'interaction_resolved')
    return { kind: 'resolved', toolCallId: str(update.tool_call_id) };
  return null;
}

export interface PromptResultFacts {
  stopReason: string | null;
  cancellationCategory: string | null;
}

export function readPromptResult(result: unknown): PromptResultFacts {
  const r = isRecord(result) ? result : {};
  const meta = metaOf(r);
  return { stopReason: str(r.stopReason), cancellationCategory: str(meta.cancellationCategory) };
}

/**
 * The agent's compaction updates (xAI only): `started` before it summarizes older messages, then one of `completed`,
 * `failed` or `cancelled`. A manual `/compact` sends `completed` alone. A count the agent did not send is null.
 */
export type CompactionUpdate =
  | { kind: 'started'; tokensUsed: number | null; contextWindow: number | null }
  | { kind: 'completed'; tokensBefore: number | null; tokensAfter: number | null; elapsedMs: number | null }
  | { kind: 'failed'; error: string; retry: boolean }
  | { kind: 'cancelled' };

/** The agent's own first words of a failure it retries at the next step. */
export const COMPACT_RETRY_TEXT = "it'll retry on the next turn";

function count(value: unknown): number | null {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0 ? value : null;
}

export function readCompaction(update: Json): CompactionUpdate | null {
  switch (update.sessionUpdate) {
    case 'auto_compact_started':
      return { kind: 'started', tokensUsed: count(update.tokens_used), contextWindow: count(update.context_window) };
    case 'auto_compact_completed':
      return {
        kind: 'completed',
        tokensBefore: count(update.tokens_before),
        tokensAfter: count(update.tokens_after),
        elapsedMs: count(update.elapsed_ms),
      };
    case 'auto_compact_failed': {
      const error = str(update.error) ?? '';
      return { kind: 'failed', error, retry: error.startsWith(COMPACT_RETRY_TEXT) };
    }
    case 'auto_compact_cancelled':
      return { kind: 'cancelled' };
    default:
      return null;
  }
}

/**
 * A batch of hooks the agent waits on (xAI only): `started` is `hook_run_started` (live only, sent only when a hook
 * will run: `S/session/acp_session_impl/hook_dispatch.rs` `announce_hook_run`), `ended` its `hook_execution` (the same
 * event, tool and prompt id). Null for any other update.
 */
export type HookRunLine =
  | { kind: 'started'; event: string; tool: string | null; promptId: string | null; count: number }
  | { kind: 'ended'; event: string; tool: string | null; promptId: string | null };

export function readHookRun(update: Json): HookRunLine | null {
  const kind =
    update.sessionUpdate === 'hook_run_started'
      ? 'started'
      : update.sessionUpdate === 'hook_execution'
        ? 'ended'
        : null;
  const event = str(update.event_name);
  if (kind === null || event === null) return null;
  const batch = { event, tool: str(update.tool_name), promptId: str(update.prompt_id) };
  if (kind === 'ended') return { kind, ...batch };
  return { kind, ...batch, count: count(update.count) ?? 0 };
}

export interface WorkflowUpdate {
  runId: string;
  revision: number;
  name: string;
  /** The agent's word: `active`, a paused one, `blocked`, `budget_limited` or a terminal one. */
  status: string;
  /** `workflow_outcome_ignored` after a stop: the run's agents drained. */
  lastEvent: string | null;
}

/** A `workflow_updated` line (xAI only) with a run id, a revision and a status; any other shape is null. */
export function readWorkflowUpdate(update: Json): WorkflowUpdate | null {
  if (update.sessionUpdate !== 'workflow_updated') return null;
  const runId = str(update.run_id);
  const revision = count(update.revision);
  const status = str(update.status);
  if (runId === null || runId === '' || revision === null || status === null || status === '') return null;
  return { runId, revision, name: str(update.name) ?? '', status, lastEvent: str(update.last_event) };
}

/** A line of a tool call: its `_meta['x.ai/tool']` name, the pause or stop its `rawInput.source` asks, its end. */
export interface WorkflowCallLine {
  toolCallId: string;
  tool: string | null;
  /** `key` is the run id or name, as the agent's `find_run_id` takes it. */
  control: { verb: 'stop' | 'pause'; key: string } | null;
  ended: 'completed' | 'failed' | null;
}

export function readWorkflowCall(update: Json): WorkflowCallLine | null {
  if (update.sessionUpdate !== 'tool_call' && update.sessionUpdate !== 'tool_call_update') return null;
  const toolCallId = str(update.toolCallId);
  if (toolCallId === null) return null;
  const tool = metaOf(update)['x.ai/tool'];
  const input = isRecord(update.rawInput) ? update.rawInput : null;
  const source = input !== null && isRecord(input.source) ? input.source : null;
  const verb = source?.type === 'stop' || source?.type === 'pause' ? source.type : null;
  const key = source === null ? null : str(source.run_id);
  return {
    toolCallId,
    tool: isRecord(tool) ? str(tool.name) : null,
    control: verb === null || key === null || key === '' ? null : { verb, key },
    ended: update.status === 'completed' || update.status === 'failed' ? update.status : null,
  };
}

/** The answer to `_x.ai/workflows/stop`; any other shape is null. */
export function readWorkflowStop(result: unknown): 'stopped' | 'rejected' | null {
  if (!isRecord(result)) return null;
  if (result.outcome === 'stopped') return 'stopped';
  if (result.outcome === 'rejected') return 'rejected';
  return null;
}

export type TaskKillOutcome = 'killed' | 'already_exited' | 'not_found';

/**
 * The `_x.ai/task/kill` reply, `{result: {taskId, outcome}}`: its outcome. Null for the agent's error reply
 * (`{result: null, error}`, a session it does not hold) and any other shape.
 */
export function readTaskKill(result: unknown): TaskKillOutcome | null {
  if (!isRecord(result) || !isRecord(result.result)) return null;
  const outcome = result.result.outcome;
  return outcome === 'killed' || outcome === 'already_exited' || outcome === 'not_found' ? outcome : null;
}

export type SubagentCancelOutcome = 'cancelled' | 'already_finished' | 'not_found';

/**
 * The `_x.ai/subagent/cancel` reply, `{result: {subagentId, cancelled, outcome: {kind}}}` (`CancelSubagentResponse`,
 * `S/extensions/task.rs`): the outcome's kind. `cancelled` is followed by the helper's `subagent_finished`; the other two
 * by no end line. Null for any other shape, an unknown kind included.
 */
export function readSubagentCancel(result: unknown): SubagentCancelOutcome | null {
  if (!isRecord(result) || !isRecord(result.result) || !isRecord(result.result.outcome)) return null;
  const kind = result.result.outcome.kind;
  return kind === 'cancelled' || kind === 'already_finished' || kind === 'not_found' ? kind : null;
}

export const SUBAGENT_MESSAGE_KINDS = [
  'accepted',
  'rejected',
  'not_active',
  'unsupported_content',
  'limit',
  'admission_uncertain',
  'not_accepted_before_deadline',
  'saturated',
  'channel_closed',
] as const;
export type SubagentMessageKind = (typeof SUBAGENT_MESSAGE_KINDS)[number];

/**
 * The `_x.ai/subagent/message` reply, `{result: {kind, …}}` (`SendSubagentMessageOutcome`,
 * `S/extensions/subagent_message.rs`): its kind. Null for any other shape, an unknown kind included.
 */
export function readSubagentMessage(result: unknown): SubagentMessageKind | null {
  if (!isRecord(result) || !isRecord(result.result)) return null;
  const kind = result.result.kind;
  for (const known of SUBAGENT_MESSAGE_KINDS) if (kind === known) return known;
  return null;
}

/**
 * The `_x.ai/scheduler/delete` reply, `{result: {taskId, deleted}}`: `deleted`, or `gone` when the agent held no such
 * schedule. Null for the agent's error reply (`{result: null, error}`) and any other shape.
 */
export function readScheduleDelete(result: unknown): 'deleted' | 'gone' | null {
  if (!isRecord(result) || !isRecord(result.result)) return null;
  const deleted = result.result.deleted;
  return deleted === true ? 'deleted' : deleted === false ? 'gone' : null;
}

/**
 * The `_x.ai/rewind/execute` reply, a bare object (`RewindResponse`, snake_case). `files`: the dry run's clean files
 * and conflicts; `reverted`: the files a commit put back (a file it could not write is not among them). The agent's
 * `error` text is kept only as a yes or no. Null for any other shape.
 */
export interface RewindResult {
  success: boolean;
  target: number;
  files: Array<{ path: string; state: RewindFileState }>;
  reverted: string[];
  error: boolean;
}

const CONFLICT_STATES: Readonly<Record<string, RewindFileState>> = {
  modified_externally: 'modified',
  deleted_externally: 'deleted',
  created_externally: 'created',
};

export function readRewindResult(result: unknown): RewindResult | null {
  if (!isRecord(result) || typeof result.success !== 'boolean') return null;
  const target = count(result.target_prompt_index);
  if (target === null) return null;
  const files: RewindResult['files'] = [];
  if (Array.isArray(result.clean_files)) {
    for (const path of result.clean_files) if (typeof path === 'string') files.push({ path, state: 'clean' });
  }
  if (Array.isArray(result.conflicts)) {
    for (const conflict of result.conflicts) {
      if (!isRecord(conflict) || typeof conflict.path !== 'string') continue;
      const state = CONFLICT_STATES[str(conflict.conflict_type) ?? ''] ?? 'modified';
      files.push({ path: conflict.path, state });
    }
  }
  const reverted = Array.isArray(result.reverted_files)
    ? result.reverted_files.filter((path): path is string => typeof path === 'string')
    : [];
  const error = typeof result.error === 'string' && result.error !== '';
  return { success: result.success, target, files, reverted, error };
}

/**
 * The `_x.ai/session/fork` reply, a bare camelCase `ForkSessionResponse`: the copy's session id. Null for any other
 * shape.
 */
export function readForkResult(result: unknown): { sessionId: string } | null {
  if (!isRecord(result)) return null;
  const sessionId = str(result.newSessionId);
  return sessionId === null || sessionId === '' ? null : { sessionId };
}

/**
 * The `_x.ai/recap` ack, `{result: {ok: true}}`, or `{result: {ok: true, disabled: true}}` when recaps are turned off in
 * the agent's settings. Null for any other shape.
 */
export function readRecapAck(result: unknown): { disabled: boolean } | null {
  if (!isRecord(result) || !isRecord(result.result) || result.result.ok !== true) return null;
  return { disabled: result.result.disabled === true };
}

/** The `_x.ai/interject` reply, `{result: {status: "queued"}}`: the agent took the message. Any other shape is not. */
export function readInterject(result: unknown): boolean {
  return isRecord(result) && isRecord(result.result) && result.result.status === 'queued';
}

/** A `subagent_spawned` line's child session id, or null. */
export function readSpawnedChild(update: Json): string | null {
  if (update.sessionUpdate !== 'subagent_spawned') return null;
  const child = str(update.child_session_id);
  return child === null || child === '' ? null : child;
}

/** A `subagent_progress` tick's numbers, by the helper's child session id. */
export interface SubagentProgress {
  childId: string;
  turns: number;
  toolCalls: number;
  tokens: number;
  /** Clamped to 0..100. */
  contextPct: number;
  errors: number;
  durationMs: number;
  /** `context_window_tokens`; null when the tick names none. */
  contextWindow: number | null;
}

/**
 * A `subagent_progress` line (xAI only, never stored: `S/agent/subagent/mod.rs` `spawn_progress_publisher`) with a
 * child session id and every count a whole number (the window may be missing); any other shape is null.
 */
export function readSubagentProgress(update: Json): SubagentProgress | null {
  if (update.sessionUpdate !== 'subagent_progress') return null;
  const childId = str(update.child_session_id);
  const turns = count(update.turn_count);
  const toolCalls = count(update.tool_call_count);
  const tokens = count(update.tokens_used);
  const pct = count(update.context_usage_pct);
  const errors = count(update.error_count);
  const durationMs = count(update.duration_ms);
  if (childId === null || childId === '' || turns === null || toolCalls === null || tokens === null) return null;
  if (pct === null || errors === null || durationMs === null) return null;
  const contextWindow = count(update.context_window_tokens);
  return { childId, turns, toolCalls, tokens, contextPct: Math.min(pct, 100), errors, durationMs, contextWindow };
}

/** A monitor's output line or batch: its task id, its description and its raw text (not the model's XML copy). */
export interface MonitorEventLine {
  taskId: string;
  description: string;
  text: string;
}

/**
 * A `monitor_event` line (`_x.ai/monitor_event`, never stored, no `_meta`: `S/tools/notification_bridge.rs`
 * `MonitorEvent`) with a task id and its texts; any other shape is null.
 */
export function readMonitorEvent(update: Json): MonitorEventLine | null {
  if (update.sessionUpdate !== 'monitor_event') return null;
  const taskId = str(update.task_id);
  const description = str(update.description);
  const text = str(update.event_text);
  if (taskId === null || taskId === '' || description === null || text === null) return null;
  return { taskId, description, text };
}

/** A `model_auto_switched` line: the agent's ids (`''` for none) and its words. */
export interface ModelSwitchLine {
  previous: string;
  next: string;
  reason: string;
}

/**
 * A `model_auto_switched` line (xAI only, never stored, no `_meta`: `S/agent/mvp_agent/agent_ops.rs`
 * `send_model_auto_switched`); null for any other update or a field that is not text.
 */
export function readModelSwitch(update: Json): ModelSwitchLine | null {
  if (update.sessionUpdate !== 'model_auto_switched') return null;
  const previous = str(update.previous_model_id);
  const next = str(update.new_model_id);
  const reason = str(update.reason);
  if (previous === null || next === null || reason === null) return null;
  return { previous, next, reason };
}

/**
 * The helpers the reply of `_x.ai/subagent/list_running` names (`{result: {subagents: [...]}}`): each one's
 * `childSessionId` and `subagentId`. Null for any other shape, an error reply included.
 */
export function readRunningSubagents(result: unknown): Set<string> | null {
  if (!isRecord(result) || !isRecord(result.result) || !Array.isArray(result.result.subagents)) return null;
  const ids = new Set<string>();
  for (const helper of result.result.subagents) {
    if (!isRecord(helper)) return null;
    for (const value of [helper.childSessionId, helper.subagentId]) {
      if (typeof value === 'string' && value !== '') ids.add(value);
    }
  }
  return ids;
}

/** A `session_info_update` title, or null. */
export function readTitle(update: Json): string | null {
  return update.sessionUpdate === 'session_info_update' ? str(update.title) : null;
}

/**
 * One `AvailableCommand` (`{name, description, input: {hint} | null, _meta?}`); null when the name or the words are
 * not strings. A skill row's `_meta` carries `scope`, `path`, `bareName`, `qualifiedName`; a workflow row's
 * `workflowSource` (`S/session/slash_commands.rs` `available_commands`).
 */
function readCommandRow(value: unknown): CommandRow | null {
  if (!isRecord(value) || typeof value.name !== 'string' || typeof value.description !== 'string') return null;
  const input = isRecord(value.input) ? value.input : {};
  const meta = metaOf(value);
  return {
    name: value.name,
    description: value.description,
    hint: str(input.hint),
    workflow: typeof meta.workflowSource === 'string',
    skill: typeof meta.bareName === 'string' || typeof meta.scope === 'string',
    qualifiedName: str(meta.qualifiedName),
  };
}

function readCommandRows(value: unknown): CommandRow[] | null {
  if (!Array.isArray(value)) return null;
  const rows: CommandRow[] = [];
  for (const item of value) {
    const row = readCommandRow(item);
    if (row !== null) rows.push(row);
  }
  return rows;
}

/** The rows of an `available_commands_update` push, or null for any other update. */
export function readCommandsUpdate(update: Json): CommandRow[] | null {
  if (update.sessionUpdate !== 'available_commands_update') return null;
  return readCommandRows(update.availableCommands);
}

/** The rows of the `_x.ai/commands/list` reply (`{commands, tools?}`, no envelope); null for any other shape. */
export function readCommandList(result: unknown): CommandRow[] | null {
  return isRecord(result) ? readCommandRows(result.commands) : null;
}

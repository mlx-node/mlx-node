/** The ACP shapes the adapter reads. Every inbound value goes through one of these readers; nothing is cast unchecked. */

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
  agentVersion: string | null;
  /** null when `modelState.availableModels` is not an array of `{modelId: string}`. */
  models: Array<{ modelId: string; contextTokens: number | null }> | null;
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
      });
    }
  }
  return {
    protocolVersion: r.protocolVersion,
    loadSession: caps.loadSession === true,
    embeddedContext: prompt.embeddedContext === true,
    pluginDirs: meta['x.ai/pluginDirs'] === true,
    agentVersion: str(meta.agentVersion),
    models,
  };
}

export interface SessionReplyFacts {
  sessionId: string | null;
  currentModelId: string | null;
  /** `_meta.currentWorkingDirectory` (new), else `_meta["x.ai/sessionDetail"].cwd` (load). */
  cwd: string | null;
}

export function readSessionReply(result: unknown): SessionReplyFacts {
  const r = isRecord(result) ? result : {};
  const meta = metaOf(r);
  const models = isRecord(r.models) ? r.models : {};
  const detail = isRecord(meta['x.ai/sessionDetail']) ? meta['x.ai/sessionDetail'] : {};
  return {
    sessionId: str(r.sessionId) ?? str(meta.sessionId),
    currentModelId: str(models.currentModelId),
    cwd: str(meta.currentWorkingDirectory) ?? str(detail.cwd),
  };
}

export interface SessionInfoFacts {
  agentName: string | null;
  model: string | null;
  apiBackend: string | null;
  toolDefinitionsCount: number | null;
  systemPromptTokens: number | null;
  toolDefinitionsTokens: number | null;
  usageCategories: Array<{ label: string; detail: string }>;
}

/** `_x.ai/session/info` nests its payload once more: `reply.result.result`. Any other shape is null. */
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
  return {
    agentName: str(r.agentName),
    model: str(r.model),
    apiBackend: str(r.apiBackend),
    toolDefinitionsCount: count(ctx.toolDefinitionsCount),
    systemPromptTokens: count(ctx.systemPromptTokens),
    toolDefinitionsTokens: count(ctx.toolDefinitionsTokens),
    usageCategories: categories,
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
  const ask = metaOf(params).hookAsk;
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
  };
}

/** The three methods that carry session updates. */
export const UPDATE_METHODS = ['session/update', '_x.ai/session_notification', '_x.ai/session/update'] as const;

export interface SessionUpdateMessage {
  sessionId: string;
  /** True for the xAI methods (`_x.ai/session_notification`, `_x.ai/session/update`). */
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

export interface PromptResultFacts {
  stopReason: string | null;
  cancellationCategory: string | null;
}

export function readPromptResult(result: unknown): PromptResultFacts {
  const r = isRecord(result) ? result : {};
  const meta = metaOf(r);
  return { stopReason: str(r.stopReason), cancellationCategory: str(meta.cancellationCategory) };
}

/** A `session_info_update` title, or null. */
export function readTitle(update: Json): string | null {
  return update.sessionUpdate === 'session_info_update' ? str(update.title) : null;
}

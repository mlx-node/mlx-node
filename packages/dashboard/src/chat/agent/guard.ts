/**
 * The start-up self-check of an agent process and the post-create guard of every new or loaded session. A changed
 * binary, an unparsed profile or an unknown tool name silently gives Chat the full default agent with a shell; these
 * checks are what catch it, and every failure fails closed.
 */

import type { InitializeFacts, SessionInfoFacts, SessionReplyFacts } from '../acp/messages.js';
import type { ChatMode } from '../types.js';
import { CHAT_AGENT_NAME, CHAT_MCP_SERVER_COUNT, CHAT_SKILL_COUNT, CHAT_TOOL_COUNT } from './profiles.js';

export type SelfCheck =
  | {
      ok: true;
      agentVersion: string;
      models: Array<{ modelId: string; contextTokens: number | null }>;
      embeddedContext: boolean;
    }
  /** Rows 1–4 block the mode as `agent-incompatible`; row 5 (no models) is resolved by the host. */
  | { ok: false; row: 1 | 2 | 3 | 4 | 5 };

export function selfCheck(facts: InitializeFacts): SelfCheck {
  if (facts.protocolVersion !== 1) return { ok: false, row: 1 };
  if (!facts.loadSession) return { ok: false, row: 2 };
  if (!facts.pluginDirs) return { ok: false, row: 3 };
  if (facts.agentVersion === null) return { ok: false, row: 4 };
  if (facts.models === null || facts.models.length === 0) return { ok: false, row: 5 };
  return { ok: true, agentVersion: facts.agentVersion, models: facts.models, embeddedContext: facts.embeddedContext };
}

/** The leading decimal integer of a usage detail (`"1 skill"` → 1), or null. */
export function leadingInteger(text: string): number | null {
  let i = 0;
  let value = 0;
  while (i < text.length) {
    const c = text.charCodeAt(i);
    if (c < 48 || c > 57) break;
    value = value * 10 + (c - 48);
    if (!Number.isSafeInteger(value)) return null;
    i++;
  }
  return i === 0 ? null : value;
}

/** The count of a usage category; a missing category counts 0, a detail without a leading number is null. */
export function categoryCount(info: SessionInfoFacts, label: string): number | null {
  const entry = info.usageCategories.find((c) => c.label === label);
  return entry === undefined ? 0 : leadingInteger(entry.detail);
}

export interface GuardInput {
  mode: ChatMode;
  reply: SessionReplyFacts;
  /** null when `_x.ai/session/info` failed or had another shape. */
  info: SessionInfoFacts | null;
  /** `_x.ai/mcp/server_status {name: "mlx_web", status: "ready"}` was seen for this session (Chat only). */
  mcpReady: boolean;
  requestedModelId: string;
  /** realpath of the requested cwd. */
  cwd: string;
}

export type GuardRow = 1 | 2 | 3 | 4 | 5 | 6 | 7;

export type GuardResult =
  | { ok: true; agentName: string; toolCount: number | null }
  | { ok: false; row: GuardRow; reason: 'profile-mismatch' | 'model-unavailable' | 'cwd-mismatch' };

export function checkSession(input: GuardInput): GuardResult {
  const { info, mode } = input;
  const fail = (row: GuardRow): GuardResult => ({
    ok: false,
    row,
    reason: row === 5 ? 'model-unavailable' : row === 6 ? 'cwd-mismatch' : 'profile-mismatch',
  });
  if (info === null || info.agentName === null || info.agentName === '') return fail(1);
  if (mode === 'chat' ? info.agentName !== CHAT_AGENT_NAME : info.agentName === CHAT_AGENT_NAME) return fail(1);
  if (mode === 'chat') {
    if (info.toolDefinitionsCount !== CHAT_TOOL_COUNT) return fail(2);
    if (categoryCount(info, 'Skills') !== CHAT_SKILL_COUNT) return fail(3);
    if (!input.mcpReady || categoryCount(info, 'MCP servers') !== CHAT_MCP_SERVER_COUNT) return fail(4);
  } else if (info.toolDefinitionsCount === null) {
    return fail(2);
  }
  if (input.reply.currentModelId !== input.requestedModelId) return fail(5);
  if (input.reply.cwd !== input.cwd) return fail(6);
  if (info.apiBackend !== 'Messages') return fail(7);
  return { ok: true, agentName: info.agentName, toolCount: info.toolDefinitionsCount };
}

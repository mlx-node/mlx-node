/** Writes synthetic grok session folders (summary, events, subagent meta) under a temp agent home. */

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

export interface EventLine {
  /** Epoch ms of the line (`agentTimestampMs`). */
  at: number;
  /** The `_x.ai/session/update` method; otherwise an ACP `session/update` line. */
  xai?: boolean;
  update: Record<string, unknown>;
}

export const userLine = (at: number, text: string, promptIndex = 0): EventLine => ({
  at,
  update: { sessionUpdate: 'user_message_chunk', content: { type: 'text', text }, _meta: { promptIndex } },
});
export const answerLine = (at: number, text: string): EventLine => ({
  at,
  update: { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text } },
});
export const toolLine = (at: number, id: string, name: string, status: string): EventLine => ({
  at,
  update: {
    sessionUpdate: status === 'pending' ? 'tool_call' : 'tool_call_update',
    toolCallId: id,
    title: `${name} title`,
    status,
    _meta: { 'x.ai/tool': { name } },
  },
});
export const spawnedLine = (at: number, child: string, description: string): EventLine => ({
  at,
  xai: true,
  update: { sessionUpdate: 'subagent_spawned', child_session_id: child, subagent_type: 'explore', description },
});
export const finishedLine = (at: number, child: string, status = 'completed'): EventLine => ({
  at,
  xai: true,
  update: { sessionUpdate: 'subagent_finished', child_session_id: child, status, duration_ms: 1, tokens_used: 1 },
});
export const turnLine = (at: number, elapsedMs: number, tokens = 0): EventLine => ({
  at,
  xai: true,
  update: {
    sessionUpdate: 'turn_completed',
    prompt_id: `p${String(at)}`,
    stop_reason: 'end_turn',
    elapsed_ms: elapsedMs,
    ...(tokens > 0 ? { usage: { inputTokens: tokens, outputTokens: 0 } } : {}),
  },
});
export const modelLine = (at: number, modelId: string): EventLine => ({
  at,
  xai: true,
  update: { sessionUpdate: 'model_changed', model_id: modelId },
});

export function eventsText(sessionId: string, lines: readonly EventLine[]): string {
  return lines
    .map((line) =>
      JSON.stringify({
        timestamp: Math.floor(line.at / 1000),
        method: line.xai === true ? '_x.ai/session/update' : 'session/update',
        params: { sessionId, update: line.update, _meta: { agentTimestampMs: line.at } },
      }),
    )
    .map((l) => `${l}\n`)
    .join('');
}

export interface SessionFiles {
  id: string;
  cwd?: string;
  title?: string;
  /** null = no `updates.jsonl`. */
  lines: readonly EventLine[] | null;
  /** Writes `<parent>/subagents/<id>/meta.json`. */
  parent?: { id: string; cwd?: string; status: string; description: string };
}

/** The session folder it wrote. */
export function writeSession(home: string, files: SessionFiles): string {
  const cwd = files.cwd ?? '/Users/dev/timeline';
  const dir = join(home, 'sessions', encodeURIComponent(cwd), files.id);
  mkdirSync(dir, { recursive: true });
  writeFileSync(
    join(dir, 'summary.json'),
    JSON.stringify({
      info: { id: files.id, cwd },
      session_summary: files.title ?? files.id,
      created_at: '2026-10-10T14:41:17Z',
      updated_at: '2026-10-10T14:41:17Z',
      num_messages: 1,
      current_model_id: 'summary-model',
      ...(files.parent === undefined ? {} : { session_kind: 'subagent', parent_session_id: files.parent.id }),
    }),
  );
  if (files.lines !== null) writeFileSync(join(dir, 'updates.jsonl'), eventsText(files.id, files.lines));
  if (files.parent !== undefined) {
    const parentDir = join(home, 'sessions', encodeURIComponent(files.parent.cwd ?? cwd), files.parent.id);
    const metaDir = join(parentDir, 'subagents', files.id);
    mkdirSync(metaDir, { recursive: true });
    writeFileSync(
      join(metaDir, 'meta.json'),
      JSON.stringify({
        parent_session_id: files.parent.id,
        child_session_id: files.id,
        subagent_type: 'explore',
        description: files.parent.description,
        status: files.parent.status,
        started_at: '2026-10-10T14:49:25Z',
        effective_model_id: 'child-model',
      }),
    );
  }
  return dir;
}

/**
 * A workflow run's `workflows/<run id>/state.json`: the agent rewrites it on every update, stored or live (grok
 * `store.rs`, manifest version 4, the run under `state`). Only it holds the steps' `detail` and the run's history.
 * The agent writes it through a queue, so it can be older than the last stored `workflow_updated`, or newer.
 */

import { lstat, readFile } from 'node:fs/promises';
import { join } from 'node:path';

import { applyRunUpdate, rfc3339Ms } from './transcript.js';
import type { TranscriptEntry } from './types.js';

type Json = Record<string, unknown>;
type WorkflowEntry = Extract<TranscriptEntry, { kind: 'workflow' }>;
type RunHistory = NonNullable<WorkflowEntry['history']>;

/** grok `WORKFLOW_RUN_MANIFEST_VERSION`. */
export const RUN_STATE_VERSION = 4;
/** Larger files are not read; the agent itself restores none over 512 KiB. */
export const MAX_RUN_STATE_BYTES = 1024 * 1024;

export interface RunState {
  runId: string;
  revision: number;
  /** The state as the `workflow_updated` the agent builds of it (`build_workflow_updated`). */
  update: Json;
  /** Each step's `detail` by its title. */
  details: ReadonlyMap<string, string>;
  /** Oldest first, at most 64 lines (the agent drops older ones). */
  history: RunHistory;
}

function isRecord(value: unknown): value is Json {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isUsize(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0;
}

function str(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

/** `<session dir>/workflows/<run id>/state.json`; null for a run id that is not one folder name. */
export function runStatePath(sessionDir: string, runId: string): string | null {
  if (runId === '' || runId === '.' || runId === '..') return null;
  for (let i = 0; i < runId.length; i++) {
    const code = runId.charCodeAt(i);
    if (code === 0x2f || code === 0x5c || code === 0) return null;
  }
  return join(sessionDir, 'workflows', runId, 'state.json');
}

/** A `state.json` text → the run it holds; null for another version, a run with no id or revision, or no JSON. */
export function parseRunState(text: string): RunState | null {
  let doc: unknown;
  try {
    doc = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(doc) || doc.version !== RUN_STATE_VERSION || !isRecord(doc.state)) return null;
  const state = doc.state;
  const runId = str(state.run_id);
  if (runId === null || runId === '' || !isUsize(state.revision)) return null;
  const phases = Array.isArray(state.phases) ? state.phases.filter(isRecord) : [];
  const status = str(state.status) ?? '';
  const current = str(state.current_phase);
  const at = current === null ? -1 : phases.findIndex((phase) => phase.title === current);
  const details = new Map<string, string>();
  for (const phase of phases) {
    const title = str(phase.title);
    const detail = str(phase.detail);
    if (title !== null && detail !== null && detail !== '' && !details.has(title)) details.set(title, detail);
  }
  const lines = Array.isArray(state.history) ? state.history.filter(isRecord) : [];
  const history: RunHistory = [];
  for (const line of lines) {
    const event = str(line.event);
    if (event === null || event === '') continue;
    history.push({ event, detail: str(line.detail), ts: rfc3339Ms(line.at) ?? 0 });
  }
  const last = lines.at(-1);
  const update: Json = {
    run_id: runId,
    revision: state.revision,
    name: state.name,
    objective: state.objective,
    status,
    phases: phases.map((phase, i) => ({
      title: str(phase.title) ?? '',
      state: at === -1 || i > at ? 'pending' : i < at || status === 'complete' ? 'done' : 'active',
    })),
    current_phase: current,
    agent_budget: state.agent_budget,
    agents_used: state.agents_used,
    agent_usage_incomplete: state.agent_usage_incomplete,
    elapsed_ms: state.elapsed_ms_floor,
    agents: state.agents,
    last_event: last?.event,
    last_event_detail: last?.detail,
    last_event_timestamp: last?.at,
    pause_message: state.pause_message,
    result_summary: state.result_summary,
  };
  return { runId, revision: state.revision, update, details, history };
}

/**
 * A run's `state.json` in `sessionDir` read now, with its mtime; null when it is missing, not a plain file, over
 * {@link MAX_RUN_STATE_BYTES}, of another version, or of another run.
 */
export async function readRunState(
  sessionDir: string,
  runId: string,
): Promise<{ state: RunState; fileTs: number } | null> {
  const path = runStatePath(sessionDir, runId);
  if (path === null) return null;
  try {
    const st = await lstat(path);
    if (!st.isFile() || st.size > MAX_RUN_STATE_BYTES) return null;
    const state = parseRunState(await readFile(path, 'utf8'));
    return state === null || state.runId !== runId ? null : { state, fileTs: Math.floor(st.mtimeMs) };
  } catch {
    return null;
  }
}

/**
 * The run's entry with its `state.json` (written at `fileTs`): a higher revision there replaces the update's fields
 * (its run time is a floor, so the larger time stays); the steps get their `detail`, `phasesEntered` the steps the
 * history names, and the entry the history. A new entry; `entry` is not changed.
 */
export function mergeRunState(entry: WorkflowEntry, state: RunState, fileTs: number): WorkflowEntry {
  const out: WorkflowEntry = { ...entry };
  if (state.revision > entry.revision) {
    applyRunUpdate(out, state.update, fileTs);
    out.elapsedMs = Math.max(out.elapsedMs, entry.elapsedMs);
  }
  out.phases = out.phases.map((phase) => {
    const detail = state.details.get(phase.title);
    return detail === undefined ? phase : { ...phase, detail };
  });
  const entered = [...out.phasesEntered];
  for (const line of state.history) {
    if (line.event !== 'phase_entered' || line.detail === null || line.detail === '' || line.ts === 0) continue;
    if (!entered.some((p) => p.title === line.detail && p.ts === line.ts))
      entered.push({ title: line.detail, ts: line.ts });
  }
  out.phasesEntered = entered.sort((a, b) => a.ts - b.ts);
  out.history = state.history;
  return out;
}

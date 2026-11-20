/**
 * `ROOT/model-windows.json`: the context window the agent process last listed for each model, so the window stays known
 * while no agent runs (between turns after its idle stop, after an app restart). Model ids and token counts only.
 */

import { readFileSync } from 'node:fs';

import { writeFileAtomic } from './paths.js';

/** The file's windows when it is valid; anything else (missing, broken, wrong shape) reads as none. */
export function parseModelWindows(text: string): Map<string, number> {
  const out = new Map<string, number>();
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return out;
  }
  if (typeof raw !== 'object' || raw === null) return out;
  const record = raw as { version?: unknown; models?: unknown };
  if (record.version !== 1 || !Array.isArray(record.models)) return out;
  for (const item of record.models as unknown[]) {
    if (typeof item !== 'object' || item === null) continue;
    const { modelId, contextTokens } = item as { modelId?: unknown; contextTokens?: unknown };
    if (typeof modelId !== 'string' || modelId === '') continue;
    if (typeof contextTokens !== 'number' || !Number.isSafeInteger(contextTokens) || contextTokens <= 0) continue;
    out.set(modelId, contextTokens);
  }
  return out;
}

export function readModelWindowsSync(path: string): Map<string, number> {
  try {
    return parseModelWindows(readFileSync(path, 'utf8'));
  } catch {
    return new Map();
  }
}

/** Temp file and rename, mode 0600. */
export async function writeModelWindows(path: string, windows: ReadonlyMap<string, number>): Promise<void> {
  const models = [...windows].map(([modelId, contextTokens]) => ({ modelId, contextTokens }));
  await writeFileAtomic(path, `${JSON.stringify({ version: 1, models })}\n`);
}

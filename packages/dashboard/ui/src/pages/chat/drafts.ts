/**
 * Window memory of the Chat page: the draft per place and the last open conversation per workspace. Prompt text is
 * never written to storage; only the last used mode and the chosen model per mode are.
 */

import type { ChatMode } from '../../lib/chat-types';

/** The New screen of both modes shares one draft: it is one composer. */
export const NEW_PLACE = 'new';

const drafts = new Map<string, string>();
const lastPlace: Record<ChatMode, string | null> = { chat: null, work: null };

export function readDraft(place: string): string {
  return drafts.get(place) ?? '';
}

export function writeDraft(place: string, text: string): void {
  if (text === '') drafts.delete(place);
  else drafts.set(place, text);
}

export function lastPlaceOf(mode: ChatMode): string | null {
  return lastPlace[mode];
}

export function rememberPlace(mode: ChatMode, id: string): void {
  lastPlace[mode] = id;
}

/** A conversation that was deleted or is gone is nobody's place any more. */
export function forgetPlace(id: string): void {
  if (lastPlace.chat === id) lastPlace.chat = null;
  if (lastPlace.work === id) lastPlace.work = null;
  drafts.delete(id);
}

/** The folder the user chose for the next Work conversation, in this window. */
let workFolder: string | null = null;

export function chosenFolder(): string | null {
  return workFolder;
}

export function chooseFolder(path: string | null): void {
  workFolder = path;
}

/** The longest file name the agent can make: it keeps a folder's conversations under its encoded path. */
const AGENT_NAME_BYTES = 255;

/** The agent keeps this folder's conversations in a directory named by the encoded path, and that name is too long. */
export function folderTooLong(path: string): boolean {
  return encodeURIComponent(path).length > AGENT_NAME_BYTES;
}

/** The newest distinct folders of the Work list, at most `n`. */
export function recentFolders(rows: ReadonlyArray<{ cwd: string | null; updatedAt: number }>, n = 6): string[] {
  const out: string[] = [];
  for (const row of [...rows].sort((a, b) => b.updatedAt - a.updatedAt)) {
    if (row.cwd === null || row.cwd === '' || out.includes(row.cwd)) continue;
    out.push(row.cwd);
    if (out.length === n) break;
  }
  return out;
}

const MODE_KEY = 'mlx-node:chat-mode:v1';
const MODEL_KEY = 'mlx-node:chat-model:v1';

export function isChatMode(value: string | null | undefined): value is ChatMode {
  return value === 'chat' || value === 'work';
}

export function storedMode(): ChatMode {
  try {
    const value = globalThis.localStorage?.getItem(MODE_KEY);
    return isChatMode(value) ? value : 'chat';
  } catch {
    return 'chat';
  }
}

export function storeMode(mode: ChatMode): void {
  try {
    globalThis.localStorage?.setItem(MODE_KEY, mode);
  } catch {
    // The mode still applies to this window.
  }
}

export function storedModel(mode: ChatMode): string | null {
  try {
    const raw = globalThis.localStorage?.getItem(MODEL_KEY);
    if (raw === null || raw === undefined) return null;
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== 'object' || parsed === null) return null;
    const value = (parsed as Record<string, unknown>)[mode];
    return typeof value === 'string' ? value : null;
  } catch {
    return null;
  }
}

export function storeModel(mode: ChatMode, id: string): void {
  try {
    const raw = globalThis.localStorage?.getItem(MODEL_KEY);
    let current: Record<string, unknown> = {};
    if (raw !== null && raw !== undefined) {
      const parsed: unknown = JSON.parse(raw);
      if (typeof parsed === 'object' && parsed !== null) current = parsed as Record<string, unknown>;
    }
    globalThis.localStorage?.setItem(MODEL_KEY, JSON.stringify({ ...current, [mode]: id }));
  } catch {
    // The choice still applies to this window.
  }
}

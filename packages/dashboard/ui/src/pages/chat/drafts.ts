/**
 * Window memory of the Chat page: the draft, the attached files and the failed send per place and the last open
 * conversation per workspace. Prompt text and files are never written to storage; only the last used mode, the chosen model per mode
 * and the last model picked per base are.
 */

import { attachQueue, type DraftAttachment } from '../../lib/chat-attachments';
import type { ChatMode } from '../../lib/chat-types';
import type { StripNotice } from './blockers';

/** The New screen of both modes shares one draft: it is one composer. */
export const NEW_PLACE = 'new';

const drafts = new Map<string, string>();
const attachments = new Map<string, readonly DraftAttachment[]>();
/** How often `forgetPlace` removed each place: a read that started before a delete must not bring its files back. */
const forgets = new Map<string, number>();
const NO_ATTACHMENTS: readonly DraftAttachment[] = [];
const lastPlace: Record<ChatMode, string | null> = { chat: null, work: null };

/** A message whose send failed, kept for Retry until it is sent, a send from its place succeeds, or the place is gone. */
export interface FailedSend {
  text: string;
  files: readonly DraftAttachment[];
  /** What the failure shows over the composer; null when the thread's own end strip shows it. */
  strip: StripNotice | null;
  /** It was put back into the composer, so the composer holds it too. */
  restored: boolean;
}

const failures = new Map<string, FailedSend>();
const listeners = new Set<() => void>();

/** Called after every change of the attached files or the failed sends, whichever page made it. */
export function subscribeDrafts(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function changed(): void {
  for (const listener of listeners) listener();
}

/** One queue for every read of files into this store, whichever Chat mount started it. */
export const attachDrafts = attachQueue();

export function readDraft(place: string): string {
  return drafts.get(place) ?? '';
}

export function writeDraft(place: string, text: string): void {
  if (text === '') drafts.delete(place);
  else drafts.set(place, text);
}

export function readAttachments(place: string): readonly DraftAttachment[] {
  return attachments.get(place) ?? NO_ATTACHMENTS;
}

/** Taken when a read or a send starts and given to its `writeAttachments`. */
export function placeEpoch(place: string): number {
  return forgets.get(place) ?? 0;
}

/** Dropped when `forgetPlace` removed the place after `epoch` was taken. */
export function writeAttachments(
  place: string,
  items: readonly DraftAttachment[],
  epoch: number = placeEpoch(place),
): void {
  if (epoch !== placeEpoch(place)) return;
  if (items.length === 0) attachments.delete(place);
  else attachments.set(place, items);
  changed();
}

export function readFailed(place: string): FailedSend | null {
  return failures.get(place) ?? null;
}

/** Dropped when `forgetPlace` removed the place after `epoch` was taken. */
export function writeFailed(place: string, record: FailedSend | null, epoch: number = placeEpoch(place)): void {
  if (epoch !== placeEpoch(place)) return;
  if (record === null) failures.delete(place);
  else failures.set(place, record);
  changed();
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
  attachments.delete(id);
  failures.delete(id);
  forgets.set(id, placeEpoch(id) + 1);
  changed();
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
/** The last model picked per base, in both modes: `{ [base in lower case]: id }`. */
const VARIANT_KEY = 'mlx-node:chat-variant:v1';

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

/** The last model picked per base (keys in lower case); empty when storage is unreadable. */
export function pickedVariants(): Record<string, string> {
  try {
    const raw = globalThis.localStorage?.getItem(VARIANT_KEY);
    if (raw === null || raw === undefined) return {};
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return {};
    const out: Record<string, string> = {};
    for (const [base, id] of Object.entries(parsed)) if (typeof id === 'string') out[base] = id;
    return out;
  } catch {
    return {};
  }
}

export function storeVariant(base: string, id: string): void {
  try {
    globalThis.localStorage?.setItem(VARIANT_KEY, JSON.stringify({ ...pickedVariants(), [base.toLowerCase()]: id }));
  } catch {
    // The pick still applies to this window.
  }
}

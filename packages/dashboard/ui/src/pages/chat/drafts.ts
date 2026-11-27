/**
 * Window memory of the Chat page: the draft, the attached files, the "Plan first" value and the failed send per place and the last open
 * conversation per workspace. Prompt text and files are never written to storage; only the last used mode, the chosen model and
 * thinking level per mode and the last model picked per base are.
 */

import { isThinkingLevel } from '../../../../src/chat/thinking-levels.js';
import { attachQueue, type DraftAttachment, type PdfDrawing } from '../../lib/chat-attachments';
import { CHAT } from '../../lib/chat-strings';
import type { ChatMode, ThinkingLevel } from '../../lib/chat-types';
import type { StripNotice } from './blockers';

/** The New screen of both modes shares one draft: it is one composer. */
export const NEW_PLACE = 'new';

const drafts = new Map<string, string>();
const attachments = new Map<string, readonly DraftAttachment[]>();

/**
 * A PDF whose read runs: its chip shows from the start. `total` is null until the file is open; `bytes` are the JPEG
 * bytes of the pages drawn so far; `thumbnail` is page 1 once drawn. `stop` cancels the read when the chip is removed.
 */
export interface PendingPdf extends Omit<PdfDrawing, 'total'> {
  id: string;
  name: string;
  total: number | null;
  bytes: number;
  thumbnail: string | null;
  stop: AbortController;
}

const pending = new Map<string, readonly PendingPdf[]>();
const NO_PENDING: readonly PendingPdf[] = [];
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

/**
 * Work: the "Plan first" value the user set for a place, kept beside its draft. `seen`: the key of the conversation's
 * last mode line when it was set (null: none yet); a newer mode line takes over from it.
 */
export interface PlanArm {
  next: boolean;
  seen: string | null;
}

const plans = new Map<string, PlanArm>();
/** Places whose "no plan mode" notice already turned the chip off; the notice goes at the next send. */
const planNoticed = new Set<string>();

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

export function readPending(place: string): readonly PendingPdf[] {
  return pending.get(place) ?? NO_PENDING;
}

/** Adds, replaces (same id) or with `null` removes one pending PDF; dropped when the place was forgotten since `epoch`. */
export function writePending(
  place: string,
  id: string,
  item: PendingPdf | null,
  epoch: number = placeEpoch(place),
): void {
  if (epoch !== placeEpoch(place)) return;
  const items = readPending(place);
  const at = items.findIndex((entry) => entry.id === id);
  if (item === null && at === -1) return;
  const next = item === null ? items.filter((entry) => entry.id !== id) : [...items];
  if (item !== null) {
    if (at === -1) next.push(item);
    else next[at] = item;
  }
  if (next.length === 0) pending.delete(place);
  else pending.set(place, next);
  changed();
}

/** The pending PDFs of a place whose draft is not among its attached files yet: a send now would go without them. */
export function drawingCount(place: string): number {
  const held = readAttachments(place);
  let count = 0;
  for (const entry of readPending(place)) if (!held.some((item) => item.id === entry.id)) count += 1;
  return count;
}

/** Cancels the read of a pending PDF and drops its chip; false when no read of that id runs. */
export function stopPending(place: string, id: string): boolean {
  const item = readPending(place).find((entry) => entry.id === id);
  if (item === undefined) return false;
  item.stop.abort();
  writePending(place, id, null);
  return true;
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

export function readPlan(place: string): PlanArm | null {
  return plans.get(place) ?? null;
}

export function writePlan(place: string, value: PlanArm | null): void {
  if (value === null) plans.delete(place);
  else plans.set(place, value);
  changed();
}

/** Whether the place's "no plan mode" notice still has to turn the chip off: true once per notice. */
export function takePlanNotice(place: string, shown: boolean): boolean {
  if (!shown) {
    planNoticed.delete(place);
    return false;
  }
  if (planNoticed.has(place)) return false;
  planNoticed.add(place);
  return true;
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
  for (const item of readPending(id)) item.stop.abort();
  pending.delete(id);
  failures.delete(id);
  plans.delete(id);
  planNoticed.delete(id);
  forgets.set(id, placeEpoch(id) + 1);
  changed();
}

/** Where a new Work conversation starts: a folder, or none (the app makes it a new empty folder). */
export type FolderPick = { kind: 'folder'; path: string } | { kind: 'none' };

/** The place the user chose for the next Work conversation, in this window. */
let workPlace: FolderPick | null = null;

/** Null: no pick in this window. */
export function chosenFolder(): FolderPick | null {
  return workPlace;
}

export function chooseFolder(pick: FolderPick | null): void {
  workPlace = pick;
}

/** With no pick in this window: the place of the newest Work conversation, else none. */
export function defaultPick(
  rows: ReadonlyArray<{ cwd: string | null; updatedAt: number; noFolder?: boolean }>,
): FolderPick {
  let newest: (typeof rows)[number] | null = null;
  for (const row of rows) if (newest === null || row.updatedAt > newest.updatedAt) newest = row;
  if (newest === null || newest.noFolder === true || newest.cwd === null || newest.cwd === '') return { kind: 'none' };
  return { kind: 'folder', path: newest.cwd };
}

/** The place of the next Work conversation: a pick, or none yet (`loading`: the Work list is on its way; `unset`: it failed). */
export type NewPlace = FolderPick | { kind: 'loading' } | { kind: 'unset' };

type ListedRows = ReadonlyArray<{ cwd: string | null; updatedAt: number; noFolder?: boolean }>;

/**
 * The window's pick, else the place of the newest Work conversation once the Work list is in. Before that there is no
 * place: a send would start without a folder the person did not pick. A list that failed gives no default.
 */
export function newPlace(choice: FolderPick | null, list: { rows: ListedRows | undefined; failed: boolean }): NewPlace {
  if (choice !== null) return choice;
  if (list.rows !== undefined) return defaultPick(list.rows);
  return list.failed ? { kind: 'unset' } : { kind: 'loading' };
}

/** Why Send and a move up wait for the place; undefined once there is one. */
export function placeBlock(place: NewPlace): string | undefined {
  if (place.kind === 'loading') return CHAT.folderLoading;
  if (place.kind === 'unset') return CHAT.folderUnsetBlock;
  return undefined;
}

/** The folder of a place; null for none, and while there is no place yet. */
export function pickPath(pick: NewPlace): string | null {
  return pick.kind === 'folder' ? pick.path : null;
}

/** The folder a mode draft is for: a path, null for none, undefined while there is no place yet. */
function draftFolder(place: NewPlace): string | null | undefined {
  if (place.kind === 'folder') return place.path;
  return place.kind === 'none' ? null : undefined;
}

/** The New Work screen's mode draft: the host keeps a mode picked there for this id and folder only. */
export interface ModeDraft {
  id: string;
  /** Undefined: made before the place was known. */
  folder: string | null | undefined;
  visit: string;
  mode: ChatMode;
}

/**
 * The mode draft for the screen now (the same object when it holds). A new id when the visit, the mode or a known
 * folder changes, so a mode picked for one folder never reaches another; a draft made before the place was known
 * takes the first place with its id, so the Work list coming in loses nothing.
 */
export function nextModeDraft(
  draft: ModeDraft,
  now: { place: NewPlace; visit: string; mode: ChatMode },
  newId: () => string,
): ModeDraft {
  const folder = draftFolder(now.place);
  const moved = draft.folder !== undefined && folder !== undefined && draft.folder !== folder;
  if (draft.visit !== now.visit || draft.mode !== now.mode || moved)
    return { id: newId(), folder, visit: now.visit, mode: now.mode };
  if (draft.folder === undefined && folder !== undefined) return { ...draft, folder };
  return draft;
}

/** A fresh mode draft for the screen now. */
export function firstModeDraft(now: { place: NewPlace; visit: string; mode: ChatMode }, id: string): ModeDraft {
  return { id, folder: draftFolder(now.place), visit: now.visit, mode: now.mode };
}

/** The longest file name the agent can make: it keeps a folder's conversations under its encoded path. */
const AGENT_NAME_BYTES = 255;

/** The agent keeps this folder's conversations in a directory named by the encoded path, and that name is too long. */
export function folderTooLong(path: string): boolean {
  return encodeURIComponent(path).length > AGENT_NAME_BYTES;
}

/** The newest distinct folders of the Work list, at most `n`; the folders the app made for No folder are not offered. */
export function recentFolders(
  rows: ReadonlyArray<{ cwd: string | null; updatedAt: number; noFolder?: boolean }>,
  n = 6,
): string[] {
  const out: string[] = [];
  for (const row of [...rows].sort((a, b) => b.updatedAt - a.updatedAt)) {
    if (row.noFolder === true || row.cwd === null || row.cwd === '' || out.includes(row.cwd)) continue;
    out.push(row.cwd);
    if (out.length === n) break;
  }
  return out;
}

const MODE_KEY = 'mlx-node:chat-mode:v1';
const MODEL_KEY = 'mlx-node:chat-model:v1';
/** The thinking level a new conversation starts with, per mode, next to its model. */
const THINKING_KEY = 'mlx-node:chat-thinking:v1';
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

export function storedThinking(mode: ChatMode): ThinkingLevel | undefined {
  try {
    const raw = globalThis.localStorage?.getItem(THINKING_KEY);
    if (raw === null || raw === undefined) return undefined;
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed !== 'object' || parsed === null) return undefined;
    const value = (parsed as Record<string, unknown>)[mode];
    return isThinkingLevel(value) ? value : undefined;
  } catch {
    return undefined;
  }
}

export function storeThinking(mode: ChatMode, level: ThinkingLevel): void {
  try {
    const raw = globalThis.localStorage?.getItem(THINKING_KEY);
    let current: Record<string, unknown> = {};
    if (raw !== null && raw !== undefined) {
      const parsed: unknown = JSON.parse(raw);
      if (typeof parsed === 'object' && parsed !== null) current = parsed as Record<string, unknown>;
    }
    globalThis.localStorage?.setItem(THINKING_KEY, JSON.stringify({ ...current, [mode]: level }));
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

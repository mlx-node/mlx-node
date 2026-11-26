/**
 * `composer/<id>.json`: what one conversation's composer keeps across an app restart, its queued messages and its
 * thinking level. It holds text the user typed (and their files), so it is mode 0600, written whole through a temp
 * file, and removed with the conversation. A file that does not read back as rows is no rows.
 */

import { chmod, lstat, mkdir, readFile, unlink } from 'node:fs/promises';
import { join } from 'node:path';

import { writeFileAtomic } from './paths.js';
import { safeId } from './permission-log.js';
import type { QueueRow } from './queue.js';
import { checkAttachments, checkText } from './route-input.js';
import { isThinkingLevel } from './thinking-levels.js';
import { QUEUE_MAX, type ThinkingLevel } from './types.js';

/** Five rows of the largest message the prompt route takes, with room for JSON. */
const STORE_READ_MAX = 48 * 1024 * 1024;

export interface ComposerRecord {
  queue: QueueRow[];
  /** Absent: the model's default level. */
  thinking?: ThinkingLevel;
}

function rowOf(value: unknown): QueueRow | null {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return null;
  const r = value as Record<string, unknown>;
  if (typeof r.id !== 'string' || !safeId(r.id) || typeof r.text !== 'string') return null;
  if (typeof r.queuedAt !== 'number' || !Number.isFinite(r.queuedAt)) return null;
  if (!Array.isArray(r.attachments)) return null;
  if (r.plan !== undefined && typeof r.plan !== 'boolean') return null;
  try {
    const text = checkText(r.text);
    const attachments = checkAttachments(r.attachments, text) ?? [];
    return {
      id: r.id,
      text,
      attachments,
      ...(r.plan === undefined ? {} : { plan: r.plan }),
      queuedAt: r.queuedAt,
    };
  } catch {
    return null;
  }
}

/**
 * The record a file's text holds: rows that do not read are dropped, and at most {@link QUEUE_MAX} are kept; a level
 * that is not one is no level.
 */
export function readComposerRecord(text: string): ComposerRecord {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return { queue: [] };
  }
  if (typeof parsed !== 'object' || parsed === null || !Array.isArray((parsed as { queue?: unknown }).queue)) {
    return { queue: [] };
  }
  const queue: QueueRow[] = [];
  const seen = new Set<string>();
  for (const item of (parsed as { queue: unknown[] }).queue) {
    const row = rowOf(item);
    if (row === null || seen.has(row.id)) continue;
    seen.add(row.id);
    queue.push(row);
    if (queue.length === QUEUE_MAX) break;
  }
  const thinking = (parsed as { thinking?: unknown }).thinking;
  return isThinkingLevel(thinking) ? { queue, thinking } : { queue };
}

export class ComposerStore {
  readonly dir: string;
  private chain: Promise<void> = Promise.resolve();
  /** Conversations whose file was removed: a write queued after the delete must not make it again. */
  private readonly removed = new Set<string>();

  constructor(dir: string) {
    this.dir = dir;
  }

  private path(id: string): string {
    if (!safeId(id)) throw new Error('bad conversation id');
    return join(this.dir, `${id}.json`);
  }

  private serial<T>(run: () => Promise<T>): Promise<T> {
    const next = this.chain.then(run, run);
    this.chain = next.then(
      () => {},
      () => {},
    );
    return next;
  }

  /** Resolves once every read, write and remove asked for so far has ended. */
  settled(): Promise<void> {
    return this.chain;
  }

  /** The record of a conversation; empty when the file is missing, a link, too large or unreadable. */
  read(id: string): Promise<ComposerRecord> {
    return this.serial(() => this.readNow(id));
  }

  private async readNow(id: string): Promise<ComposerRecord> {
    try {
      const path = this.path(id);
      const info = await lstat(path);
      if (!info.isFile() || info.size > STORE_READ_MAX) return { queue: [] };
      return readComposerRecord(await readFile(path, 'utf8'));
    } catch {
      return { queue: [] };
    }
  }

  /** Writes the rows; the stored level stays. Rejects when it was not written. */
  write(id: string, record: Pick<ComposerRecord, 'queue'>): Promise<void> {
    return this.serial(async () => {
      const { thinking } = await this.readNow(id);
      await this.writeNow(id, thinking === undefined ? { queue: record.queue } : { queue: record.queue, thinking });
    });
  }

  /** Writes the level (null: the model's default); the stored rows stay. Rejects when it was not written. */
  setThinking(id: string, level: ThinkingLevel | null): Promise<void> {
    return this.serial(async () => {
      const { queue } = await this.readNow(id);
      await this.writeNow(id, level === null ? { queue } : { queue, thinking: level });
    });
  }

  /** A record with no rows and no level removes the file. */
  private async writeNow(id: string, record: ComposerRecord): Promise<void> {
    const path = this.path(id);
    if (this.removed.has(id)) throw new Error('conversation deleted');
    if (record.queue.length === 0 && record.thinking === undefined) {
      await unlinkQuiet(path);
      return;
    }
    await mkdir(this.dir, { recursive: true, mode: 0o700 });
    await chmod(this.dir, 0o700);
    await writeFileAtomic(path, `${JSON.stringify(record)}\n`);
  }

  /** Deletes the file; every later write of this conversation is refused for the rest of the run. */
  remove(id: string): Promise<void> {
    return this.serial(async () => {
      this.removed.add(id);
      await unlinkQuiet(this.path(id));
    });
  }
}

async function unlinkQuiet(path: string): Promise<void> {
  try {
    await unlink(path);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
  }
}

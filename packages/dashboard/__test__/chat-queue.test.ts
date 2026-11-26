/**
 * The host's queue of one conversation and its store file: the cap, the order, a row open for edit, the holds, the
 * page's view, and `composer/<id>.json` (mode 0600, rows that do not read dropped, gone with the last row and with the
 * conversation).
 */

import { existsSync, mkdtempSync, readFileSync, rmSync, statSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { ComposerStore, readComposerRecord } from '../src/chat/composer-store.js';
import { chatPaths } from '../src/chat/paths.js';
import { ConversationQueue, type QueueRow } from '../src/chat/queue.js';
import { QUEUE_MAX } from '../src/chat/types.js';
import { CHAT } from '../ui/src/lib/chat-strings.js';
import { QUEUE_MAX as PAGE_QUEUE_MAX } from '../ui/src/lib/chat-types.js';

function row(text: string, extra: Partial<QueueRow> = {}): Omit<QueueRow, 'id'> {
  return { text, attachments: [], queuedAt: 1_000, ...extra };
}

describe('ConversationQueue', () => {
  it('keeps rows in order and refuses the sixth', () => {
    const queue = new ConversationQueue();
    for (let i = 1; i <= QUEUE_MAX; i++) expect(queue.add(row(`m${i}`))).not.toBeNull();
    expect(queue.add(row('m6'))).toBeNull();
    expect(queue.view().items.map((item) => item.text)).toEqual(['m1', 'm2', 'm3', 'm4', 'm5']);
    expect(queue.next()?.text).toBe('m1');
    expect(QUEUE_MAX).toBe(5);
    expect(PAGE_QUEUE_MAX).toBe(QUEUE_MAX);
    expect(CHAT.queue.full.startsWith(String(QUEUE_MAX))).toBe(true);
  });

  it('sends nothing while a row is open for edit; a save or the end of the edit gives the list back', () => {
    const queue = new ConversationQueue();
    const a = queue.add(row('a'))!;
    const b = queue.add(row('b'))!;
    expect(queue.setEditing(b.id, true)).toBe(true);
    expect(queue.next()).toBeNull();
    expect(queue.view().items.map((item) => item.editing)).toEqual([false, true]);
    expect(queue.edit(b.id, 'b2')).toBe(true);
    expect(queue.next()?.id).toBe(a.id);
    expect(queue.view().items[1]).toMatchObject({ text: 'b2', editing: false });
    queue.setEditing(a.id, true);
    expect(queue.endEdits()).toBe(true);
    expect(queue.endEdits()).toBe(false);
    expect(queue.next()?.id).toBe(a.id);
    expect(queue.setEditing('gone', true)).toBe(false);
    expect(queue.edit('gone', 'x')).toBe(false);
  });

  it('holds: the first hold stays, an empty queue takes none, the last row out ends it, release gives the list back', () => {
    const queue = new ConversationQueue();
    expect(queue.holdFor({ why: 'stopped' })).toBe(false);
    expect(queue.held).toBeNull();
    const a = queue.add(row('a'))!;
    expect(queue.holdFor({ why: 'stopped' })).toBe(true);
    expect(queue.holdFor({ why: 'failed' })).toBe(false);
    expect(queue.held).toEqual({ why: 'stopped' });
    expect(queue.next()).toBeNull();
    expect(queue.release()).toBe(true);
    expect(queue.release()).toBe(false);
    expect(queue.next()?.id).toBe(a.id);
    queue.holdFor({ why: 'blocked', reason: 'That folder does not exist any more.', code: 'cwd-invalid' });
    expect(queue.view().held).toEqual({
      why: 'blocked',
      reason: 'That folder does not exist any more.',
      code: 'cwd-invalid',
    });
    expect(queue.remove(a.id)).toBe(true);
    expect(queue.held).toBeNull();
    expect(queue.remove(a.id)).toBe(false);
    queue.add(row('b'));
    queue.holdFor({ why: 'failed' });
    queue.clear();
    expect(queue.view()).toEqual({ items: [], held: null, count: 0 });
  });

  // Mutation: drop the row at `sending` → a Stop before its prompt went loses it and its files.
  it('a row whose turn starts leaves the view and the routes but stays stored until its prompt went', () => {
    const queue = new ConversationQueue();
    const file = { kind: 'file' as const, name: 'notes.md', text: '# notes\n' };
    const a = queue.add(row('a', { attachments: [file] }))!;
    const b = queue.add(row('b'))!;
    queue.sending(a.id, 'turn-1');
    expect(queue.view().items.map((item) => item.text)).toEqual(['b']);
    expect(queue.stored().map((r) => r.text)).toEqual(['a', 'b']);
    expect(queue.next()).toBeNull();
    expect(queue.row(a.id)).toBeUndefined();
    expect(queue.setEditing(a.id, true)).toBe(false);
    expect(queue.remove(a.id)).toBe(false);
    expect(queue.size).toBe(2);
    expect(queue.notSent('turn-other')).toBe(false);
    expect(queue.notSent('turn-1')).toBe(true);
    expect(queue.view().items.map((item) => item.text)).toEqual(['a', 'b']);
    expect(queue.next()?.attachments).toEqual([file]);

    queue.sending(a.id, 'turn-2');
    queue.clear();
    expect(queue.view().items).toEqual([]);
    expect(queue.stored().map((r) => r.text)).toEqual(['a']);
    expect(queue.sent('turn-1')).toBe(false);
    expect(queue.sent('turn-2')).toBe(true);
    expect(queue.size).toBe(0);
    expect(queue.stored()).toEqual([]);
    expect(queue.remove(b.id)).toBe(false);
  });

  // Mutation: return null from `next()` whenever the list is held → the row typed during an op waits for Send next.
  it('an unheld row goes though older rows are held; a hold made after it holds it too; the store never keeps it', () => {
    const queue = new ConversationQueue();
    const old = queue.add(row('old'))!;
    queue.holdFor({ why: 'stopped' });
    const typed = queue.add(row('typed during the op', { unheld: true }))!;
    const later = queue.add(row('later', { unheld: true }))!;
    expect(queue.next()?.id).toBe(typed.id);
    queue.setEditing(later.id, true);
    expect(queue.next()).toBeNull();
    queue.setEditing(later.id, false);
    expect(queue.remove(typed.id)).toBe(true);
    expect(queue.next()?.id).toBe(later.id);
    expect(queue.held).toEqual({ why: 'stopped' });
    expect(queue.stored().map((r) => Object.hasOwn(r, 'unheld'))).toEqual([false, false]);
    expect(queue.view().items.map((item) => item.text)).toEqual(['old', 'later']);
    expect(queue.holdFor({ why: 'failed' })).toBe(false);
    expect(queue.next()).toBeNull();
    expect(queue.release()).toBe(true);
    expect(queue.next()?.id).toBe(old.id);

    const free = new ConversationQueue();
    const first = free.add(row('first'))!;
    free.add(row('second', { unheld: true }));
    expect(free.next()?.id).toBe(first.id);
    expect(free.holdFor({ why: 'failed' })).toBe(true);
    expect(free.next()).toBeNull();
  });

  it('a restart hold comes only with rows; the view shows files by name and size, and the plan value', () => {
    expect(new ConversationQueue([], { why: 'restart' }).held).toBeNull();
    const queue = new ConversationQueue(
      [
        {
          id: 'r1',
          text: 'see the log',
          attachments: [{ kind: 'file', name: 'build.log', text: 'line one\n' }],
          plan: true,
          queuedAt: 5,
        },
      ],
      { why: 'restart' },
    );
    expect(queue.view()).toEqual({
      items: [
        {
          id: 'r1',
          text: 'see the log',
          files: [{ kind: 'file', name: 'build.log', bytes: 9 }],
          plan: true,
          queuedAt: 5,
          editing: false,
        },
      ],
      held: { why: 'restart' },
      count: 1,
    });
    expect(queue.stored()[0]!.attachments).toEqual([{ kind: 'file', name: 'build.log', text: 'line one\n' }]);
  });
});

describe('the composer store', () => {
  let root: string;
  let store: ComposerStore;
  const id = 'conv-1';

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'mlx-composer-'));
    store = new ComposerStore(chatPaths(join(root, 'chat')).composer);
  });

  afterEach(() => {
    rmSync(root, { recursive: true, force: true });
  });

  it('writes the rows whole, mode 0600, reads them back, and removes the file with the last row', async () => {
    const rows: QueueRow[] = [
      { id: 'a1', text: 'first', attachments: [], queuedAt: 1 },
      { id: 'b2', text: 'second', attachments: [{ kind: 'file', name: 'n.txt', text: 'x' }], plan: false, queuedAt: 2 },
    ];
    await store.write(id, { queue: rows });
    const path = join(store.dir, `${id}.json`);
    expect(statSync(path).mode & 0o777).toBe(0o600);
    expect(statSync(store.dir).mode & 0o777).toBe(0o700);
    expect(await store.read(id)).toEqual({ queue: rows });
    await store.write(id, { queue: [] });
    expect(existsSync(path)).toBe(false);
    expect(await store.read(id)).toEqual({ queue: [] });

    // The service's close waits on this, so a row queued just before a quit is on disk.
    void store.write(id, { queue: rows.slice(0, 1) });
    await store.settled();
    expect(JSON.parse(readFileSync(path, 'utf8'))).toEqual({ queue: rows.slice(0, 1) });
  });

  it('a removed conversation refuses later writes; a link or a bad id reads as no rows', async () => {
    await store.write(id, { queue: [{ id: 'a1', text: 'first', attachments: [], queuedAt: 1 }] });
    await store.remove(id);
    expect(existsSync(join(store.dir, `${id}.json`))).toBe(false);
    await expect(store.write(id, { queue: [{ id: 'a1', text: 'x', attachments: [], queuedAt: 1 }] })).rejects.toThrow();
    const target = join(root, 'elsewhere.json');
    writeFileSync(target, JSON.stringify({ queue: [{ id: 'z9', text: 'leak', attachments: [], queuedAt: 1 }] }));
    symlinkSync(target, join(store.dir, 'linked.json'));
    expect(await store.read('linked')).toEqual({ queue: [] });
    expect(await store.read('../escape')).toEqual({ queue: [] });
    expect(readFileSync(target, 'utf8')).toContain('leak');
  });

  it('drops rows that do not read: bad text, slash text, bad files, repeated ids; keeps at most five', () => {
    const good = (n: number): unknown => ({ id: `r${n}`, text: `row ${n}`, attachments: [], queuedAt: n });
    const record = readComposerRecord(
      JSON.stringify({
        queue: [
          good(1),
          { id: 'r2', text: '/yolo', attachments: [], queuedAt: 2 },
          { id: 'r3', text: '   ', attachments: [], queuedAt: 3 },
          { id: 'r4', text: 'files', attachments: [{ kind: 'file', name: 'a', text: '' }], queuedAt: 4 },
          { id: 'bad id!', text: 'x', attachments: [], queuedAt: 5 },
          { id: 'r6', text: 'x', attachments: [], queuedAt: 'now' },
          { id: 'r7', text: 'x', attachments: [], plan: 'yes', queuedAt: 7 },
          good(1),
          good(8),
          good(9),
          good(10),
          good(11),
          good(12),
        ],
      }),
    );
    expect(record.queue.map((r) => r.id)).toEqual(['r1', 'r8', 'r9', 'r10', 'r11']);
    expect(readComposerRecord('not json')).toEqual({ queue: [] });
    expect(readComposerRecord('{"queue":3}')).toEqual({ queue: [] });
  });

  it('the thinking level and the rows share the file: each write keeps the other, the file goes with neither', async () => {
    const rows: QueueRow[] = [{ id: 'a1', text: 'first', attachments: [], queuedAt: 1 }];
    const path = join(store.dir, `${id}.json`);
    await store.setThinking(id, 'low');
    expect(statSync(path).mode & 0o777).toBe(0o600);
    expect(await store.read(id)).toEqual({ queue: [], thinking: 'low' });
    await store.write(id, { queue: rows });
    expect(await store.read(id)).toEqual({ queue: rows, thinking: 'low' });
    await store.write(id, { queue: [] });
    expect(await store.read(id)).toEqual({ queue: [], thinking: 'low' });
    await store.write(id, { queue: rows });
    await store.setThinking(id, null);
    expect(await store.read(id)).toEqual({ queue: rows });
    await store.write(id, { queue: [] });
    expect(existsSync(path)).toBe(false);
    expect(readComposerRecord('{"queue":[],"thinking":"minimal"}')).toEqual({ queue: [] });
    await store.remove(id);
    await expect(store.setThinking(id, 'high')).rejects.toThrow();
  });
});

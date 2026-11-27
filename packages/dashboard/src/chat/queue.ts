/**
 * The queue of one conversation: messages typed while an answer runs, sent one at a time when the conversation is
 * free. The host owns it (the agent's own queue starts its next turn with no host step between). A hold stops it until
 * the user's Send next; an empty queue is never held.
 */

import { randomUUID } from 'node:crypto';

import { attachmentView, type AttachmentInput } from './attachment-rule.js';
import { QUEUE_MAX, type QueueHold, type QueueState } from './types.js';

export const QUEUE_FULL_MESSAGE = '5 messages wait. Remove one to queue more.';

/** A queued message as the host keeps it (and `composer/<id>.json` stores it). */
export interface QueueRow {
  id: string;
  text: string;
  attachments: AttachmentInput[];
  /** Work: what the chip said when the row was queued. */
  plan?: boolean;
  queuedAt: number;
  /**
   * Typed while a history op or a recap ran, where a free conversation would have sent it at once: it goes when the op
   * ends though older rows are held. A hold made after it holds it too. Memory only.
   */
  unheld?: true;
}

export class ConversationQueue {
  private rows: QueueRow[];
  private readonly editing = new Set<string>();
  /** Rows whose Send now waits for the agent's reply: sent by the list too, the agent would have them twice. */
  private readonly steering = new Set<string>();
  /**
   * The row whose turn starts: out of the view and of the routes, but kept (and stored) until its prompt went to the
   * agent, so a Stop or a failure before that gives it back with its files.
   */
  private starting: { id: string; turnId: string } | null = null;
  private hold: QueueHold | null;

  constructor(rows: readonly QueueRow[] = [], held: QueueHold | null = null) {
    this.rows = rows.slice(0, QUEUE_MAX).map((row) => ({ ...row }));
    this.hold = this.rows.length === 0 ? null : held;
  }

  get size(): number {
    return this.rows.length;
  }

  get held(): QueueHold | null {
    return this.hold;
  }

  /** {@link QUEUE_MAX} rows wait: no row can be added. */
  get full(): boolean {
    return this.rows.length >= QUEUE_MAX;
  }

  /** The new row, or null when the queue is {@link full}. */
  add(input: Omit<QueueRow, 'id'>): QueueRow | null {
    if (this.full) return null;
    const row: QueueRow = { id: randomUUID(), ...input, attachments: [...input.attachments] };
    this.rows.push(row);
    return row;
  }

  /** A row the user can act on: the one whose turn starts is not one. */
  row(id: string): QueueRow | undefined {
    if (id === this.starting?.id) return undefined;
    return this.rows.find((r) => r.id === id);
  }

  /** The row's turn was started: it waits out of the list for {@link sent} or {@link notSent} of that turn. */
  sending(id: string, turnId: string): void {
    if (this.row(id) !== undefined) this.starting = { id, turnId };
  }

  /** The turn's prompt went to the agent: its row leaves. False when no row waits for this turn. */
  sent(turnId: string): boolean {
    const starting = this.starting;
    if (starting === null || starting.turnId !== turnId) return false;
    this.starting = null;
    return this.remove(starting.id);
  }

  /** The turn ended before its prompt went to the agent: its row is back in its place. */
  notSent(turnId: string): boolean {
    if (this.starting === null || this.starting.turnId !== turnId) return false;
    this.starting = null;
    return true;
  }

  /** New text for a row; its edit ends. False when no row has this id. */
  edit(id: string, text: string): boolean {
    const row = this.row(id);
    if (row === undefined) return false;
    row.text = text;
    this.editing.delete(id);
    return true;
  }

  /** A page opened or closed the row for edit. False when no row has this id. */
  setEditing(id: string, on: boolean): boolean {
    if (this.row(id) === undefined) return false;
    if (on) this.editing.add(id);
    else this.editing.delete(id);
    return true;
  }

  /** A Send now of the row went out (`on`) or settled. False when no row has this id. */
  setSteering(id: string, on: boolean): boolean {
    if (this.row(id) === undefined) return false;
    if (on) this.steering.add(id);
    else this.steering.delete(id);
    return true;
  }

  isSteering(id: string): boolean {
    return this.steering.has(id);
  }

  /** Every edit ends (no page is left to finish it); true when one was open. */
  endEdits(): boolean {
    const any = this.editing.size > 0;
    this.editing.clear();
    return any;
  }

  remove(id: string): boolean {
    if (id === this.starting?.id) return false;
    const before = this.rows.length;
    this.rows = this.rows.filter((r) => r.id !== id);
    this.editing.delete(id);
    this.steering.delete(id);
    if (this.rows.length === 0) this.hold = null;
    return this.rows.length !== before;
  }

  /** Every row the user can act on leaves. */
  clear(): void {
    this.rows = this.rows.filter((r) => r.id === this.starting?.id);
    this.editing.clear();
    this.steering.clear();
    this.hold = null;
  }

  /**
   * The first hold stays: it is why the user's last say stopped the list. An empty queue takes none. Every row, the
   * unheld ones too, waits under it from now.
   */
  holdFor(hold: QueueHold): boolean {
    for (const row of this.rows) delete row.unheld;
    if (this.rows.length === 0 || this.hold !== null) return false;
    this.hold = hold;
    return true;
  }

  release(): boolean {
    const was = this.hold !== null;
    this.hold = null;
    return was;
  }

  /**
   * The row the queue sends next: the first one, or while it is held the first unheld one; none while it is empty, a
   * row is open for edit, a Send now is in flight or a row's turn starts.
   */
  next(): QueueRow | null {
    if (this.editing.size > 0 || this.steering.size > 0 || this.starting !== null) return null;
    if (this.hold !== null) return this.rows.find((row) => row.unheld === true) ?? null;
    return this.rows[0] ?? null;
  }

  /** The rows as the store keeps them. */
  stored(): QueueRow[] {
    return this.rows.map(({ unheld: _unheld, ...row }) => ({ ...row, attachments: [...row.attachments] }));
  }

  /**
   * The page's view, not yet redacted: text, files as names and sizes. The row whose turn starts is in the thread, and
   * still in `count`, as in {@link full}.
   */
  view(): QueueState {
    return {
      items: this.rows
        .filter((row) => row.id !== this.starting?.id)
        .map((row) => ({
          id: row.id,
          text: row.text,
          files: row.attachments.map(attachmentView),
          ...(row.plan === undefined ? {} : { plan: row.plan }),
          queuedAt: row.queuedAt,
          editing: this.editing.has(row.id),
        })),
      held: this.hold === null ? null : { ...this.hold },
      count: this.rows.length,
    };
  }
}

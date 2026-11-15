import type { PermissionMode } from './types.js';

type Raised = 'auto' | 'bypass';

interface Slot {
  draft: string;
  mode: Raised;
  /** The folder `checkWorkFolder` returned when the mode was picked. */
  cwd: string;
}

/** The mode picked on the New Work screen before a conversation exists. One at a time; never on disk. */
export class PermissionDraft {
  /** Every draft route call and every claim adds 1: a move up that waited is refused when it changed. */
  epoch = 0;
  private slot: Slot | null = null;
  /** Draft -> the conversation that took its pick; apart from the slot, so a later pick or `clear` keeps the forward. */
  private readonly taken = new Map<string, string>();

  /** 'ask' unless the slot is this draft's. */
  modeOf(draft: string): PermissionMode {
    const slot = this.slot;
    return slot !== null && slot.draft === draft ? slot.mode : 'ask';
  }

  /** The conversation that took this draft's pick, else null. */
  spent(draft: string): string | null {
    return this.taken.get(draft) ?? null;
  }

  /** Replaces whatever the slot held. */
  set(draft: string, mode: Raised, cwd: string): void {
    this.slot = { draft, mode, cwd };
  }

  /** 'ask' empties the slot. */
  lower(draft: string, target: 'ask' | 'auto'): void {
    const slot = this.slot;
    if (slot === null || slot.draft !== draft) return;
    if (target === 'ask') this.slot = null;
    else slot.mode = target;
  }

  /** The pick of this draft for the conversation just made in `cwd`; the slot empties. Null when there is none. */
  claim(draft: string | undefined, cwd: string, conversation: string): Raised | null {
    if (draft === undefined) return null;
    this.epoch += 1;
    const slot = this.slot;
    if (slot === null || slot.draft !== draft || this.taken.has(draft)) return null;
    this.slot = null;
    if (slot.cwd !== cwd) return null;
    this.taken.set(draft, conversation);
    return slot.mode;
  }

  /** The conversation was deleted: its draft goes. */
  forget(conversation: string): void {
    for (const [draft, owner] of this.taken) if (owner === conversation) this.taken.delete(draft);
  }

  /** True when something was shown. */
  clear(): boolean {
    const was = this.shown() !== null;
    this.slot = null;
    this.epoch += 1;
    return was;
  }

  /** Null when empty. */
  shown(): { draft: string; mode: Raised } | null {
    const slot = this.slot;
    return slot === null ? null : { draft: slot.draft, mode: slot.mode };
  }
}

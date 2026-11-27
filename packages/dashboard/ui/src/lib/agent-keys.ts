/**
 * The keys of agent rows, the Agent view and a run's head. Letters act only while a row (or the head) has focus,
 * never from a field, and never with ⌘, Ctrl or Option (those are the shell's).
 */

/** Where the key was pressed: a row of an agent list, the open Agent view, or a run's head. */
export type AgentKeyPlace = 'row' | 'view' | 'run-head';

export type AgentKeyAction =
  | 'up'
  | 'down'
  | 'first'
  | 'last'
  | 'open'
  | 'message'
  | 'pause'
  | 'stop'
  | 'close'
  | 'previous-agent'
  | 'next-agent';

/** What a key's target must have for the field check: an element, or anything else (no field). */
export interface KeyTarget {
  tagName?: string;
  isContentEditable?: boolean;
  getAttribute?: (name: string) => string | null;
  parentElement?: KeyTarget | null;
}

export interface AgentKey {
  key: string;
  metaKey?: boolean;
  ctrlKey?: boolean;
  altKey?: boolean;
  target: KeyTarget | EventTarget | null;
}

export interface AgentKeyOptions {
  /** The agent build can pause and resume: without it `P` does nothing (no dead keys). */
  pause: boolean;
}

/** Input types that take no typing: a key there is the row's. */
const BUTTON_INPUTS: ReadonlySet<string> = new Set(['checkbox', 'radio', 'button', 'submit', 'reset']);

/** The target, or an element it sits in, takes typing: a text field, a select, or editable text. */
export function inField(target: KeyTarget | EventTarget | null): boolean {
  let at = target as KeyTarget | null;
  while (at !== null && typeof at === 'object') {
    if (at.isContentEditable === true) return true;
    const tag = typeof at.tagName === 'string' ? at.tagName.toUpperCase() : '';
    if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
    if (tag === 'INPUT') {
      const type = (at.getAttribute?.('type') ?? 'text').toLowerCase();
      return !BUTTON_INPUTS.has(type);
    }
    at = at.parentElement ?? null;
  }
  return false;
}

/** Enter and Space on a button or a link are its own press. */
function pressable(target: KeyTarget | EventTarget | null): boolean {
  const at = target as KeyTarget | null;
  if (at === null || typeof at !== 'object') return false;
  const tag = typeof at.tagName === 'string' ? at.tagName.toUpperCase() : '';
  return tag === 'BUTTON' || tag === 'A' || at.getAttribute?.('role') === 'button';
}

/** The action a key takes at `place`; null when it takes none there. */
export function agentKeyOf(event: AgentKey, place: AgentKeyPlace, options: AgentKeyOptions): AgentKeyAction | null {
  if (event.metaKey === true || event.ctrlKey === true || event.altKey === true) return null;
  if (inField(event.target)) return null;
  const key = event.key;
  const letter = key.length === 1 ? key.toLowerCase() : '';
  if (place === 'view') {
    if (key === 'Escape') return 'close';
    if (key === '[') return 'previous-agent';
    if (key === ']') return 'next-agent';
    return null;
  }
  if (letter === 'p') return options.pause ? 'pause' : null;
  if (letter === 'x') return 'stop';
  if (place === 'run-head') return null;
  switch (key) {
    case 'ArrowUp':
      return 'up';
    case 'ArrowDown':
      return 'down';
    case 'Home':
      return 'first';
    case 'End':
      return 'last';
    case 'Enter':
    case ' ':
      return pressable(event.target) ? null : 'open';
    default:
      return letter === 'm' ? 'message' : null;
  }
}

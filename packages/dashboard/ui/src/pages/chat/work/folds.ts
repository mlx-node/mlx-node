/** The Work list's folder groups: which are folded (per viewer, kept), which show every row (this window only). */

const FOLDED_KEY = 'mlx-node:work-folded:v1';

/** The folded group keys; a failed or odd read means every group is open. */
export function readFolded(): Set<string> {
  try {
    const raw = globalThis.localStorage?.getItem(FOLDED_KEY);
    if (raw === null || raw === undefined) return new Set();
    const value: unknown = JSON.parse(raw);
    if (!Array.isArray(value)) return new Set();
    return new Set(value.filter((key): key is string => typeof key === 'string'));
  } catch {
    return new Set();
  }
}

export function writeFolded(folded: ReadonlySet<string>): void {
  try {
    globalThis.localStorage?.setItem(FOLDED_KEY, JSON.stringify([...folded]));
  } catch {
    // A per-viewer convenience: the folds hold until the list is drawn again.
  }
}

/** The folds after `key` opens or folds; keys of folders no longer in the list are dropped. */
export function nextFolded(
  folded: ReadonlySet<string>,
  key: string,
  open: boolean,
  keysNow: ReadonlySet<string>,
): Set<string> {
  const next = new Set([...folded].filter((entry) => keysNow.has(entry)));
  if (open) next.delete(key);
  else if (keysNow.has(key)) next.add(key);
  return next;
}

let allShown: ReadonlySet<string> = new Set();

/** Groups whose every row shows after "Show n more", in this window. */
export function shownAll(): ReadonlySet<string> {
  return allShown;
}

export function setShownAll(next: ReadonlySet<string>): void {
  allShown = next;
}

let lastOpen: string | null | undefined;
let revealFor: string | null = null;

/**
 * The open conversation whose folded group should open: set when the open id changes after the window's first look,
 * and kept until `revealed` (its row may be listed only later).
 */
export function toReveal(openId: string | null): string | null {
  if (lastOpen === undefined) {
    lastOpen = openId;
    return null;
  }
  if (openId !== lastOpen) {
    lastOpen = openId;
    revealFor = openId;
  }
  return revealFor;
}

export function revealed(): void {
  revealFor = null;
}

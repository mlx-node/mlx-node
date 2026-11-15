import { type RefObject, useEffect, useRef } from 'react';

export interface Hotkey {
  key: string;
  meta?: boolean;
  shift?: boolean;
  alt?: boolean;
  allowInInput?: boolean;
  run: (event: KeyboardEvent) => void;
}

function isTyping(target: EventTarget | null): boolean {
  if (typeof HTMLElement === 'undefined' || !(target instanceof HTMLElement)) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName;
  if (tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (tag !== 'INPUT') return false;
  const type = (target.getAttribute('type') ?? 'text').toLowerCase();
  return type !== 'checkbox' && type !== 'radio' && type !== 'button' && type !== 'submit' && type !== 'reset';
}

/** The binding `event` triggers, or undefined. `meta` means ⌘ or Ctrl. */
export function matchHotkey(hotkeys: readonly Hotkey[], event: KeyboardEvent): Hotkey | undefined {
  const meta = event.metaKey || event.ctrlKey;
  const key = event.key.toLowerCase();
  return hotkeys.find(
    (hotkey) =>
      hotkey.key.toLowerCase() === key &&
      (hotkey.meta === true) === meta &&
      (hotkey.shift === true) === event.shiftKey &&
      (hotkey.alt === true) === event.altKey &&
      (hotkey.allowInInput === true || !isTyping(event.target)),
  );
}

/** One keydown listener on window for the given bindings. */
export function useHotkeys(hotkeys: readonly Hotkey[]): void {
  const latest = useRef(hotkeys);
  useEffect(() => {
    latest.current = hotkeys;
  });
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent): void => {
      const hit = matchHotkey(latest.current, event);
      if (hit === undefined) return;
      event.preventDefault();
      hit.run(event);
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);
}

let pageSearch: RefObject<HTMLElement | null> | null = null;

/** A page names its search field; `/` focuses it while that page is mounted. */
export function usePageSearch(ref: RefObject<HTMLElement | null>): void {
  useEffect(() => {
    pageSearch = ref;
    return () => {
      if (pageSearch === ref) pageSearch = null;
    };
  }, [ref]);
}

/** Focus the mounted page's search field, if it named one. */
export function focusPageSearch(): void {
  pageSearch?.current?.focus();
}

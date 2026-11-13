
import { useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent, ReactNode } from "react";
import { AnimatePresence, animate, motion, useMotionValue, useReducedMotion } from "motion/react";
import type { Transition, ValueAnimationTransition, Variants } from "motion/react";
import { Command as CommandIcon, Search, X } from "lucide-react";
import { motionTokens } from "../lib/motion-tokens";
import { isWordChar, splitShortcut, splitWhitespace } from "../../../lib/text-scan";
import styles from "./command-palette.module.css";

export interface CommandItem { id: string; label: string; description?: string; group?: string; keywords?: string[]; icon?: ReactNode; shortcut?: string; }
export interface CommandPaletteProps { items: CommandItem[]; placeholder?: string; onSelect?: (item: CommandItem) => void; onClose?: () => void; label?: string; /** Focus the search field on mount, for a palette that opens on demand. */ autoFocus?: boolean; }

const enter: Transition = { duration: motionTokens.duration.standard, ease: [...motionTokens.ease.enter] };
const leave: Transition = { duration: motionTokens.duration.instant, ease: [...motionTokens.ease.standard] };
const GLIDE_ROWS = 6;
/** Rows fade out while the rest glide into their place; when the list snaps they leave at once so nothing overlaps the new rows. */
const rowExit: Variants = { exit: (glide: boolean) => ({ opacity: 0, transition: glide ? leave : { duration: 0 } }) };

type Range = [start: number, end: number];
interface Ranked { item: CommandItem; score: number; ranges: Range[]; order: number }

/** The first index where `word` starts a word of `text`, or -1. */
function wordStart(text: string, word: string) {
  for (let at = text.indexOf(word); at !== -1; at = text.indexOf(word, at + 1)) if (at === 0 || !isWordChar(text.charAt(at - 1))) return at;
  return -1;
}
/** Letters typed in order, starting at a word, within a short span: "dup pr" finds "Duplicate project", "shpr" finds "Share project". */
function subsequence(text: string, word: string): Range[] | null {
  if (word.length < 2) return null;
  for (let start = text.indexOf(word.charAt(0)); start !== -1; start = text.indexOf(word.charAt(0), start + 1)) {
    if (start > 0 && isWordChar(text.charAt(start - 1))) continue;
    const ranges: Range[] = [[start, start + 1]];
    let at = start;
    for (let i = 1; i < word.length && at !== -1; i++) {
      at = text.indexOf(word.charAt(i), at + 1);
      if (at === -1) break;
      const last = ranges[ranges.length - 1];
      if (last && last[1] === at) last[1] = at + 1; else ranges.push([at, at + 1]);
    }
    if (at !== -1 && at - start < word.length * 3) return ranges;
  }
  return null;
}
/** Scores one search word: the label counts most and a prefix ranks first. Descriptions and keywords match only at a word start, so "test" never finds "latest". */
function scoreWord(item: CommandItem, label: string, word: string): { score: number; ranges: Range[] } | null {
  const inLabel = wordStart(label, word);
  if (inLabel === 0) return { score: 100, ranges: [[0, word.length]] };
  if (inLabel > 0) return { score: 80, ranges: [[inLabel, inLabel + word.length]] };
  const inside = word.length >= 3 ? label.indexOf(word) : -1;
  if (inside > 0) return { score: 50, ranges: [[inside, inside + word.length]] };
  if (item.keywords?.some(keyword => wordStart(keyword.toLowerCase(), word) !== -1)) return { score: 40, ranges: [] };
  if (item.description && wordStart(item.description.toLowerCase(), word) !== -1) return { score: 25, ranges: [] };
  const loose = subsequence(label, word);
  return loose ? { score: 10, ranges: loose } : null;
}
/** Every word of the query has to match somewhere; the best matches come first, ties keep the order they were given in. */
function rank(items: CommandItem[], query: string): Ranked[] {
  const words = splitWhitespace(query.trim().toLowerCase());
  if (!words.length) return items.map((item, order) => ({ item, score: 0, ranges: [], order }));
  const ranked: Ranked[] = [];
  items.forEach((item, order) => {
    const label = item.label.toLowerCase();
    let score = 0;
    const ranges: Range[] = [];
    for (const word of words) {
      const match = scoreWord(item, label, word);
      if (!match) return;
      score += match.score;
      ranges.push(...match.ranges);
    }
    ranked.push({ item, score, ranges, order });
  });
  return ranked.sort((a, b) => b.score - a.score || a.order - b.order);
}
/** The label with its matched letters in medium weight. */
function Highlighted({ text, ranges }: { text: string; ranges: Range[] }) {
  if (!ranges.length) return text;
  const marked = new Array<boolean>(text.length).fill(false);
  for (const [start, end] of ranges) for (let i = start; i < end; i++) marked[i] = true;
  const parts: ReactNode[] = [];
  for (let i = 0; i < text.length;) {
    let end = i;
    while (end < text.length && marked[end] === marked[i]) end++;
    parts.push(marked[i] ? <mark key={i} className={styles.match}>{text.slice(i, end)}</mark> : text.slice(i, end));
    i = end;
  }
  return parts;
}
/** "⌘ K" or "Shift+D" becomes one keycap per key. */
const keysOf = (shortcut: string) => splitShortcut(shortcut);

export function CommandPalette({ items, placeholder = "Search commands", onSelect, onClose, label = "Command palette", autoFocus = false }: CommandPaletteProps) {
  const inputId = useId();
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [listHeight, setListHeight] = useState<number | "auto">("auto");
  const reduced = useReducedMotion();
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const lastPointer = useRef({ x: -1, y: -1 });
  const highlightShown = useRef(false);
  const highlightY = useMotionValue(0);
  const highlightHeight = useMotionValue(0);
  const highlightOpacity = useMotionValue(0);
  // Groups follow their best match, and the list reads top to bottom in that order, so arrow keys walk what you see.
  const { groupedItems, filtered, rangesById } = useMemo(() => {
    const grouped = new Map<string, Ranked[]>();
    for (const entry of rank(items, query)) {
      const group = entry.item.group ?? "Actions";
      grouped.set(group, [...(grouped.get(group) ?? []), entry]);
    }
    const flat: CommandItem[] = [];
    const groups = [...grouped.entries()].map(([group, entries]) => [group, entries.map(({ item }) => ({ item, index: flat.push(item) - 1 }))] as const);
    return { groupedItems: groups, filtered: flat, rangesById: new Map([...grouped.values()].flat().map(entry => [entry.item.id, entry.ranges])) };
  }, [items, query]);
  const [shift, setShift] = useState({ list: filtered, glide: true });
  if (shift.list !== filtered) {
    const before = new Map(shift.list.map((item, index) => [item.id, index]));
    setShift({ list: filtered, glide: filtered.every((item, index) => Math.abs((before.get(item.id) ?? index) - index) <= GLIDE_ROWS) });
  }
  const glide = shift.glide && !reduced;
  // The first result is active until the pointer or the arrow keys pick another, so Enter always has a target.
  const safeActiveIndex = filtered.length ? Math.min(activeIndex ?? 0, filtered.length - 1) : -1;
  const activeId = filtered[safeActiveIndex]?.id;

  useEffect(() => {
    if (autoFocus) inputRef.current?.focus({ preventScroll: true });
  }, [autoFocus]);

  useEffect(() => {
    const focusShortcut = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        inputRef.current?.focus();
      }
    };
    document.addEventListener("keydown", focusShortcut);
    return () => document.removeEventListener("keydown", focusShortcut);
  }, []);

  // The results frame follows the list height, so filtering closes or opens the gap instead of snapping.
  useLayoutEffect(() => {
    const list = listRef.current;
    if (!list) return;
    const observer = new ResizeObserver(([entry]) => setListHeight(Math.round(entry?.borderBoxSize?.[0]?.blockSize ?? list.offsetHeight)));
    observer.observe(list);
    return () => observer.disconnect();
  }, []);

  // One highlight follows the active result on a snappy spring, for the pointer and arrow keys alike, and fades in when it first appears.
  useLayoutEffect(() => {
    const node = activeId ? document.getElementById(`${inputId}-${activeId}`) : null;
    if (!node) {
      highlightShown.current = false;
      animate(highlightOpacity, 0, { duration: reduced ? 0 : .1 });
      return;
    }
    const move: ValueAnimationTransition<number> = !highlightShown.current || reduced ? { duration: 0 } : motionTokens.spring.snappy;
    // Rows sit inside positioned groups, so their offset is summed up to the list (layout transforms stay out of it).
    let top = 0;
    for (let element: HTMLElement | null = node; element && element !== listRef.current; element = element.offsetParent as HTMLElement | null) top += element.offsetTop + (element === node ? 0 : element.clientTop);
    animate(highlightY, top, move);
    animate(highlightHeight, node.offsetHeight, move);
    animate(highlightOpacity, 1, { duration: reduced ? 0 : .08 });
    highlightShown.current = true;
  }, [activeId, filtered, inputId, reduced, highlightY, highlightHeight, highlightOpacity]);

  useEffect(() => {
    if (!activeId) return;
    document.getElementById(`${inputId}-${activeId}`)?.scrollIntoView({ block: "nearest" });
  }, [activeId, inputId]);

  function choose(item: CommandItem) {
    onSelect?.(item);
    setQuery("");
    setActiveIndex(null);
    inputRef.current?.focus();
  }
  function handleKeyDown(event: ReactKeyboardEvent<HTMLInputElement>) {
    if (event.key === "ArrowDown") { event.preventDefault(); setActiveIndex(Math.min(safeActiveIndex + 1, Math.max(filtered.length - 1, 0))); }
    if (event.key === "ArrowUp") { event.preventDefault(); setActiveIndex(Math.max(safeActiveIndex - 1, 0)); }
    if (event.key === "Home") { event.preventDefault(); setActiveIndex(0); }
    if (event.key === "End") { event.preventDefault(); setActiveIndex(Math.max(filtered.length - 1, 0)); }
    // Enter runs the highlighted result; after typing, with nothing highlighted yet, it runs the top match.
    const target = filtered[safeActiveIndex];
    if (event.key === "Enter" && target) { event.preventDefault(); choose(target); }
    if (event.key === "Escape") {
      event.preventDefault();
      if (query) { setQuery(""); setActiveIndex(null); }
      else onClose?.();
    }
  }

  return <motion.div className={styles.palette} initial={reduced ? false : { opacity: 0, y: 6, scale: .98 }} animate={{ opacity: 1, y: 0, scale: 1 }} transition={reduced ? { duration: 0 } : { default: motionTokens.spring.smooth, opacity: { duration: motionTokens.duration.fast, ease: [...motionTokens.ease.enter] } }}>
    <div className={styles.searchRow}>
      <span className={styles.searchIcon}><Search width={18} height={18} strokeWidth={1.75} aria-hidden="true"/></span>
      <label className={styles.visuallyHidden} htmlFor={inputId}>{label}</label>
      <input ref={inputRef} id={inputId} role="combobox" aria-autocomplete="list" aria-expanded="true" aria-controls={`${inputId}-results`} aria-activedescendant={activeId ? `${inputId}-${activeId}` : undefined} value={query} onChange={event => { setQuery(event.target.value); setActiveIndex(null); }} onKeyDown={handleKeyDown} placeholder={placeholder} autoComplete="off"/>
      <AnimatePresence mode="popLayout" initial={false}>
        {query ? <motion.button key="clear" className={styles.clearButton} type="button" aria-label="Clear search" onClick={() => { setQuery(""); setActiveIndex(null); inputRef.current?.focus(); }} initial={reduced ? false : { opacity: 0, scale: .6, filter: `blur(${motionTokens.blur.subtle}px)` }} animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }} exit={reduced ? { opacity: 0, transition: { duration: 0 } } : { opacity: 0, scale: .6, filter: `blur(${motionTokens.blur.subtle}px)`, transition: leave }} transition={reduced ? { duration: 0 } : { default: motionTokens.spring.snappy, opacity: { duration: motionTokens.duration.instant } }}><X width={15} height={15} aria-hidden="true"/></motion.button> : null}
      </AnimatePresence>
      {onClose ? <button className={styles.closeButton} type="button" aria-label="Close command palette" onClick={onClose}>Esc</button> : null}
      <kbd className={styles.commandKey}>⌘ K</kbd>
    </div>
    <motion.div className={styles.resultsFrame} initial={false} animate={{ height: listHeight }} transition={reduced ? { duration: 0 } : motionTokens.spring.smooth}>
      <motion.div ref={listRef} layoutScroll id={`${inputId}-results`} className={styles.results} role="listbox" aria-label="Command results">
        <motion.span className={styles.highlight} style={{ y: highlightY, height: highlightHeight, opacity: highlightOpacity }} aria-hidden="true"/>
        {/* Results leave before the empty state arrives, so the frame makes one height change instead of two. */}
        <AnimatePresence mode="wait" initial={false}>
          {filtered.length ? <motion.div key="results" initial={reduced ? false : { opacity: 0 }} animate={{ opacity: 1 }} exit={reduced ? { opacity: 0, transition: { duration: 0 } } : { opacity: 0, transition: leave }} transition={enter}>
            {/* Filtered-out rows pop out of the flow as they fade, so the rest glide up and the frame starts closing on the same keystroke. */}
            <AnimatePresence mode="popLayout" initial={false} custom={glide}>
              {groupedItems.map(([group, entries]) => <motion.div layout={glide ? "position" : false} className={styles.group} role="group" aria-label={group} key={group} variants={rowExit} initial={reduced ? false : { opacity: 0 }} animate={{ opacity: 1 }} exit="exit" transition={reduced ? { duration: 0 } : { default: enter, layout: motionTokens.spring.smooth }}>
                <span className={styles.groupHeading} aria-hidden="true">{group}</span>
                <AnimatePresence mode="popLayout" initial={false} custom={glide}>{entries.map(({ item, index }) => <motion.button layout={glide ? "position" : false} key={item.id} id={`${inputId}-${item.id}`} className={`${styles.result} ${index === safeActiveIndex ? styles.activeResult : ""}`} type="button" role="option" aria-selected={index === safeActiveIndex} onClick={() => choose(item)} onPointerMove={event => {
                  // Ignore the synthetic moves browsers send while the list scrolls under a still pointer.
                  if (event.clientX === lastPointer.current.x && event.clientY === lastPointer.current.y) return;
                  lastPointer.current = { x: event.clientX, y: event.clientY };
                  if (index !== safeActiveIndex) setActiveIndex(index);
                }} variants={rowExit} initial={reduced ? false : { opacity: 0 }} animate={{ opacity: 1 }} exit="exit" transition={reduced ? { duration: 0 } : { default: enter, layout: motionTokens.spring.smooth }}><span className={styles.itemIcon} aria-hidden="true">{item.icon ?? <CommandIcon width={16} height={16}/>}</span><span className={styles.resultCopy}><span className={styles.resultLabel}><Highlighted text={item.label} ranges={rangesById.get(item.id) ?? []}/></span>{item.description && <span className={styles.resultDescription}>{item.description}</span>}</span>{item.shortcut && <span className={styles.shortcut}>{keysOf(item.shortcut).map((key, i) => <kbd key={i}>{key}</kbd>)}</span>}</motion.button>)}</AnimatePresence>
              </motion.div>)}
            </AnimatePresence>
          </motion.div> : <motion.div key="empty" className={styles.empty} initial={reduced ? false : { opacity: 0, y: 6, filter: `blur(${motionTokens.blur.subtle}px)` }} animate={{ opacity: 1, y: 0, filter: "blur(0px)" }} exit={reduced ? { opacity: 0, transition: { duration: 0 } } : { opacity: 0, y: -4, transition: leave }} transition={enter}><span><Search width={20} height={20} aria-hidden="true"/></span><strong>No results for “{query.trim()}”</strong><small>Try a different word or clear the search.</small></motion.div>}
        </AnimatePresence>
      </motion.div>
    </motion.div>
    <span className={styles.visuallyHidden} role="status" aria-live="polite">{query.trim() ? filtered.length ? `${filtered.length} ${filtered.length === 1 ? "result" : "results"}` : "No results" : ""}</span>
  </motion.div>;
}

export default CommandPalette;

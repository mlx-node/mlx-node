import { useState, type AnimationEvent, type CSSProperties, type ReactElement } from 'react';

import styles from './swap-text.module.css';

export interface SwapTextProps {
  text: string;
  /** min-width, e.g. '6ch', for a place whose neighbours must not move. */
  reserve?: string;
  /** 'text' (default): the whole text swaps. 'char' (numbers, timers): only the characters that changed swap. */
  granularity?: 'text' | 'char';
}

const MAX_CHAR_STEP = 4;

interface Cell {
  key: number;
  /** The character this cell showed before its last swap; null when it has not swapped. */
  prev: string | null;
  /** Order of this cell among the cells that swapped in the last change, counted from the right. */
  rank: number;
}

interface SwapState {
  text: string;
  /** Whole-text form: the generation key and the text it replaced. */
  whole: { key: number; prev: string | null } | null;
  /** Per-character form, one cell per character of `text`. */
  cells: Cell[] | null;
  nextKey: number;
}

function freshCells(length: number, firstKey: number): Cell[] {
  const cells: Cell[] = [];
  for (let i = 0; i < length; i += 1) cells.push({ key: firstKey + i, prev: null, rank: 0 });
  return cells;
}

function initial(text: string, char: boolean): SwapState {
  return char
    ? { text, whole: null, cells: freshCells(text.length, 0), nextKey: text.length }
    : { text, whole: { key: 0, prev: null }, cells: null, nextKey: 1 };
}

function advance(state: SwapState, text: string, char: boolean): SwapState {
  const previous = state.text;
  if (char && state.cells !== null && previous.length === text.length) {
    let nextKey = state.nextKey;
    let rank = 0;
    const cells = state.cells.slice();
    for (let i = text.length - 1; i >= 0; i -= 1) {
      if (previous[i] === text[i]) continue;
      cells[i] = { key: nextKey, prev: previous[i], rank: Math.min(rank, MAX_CHAR_STEP) };
      nextKey += 1;
      rank += 1;
    }
    return { text, whole: null, cells, nextKey };
  }
  if (char && state.cells === null && previous.length === text.length) {
    const cells = freshCells(text.length, state.nextKey);
    return advance({ text: previous, whole: null, cells, nextKey: state.nextKey + text.length }, text, true);
  }
  return { text, whole: { key: state.nextKey, prev: previous }, cells: null, nextKey: state.nextKey + 1 };
}

/**
 * A text that swaps in place: the leaving text is drawn by a ::before from `data-prev`, so `textContent` is always
 * exactly the current text. Old and new share one grid cell, so the box never jumps mid-swap.
 */
export function SwapText({ text, reserve, granularity = 'text' }: SwapTextProps): ReactElement {
  const char = granularity === 'char';
  const [state, setState] = useState<SwapState>(() => initial(text, char));
  if (state.text !== text) setState(advance(state, text, char));
  const shown = state;

  const style: CSSProperties | undefined = reserve === undefined ? undefined : { minWidth: reserve };

  const settle = (event: AnimationEvent<HTMLElement>): void => {
    if (event.target !== event.currentTarget || event.pseudoElement !== '::before') return;
    const key = Number(event.currentTarget.dataset.key);
    setState((current) => {
      if (current.whole !== null && current.whole.key === key && current.whole.prev !== null) {
        return { ...current, whole: { key, prev: null } };
      }
      if (current.cells === null) return current;
      const index = current.cells.findIndex((cell) => cell.key === key);
      if (index === -1 || current.cells[index].prev === null) return current;
      const cells = current.cells.slice();
      cells[index] = { ...cells[index], prev: null };
      return { ...current, cells };
    });
  };

  if (shown.cells !== null) {
    return (
      <span className={styles.root} data-granularity="char" style={style}>
        {shown.cells.map((cell, index) => (
          <span
            key={cell.key}
            className={styles.cell}
            data-key={cell.key}
            data-prev={cell.prev ?? undefined}
            style={{ '--rank': cell.rank } as CSSProperties}
            onAnimationEnd={settle}
          >
            <span className={styles.now}>{shown.text[index]}</span>
          </span>
        ))}
      </span>
    );
  }

  const whole = shown.whole ?? { key: 0, prev: null };
  return (
    <span className={styles.root} data-granularity="text" style={style}>
      <span
        key={whole.key}
        className={styles.cell}
        data-key={whole.key}
        data-prev={whole.prev ?? undefined}
        onAnimationEnd={settle}
      >
        <span className={styles.now}>{shown.text}</span>
      </span>
    </span>
  );
}

import { Check } from 'lucide-react';
import { forwardRef, useId, useRef, useState, type KeyboardEvent, type ReactElement } from 'react';

import styles from './answer-list.module.css';

export interface AnswerOption {
  /** Plain text, one line. */
  label: string;
  /** Plain text under the label. */
  description?: string;
  /** A short mark after the label ("Recommended"). */
  tag?: string;
}

/** The answer to one question: picked option indexes, and the "Other" row with its own text. */
export interface AnswerState {
  picks: number[];
  otherOn: boolean;
  otherText: string;
}

export const NO_ANSWER: AnswerState = { picks: [], otherOn: false, otherText: '' };

/**
 * Row `row` picked: options are rows 0…n-1, "Other" is row n. Single: the row alone. Multi: the row added (or, with
 * `flip`, switched off when it was on).
 */
export function pickRow(state: AnswerState, row: number, count: number, multi: boolean, flip = false): AnswerState {
  if (row === count) {
    if (!multi) return { ...state, picks: [], otherOn: true };
    return { ...state, otherOn: flip ? !state.otherOn : true };
  }
  if (!multi) return { ...state, picks: [row], otherOn: false };
  const on = state.picks.includes(row);
  if (on && !flip) return state;
  const picks = on ? state.picks.filter((p) => p !== row) : [...state.picks, row].sort((a, b) => a - b);
  return { ...state, picks };
}

/** Whether a row is picked. */
export function rowOn(state: AnswerState, row: number, count: number): boolean {
  return row === count ? state.otherOn : state.picks.includes(row);
}

export interface AnswerListProps {
  /** The question: the group's name. */
  label: string;
  /** Ids of what describes the group (the question's text, the hint). */
  describedBy?: string;
  multi: boolean;
  options: readonly AnswerOption[];
  value: AnswerState;
  onChange: (next: AnswerState) => void;
  /** The "Other" row's word, its field's name and placeholder. */
  other: { label: string; fieldLabel: string; placeholder: string };
  /** Enter on a picked row, or in the "Other" field: go on (Next, or Send). */
  onEnter?: () => void;
  /** Whether Enter may act now (a card's arm delay); default always. */
  enterReady?: (event: KeyboardEvent) => boolean;
  /** Characters the "Other" field takes. */
  otherMax?: number;
  disabled?: boolean;
  /** The row that holds focus (the "Other" field counts as its row), or null when focus left the list. */
  onFocusRow?: (row: number | null) => void;
}

/**
 * The options of a question as a radio group (single) or a group of checkboxes (multi), each numbered, plus an "Other"
 * row whose own text field follows the group. Nothing is picked by focus alone except by arrow keys in a single group.
 * A radio group is one tab stop; each checkbox is its own. Picking "Other" moves focus into its field. Enter on a row that is not picked only picks it; Enter on a picked row
 * goes on. Every text is drawn as plain text.
 */
export const AnswerList = forwardRef<HTMLDivElement, AnswerListProps>(function AnswerList(props, ref): ReactElement {
  const { multi, options, value, disabled = false } = props;
  const count = options.length;
  const rows = useRef<Array<HTMLDivElement | null>>([]);
  const field = useRef<HTMLTextAreaElement | null>(null);
  const [focused, setFocusedRow] = useState<number | null>(null);
  const setFocused = (row: number | null): void => {
    setFocusedRow(row);
    props.onFocusRow?.(row);
  };
  const fieldId = useId();
  const firstOn = [...Array(count + 1).keys()].find((row) => rowOn(value, row, count));
  const tabStop = focused ?? firstOn ?? 0;
  const ready = (event: KeyboardEvent): boolean => props.enterReady?.(event) ?? true;

  const set = (row: number, flip: boolean): void => {
    if (disabled) return;
    const next = pickRow(value, row, count, multi, flip);
    props.onChange(next);
    if (row === count && next.otherOn) field.current?.focus();
  };

  const onRowKey = (event: KeyboardEvent<HTMLDivElement>, row: number): void => {
    const last = count;
    const step = (to: number): void => {
      rows.current[to]?.focus();
      if (!multi) set(to, false);
    };
    switch (event.key) {
      case 'ArrowDown':
      case 'ArrowRight':
        step(row === last ? 0 : row + 1);
        break;
      case 'ArrowUp':
      case 'ArrowLeft':
        step(row === 0 ? last : row - 1);
        break;
      case 'Home':
        step(0);
        break;
      case 'End':
        step(last);
        break;
      case ' ':
        if (event.repeat) break;
        set(row, multi);
        break;
      case 'Enter':
        if (event.repeat || !ready(event)) break;
        if (rowOn(value, row, count)) props.onEnter?.();
        else set(row, false);
        break;
      default:
        return;
    }
    event.preventDefault();
  };

  const onFieldKey = (event: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      rows.current[count]?.focus();
      return;
    }
    if (event.key !== 'Enter' || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    if (!event.repeat && ready(event)) props.onEnter?.();
  };

  const role = multi ? 'checkbox' : 'radio';
  const row = (index: number, label: string, option?: AnswerOption): ReactElement => {
    const on = rowOn(value, index, count);
    return (
      <div
        key={index}
        ref={(node) => {
          rows.current[index] = node;
        }}
        role={role}
        aria-checked={on}
        aria-disabled={disabled ? true : undefined}
        aria-describedby={index === count ? fieldId : undefined}
        tabIndex={multi || index === tabStop ? 0 : -1}
        className={styles.row}
        data-checked={on ? '' : undefined}
        data-row={index}
        onClick={() => set(index, multi)}
        onFocus={() => setFocused(index)}
        onKeyDown={(event) => onRowKey(event, index)}
      >
        <span className={styles.mark} data-multi={multi ? '' : undefined} aria-hidden="true">
          {on ? multi ? <Check size={11} strokeWidth={2.5} /> : <span className={styles.dot} /> : null}
        </span>
        <span className={styles.key} aria-hidden="true">
          {index < 9 ? index + 1 : ''}
        </span>
        <span className={styles.text}>
          <span className={styles.title}>
            {label}
            {option?.tag !== undefined ? <span className={styles.tag}>{option.tag}</span> : null}
          </span>
          {option?.description !== undefined && option.description !== '' ? (
            <span className={styles.note}>{option.description}</span>
          ) : null}
        </span>
      </div>
    );
  };

  return (
    <div className={styles.root} data-disabled={disabled ? '' : undefined}>
      <div
        ref={ref}
        role={multi ? 'group' : 'radiogroup'}
        aria-label={props.label}
        aria-describedby={props.describedBy}
        aria-disabled={disabled ? true : undefined}
        className={styles.list}
        onBlur={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget)) setFocused(null);
        }}
      >
        {options.map((option, index) => row(index, option.label, option))}
        {row(count, props.other.label)}
      </div>
      <textarea
        id={fieldId}
        ref={field}
        rows={1}
        className={styles.field}
        aria-label={props.other.fieldLabel}
        placeholder={props.other.placeholder}
        value={value.otherText}
        maxLength={props.otherMax}
        disabled={disabled}
        data-other-field=""
        onChange={(event) => {
          const text = event.target.value;
          const next = text.trim() === '' ? { ...value, otherText: text } : pickRow(value, count, count, multi);
          props.onChange({ ...next, otherText: text });
        }}
        onFocus={() => setFocused(count)}
        onKeyDown={onFieldKey}
      />
    </div>
  );
});

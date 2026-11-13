import { type KeyboardEvent, type ReactElement, type ReactNode, useEffect, useRef, useState } from 'react';

import { Tooltip } from '../../arc/tooltip/tooltip';
import { LiveLabel } from '../live-label/live-label';
import { MorphLoader } from '../morph-loader/morph-loader';
import { motionX } from '../motion/motion';
import { useStaticMotion } from '../motion/use-static-motion';
import styles from './option-list.module.css';

export interface OptionItem {
  value: string;
  title: string;
  note?: string;
  trailing?: ReactNode;
  disabled?: boolean;
  disabledReason?: string;
  /** A change to this option is in flight: ring + `workingLabel` in place of the note. */
  working?: boolean;
  workingLabel?: string;
  /** 'danger': the title and the trailing part in the danger ink. */
  tone?: 'danger';
}

export interface OptionListProps {
  label: string;
  value: string;
  onValueChange: (value: string) => void;
  options: readonly OptionItem[];
  disabled?: boolean;
  /** Notes wrap onto more lines instead of ending in an ellipsis. */
  wrap?: boolean;
}

/**
 * A compact radio group. Arrow keys move focus without selecting; Space or Enter selects, because a selection here
 * can install files and restart processes.
 */
export function OptionList({
  label,
  value,
  onValueChange,
  options,
  disabled = false,
  wrap = false,
}: OptionListProps): ReactElement {
  const rows = useRef<Array<HTMLDivElement | null>>([]);
  const staticMotion = useStaticMotion();
  const [held, setHeld] = useState<string | null>(null);
  const [focused, setFocused] = useState<number | null>(null);
  const wasWorking = useRef<string | null>(null);

  const working = options.find((option) => option.working === true)?.value ?? null;
  useEffect(() => {
    const previous = wasWorking.current;
    wasWorking.current = working;
    if (previous === null || working !== null || previous !== value || staticMotion) return;
    setHeld(previous);
    const timer = setTimeout(() => setHeld(null), motionX.successHold * 1000);
    return () => clearTimeout(timer);
  }, [working, value, staticMotion]);

  const blocked = (option: OptionItem): boolean => disabled || option.disabled === true;
  const checkedIndex = options.findIndex((option) => option.value === value);
  const firstOpen = options.findIndex((option) => !blocked(option));
  const tabStop = focused ?? (checkedIndex !== -1 ? checkedIndex : firstOpen);

  const move = (from: number, step: 1 | -1 | 'home' | 'end'): void => {
    const order = options.map((_, index) => index).filter((index) => !blocked(options[index]!));
    if (order.length === 0) return;
    let target: number;
    if (step === 'home') target = order[0]!;
    else if (step === 'end') target = order[order.length - 1]!;
    else {
      const at = order.indexOf(from);
      target = at === -1 ? order[0]! : order[(at + step + order.length) % order.length]!;
    }
    rows.current[target]?.focus();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>, index: number, option: OptionItem): void => {
    if (event.key === 'ArrowDown' || event.key === 'ArrowRight') move(index, 1);
    else if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') move(index, -1);
    else if (event.key === 'Home') move(index, 'home');
    else if (event.key === 'End') move(index, 'end');
    else if (event.key === ' ' || event.key === 'Enter') {
      if (!blocked(option) && option.value !== value) onValueChange(option.value);
    } else return;
    event.preventDefault();
  };

  return (
    <div
      role="radiogroup"
      aria-label={label}
      aria-disabled={disabled ? true : undefined}
      className={styles.root}
      data-wrap={wrap ? '' : undefined}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) setFocused(null);
      }}
    >
      {options.map((option, index) => {
        const checked = option.value === value;
        const isBlocked = blocked(option);
        const busy = option.working === true;
        const showHeld = held === option.value && !busy;
        const row = (
          <div
            key={option.value}
            ref={(node) => {
              rows.current[index] = node;
            }}
            role="radio"
            aria-checked={checked}
            aria-disabled={isBlocked ? true : undefined}
            aria-busy={busy ? true : undefined}
            tabIndex={index === tabStop ? 0 : -1}
            className={styles.row}
            data-checked={checked ? '' : undefined}
            data-blocked={isBlocked ? '' : undefined}
            data-tone={option.tone}
            onClick={() => {
              if (!isBlocked && !checked) onValueChange(option.value);
            }}
            onFocus={() => setFocused(index)}
            onKeyDown={(event) => onKeyDown(event, index, option)}
          >
            <span className={styles.lead} aria-hidden="true">
              {busy || showHeld ? (
                <MorphLoader size={14} status={busy ? 'loading' : 'success'} />
              ) : (
                <span className={styles.mark} data-checked={checked ? '' : undefined}>
                  {checked ? <span className={styles.dot} /> : null}
                </span>
              )}
            </span>
            <span className={styles.text}>
              <span className={styles.title}>{option.title}</span>
              {busy && option.workingLabel !== undefined ? (
                <LiveLabel active className={styles.note}>
                  {option.workingLabel}
                </LiveLabel>
              ) : option.note !== undefined ? (
                <span className={styles.note}>{option.note}</span>
              ) : null}
            </span>
            {option.trailing !== undefined && option.trailing !== null ? (
              <span className={styles.trailing}>{option.trailing}</span>
            ) : (
              <span />
            )}
          </div>
        );
        if (isBlocked && option.disabledReason !== undefined && option.disabledReason !== '') {
          return (
            <Tooltip key={option.value} content={option.disabledReason}>
              {row}
            </Tooltip>
          );
        }
        return row;
      })}
    </div>
  );
}

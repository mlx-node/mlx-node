import { X } from 'lucide-react';
import { AnimatePresence, motion } from 'motion/react';
import { type KeyboardEvent, type ReactElement, useId, useState } from 'react';

import { cn } from '../../../lib/cn';
import { motionX } from '../motion/motion';
import { useStaticMotion } from '../motion/use-static-motion';
import styles from './tag-input.module.css';

export interface TagInputProps {
  label?: string;
  values: readonly string[];
  onChange: (next: string[]) => void;
  placeholder?: string;
  hint?: string;
  allowDuplicates?: boolean;
  className?: string;
}

/** Interactive chip/tag editor for enum choices and CLEF score levels. */
export function TagInput({
  label,
  values,
  onChange,
  placeholder = 'Type value and press Enter',
  hint,
  allowDuplicates = false,
  className,
}: TagInputProps): ReactElement {
  const id = useId();
  const staticMotion = useStaticMotion();
  const [draft, setDraft] = useState('');

  const commit = (raw: string): void => {
    const parts = raw
      .split(',')
      .map((part) => part.trim())
      .filter((part) => part.length > 0);
    if (parts.length === 0) return;
    const next = [...values];
    for (const part of parts) {
      if (!allowDuplicates && next.includes(part)) continue;
      next.push(part);
    }
    onChange(next);
    setDraft('');
  };

  const removeAt = (index: number): void => {
    onChange(values.filter((_, i) => i !== index));
  };

  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>): void => {
    if (event.key === 'Enter' || event.key === ',') {
      event.preventDefault();
      commit(draft);
    } else if (event.key === 'Backspace' && draft === '' && values.length > 0) {
      removeAt(values.length - 1);
    }
  };

  return (
    <div className={cn(styles.root, className)}>
      {label !== undefined ? (
        <label htmlFor={id} className={styles.label}>
          {label}
        </label>
      ) : null}
      <div className={styles.box}>
        <AnimatePresence initial={false}>
          {values.map((tag, index) => (
            <motion.span
              key={`${tag}-${String(index)}`}
              layout={!staticMotion}
              initial={staticMotion ? false : { opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={staticMotion ? { opacity: 0, transition: { duration: 0 } } : { opacity: 0, scale: 0.85 }}
              transition={staticMotion ? { duration: 0 } : motionX.spring.expand}
              className={styles.chip}
            >
              <span className={styles.chipText}>{tag}</span>
              <button
                type="button"
                className={styles.remove}
                aria-label={`Remove ${tag}`}
                onClick={() => removeAt(index)}
              >
                <X size={11} strokeWidth={2} aria-hidden="true" />
              </button>
            </motion.span>
          ))}
        </AnimatePresence>
        <input
          id={id}
          type="text"
          className={styles.input}
          value={draft}
          placeholder={placeholder}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={onKeyDown}
          onBlur={() => {
            if (draft.trim() !== '') commit(draft);
          }}
        />
      </div>
      {hint !== undefined ? <p className={styles.hint}>{hint}</p> : null}
    </div>
  );
}

import { Eye, EyeOff } from 'lucide-react';
import { type ReactElement, useId, useState } from 'react';

import { IconButton } from '../icon-button/icon-button';
import styles from './secret-field.module.css';

export interface SecretFieldProps {
  label: string;
  /** What the secret is for, added to the field's and the Show button's accessible names (not shown). */
  context?: string;
  value: string;
  onValueChange(value: string): void;
  placeholder?: string;
  description?: string;
  error?: string;
  disabled?: boolean;
  readOnly?: boolean;
  /** Enter in the field. */
  onSubmit?(): void;
  autoFocus?: boolean;
}

/**
 * A secret entry field: hidden while typed until the Show button is pressed. It never reads the clipboard and holds
 * no copy of the value; the caller owns it and clears it.
 */
export function SecretField(props: SecretFieldProps): ReactElement {
  const id = useId();
  const [shown, setShown] = useState(false);
  const hintId = props.description !== undefined && props.description !== '' ? `${id}-hint` : undefined;
  const errorId = props.error !== undefined && props.error !== '' ? `${id}-error` : undefined;
  const describedBy = [hintId, errorId].filter((part) => part !== undefined).join(' ') || undefined;
  return (
    <div className={styles.field}>
      <label className={styles.label} htmlFor={id}>
        {props.label}
        {props.context !== undefined ? <span className={styles.hidden}> for {props.context}</span> : null}
      </label>
      <div className={styles.box}>
        <input
          id={id}
          className={styles.input}
          type={shown ? 'text' : 'password'}
          data-shown={shown ? '' : undefined}
          value={props.value}
          placeholder={props.placeholder}
          disabled={props.disabled}
          readOnly={props.readOnly}
          autoFocus={props.autoFocus}
          autoComplete="off"
          spellCheck={false}
          autoCapitalize="off"
          autoCorrect="off"
          aria-invalid={errorId !== undefined ? true : undefined}
          aria-describedby={describedBy}
          onChange={(event) => props.onValueChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key !== 'Enter' || event.nativeEvent.isComposing || props.onSubmit === undefined) return;
            event.preventDefault();
            props.onSubmit();
          }}
        />
        <IconButton
          className={styles.reveal}
          label={`${shown ? 'Hide key' : 'Show key'}${props.context !== undefined ? ` for ${props.context}` : ''}`}
          tooltip={shown ? 'Hide key' : 'Show key'}
          icon={shown ? <EyeOff size={14} aria-hidden="true" /> : <Eye size={14} aria-hidden="true" />}
          aria-pressed={shown}
          disabled={props.disabled}
          onClick={() => setShown((value) => !value)}
        />
      </div>
      {hintId !== undefined ? (
        <span id={hintId} className={styles.description}>
          {props.description}
        </span>
      ) : null}
      {errorId !== undefined ? (
        <span id={errorId} className={styles.error} role="alert">
          {props.error}
        </span>
      ) : null}
    </div>
  );
}

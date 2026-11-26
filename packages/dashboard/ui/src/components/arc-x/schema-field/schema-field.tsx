import type { KeyboardEvent, ReactElement, ReactNode } from 'react';
import { useId } from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import type { McpField } from '../../../lib/chat-types';
import { Switch } from '../../arc/switch/switch';
import styles from './schema-field.module.css';

/** What a field holds on the page: text, a switch, an option index (null for none), option indexes. */
export type SchemaValue = string | boolean | number | number[] | null;

/** A text field longer than this is a multi-line box. */
export const LONG_TEXT_CHARS = 200;
/** The longest value a box takes: the host refuses a longer one as a bad body, as the agent's own form stops there. */
export const VALUE_MAX_CHARS = 4_096;
/** A single select with more options than this is a menu; fewer are radio rows. */
export const SELECT_MENU_OVER = 4;

/** The value a field starts with: the server's default, else empty. */
export function initialValue(field: McpField): SchemaValue {
  switch (field.type) {
    case 'boolean':
      return field.initial;
    case 'choice':
      return field.initial;
    case 'choices':
      return [...field.initial];
    case 'text':
      return field.format === 'date-time' ? localDateTime(field.initial) : field.initial;
    default:
      return field.initial;
  }
}

/** A date and time the browser's date-time box can hold (`YYYY-MM-DDTHH:MM[:SS]`); anything else starts empty. */
function localDateTime(text: string): string {
  const shape = text.length === 16 || text.length === 19;
  return shape && text[10] === 'T' && text[13] === ':' ? text : '';
}

function pad(n: number): string {
  return String(Math.trunc(Math.abs(n))).padStart(2, '0');
}

/**
 * The date-time box's value as RFC 3339, the local zone's offset added: `2026-08-19T10:00` becomes
 * `2026-08-19T10:00:00+02:00`. Text of another shape goes as it is (the host refuses it).
 */
export function rfc3339(value: string): string {
  if (value === '' || localDateTime(value) === '') return value;
  const withSeconds = value.length === 16 ? `${value}:00` : value;
  const [day, time] = withSeconds.split('T') as [string, string];
  const [y, m, d] = day.split('-').map(Number) as [number, number, number];
  const [h, min, s] = time.split(':').map(Number) as [number, number, number];
  const offset = -new Date(y, m - 1, d, h, min, s).getTimezoneOffset();
  const sign = offset < 0 ? '-' : '+';
  return `${withSeconds}${sign}${pad(offset / 60)}:${pad(offset % 60)}`;
}

function inputType(field: Extract<McpField, { type: 'text' }>): string {
  switch (field.format) {
    case 'email':
      return 'email';
    case 'uri':
      return 'url';
    case 'date':
      return 'date';
    case 'date-time':
      return 'datetime-local';
    default:
      return 'text';
  }
}

export interface SchemaFieldProps {
  field: McpField;
  /** The field's place in the form: its `data-field`. */
  index: number;
  value: SchemaValue;
  onChange: (value: SchemaValue) => void;
  /** The host's word on what is wrong with the value; null when it is fine. */
  error: string | null;
  disabled?: boolean;
  /** Enter in a one-line field (a multi-line box takes Enter as a new line). */
  onEnter?: (event: KeyboardEvent<HTMLInputElement>) => void;
}

/**
 * One field of a server's form: its label (a needed field carries "*", and "needed" in its name), the server's
 * description, the control for its kind, and the host's error under it. Description and error are tied to the control
 * by `aria-describedby`; a wrong value sets `aria-invalid`. Every text is drawn as plain text.
 */
export function SchemaField(props: SchemaFieldProps): ReactElement {
  const { field, value, disabled = false } = props;
  const id = useId();
  const labelId = `${id}-label`;
  const descriptionId = field.description !== null && field.description !== '' ? `${id}-description` : undefined;
  const errorId = props.error !== null ? `${id}-error` : undefined;
  const describedBy = [descriptionId, errorId].filter((part) => part !== undefined).join(' ') || undefined;
  const name = field.required ? fill(CHAT.mcp.fieldName, { title: field.title, needed: CHAT.mcp.needed }) : field.title;
  const invalid = props.error !== null ? true : undefined;
  const common = {
    'aria-label': name,
    'aria-describedby': describedBy,
    'aria-invalid': invalid,
  };
  const text = typeof value === 'string' ? value : '';
  const onKeyDown = (event: KeyboardEvent<HTMLInputElement>): void => {
    if (event.key !== 'Enter' || event.shiftKey || event.nativeEvent.isComposing || props.onEnter === undefined) return;
    event.preventDefault();
    props.onEnter(event);
  };

  let control: ReactNode;
  let group = false;
  switch (field.type) {
    case 'text':
      control =
        field.maxLength !== null && field.maxLength > LONG_TEXT_CHARS ? (
          <textarea
            id={id}
            className={styles.box}
            rows={3}
            value={text}
            maxLength={VALUE_MAX_CHARS}
            disabled={disabled}
            aria-required={field.required ? true : undefined}
            {...common}
            onChange={(event) => props.onChange(event.target.value)}
          />
        ) : (
          <input
            id={id}
            className={styles.input}
            type={inputType(field)}
            value={text}
            maxLength={VALUE_MAX_CHARS}
            disabled={disabled}
            aria-required={field.required ? true : undefined}
            {...common}
            onChange={(event) => props.onChange(event.target.value)}
            onKeyDown={onKeyDown}
          />
        );
      break;
    case 'number':
    case 'integer':
      control = (
        <input
          id={id}
          className={styles.input}
          type="text"
          inputMode={field.type === 'integer' ? 'numeric' : 'decimal'}
          value={text}
          maxLength={VALUE_MAX_CHARS}
          disabled={disabled}
          aria-required={field.required ? true : undefined}
          {...common}
          onChange={(event) => props.onChange(event.target.value)}
          onKeyDown={onKeyDown}
        />
      );
      break;
    case 'boolean':
      control = (
        <span className={styles.switchRow}>
          <Switch
            id={id}
            checked={value === true}
            disabled={disabled}
            {...common}
            onCheckedChange={(checked) => props.onChange(checked)}
          />
        </span>
      );
      break;
    case 'choice': {
      const picked = typeof value === 'number' ? value : null;
      if (field.options.length > SELECT_MENU_OVER) {
        control = (
          <select
            id={id}
            className={styles.input}
            value={picked === null ? '' : String(picked)}
            disabled={disabled}
            aria-required={field.required ? true : undefined}
            {...common}
            onChange={(event) => props.onChange(event.target.value === '' ? null : Number(event.target.value))}
          >
            <option value="">{CHAT.mcp.pick}</option>
            {field.options.map((option, i) => (
              <option key={i} value={String(i)}>
                {option.label}
              </option>
            ))}
          </select>
        );
        break;
      }
      group = true;
      control = (
        <div
          role="radiogroup"
          className={styles.list}
          aria-required={field.required ? true : undefined}
          {...common}
          data-control=""
        >
          {field.options.map((option, i) => (
            <label key={i} className={styles.option}>
              <input
                type="radio"
                name={id}
                checked={picked === i}
                disabled={disabled}
                onChange={() => props.onChange(i)}
              />
              <span className={styles.optionText}>{option.label}</span>
            </label>
          ))}
        </div>
      );
      break;
    }
    case 'choices': {
      const picks = Array.isArray(value) ? value : [];
      group = true;
      control = (
        <div role="group" className={styles.list} {...common} data-control="">
          {field.options.map((option, i) => (
            <label key={i} className={styles.option}>
              <input
                type="checkbox"
                checked={picks.includes(i)}
                disabled={disabled}
                onChange={(event) =>
                  props.onChange(
                    event.target.checked ? [...picks, i].sort((a, b) => a - b) : picks.filter((pick) => pick !== i),
                  )
                }
              />
              <span className={styles.optionText}>{option.label}</span>
            </label>
          ))}
        </div>
      );
      break;
    }
  }

  const label = (
    <>
      {field.title}
      {field.required ? (
        <span className={styles.mark} aria-hidden="true">
          {' *'}
        </span>
      ) : null}
    </>
  );
  return (
    <div className={styles.field} data-field={props.index} data-invalid={invalid ? '' : undefined}>
      {group ? (
        <span id={labelId} className={styles.label} aria-hidden="true">
          {label}
        </span>
      ) : (
        <label id={labelId} htmlFor={id} className={styles.label}>
          {label}
        </label>
      )}
      <div className={styles.body}>
        {control}
        {descriptionId !== undefined ? (
          <p id={descriptionId} className={styles.description}>
            {field.description}
          </p>
        ) : null}
        {errorId !== undefined ? (
          <p id={errorId} className={styles.error}>
            {props.error}
          </p>
        ) : null}
      </div>
    </div>
  );
}

/** The fields of one form: label left of its control from 520 px of form width, above it below that. */
export function SchemaForm({ children }: { children: ReactNode }): ReactElement {
  return <div className={styles.form}>{children}</div>;
}

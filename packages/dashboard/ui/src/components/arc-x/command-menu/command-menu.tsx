import { useEffect, useRef, useState, type ReactElement } from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import { matchCommands } from '../../../lib/command-match';
import styles from './command-menu.module.css';

/** One command the `/` list offers. */
export interface CommandRow {
  name: string;
  group: 'builtin' | 'skill' | 'workflow';
  /** One line: what it does. */
  about: string;
  /** The words it takes, at the row's end; null when none. */
  words: string | null;
  /** Other names it is typed as (`/usage` for `/context`): typed in full, Enter runs it as the name does. */
  aliases?: readonly string[];
  /** What to know before running it: the row's hover title, and read with the row. */
  tip?: string;
}

/** The row is the command typed in full, by its name or another name it is typed as. */
export function typedInFull(row: CommandRow | undefined, typed: string): boolean {
  const name = typed.toLowerCase();
  return row !== undefined && (row.name === name || row.aliases?.includes(name) === true);
}

const GROUPS: ReadonlyArray<CommandRow['group']> = ['builtin', 'skill', 'workflow'];

/** The rows that match the typed name, as the list shows them: by group, the best match first in each. */
export function commandOrder(rows: readonly CommandRow[], typed: string): CommandRow[] {
  const ranked = matchCommands(rows, typed);
  return GROUPS.flatMap((group) => ranked.filter((row) => row.group === group));
}

export interface CommandMenuProps {
  /** The listbox id: the text field's `aria-controls`. */
  id: string;
  /** The rows in the order {@link commandOrder} gives. */
  rows: readonly CommandRow[];
  /** The highlighted row. */
  active: number;
  /** The id of row `index`: the text field's `aria-activedescendant`. */
  optionId: (index: number) => string;
  /** A muted last line: more commands come once the conversation is loaded. */
  later: boolean;
  /** A row was clicked. The list never takes focus: a press keeps it in the text field. */
  onPick: (index: number) => void;
  onHover: (index: number) => void;
}

/**
 * The `/` command list: it opens upward over the composer and is driven from the text field (`aria-activedescendant`);
 * rows are grouped as Commands, Skills, Workflows. Agent text is drawn as plain text.
 */
export function CommandMenu(props: CommandMenuProps): ReactElement {
  const list = useRef<HTMLDivElement | null>(null);
  const [said, setSaid] = useState('');
  const count = props.rows.length === 1 ? CHAT.slash.countOne : fill(CHAT.slash.count, { n: props.rows.length });

  useEffect(() => {
    setSaid(count);
  }, [count]);

  const { active, optionId } = props;
  useEffect(() => {
    const id = optionId(active);
    const row = [...(list.current?.querySelectorAll<HTMLElement>('[role="option"]') ?? [])].find((r) => r.id === id);
    row?.scrollIntoView?.({ block: 'nearest' });
  }, [active, optionId]);

  let index = 0;
  return (
    <div className={styles.panel} data-command-menu="">
      <div ref={list} id={props.id} role="listbox" aria-label={CHAT.slash.label} className={styles.list}>
        {GROUPS.map((group) => {
          const rows = props.rows.filter((row) => row.group === group);
          if (rows.length === 0) return null;
          const headingId = `${props.id}-${group}`;
          return (
            <div key={group} role="group" aria-labelledby={headingId} className={styles.group}>
              <div id={headingId} className={styles.heading}>
                {CHAT.slash.groups[group]}
              </div>
              {rows.map((row) => {
                const at = index;
                index += 1;
                return (
                  <div
                    key={row.name}
                    id={props.optionId(at)}
                    role="option"
                    aria-selected={at === props.active}
                    className={styles.row}
                    title={row.tip}
                    onMouseDown={(event) => event.preventDefault()}
                    onClick={() => props.onPick(at)}
                    onMouseMove={() => {
                      if (at !== props.active) props.onHover(at);
                    }}
                  >
                    <span className={styles.name}>/{row.name}</span>
                    <span className={styles.about}>{row.about}</span>
                    {row.words !== null ? <span className={styles.words}>{row.words}</span> : null}
                    {row.tip !== undefined ? <span className={styles.srOnly}>{` ${row.tip}`}</span> : null}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
      {props.later ? <div className={styles.later}>… {CHAT.slash.later}</div> : null}
      <span className={styles.srOnly} role="status">
        {said}
      </span>
    </div>
  );
}

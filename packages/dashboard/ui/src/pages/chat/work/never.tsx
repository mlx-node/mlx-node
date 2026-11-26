import { Ban, ChevronLeft, ChevronRight, X } from 'lucide-react';
import { type ReactElement, type RefObject, useEffect, useLayoutEffect, useRef, useState } from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import type { NeverRuleView } from '../../../lib/chat-types';
import { formatClock, formatDay } from '../../../lib/format';
import { folderName } from '../../../lib/permission-text';
import { oneLine } from '../../../lib/web-tools';
import { visibleText } from '../../../lib/work-tools';
import styles from './never.module.css';

/** Work: the never-allow rules of the open conversation's folder, as the permission chip reads and removes them. */
export interface NeverRulesControl {
  /** Rules of the folder now (`ConversationInfo.neverRules`). */
  count: number;
  folder: string;
  load: () => Promise<NeverRuleView[]>;
  /** Resolves false when the rule was not removed (the page says why). */
  remove: (ruleId: string, words: string) => Promise<boolean>;
}

function shownWords(words: string): string {
  return oneLine(visibleText(words)).trim();
}

/** When a rule was made: the time for today, the day before that. */
export function ruleWhen(at: number, now: number = Date.now()): string {
  const day = new Date(now);
  day.setHours(0, 0, 0, 0);
  return at >= day.getTime() ? fill(CHAT.never.today, { time: formatClock(at) }) : formatDay(at);
}

/** The modes view's row under the radio group: "Never allowed · 2 ›". Not drawn with no rules. */
export function NeverChipRow(props: { count: number; onOpen: () => void }): ReactElement | null {
  if (props.count <= 0) return null;
  return (
    <button type="button" className={styles.chipRow} data-never-chip-row="" onClick={props.onOpen}>
      <Ban size={14} strokeWidth={1.75} aria-hidden="true" />
      <span className={styles.chipText}>{fill(CHAT.never.chipRow, { n: props.count })}</span>
      <ChevronRight size={14} strokeWidth={1.75} aria-hidden="true" />
    </button>
  );
}

/**
 * The rules list, a view of the permission chip's popover: each rule's words, when it was made, and its remove
 * button. After a removal focus stays in the list (the next row, else the way back).
 */
export function NeverRules(props: {
  control: NeverRulesControl;
  onBack: () => void;
  backRef: RefObject<HTMLButtonElement | null>;
}): ReactElement {
  const { control } = props;
  const [rules, setRules] = useState<NeverRuleView[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [removing, setRemoving] = useState<string | null>(null);
  const buttons = useRef(new Map<string, HTMLButtonElement>());
  const focusNext = useRef<string | null | undefined>(undefined);
  const load = control.load;

  useEffect(() => {
    let live = true;
    load().then(
      (list) => {
        if (!live) return;
        setRules(list);
        setFailed(false);
      },
      () => {
        if (live) setFailed(true);
      },
    );
    return () => {
      live = false;
    };
  }, [load, control.count]);

  useLayoutEffect(() => {
    const next = focusNext.current;
    if (next === undefined) return;
    focusNext.current = undefined;
    const target = next === null ? null : (buttons.current.get(next) ?? null);
    (target ?? props.backRef.current)?.focus({ preventScroll: true });
  }, [rules]);

  const remove = (rule: NeverRuleView, index: number): void => {
    if (removing !== null || rules === null) return;
    setRemoving(rule.id);
    void control.remove(rule.id, rule.words).then((ok) => {
      setRemoving(null);
      if (!ok) return;
      focusNext.current = rules[index + 1]?.id ?? rules[index - 1]?.id ?? null;
      setRules((list) => (list === null ? list : list.filter((r) => r.id !== rule.id)));
    });
  };

  return (
    <div className={styles.view} data-never-rules="">
      <div className={styles.head}>
        <button
          ref={props.backRef}
          type="button"
          className={styles.back}
          aria-label={CHAT.never.back}
          onClick={props.onBack}
        >
          <ChevronLeft size={14} strokeWidth={1.75} aria-hidden="true" />
        </button>
        <p className={styles.title} title={control.folder}>
          {fill(CHAT.never.listTitle, { folder: folderName(control.folder) })}
        </p>
      </div>
      {failed ? <p className={styles.note}>{CHAT.never.loadFailed}</p> : null}
      {rules !== null && rules.length > 0 ? (
        <ul className={styles.list}>
          {rules.map((rule, index) => {
            const words = shownWords(rule.words);
            return (
              <li key={rule.id} className={styles.row} data-never-rule={rule.id}>
                <span className={styles.words} title={words}>
                  {words}
                </span>
                <span className={styles.when}>{ruleWhen(rule.at)}</span>
                <button
                  ref={(node) => {
                    if (node === null) buttons.current.delete(rule.id);
                    else buttons.current.set(rule.id, node);
                  }}
                  type="button"
                  className={styles.remove}
                  aria-label={fill(CHAT.never.removeLabel, { words })}
                  aria-busy={removing === rule.id ? true : undefined}
                  disabled={removing !== null && removing !== rule.id}
                  onClick={() => remove(rule, index)}
                >
                  <X size={14} strokeWidth={1.75} aria-hidden="true" />
                </button>
              </li>
            );
          })}
        </ul>
      ) : null}
      <p className={styles.note}>{CHAT.never.listNote}</p>
    </div>
  );
}

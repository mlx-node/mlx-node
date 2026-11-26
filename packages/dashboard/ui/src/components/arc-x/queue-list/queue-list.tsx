import { ArrowUpToLine, FileText, Image as ImageIcon, Pencil, X } from 'lucide-react';
import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactElement,
  type RefObject,
} from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import type { QueuedMessage } from '../../../lib/chat-types';
import { formatBytes } from '../../../lib/format';
import { hasNonWhitespace } from '../../../lib/text-scan';
import { Button } from '../../arc/button/button';
import { Tooltip } from '../../arc/tooltip/tooltip';
import { IconButton } from '../icon-button/icon-button';
import styles from './queue-list.module.css';

export interface QueueListProps {
  items: readonly QueuedMessage[];
  /** The header's hold sentence; null while the list sends by itself. */
  held: string | null;
  /** Not held: what the list waits for before it sends by itself (a goal that runs). Shown only while not held. */
  waits?: string | null;
  /** The row open for edit in this window, or null. */
  editing: string | null;
  onEditingChange: (id: string | null) => void;
  /** Saves a row's new text and closes its edit; false: it was not saved and the edit stays open. */
  onSave: (id: string, text: string) => Promise<boolean>;
  onRemove: (id: string) => void;
  onClear: () => void;
  /** Shown while held: the hold ends and the first row goes. */
  onSendNext: () => void;
  /** Focus goes back to the message box. */
  onFocusOut: () => void;
  /** Work, while an answer runs: each row gets Send now. Without it the rows have no such button (Chat, idle). */
  sendNow?: SendNowControl;
}

export interface SendNowControl {
  /** Why the row cannot go now (its tooltip), or null when it can. */
  off: (item: QueuedMessage) => string | null;
  /** Rows whose Send now waits for its reply. */
  busy: ReadonlySet<string>;
  onSend: (id: string) => void;
}

function SendNowButton({ item, control }: { item: QueuedMessage; control: SendNowControl }): ReactElement {
  const off = control.off(item);
  const busy = control.busy.has(item.id);
  return (
    <Tooltip content={off ?? CHAT.queue.sendNowTip}>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        className={styles.sendNow}
        aria-label={CHAT.queue.sendNow}
        aria-disabled={off !== null || busy ? true : undefined}
        onClick={() => {
          if (off === null && !busy) control.onSend(item.id);
        }}
      >
        <ArrowUpToLine className={styles.sendNowIcon} size={14} strokeWidth={1.75} aria-hidden="true" />
        <span className={styles.sendNowLabel}>{CHAT.queue.sendNow}</span>
      </Button>
    </Tooltip>
  );
}

function grow(node: HTMLTextAreaElement | null): void {
  if (node === null) return;
  node.style.height = 'auto';
  if (node.scrollHeight > 0) node.style.height = `${node.scrollHeight}px`;
}

function EditRow({
  item,
  onSave,
  onCancel,
}: {
  item: QueuedMessage;
  onSave: (text: string) => void;
  onCancel: () => void;
}): ReactElement {
  const [text, setText] = useState(item.text);
  const area = useRef<HTMLTextAreaElement | null>(null);
  useLayoutEffect(() => grow(area.current), [text]);
  useEffect(() => {
    const node = area.current;
    if (node === null) return;
    node.focus({ preventScroll: true });
    node.setSelectionRange(node.value.length, node.value.length);
  }, []);
  const ready = hasNonWhitespace(text);
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>): void => {
    const native = event.nativeEvent;
    if (native.isComposing || native.keyCode === 229) return;
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      if (ready) onSave(text);
      return;
    }
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      onCancel();
    }
  };
  return (
    <div className={styles.edit}>
      <textarea
        ref={area}
        className={styles.editInput}
        rows={1}
        value={text}
        aria-label={CHAT.queue.editLabel}
        spellCheck
        onChange={(event) => setText(event.target.value)}
        onKeyDown={onKeyDown}
      />
      <Button type="button" variant="ghost" size="sm" disabled={!ready} onClick={() => onSave(text)}>
        {CHAT.queue.save}
      </Button>
      <Button type="button" variant="ghost" size="sm" onClick={onCancel}>
        {CHAT.queue.cancel}
      </Button>
    </div>
  );
}

function Files({ item }: { item: QueuedMessage }): ReactElement | null {
  if (item.files.length === 0) return null;
  return (
    <span className={styles.files}>
      {item.files.map((file, i) => {
        const Icon = file.kind === 'image' ? ImageIcon : FileText;
        const name = file.kind === 'file' ? file.name : CHAT.attach.image;
        return (
          <span key={i} className={styles.file}>
            <Icon size={12} strokeWidth={1.75} aria-hidden="true" />
            <span className={styles.fileName}>{name}</span>
            <span className={styles.fileSize}>{formatBytes(file.bytes)}</span>
          </span>
        );
      })}
    </span>
  );
}

/**
 * The queued messages, above the composer's box: a header (count, why the list waits, Send next, Clear) and up to five
 * rows, three in view. A row's text opens its edit (Enter saves, Esc leaves); the queue sends no row while one is open.
 */
export function QueueList(props: QueueListProps): ReactElement | null {
  const editButtons = useRef(new Map<string, HTMLButtonElement>());
  const removeButtons = useRef(new Map<string, HTMLButtonElement>());
  /** After Remove: the row whose Remove takes focus once the list has changed. */
  const focusNext = useRef<string | null>(null);
  const backTo = useRef<string | null>(null);
  const { items, editing, onFocusOut } = props;

  useEffect(() => {
    const next = focusNext.current;
    if (next === null) return;
    const button = removeButtons.current.get(next);
    if (button === undefined) return;
    focusNext.current = null;
    button.focus();
  }, [items]);

  useEffect(() => {
    const back = backTo.current;
    if (back === null || editing !== null) return;
    backTo.current = null;
    editButtons.current.get(back)?.focus();
  }, [editing]);

  if (items.length === 0) return null;

  const remove = (index: number): void => {
    const item = items[index]!;
    const next = items[index + 1];
    focusNext.current = next?.id ?? null;
    props.onRemove(item.id);
    if (next === undefined) onFocusOut();
  };

  const cancel = (id: string): void => {
    backTo.current = id;
    props.onEditingChange(null);
  };

  // A save that went through closes the edit on the page's side (`editing` turns null); focus then goes back to ✎.
  const save = (id: string, text: string): void => {
    backTo.current = id;
    void props.onSave(id, text).then((saved) => {
      if (!saved && backTo.current === id) backTo.current = null;
    });
  };

  const bind =
    (map: RefObject<Map<string, HTMLButtonElement>>, id: string) =>
    (node: HTMLButtonElement | null): void => {
      if (node === null) map.current.delete(id);
      else map.current.set(id, node);
    };

  const title = fill(CHAT.queue.title, { n: items.length });
  const why = props.held ?? props.waits ?? null;
  return (
    <div className={styles.root} data-held={props.held !== null ? '' : undefined}>
      <div className={styles.head}>
        <span className={styles.title} title={why === null ? undefined : `${title} · ${why}`}>
          {title}
          {why !== null ? <span className={styles.held}>{` · ${why}`}</span> : null}
        </span>
        {props.held !== null ? (
          <Button type="button" variant="ghost" size="sm" onClick={props.onSendNext}>
            {CHAT.queue.sendNext}
          </Button>
        ) : null}
        <Button type="button" variant="ghost" size="sm" onClick={props.onClear}>
          {CHAT.queue.clear}
        </Button>
      </div>
      <ol className={styles.rows} role="list" aria-label={CHAT.queue.label}>
        {items.map((item, index) => (
          <li key={item.id} className={styles.row} role="listitem" data-editing={item.editing ? '' : undefined}>
            <span className={styles.number} aria-hidden="true">
              {index + 1}
            </span>
            {editing === item.id ? (
              <EditRow item={item} onSave={(text) => save(item.id, text)} onCancel={() => cancel(item.id)} />
            ) : (
              <>
                <button
                  type="button"
                  className={styles.text}
                  title={item.text}
                  aria-label={`${CHAT.queue.edit}: ${item.text}`}
                  onClick={() => props.onEditingChange(item.id)}
                >
                  {item.text}
                </button>
                {item.plan === true ? <span className={styles.tag}>{CHAT.queue.plan}</span> : null}
                <Files item={item} />
                {props.sendNow !== undefined ? <SendNowButton item={item} control={props.sendNow} /> : null}
                <IconButton
                  ref={bind(editButtons, item.id)}
                  label={CHAT.queue.edit}
                  icon={<Pencil size={14} strokeWidth={1.75} />}
                  onClick={() => props.onEditingChange(item.id)}
                />
                <IconButton
                  ref={bind(removeButtons, item.id)}
                  label={CHAT.queue.remove}
                  icon={<X size={14} strokeWidth={1.75} />}
                  onClick={() => remove(index)}
                />
              </>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}

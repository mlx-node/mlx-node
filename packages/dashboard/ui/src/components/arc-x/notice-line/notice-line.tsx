import { CircleAlert, TriangleAlert, X } from 'lucide-react';
import type { ReactElement, ReactNode } from 'react';

import { Button } from '../../arc/button/button';
import { Popover, PopoverContent, PopoverTrigger } from '../../arc/popover/popover';
import { IconButton } from '../icon-button/icon-button';
import styles from './notice-line.module.css';

export interface NoticeLineProps {
  tone: 'warning' | 'danger';
  /** One sentence; cut at the line's end, whole in the hover title. */
  text: string;
  /** Closes the notice; absent when it cannot be closed. */
  onClose?: { label: string; run: () => void };
  /** Before the close button: the "+n" of the notices behind this one. */
  more?: ReactNode;
}

/** A one-shot fact as one closable line: icon, sentence, the others behind "+n", close. */
export function NoticeLine({ tone, text, onClose, more }: NoticeLineProps): ReactElement {
  const Icon = tone === 'danger' ? CircleAlert : TriangleAlert;
  return (
    <div className={styles.root} data-tone={tone} role="status" data-notice-line="">
      <Icon className={styles.icon} size={14} strokeWidth={1.75} aria-hidden="true" />
      <span className={styles.text} title={text}>
        {text}
      </span>
      {more}
      {onClose !== undefined ? (
        <IconButton label={onClose.label} icon={<X size={14} strokeWidth={1.75} />} onClick={onClose.run} />
      ) : null}
    </div>
  );
}

/** "+n": a popover that lists the notices behind the one shown, each as a line. */
export function NoticeMore(props: {
  /** The button's text, e.g. "+2". */
  text: string;
  /** Its name, e.g. "Show 2 more notices". */
  label: string;
  notices: ReadonlyArray<{ key: string; tone: 'warning' | 'danger'; text: string }>;
}): ReactElement {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button type="button" variant="ghost" size="sm" aria-label={props.label} className={styles.more}>
          {props.text}
        </Button>
      </PopoverTrigger>
      <PopoverContent side="top" align="end" aria-label={props.label}>
        <ul className={styles.list}>
          {props.notices.map((notice) => (
            <li key={notice.key}>
              <NoticeLine tone={notice.tone} text={notice.text} />
            </li>
          ))}
        </ul>
      </PopoverContent>
    </Popover>
  );
}

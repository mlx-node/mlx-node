import type { ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import { useHomeDir } from '../../../lib/home-dir';
import { truncateMiddle } from '../../../lib/labels';
import { tildePath } from '../../../lib/paths';
import styles from './path-text.module.css';

export interface PathTextProps {
  /** An absolute path; the hover title always shows it in full. */
  path: string;
  /** Cut in the middle past this many characters, after the home folder became `~`. */
  maxChars?: number;
  /** Hover text, when a resolved form of the same path says more. Default `path`. */
  title?: string;
  /** Code font. Default true. */
  mono?: boolean;
  /** Break onto more lines at any character instead of giving way with an ellipsis: the one place a path shows whole. */
  wrap?: boolean;
  /** Give way in the folders above, never in the last name: a row that names a folder. */
  keepEnd?: boolean;
  className?: string;
}

/** A filesystem path as the user reads it: the home folder as `~`, the full path on hover. */
export function PathText({
  path,
  maxChars,
  title,
  mono = true,
  wrap = false,
  keepEnd = false,
  className,
}: PathTextProps): ReactElement {
  const short = tildePath(path, useHomeDir());
  const cut = keepEnd ? short.lastIndexOf('/', short.length - 2) : -1;
  if (cut > 0) {
    return (
      <span
        className={cn(styles.root, className)}
        data-mono={mono ? '' : undefined}
        data-keep-end=""
        title={title ?? path}
      >
        <span className={styles.head}>{short.slice(0, cut)}</span>
        <span className={styles.tail}>{short.slice(cut)}</span>
      </span>
    );
  }
  return (
    <span
      className={cn(styles.root, className)}
      data-mono={mono ? '' : undefined}
      data-wrap={wrap ? '' : undefined}
      title={title ?? path}
    >
      {maxChars === undefined ? short : truncateMiddle(short, maxChars)}
    </span>
  );
}

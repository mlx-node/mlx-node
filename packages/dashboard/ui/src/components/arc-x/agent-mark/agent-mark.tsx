import { Terminal } from 'lucide-react';
import type { ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import styles from './agent-mark.module.css';

export interface AgentMarkProps {
  id: 'claude' | 'codex' | 'grok';
  size?: 20 | 28;
  className?: string;
}

const RAYS = Array.from({ length: 12 }, (_, index) => index * 15);

/** A coding agent's mark, monochrome in the text color. */
export function AgentMark({ id, size = 28, className }: AgentMarkProps): ReactElement {
  if (id === 'codex') {
    return (
      <span className={cn(styles.root, styles.boxed, className)} data-size={size} data-agent={id} aria-hidden="true">
        <Terminal className={styles.terminal} strokeWidth={1.75} />
      </span>
    );
  }
  return (
    <span className={cn(styles.root, className)} data-size={size} data-agent={id} aria-hidden="true">
      <svg viewBox="0 0 40 40" fill="none" stroke="currentColor" strokeWidth={2.4} strokeLinecap="round">
        {id === 'claude' ? (
          RAYS.map((angle) => <path key={angle} d="M20 4v32" transform={`rotate(${angle} 20 20)`} />)
        ) : (
          <path d="M30 9a14 14 0 1 0 2 19M8 34 34 5M19 22l14-13-8 17" strokeLinejoin="round" />
        )}
      </svg>
    </span>
  );
}

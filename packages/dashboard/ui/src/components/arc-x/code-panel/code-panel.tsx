import { animate } from 'motion/react';
import { useLayoutEffect, useRef, useState, type CSSProperties, type ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import { CopyButton } from '../../arc/copy-button/copy-button';
import { motionX } from '../motion/motion';
import { useStaticMotion } from '../motion/use-static-motion';
import styles from './code-panel.module.css';

export interface CodePanelProps {
  code: string;
  /** Names the block for assistive tech and the copy button. */
  label?: string;
  /** Show this many lines, then a "Show n more lines" toggle. */
  maxLines?: number;
  /** Wrap long lines instead of scrolling sideways. */
  wrap?: boolean;
  className?: string;
}

function lineCount(code: string): number {
  let lines = 1;
  for (let i = 0; i < code.length; i += 1) if (code[i] === '\n') lines += 1;
  return code.endsWith('\n') ? lines - 1 : lines;
}

/** A mono well with a copy button; no syntax colors. */
export function CodePanel({ code, label, maxLines, wrap = false, className }: CodePanelProps): ReactElement {
  const staticMotion = useStaticMotion();
  const total = lineCount(code);
  const clamps = maxLines !== undefined && maxLines > 0 && total > maxLines;
  const [expanded, setExpanded] = useState(false);
  const clipRef = useRef<HTMLDivElement | null>(null);
  const fromHeight = useRef<number | null>(null);

  useLayoutEffect(() => {
    const clip = clipRef.current;
    const from = fromHeight.current;
    fromHeight.current = null;
    if (clip === null || from === null || staticMotion) return;
    const to = clip.getBoundingClientRect().height;
    if (from === to) return;
    clip.style.maxHeight = 'none';
    const controls = animate(clip, { height: [from, to] }, motionX.spring.expand);
    const settle = (): void => {
      clip.style.height = '';
      clip.style.maxHeight = '';
    };
    void Promise.resolve(controls).then(settle);
    return () => {
      controls.stop();
      settle();
    };
  }, [expanded, staticMotion]);

  const toggle = (): void => {
    fromHeight.current = clipRef.current?.getBoundingClientRect().height ?? null;
    setExpanded((value) => !value);
  };

  const collapsed = clamps && !expanded;
  const clipStyle = collapsed ? ({ '--lines': maxLines } as CSSProperties) : undefined;
  const hidden = clamps ? total - (maxLines ?? 0) : 0;

  return (
    <figure className={cn(styles.root, className)} aria-label={label}>
      <div ref={clipRef} className={styles.clip} data-collapsed={collapsed ? '' : undefined} style={clipStyle}>
        <pre className={styles.code} data-wrap={wrap ? '' : undefined}>
          <code>{code}</code>
        </pre>
      </div>
      <CopyButton
        value={code}
        label={label === undefined ? 'Copy' : `Copy ${label}`}
        iconOnly
        variant="plain"
        className={styles.copy}
      />
      {clamps ? (
        <button type="button" className={styles.more} aria-expanded={expanded} onClick={toggle}>
          {expanded ? 'Show fewer lines' : `Show ${hidden} more ${hidden === 1 ? 'line' : 'lines'}`}
        </button>
      ) : null}
    </figure>
  );
}

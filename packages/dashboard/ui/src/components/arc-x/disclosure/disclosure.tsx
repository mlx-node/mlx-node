import { ChevronRight } from 'lucide-react';
import { motion } from 'motion/react';
import { useId, useState, type ReactElement, type ReactNode } from 'react';

import { cn } from '../../../lib/cn';
import { motionX } from '../motion/motion';
import { useStaticMotion } from '../motion/use-static-motion';
import styles from './disclosure.module.css';

export interface DisclosureProps {
  title: ReactNode;
  /** Muted text at the end of the header (a count, a size). */
  meta?: ReactNode;
  defaultOpen?: boolean;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  /** Children mount on first open and stay mounted after. Default true. */
  lazy?: boolean;
  children: ReactNode;
  className?: string;
}

const INSTANT = { duration: 0 } as const;

/** A header button that opens a region below it; the height opens on `expand`, closing clips the content. */
export function Disclosure(props: DisclosureProps): ReactElement {
  const staticMotion = useStaticMotion();
  const buttonId = useId();
  const regionId = useId();
  const [uncontrolled, setUncontrolled] = useState(props.defaultOpen === true);
  const open = props.open ?? uncontrolled;
  const [everOpen, setEverOpen] = useState(open);
  if (open && !everOpen) setEverOpen(true);
  const [closedAtRest, setClosedAtRest] = useState(!open);
  if (open && closedAtRest) setClosedAtRest(false);
  const [toggled, setToggled] = useState(false);

  const toggle = (): void => {
    const next = !open;
    if (props.open === undefined) setUncontrolled(next);
    setToggled(true);
    props.onOpenChange?.(next);
  };

  const lazy = props.lazy !== false;
  const mountChildren = !lazy || everOpen;
  const hidden = !open && (staticMotion || closedAtRest);

  return (
    <div className={cn(styles.root, props.className)} data-state={open ? 'open' : 'closed'}>
      <button
        id={buttonId}
        type="button"
        className={styles.header}
        aria-expanded={open}
        aria-controls={regionId}
        onClick={toggle}
      >
        <ChevronRight className={styles.chevron} size={14} strokeWidth={1.75} aria-hidden="true" />
        <span className={styles.title}>{props.title}</span>
        {props.meta !== undefined ? <span className={styles.meta}>{props.meta}</span> : null}
      </button>
      <motion.div
        id={regionId}
        role="region"
        aria-labelledby={buttonId}
        className={styles.region}
        hidden={hidden}
        initial={false}
        animate={{ height: open ? 'auto' : 0 }}
        transition={staticMotion ? INSTANT : motionX.spring.expand}
        onAnimationComplete={() => {
          if (!open) setClosedAtRest(true);
        }}
      >
        <div className={styles.content} data-enter={open && toggled ? '' : undefined}>
          {mountChildren ? props.children : null}
        </div>
      </motion.div>
    </div>
  );
}

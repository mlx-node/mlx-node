import {
  animate,
  motion,
  useInView,
  useMotionValue,
  useMotionValueEvent,
  usePageInView,
  useTransform,
  type AnimationPlaybackControls,
} from 'motion/react';
import { useEffect, useMemo, useRef, useState, type ReactElement } from 'react';

import { cn } from '../../../lib/cn';
import { motionTokens } from '../../arc/motion-tokens';
import { motionX } from '../motion/motion';
import { useStaticMotion } from '../motion/use-static-motion';
import styles from './morph-loader.module.css';
import { LOADER_SHAPES, mixSegment, segmentPath, turnShape, type LoaderStatus, type Segment } from './shapes';

export interface MorphLoaderProps {
  status: LoaderStatus;
  /** px. Default 14. */
  size?: 14 | 16 | 20;
  /** Accessible name. Omit when a word next to it says the status (then the glyph is aria-hidden). */
  label?: string;
  /** 'loading' only. False: the arcs stand still (a record of running work, not work seen right now). Default true. */
  spinning?: boolean;
  /** The caller may set `color` for pending / loading; success and error keep their tones. */
  className?: string;
}

const QUARTER = 90;

/** One stroke that turns from its last shape into `target` on the morph spring, `delay` seconds after the first. */
function MorphStroke({ target, delay }: { target: Segment; delay: number }): ReactElement {
  const progress = useMotionValue(1);
  const from = useRef(target);
  const to = useRef(target);
  const d = useTransform(progress, (t) => segmentPath(mixSegment(from.current, to.current, t)));

  useEffect(() => {
    if (to.current === target) return;
    from.current = mixSegment(from.current, to.current, progress.get());
    to.current = target;
    progress.jump(0);
    const controls = animate(progress, 1, { ...motionX.spring.morph, delay });
    return () => controls.stop();
  }, [target, delay, progress]);

  return <motion.path d={d} />;
}

/** The ring that ends in a check or a cross. Always four strokes, so the shapes morph instead of swapping. */
export function MorphLoader({ status, size = 14, label, spinning = true, className }: MorphLoaderProps): ReactElement {
  const staticMotion = useStaticMotion();
  const svgRef = useRef<SVGSVGElement | null>(null);
  const groupRef = useRef<SVGGElement | null>(null);
  const inView = useInView(svgRef);
  const pageVisible = usePageInView();
  const angle = useMotionValue(0);
  const [quarters, setQuarters] = useState(0);
  const turning = !staticMotion && status === 'loading' && spinning && inView && pageVisible;

  useMotionValueEvent(angle, 'change', (deg) => {
    groupRef.current?.setAttribute('transform', `rotate(${deg} 12 12)`);
  });

  useEffect(() => {
    if (staticMotion) return;
    let controls: AnimationPlaybackControls | undefined;
    if (turning) {
      const start = angle.get();
      controls = animate(angle, start + 360, { duration: motionX.spin, ease: 'linear', repeat: Infinity });
    } else if (status !== 'loading') {
      const current = angle.get();
      const rest = Math.ceil(current / QUARTER) * QUARTER;
      setQuarters(Math.round(rest / QUARTER));
      if (rest !== current) controls = animate(angle, rest, motionX.spring.morph);
    }
    return () => controls?.stop();
  }, [turning, status, staticMotion, angle]);

  const targets = useMemo(() => turnShape(LOADER_SHAPES[status], quarters), [status, quarters]);
  const labelled = label !== undefined && label.length > 0;

  return (
    <svg
      ref={svgRef}
      className={cn(styles.root, className)}
      viewBox="0 0 24 24"
      width={size}
      height={size}
      data-status={status}
      role={labelled ? 'img' : undefined}
      aria-label={labelled ? label : undefined}
      aria-hidden={labelled ? undefined : true}
    >
      {staticMotion ? (
        <g>
          {LOADER_SHAPES[status].map((segment, index) => (
            <path key={index} d={segmentPath(segment)} />
          ))}
        </g>
      ) : (
        <g ref={groupRef}>
          {targets.map((segment, index) => (
            <MorphStroke key={index} target={segment} delay={index * motionTokens.stagger.item} />
          ))}
        </g>
      )}
    </svg>
  );
}

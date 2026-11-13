import { useMotionValueEvent, type MotionValue } from 'motion/react';
import { useLayoutEffect, useRef, type CSSProperties, type ReactElement } from 'react';

import { deltaAt, rectDelta, type Rect } from './morph';

export interface ReshapeBarProps {
  d: string;
  rect: Rect;
  /** Where the bar is drawn from when it mounts after the chart did (a zero-height rect on the baseline). */
  bornFrom?: Rect;
  signature: string;
  progress: MotionValue<number>;
  className?: string;
  style?: CSSProperties;
}

function transformFor(from: Rect | null, to: Rect, t: number): string | null {
  const delta = rectDelta(from, to);
  if (delta === null || t >= 1) return null;
  const k = deltaAt(delta, t);
  if (k.dx === 0 && k.dy === 0 && k.sx === 1 && k.sy === 1) return null;
  return `translate(${String(to.x + k.dx)} ${String(to.y + k.dy)}) scale(${String(k.sx)} ${String(k.sy)}) translate(${String(-to.x)} ${String(-to.y)})`;
}

/** A bar drawn at its new rectangle, moved there from its old one by a transform on the chart's progress value. */
export function ReshapeBar(props: ReshapeBarProps): ReactElement {
  const ref = useRef<SVGPathElement | null>(null);
  const from = useRef<Rect | null>(props.bornFrom ?? props.rect);
  const to = useRef<Rect>(props.rect);
  const lastSignature = useRef(props.signature);
  const { progress } = props;

  useLayoutEffect(() => {
    if (lastSignature.current !== props.signature) {
      lastSignature.current = props.signature;
      from.current = to.current;
    }
    to.current = props.rect;
    const node = ref.current;
    if (node === null) return;
    const transform = transformFor(from.current, to.current, progress.get());
    if (transform === null) node.removeAttribute('transform');
    else node.setAttribute('transform', transform);
  });

  useMotionValueEvent(progress, 'change', (t) => {
    const node = ref.current;
    if (node === null) return;
    const transform = transformFor(from.current, to.current, t);
    if (transform === null) node.removeAttribute('transform');
    else node.setAttribute('transform', transform);
  });

  return <path ref={ref} d={props.d} className={props.className} style={props.style} />;
}

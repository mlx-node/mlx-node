import { motion, useMotionValue, useMotionValueEvent, type MotionValue } from 'motion/react';
import { useLayoutEffect, useRef, type CSSProperties, type ReactElement } from 'react';

import { mixSamples } from './morph';

export interface MorphPathProps {
  /** The exact path of the new data, drawn whenever nothing moves. */
  exact: string;
  /** The new data resampled to `motionX.morphSamples` pixel ys (null = gap). */
  samples: readonly (number | null)[];
  /** Turns resampled pixel ys into a path (the caller knows the x positions and the curve). */
  toPath: (samples: readonly (number | null)[]) => string;
  /** Changes exactly when the data changes; a new value takes the line from where it is now to `samples`. */
  signature: string;
  progress: MotionValue<number>;
  className?: string;
  style?: CSSProperties;
  'data-entrance-last'?: '';
}

/**
 * A path that reshapes from the line on screen to new data on the chart's one progress value, writing `d` per frame
 * without a React render. A gap in the new data is a gap from the first frame.
 */
export function MorphPath(props: MorphPathProps): ReactElement {
  const { exact, samples, toPath, signature, progress } = props;
  const d = useMotionValue(exact);
  const from = useRef<readonly (number | null)[]>(samples);
  const to = useRef<readonly (number | null)[]>(samples);
  const shown = useRef<readonly (number | null)[]>(samples);
  const latest = useRef({ exact, toPath });
  const lastSignature = useRef(signature);

  useLayoutEffect(() => {
    latest.current = { exact, toPath };
    if (lastSignature.current !== signature) {
      lastSignature.current = signature;
      from.current = shown.current;
      to.current = samples;
    }
    if (progress.get() >= 1) {
      shown.current = samples;
      d.set(exact);
    }
  });

  useMotionValueEvent(progress, 'change', (t) => {
    if (t >= 1) {
      shown.current = to.current;
      d.set(latest.current.exact);
      return;
    }
    const mixed = mixSamples(from.current, to.current, t);
    shown.current = mixed;
    d.set(latest.current.toPath(mixed));
  });

  return (
    <motion.path
      d={d}
      className={props.className}
      style={props.style}
      data-entrance-last={props['data-entrance-last']}
    />
  );
}

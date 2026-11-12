import { animate, useMotionValue, type MotionValue } from 'motion/react';
import { useEffect, useLayoutEffect, useRef } from 'react';

import { motionX } from '../motion/motion';
import { useRichMotion } from '../motion/use-entrance';

/**
 * One progress value per chart: set to 0 and sprung to 1 on `reshape` each time `signature` changes (series keys,
 * column count and formatted values, so equal data moves nothing). Stays at 1 under the static branch.
 */
export function useReshape(signature: string): MotionValue<number> {
  const rich = useRichMotion();
  const progress = useMotionValue(1);
  const last = useRef(signature);
  const controls = useRef<{ stop(): void } | null>(null);

  useLayoutEffect(() => {
    if (last.current === signature) return;
    last.current = signature;
    controls.current?.stop();
    controls.current = null;
    if (!rich) {
      progress.jump(1);
      return;
    }
    progress.jump(0);
    controls.current = animate(progress, 1, motionX.spring.reshape);
  }, [signature, rich, progress]);

  useEffect(
    () => () => {
      controls.current?.stop();
    },
    [],
  );

  return progress;
}

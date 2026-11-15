import { MotionGlobalConfig, useReducedMotion } from 'motion/react';

/** True when nothing may travel: the OS asks for reduced motion, or a harness set MotionGlobalConfig.skipAnimations. */
export function useStaticMotion(): boolean {
  const reduced = useReducedMotion();
  return reduced === true || MotionGlobalConfig.skipAnimations === true;
}

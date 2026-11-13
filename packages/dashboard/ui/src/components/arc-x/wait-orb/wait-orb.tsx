import { useInView, usePageInView } from 'motion/react';
import { useEffect, useRef, type ReactElement } from 'react';

import { useTheme } from '../../../theme/theme';
import { motionTokens } from '../../arc/motion-tokens';
import { motionX } from '../motion/motion';
import { useStaticMotion } from '../motion/use-static-motion';
import { ORB_SIZE, orbFrame, TAU, type OrbShape } from './engine';

export type WaitOrbState = 'hold' | 'load' | 'think';

export interface WaitOrbProps {
  /** Which wait this is: the turn waits for something else, the engine loads the model, the model works. */
  state: WaitOrbState;
  /** Accessible name. Omit when the status text stands next to it (then the orb is aria-hidden). */
  label?: string;
}

/** `pace` multiplies the clock, `ink` every dot's alpha. */
const LOOK: Record<WaitOrbState, { shape: OrbShape; pace: number; ink: number }> = {
  hold: { shape: 'ring', pace: motionX.orb.holdPace, ink: 0.7 },
  load: { shape: 'wave', pace: 1, ink: 1 },
  think: { shape: 'band', pace: 1, ink: 1 },
};

/**
 * The dotted orb of a wait: its shape says which wait. Decoration; the status is text beside it. One canvas, drawn
 * `motionX.orb.rate` times per second only while the page is visible, the orb is in view and motion is allowed.
 */
export function WaitOrb({ state, label }: WaitOrbProps): ReactElement {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const shown = useRef(state);
  const staticMotion = useStaticMotion();
  const inView = useInView(ref);
  const pageVisible = usePageInView();
  const theme = useTheme().resolved;
  const running = !staticMotion && inView && pageVisible;

  useEffect(() => {
    const canvas = ref.current;
    // CPU-backed: the GPU process gets one 40 px upload per draw instead of every arc's raster commands.
    const ctx = canvas?.getContext('2d', { willReadFrequently: true }) ?? null;
    if (canvas === null || ctx === null) return;
    const dpr = Math.min(2, window.devicePixelRatio);
    const side = Math.round(ORB_SIZE * dpr);
    // Setting the size clears the canvas: only when it changed, so a state change keeps the last frame up.
    if (canvas.width !== side) {
      canvas.width = side;
      canvas.height = side;
    }
    const ink = getComputedStyle(canvas).color;
    const draw = (seconds: number | null, strength: number): void => {
      const look = LOOK[shown.current];
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, ORB_SIZE, ORB_SIZE);
      ctx.fillStyle = ink;
      for (const dot of orbFrame(look.shape, seconds === null ? motionX.orb.still : seconds * look.pace)) {
        ctx.globalAlpha = dot.a * (1 - dot.white) * look.ink * strength;
        ctx.beginPath();
        ctx.arc(dot.x, dot.y, dot.r, 0, TAU);
        ctx.fill();
      }
    };
    if (!running) {
      shown.current = state;
      draw(null, 1);
      return;
    }
    const out = motionTokens.duration.instant * 1000;
    const back = motionTokens.duration.fast * 1000;
    let swapAt = shown.current === state ? null : performance.now();
    let tick = -1;
    let frame = 0;
    const loop = (now: number): void => {
      frame = requestAnimationFrame(loop);
      const next = Math.floor((now * motionX.orb.rate) / 1000);
      if (next === tick) return;
      tick = next;
      let strength = 1;
      if (swapAt !== null) {
        const since = now - swapAt;
        if (since >= out) shown.current = state;
        strength = since < out ? 1 - since / out : Math.min(1, (since - out) / back);
        if (since >= out + back) swapAt = null;
      }
      draw(now / 1000, strength);
    };
    frame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frame);
  }, [state, theme, running]);

  const named = label !== undefined;
  return (
    <canvas
      ref={ref}
      style={{ display: 'block', flex: 'none', width: ORB_SIZE, height: ORB_SIZE, color: 'var(--foreground)' }}
      data-wait-orb={state}
      role={named ? 'img' : undefined}
      aria-label={label}
      aria-hidden={named ? undefined : true}
    />
  );
}

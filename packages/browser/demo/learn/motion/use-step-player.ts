import * as React from 'react';

import { usePrefersReducedMotion } from './use-prefers-reduced-motion';

export type StepPlayer = {
  /** current frame index, always in [0, total) */
  frame: number;
  /** true while the interval is advancing frames */
  playing: boolean;
  /** true when the user asked for less motion — hide controls, show a still */
  reducedMotion: boolean;
  /** toggle play/pause (no-op under reduced motion) */
  toggle: () => void;
  /** advance one frame; always pauses first, so "Step" means step */
  step: () => void;
  /** restart the sweep — or, under reduced motion, re-pin the resting frame */
  reset: () => void;
  /** jump to a frame; always pauses (used by chapter markers / scrubbing) */
  goTo: (frame: number) => void;
};

/**
 * useStepPlayer — the frame-stepper this course hand-rolls in ~32 widgets:
 * an integer `frame` advanced by a `setInterval`, with play/pause/step and a
 * reduced-motion contract. Extracted so new widgets get the contract right by
 * construction instead of by copy-paste.
 *
 * THE SSR CONTRACT (the part that is easy to get wrong):
 *
 * The first render is deterministic and identical on server and client — it is
 * always `initialFrame`, never a frame chosen from `matchMedia`. Retargeting to
 * `restFrame` for reduced-motion users happens in an EFFECT, after mount. Some
 * older widgets read the preference straight in the useState initializer; that
 * only survives because boot is `createRoot` (no hydration diff check) rather
 * than `hydrateRoot`. Do not rely on that here.
 *
 * The same effect also handles the preference flipping mid-session, which the
 * useState-initializer approach silently cannot (initializers run once).
 *
 * @param total       number of frames; the sweep wraps modulo this
 * @param opts.frameMs        ms per frame (default 1500, the course's usual pace)
 * @param opts.initialFrame   first render's frame, server and client (default 0)
 * @param opts.restFrame      frame shown to reduced-motion users. Default is the
 *                            LAST frame: for an explainer that builds up state,
 *                            the finished picture is the informative still.
 * @param opts.loop           when false, the sweep stops on the last frame
 *                            instead of wrapping (default true)
 * @param opts.initialPlaying whether the sweep runs on its own from mount
 *                            (default true). Pass `false` for a widget whose
 *                            payoff is the ANSWER — a capstone that autoplays
 *                            hands the reader the ending before they have read
 *                            the setup, and a non-looping one then parks there
 *                            forever, so the beginning is never seen at all.
 */
export function useStepPlayer(
  total: number,
  opts: {
    frameMs?: number;
    initialFrame?: number;
    restFrame?: number;
    loop?: boolean;
    initialPlaying?: boolean;
  } = {},
): StepPlayer {
  const { frameMs = 1500, initialFrame = 0, loop = true, initialPlaying = true } = opts;
  const restFrame = opts.restFrame ?? total - 1;

  const reducedMotion = usePrefersReducedMotion();

  // Deterministic on the server: a fixed frame, never a media query.
  const [frame, setFrame] = React.useState(initialFrame);
  const [playing, setPlaying] = React.useState(initialPlaying);

  // Retarget AFTER mount. Also fires if the preference flips mid-session.
  React.useEffect(() => {
    if (!reducedMotion) return;
    setPlaying(false);
    setFrame(restFrame);
  }, [reducedMotion, restFrame]);

  React.useEffect(() => {
    if (!playing || reducedMotion) return;
    // The updater must stay PURE — no setPlaying inside it. React may invoke an
    // updater more than once (StrictMode double-invokes to surface exactly this
    // kind of hidden side effect), so parking a non-looping sweep is handled by
    // the separate effect below, driven by the frame value itself.
    const t = window.setInterval(() => {
      setFrame((f) => (f + 1 < total ? f + 1 : loop ? 0 : total - 1));
    }, frameMs);
    return () => window.clearInterval(t);
  }, [playing, reducedMotion, total, frameMs, loop]);

  // Non-looping sweeps stop when they reach the end. Separate from the interval
  // so the updater above has no side effects.
  React.useEffect(() => {
    if (loop || reducedMotion) return;
    if (frame >= total - 1) setPlaying(false);
  }, [loop, reducedMotion, frame, total]);

  const toggle = React.useCallback(() => {
    if (reducedMotion) return;
    // Pressing play on a non-looping sweep that is already parked on the last
    // frame would otherwise start an interval that can never advance — the
    // button flips to "Pause" and nothing moves. Replay from the top instead.
    // `initialFrame` itself can be the last frame, in which case rewinding to it
    // is a no-op and the dead end survives; fall back to 0 so replay always
    // actually moves.
    if (!playing && !loop && frame >= total - 1) setFrame(initialFrame < total - 1 ? initialFrame : 0);
    setPlaying(!playing);
  }, [reducedMotion, playing, loop, frame, total, initialFrame]);

  const step = React.useCallback(() => {
    setPlaying(false);
    // Same end-of-sweep rule as the interval: a non-looping sweep stops on the
    // last frame. Wrapping here would let "Step" teleport a reader who is
    // reading the final state back to the beginning, which no caption expects.
    setFrame((f) => (f + 1 < total ? f + 1 : loop ? 0 : total - 1));
  }, [total, loop]);

  const goTo = React.useCallback(
    (f: number) => {
      setPlaying(false);
      // Clamp rather than wrap: a caller passing an out-of-range marker should
      // land at an end, not silently teleport to the other side of the sweep.
      setFrame(Math.max(0, Math.min(total - 1, f)));
    },
    [total],
  );

  const reset = React.useCallback(() => {
    setFrame(reducedMotion ? restFrame : initialFrame);
    // "Back to how it started" — including whether it started moving. A widget
    // that opts out of autoplay wants Reset to rewind, not to launch the sweep
    // it deliberately kept parked.
    setPlaying(!reducedMotion && initialPlaying);
  }, [reducedMotion, restFrame, initialFrame, initialPlaying]);

  return { frame, playing, reducedMotion, toggle, step, reset, goTo };
}

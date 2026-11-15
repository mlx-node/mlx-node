import { motionTokens } from '../../arc/motion-tokens';

export const motionX = {
  spring: {
    /** Heights that open and close, widths that follow new text, list rows that enter. */
    expand: motionTokens.spring.snappy,
    /** Sidebar width, shared highlights, tool rows. Slower than any Arc free preset; measured on the reference. */
    glide: { type: 'spring', visualDuration: 0.48, bounce: 0 },
    /** A surface that changes shape where it stands: trigger -> confirm panel -> success pill, loader strokes. */
    morph: motionTokens.spring.morph,
    /** A surface that crosses the page while it resizes: KPI tile -> detail panel, rows that re-sort. Measured on the reference. */
    travel: { type: 'spring', visualDuration: 0.6, bounce: 0 },
    /** Chart marks that change to new data: path samples, bar rectangles, share bars. */
    reshape: motionTokens.spring.smooth,
  },
  swap: {
    out: { duration: motionTokens.duration.instant, ease: motionTokens.ease.standard },
    in: { duration: motionTokens.duration.fast, ease: motionTokens.ease.enter },
    /** px the new whole text rises. */
    rise: 2,
    /** px one character travels, out and in. */
    shift: 4,
  },
  reveal: {
    duration: motionTokens.duration.standard,
    ease: motionTokens.ease.enter,
    /** px the arriving content rises. */
    rise: 4,
    step: motionTokens.stagger.item,
    maxSteps: 8,
  },
  page: {
    out: {
      duration: motionTokens.duration.exit,
      ease: motionTokens.ease.standard,
      y: -6,
      blur: motionTokens.blur.subtle,
    },
    in: { duration: motionTokens.duration.standard, ease: motionTokens.ease.enter, y: 6 },
  },
  /** A modal layer (the command palette): arc/dialog's rise and scale, in and out. */
  layer: {
    in: { y: 8, scale: 0.96 },
    out: { y: 4, scale: 0.98 },
  },
  /** Seconds a staggered entrance may take in total. A step is min(nominal step, budget / count). */
  staggerBudget: motionTokens.duration.considered,
  /** Seconds a changed chip keeps its tint before it drains (flash). */
  flashHold: 0.43,
  /** Seconds a success face stays before its surface returns to rest. 0 under useStaticMotion(). */
  successHold: 0.9,
  /** Ticker drift, px per second. */
  drift: 32,
  /** Seconds the pointer rests on an item before its hover panel opens. Equals --duration-hold. */
  hoverHold: 0.14,
  /** Seconds per turn of a JS-driven ring (MorphLoader). Equals --duration-spin. */
  spin: 0.8,
  orb: {
    /** Draws per second of a WaitOrb canvas. */
    rate: 30,
    /** Seconds on the orb's clock of the still frame (static motion, a stopped loop). */
    still: 0.6,
    /** Clock multiplier of the `hold` orb. */
    holdPace: 0.45,
  },
  /** x positions both paths are resampled to before a line reshape. */
  morphSamples: 48,
} as const;

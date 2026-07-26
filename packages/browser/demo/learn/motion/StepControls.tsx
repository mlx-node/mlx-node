import * as React from 'react';

import type { Locale } from '../../lib/i18n';
import type { StepPlayer } from './use-step-player';

/**
 * StepControls — the play / step / reset button row this course repeats in ~32
 * widgets, extracted verbatim in look and behaviour so a new animated widget is
 * indistinguishable from the existing ones.
 *
 * Two conventions worth stating because they are deliberate, not accidental:
 *
 *  - Under reduced motion the PLAY button disappears but Step and Reset stay.
 *    The preference asks to drop non-essential motion, not to drop content or
 *    navigation — and stepping is an instantaneous state change, not motion.
 *    Hiding the whole row would strand these readers on the resting frame with
 *    no way to reach the intermediate states, which for a build-up explainer is
 *    where the actual lesson lives. Same call TokenJourney makes.
 *  - "Step" always pauses first (see useStepPlayer.step). Stepping while the
 *    interval keeps running is the single most confusing thing these controls
 *    can do.
 *
 * Tap targets are >= 24px tall via `py-1.5` + `text-xs` so the row satisfies
 * WCAG 2.5.8 on a phone, where the widget's SVG is scaled well below its
 * viewBox size.
 */

const COPY = {
  en: { play: 'Play', pause: 'Pause', step: 'Step', reset: 'Reset' },
  // Glossary-free UI verbs — safe to translate.
  zh: { play: '播放', pause: '暂停', step: '单步', reset: '重置' },
} as const satisfies Record<Locale, { play: string; pause: string; step: string; reset: string }>;

// `py-1.5` at `text-xs` is a 28px button — fine for a mouse, under the course's
// 44px touch floor. The `max-sm:` bump is the same one the unskinned widgets
// (FlashTrafficDiagram, Fp8QuantDiagram, BaseVsInstructDiagram, …) already
// carry; `inline-flex items-center` keeps the label centred once min-height
// exceeds the line box. This is the play/step control of every skinned widget,
// so the floor has to live here rather than in sixteen call sites.
const BTN =
  'inline-flex items-center rounded px-2.5 py-1.5 max-sm:min-h-[44px] max-sm:px-3 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary';

export function StepControls({
  player,
  locale,
  showReset = false,
}: {
  player: StepPlayer;
  locale: Locale;
  /** Show a Reset button. Worth it for long sweeps; noise for short loops. */
  showReset?: boolean;
}) {
  const copy = COPY[locale];

  return (
    <div className="flex items-center gap-1">
      {player.reducedMotion ? null : (
        <button type="button" onClick={player.toggle} aria-pressed={player.playing} className={BTN}>
          {player.playing ? copy.pause : copy.play}
        </button>
      )}
      <button type="button" onClick={player.step} className={BTN}>
        {copy.step}
      </button>
      {showReset || player.reducedMotion ? (
        <button type="button" onClick={player.reset} className={BTN}>
          {copy.reset}
        </button>
      ) : null}
    </div>
  );
}

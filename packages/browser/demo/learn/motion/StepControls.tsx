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
 *  - Under reduced motion the controls RENDER NOTHING. A reader who asked for
 *    less motion is shown the resting frame as a still diagram; offering them a
 *    Play button would be offering them the thing they opted out of.
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

const BTN =
  'rounded px-2.5 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary';

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
  if (player.reducedMotion) return null;

  return (
    <div className="flex items-center gap-1">
      <button type="button" onClick={player.toggle} aria-pressed={player.playing} className={BTN}>
        {player.playing ? copy.pause : copy.play}
      </button>
      <button type="button" onClick={player.step} className={BTN}>
        {copy.step}
      </button>
      {showReset ? (
        <button type="button" onClick={player.reset} className={BTN}>
          {copy.reset}
        </button>
      ) : null}
    </div>
  );
}

import { CHAPTERS } from '../../learn/chapters';
import { useUiStrings } from '../../learn/i18n/ui-react';
import type { ModelBlockReason } from '../../lib/device-capability';
import { useLocale } from '../../lib/i18n-react';
import { modelBlockedShort } from '../../lib/model-blocked-copy';
import { LanguageSwitcher } from '../LanguageSwitcher';
import { DriftingLibrary } from './DriftingLibrary';

export type LandingProps = {
  onLoad: () => void;
  onLocalModel: () => void;
  onStartLearning: () => void;
  onOpenJSpace: () => void;
  errorBanner: string | null;
  hostedModelAvailable?: boolean | null;
  loadDisabled?: boolean;
  /** Non-null when this device can't run the in-browser model (iOS Safari,
   *  low-RAM, no WebGPU). We disable the "Load model" CTA and show a short
   *  "open on desktop" note instead of a button that silently no-ops. */
  modelBlockReason?: ModelBlockReason;
  /** When true, the model is already loaded in the worker. We flip the
   *  primary CTA label to "Open Chat →" so a returning user (or anyone
   *  who already loaded the model on a previous visit and got cached
   *  weights) sees an obvious next action instead of a "Load Model"
   *  button that silently no-ops because there's nothing to load. */
  modelReady?: boolean;
};

const MODEL_LABEL = '0.8B';

export function Landing({
  onLoad,
  onLocalModel,
  onStartLearning,
  onOpenJSpace,
  errorBanner,
  hostedModelAvailable = null,
  loadDisabled = false,
  modelReady = false,
  modelBlockReason = null,
}: LandingProps) {
  const ui = useUiStrings();
  const locale = useLocale();
  const blocked = modelBlockReason != null;

  const primaryLabel =
    hostedModelAvailable === false ? (
      ui.landing.chooseLocalModel
    ) : modelReady ? (
      <>{ui.landing.openChat}</>
    ) : (
      <>{ui.landing.loadModel(MODEL_LABEL)}</>
    );

  return (
    <div className="overlay-screen">
      <DriftingLibrary />
      {/* Language switcher pinned to the top-right corner — the landing has no
          header bar, so this floats above the hero like the footer floats below.
          Hidden below `sm`: at 375px the near-full-width hero badge (the "Real
          LLM · 100% Local · WebGPU" pill spans almost edge-to-edge) sits right
          under this corner, so the toggle would occlude the badge's right end.
          On a phone the in-flow switcher below the header renders instead. */}
      <div className="absolute right-5 top-5 z-20 hidden sm:block">
        <LanguageSwitcher />
      </div>
      {/* `self-start` top-anchors the hero on a phone (overriding the parent's
          vertical centering from the child — no stylesheet edit). The overlay
          does not scroll, so a vertically-centered hero taller than a short
          viewport would clip its TOP — which would hide the in-flow switcher
          below. Top-anchoring keeps that switcher (and the badge) on-screen.
          `sm:self-center` restores the centered hero on wider screens. */}
      <div className="landing-content self-start sm:self-center">
        {/* Mobile-only in-flow switcher: rides at the top of the hero, right-
            aligned, so it always sits ABOVE the wide badge and can never overlap
            it at any viewport height. Replaced by the corner switcher from `sm` up. */}
        <div className="mb-5 self-end sm:hidden">
          <LanguageSwitcher />
        </div>
        <div className="landing-tag">{ui.landing.tag}</div>
        <h1 className="landing-title">
          {ui.landing.heroTitlePre}
          <em>{ui.landing.heroTitleEm}</em>
          {ui.landing.heroTitlePost}
        </h1>
        <p className="landing-sub">
          {ui.landing.heroSubBeforeCode}
          <code>@mlx-node/browser</code>
          {ui.landing.heroSubAfterCode}
        </p>

        <div className="landing-void-ad" aria-label={ui.landing.voidAdAria}>
          <span className="landing-void-eyebrow">{ui.landing.voidEyebrow}</span>
          <span className="landing-void-mark">{ui.landing.voidMark}</span>
          <span className="landing-void-copy">{ui.landing.voidCopy}</span>
        </div>

        <div className="landing-specs">
          <div className="spec">
            <div className="spec-value">{ui.landing.specParams}</div>
            <div className="spec-label">{ui.landing.specParamsLabel}</div>
          </div>
          <div className="spec">
            <div className="spec-value">{ui.landing.specChapters(CHAPTERS.length)}</div>
            <div className="spec-label">{ui.landing.specChaptersLabel}</div>
          </div>
          <div className="spec">
            <div className="spec-value">{ui.landing.specLocal}</div>
            <div className="spec-label">{ui.landing.specLocalLabel}</div>
          </div>
        </div>

        <div className="btn-load-group">
          <button
            type="button"
            className="btn-load"
            onClick={onLoad}
            disabled={loadDisabled || hostedModelAvailable === null || blocked}
          >
            {hostedModelAvailable === null ? ui.landing.checkingModel : primaryLabel}
          </button>
          <button
            type="button"
            className="btn-load-arrow"
            title={ui.landing.localPickerTitle}
            aria-label={ui.landing.localPickerTitle}
            onClick={onLocalModel}
            disabled={loadDisabled || blocked}
          >
            ▾
          </button>
        </div>
        {blocked && <p className="landing-learn-hint">{modelBlockedShort(locale)}</p>}

        <button type="button" className="btn-start-learning" onClick={onStartLearning}>
          {ui.landing.startLearning}
        </button>
        <p className="landing-learn-hint">{ui.landing.learnHint(CHAPTERS.length)}</p>

        <button type="button" className="landing-jspace-link" onClick={onOpenJSpace}>
          {ui.landing.jspaceLink}
        </button>

        {errorBanner && (
          <div className="error-banner" style={{ marginTop: 24 }}>
            {errorBanner}
          </div>
        )}
      </div>

      <img className="landing-mascot" src="/capybara.png" alt={ui.landing.mascotAlt} />

      <div className="landing-footer">
        {ui.landing.builtWith} <code>@mlx-node/browser</code>
        <span className="landing-footer-void">{ui.landing.hostedOnVoid}</span>
        {/* The local-model picker starts the same WebGPU load, so a blocked
            device (iOS/low-RAM/no-WebGPU) can't use it — hide this footer link
            there so it isn't a dead affordance. The blocked hint under the CTA
            already tells the user to open the page on a desktop. */}
        {!blocked && (
          <span className="local-link" onClick={onLocalModel}>
            {ui.landing.localModelLink}
          </span>
        )}
      </div>
    </div>
  );
}

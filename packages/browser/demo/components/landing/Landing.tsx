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
          header bar, so this floats above the hero like the footer floats below. */}
      <div className="absolute right-5 top-5 z-20">
        <LanguageSwitcher />
      </div>
      <div className="landing-content">
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
        <span className="local-link" onClick={onLocalModel}>
          {ui.landing.localModelLink}
        </span>
      </div>
    </div>
  );
}

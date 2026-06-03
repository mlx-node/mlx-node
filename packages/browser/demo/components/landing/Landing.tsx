import { CHAPTERS } from '../../learn/chapters';
import { DriftingLibrary } from './DriftingLibrary';

export type LandingProps = {
  onLoad: () => void;
  onLocalModel: () => void;
  onStartLearning: () => void;
  errorBanner: string | null;
  hostedModelAvailable?: boolean | null;
  loadDisabled?: boolean;
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
  errorBanner,
  hostedModelAvailable = null,
  loadDisabled = false,
  modelReady = false,
}: LandingProps) {
  const primaryLabel =
    hostedModelAvailable === false ? (
      'Choose Local Model'
    ) : modelReady ? (
      <>Open Chat →</>
    ) : (
      <>
        Load Model (<span>{MODEL_LABEL}</span>)
      </>
    );

  return (
    <div className="overlay-screen">
      <DriftingLibrary />
      <div className="landing-content">
        <div className="landing-tag">Real LLM · 100% Local · WebGPU</div>
        <h1 className="landing-title">
          How LLMs <em>Work</em>
        </h1>
        <p className="landing-sub">
          Run a real 0.8B language model (Qwen3.5) entirely in your browser, then learn exactly how it works —
          layer by layer. No server, no API keys — powered by <code>@mlx-node/browser</code> and WebGPU.
        </p>

        <div className="landing-void-ad" aria-label="Hosted on Void Platform">
          <span className="landing-void-eyebrow">Hosted on</span>
          <span className="landing-void-mark">Void Platform</span>
          <span className="landing-void-copy">Edge assets · WebGPU isolation</span>
        </div>

        <div className="landing-specs">
          <div className="spec">
            <div className="spec-value">0.8B params</div>
            <div className="spec-label">Qwen3.5 · runs live</div>
          </div>
          <div className="spec">
            <div className="spec-value">{CHAPTERS.length} chapters</div>
            <div className="spec-label">Interactive course</div>
          </div>
          <div className="spec">
            <div className="spec-value">100% local</div>
            <div className="spec-label">WebGPU · no server</div>
          </div>
        </div>

        <div className="btn-load-group">
          <button
            type="button"
            className="btn-load"
            onClick={onLoad}
            disabled={loadDisabled || hostedModelAvailable === null}
          >
            {hostedModelAvailable === null ? 'Checking Model...' : primaryLabel}
          </button>
          <button
            type="button"
            className="btn-load-arrow"
            title="Choose local model directory"
            aria-label="Choose local model directory"
            onClick={onLocalModel}
            disabled={loadDisabled}
          >
            ▾
          </button>
        </div>

        <button type="button" className="btn-start-learning" onClick={onStartLearning}>
          Start learning →
        </button>
        <p className="landing-learn-hint">
          New to LLMs? Step through {CHAPTERS.length} chapters that explain how this model actually works.
        </p>

        {errorBanner && (
          <div className="error-banner" style={{ marginTop: 24 }}>
            {errorBanner}
          </div>
        )}
      </div>

      <img className="landing-mascot" src="/capybara.png" alt="Capybara mascot" />

      <div className="landing-footer">
        Built with <code>@mlx-node/browser</code>
        <span className="landing-footer-void">Hosted on Void</span>
        <span className="local-link" onClick={onLocalModel}>
          Local model…
        </span>
      </div>
    </div>
  );
}

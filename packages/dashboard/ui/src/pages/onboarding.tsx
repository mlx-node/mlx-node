import { ModelLogo, prettyModelName } from '@/components/model-logos';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { dismissOnboarding } from '@/lib/onboarding';
import type { CatalogItem, CatalogResponse, ModelsResponse } from '@/lib/types';
import type { AsyncState } from '@/lib/use-api';
import {
  ArrowDown,
  ArrowRight,
  Boxes,
  Check,
  CheckCheck,
  CircleHelp,
  Download,
  ExternalLink,
  HardDrive,
  Laptop,
  LockKeyhole,
  RefreshCw,
  Sparkles,
} from 'lucide-react';
import { type CSSProperties, type ReactNode, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

export interface ModelDownloadError {
  repo: string;
  message: string;
}

interface OnboardingProps {
  catalog: AsyncState<CatalogResponse>;
  models: AsyncState<ModelsResponse>;
  activeRepos: string[];
  downloadError: ModelDownloadError | null;
  downloadsLoading: boolean;
  downloadsError: Error | undefined;
  reloadDownloads: () => void;
  renderDownload: (item: CatalogItem) => ReactNode;
}

function modelStyle(item: CatalogItem): 'violet' | 'blue' | 'green' {
  if (/gemma/i.test(item.label)) return 'green';
  if (/agentworld/i.test(item.label)) return 'blue';
  return 'violet';
}

function modelDescription(item: CatalogItem): string {
  if (/agentworld/i.test(item.label)) return 'Tuned for agent workflows, with fast responses.';
  if (/gemma-4-26b/i.test(item.label)) return 'Fast responses, with a smaller download to get started.';
  if (item.isDefault) return 'Our recommended starting point for working with tools.';
  return item.description;
}

export default function Onboarding({
  catalog,
  models,
  activeRepos,
  downloadError,
  downloadsLoading,
  downloadsError,
  reloadDownloads,
  renderDownload,
}: OnboardingProps) {
  const navigate = useNavigate();
  const [selectedRepo, setSelectedRepo] = useState<string | null>(null);
  const items = (catalog.data?.items ?? []).filter((item) => !item.hidden);
  const activeRepo = activeRepos.find((repo) => items.some((item) => item.hfRepo === repo));
  const selected =
    items.find((item) => item.hfRepo === (activeRepo ?? selectedRepo)) ??
    items.find((item) => item.isDefault && !item.blockedByForeignDir) ??
    items.find((item) => !item.blockedByForeignDir) ??
    items[0];
  const downloading = activeRepo !== undefined;
  const ready = selected?.present === true && !downloading;
  const error = catalog.error ?? models.error ?? downloadsError;
  const modelsDir = models.data?.dir;
  const canFinish = Boolean(modelsDir) && !models.loading && !models.refreshing && !models.error;

  useEffect(() => {
    if (activeRepo) setSelectedRepo(activeRepo);
  }, [activeRepo]);

  const finish = (to: string): void => {
    // The root route checks dismissal for this library. Leaving before its
    // identity is known would lose the skip and immediately reopen onboarding.
    if (!canFinish || !modelsDir) return;
    dismissOnboarding(modelsDir);
    void navigate(to, { replace: true });
  };

  return (
    <div className="onboarding h-screen overflow-hidden text-foreground">
      <div className="h-9 shrink-0" style={{ WebkitAppRegion: 'drag' } as CSSProperties} aria-hidden />
      <main className="onboarding-scroll overflow-y-auto">
        <div className="onboarding-wrap">
          <header className="onboarding-header">
            <button
              type="button"
              disabled={!canFinish}
              onClick={() => finish('/')}
              className="onboarding-brand disabled:cursor-default disabled:opacity-60"
              aria-label="mlx-node home"
            >
              <span className="bg-brand-gradient text-primary-foreground flex size-9 items-center justify-center rounded-xl">
                <Boxes className="size-5" aria-hidden />
              </span>
              mlx-node
            </button>
            <Button variant="ghost" className="text-muted-foreground" disabled={!canFinish} onClick={() => finish('/')}>
              {downloading ? 'Explore while downloading' : 'Set up later'} <ArrowRight className="size-4" aria-hidden />
            </Button>
          </header>

          <section className="onboarding-hero" aria-labelledby="welcome-title">
            <div className="onboarding-eyebrow">
              <span /> YOUR LOCAL AI STARTS HERE
            </div>
            <h1 id="welcome-title">
              Powerful AI. <span>All yours.</span>
            </h1>
            <p>Start with a model. Download once, then run it on your own device.</p>
            <div className="onboarding-promises">
              <span>
                <LockKeyhole aria-hidden /> Runs on your device
              </span>
              <span>
                <Laptop aria-hidden /> Yours to use offline
              </span>
              <span>
                <Sparkles aria-hidden /> No API key needed
              </span>
            </div>
          </section>

          <ol className="onboarding-steps" aria-label="Getting started">
            {['Choose a model', 'Download', 'Make it yours'].map((label, index) => {
              const step = ready ? 2 : downloading ? 1 : 0;
              return (
                <li key={label} aria-current={index === step ? 'step' : undefined} data-complete={index < step}>
                  <span className="onboarding-step-number">
                    {index < step ? <Check className="size-3.5" aria-hidden /> : `0${index + 1}`}
                  </span>
                  {label}
                </li>
              );
            })}
          </ol>

          <section aria-labelledby="model-choice-title" className="onboarding-models">
            <div className="onboarding-section-heading">
              <div>
                <h2 id="model-choice-title">Choose your first model</h2>
                <p>Choose one for now. You can add more anytime.</p>
              </div>
              <span className="onboarding-curated">
                <CheckCheck className="size-4" aria-hidden /> Curated for mlx-node
              </span>
            </div>

            {error ? (
              <div className="onboarding-notice" role="alert">
                <div>
                  <h3>We couldn’t load your model library.</h3>
                  <p>{error.message}</p>
                </div>
                <Button
                  variant="outline"
                  onClick={() => {
                    catalog.reload();
                    models.reload();
                    reloadDownloads();
                  }}
                >
                  <RefreshCw className="size-4" aria-hidden /> Try again
                </Button>
              </div>
            ) : catalog.loading || models.loading || downloadsLoading ? (
              <div className="onboarding-card-grid" role="status" aria-label="Loading available models">
                {[0, 1, 2].map((i) => (
                  <Skeleton key={i} className="onboarding-model-skeleton rounded-3xl" />
                ))}
              </div>
            ) : items.length === 0 ? (
              <div className="onboarding-notice" role="status">
                <div>
                  <h3>No models available yet</h3>
                  <p>You can explore the app and return to Models later.</p>
                </div>
                <Button variant="outline" onClick={catalog.reload}>
                  Refresh models
                </Button>
              </div>
            ) : (
              <fieldset disabled={downloading} className="onboarding-card-grid">
                <legend className="sr-only">Choose your first model</legend>
                {items.map((item) => {
                  const checked = selected?.hfRepo === item.hfRepo;
                  return (
                    <label
                      key={item.hfRepo}
                      className={`onboarding-model-card onboarding-${modelStyle(item)}`}
                      data-selected={checked}
                      data-blocked={item.blockedByForeignDir}
                    >
                      <input
                        type="radio"
                        name="first-model"
                        value={item.hfRepo}
                        checked={checked}
                        disabled={item.blockedByForeignDir}
                        onChange={() => setSelectedRepo(item.hfRepo)}
                        aria-label={item.label}
                        aria-describedby={`description-${item.slug}`}
                      />
                      <div className="onboarding-card-art" aria-hidden>
                        <div className="onboarding-orbit onboarding-orbit-one" />
                        <div className="onboarding-orbit onboarding-orbit-two" />
                        <div className="onboarding-orbit onboarding-orbit-three" />
                        <span className="onboarding-model-mark">
                          <ModelLogo model={item.label} className="size-12" />
                        </span>
                      </div>
                      <div className="onboarding-card-body">
                        <span className="onboarding-card-badge">
                          {item.present ? 'On your device' : item.isDefault ? 'Recommended' : 'Ready for local AI'}
                        </span>
                        <h3>{prettyModelName(item.label)}</h3>
                        <p id={`description-${item.slug}`}>{modelDescription(item)}</p>
                        <div className="onboarding-card-meta">
                          <span>
                            <HardDrive className="size-3.5" aria-hidden /> ~{item.sizeGb} GB
                          </span>
                          <span>
                            {item.blockedByForeignDir
                              ? 'Folder needs cleanup'
                              : item.present
                                ? 'Downloaded'
                                : 'One-time download'}
                          </span>
                        </div>
                      </div>
                      <span className="onboarding-selection" aria-hidden>
                        {checked && <Check className="size-3" />}
                      </span>
                    </label>
                  );
                })}
              </fieldset>
            )}

            <div className="onboarding-guidance">
              <CircleHelp className="size-4 shrink-0" aria-hidden />
              <p>
                New to local models? Start with the recommended option. Download size is disk space; running a model
                needs additional memory.
              </p>
            </div>
          </section>

          <footer className="onboarding-footer">
            <LockKeyhole className="size-3.5" aria-hidden />
            <span>Downloaded from Hugging Face. Run locally, on your terms.</span>
          </footer>
        </div>
      </main>

      {!error && !catalog.loading && !models.loading && !downloadsLoading && selected && (
        <div className="onboarding-dock">
          <section className="onboarding-download-panel" aria-label="Selected model">
            <div className="onboarding-download-summary" aria-live="polite">
              <span className={`onboarding-download-icon ${ready ? 'onboarding-ready' : ''}`}>
                {ready ? (
                  <Check className="size-5" aria-hidden />
                ) : downloading ? (
                  <Download className="size-5" aria-hidden />
                ) : (
                  <ArrowDown className="size-5" aria-hidden />
                )}
              </span>
              <div>
                <p className="onboarding-selected-label">
                  {ready ? 'YOUR MODEL IS READY' : downloading ? 'MAKING IT LOCAL' : 'YOUR FIRST MODEL'}
                </p>
                <h3>{prettyModelName(selected.label)}</h3>
                <p className="onboarding-download-note">
                  {ready
                    ? 'The download is complete. Let’s put your model to work.'
                    : downloading
                      ? 'Keep the app open. You can explore while your model downloads.'
                      : `About ${selected.sizeGb} GB to download. A little setup, a lot of possibility.`}
                </p>
              </div>
            </div>
            <div className="onboarding-download-action">
              {ready ? (
                <Button className="w-full" disabled={!canFinish} onClick={() => finish('/coding-agents')}>
                  Set up coding agents <ArrowRight className="size-4" aria-hidden />
                </Button>
              ) : (
                renderDownload(selected)
              )}
              {ready ? (
                <button className="onboarding-text-link" disabled={!canFinish} onClick={() => finish('/')}>
                  Go to overview
                </button>
              ) : (
                !downloading && (
                  <a
                    className="onboarding-text-link"
                    href={`https://huggingface.co/${selected.hfRepo}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    View model details <ExternalLink className="size-3" aria-hidden />
                  </a>
                )
              )}
            </div>
            {downloadError?.repo === selected.hfRepo && (
              <p className="onboarding-download-error text-destructive" role="alert">
                {downloadError.message} Please try again.
              </p>
            )}
          </section>
        </div>
      )}
    </div>
  );
}

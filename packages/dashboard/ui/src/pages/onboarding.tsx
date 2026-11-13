import { ArrowRight, ExternalLink } from 'lucide-react';
import { type ReactElement, type ReactNode, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import { ChoiceCard } from '../components/arc-x/choice-card/choice-card';
import { LiveLabel } from '../components/arc-x/live-label/live-label';
import { ModelLogo, prettyModelName } from '../components/arc-x/model-badge/model-logos';
import { Panel } from '../components/arc-x/panel/panel';
import { Steps } from '../components/arc-x/steps/steps';
import { Alert } from '../components/arc/alert/alert';
import { Badge } from '../components/arc/badge/badge';
import { Button } from '../components/arc/button/button';
import { EmptyState } from '../components/arc/empty-state/empty-state';
import { modelBlurb } from '../lib/catalog';
import { dismissOnboarding } from '../lib/onboarding';
import type { CatalogItem, CatalogResponse, ModelsResponse } from '../lib/types';
import type { AsyncState } from '../lib/use-api';
import { WindowDragStrip } from '../shell/window-chrome';
import { hostNote } from './models/catalog';
import type { ModelDownloadError } from './models/use-model-downloads';

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

const STEPS = ['Choose a model', 'Download', 'Set up coding agents'] as const;

function availability(item: CatalogItem): string {
  if (item.blockedByForeignDir) return 'Folder needs cleanup';
  if (item.present) return 'Downloaded';
  return 'One-time download';
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
}: OnboardingProps): ReactElement {
  const navigate = useNavigate();
  const [selectedRepo, setSelectedRepo] = useState<string | null>(null);
  const items = (catalog.data?.items ?? []).filter((item) => !item.hidden);
  const activeRepo = activeRepos.find((repo) => items.some((item) => item.hfRepo === repo));
  const selected =
    items.find((item) => item.hfRepo === (activeRepo ?? selectedRepo)) ??
    items.find((item) => item.isDefault === true && !item.blockedByForeignDir) ??
    items.find((item) => !item.blockedByForeignDir) ??
    items[0];
  const downloading = activeRepo !== undefined;
  const ready = selected?.present === true && !downloading;
  const error = catalog.error ?? models.error ?? downloadsError;
  const loading = catalog.loading || models.loading || downloadsLoading;
  const modelsDir = models.data?.dir;
  const canFinish =
    modelsDir !== undefined && modelsDir !== '' && !models.loading && !models.refreshing && models.error === undefined;
  const host = models.data?.host;

  useEffect(() => {
    if (activeRepo !== undefined) setSelectedRepo(activeRepo);
  }, [activeRepo]);

  const finish = (to: string): void => {
    // The root route checks dismissal for this library; leaving before its identity is known would reopen onboarding.
    if (!canFinish || modelsDir === undefined) return;
    dismissOnboarding(modelsDir);
    void navigate(to, { replace: true });
  };

  let section: ReactNode;
  if (error !== undefined) {
    section = (
      <Alert tone="danger" title="Could not load the model list">
        <span className="grid justify-items-start gap-2">
          <span>{error.message}</span>
          <Button
            variant="secondary"
            size="sm"
            onClick={() => {
              catalog.reload();
              models.reload();
              reloadDownloads();
            }}
          >
            Try again
          </Button>
        </span>
      </Alert>
    );
  } else if (loading) {
    section = (
      <div
        role="status"
        aria-label="Loading available models"
        className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,12.5rem),1fr))] gap-3"
      >
        {[0, 1, 2].map((index) => (
          <ChoiceCard
            key={index}
            loading
            name="first-model"
            value=""
            checked={false}
            onChange={() => {}}
            ariaLabel=""
            title=""
            description=""
          />
        ))}
      </div>
    );
  } else if (items.length === 0) {
    section = (
      <EmptyState
        title="No models available yet"
        description="You can explore the app and return to Models later."
        action={
          <Button variant="secondary" onClick={catalog.reload}>
            Refresh models
          </Button>
        }
      />
    );
  } else {
    section = (
      <fieldset
        disabled={downloading}
        className="m-0 grid min-w-0 grid-cols-[repeat(auto-fit,minmax(min(100%,12.5rem),1fr))] gap-3 p-0"
      >
        <legend className="sr-only">Choose your first model</legend>
        {items.map((item) => {
          const note = hostNote(item, host);
          return (
            <ChoiceCard
              key={item.hfRepo}
              name="first-model"
              value={item.hfRepo}
              checked={selected?.hfRepo === item.hfRepo}
              disabled={item.blockedByForeignDir}
              onChange={setSelectedRepo}
              ariaLabel={item.label}
              describedBy={`description-${item.slug}`}
              descriptionId={`description-${item.slug}`}
              media={<ModelLogo model={item.label} className="size-7" />}
              badge={
                item.present ? (
                  <Badge tone="success" size="sm">
                    On this Mac
                  </Badge>
                ) : item.isDefault === true ? (
                  <Badge tone="info" size="sm">
                    Recommended
                  </Badge>
                ) : undefined
              }
              title={prettyModelName(item.label)}
              description={modelBlurb(item)}
              meta={`~${item.sizeGb} GB · ${availability(item)}`}
              warning={note ?? undefined}
            />
          );
        })}
      </fieldset>
    );
  }

  const showDock = error === undefined && !loading && selected !== undefined;

  return (
    <div data-onboarding="" className="grid h-screen grid-rows-[40px_minmax(0,1fr)_auto] text-foreground">
      <WindowDragStrip />
      <main data-scroll-root="" className="min-h-0 overflow-y-auto">
        <div className="mx-auto grid w-full max-w-[880px] gap-6 px-4 pt-2 pb-8 md:px-6">
          <header className="flex items-center justify-between gap-3">
            <button
              type="button"
              disabled={!canFinish}
              onClick={() => finish('/')}
              aria-label="mlx-node home"
              className="cursor-pointer font-heading text-strong font-medium text-foreground disabled:cursor-default disabled:text-muted"
            >
              mlx-node
            </button>
            <Button variant="ghost" disabled={!canFinish} onClick={() => finish('/')}>
              {downloading ? 'Explore while downloading' : 'Set up later'}
              <ArrowRight size={16} aria-hidden="true" />
            </Button>
          </header>

          <div className="grid gap-2">
            <h1 className="m-0 font-heading text-hero font-normal text-foreground">Choose your first model</h1>
            <p className="m-0 text-body text-secondary">
              Download one model to run on this Mac. You can add more later.
            </p>
          </div>

          <Steps label="Getting started" steps={STEPS} current={ready ? 2 : downloading ? 1 : 0} />

          <div className="grid gap-3">
            {section}
            <p className="m-0 text-caption text-secondary">
              New to local models? Start with the recommended one. Download size is disk space; running a model needs
              additional memory.
            </p>
          </div>

          <p className="m-0 text-caption text-muted">Models download from Hugging Face and run on this Mac.</p>
        </div>
      </main>

      {showDock ? (
        <div className="px-4 pb-4 md:px-6">
          <div className="mx-auto w-full max-w-[880px]">
            <Panel ariaLabel="Selected model">
              <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3">
                <div className="grid min-w-[min(100%,16rem)] flex-1 gap-1" aria-live="polite">
                  {downloading ? (
                    <LiveLabel active as="p" className="m-0 text-caption text-secondary">
                      Downloading
                    </LiveLabel>
                  ) : (
                    <p className="m-0 text-caption text-secondary">{ready ? 'Model ready' : 'Your first model'}</p>
                  )}
                  <h2 className="m-0 text-heading font-medium text-foreground">{prettyModelName(selected.label)}</h2>
                  <p className="m-0 text-body text-secondary">
                    {ready
                      ? 'The download is complete.'
                      : downloading
                        ? 'Keep the app open. You can look around while it downloads.'
                        : `About ${selected.sizeGb} GB to download.`}
                  </p>
                </div>
                <div className="grid w-[17.5rem] max-w-full shrink-0 justify-items-center gap-2">
                  {ready ? (
                    <>
                      <Button
                        variant="primary"
                        size="lg"
                        className="w-full"
                        disabled={!canFinish}
                        onClick={() => finish('/coding-agents')}
                      >
                        Set up coding agents
                        <ArrowRight size={16} aria-hidden="true" />
                      </Button>
                      <Button variant="ghost" size="sm" disabled={!canFinish} onClick={() => finish('/')}>
                        Go to overview
                      </Button>
                    </>
                  ) : (
                    <>
                      <div className="w-full">{renderDownload(selected)}</div>
                      {downloading ? null : (
                        <a
                          className="link inline-flex items-center gap-1 text-caption"
                          href={`https://huggingface.co/${selected.hfRepo}`}
                          target="_blank"
                          rel="noreferrer"
                        >
                          View model details
                          <ExternalLink size={12} aria-hidden="true" />
                        </a>
                      )}
                    </>
                  )}
                </div>
              </div>
              {downloadError?.repo === selected.hfRepo ? (
                <div className="mt-3">
                  <Alert tone="danger" title="Download failed">
                    {downloadError.message} Please try again.
                  </Alert>
                </div>
              ) : null}
            </Panel>
          </div>
        </div>
      ) : null}
    </div>
  );
}

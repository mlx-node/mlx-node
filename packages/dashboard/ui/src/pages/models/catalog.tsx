import { AlertTriangle, Download, ExternalLink } from 'lucide-react';
import { useState, type ReactElement, type ReactNode } from 'react';
import { Link } from 'react-router-dom';

import { ConfirmPopover, type ConfirmContent } from '../../components/arc-x/confirm-popover/confirm-popover';
import { ModelLogo } from '../../components/arc-x/model-badge/model-logos';
import { MorphLoader } from '../../components/arc-x/morph-loader/morph-loader';
import { Panel } from '../../components/arc-x/panel/panel';
import { Placeholder } from '../../components/arc-x/placeholder/placeholder';
import { Rule } from '../../components/arc-x/rule/rule';
import { Alert } from '../../components/arc/alert/alert';
import { Badge } from '../../components/arc/badge/badge';
import { Button } from '../../components/arc/button/button';
import { DownloadProgress } from '../../components/download-progress';
import { catalogBuildLabel } from '../../lib/catalog';
import { formatBytes, formatBytesParts, formatCount } from '../../lib/format';
import { truncateMiddle } from '../../lib/labels';
import type { CatalogBlocker, CatalogItem, ModelsResponse } from '../../lib/types';
import type { CatalogDownloadProps } from './use-model-downloads';

const GB = 1e9;
const GIB = 2 ** 30;

/** The one note a catalog card or an onboarding choice shows about room on this Mac, or null. */
export function hostNote(
  item: Pick<CatalogItem, 'sizeGb' | 'present'>,
  host: ModelsResponse['host'] | undefined,
): string | null {
  if (host === undefined) return null;
  if (!item.present && host.diskFreeBytes !== null && item.sizeGb * GB > host.diskFreeBytes) {
    return `Needs about ${item.sizeGb} GB. ${formatBytes(host.diskFreeBytes)} is free on this disk.`;
  }
  if (item.sizeGb * GIB > host.totalMemBytes) {
    return `Weights are larger than this Mac's memory (${formatBytes(host.totalMemBytes)}).`;
  }
  return null;
}

export function HostNote({ text }: { text: string }): ReactElement {
  return (
    <p className="m-0 flex items-start gap-1.5 text-caption text-warning">
      <AlertTriangle size={12} aria-hidden="true" className="mt-0.5 shrink-0" />
      <span>{text}</span>
    </p>
  );
}

export interface CatalogDownloadActionProps extends CatalogDownloadProps {
  /** `/models?focus=…` when a local model is the installed one. */
  detailsTo?: string;
  /** The DFlash2 draft block: only the Reinstall confirmation title differs. */
  companion?: boolean;
}

/** A byte size whose number and unit never wrap apart. */
function sizeText(bytes: number): string {
  const { value, unit } = formatBytesParts(bytes);
  return unit === '' ? value : `${value}\u00a0${unit}`;
}

/** How the card names what is in the way, and the word for it. */
export function blockerPhrase(blocker: CatalogBlocker): { phrase: string; noun: string } {
  if (blocker.kind === 'symlink') return { phrase: 'a link', noun: 'link' };
  if (blocker.kind === 'file') return { phrase: `a file (${sizeText(blocker.sizeBytes)})`, noun: 'file' };
  if (blocker.truncated) {
    if (blocker.fileCount === 0) return { phrase: 'a directory the app could not fully read', noun: 'directory' };
    return { phrase: `a directory with more than ${formatCount(blocker.fileCount)} files`, noun: 'directory' };
  }
  if (blocker.fileCount === 0) return { phrase: 'an empty directory', noun: 'directory' };
  const files = blocker.fileCount === 1 ? '1 file' : `${formatCount(blocker.fileCount)} files`;
  return { phrase: `a directory with ${files} (${sizeText(blocker.sizeBytes)})`, noun: 'directory' };
}

const NAME_MAX_CHARS = 40;

/** What the removal takes away, or null when there is nothing to say (a link whose text is unknown). */
function blockerContents(blocker: CatalogBlocker): ReactNode {
  const size = sizeText(blocker.sizeBytes);
  if (blocker.kind === 'symlink') {
    if (blocker.linkTarget === null) return null;
    return blocker.dangling
      ? `It points to ${blocker.linkTarget}, which does not exist.`
      : `It points to ${blocker.linkTarget}.`;
  }
  if (blocker.kind === 'file') return `The file is ${size}.`;
  if (blocker.truncated) {
    if (blocker.fileCount === 0) return 'The app could not read all of it, so its size is not known.';
    return `It holds more than ${formatCount(blocker.fileCount)} files (more than ${size}).`;
  }
  if (blocker.fileCount === 0) return 'The directory is empty.';
  const files = blocker.fileCount === 1 ? '1 file' : `${formatCount(blocker.fileCount)} files`;
  if (blocker.names.length === 0) return `It holds ${files} (${size}).`;
  return (
    <>
      {`It holds ${files} (${size}): `}
      {blocker.names.map((name, index) => (
        <span key={name} title={name}>
          {index > 0 ? ', ' : ''}
          {truncateMiddle(name, NAME_MAX_CHARS)}
        </span>
      ))}
      .
    </>
  );
}

function blockerNotice(blocker: CatalogBlocker): string | undefined {
  if (blocker.kind === 'symlink') return 'This is a link. Only the link is removed. The target is not changed.';
  if (blocker.listed) return 'Local models lists this directory. Make sure that you do not need it.';
  if (blocker.modelLike) return 'This directory has a config.json. Make sure that you do not need it.';
  return undefined;
}

export interface ReinstallHandlers {
  companion?: boolean;
  onReinstall: () => Promise<void>;
  onReinstallStarted: () => void;
}

/** The confirmation of a Reinstall over the entry in the way of `item`. */
export function reinstallContent(
  item: { slug: string; blocker: CatalogBlocker },
  handlers: ReinstallHandlers,
): ConfirmContent {
  const { noun } = blockerPhrase(item.blocker);
  const contents = blockerContents(item.blocker);
  return {
    title: handlers.companion === true ? 'Reinstall companion weights?' : 'Reinstall model?',
    description: (
      <>
        This downloads the model, then permanently deletes the {noun}{' '}
        <span className="font-medium text-foreground">{item.slug}</span> and installs the model in its place.{' '}
        {contents === null ? null : <>{contents} </>}
        Nothing is deleted if the download fails or you cancel it.
      </>
    ),
    notice: blockerNotice(item.blocker),
    confirmLabel: 'Reinstall',
    workingLabel: 'Starting…',
    successLabel: 'Started',
    tone: 'danger',
    errorTitle: 'Could not start the download',
    onConfirm: handlers.onReinstall,
    onDone: handlers.onReinstallStarted,
  };
}

/** The action of one catalog entry: progress, Update, the Installed status, the cleanup note, or Install. */
export function CatalogDownloadAction({
  item,
  installLabel = 'Install',
  settlingLabel = installLabel,
  prominent = false,
  updateAvailable,
  settling,
  updateSettling,
  job,
  onInstall,
  onReinstall,
  onReinstallStarted,
  onDone,
  onError,
  onCancelled,
  onCancel,
  detailsTo,
  companion = false,
}: CatalogDownloadActionProps): ReactElement {
  const [starting, setStarting] = useState(false);
  const start = async (): Promise<void> => {
    if (starting) return;
    setStarting(true);
    try {
      await onInstall();
    } finally {
      setStarting(false);
    }
  };
  // A live job is the only safe gate and must not be qualified by catalog state: an update re-installs something
  // already present, and the update probe may still be loading.
  if (job !== undefined) {
    return (
      <div className="grid gap-2">
        <DownloadProgress id={job.id} onDone={onDone} onError={onError} onCancelled={onCancelled} />
        {item.blocker !== null ? (
          <p className="m-0 text-caption text-secondary">
            {`Replaces the old ${blockerPhrase(item.blocker).noun} when the download is complete.`}
          </p>
        ) : null}
        {/* cancel() refuses a committing job, and a second cancel answers 404. */}
        {!job.committing && !job.cancelling ? (
          <Button type="button" variant="ghost" size="sm" className="w-full" onClick={() => onCancel(job.id)}>
            Cancel
          </Button>
        ) : null}
      </div>
    );
  }
  if (item.present && updateAvailable) {
    return (
      <Button
        type="button"
        variant="secondary"
        className="w-full"
        disabled={starting || settling || updateSettling}
        onClick={() => void start()}
      >
        <Download size={16} aria-hidden="true" />
        Update available
      </Button>
    );
  }
  // `present`, not `installed`: an unowned install (mlx download) cannot be overwritten, so it gets no Install.
  if (item.present) {
    return (
      <p className="m-0 flex min-h-8 items-center gap-1.5 text-body font-medium text-secondary" data-status="installed">
        <MorphLoader size={14} status="success" />
        <span>Installed</span>
        {detailsTo !== undefined ? (
          <>
            <span aria-hidden="true">·</span>
            <Link className="link" to={detailsTo}>
              Details
            </Link>
          </>
        ) : null}
      </p>
    );
  }
  if (item.blocker?.holdsModel === true) {
    return (
      <Alert tone="warning" title="Needs cleanup">
        <span className="font-mono">{item.slug}</span> holds a model that the app did not create, so Reinstall cannot
        replace it.{' '}
        {item.blocker.listed
          ? 'Delete it in Local models to install this one.'
          : 'Move or delete it in Finder to install this one.'}
      </Alert>
    );
  }
  if (item.blocker !== null) {
    const { phrase, noun } = blockerPhrase(item.blocker);
    return (
      <Alert tone="warning" title="Needs cleanup">
        <span className="grid justify-items-start gap-2">
          <span>
            <span className="font-mono">{item.slug}</span> is in the way: {phrase} that the app did not create.
            Reinstall downloads the model, then removes this {noun}.
          </span>
          <ConfirmPopover
            trigger={{
              kind: 'button',
              label: 'Reinstall',
              variant: 'secondary',
              size: 'sm',
              icon: <Download size={16} aria-hidden="true" />,
            }}
            align="start"
            disabled={settling}
            disabledReason="Checking the last download."
            {...reinstallContent(
              { slug: item.slug, blocker: item.blocker },
              { companion, onReinstall, onReinstallStarted },
            )}
          />
        </span>
      </Alert>
    );
  }
  return (
    <Button
      type="button"
      variant={prominent ? 'primary' : 'secondary'}
      size={prominent ? 'lg' : 'md'}
      className="w-full"
      loading={starting}
      disabled={starting || settling}
      onClick={() => void start()}
    >
      {starting ? null : <Download size={16} aria-hidden="true" />}
      {starting ? 'Starting download…' : settling ? settlingLabel : installLabel}
    </Button>
  );
}

function hfUrl(repo: string): string {
  return `https://huggingface.co/${repo}`;
}

export interface CatalogCardProps extends CatalogDownloadActionProps {
  item: CatalogItem;
  draftDownload?: CatalogDownloadProps;
  host: ModelsResponse['host'] | undefined;
}

/** One recommended model: link, badges, size and build, a host note, its action, and its optional draft. */
export function CatalogCard({ item, draftDownload, host, ...download }: CatalogCardProps): ReactElement {
  const build = catalogBuildLabel(item);
  const note = hostNote(item, host);
  const meta = [`~${item.sizeGb} GB`, build].filter((part): part is string => part !== null).join(' · ');
  return (
    <div data-model={item.slug} className="min-w-0">
      <Panel ariaLabel={item.label}>
        <div className="grid min-w-0 gap-3">
          <div className="flex min-w-0 items-start justify-between gap-2">
            <a
              className="flex min-w-0 items-center gap-2 text-strong font-medium text-foreground"
              href={hfUrl(item.hfRepo)}
              target="_blank"
              rel="noreferrer"
            >
              <ModelLogo model={item.label} className="size-[18px] shrink-0" />
              <span className="min-w-0 truncate" title={item.label}>
                {item.label}
              </span>
              <ExternalLink size={12} aria-hidden="true" className="shrink-0 text-muted" />
            </a>
            <span className="flex shrink-0 items-center gap-1">
              {item.isDefault === true ? (
                <Badge tone="info" size="sm">
                  Default
                </Badge>
              ) : null}
              {item.present && download.updateAvailable ? (
                <Badge tone="warning" size="sm">
                  Update
                </Badge>
              ) : null}
            </span>
          </div>
          <p className="m-0 line-clamp-2 text-body text-secondary">{item.description}</p>
          <p className="m-0 text-caption text-secondary tabular-nums">{meta}</p>
          {note !== null ? <HostNote text={note} /> : null}
          <CatalogDownloadAction item={item} {...download} />
          {item.draft !== undefined && draftDownload !== undefined ? (
            <>
              <Rule />
              <div role="group" aria-label={`${item.draft.label} companion`} className="grid gap-2">
                <div className="flex min-w-0 flex-wrap items-center justify-between gap-2">
                  <span className="flex min-w-0 items-center gap-2">
                    <a
                      className="min-w-0 truncate text-body font-medium text-foreground"
                      href={hfUrl(item.draft.hfRepo)}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {item.draft.label}
                    </a>
                    <Badge size="sm">Optional</Badge>
                  </span>
                  <span className="text-caption text-secondary tabular-nums">~{item.draft.sizeGb} GB</span>
                </div>
                <p className="m-0 text-caption text-secondary">
                  Can speed up replies. Used automatically the next time this model loads.
                </p>
                <CatalogDownloadAction {...draftDownload} companion installLabel={`Install ${item.draft.label}`} />
              </div>
            </>
          ) : null}
        </div>
      </Panel>
    </div>
  );
}

/** The loading form of a card: the same rows as placeholders; `withDraft` holds the draft block too. */
export function CatalogCardPlaceholder({ withDraft = false }: { withDraft?: boolean }): ReactElement {
  return (
    <div className="min-w-0">
      <Panel ariaLabel="Loading recommended model" busy>
        <div className="grid gap-3">
          <Placeholder width="60%" />
          <Placeholder width="90%" />
          <Placeholder width="35%" />
          <Placeholder height={32} radius="control" />
          {withDraft ? (
            <>
              <Rule />
              <Placeholder width="45%" />
              <Placeholder width="80%" />
              <Placeholder height={32} radius="control" />
            </>
          ) : null}
        </div>
      </Panel>
    </div>
  );
}

export function CatalogGrid({ children }: { children: ReactNode }): ReactElement {
  return (
    <div className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,17.5rem),1fr))] items-start gap-3">{children}</div>
  );
}

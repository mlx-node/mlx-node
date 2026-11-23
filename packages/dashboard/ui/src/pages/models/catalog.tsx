import { AlertTriangle, ArrowUp, Check, Download, ExternalLink } from 'lucide-react';
import { useCallback, useEffect, useRef, useState, type ReactElement, type ReactNode } from 'react';
import { Link } from 'react-router-dom';

import { ButtonLink } from '../../components/arc-x/button-link/button-link';
import { ConfirmPopover, type ConfirmContent } from '../../components/arc-x/confirm-popover/confirm-popover';
import { ModelLogo } from '../../components/arc-x/model-badge/model-logos';
import { MorphLoader } from '../../components/arc-x/morph-loader/morph-loader';
import { Panel } from '../../components/arc-x/panel/panel';
import { Placeholder } from '../../components/arc-x/placeholder/placeholder';
import { Alert } from '../../components/arc/alert/alert';
import { Badge } from '../../components/arc/badge/badge';
import { Button } from '../../components/arc/button/button';
import { Popover, PopoverContent, PopoverTrigger } from '../../components/arc/popover/popover';
import SegmentedControl from '../../components/arc/segmented-control/segmented-control';
import { Tooltip } from '../../components/arc/tooltip/tooltip';
import {
  bytesOfTotal,
  DownloadBar,
  DownloadProgress,
  progressFileLine,
  progressPercent,
  progressSummary,
  useDownloadProgress,
  type ProgressState,
} from '../../components/download-progress';
import { modelPresent, shownBuild } from '../../lib/catalog';
import { cn } from '../../lib/cn';
import { errMessage } from '../../lib/errors';
import { formatBytes, formatBytesParts, formatCount, formatCountNoun } from '../../lib/format';
import { truncateMiddle } from '../../lib/labels';
import { pickFolder } from '../../lib/pick-folder';
import { toast } from '../../lib/toast';
import type {
  AdoptPlanResponse,
  CatalogBlocker,
  CatalogBuildItem,
  CatalogItem,
  LocalModel,
  LocatePlanResponse,
  ModelsResponse,
} from '../../lib/types';
import styles from './catalog.module.css';
import {
  entryDownloads,
  type ActiveJob,
  type CatalogDownloadProps,
  type DownloadItem,
  type ModelDownloads,
} from './use-model-downloads';

const GB = 1e9;
const GIB = 2 ** 30;

type HostWarning = { kind: 'disk'; free: number } | { kind: 'memory'; memory: number };

function hostWarning(
  build: Pick<CatalogBuildItem, 'sizeGb' | 'present'>,
  minMemoryGb: number | undefined,
  host: ModelsResponse['host'] | undefined,
): HostWarning | null {
  if (host === undefined) return null;
  if (!build.present && host.diskFreeBytes !== null && build.sizeGb * GB > host.diskFreeBytes) {
    return { kind: 'disk', free: host.diskFreeBytes };
  }
  if (minMemoryGb === undefined && build.sizeGb * GIB > host.totalMemBytes) {
    return { kind: 'memory', memory: host.totalMemBytes };
  }
  return null;
}

/** The one note a catalog card or an onboarding choice shows about room on this Mac for `build`, or null. */
export function hostNote(
  build: Pick<CatalogBuildItem, 'sizeGb' | 'present'>,
  minMemoryGb: number | undefined,
  host: ModelsResponse['host'] | undefined,
): string | null {
  const warning = hostWarning(build, minMemoryGb, host);
  if (warning === null) return null;
  if (warning.kind === 'disk')
    return `Needs about ${build.sizeGb} GB. ${formatBytes(warning.free)} is free on this disk.`;
  return `Weights are larger than this Mac's memory (${formatBytes(warning.memory)}).`;
}

/** The short form of {@link hostNote} that the build line of a card shows, or null. */
export function hostLine(
  build: Pick<CatalogBuildItem, 'sizeGb' | 'present'>,
  minMemoryGb: number | undefined,
  host: ModelsResponse['host'] | undefined,
): string | null {
  const warning = hostWarning(build, minMemoryGb, host);
  if (warning === null) return null;
  if (warning.kind === 'disk') return `${build.sizeGb} GB · only ${formatBytes(warning.free)} free on this disk`;
  return `${build.sizeGb} GB · more than this Mac's ${formatBytes(warning.memory)}`;
}

/** `as="span"` inside an Alert, whose body is a `<p>`. */
export function HostNote({ text, as: Tag = 'p' }: { text: string; as?: 'p' | 'span' }): ReactElement {
  return (
    <Tag className="m-0 flex items-start gap-1.5 text-caption text-warning">
      <AlertTriangle size={12} aria-hidden="true" className="mt-0.5 shrink-0" />
      <span>{text}</span>
    </Tag>
  );
}

/** A byte size whose number and unit never wrap apart. */
function sizeText(bytes: number): string {
  const { value, unit } = formatBytesParts(bytes);
  return unit === '' ? value : `${value} ${unit}`;
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
  const { phrase, noun } = blockerPhrase(item.blocker);
  const contents = blockerContents(item.blocker);
  const name = <span className="font-medium text-foreground">{item.slug}</span>;
  return {
    title: handlers.companion === true ? 'Reinstall companion weights?' : 'Reinstall model?',
    description: (
      <>
        {name} is in the way: {phrase} that the app did not create. This downloads the model, then permanently deletes
        the {noun} {name} and installs the model in its place. {contents === null ? null : <>{contents} </>}
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

export interface UpdateHandlers {
  companion?: boolean;
  onUpdateInPlace: (token: string) => Promise<void>;
  onReinstallStarted: () => void;
}

function filesText(count: number): string {
  return formatCountNoun(count, 'file', 'files', formatCount);
}

/** The confirmation of an Update of `slug` in place, in the numbers of `plan`; a part with a zero count is left out. */
export function updateContent(slug: string, plan: AdoptPlanResponse, handlers: UpdateHandlers): ConfirmContent {
  const { check, download, other } = plan;
  const name = <span className="font-medium text-foreground">{slug}</span>;
  return {
    title: handlers.companion === true ? 'Update companion weights?' : 'Update model?',
    description: (
      <>
        {name} holds a model that the app did not create.{' '}
        {check.count > 0 ? (
          <>
            Update checks {filesText(check.count)} ({sizeText(check.bytes)}) in {name} against the catalog.{' '}
          </>
        ) : null}
        {download.count > 0 ? (
          <>
            {check.count > 0 ? 'It' : 'Update'} downloads {filesText(download.count)} ({sizeText(download.bytes)}) and
            adds {download.count === 1 ? 'it' : 'them'} to the folder.{' '}
          </>
        ) : null}
        {other.count > 0 ? (
          <>
            {other.count === 1
              ? '1 other file stays where it is'
              : `${formatCount(other.count)} other files stay where they are`}
            {other.names.length > 0 ? (
              <>
                {': '}
                {other.names.map((file, index) => (
                  <span key={file} title={file}>
                    {index > 0 ? ', ' : ''}
                    {truncateMiddle(file, NAME_MAX_CHARS)}
                  </span>
                ))}
              </>
            ) : null}
            .{' '}
          </>
        ) : null}
        Update deletes nothing and moves nothing. If a file is not the catalog file, Update stops and changes nothing.
      </>
    ),
    notice:
      'After Update the app manages this folder. A later update replaces it. Files that you add or change in it are then moved to .staging in the models folder. The app does not delete them.',
    confirmLabel: 'Update',
    workingLabel: 'Starting…',
    successLabel: 'Started',
    tone: 'neutral',
    errorTitle: 'Could not start the update',
    onConfirm: () => handlers.onUpdateInPlace(plan.token),
    onDone: handlers.onReinstallStarted,
  };
}

const FOLDER_MAX_CHARS = 48;

/** A folder path in a sentence: shortened in the middle, the whole path in its title. */
function FolderName({ path }: { path: string }): ReactElement {
  return (
    <span className="font-medium text-foreground" title={path}>
      {truncateMiddle(path, FOLDER_MAX_CHARS)}
    </span>
  );
}

export interface LocateHandlers {
  onLocate: (plan: LocatePlanResponse) => Promise<void>;
  onStarted: () => void;
}

/** The confirmation of a Locate of `name` (`Qwen3.8-27B UD-Q4_K_XL`), in the numbers and the mode of `plan`. */
export function locateContent(name: string, plan: LocatePlanResponse, handlers: LocateHandlers): ConfirmContent {
  const { use, download } = plan;
  return {
    title: 'Use these files?',
    description: (
      <>
        The app found {filesText(use.count)} ({sizeText(use.bytes)}) of {name} in <FolderName path={plan.folder} />.{' '}
        {plan.byRecord
          ? `This folder has this app's install record for ${name}. The app trusts the record: it checks that the files are there and does not read them. `
          : 'So far it checked names and sizes. Next it reads each file once to check its content. This can take a few minutes. '}
        {plan.mode === 'shared'
          ? 'Nothing is copied: the models folder shares these files on disk, so they take no extra space.'
          : 'Nothing is copied: the app links to these files. The model works only while this disk is connected.'}
        {download.count > 0
          ? ` It downloads ${filesText(download.count)} (${sizeText(download.bytes)}) into the models folder.`
          : null}
      </>
    ),
    notice: 'Nothing in your folder is changed, moved or deleted.',
    confirmLabel: 'Use files',
    workingLabel: 'Starting…',
    successLabel: 'Started',
    tone: 'neutral',
    errorTitle: 'Could not start Locate',
    onConfirm: () => handlers.onLocate(plan),
    onDone: handlers.onStarted,
  };
}

/** The confirmation of an update of a located build: the job downloads the whole build into the models folder. */
export function updateLocatedContent(
  item: { sizeGb: number; located: { from: string } },
  handlers: { onDownload: () => Promise<void>; onStarted: () => void },
): ConfirmContent {
  return {
    title: 'Download the new version?',
    description: (
      <>
        The app downloads it (about {item.sizeGb} GB) into the models folder. The files in{' '}
        <FolderName path={item.located.from} /> are not changed, and the app stops using them.
      </>
    ),
    confirmLabel: 'Download',
    workingLabel: 'Starting…',
    successLabel: 'Started',
    tone: 'neutral',
    errorTitle: 'Could not start the download',
    onConfirm: handlers.onDownload,
    onDone: handlers.onStarted,
  };
}

/** The sentence of a located build whose files cannot be reached. */
export function missingText(from: string): string {
  return `${from} is not available. Connect the disk, or locate the files again.`;
}

/**
 * What a build's folder offers, first match: its job, an update, installed, located files that cannot be reached, a
 * folder holding another build, a full folder the app did not make (Update in place), one it cannot replace, something
 * else in the way (Reinstall), Install.
 */
export type CatalogActionState =
  | 'job'
  | 'update'
  | 'installed'
  | 'missing'
  | 'held'
  | 'adopt'
  | 'stuck'
  | 'replace'
  | 'install';

export function catalogActionState(
  item: Pick<DownloadItem, 'present' | 'heldBy' | 'blocker' | 'located'>,
  download: { job: ActiveJob | undefined; updateAvailable: boolean },
): CatalogActionState {
  // A live job is the only safe gate and must not be qualified by catalog state: an update re-installs something
  // already present, and the update probe may still be loading.
  if (download.job !== undefined) return 'job';
  if (item.present && download.updateAvailable) return 'update';
  // `present`, not `installed`: an unowned install (mlx download) cannot be overwritten, so it gets no Install.
  if (item.present) return 'installed';
  if (item.located?.filesAvailable === false) return 'missing';
  if (item.heldBy !== null) return 'held';
  const blocker = item.blocker;
  if (blocker?.holdsModel === true && blocker.kind === 'directory' && !blocker.truncated) return 'adopt';
  if (blocker?.holdsModel === true) return 'stuck';
  if (blocker !== null) return 'replace';
  return 'install';
}

interface UpdateInPlaceButtonProps extends UpdateHandlers {
  slug: string;
  settling: boolean;
  label: string;
  variant: 'secondary' | 'ghost';
  align: 'start' | 'end';
  withIcon: boolean;
  onPlanUpdate: () => Promise<AdoptPlanResponse>;
  /** A click that loads a new plan. */
  onPlanStart?: () => void;
  onPlanError: (message: string) => void;
}

/**
 * Update over a real folder that holds a model the app did not create. The plan is loaded on the click and the
 * confirmation opens only with it; closing the confirmation drops it, so the next click loads a new one.
 */
function UpdateInPlaceButton({
  slug,
  settling,
  label,
  variant,
  align,
  withIcon,
  companion,
  onPlanUpdate,
  onPlanStart,
  onPlanError,
  onUpdateInPlace,
  onReinstallStarted,
}: UpdateInPlaceButtonProps): ReactElement {
  const [loading, setLoading] = useState(false);
  const [plan, setPlan] = useState<AdoptPlanResponse | null>(null);
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const refocus = useRef(false);
  useEffect(() => {
    if (plan !== null || !refocus.current) return;
    refocus.current = false;
    buttonRef.current?.focus({ preventScroll: true });
  }, [plan]);
  const load = async (): Promise<void> => {
    if (loading || settling) return;
    setLoading(true);
    onPlanStart?.();
    try {
      setPlan(await onPlanUpdate());
    } catch (err) {
      onPlanError(errMessage(err));
    } finally {
      setLoading(false);
    }
  };
  const icon = withIcon ? <Download size={16} aria-hidden="true" /> : undefined;
  if (plan !== null) {
    return (
      <ConfirmPopover
        trigger={{ kind: 'button', label, variant, size: 'sm', icon }}
        align={align}
        open
        onOpenChange={(open) => {
          if (open) return;
          refocus.current = true;
          setPlan(null);
        }}
        {...updateContent(slug, plan, { companion, onUpdateInPlace, onReinstallStarted })}
      />
    );
  }
  const button = (
    <Button
      ref={buttonRef}
      type="button"
      variant={variant}
      size="sm"
      loading={loading}
      aria-disabled={settling ? true : undefined}
      onClick={() => void load()}
    >
      {loading ? null : icon}
      {loading ? 'Checking…' : label}
    </Button>
  );
  return settling ? <Tooltip content="Checking the last download.">{button}</Tooltip> : button;
}

function UpdateInPlaceAction(
  props: Omit<UpdateInPlaceButtonProps, 'label' | 'variant' | 'align' | 'withIcon' | 'onPlanStart' | 'onPlanError'>,
): ReactElement {
  const [planError, setPlanError] = useState<string | null>(null);
  return (
    <Alert tone="warning" title="Needs update">
      <span className="grid justify-items-start gap-2">
        <span>
          <span className="font-mono">{props.slug}</span> holds a model that the app did not create. Update checks its
          files against the catalog and downloads the files that are missing. It deletes nothing.
        </span>
        {planError !== null ? <HostNote text={planError} as="span" /> : null}
        <UpdateInPlaceButton
          {...props}
          label="Update"
          variant="secondary"
          align="start"
          withIcon
          onPlanStart={() => setPlanError(null)}
          onPlanError={setPlanError}
        />
      </span>
    </Alert>
  );
}

/** Locate on one build: the folder dialog, the server's check, a refusal, or a plan to confirm. */
export type LocateView =
  | { step: 'idle' }
  | { step: 'pick' }
  | { step: 'plan' }
  | { step: 'refused'; message: string }
  | { step: 'confirm'; plan: LocatePlanResponse };

export interface LocateControl {
  view: LocateView;
  /** Opens the folder dialog, then checks the folder. Does nothing while the dialog or the check runs. */
  start: () => void;
  /** Closes the confirmation, or forgets a refusal. */
  close: () => void;
}

const LOCATE_IDLE: LocateView = { step: 'idle' };

/** Locate for the build that `download` installs. A cancelled dialog goes back to idle and says nothing. */
export function useLocate(download: Pick<CatalogDownloadProps, 'item' | 'onPlanLocate'>): LocateControl {
  const slug = download.item.slug;
  const [state, setState] = useState<{ slug: string; view: LocateView }>({ slug, view: LOCATE_IDLE });
  const busy = useRef(false);
  const plan = download.onPlanLocate;
  const start = (): void => {
    if (busy.current) return;
    busy.current = true;
    setState({ slug, view: { step: 'pick' } });
    void (async () => {
      try {
        const path = await pickFolder('model');
        if (path === null) {
          setState({ slug, view: LOCATE_IDLE });
          return;
        }
        setState({ slug, view: { step: 'plan' } });
        const found = await plan(path);
        setState({ slug, view: { step: 'confirm', plan: found } });
      } catch (err) {
        setState({ slug, view: { step: 'refused', message: errMessage(err) } });
      } finally {
        busy.current = false;
      }
    })();
  };
  const close = (): void => setState((prev) => (prev.slug === slug ? { slug, view: LOCATE_IDLE } : prev));
  return { view: state.slug === slug ? state.view : LOCATE_IDLE, start, close };
}

interface LocateButtonProps {
  control: LocateControl;
  /** The model the confirmation names: `Qwen3.8-27B UD-Q4_K_XL`. */
  name: string;
  download: Pick<CatalogDownloadProps, 'onLocate' | 'onReinstallStarted'>;
  size: 'sm' | 'md';
  align: 'start' | 'end';
  /** Install is starting, or the last job is settling (`reason` says so). */
  disabled: boolean;
  reason?: string;
  className?: string;
}

/**
 * Locate…: picks a folder, shows "Checking…" while the server checks it, then grows into the confirmation. The button
 * keeps its focus while it works (`aria-disabled`), and gets it back when the confirmation closes.
 */
function LocateButton({
  control,
  name,
  download,
  size,
  align,
  disabled,
  reason,
  className,
}: LocateButtonProps): ReactElement {
  const { view } = control;
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const refocus = useRef(false);
  useEffect(() => {
    if (view.step === 'confirm' || !refocus.current) return;
    refocus.current = false;
    buttonRef.current?.focus({ preventScroll: true });
  }, [view.step]);
  if (view.step === 'confirm') {
    return (
      <ConfirmPopover
        trigger={{ kind: 'button', label: 'Locate…', variant: 'ghost', size }}
        align={align}
        open
        onOpenChange={(open) => {
          if (open) return;
          refocus.current = true;
          control.close();
        }}
        {...locateContent(name, view.plan, {
          onLocate: download.onLocate,
          onStarted: () => {
            control.close();
            download.onReinstallStarted();
          },
        })}
      />
    );
  }
  const busy = view.step === 'pick' || view.step === 'plan';
  const off = disabled && !busy;
  const button = (
    <Button
      ref={buttonRef}
      type="button"
      variant="ghost"
      size={size}
      className={className}
      disabled={off}
      aria-disabled={busy ? true : undefined}
      onClick={() => {
        if (!busy) control.start();
      }}
    >
      {/* The word stays on screen: the check can take some seconds, and a bare spinner does not say what runs. */}
      {view.step === 'plan' ? <MorphLoader size={14} status="loading" spinning className="text-muted" /> : null}
      {view.step === 'plan' ? 'Checking…' : 'Locate…'}
    </Button>
  );
  return (
    <Tooltip content={off && reason !== undefined ? reason : 'Use model files that are already on this Mac'}>
      {off ? <span className="inline-flex">{button}</span> : button}
    </Tooltip>
  );
}

/** The model name a Locate confirmation uses: the entry, and the build when the entry has more than one. */
export function locateName(entry: Pick<CatalogItem, 'label' | 'builds'>, build: Pick<CatalogBuildItem, 'id'>): string {
  return entry.builds.length > 1 ? `${entry.label} ${build.id}` : entry.label;
}

export interface CatalogDownloadActionProps extends CatalogDownloadProps {
  /** The name of what `item` installs (`UD-Q4_K_XL`), for a folder that holds another build. */
  buildName: string;
  /** The model a Locate confirmation names ({@link locateName}). */
  modelName: string;
}

/** Onboarding's action for one build: progress, Update, the Installed status, the cleanup note, or Install. */
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
  onPlanUpdate,
  onUpdateInPlace,
  onPlanLocate,
  onLocate,
  onRecheck,
  onReplaceLocated,
  onDone,
  onError,
  onCancelled,
  onCancel,
  buildName,
  modelName,
}: CatalogDownloadActionProps): ReactElement {
  const [starting, setStarting] = useState(false);
  const locate = useLocate({ item, onPlanLocate });
  const start = async (): Promise<void> => {
    if (starting) return;
    setStarting(true);
    try {
      await onInstall();
    } finally {
      setStarting(false);
    }
  };
  const state = catalogActionState(item, { job, updateAvailable });
  if (state === 'job' && job !== undefined) {
    return (
      <div className="grid gap-2">
        <DownloadProgress id={job.id} onDone={onDone} onError={onError} onCancelled={onCancelled} />
        {item.blocker !== null ? <p className="m-0 text-caption text-secondary">{jobBlockerLine(item)}</p> : null}
        {/* cancel() refuses a committing job, and a second cancel answers 404. */}
        {!job.committing && !job.cancelling ? (
          <Button type="button" variant="ghost" size="sm" className="w-full" onClick={() => onCancel(job.id)}>
            Cancel
          </Button>
        ) : null}
      </div>
    );
  }
  if (state === 'update' && item.located !== null) {
    return (
      <ConfirmPopover
        trigger={{
          kind: 'button',
          label: 'Update available',
          variant: 'secondary',
          size: 'md',
          icon: <Download size={16} aria-hidden="true" />,
        }}
        align="start"
        disabled={settling || updateSettling}
        disabledReason="Checking the last download."
        {...updateLocatedContent(
          { sizeGb: item.sizeGb, located: item.located },
          { onDownload: onReplaceLocated, onStarted: onReinstallStarted },
        )}
      />
    );
  }
  if (state === 'update') {
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
  if (state === 'installed') {
    return (
      <p className="m-0 flex min-h-8 items-center gap-1.5 text-body font-medium text-secondary" data-status="installed">
        <MorphLoader size={14} status="success" />
        <span>Installed</span>
      </p>
    );
  }
  const locateButton = (
    <LocateButton
      control={locate}
      name={modelName}
      download={{ onLocate, onReinstallStarted }}
      size="sm"
      align="start"
      disabled={starting || settling}
      reason={settling ? 'Checking the last download.' : undefined}
    />
  );
  const refusal =
    locate.view.step === 'refused' ? (
      <HostNote text={locate.view.message} as={state === 'missing' ? 'span' : 'p'} />
    ) : null;
  if (state === 'missing' && item.located !== null) {
    return (
      <Alert tone="warning" title="Files not found">
        <span className="grid justify-items-start gap-2">
          <span>{missingText(item.located.from)}</span>
          {refusal}
          <span className="flex flex-wrap items-center gap-2">
            <Button type="button" variant="secondary" size="sm" onClick={onRecheck}>
              Check again
            </Button>
            {locateButton}
            <Button
              type="button"
              variant="ghost"
              size="sm"
              disabled={starting || settling}
              onClick={() => void start()}
            >
              Download instead
            </Button>
          </span>
        </span>
      </Alert>
    );
  }
  if (state === 'held') {
    return (
      <Alert tone="warning" title="Needs cleanup">
        {heldSentence(item.slug, item.heldBy ?? '', buildName)}
      </Alert>
    );
  }
  if (state === 'adopt') {
    return (
      <UpdateInPlaceAction
        slug={item.slug}
        settling={settling}
        onPlanUpdate={onPlanUpdate}
        onUpdateInPlace={onUpdateInPlace}
        onReinstallStarted={onReinstallStarted}
      />
    );
  }
  if (state === 'stuck' && item.blocker !== null) {
    return <Alert tone="warning" title="Needs cleanup">{`${stuckSentence(item.slug, item.blocker)}`}</Alert>;
  }
  if (state === 'replace' && item.blocker !== null) {
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
            {...reinstallContent({ slug: item.slug, blocker: item.blocker }, { onReinstall, onReinstallStarted })}
          />
        </span>
      </Alert>
    );
  }
  const locating = locate.view.step === 'pick' || locate.view.step === 'plan';
  return (
    <div className="grid justify-items-center gap-2">
      <Button
        type="button"
        variant={prominent ? 'primary' : 'secondary'}
        size={prominent ? 'lg' : 'md'}
        className="w-full"
        loading={starting}
        disabled={starting || settling || locating}
        onClick={() => {
          locate.close();
          void start();
        }}
      >
        {starting ? null : <Download size={16} aria-hidden="true" />}
        {starting ? 'Starting download…' : settling ? settlingLabel : installLabel}
      </Button>
      {locateButton}
      {refusal}
    </div>
  );
}

/** What a job does to the old folder in its way. */
function jobBlockerLine(item: Pick<DownloadItem, 'slug' | 'blocker'>): string | null {
  if (item.blocker === null) return null;
  return item.blocker.holdsModel
    ? `Checks the files in ${item.slug}. Nothing is deleted.`
    : `Replaces the old ${blockerPhrase(item.blocker).noun} when the download is complete.`;
}

function heldSentence(slug: string, heldBy: string, build: string): string {
  return `${slug} holds ${heldBy}. Delete it in Local models to install ${build} here.`;
}

function stuckSentence(slug: string, blocker: CatalogBlocker): string {
  return `${slug} holds a model that the app did not create, so Reinstall cannot replace it. ${
    blocker.listed
      ? 'Delete it in Local models to install this one.'
      : 'Move or delete it in Finder to install this one.'
  }`;
}

/** `/models?focus=…` for `slug` when Local models lists it. */
export function detailsLink(localModels: readonly Pick<LocalModel, 'name'>[], slug: string): string | undefined {
  return localModels.some((model) => model.name === slug) ? `/models?focus=${encodeURIComponent(slug)}` : undefined;
}

function hfUrl(repo: string): string {
  return `https://huggingface.co/${repo}`;
}

/** The build each card shows on this page visit, by entry label. Not stored: a new visit starts from the rule. */
export function useChosenBuilds(): {
  chosen: Readonly<Record<string, string>>;
  choose: (label: string, slug: string) => void;
} {
  const [chosen, setChosen] = useState<Readonly<Record<string, string>>>({});
  const choose = useCallback((label: string, slug: string): void => {
    setChosen((prev) => (prev[label] === slug ? prev : { ...prev, [label]: slug }));
  }, []);
  return { chosen, choose };
}

type BuildMarkKind = 'downloading' | 'attention' | 'update' | 'installed';

const MARK_WORD: Record<BuildMarkKind, string> = {
  downloading: 'downloading',
  attention: 'needs attention',
  update: 'update available',
  installed: 'installed',
};

/** The mark on a build's segment, first match, or null for a build that is simply absent. */
export function buildMark(
  download: Pick<CatalogDownloadProps, 'item' | 'job' | 'updateAvailable'>,
): BuildMarkKind | null {
  if (download.job !== undefined) return 'downloading';
  const { item } = download;
  if (item.blocker !== null || item.heldBy !== null) return 'attention';
  if (!item.present && item.located?.filesAvailable === false) return 'attention';
  if (download.item.present && download.updateAvailable) return 'update';
  if (download.item.present) return 'installed';
  return null;
}

function BuildMark({ kind }: { kind: BuildMarkKind | null }): ReactElement | null {
  if (kind === null) return null;
  return (
    <span className="inline-flex items-center" data-mark={kind}>
      {kind === 'downloading' ? (
        <MorphLoader size={14} status="loading" spinning className="text-muted" />
      ) : kind === 'attention' ? (
        <AlertTriangle size={12} aria-hidden="true" className="text-warning" />
      ) : kind === 'update' ? (
        <ArrowUp size={12} aria-hidden="true" className="text-warning" />
      ) : (
        <Check size={12} aria-hidden="true" className="text-success" />
      )}
      <span className="sr-only">{MARK_WORD[kind]}</span>
    </span>
  );
}

/** One line that never wraps: the text truncates and the Tooltip holds all of it. */
function OneLine({
  tooltip,
  className,
  children,
}: {
  tooltip: ReactNode;
  className?: string;
  children: ReactNode;
}): ReactElement {
  return (
    <Tooltip content={tooltip}>
      <span className={cn('flex min-w-0 items-center gap-1.5', className)}>{children}</span>
    </Tooltip>
  );
}

/** A warning that fits one line: the line opens a popover with all of it, for the pointer and the keyboard. */
function WarningLine({ line, title, text }: { line: string; title: string; text: string }): ReactElement {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="flex max-w-full min-w-0 cursor-pointer items-center gap-1.5 text-start text-caption text-warning"
        >
          <AlertTriangle size={12} aria-hidden="true" className="shrink-0" />
          <span className="min-w-0 truncate">{line}</span>
        </button>
      </PopoverTrigger>
      <PopoverContent align="start">
        <p className="m-0 text-body font-medium text-foreground">{title}</p>
        <p className="m-0 mt-1 text-body text-secondary">{text}</p>
      </PopoverContent>
    </Popover>
  );
}

function BuildLine({
  entry,
  build,
  download,
  progress,
  host,
  refusal,
}: {
  entry: CatalogItem;
  build: CatalogBuildItem;
  download: CatalogDownloadProps;
  progress: ProgressState;
  host: ModelsResponse['host'] | undefined;
  /** Why the folder the user picked for Locate cannot be used. */
  refusal: string | null;
}): ReactElement {
  const job = download.job;
  const located = download.item.located;
  if (job === undefined && (refusal !== null || (located !== null && !located.filesAvailable && !build.present))) {
    return (
      <p className={cn('m-0', styles.line)}>
        {refusal !== null ? (
          <WarningLine line={refusal} title="Could not use this folder" text={refusal} />
        ) : located !== null ? (
          <WarningLine
            line={`Files not found · ${located.from}`}
            title="Files not found"
            text={missingText(located.from)}
          />
        ) : null}
      </p>
    );
  }
  if (job === undefined && located !== null && build.present) {
    const text = `Uses files in ${located.from}`;
    return (
      <p className={cn('m-0 text-caption text-secondary', styles.line)} data-located={located.mode}>
        <OneLine tooltip={text}>
          <span className="min-w-0 truncate">Uses files in {truncateMiddle(located.from, FOLDER_MAX_CHARS)}</span>
        </OneLine>
      </p>
    );
  }
  const warning = job === undefined ? hostLine(build, entry.minMemoryGb, host) : null;
  const text =
    job !== undefined
      ? progressSummary(progress, job)
      : (warning ?? [`${build.sizeGb} GB`, build.source, build.note].filter((part) => part !== undefined).join(' · '));
  return (
    <p
      className={cn('m-0 text-caption tabular-nums', warning !== null ? 'text-warning' : 'text-secondary', styles.line)}
    >
      <OneLine tooltip={warning !== null ? (hostNote(build, entry.minMemoryGb, host) ?? text) : text}>
        {warning !== null ? <AlertTriangle size={12} aria-hidden="true" className="shrink-0" /> : null}
        <span className="min-w-0 truncate">{text}</span>
      </OneLine>
    </p>
  );
}

function StatusWord({ text, tooltip }: { text: string; tooltip: string }): ReactElement {
  return (
    <OneLine tooltip={tooltip} className="text-body font-medium text-warning">
      <AlertTriangle size={14} aria-hidden="true" className="shrink-0" />
      <span className="min-w-0 truncate">{text}</span>
      {/* The trigger cannot take focus, so the hover Tooltip never reaches a keyboard or screen reader user. */}
      <span className="sr-only">{tooltip}</span>
    </OneLine>
  );
}

function ShowWhy({ text, variant }: { text: string; variant: 'secondary' | 'ghost' }): ReactElement {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button type="button" variant={variant} size="sm">
          Show why
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end">
        <p className="m-0 text-body font-medium text-foreground">Needs cleanup</p>
        <p className="m-0 mt-1 text-body text-secondary">{text}</p>
      </PopoverContent>
    </Popover>
  );
}

function InstalledText({
  detailsTo,
  openTo,
}: {
  detailsTo: string | undefined;
  openTo: string | undefined;
}): ReactElement {
  return (
    <p className="m-0 flex min-w-0 items-center gap-1.5 text-body font-medium text-secondary" data-status="installed">
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
      {openTo !== undefined ? (
        <>
          <span aria-hidden="true">·</span>
          <Link className="link" to={openTo}>
            Open
          </Link>
        </>
      ) : null}
    </p>
  );
}

interface RowProps {
  entry: CatalogItem;
  build: CatalogBuildItem;
  download: CatalogDownloadProps;
  progress: ProgressState;
  starting: boolean;
  onStart: (download: CatalogDownloadProps) => void;
  detailsTo: string | undefined;
  openTo: string | undefined;
  locate: LocateControl | undefined;
}

/** The bottom row: one control that acts on the chosen build, and at most one in the right slot. */
function ActionRow({
  entry,
  build,
  download,
  progress,
  starting,
  onStart,
  detailsTo,
  openTo,
  locate,
}: RowProps): ReactElement {
  const { item, job, settling, updateSettling } = download;
  const state = catalogActionState(item, download);
  if (state === 'job' && job !== undefined) {
    const fileLine = progressFileLine(progress);
    const blockerLine = jobBlockerLine(item);
    const lines = [fileLine, blockerLine].filter((line): line is string => line !== null);
    const moving = (progress.phase === 'connecting' || progress.phase === 'running') && !progress.stalled;
    const bar = (
      <span className="block min-w-0 flex-1">
        <DownloadBar state={progress} label={fileLine ?? 'Download progress'} />
      </span>
    );
    return (
      <>
        <div className={styles.primary}>
          <MorphLoader size={14} status="loading" spinning={moving} className="text-muted" />
          {lines.length > 0 ? (
            <Tooltip
              content={
                <span className="grid">
                  {lines.map((line) => (
                    <span key={line}>{line}</span>
                  ))}
                </span>
              }
            >
              {bar}
            </Tooltip>
          ) : (
            bar
          )}
          <span className={cn('text-caption text-secondary tabular-nums', styles.percent)}>
            {progressPercent(progress)}%
          </span>
          {blockerLine !== null ? <span className="sr-only">{blockerLine}</span> : null}
        </div>
        {/* cancel() refuses a committing job, and a second cancel answers 404. */}
        {!job.committing && !job.cancelling ? (
          <Button type="button" variant="ghost" size="md" onClick={() => download.onCancel(job.id)}>
            Cancel
          </Button>
        ) : null}
      </>
    );
  }
  if (state === 'installed' || state === 'update') {
    const located = item.located;
    return (
      <>
        <div className={styles.primary}>
          <InstalledText detailsTo={detailsTo} openTo={openTo} />
        </div>
        {state === 'update' && located !== null ? (
          <ConfirmPopover
            trigger={{
              kind: 'button',
              label: 'Update',
              variant: 'secondary',
              size: 'sm',
              icon: <Download size={16} aria-hidden="true" />,
            }}
            align="end"
            disabled={starting || settling || updateSettling}
            disabledReason="Checking the last download."
            {...updateLocatedContent(
              { sizeGb: build.sizeGb, located },
              { onDownload: download.onReplaceLocated, onStarted: download.onReinstallStarted },
            )}
          />
        ) : state === 'update' ? (
          <Button
            type="button"
            variant="secondary"
            size="sm"
            disabled={starting || settling || updateSettling}
            onClick={() => onStart(download)}
          >
            <Download size={16} aria-hidden="true" />
            Update
          </Button>
        ) : null}
      </>
    );
  }
  const locateButton = (size: 'sm' | 'md') =>
    locate === undefined ? null : (
      <LocateButton
        control={locate}
        name={locateName(entry, build)}
        download={download}
        size={size}
        align="end"
        disabled={starting || settling}
        reason={settling ? 'Checking the last download.' : undefined}
      />
    );
  const locating = locate?.view.step === 'pick' || locate?.view.step === 'plan';
  if (state === 'missing') {
    return (
      <>
        <div className={styles.primary}>
          <Button type="button" variant="secondary" size="sm" onClick={download.onRecheck}>
            Check again
          </Button>
        </div>
        {locateButton('sm')}
        <Button
          type="button"
          variant="ghost"
          size="sm"
          disabled={starting || settling || locating}
          onClick={() => {
            locate?.close();
            onStart(download);
          }}
        >
          Download instead
        </Button>
      </>
    );
  }
  if (state === 'held') {
    const heldBy = item.heldBy ?? '';
    const sentence = heldSentence(item.slug, heldBy, build.id);
    return (
      <>
        <div className={styles.primary}>
          <StatusWord text={`Holds ${heldBy}`} tooltip={sentence} />
        </div>
        {detailsTo !== undefined ? (
          <ButtonLink to={detailsTo} variant="secondary" size="sm">
            Show in Local models
          </ButtonLink>
        ) : (
          <ShowWhy text={sentence} variant="secondary" />
        )}
      </>
    );
  }
  if (state === 'adopt') {
    return (
      <>
        <div className={styles.primary}>
          <StatusWord text="Needs update" tooltip={`${item.slug} holds a model that the app did not create.`} />
        </div>
        <UpdateInPlaceButton
          slug={item.slug}
          settling={settling}
          label="Update…"
          variant="secondary"
          align="end"
          withIcon={false}
          onPlanUpdate={download.onPlanUpdate}
          onPlanError={(message) => toast.error('Could not check the folder', { description: message })}
          onUpdateInPlace={download.onUpdateInPlace}
          onReinstallStarted={download.onReinstallStarted}
        />
      </>
    );
  }
  if (state === 'stuck' && item.blocker !== null) {
    return (
      <>
        <div className={styles.primary}>
          <StatusWord text="Needs cleanup" tooltip={`${item.slug} holds a model that the app did not create.`} />
        </div>
        {detailsTo !== undefined && item.blocker.listed ? (
          <ButtonLink to={detailsTo} variant="secondary" size="sm">
            Show in Local models
          </ButtonLink>
        ) : (
          <ShowWhy text={stuckSentence(item.slug, item.blocker)} variant="secondary" />
        )}
      </>
    );
  }
  if (state === 'replace' && item.blocker !== null) {
    return (
      <>
        <div className={styles.primary}>
          <StatusWord text="Needs cleanup" tooltip={`${item.slug} is in the way.`} />
        </div>
        <ConfirmPopover
          trigger={{ kind: 'button', label: 'Reinstall…', variant: 'secondary', size: 'sm' }}
          align="end"
          disabled={settling}
          disabledReason="Checking the last download."
          {...reinstallContent(
            { slug: item.slug, blocker: item.blocker },
            { onReinstall: download.onReinstall, onReinstallStarted: download.onReinstallStarted },
          )}
        />
      </>
    );
  }
  const install = (
    <Button
      type="button"
      variant="secondary"
      size="md"
      className="max-w-72"
      loading={starting}
      disabled={starting || settling || locating}
      onClick={() => {
        locate?.close();
        onStart(download);
      }}
    >
      {starting ? null : <Download size={16} aria-hidden="true" />}
      {starting ? 'Starting download…' : entry.builds.length > 1 ? `Install ${build.id}` : 'Install'}
    </Button>
  );
  return (
    <>
      <div className={styles.primary}>
        {settling && !starting ? (
          <Tooltip content="Checking the last download.">
            <span className="inline-flex min-w-0">{install}</span>
          </Tooltip>
        ) : (
          install
        )}
      </div>
      {locateButton('md')}
    </>
  );
}

interface DraftLineProps {
  entry: CatalogItem;
  draft: NonNullable<CatalogItem['draft']>;
  download: CatalogDownloadProps;
  progress: ProgressState;
  starting: boolean;
  onStart: (download: CatalogDownloadProps) => void;
}

/** The draft weights of an entry: one line with its own state and job, the same for every build of the entry. */
function DraftLine({ entry, draft, download, progress, starting, onStart }: DraftLineProps): ReactElement {
  const { item, job, settling, updateSettling } = download;
  const state = catalogActionState(item, download);
  const both = entry.builds.length === 2 ? ' Works with both builds.' : '';
  let glyph: ReactNode;
  let text: string;
  let tooltip: string;
  let control: ReactNode = null;
  if (state === 'job' && job !== undefined) {
    const moving = (progress.phase === 'connecting' || progress.phase === 'running') && !progress.stalled;
    glyph = <MorphLoader size={14} status="loading" spinning={moving} className="text-muted" />;
    text = job.cancelling
      ? 'cancelling…'
      : job.committing || progress.phase === 'done'
        ? 'finishing…'
        : progress.phase === 'connecting' || progress.totalBytes <= 0
          ? 'starting…'
          : `${bytesOfTotal(progress.receivedBytes, progress.totalBytes)} · ${progressPercent(progress)}%`;
    tooltip = progressFileLine(progress) ?? `${draft.label} · ${text}`;
    if (!job.committing && !job.cancelling) {
      control = (
        <Button type="button" variant="ghost" size="sm" onClick={() => download.onCancel(job.id)}>
          Cancel
        </Button>
      );
    }
  } else if (state === 'update') {
    glyph = <ArrowUp size={12} aria-hidden="true" className="shrink-0 text-warning" />;
    text = 'update available';
    tooltip = `Used when ${entry.label} loads. To remove it, open Local models, Companion weights.`;
    control = (
      <Button
        type="button"
        variant="ghost"
        size="sm"
        disabled={starting || settling || updateSettling}
        onClick={() => onStart(download)}
      >
        Update
      </Button>
    );
  } else if (state === 'installed') {
    glyph = <Check size={12} aria-hidden="true" className="shrink-0 text-success" />;
    const waits = !modelPresent(entry);
    text = waits ? 'added · waits for the model' : 'added';
    tooltip = waits
      ? `Install a ${entry.label} build to use it.`
      : `Used when ${entry.label} loads. To remove it, open Local models, Companion weights.`;
  } else if (state === 'install') {
    glyph = <Badge size="sm">Optional</Badge>;
    text = `${draft.sizeGb} GB`;
    tooltip = `A draft model that can speed up replies. The answers stay the same. Used the next time ${entry.label} loads.${both}`;
    const add = (
      <Button
        type="button"
        variant="ghost"
        size="sm"
        aria-label={starting ? undefined : `Add ${draft.label}, ${draft.sizeGb} GB`}
        loading={starting}
        disabled={starting || settling}
        onClick={() => onStart(download)}
      >
        {starting ? 'Starting…' : 'Add'}
      </Button>
    );
    control =
      settling && !starting ? (
        <Tooltip content="Checking the last download.">
          <span>{add}</span>
        </Tooltip>
      ) : (
        add
      );
  } else {
    glyph = <AlertTriangle size={12} aria-hidden="true" className="shrink-0 text-warning" />;
    const blocker = item.blocker;
    if (state === 'adopt') {
      text = 'needs update';
      tooltip = `${item.slug} holds a model that the app did not create.`;
      control = (
        <UpdateInPlaceButton
          slug={item.slug}
          settling={settling}
          companion
          label="Update…"
          variant="ghost"
          align="end"
          withIcon={false}
          onPlanUpdate={download.onPlanUpdate}
          onPlanError={(message) => toast.error('Could not check the folder', { description: message })}
          onUpdateInPlace={download.onUpdateInPlace}
          onReinstallStarted={download.onReinstallStarted}
        />
      );
    } else if (state === 'replace' && blocker !== null) {
      text = 'needs cleanup';
      tooltip = `${item.slug} is in the way.`;
      control = (
        <ConfirmPopover
          trigger={{ kind: 'button', label: 'Reinstall…', variant: 'ghost', size: 'sm' }}
          align="end"
          disabled={settling}
          disabledReason="Checking the last download."
          {...reinstallContent(
            { slug: item.slug, blocker },
            { companion: true, onReinstall: download.onReinstall, onReinstallStarted: download.onReinstallStarted },
          )}
        />
      );
    } else {
      text = 'needs cleanup';
      tooltip = `${item.slug} holds a model that the app did not create.`;
      control =
        blocker !== null ? (
          <ShowWhy text={stuckSentence(item.slug, blocker)} variant="ghost" />
        ) : (
          <ShowWhy text={heldSentence(item.slug, item.heldBy ?? '', draft.label)} variant="ghost" />
        );
    }
  }
  return (
    <div
      role="group"
      aria-label={`${draft.label} add-on`}
      className={cn('min-w-0 flex-1 justify-between', styles.line)}
    >
      <OneLine tooltip={tooltip} className="text-caption text-secondary tabular-nums">
        {glyph}
        <a
          className="inline-flex shrink-0 items-center gap-1 text-body font-medium text-foreground"
          href={hfUrl(draft.hfRepo)}
          target="_blank"
          rel="noreferrer"
        >
          {draft.label}
          <ExternalLink size={12} aria-hidden="true" className="text-muted" />
        </a>
        <span aria-hidden="true">·</span>
        <span className="min-w-0 truncate" data-draft-state={state}>
          {text}
        </span>
      </OneLine>
      {control}
    </div>
  );
}

export interface CatalogCardViewProps {
  entry: CatalogItem;
  /** The folder of the build the card shows. */
  chosen: string;
  /** A segment was clicked; nothing starts. */
  onChoose: (slug: string) => void;
  /** The download state of every build, in `entry.builds` order. */
  downloads: readonly CatalogDownloadProps[];
  /** The live progress of the chosen build's job. */
  progress: ProgressState;
  draft?: { download: CatalogDownloadProps; progress: ProgressState };
  /** Folders whose start request is in flight. */
  starting: readonly string[];
  onStart: (download: CatalogDownloadProps) => void;
  host: ModelsResponse['host'] | undefined;
  /** `/models?focus=…` when Local models lists the chosen build's folder. */
  detailsTo?: string;
  /** A workspace for the chosen build (`/system-one?model=…`). */
  openTo?: string;
  /** Locate of the chosen build: Locate… beside Install and in the "Files not found" state. */
  locate?: LocateControl;
}

/** One recommended model in five fixed rows: title, description, build, extras, action. */
export function CatalogCardView({
  entry,
  chosen,
  onChoose,
  downloads,
  progress,
  draft,
  starting,
  onStart,
  host,
  detailsTo,
  openTo,
  locate,
}: CatalogCardViewProps): ReactElement {
  const index = Math.max(
    0,
    entry.builds.findIndex((build) => build.slug === chosen),
  );
  const build = entry.builds[index];
  const download = downloads[index];
  const busy = downloads.some((props) => props.job !== undefined) || draft?.download.job !== undefined;
  const included = build.included ?? '';
  const state = catalogActionState(download.item, download);
  const refusal =
    locate?.view.step === 'refused' && (state === 'install' || state === 'missing') ? locate.view.message : null;
  return (
    <div
      data-model={entry.builds[0].slug}
      data-build={build.slug}
      data-catalog-card=""
      data-downloading={busy ? '' : undefined}
      className={styles.card}
    >
      <Panel ariaLabel={entry.label} className={styles.card}>
        <div className={styles.body}>
          <div data-row="title" className="flex min-w-0 items-center justify-between gap-2">
            <a
              className="flex min-w-0 items-center gap-2 text-strong font-medium text-foreground"
              href={hfUrl(build.hfRepo)}
              target="_blank"
              rel="noreferrer"
            >
              <ModelLogo model={entry.label} className="size-[18px] shrink-0" />
              <span className="min-w-0 truncate" title={entry.label}>
                {entry.label}
              </span>
              <ExternalLink size={12} aria-hidden="true" className="shrink-0 text-muted" />
            </a>
            <span className="flex shrink-0 items-center gap-1">
              {entry.category === 'systemone' ? (
                <Badge color="violet" size="sm">
                  System One
                </Badge>
              ) : entry.isDefault === true ? (
                <Badge tone="info" size="sm">
                  Default
                </Badge>
              ) : null}
              {download.item.present && download.updateAvailable ? (
                <Badge tone="warning" size="sm">
                  Update
                </Badge>
              ) : null}
            </span>
          </div>
          <p
            data-row="description"
            className={cn('m-0 text-body text-secondary', styles.description)}
            title={entry.description}
          >
            {entry.description}
          </p>
          <div data-row="build" className={styles.build}>
            <div className={styles.slot}>
              {entry.builds.length > 1 ? (
                <SegmentedControl
                  label={`Build of ${entry.label}`}
                  value={build.slug}
                  onValueChange={onChoose}
                  options={entry.builds.map((candidate, at) => ({
                    value: candidate.slug,
                    label: candidate.id,
                    accessory: <BuildMark kind={buildMark(downloads[at])} />,
                  }))}
                />
              ) : (
                <Badge outline size="lg">
                  {build.id}
                </Badge>
              )}
            </div>
            <BuildLine
              entry={entry}
              build={build}
              download={download}
              progress={progress}
              host={host}
              refusal={refusal}
            />
            <span className="sr-only" aria-live="polite">
              {refusal}
            </span>
          </div>
          <div data-row="extras" className={styles.line}>
            {entry.draft !== undefined && draft !== undefined ? (
              <DraftLine
                entry={entry}
                draft={entry.draft}
                download={draft.download}
                progress={draft.progress}
                starting={starting.includes(draft.download.item.slug)}
                onStart={onStart}
              />
            ) : included !== '' ? (
              <OneLine tooltip={included} className="text-caption text-secondary">
                <span className="min-w-0 truncate">{included}</span>
              </OneLine>
            ) : null}
          </div>
          <div data-row="action" className={styles.action}>
            <ActionRow
              entry={entry}
              build={build}
              download={download}
              progress={progress}
              starting={starting.includes(build.slug)}
              onStart={onStart}
              detailsTo={detailsTo}
              openTo={openTo}
              locate={locate}
            />
          </div>
        </div>
      </Panel>
    </div>
  );
}

/** Subscribes to a job the card does not show, so its end still reaches the page. */
function JobFollower({ download }: { download: CatalogDownloadProps }): null {
  useDownloadProgress(download.job?.id, download);
  return null;
}

export interface CatalogCardProps extends Omit<
  CatalogCardViewProps,
  'progress' | 'draft' | 'starting' | 'onStart' | 'locate'
> {
  draft?: CatalogDownloadProps;
}

/** The live card: one subscription per job of its builds and its draft, and the in-flight start requests. */
export function CatalogCard({ draft, ...props }: CatalogCardProps): ReactElement {
  const index = Math.max(
    0,
    props.entry.builds.findIndex((build) => build.slug === props.chosen),
  );
  const shown = props.downloads[index];
  const progress = useDownloadProgress(shown.job?.id, shown);
  const draftProgress = useDownloadProgress(draft?.job?.id, draft ?? {});
  const locate = useLocate(shown);
  const [starting, setStarting] = useState<readonly string[]>([]);
  const inFlight = useRef(new Set<string>());
  const start = (download: CatalogDownloadProps): void => {
    const slug = download.item.slug;
    if (inFlight.current.has(slug)) return;
    inFlight.current.add(slug);
    setStarting([...inFlight.current]);
    void download.onInstall().finally(() => {
      inFlight.current.delete(slug);
      setStarting([...inFlight.current]);
    });
  };
  return (
    <>
      {props.downloads.map((download, at) =>
        at !== index && download.job !== undefined ? (
          <JobFollower key={download.item.slug} download={download} />
        ) : null,
      )}
      <CatalogCardView
        {...props}
        progress={progress}
        draft={draft === undefined ? undefined : { download: draft, progress: draftProgress }}
        starting={starting}
        onStart={start}
        locate={locate}
      />
    </>
  );
}

/** The card of `item` on a page: the build the page shows, its downloads and its links. */
export function CatalogEntryCard(props: {
  item: CatalogItem;
  downloads: Pick<ModelDownloads, 'downloadProps' | 'active'>;
  host: ModelsResponse['host'] | undefined;
  localModels: readonly Pick<LocalModel, 'name'>[];
  chosen: Readonly<Record<string, string>>;
  choose: (label: string, slug: string) => void;
  /** The card links its installed build to its System One workspace. */
  systemOne?: boolean;
}): ReactElement {
  const { item, downloads, choose } = props;
  const { builds, draft } = entryDownloads(downloads, item);
  // A build whose job just ended stays shown until the refreshed catalog says where it is.
  const active: Record<string, unknown> = { ...downloads.active };
  for (const download of builds) if (download.settling) active[download.item.slug] = true;
  const build = shownBuild(item, props.chosen, active);
  return (
    <CatalogCard
      entry={item}
      chosen={build.slug}
      onChoose={(slug) => choose(item.label, slug)}
      downloads={builds}
      draft={draft}
      host={props.host}
      detailsTo={detailsLink(props.localModels, build.slug)}
      openTo={
        props.systemOne === true && build.present ? `/system-one?model=${encodeURIComponent(build.slug)}` : undefined
      }
    />
  );
}

/** The loading form of a card: the same five rows as placeholders. */
export function CatalogCardPlaceholder(): ReactElement {
  return (
    <div data-catalog-card="" className={styles.card}>
      <Panel ariaLabel="Loading recommended model" busy className={styles.card}>
        <div className={styles.body}>
          <div className="flex items-center">
            <Placeholder width="45%" />
          </div>
          <div>
            <Placeholder width="85%" />
          </div>
          <div className={styles.build}>
            <Placeholder width="40%" height={28} radius="control" />
            <Placeholder width="30%" />
          </div>
          <div className="flex items-center">
            <Placeholder width="50%" />
          </div>
          <div className={styles.action}>
            <Placeholder width="55%" height={32} radius="control" />
          </div>
        </div>
      </Panel>
    </div>
  );
}

export function CatalogGrid({ children }: { children: ReactNode }): ReactElement {
  return (
    <div className={styles.grid} data-catalog-grid="">
      {children}
    </div>
  );
}

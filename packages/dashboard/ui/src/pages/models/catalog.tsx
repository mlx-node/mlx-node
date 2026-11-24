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
import { DashboardApiError } from '../../lib/api';
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
  HostResponse,
  LocatePlanResponse,
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
  host: HostResponse | undefined,
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

function diskText(sizeGb: number, free: number): string {
  return `Needs about ${sizeGb} GB. ${formatBytes(free)} is free on this disk.`;
}

/** The one note a catalog card or an onboarding choice shows about room on this Mac for `build`, or null. */
export function hostNote(
  build: Pick<CatalogBuildItem, 'sizeGb' | 'present'>,
  minMemoryGb: number | undefined,
  host: HostResponse | undefined,
): string | null {
  const warning = hostWarning(build, minMemoryGb, host);
  if (warning === null) return null;
  if (warning.kind === 'disk') return diskText(build.sizeGb, warning.free);
  return `Weights are larger than this Mac's memory (${formatBytes(warning.memory)}).`;
}

/** The disk half of {@link hostNote}: the line a cleanup confirmation adds when the disk is short, or null. */
export function diskNote(
  build: Pick<CatalogBuildItem, 'sizeGb' | 'present'>,
  host: HostResponse | undefined,
): string | null {
  if (host === undefined || build.present || host.diskFreeBytes === null) return null;
  return build.sizeGb * GB > host.diskFreeBytes ? diskText(build.sizeGb, host.diskFreeBytes) : null;
}

/** The short form of {@link hostNote} that the build line of a card shows, or null. */
export function hostLine(
  build: Pick<CatalogBuildItem, 'sizeGb' | 'present'>,
  minMemoryGb: number | undefined,
  host: HostResponse | undefined,
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

const NAME_MAX_CHARS = 40;

/** File names in a sentence, each shortened in the middle with the whole name in its title. */
function NameList({ names }: { names: readonly string[] }): ReactElement {
  return (
    <>
      {names.map((name, index) => (
        <span key={name} title={name}>
          {index > 0 ? ', ' : ''}
          {truncateMiddle(name, NAME_MAX_CHARS)}
        </span>
      ))}
    </>
  );
}

/** What moves to the Trash: the whole folder or file, in the numbers the card read. */
function trashText(slug: string, blocker: CatalogBlocker): ReactNode {
  const size = sizeText(blocker.sizeBytes);
  if (blocker.kind === 'file') return `The file ${slug} (${size}).`;
  if (blocker.truncated) {
    if (blocker.fileCount === 0)
      return `The whole folder ${slug}. The app could not read it, so its size is not known.`;
    return `The whole folder ${slug}: more than ${filesText(blocker.fileCount)} (more than ${size}). The app could not list all of it.`;
  }
  if (blocker.fileCount === 0) return `The empty folder ${slug}.`;
  const counted = `The folder ${slug}: ${filesText(blocker.fileCount)} (${size})`;
  if (blocker.names.length === 0) return `${counted}.`;
  return (
    <>
      {`${counted}: `}
      <NameList names={blocker.names} />.
    </>
  );
}

/** What the cleanup of a link removes: the link alone. */
function removedText(blocker: CatalogBlocker): ReactNode {
  if (blocker.linkTarget === null) return 'Only the link.';
  return (
    <>
      Only the link. It points to <FolderName path={blocker.linkTarget} />
      {blocker.dangling ? ', which is not there.' : '.'}
    </>
  );
}

/**
 * One label and its text in the grid of a cleanup confirmation. Each cell starts with a space, which the grid does not
 * draw, so the text a screen reader or a copy reads keeps its words apart.
 */
function CleanupRow({ label, children }: { label: string; children: ReactNode }): ReactElement {
  return (
    <>
      <span className="text-secondary"> {label}</span>
      <span className="min-w-0 text-foreground"> {children}</span>
    </>
  );
}

/** The model a cleanup downloads, and its size. */
export interface CleanupWhat {
  /** `Qwen3.8-Flash-Next UD-Q4_K_XL`, or the companion's label. */
  name: string;
  sizeGb: number;
  /** {@link diskNote} for this build, or null. */
  spaceNote: string | null;
}

export interface CleanupHandlers {
  companion?: boolean;
  onCleanup: () => Promise<void>;
  onConfirmStarted: () => void;
}

/** From this size the confirmation says that the download can take an hour or more. */
const LONG_DOWNLOAD_GB = 20;

/**
 * The confirmation of Clean up & update over the entry in the way of `item`: what is removed or goes to the Trash,
 * what stays, and what is downloaded. `refusedSentence` is the server's reason when a refused Update led here.
 */
export function cleanupContent(
  item: { slug: string; blocker: CatalogBlocker },
  what: CleanupWhat,
  handlers: CleanupHandlers,
  refusedSentence?: string,
): ConfirmContent {
  const { slug, blocker } = item;
  const name = <span className="font-medium text-foreground">{slug}</span>;
  const link = blocker.kind === 'symlink';
  const noun = link ? 'link' : blocker.kind === 'file' ? 'file' : 'folder';
  const long = what.sizeGb >= LONG_DOWNLOAD_GB ? ' A download of this size can take an hour or more.' : '';
  const notices = [
    blocker.listed ? 'Local models lists this folder as a model. Make sure that you do not need it.' : null,
    link
      ? `The link is removed only when the download is complete and checked. If you cancel or the download fails, nothing changes.${
          blocker.holdsModel ? ' The old files keep their disk space.' : ''
        }`
      : 'It moves to the Trash only when the download is complete and checked. If you cancel or the download fails, nothing changes. You can take it back from the Trash. Its disk space is free only after you empty the Trash.',
    what.spaceNote,
  ].filter((line): line is string => line !== null);
  return {
    title: handlers.companion === true ? 'Clean up and update companion weights?' : 'Clean up and update?',
    description: (
      <span className="grid gap-2">
        {refusedSentence !== undefined ? (
          <span>
            Update cannot use {name} as it is. {refusedSentence}{' '}
            <span className="mt-1 block">Clean up &amp; update does this instead:</span>
          </span>
        ) : (
          <span>
            {name} is a {noun} that the app did not create.
          </span>
        )}
        <span className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1">
          {link ? (
            <>
              <CleanupRow label="Removed">{removedText(blocker)}</CleanupRow>
              {blocker.dangling || blocker.linkToFile ? null : (
                <CleanupRow label="Stays">
                  That folder and every file in it. The app does not change it and no longer uses it.
                </CleanupRow>
              )}
            </>
          ) : (
            <CleanupRow label="To the Trash">{trashText(slug, blocker)}</CleanupRow>
          )}
          <CleanupRow label="Downloaded">
            {`${refusedSentence !== undefined ? 'The whole model: ' : ''}${what.name}, ${what.sizeGb} GB, into a new folder.${long}`}
          </CleanupRow>
        </span>
      </span>
    ),
    notice: (
      <span className="grid gap-1">
        {notices.map((line, index) => (
          <span key={line}>
            {index > 0 ? ' ' : null}
            {line}
          </span>
        ))}
      </span>
    ),
    confirmLabel: 'Clean up & update',
    workingLabel: 'Starting…',
    successLabel: 'Started',
    tone: 'neutral',
    errorTitle: 'Could not start the download',
    onConfirm: handlers.onCleanup,
    onDone: handlers.onConfirmStarted,
  };
}

export interface UpdateHandlers {
  companion?: boolean;
  onUpdateInPlace: (token: string) => Promise<void>;
  onConfirmStarted: () => void;
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
                <NameList names={other.names} />
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
    onDone: handlers.onConfirmStarted,
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
 * folder holding another build, a folder the app may not write (no action), a full folder the app did not make
 * (Update in place), anything else in the way (Clean up & update), Install.
 */
export type CatalogActionState =
  | 'job'
  | 'update'
  | 'installed'
  | 'missing'
  | 'held'
  | 'locked'
  | 'adopt'
  | 'cleanup'
  | 'install';

/** `download.refused`: Update refused this folder for a reason a cleanup solves, so it is a cleanup, not an adopt. */
export function catalogActionState(
  item: Pick<DownloadItem, 'present' | 'heldBy' | 'blocker' | 'located'>,
  download: { job: ActiveJob | undefined; updateAvailable: boolean; refused?: boolean },
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
  if (blocker === null) return 'install';
  if (blocker.locked) return 'locked';
  if (blocker.holdsModel && blocker.kind === 'directory' && !blocker.truncated && download.refused !== true)
    return 'adopt';
  return 'cleanup';
}

/** The open state of a cleanup confirmation, for the folder it was opened for. */
function useCleanupOpen(slug: string): { open: boolean; onOpenChange: (open: boolean) => void } {
  const [openFor, setOpenFor] = useState<string | null>(null);
  return { open: openFor === slug, onOpenChange: (open) => setOpenFor(open ? slug : null) };
}

const CLEANUP_LABEL = 'Clean up & update…';

interface CleanupButtonProps extends CleanupHandlers {
  slug: string;
  blocker: CatalogBlocker;
  what: CleanupWhat;
  refusedSentence: string | undefined;
  settling: boolean;
  variant: 'secondary' | 'ghost';
  align: 'start' | 'end';
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/** Clean up & update…: opens its confirmation from the catalog's numbers, with no request before the confirm. */
function CleanupButton({
  slug,
  blocker,
  what,
  refusedSentence,
  settling,
  variant,
  align,
  open,
  onOpenChange,
  ...handlers
}: CleanupButtonProps): ReactElement {
  return (
    <ConfirmPopover
      trigger={{ kind: 'button', label: CLEANUP_LABEL, variant, size: 'sm' }}
      align={align}
      disabled={settling}
      disabledReason="Checking the last download."
      open={open}
      onOpenChange={onOpenChange}
      {...cleanupContent({ slug, blocker }, what, handlers, refusedSentence)}
    />
  );
}

/** The locked state's sentence: the one state the app cannot act on, and what the user can change. */
function lockedSentence(slug: string): string {
  return `The app has no write permission on ${slug}, so it cannot move it or add files to it. Change its permissions (Finder, Get Info), then open this page again.`;
}

function cleanupStatus(slug: string): string {
  return `${slug} is in the way of this model.`;
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
  /** The plan was refused for a reason a cleanup solves (`needs-cleanup`). */
  onNeedsCleanup: (message: string) => void;
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
  onNeedsCleanup,
  onUpdateInPlace,
  onConfirmStarted,
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
      if (err instanceof DashboardApiError && err.reason === 'needs-cleanup') onNeedsCleanup(err.message);
      else onPlanError(errMessage(err));
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
        {...updateContent(slug, plan, { companion, onUpdateInPlace, onConfirmStarted })}
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
  download: Pick<CatalogDownloadProps, 'onLocate' | 'onConfirmStarted'>;
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
            download.onConfirmStarted();
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
  /** The model a cleanup confirmation names ({@link buildFullName}). */
  cleanupName: string;
  host: HostResponse | undefined;
}

/** Onboarding's action for one build: progress, Update, the Installed status, a note on the folder, or Install. */
export function CatalogDownloadAction({
  item,
  installLabel = 'Install',
  settlingLabel = installLabel,
  prominent = false,
  updateAvailable,
  settling,
  updateSettling,
  job,
  refused,
  refusedSentence,
  onInstall,
  onCleanup,
  onConfirmStarted,
  onPlanUpdate,
  onUpdateRefused,
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
  cleanupName,
  host,
}: CatalogDownloadActionProps): ReactElement {
  const [starting, setStarting] = useState(false);
  const locate = useLocate({ item, onPlanLocate });
  const cleanupOpen = useCleanupOpen(item.slug);
  const start = async (): Promise<void> => {
    if (starting) return;
    setStarting(true);
    try {
      await onInstall();
    } finally {
      setStarting(false);
    }
  };
  const state = catalogActionState(item, { job, updateAvailable, refused });
  if (state === 'job' && job !== undefined) {
    const line = jobBlockerLine(item, job);
    return (
      <div className="grid gap-2">
        <DownloadProgress id={job.id} onDone={onDone} onError={onError} onCancelled={onCancelled} />
        {line !== null ? <p className="m-0 text-caption text-secondary">{line}</p> : null}
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
          { onDownload: onReplaceLocated, onStarted: onConfirmStarted },
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
      download={{ onLocate, onConfirmStarted }}
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
    const heldBy = item.heldBy ?? '';
    return (
      <Alert tone="warning" title={`Holds ${heldBy}`}>
        {heldSentence(item.slug, heldBy, buildName)}
      </Alert>
    );
  }
  if (state === 'locked') {
    return (
      <Alert tone="warning" title="No access">
        {lockedSentence(item.slug)}
      </Alert>
    );
  }
  if (state === 'adopt') {
    return (
      <UpdateInPlaceAction
        slug={item.slug}
        settling={settling}
        onPlanUpdate={onPlanUpdate}
        onNeedsCleanup={(message) => {
          onUpdateRefused(message);
          cleanupOpen.onOpenChange(true);
        }}
        onUpdateInPlace={onUpdateInPlace}
        onConfirmStarted={onConfirmStarted}
      />
    );
  }
  if (state === 'cleanup' && item.blocker !== null) {
    return (
      <Alert tone="warning" title="Needs cleanup">
        <span className="grid justify-items-start gap-2">
          <span>{cleanupStatus(item.slug)}</span>
          <CleanupButton
            slug={item.slug}
            blocker={item.blocker}
            what={{ name: cleanupName, sizeGb: item.sizeGb, spaceNote: diskNote(item, host) }}
            refusedSentence={refusedSentence}
            settling={settling}
            variant="secondary"
            align="start"
            {...cleanupOpen}
            onCleanup={onCleanup}
            onConfirmStarted={onConfirmStarted}
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

/**
 * What a job does to the old entry in its way, by the job's kind; null when it touches none. A cleanup names the entry
 * it was started for, which the catalog no longer shows once it reloads.
 */
function jobBlockerLine(item: Pick<DownloadItem, 'slug' | 'blocker'>, job: ActiveJob): string | null {
  if (job.kind === 'update') return `Checks the files in ${item.slug}. Nothing is deleted.`;
  const blocker = job.blocker ?? item.blocker;
  if (job.kind !== 'cleanup' || blocker === null) return null;
  if (blocker.kind === 'symlink') {
    return `Removes the link when the download is complete.${blocker.linkToFile ? '' : ' Its folder is not changed.'}`;
  }
  return `Moves the old ${blocker.kind === 'file' ? 'file' : 'folder'} to the Trash when the download is complete.`;
}

function heldSentence(slug: string, heldBy: string, build: string): string {
  return `${slug} holds ${heldBy}. It is a working model, so this card does not remove it. Delete it in Local models to install ${build} here.`;
}

/** `/models?focus=…` for the folder of `build` when Local models lists it. */
export function detailsLink(build: Pick<CatalogBuildItem, 'slug' | 'listed'>): string | undefined {
  return build.listed ? `/models?focus=${encodeURIComponent(build.slug)}` : undefined;
}

/** The model and build a cleanup confirmation names: `Qwen3.8-Flash-Next UD-Q4_K_XL`. */
export function buildFullName(entry: Pick<CatalogItem, 'label'>, build: Pick<CatalogBuildItem, 'id'>): string {
  return `${entry.label} ${build.id}`;
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
  host: HostResponse | undefined;
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

function ShowWhy({
  title,
  text,
  variant,
}: {
  title: string;
  text: string;
  variant: 'secondary' | 'ghost';
}): ReactElement {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button type="button" variant={variant} size="sm">
          Show why
        </Button>
      </PopoverTrigger>
      <PopoverContent align="end">
        <p className="m-0 text-body font-medium text-foreground">{title}</p>
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
  host: HostResponse | undefined;
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
  host,
  openTo,
  locate,
}: RowProps): ReactElement {
  const { item, job, settling, updateSettling } = download;
  const cleanupOpen = useCleanupOpen(item.slug);
  const detailsTo = detailsLink(build);
  const state = catalogActionState(item, download);
  if (state === 'job' && job !== undefined) {
    const fileLine = progressFileLine(progress);
    const blockerLine = jobBlockerLine(item, job);
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
              { onDownload: download.onReplaceLocated, onStarted: download.onConfirmStarted },
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
          <ShowWhy title={`Holds ${heldBy}`} text={sentence} variant="secondary" />
        )}
      </>
    );
  }
  if (state === 'locked') {
    const sentence = lockedSentence(item.slug);
    return (
      <>
        <div className={styles.primary}>
          <StatusWord text="No access" tooltip={sentence} />
        </div>
        <ShowWhy title="No access" text={sentence} variant="secondary" />
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
          onNeedsCleanup={(message) => {
            download.onUpdateRefused(message);
            cleanupOpen.onOpenChange(true);
          }}
          onUpdateInPlace={download.onUpdateInPlace}
          onConfirmStarted={download.onConfirmStarted}
        />
      </>
    );
  }
  if (state === 'cleanup' && item.blocker !== null) {
    return (
      <>
        <div className={styles.primary}>
          <StatusWord text="Needs cleanup" tooltip={cleanupStatus(item.slug)} />
        </div>
        <CleanupButton
          slug={item.slug}
          blocker={item.blocker}
          what={{ name: buildFullName(entry, build), sizeGb: build.sizeGb, spaceNote: diskNote(build, host) }}
          refusedSentence={download.refusedSentence}
          settling={settling}
          variant="secondary"
          align="end"
          {...cleanupOpen}
          onCleanup={download.onCleanup}
          onConfirmStarted={download.onConfirmStarted}
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
  host: HostResponse | undefined;
}

/** The draft weights of an entry: one line with its own state and job, the same for every build of the entry. */
function DraftLine({ entry, draft, download, progress, starting, onStart, host }: DraftLineProps): ReactElement {
  const { item, job, settling, updateSettling } = download;
  const cleanupOpen = useCleanupOpen(item.slug);
  const state = catalogActionState(item, download);
  const autoInstall = draft.autoInstall === true;
  const buildsNote =
    entry.builds.length === 2 ? ' Works with both builds.' : entry.builds.length > 2 ? ' Works with every build.' : '';
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
    glyph = <Badge size="sm">{autoInstall ? 'Included' : 'Optional'}</Badge>;
    text = `${draft.sizeGb} GB`;
    tooltip = autoInstall
      ? `Downloaded with this model for faster replies. The answers stay the same. Used the next time ${entry.label} loads.${buildsNote}`
      : `A draft model that can speed up replies. The answers stay the same. Used the next time ${entry.label} loads.${buildsNote}`;
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
          onNeedsCleanup={(message) => {
            download.onUpdateRefused(message);
            cleanupOpen.onOpenChange(true);
          }}
          onUpdateInPlace={download.onUpdateInPlace}
          onConfirmStarted={download.onConfirmStarted}
        />
      );
    } else if (state === 'locked') {
      text = 'no access';
      tooltip = lockedSentence(item.slug);
      control = <ShowWhy title="No access" text={tooltip} variant="ghost" />;
    } else {
      // A draft has no other build and no located files, so the only state left is a cleanup.
      text = 'needs cleanup';
      tooltip = cleanupStatus(item.slug);
      control =
        blocker === null ? null : (
          <CleanupButton
            slug={item.slug}
            blocker={blocker}
            what={{ name: draft.label, sizeGb: draft.sizeGb, spaceNote: diskNote(item, host) }}
            refusedSentence={download.refusedSentence}
            settling={settling}
            variant="ghost"
            align="end"
            {...cleanupOpen}
            companion
            onCleanup={download.onCleanup}
            onConfirmStarted={download.onConfirmStarted}
          />
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
  host: HostResponse | undefined;
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
                host={host}
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
              host={host}
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
export function JobFollower({ download }: { download: CatalogDownloadProps }): null {
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
  host: HostResponse | undefined;
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
      openTo={
        props.systemOne === true && build.present
          ? `/system-one?model=${encodeURIComponent(build.slug)}&build=${encodeURIComponent(build.id)}`
          : undefined
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

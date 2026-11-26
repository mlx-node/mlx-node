import { Info, TriangleAlert } from 'lucide-react';
import { useEffect, useId, useRef, type ReactElement, type ReactNode } from 'react';

import { CHAT, fill } from '../../../lib/chat-strings';
import type { ChatMode, RewindFile, RewindMode, RewindPlan } from '../../../lib/chat-types';
import { filesWord } from '../../../lib/permission-text';
import { Alert } from '../../arc/alert/alert';
import { Button } from '../../arc/button/button';
import { Dialog, DialogContent } from '../../arc/dialog/dialog';
import { Disclosure } from '../disclosure/disclosure';
import { MiddleText } from '../middle-text/middle-text';
import { MorphLoader } from '../morph-loader/morph-loader';
import { OptionList, type OptionItem } from '../option-list/option-list';
import styles from './rewind-dialog.module.css';

/**
 * Where a rewind stands. `loading`: the plan is read (`checking`: the agent's dry run of the files, else the
 * conversation is opened). `ready`: the plan is shown (`changed`: read again because a file changed after the last
 * check). `working`: the rewind runs (`slow` after 20 s). `long`: it runs past 120 s; the user may wait or restart the
 * agent. `error`: why it did not run, or what it could not do (`retry`: the plan may be asked again; `files`: the files
 * a restore could not put back, and how many more).
 */
export type RewindDialogState =
  | { phase: 'loading'; checking?: boolean }
  | { phase: 'ready'; plan: RewindPlan; changed?: boolean }
  | { phase: 'working'; plan: RewindPlan; slow: boolean }
  | { phase: 'long'; plan: RewindPlan }
  | {
      phase: 'error';
      tone: 'info' | 'warning' | 'danger';
      text: string;
      retry?: boolean;
      files?: { paths: readonly string[]; more: number };
    };

export interface RewindDialogProps {
  open: boolean;
  state: RewindDialogState;
  mode: ChatMode;
  /** The message rewound to, as typed; its first line shows. */
  message: string;
  /** The agent's live commands (every conversation's): a restart stops them. */
  commands: number;
  /** What the rewind puts back; Work picks it when the plan has files (see {@link defaultChoice}). */
  choice: RewindMode;
  onChoice: (mode: RewindMode) => void;
  /** Cancel, Close, Esc or a press outside; nothing while the rewind runs. */
  onClose: () => void;
  onConfirm: () => void;
  onKeepWaiting: () => void;
  onRestart: () => void;
  /** Try again after a plan the agent did not answer in time. */
  onRetry?: () => void;
  /** Where focus goes when the dialog closes; none: Radix's own. */
  returnFocus?: () => HTMLElement | null;
}

/** The rows a file list shows before the rest folds into "N more". */
export const REWIND_FILES_SHOWN = 8;

/** Every file the plan names, the ones past the list's limit included. */
function fileTotal(plan: RewindPlan): number {
  return plan.files.length + plan.filesMore;
}

/** Files changed outside the agent since it wrote them: a restore overwrites them. */
function conflictCount(plan: RewindPlan): number {
  return plan.files.filter((file) => file.state !== 'clean').length;
}

/** Whether the dialog offers the three choices: Work, and the agent would put files back. */
export function hasFileChoices(mode: ChatMode, plan: RewindPlan): boolean {
  return mode === 'work' && fileTotal(plan) > 0;
}

/** The choice a plan opens on: the conversation and the files, or the conversation only when a file changed outside. */
export function defaultChoice(mode: ChatMode, plan: RewindPlan): RewindMode {
  if (!hasFileChoices(mode, plan)) return 'conversation';
  return conflictCount(plan) > 0 ? 'conversation' : 'all';
}

/** The confirm button's words: a restore over files changed outside says so. */
export function confirmText(choice: RewindMode, plan: RewindPlan): string {
  const h = CHAT.history;
  if (choice !== 'conversation' && conflictCount(plan) > 0) return h.confirmOver;
  return choice === 'files' ? h.confirmFiles : h.confirm;
}

function stateText(file: RewindFile): string | null {
  const h = CHAT.history;
  if (file.state === 'modified') return h.fileModified;
  if (file.state === 'deleted') return h.fileDeleted;
  if (file.state === 'created') return h.fileCreated;
  return null;
}

function FileRow({ file }: { file: RewindFile }): ReactElement {
  const said = stateText(file);
  return (
    <li className={styles.file} data-file-state={file.state}>
      {said === null ? (
        <span className={styles.fileLead} aria-hidden="true" />
      ) : (
        <TriangleAlert size={12} strokeWidth={1.75} aria-hidden="true" className={styles.warningIcon} />
      )}
      <MiddleText text={file.path} className={styles.path} />
      {said === null ? null : (
        <>
          <span className={styles.srOnly}>, </span>
          <span className={styles.fileNote}>{said}</span>
        </>
      )}
    </li>
  );
}

function FileList({ plan }: { plan: RewindPlan }): ReactElement {
  const h = CHAT.history;
  const total = fileTotal(plan);
  const shown = plan.files.slice(0, REWIND_FILES_SHOWN);
  const rest = plan.files.slice(REWIND_FILES_SHOWN);
  return (
    <div className={styles.files}>
      <p className={styles.label}>{fill(h.files, { n: total })}</p>
      <ul className={styles.fileList} aria-label={fill(h.filesLabel, { n: total })}>
        {shown.map((file) => (
          <FileRow key={file.path} file={file} />
        ))}
      </ul>
      {total > shown.length ? (
        <Disclosure title={fill(h.filesMore, { n: total - shown.length })}>
          <ul className={styles.fileList}>
            {rest.map((file) => (
              <FileRow key={file.path} file={file} />
            ))}
          </ul>
          {plan.filesMore > 0 ? <p className={styles.muted}>{fill(h.filesMore, { n: plan.filesMore })}</p> : null}
        </Disclosure>
      ) : null}
    </div>
  );
}

/** The files a restore could not put back, under the error that says so. */
function FailedFiles({ paths, more }: { paths: readonly string[]; more: number }): ReactElement {
  const h = CHAT.history;
  return (
    <>
      <ul className={styles.fileList} aria-label={fill(h.failedLabel, { n: paths.length + more })}>
        {paths.map((path) => (
          <li key={path} className={styles.file}>
            <MiddleText text={path} className={styles.path} />
          </li>
        ))}
      </ul>
      {more > 0 ? <p className={styles.muted}>{fill(h.filesMore, { n: more })}</p> : null}
    </>
  );
}

function firstLine(text: string): string {
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (code === 0x0a || code === 0x0d) return text.slice(0, i);
  }
  return text;
}

function removesText(plan: RewindPlan): string {
  const h = CHAT.history;
  const steps = plan.removes.steps;
  if (steps === 0) return h.removesOne;
  return steps === 1 ? h.removesStep : fill(h.removes, { n: steps });
}

/** What the rewind also ends or cannot bring back, any that apply; a restore of files alone ends nothing. */
function warnings(plan: RewindPlan, choice: RewindMode): string[] {
  const h = CHAT.history;
  const out: string[] = [];
  if (choice === 'files') return out;
  const { commands, schedules, runs } = plan.background;
  if (commands > 0) out.push(commands === 1 ? h.backgroundOne : fill(h.background, { n: commands }));
  if (schedules > 0) out.push(schedules === 1 ? h.schedulesOne : fill(h.schedules, { n: schedules }));
  if (runs > 0) out.push(runs === 1 ? h.runsOne : fill(h.runs, { n: runs }));
  if (plan.attachments) out.push(h.attachments);
  return out;
}

/** The three choices of Work when the agent would put files back; a file choice carries a dot when files changed outside. */
function choiceOptions(plan: RewindPlan): OptionItem[] {
  const h = CHAT.history;
  const total = fileTotal(plan);
  const conflicts = conflictCount(plan);
  const dot: ReactNode =
    conflicts > 0 ? <span className={styles.conflictDot} data-conflict-dot="" aria-hidden="true" /> : undefined;
  return [
    {
      value: 'all',
      title: h.modeAll,
      note:
        conflicts > 0
          ? fill(h.modeAllConflict, { files: filesWord(total), c: conflicts })
          : total === 1
            ? h.modeAllNoteOne
            : fill(h.modeAllNote, { n: total }),
      trailing: dot,
    },
    { value: 'conversation', title: h.modeConversation, note: h.modeConversationNote },
    { value: 'files', title: h.modeFiles, note: h.modeFilesNote, trailing: dot },
  ];
}

/**
 * Rewind to here: what goes, then the confirm. The same dialog in Ask, Auto and Bypass, Chat and Work. Work with files
 * to put back offers three choices, the file list and the conflicts; with a file changed outside the agent it opens on
 * the conversation only and the confirm says it overwrites. While the rewind runs Esc and a press outside do nothing.
 * First focus is the checked choice, else Cancel; never the confirm button.
 */
export function RewindDialog(props: RewindDialogProps): ReactElement {
  const h = CHAT.history;
  const describedBy = useId();
  const cancel = useRef<HTMLButtonElement>(null);
  const choices = useRef<HTMLDivElement>(null);
  const { state, choice } = props;
  const running = state.phase === 'working' || state.phase === 'long';
  const plan = state.phase === 'ready' || state.phase === 'working' || state.phase === 'long' ? state.plan : null;
  const withFiles = plan !== null && hasFileChoices(props.mode, plan);
  const restores = withFiles && choice !== 'conversation';
  const lines = plan === null ? [] : warnings(plan, choice);
  const total = plan === null ? 0 : fileTotal(plan);
  const conflicts = plan === null ? 0 : conflictCount(plan);

  const token = props.open && state.phase === 'ready' && withFiles ? state.plan.token : null;
  useEffect(() => {
    if (token === null) return;
    choices.current?.querySelector<HTMLElement>('[role="radio"][aria-checked="true"]')?.focus({ preventScroll: true });
  }, [token]);

  return (
    <Dialog
      open={props.open}
      onOpenChange={(next) => {
        if (!next && !running) props.onClose();
      }}
    >
      <DialogContent
        title={h.rewindTitle}
        aria-describedby={describedBy}
        data-rewind-dialog=""
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          const picked = choices.current?.querySelector<HTMLElement>('[role="radio"][aria-checked="true"]');
          (picked ?? cancel.current)?.focus({ preventScroll: true });
        }}
        onEscapeKeyDown={(event) => {
          if (running) event.preventDefault();
        }}
        onInteractOutside={(event) => {
          if (running) event.preventDefault();
        }}
        onCloseAutoFocus={(event) => {
          const target = props.returnFocus?.();
          if (target === null || target === undefined || !target.isConnected) return;
          event.preventDefault();
          target.focus({ preventScroll: true });
        }}
      >
        <div className={styles.body}>
          {state.phase === 'loading' ? (
            <p id={describedBy} className={styles.loading}>
              <MorphLoader status="loading" />
              <span>{state.checking === true ? h.checking : h.opening}</span>
            </p>
          ) : null}
          {state.phase === 'error' ? (
            <div id={describedBy}>
              <Alert tone={state.tone} title={state.text}>
                {state.files === undefined ? undefined : <FailedFiles {...state.files} />}
              </Alert>
            </div>
          ) : null}
          {plan !== null ? (
            <>
              {state.phase === 'ready' && state.changed === true ? (
                <Alert tone="warning" title={h.planChanged} />
              ) : null}
              <blockquote className={styles.quote}>{firstLine(props.message)}</blockquote>
              <div className={styles.facts}>
                <p id={describedBy} className={styles.lead}>
                  {withFiles && choice === 'files'
                    ? total === 1
                      ? h.filesOnlyLeadOne
                      : fill(h.filesOnlyLead, { n: total })
                    : removesText(plan)}
                </p>
                {withFiles && choice === 'files' ? null : <p className={styles.line}>{h.refill}</p>}
                {props.mode === 'work' && !withFiles ? <p className={styles.muted}>{h.filesNone}</p> : null}
              </div>
              {withFiles ? (
                <div className={styles.choices} ref={choices}>
                  <p className={styles.label} aria-hidden="true">
                    {h.what}
                  </p>
                  <OptionList
                    label={h.what}
                    value={choice}
                    onValueChange={(value) => props.onChoice(value as RewindMode)}
                    options={choiceOptions(plan)}
                    disabled={running}
                    wrap
                  />
                  {choice === 'conversation' ? <p className={styles.muted}>{h.filesStay}</p> : null}
                </div>
              ) : null}
              {restores ? <FileList plan={plan} /> : null}
              {restores && conflicts > 0 ? (
                <p className={styles.warning} data-rewind-conflicts="">
                  <TriangleAlert size={12} strokeWidth={1.75} aria-hidden="true" className={styles.warningIcon} />
                  <span>{fill(h.conflicts, { files: filesWord(conflicts) })}</span>
                </p>
              ) : null}
              {lines.length > 0 ? (
                <ul className={styles.warnings}>
                  {lines.map((line) => (
                    <li key={line} className={styles.warning}>
                      <TriangleAlert size={12} strokeWidth={1.75} aria-hidden="true" className={styles.warningIcon} />
                      <span>{line}</span>
                    </li>
                  ))}
                </ul>
              ) : null}
              {restores ? (
                <p className={styles.note} data-rewind-limits="">
                  <Info size={12} strokeWidth={1.75} aria-hidden="true" className={styles.noteIcon} />
                  <span>{h.limits}</span>
                </p>
              ) : null}
              <p className={styles.muted}>{h.noUndo}</p>
            </>
          ) : null}
          {state.phase === 'long' ? (
            <Alert tone="warning" title={h.slow}>
              {props.commands > 0
                ? props.commands === 1
                  ? h.restartStopsOne
                  : fill(h.restartStops, { n: props.commands })
                : undefined}
            </Alert>
          ) : null}
          <div className={styles.footer}>
            {state.phase === 'long' ? (
              <>
                <Button ref={cancel} type="button" variant="ghost" size="md" onClick={props.onKeepWaiting}>
                  {h.keepWaiting}
                </Button>
                <Button type="button" variant="danger" size="md" onClick={props.onRestart}>
                  {h.restartAgent}
                </Button>
              </>
            ) : state.phase === 'error' ? (
              state.retry === true && props.onRetry !== undefined ? (
                <>
                  <Button ref={cancel} type="button" variant="ghost" size="md" onClick={props.onClose}>
                    {CHAT.cancel}
                  </Button>
                  <Button type="button" variant="secondary" size="md" onClick={props.onRetry}>
                    {h.tryAgain}
                  </Button>
                </>
              ) : (
                <Button ref={cancel} type="button" variant="secondary" size="md" onClick={props.onClose}>
                  {h.close}
                </Button>
              )
            ) : (
              <>
                <Button ref={cancel} type="button" variant="ghost" size="md" disabled={running} onClick={props.onClose}>
                  {CHAT.cancel}
                </Button>
                {state.phase !== 'loading' && plan !== null ? (
                  <Button
                    type="button"
                    variant="danger"
                    size="md"
                    loading={running}
                    aria-busy={running ? true : undefined}
                    onClick={() => {
                      if (state.phase === 'ready') props.onConfirm();
                    }}
                  >
                    {state.phase === 'working'
                      ? state.slow
                        ? h.stillRestoring
                        : h.working
                      : confirmText(withFiles ? choice : 'conversation', plan)}
                  </Button>
                ) : null}
              </>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

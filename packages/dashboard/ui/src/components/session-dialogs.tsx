import { Copy, Pencil, Trash2 } from 'lucide-react';
import { useRef, useState, type ReactElement, type ReactNode, type RefObject } from 'react';

import { getJson, mutate } from '../lib/api';
import { CHAT, fill } from '../lib/chat-strings';
import type { ConversationInfo } from '../lib/chat-types';
import { formatNumber } from '../lib/format';
import { copyResumeCommand, errMessage, isConflict, LIVE_HINT, sessionTitle } from '../lib/sessions';
import { toast } from '../lib/toast';
import type {
  SessionDeleteResponse,
  SessionDetailResponse,
  SessionOrigin,
  SessionRenameResponse,
  SessionRow,
} from '../lib/types';
import { ConfirmDialog } from './arc-x/confirm-dialog/confirm-dialog';
import { ConfirmPopover, type ConfirmContent } from './arc-x/confirm-popover/confirm-popover';
import { IconButton } from './arc-x/icon-button/icon-button';
import { PathText } from './arc-x/path-text/path-text';
import { Alert } from './arc/alert/alert';
import { Button } from './arc/button/button';
import { Dialog, DialogContent } from './arc/dialog/dialog';
import { Input } from './arc/input/input';
import { Tooltip } from './arc/tooltip/tooltip';

export interface SessionActionsProps {
  session: Pick<SessionRow, 'id' | 'title' | 'live' | 'subagentCount' | 'origin' | 'cwd'>;
  /** Icon buttons only (list rows). False: "Copy resume command" is a labelled secondary button (detail header). */
  compact?: boolean;
  /** Known on the detail page; on the list it is read from GET /sessions/:id on click. Null: there is none. */
  resumeCommand?: string | null;
  onRenamed: (title: string) => void;
  onDeleted: (deleted: string[]) => void;
  /** A 409 arrived: the caller refreshes so the Live badge catches up. */
  onConflict: () => void;
}

type DeleteHandlers = Pick<SessionActionsProps, 'onDeleted' | 'onConflict'>;

export function sessionUrl(id: string): string {
  return `/sessions/${encodeURIComponent(id)}`;
}

/** Characters of a folder path inside a sentence: the path never breaks its line, the hover title has it all. */
const NOTICE_PATH_CHARS = 28;

/** The Work delete notice, with the folder as one unbroken path. */
export function workDeleteNotice(folder: string): ReactNode {
  const [before, after] = splitAt(CHAT.deleteWorkNotice, '{folder}');
  return (
    <>
      {before}
      <PathText path={folder} maxChars={NOTICE_PATH_CHARS} mono={false} />
      {after}
    </>
  );
}

function splitAt(text: string, mark: string): [string, string] {
  const at = text.indexOf(mark);
  return at === -1 ? [text, ''] : [text.slice(0, at), text.slice(at + mark.length)];
}

/** A conversation of the app's Chat page (Chat or Work), not a terminal session. */
export function isAppConversation(origin: SessionOrigin): origin is 'chat' | 'work' {
  return origin === 'chat' || origin === 'work';
}

/** Where a rename or delete goes: app conversations through the Chat routes, so a loaded one is never changed behind the agent's back. */
export function sessionWritePath(session: Pick<SessionRow, 'id' | 'origin'>): string {
  return isAppConversation(session.origin)
    ? `/chat/conversations/${encodeURIComponent(session.id)}`
    : sessionUrl(session.id);
}

/**
 * A live terminal session cannot be renamed or deleted here. An app conversation can: the Chat routes act through the
 * agent that holds it and refuse only while an answer runs.
 */
export function locked(session: Pick<SessionRow, 'live' | 'origin'>): boolean {
  return session.live && !isAppConversation(session.origin);
}

/** The Chat page address of an app conversation. */
export function openInChatUrl(id: string): string {
  return `/chat?c=${encodeURIComponent(id)}`;
}

/** The ids the last successful delete of a session returned, read by its onDone (a re-render may rebuild the content between the two). */
const deletedIds = new Map<string, string[]>();

/** The one confirmation for deleting a session: shown by the ConfirmPopover at the button, or by DeleteSessionDialog. */
export function deleteSessionContent(
  session: SessionActionsProps['session'],
  handlers: DeleteHandlers,
): ConfirmContent {
  const title = sessionTitle(session);
  const children = session.subagentCount;
  if (isAppConversation(session.origin)) {
    return {
      title: CHAT.deleteTitle,
      description: fill(CHAT.deleteText, { title }),
      notice: session.origin === 'work' && session.cwd !== '' ? workDeleteNotice(session.cwd) : undefined,
      confirmLabel: CHAT.deleteConfirm,
      workingLabel: CHAT.deleting,
      successLabel: CHAT.deleted,
      tone: 'danger',
      errorTitle: CHAT.deleteError,
      onConfirm: async () => {
        try {
          await mutate('DELETE', sessionWritePath(session));
        } catch (err) {
          if (isConflict(err)) handlers.onConflict();
          throw err;
        }
      },
      onDone: () => {
        toast.success('Session deleted', { description: title });
        handlers.onDeleted([session.id]);
      },
    };
  }
  return {
    title: 'Delete session?',
    description: (
      <>
        This permanently removes <span className="font-medium text-foreground">{title}</span>
        {children > 0 ? ` and its ${formatNumber(children)} subagent session${children === 1 ? '' : 's'}` : ''} from
        disk. This cannot be undone.
      </>
    ),
    notice: session.live ? `Session is in use. ${LIVE_HINT}` : undefined,
    confirmLabel: 'Delete',
    workingLabel: 'Deleting…',
    successLabel: 'Deleted',
    tone: 'danger',
    errorTitle: 'Could not delete',
    onConfirm: async () => {
      try {
        const result = await mutate<SessionDeleteResponse>('DELETE', sessionUrl(session.id));
        deletedIds.set(session.id, result.deleted);
      } catch (err) {
        if (isConflict(err)) handlers.onConflict();
        throw err;
      }
    },
    onDone: () => {
      const deleted = deletedIds.get(session.id) ?? [session.id];
      deletedIds.delete(session.id);
      const more = deleted.length - 1;
      toast.success('Session deleted', {
        description: more > 0 ? `${title} and ${more} subagent session${more === 1 ? '' : 's'}` : title,
      });
      handlers.onDeleted(deleted);
    },
  };
}

export interface RenameSessionDialogProps {
  session: Pick<SessionRow, 'id' | 'title' | 'live' | 'origin'>;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onRenamed: (title: string) => void;
  onConflict: () => void;
  /** Where focus goes when the dialog closes (it opens from state, so Radix has no trigger); else the scroll root. */
  returnFocus?: RefObject<HTMLElement | null>;
}

/** The rename form: a centered dialog with one field. A 409 keeps it open with the server's reason. */
export function RenameSessionDialog(props: RenameSessionDialogProps): ReactElement {
  const { session } = props;
  const [value, setValue] = useState(session.title);
  const [busy, setBusy] = useState(false);
  const [conflict, setConflict] = useState<string | null>(null);
  const titleInput = useRef<HTMLInputElement>(null);
  const [wasOpen, setWasOpen] = useState(props.open);
  if (props.open !== wasOpen) {
    setWasOpen(props.open);
    if (props.open) {
      setValue(session.title);
      setConflict(null);
    }
  }
  const trimmed = value.trim();
  const canSave = trimmed !== '' && !locked(session) && !busy;

  const submit = async (): Promise<void> => {
    if (!canSave) return;
    setBusy(true);
    setConflict(null);
    try {
      const result = await mutate<SessionRenameResponse | ConversationInfo>('PATCH', sessionWritePath(session), {
        title: trimmed,
      });
      toast.success('Session renamed', { description: result.title });
      props.onOpenChange(false);
      props.onRenamed(result.title);
    } catch (err) {
      if (isConflict(err)) {
        setConflict(err.message);
        props.onConflict();
      } else {
        toast.error('Failed to rename session', { description: errMessage(err) });
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog
      open={props.open}
      onOpenChange={(next) => {
        if (!next && busy) return;
        props.onOpenChange(next);
      }}
    >
      <DialogContent
        title="Rename session"
        description="Sets the title the agent shows in its session picker. The agent keeps a manual title and stops generating one."
        onOpenAutoFocus={(event) => {
          const input = titleInput.current;
          if (input === null || input.disabled) return;
          event.preventDefault();
          input.focus({ preventScroll: true });
          input.select();
        }}
        onCloseAutoFocus={(event) => {
          event.preventDefault();
          const target = props.returnFocus?.current ?? null;
          target?.focus({ preventScroll: true });
          if (target === null || document.activeElement !== target) {
            document.querySelector<HTMLElement>('[data-scroll-root]')?.focus({ preventScroll: true });
          }
        }}
      >
        <div className="grid gap-3">
          <Input
            ref={titleInput}
            label="Session title"
            value={value}
            disabled={locked(session)}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') void submit();
            }}
          />
          {locked(session) ? (
            <Alert tone="warning" title="Session is in use">
              {LIVE_HINT}
            </Alert>
          ) : null}
          {conflict !== null ? (
            <Alert tone="danger" title="Could not save">
              {conflict}
            </Alert>
          ) : null}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="ghost" disabled={busy} onClick={() => props.onOpenChange(false)}>
              Cancel
            </Button>
            <Button
              type="button"
              variant="primary"
              loading={busy}
              disabled={trimmed === '' || locked(session)}
              onClick={() => void submit()}
            >
              Save
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

export interface DeleteSessionDialogProps extends DeleteHandlers {
  session: SessionActionsProps['session'];
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/** The centered delete confirmation, for the palette's "Delete session…" (no button to grow from). */
export function DeleteSessionDialog(props: DeleteSessionDialogProps): ReactElement {
  return (
    <ConfirmDialog
      {...deleteSessionContent(props.session, { onDeleted: props.onDeleted, onConflict: props.onConflict })}
      open={props.open}
      onOpenChange={props.onOpenChange}
    />
  );
}

/** Copy the resume command; on the list it is read from the detail route first (it depends on the agent home). */
async function copyResume(id: string, known: string | null | undefined): Promise<void> {
  if (known !== undefined && known !== null) {
    copyResumeCommand(known);
    return;
  }
  try {
    const detail = await getJson<SessionDetailResponse>(sessionUrl(id));
    if (detail.resumeCommand === null) return;
    copyResumeCommand(detail.resumeCommand);
  } catch (err) {
    toast.error('Failed to read the resume command', { description: errMessage(err) });
  }
}

export interface SessionActionHandles {
  /** Open the rename dialog (palette "Rename session…"). */
  rename(): void;
  /** Open the centered delete dialog (palette "Delete session…"). */
  remove(): void;
  /** The two dialogs; mount once next to the actions. */
  dialogs: ReactElement;
}

/** Dialog state for one session: what the palette commands and the Rename button open. */
export function useSessionActions(
  session: SessionActionsProps['session'],
  handlers: Pick<SessionActionsProps, 'onRenamed' | 'onDeleted' | 'onConflict'>,
): SessionActionHandles {
  const [renameOpen, setRenameOpen] = useState(false);
  const [deleteOpen, setDeleteOpen] = useState(false);
  return {
    rename: () => setRenameOpen(true),
    remove: () => setDeleteOpen(true),
    dialogs: (
      <>
        <RenameSessionDialog
          session={session}
          open={renameOpen}
          onOpenChange={setRenameOpen}
          onRenamed={handlers.onRenamed}
          onConflict={handlers.onConflict}
        />
        <DeleteSessionDialog
          session={session}
          open={deleteOpen}
          onOpenChange={setDeleteOpen}
          onDeleted={handlers.onDeleted}
          onConflict={handlers.onConflict}
        />
      </>
    ),
  };
}

/** Copy resume command, Rename and Delete for one session. Rename and Delete are off while the session is live. */
export function SessionActions(props: SessionActionsProps): ReactElement {
  const { session } = props;
  const compact = props.compact === true;
  const size = compact ? 'sm' : 'md';
  const [renameOpen, setRenameOpen] = useState(false);
  const renameButton = useRef<HTMLButtonElement | null>(null);
  const rename = (
    <IconButton
      ref={renameButton}
      label="Rename"
      icon={<Pencil size={16} strokeWidth={1.75} aria-hidden="true" />}
      size={size}
      disabled={locked(session)}
      tooltip={!locked(session)}
      onClick={() => setRenameOpen(true)}
    />
  );
  // A Chat conversation lives in the Chat agent's home: no terminal can resume it.
  const resumable = session.origin !== 'chat' && props.resumeCommand !== null;
  return (
    <div className="inline-flex items-center gap-1">
      {!resumable ? null : compact ? (
        <IconButton
          label="Copy resume command"
          icon={<Copy size={16} strokeWidth={1.75} aria-hidden="true" />}
          onClick={() => void copyResume(session.id, props.resumeCommand)}
        />
      ) : (
        <Button type="button" variant="secondary" onClick={() => void copyResume(session.id, props.resumeCommand)}>
          <Copy size={16} strokeWidth={1.75} aria-hidden="true" />
          Copy resume command
        </Button>
      )}
      {locked(session) ? (
        <Tooltip content={LIVE_HINT}>
          <span className="inline-flex">{rename}</span>
        </Tooltip>
      ) : (
        rename
      )}
      <ConfirmPopover
        trigger={{
          kind: 'icon',
          label: 'Delete',
          icon: <Trash2 size={16} strokeWidth={1.75} aria-hidden="true" />,
          size,
        }}
        disabled={locked(session)}
        disabledReason={LIVE_HINT}
        align="end"
        {...deleteSessionContent(session, props)}
      />
      <RenameSessionDialog
        session={session}
        open={renameOpen}
        onOpenChange={setRenameOpen}
        onRenamed={props.onRenamed}
        onConflict={props.onConflict}
        returnFocus={renameButton}
      />
    </div>
  );
}

import { useRef, useState, type ReactElement } from 'react';

import type { ConfirmContent } from '../../components/arc-x/confirm-popover/confirm-popover';
import { Alert } from '../../components/arc/alert/alert';
import { Button } from '../../components/arc/button/button';
import { Dialog, DialogContent } from '../../components/arc/dialog/dialog';
import { Input } from '../../components/arc/input/input';
import { workDeleteNotice } from '../../components/session-dialogs';
import { DashboardApiError, mutate } from '../../lib/api';
import { CHAT, fill } from '../../lib/chat-strings';
import type { ChatMode, ConversationInfo } from '../../lib/chat-types';
import { toast } from '../../lib/toast';
import { oneLine } from '../../lib/web-tools';

export function conversationPath(id: string): string {
  return `/chat/conversations/${encodeURIComponent(id)}`;
}

/** The agent's limit, in characters (Unicode scalars) after trimming. */
export const RENAME_MAX = 100;

/** Unicode scalars, as the agent counts a title. */
function scalarCount(text: string): number {
  let n = 0;
  for (let i = 0; i < text.length; i += 1) {
    const code = text.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff && i + 1 < text.length) {
      const next = text.charCodeAt(i + 1);
      if (next >= 0xdc00 && next <= 0xdfff) i += 1;
    }
    n += 1;
  }
  return n;
}

export interface RenameTarget {
  id: string;
  title: string;
  busy: boolean;
}

export interface RenameDialogProps {
  target: RenameTarget | null;
  onOpenChange: (open: boolean) => void;
  onRenamed: (info: ConversationInfo) => void;
  /** Where focus goes when the dialog closes; by default the element that had it when the dialog opened. */
  returnFocus?: () => HTMLElement | null;
}

/** Rename one conversation: a centered dialog with one field. A busy conversation keeps Save off. */
export function RenameDialog({ target, onOpenChange, onRenamed, returnFocus }: RenameDialogProps): ReactElement {
  const open = target !== null;
  const [value, setValue] = useState(oneLine(target?.title ?? ''));
  const [busy, setBusy] = useState(false);
  const [conflict, setConflict] = useState(false);
  const [shown, setShown] = useState<RenameTarget | null>(target);
  const input = useRef<HTMLInputElement>(null);
  if (target !== shown) {
    setShown(target);
    if (target !== null) {
      setValue(oneLine(target.title));
      setConflict(false);
    }
  }
  const trimmed = value.trim();
  const locked = target?.busy === true || conflict;
  const tooLong = scalarCount(trimmed) > RENAME_MAX;
  const canSave =
    target !== null && trimmed !== '' && !tooLong && trimmed !== oneLine(target.title).trim() && !locked && !busy;

  const submit = async (): Promise<void> => {
    if (!canSave) return;
    setBusy(true);
    try {
      const info = await mutate<ConversationInfo>('PATCH', conversationPath(target.id), { title: trimmed });
      toast.success(CHAT.renamed, { description: info.title });
      onOpenChange(false);
      onRenamed(info);
    } catch (err) {
      if (err instanceof DashboardApiError && err.code === 'E_CONFLICT') setConflict(true);
      else toast.error(CHAT.renameTitle, { description: `${CHAT.tryAgain}.` });
    } finally {
      setBusy(false);
    }
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next && busy) return;
        onOpenChange(next);
      }}
    >
      <DialogContent
        title={CHAT.renameTitle}
        aria-describedby={undefined}
        onCloseAutoFocus={(event) => {
          const back = returnFocus?.();
          if (back === null || back === undefined) return;
          event.preventDefault();
          back.focus({ preventScroll: true });
        }}
        onOpenAutoFocus={(event) => {
          const node = input.current;
          if (node === null) return;
          event.preventDefault();
          node.focus({ preventScroll: true });
          node.select();
        }}
      >
        <div className="grid gap-3">
          <Input
            ref={input}
            label={CHAT.renameLabel}
            value={value}
            maxLength={200}
            error={tooLong ? CHAT.renameTooLong : undefined}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Enter') void submit();
            }}
          />
          {locked ? <Alert tone="warning" title={CHAT.renameBusy} /> : null}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="ghost" disabled={busy} onClick={() => onOpenChange(false)}>
              {CHAT.cancel}
            </Button>
            <Button type="button" variant="primary" loading={busy} disabled={!canSave} onClick={() => void submit()}>
              {CHAT.renameSave}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

/** The one delete confirmation of a conversation: at its button (ConfirmPopover) or from the palette (ConfirmDialog). */
export function deleteContent(
  item: { id: string; title: string; mode: ChatMode; folder: string | null },
  onDeleted: (id: string) => void,
): ConfirmContent {
  return {
    title: CHAT.deleteTitle,
    description: fill(CHAT.deleteText, { title: item.title }),
    notice: item.mode === 'work' && item.folder !== null ? workDeleteNotice(item.folder) : undefined,
    confirmLabel: CHAT.deleteConfirm,
    workingLabel: CHAT.deleting,
    successLabel: CHAT.deleted,
    tone: 'danger',
    errorTitle: CHAT.deleteError,
    onConfirm: async () => {
      try {
        await mutate('DELETE', conversationPath(item.id));
      } catch (err) {
        if (err instanceof DashboardApiError && err.reason === 'conversation-busy') throw new Error(CHAT.deleteBusy);
        throw new Error(`${CHAT.tryAgain}.`);
      }
    },
    onDone: () => onDeleted(item.id),
  };
}

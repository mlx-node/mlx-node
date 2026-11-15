import { Pencil, ScanSearch, SquarePen, Trash2 } from 'lucide-react';
import type { ReactElement, ReactNode } from 'react';

import { ButtonLink } from '../../components/arc-x/button-link/button-link';
import { ConfirmPopover, type ConfirmContent } from '../../components/arc-x/confirm-popover/confirm-popover';
import { IconButton } from '../../components/arc-x/icon-button/icon-button';
import { LiveDot } from '../../components/arc-x/live-dot/live-dot';
import SegmentedControl, { type Segment } from '../../components/arc/segmented-control/segmented-control';
import { Tooltip } from '../../components/arc/tooltip/tooltip';
import { CHAT, fill } from '../../lib/chat-strings';
import type { ChatHostState, ChatMode } from '../../lib/chat-types';
import { titleWhileCut } from '../../lib/hover-title';

/** The other workspace's attention: a dot on its segment, with the words in its name. */
function accessory(mode: ChatMode, active: ChatHostState['active']): ReactNode {
  const name = mode === 'work' ? CHAT.modeWork : CHAT.modeChat;
  const waiting = active.filter((entry) => entry.mode === mode && entry.waiting).length;
  if (waiting > 0) {
    return <LiveDot tone="warning" pulse size={6} label={fill(CHAT.otherWaiting, { n: waiting, mode: name })} />;
  }
  const running = active.filter((entry) => entry.mode === mode && entry.busy).length;
  if (running > 0) {
    return <LiveDot tone="success" pulse size={6} label={fill(CHAT.otherRunning, { n: running, mode: name })} />;
  }
  return undefined;
}

export function ModeSwitch(props: {
  mode: ChatMode;
  active: ChatHostState['active'];
  onChange: (mode: ChatMode) => void;
}): ReactElement {
  const options: Segment[] = (['chat', 'work'] as const).map((value) => ({
    value,
    label: value === 'work' ? CHAT.modeWork : CHAT.modeChat,
    accessory: value === props.mode ? undefined : accessory(value, props.active),
  }));
  return (
    <SegmentedControl
      label={CHAT.mode}
      options={options}
      value={props.mode}
      onValueChange={(value) => {
        if (value === 'chat' || value === 'work') props.onChange(value);
      }}
    />
  );
}

export function NewButton({ mode, onNew }: { mode: ChatMode; onNew: () => void }): ReactElement {
  const label = mode === 'work' ? CHAT.newWork : CHAT.newChat;
  return (
    <IconButton
      label={label}
      tooltip={`${label}  ⌘⇧O`}
      icon={<SquarePen size={16} strokeWidth={1.75} />}
      onClick={onNew}
    />
  );
}

export interface ThreadBarProps {
  title: string;
  /** The open conversation; null on the New screen. */
  open: { id: string; running: boolean; remove: ConfirmContent } | null;
  onRename: () => void;
  /** The New button when the list is a drawer (it sits last in the bar). */
  trailing?: ReactNode;
  /** Work: the folder chip after the title. */
  folder?: ReactNode;
  /** Work: the permission mode chip after the folder chip, while the mode is Auto or Bypass. */
  permission?: ReactNode;
}

/** The right zone of the bar: the page's one h1, then the conversation's actions. */
export function ThreadBar({ title, open, onRename, trailing, folder, permission }: ThreadBarProps): ReactElement {
  return (
    <div className="flex min-w-0 flex-1 items-center gap-1">
      <h1
        className="m-0 min-w-0 flex-1 truncate font-heading text-body font-medium text-foreground"
        onPointerEnter={(event) => titleWhileCut(event.currentTarget, title)}
      >
        {title}
      </h1>
      {folder}
      {permission}
      {open !== null ? (
        <>
          <ButtonLink to={`/sessions/${encodeURIComponent(open.id)}?tab=activity`} size="sm" variant="ghost">
            <ScanSearch className="@min-[45rem]:hidden" size={14} strokeWidth={1.75} aria-hidden="true" />
            <span className="sr-only @min-[45rem]:not-sr-only">{CHAT.inspect}</span>
          </ButtonLink>
          {open.running ? (
            <Tooltip content={CHAT.modelBusy}>
              <span>
                <IconButton
                  label={CHAT.rename}
                  tooltip={false}
                  icon={<Pencil size={16} strokeWidth={1.75} />}
                  disabled
                />
              </span>
            </Tooltip>
          ) : (
            <IconButton label={CHAT.rename} icon={<Pencil size={16} strokeWidth={1.75} />} onClick={onRename} />
          )}
          <ConfirmPopover
            trigger={{ kind: 'icon', label: CHAT.remove, icon: <Trash2 size={16} strokeWidth={1.75} /> }}
            align="end"
            disabled={open.running}
            disabledReason={open.running ? CHAT.deleteBusy : undefined}
            {...open.remove}
          />
        </>
      ) : null}
      {trailing}
    </div>
  );
}

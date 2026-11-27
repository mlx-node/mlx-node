import {
  Ellipsis,
  FoldVertical,
  GitFork,
  PanelRight,
  Pencil,
  ScanSearch,
  ScrollText,
  SquarePen,
  Trash2,
} from 'lucide-react';
import { useState, type ReactElement, type ReactNode } from 'react';

import { ButtonLink } from '../../components/arc-x/button-link/button-link';
import { ConfirmPopover, type ConfirmContent } from '../../components/arc-x/confirm-popover/confirm-popover';
import { IconButton } from '../../components/arc-x/icon-button/icon-button';
import { LiveDot } from '../../components/arc-x/live-dot/live-dot';
import { Menu, MenuItem } from '../../components/arc-x/menu/menu';
import { Button } from '../../components/arc/button/button';
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
  const running = active.filter(
    (entry) => entry.mode === mode && (entry.busy || entry.background || entry.tasks > 0),
  ).length;
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

/**
 * Shows or hides the activity rail. Open requests win: a warning dot and their count; else, while work runs, a live
 * dot and the running count.
 */
export function ActivityToggle(props: {
  open: boolean;
  running: number;
  needs: number;
  onToggle: () => void;
}): ReactElement {
  const name = props.open ? CHAT.activity.hide : CHAT.activity.show;
  const state =
    props.needs > 0
      ? {
          tone: 'warning' as const,
          n: props.needs,
          words: props.needs === 1 ? CHAT.needs.countOne : fill(CHAT.needs.count, { n: props.needs }),
        }
      : props.running > 0
        ? { tone: 'success' as const, n: props.running, words: fill(CHAT.activity.running, { n: props.running }) }
        : null;
  return (
    <Tooltip content={name}>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        aria-pressed={props.open}
        aria-label={state === null ? name : `${name}, ${state.words}`}
        data-needs={props.needs > 0 ? '' : undefined}
        onClick={props.onToggle}
      >
        {state !== null ? (
          <LiveDot tone={state.tone} pulse size={6} label={state.words} />
        ) : (
          <PanelRight size={14} strokeWidth={1.75} aria-hidden="true" />
        )}
        <span className="sr-only @min-[45rem]:not-sr-only">{CHAT.activity.label}</span>
        {state !== null ? <span className="tabular-nums">{state.n}</span> : null}
      </Button>
    </Tooltip>
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
  /**
   * The open conversation; null on the New screen. `deleteOff`: why Delete is off (an answer, a workflow run, a
   * subagent, a command), or null; rename is off only while an answer runs.
   */
  open: { id: string; running: boolean; deleteOff: string | null; remove: ConfirmContent } | null;
  onRename: () => void;
  /** The New button when the list is a drawer (it sits last in the bar). */
  trailing?: ReactNode;
  /** Work: the folder chip after the title (an open conversation's, or a new one's once a folder is chosen). */
  folder?: ReactNode;
  /** Work: the permission mode chip after the folder chip, while the mode is Auto or Bypass. */
  permission?: ReactNode;
  /** The activity rail toggle, before Inspect. */
  activity?: ReactNode;
  /** The conversation's `⋯` menu, last before the New button. */
  menu?: ReactNode;
}

/** The right zone of the bar: the page's one h1, then the conversation's actions. */
export function ThreadBar({
  title,
  open,
  onRename,
  trailing,
  folder,
  permission,
  activity,
  menu,
}: ThreadBarProps): ReactElement {
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
      {activity}
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
            disabled={open.deleteOff !== null}
            disabledReason={open.deleteOff ?? undefined}
            {...open.remove}
          />
          {menu}
        </>
      ) : null}
      {trailing}
    </div>
  );
}

export interface ThreadMenuProps {
  fork: {
    /** Why the row is off (it stays focusable and reads it); null when it acts. */
    off: string | null;
    onFork: () => void;
  };
  compact: {
    /** Why the row is off (it stays focusable and reads it); null when it acts. */
    off: string | null;
    /** How full the context is, in the row's right slot; null while unknown. */
    full: string | null;
    onCompact: () => void;
  };
  recap: {
    /** Why the row is off (it stays focusable and reads it); null when it acts. */
    off: string | null;
    onRecap: () => void;
  };
}

/** The conversation's `⋯` in the thread bar: its history ops. */
export function ThreadMenu({ fork, compact, recap }: ThreadMenuProps): ReactElement {
  const [open, setOpen] = useState(false);
  return (
    <Menu
      label={CHAT.history.threadMenu}
      open={open}
      onOpenChange={setOpen}
      side="bottom"
      align="end"
      trigger={<IconButton label={CHAT.history.threadMenu} icon={<Ellipsis size={16} strokeWidth={1.75} />} />}
    >
      <MenuItem
        icon={<GitFork size={14} strokeWidth={1.75} />}
        label={CHAT.history.fork}
        note={fork.off ?? CHAT.history.forkNote}
        disabled={fork.off !== null}
        onSelect={fork.onFork}
      />
      <MenuItem
        icon={<FoldVertical size={14} strokeWidth={1.75} />}
        label={CHAT.history.compact}
        note={compact.off ?? CHAT.history.compactNote}
        end={compact.off === null && compact.full !== null ? compact.full : undefined}
        disabled={compact.off !== null}
        onSelect={compact.onCompact}
      />
      <MenuItem
        icon={<ScrollText size={14} strokeWidth={1.75} />}
        label={CHAT.history.recap}
        note={recap.off ?? CHAT.history.recapNote}
        disabled={recap.off !== null}
        onSelect={recap.onRecap}
      />
    </Menu>
  );
}

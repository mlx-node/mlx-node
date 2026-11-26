import { Ellipsis, GitFork, Undo2 } from 'lucide-react';
import { useRef, useState, type ReactElement } from 'react';

import { CopyButton } from '../../arc/copy-button/copy-button';
import { IconButton } from '../icon-button/icon-button';
import { Menu, MenuItem } from '../menu/menu';

export interface MessageActionsProps {
  /** The message text Copy takes. */
  text: string;
  copyLabel: string;
  /** The menu's name, and the name of its button. */
  menuLabel: string;
  /** Rewind to here: its words, why it is off (the row stays focusable and reads it), and what it does. */
  rewind: {
    label: string;
    note: string;
    off: string | null;
    /** `from`: the menu's button, where focus goes back when the rewind is cancelled. */
    onSelect: (from: HTMLElement | null) => void;
  };
  /** Fork with this answer: its words, why it is off, and what it does. None: no row. */
  fork?: {
    label: string;
    note: string;
    off: string | null;
    onSelect: () => void;
  };
}

/** Under a sent message: Copy, and `⋯` with what can be done from this message. Shown on hover and on focus inside. */
export function MessageActions(props: MessageActionsProps): ReactElement {
  const [open, setOpen] = useState(false);
  const trigger = useRef<HTMLButtonElement>(null);
  const { rewind, fork } = props;
  return (
    <>
      <CopyButton value={props.text} label={props.copyLabel} iconOnly variant="plain" />
      <Menu
        label={props.menuLabel}
        open={open}
        onOpenChange={setOpen}
        side="bottom"
        align="end"
        trigger={
          <IconButton
            ref={trigger}
            label={props.menuLabel}
            icon={<Ellipsis size={16} strokeWidth={1.75} />}
            data-message-menu=""
          />
        }
      >
        <MenuItem
          icon={<Undo2 size={14} strokeWidth={1.75} />}
          label={rewind.label}
          note={rewind.off ?? rewind.note}
          disabled={rewind.off !== null}
          onSelect={() => rewind.onSelect(trigger.current)}
        />
        {fork !== undefined ? (
          <MenuItem
            icon={<GitFork size={14} strokeWidth={1.75} />}
            label={fork.label}
            note={fork.off ?? fork.note}
            disabled={fork.off !== null}
            onSelect={fork.onSelect}
          />
        ) : null}
      </Menu>
    </>
  );
}

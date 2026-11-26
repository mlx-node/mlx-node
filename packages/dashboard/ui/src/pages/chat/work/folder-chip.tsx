import { FolderOpen } from 'lucide-react';
import { type ReactElement, useRef, useState } from 'react';

import { OptionList } from '../../../components/arc-x/option-list/option-list';
import { PathText } from '../../../components/arc-x/path-text/path-text';
import { ToolChip } from '../../../components/arc-x/tool-chip/tool-chip';
import { Button } from '../../../components/arc/button/button';
import { CopyButton } from '../../../components/arc/copy-button/copy-button';
import { Popover, PopoverContent, PopoverTrigger } from '../../../components/arc/popover/popover';
import { CHAT, fill } from '../../../lib/chat-strings';
import { useHomeDir } from '../../../lib/home-dir';
import { truncateMiddle } from '../../../lib/labels';
import { tildePath } from '../../../lib/paths';
import { folderName } from '../../../lib/permission-text';

/** Characters of a folder's name on its chip. */
const NAME_CHARS = 20;
/** Characters of a recent folder's path in the list. */
const PATH_CHARS = 34;

const icon = <FolderOpen size={14} strokeWidth={1.75} />;

/** The value of the "No folder" row: no folder path is empty. */
const NONE_VALUE = '';
/** The list's value while nothing is picked yet: no path holds a NUL, so no row reads checked. */
const NO_PICK = '\u0000';

export interface NewFolderChipProps {
  /** The folder the next Work conversation opens in, or null. */
  folder: string | null;
  /** The newest folders of the Work list. */
  recent: readonly string[];
  onChoose: (path: string) => void;
  /** Open the native folder dialog. */
  onPick: () => void;
  /** The chip's words while no folder is chosen. Default "Choose folder…". */
  label?: string;
  /** The server refused the folder: its sentence, as the chip's danger dot and title. */
  problem?: string | null;
  /**
   * Work: the "No folder" row, last; the caption reads "Folder" and the popover always opens. `selected`: the next
   * conversation starts without a folder. `root`: where its folder is made, as people read it.
   */
  none?: { selected: boolean; onChoose: () => void; root: string };
  /** Work: no place yet. `loading`: the Work list is on its way; `unset`: it failed, and the person picks one. */
  unplaced?: 'loading' | 'unset';
}

/**
 * A new conversation's folder chip in the tray under the composer: recent folders, in Work "No folder", then the
 * native dialog. Outside Work, with no recent folder the dialog opens at once.
 */
export function NewFolderChip({
  folder,
  recent,
  onChoose,
  onPick,
  label,
  problem,
  none,
  unplaced,
}: NewFolderChipProps): ReactElement {
  const [open, setOpen] = useState(false);
  // A pick leaves focus to the page (the composer); Radix's return to the chip after the exit animation would undo it.
  const pickedRef = useRef(false);
  const home = useHomeDir();
  const options = recent.map((path) => ({
    value: path,
    title: folderName(path),
    note: truncateMiddle(tildePath(path, home), PATH_CHARS),
  }));
  if (none !== undefined)
    options.push({ value: NONE_VALUE, title: CHAT.folderNone, note: fill(CHAT.folderNoneNote, { root: none.root }) });
  const nothing =
    unplaced === 'loading'
      ? CHAT.folderLoadingChip
      : unplaced === 'unset'
        ? CHAT.folderUnset
        : (label ?? CHAT.folderChoose);
  const blank = unplaced === undefined && none?.selected === true;
  const shown = blank ? CHAT.folderNone : folder === null ? nothing : truncateMiddle(folderName(folder), NAME_CHARS);
  const named = blank
    ? fill(CHAT.folderLabelNone, { root: none.root })
    : folder === null
      ? unplaced === 'loading'
        ? CHAT.folderLoading
        : nothing
      : fill(CHAT.folderLabel, { folder: tildePath(folder, home) });
  const refused = problem !== undefined && problem !== null;
  const caption = none !== undefined ? CHAT.folderPath : CHAT.folderRecent;
  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        if (next && recent.length === 0 && none === undefined) {
          onPick();
          return;
        }
        if (next) pickedRef.current = false;
        setOpen(next);
      }}
    >
      <PopoverTrigger asChild>
        <ToolChip
          icon={icon}
          label={shown}
          state={blank || folder === null ? 'off' : 'on'}
          dot={refused ? 'danger' : undefined}
          title={refused ? problem : blank || folder === null ? undefined : tildePath(folder, home)}
          aria-label={refused ? `${named}. ${problem}` : named}
          data-tray-part="folder"
        />
      </PopoverTrigger>
      <PopoverContent
        side="top"
        align="start"
        aria-label={caption}
        onCloseAutoFocus={(event) => {
          if (pickedRef.current) event.preventDefault();
        }}
      >
        <div className="grid w-[18.75rem] max-w-full gap-2">
          <p className="m-0 text-caption font-medium text-secondary">{caption}</p>
          <OptionList
            label={caption}
            value={blank ? NONE_VALUE : (folder ?? (unplaced === undefined ? NONE_VALUE : NO_PICK))}
            onValueChange={(path) => {
              pickedRef.current = true;
              if (none !== undefined && path === NONE_VALUE) none.onChoose();
              else onChoose(path);
              setOpen(false);
            }}
            options={options}
          />
          <span>
            <Button
              type="button"
              variant="secondary"
              size="sm"
              onClick={() => {
                pickedRef.current = true;
                setOpen(false);
                onPick();
              }}
            >
              {recent.length === 0 ? CHAT.folderChoose : CHAT.folderChooseOther}
            </Button>
          </span>
        </div>
      </PopoverContent>
    </Popover>
  );
}

/**
 * An open Work conversation's folder in the bar: fixed for its life. The popover shows the whole path and copies it.
 * `noFolder`: the folder the app made for it, named "No folder".
 */
export function OpenFolderChip({ folder, noFolder = false }: { folder: string; noFolder?: boolean }): ReactElement {
  const home = useHomeDir();
  const shown = tildePath(folder, home);
  return (
    <Popover>
      <PopoverTrigger asChild>
        <ToolChip
          icon={icon}
          label={noFolder ? CHAT.folderNone : truncateMiddle(folderName(folder), NAME_CHARS)}
          give={1}
          title={shown}
          aria-label={noFolder ? `${CHAT.folderNone}, ${shown}` : fill(CHAT.folderLabel, { folder: shown })}
        />
      </PopoverTrigger>
      <PopoverContent side="bottom" align="end" aria-label={CHAT.folderPath}>
        <div className="grid w-[18.75rem] max-w-full gap-2">
          <p className="m-0 text-caption font-medium text-secondary">{CHAT.folderPath}</p>
          <p className="m-0 min-w-0 text-body text-foreground">
            <PathText path={folder} wrap />
          </p>
          {noFolder ? <p className="m-0 text-caption text-secondary">{CHAT.folderNoneOwn}</p> : null}
          <span>
            <CopyButton value={folder} label={CHAT.folderCopy} />
          </span>
        </div>
      </PopoverContent>
    </Popover>
  );
}

import { FolderOpen } from 'lucide-react';
import { type ReactElement, useState } from 'react';

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

export interface NewFolderChipProps {
  /** The folder the next Work conversation opens in, or null. */
  folder: string | null;
  /** The newest folders of the Work list. */
  recent: readonly string[];
  onChoose: (path: string) => void;
  /** Open the native folder dialog. */
  onPick: () => void;
}

/** The New screen's folder chip: recent folders, then the native dialog. With no recent folder the dialog opens at once. */
export function NewFolderChip({ folder, recent, onChoose, onPick }: NewFolderChipProps): ReactElement {
  const [open, setOpen] = useState(false);
  const home = useHomeDir();
  const options = recent.map((path) => ({
    value: path,
    title: folderName(path),
    note: truncateMiddle(tildePath(path, home), PATH_CHARS),
  }));
  const label = folder === null ? CHAT.folderChoose : truncateMiddle(folderName(folder), NAME_CHARS);
  return (
    <Popover
      open={open}
      onOpenChange={(next) => {
        if (next && recent.length === 0) {
          onPick();
          return;
        }
        setOpen(next);
      }}
    >
      <PopoverTrigger asChild>
        <ToolChip
          icon={icon}
          label={label}
          give={1}
          state={folder === null ? 'off' : 'on'}
          title={folder === null ? undefined : tildePath(folder, home)}
          aria-label={folder === null ? CHAT.folderChoose : fill(CHAT.folderLabel, { folder: tildePath(folder, home) })}
        />
      </PopoverTrigger>
      <PopoverContent side="top" align="start" aria-label={CHAT.folderRecent}>
        <div className="grid w-[18.75rem] max-w-full gap-2">
          <p className="m-0 text-caption font-medium text-secondary">{CHAT.folderRecent}</p>
          <OptionList
            label={CHAT.folderRecent}
            value={folder ?? ''}
            onValueChange={(path) => {
              onChoose(path);
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
                setOpen(false);
                onPick();
              }}
            >
              {CHAT.folderChooseOther}
            </Button>
          </span>
        </div>
      </PopoverContent>
    </Popover>
  );
}

/** An open Work conversation's folder: fixed for its life. The popover shows the whole path and copies it. */
export function OpenFolderChip({ folder }: { folder: string }): ReactElement {
  const home = useHomeDir();
  const shown = tildePath(folder, home);
  return (
    <Popover>
      <PopoverTrigger asChild>
        <ToolChip
          icon={icon}
          label={truncateMiddle(folderName(folder), NAME_CHARS)}
          give={1}
          title={shown}
          aria-label={fill(CHAT.folderLabel, { folder: shown })}
        />
      </PopoverTrigger>
      <PopoverContent side="bottom" align="end" aria-label={CHAT.folderPath}>
        <div className="grid w-[18.75rem] max-w-full gap-2">
          <p className="m-0 text-caption font-medium text-secondary">{CHAT.folderPath}</p>
          <p className="m-0 min-w-0 text-body text-foreground">
            <PathText path={folder} wrap />
          </p>
          <span>
            <CopyButton value={folder} label={CHAT.folderCopy} />
          </span>
        </div>
      </PopoverContent>
    </Popover>
  );
}

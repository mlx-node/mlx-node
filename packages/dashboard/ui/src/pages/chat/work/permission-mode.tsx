import { Shield, ShieldCheck, ShieldOff } from 'lucide-react';
import {
  type MouseEvent,
  type ReactElement,
  type RefObject,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from 'react';

import { OptionList, type OptionItem } from '../../../components/arc-x/option-list/option-list';
import { ToolChip } from '../../../components/arc-x/tool-chip/tool-chip';
import { Button } from '../../../components/arc/button/button';
import { Dialog, DialogContent } from '../../../components/arc/dialog/dialog';
import { Popover, PopoverContent, PopoverTrigger } from '../../../components/arc/popover/popover';
import { Tooltip } from '../../../components/arc/tooltip/tooltip';
import { DashboardApiError, getJson, mutate } from '../../../lib/api';
import { CHAT, fill } from '../../../lib/chat-strings';
import type { ChatHostState, ConversationInfo, CwdProblem, PermissionMode } from '../../../lib/chat-types';
import { folderName, modeRank, placeWords } from '../../../lib/permission-text';
import type { ProviderId, ProvidersResponse } from '../../../lib/providers-types';
import { toast } from '../../../lib/toast';
import { providerName } from '../../../lib/web-tools';
import { conversationPath } from '../conversation-actions';
import { folderTooLong } from '../drafts';
import { CARD_ARM_MS } from './permission';
import styles from './permission-mode.module.css';

type Raised = Exclude<PermissionMode, 'ask'>;

const MODES: readonly PermissionMode[] = ['ask', 'auto', 'bypass'];

const ICONS: Record<PermissionMode, ReactElement> = {
  ask: <Shield size={14} strokeWidth={1.75} />,
  auto: <ShieldCheck size={14} strokeWidth={1.75} />,
  bypass: <ShieldOff size={14} strokeWidth={1.75} />,
};

/** What the confirm text says about this conversation. */
export interface PermissionFacts {
  folder: string | null;
  /** Providers whose key is in the Work environment. */
  keys: readonly ProviderId[];
  /** A tool hook of the user was found; null on the New screen, where the confirm reads it when it opens. */
  userHooks: boolean | null;
  /** The New screen: the mode is for the conversation its first message makes. */
  draft: boolean;
  /** Open permission cards that offer "Allow once": a move to Bypass answers them. */
  openAllowOnce: number;
  /** The temp folder Auto covers beside the Work folder, as people type it (`/tmp`); null when none. */
  temp: string | null;
}

/** The mode the stream says: the host's list wins over the conversation's own field when both are known. */
export function permissionModeShown(
  raised: ChatHostState['permissionRaised'] | null,
  info: ConversationInfo | null,
): PermissionMode {
  if (info === null || info.mode !== 'work') return 'ask';
  if (raised === null) return info.permissionMode;
  return raised.find((entry) => entry.id === info.id)?.mode ?? 'ask';
}

/** Why the chip, and each move up, cannot be used now; null where it can. */
export interface PermissionBlock {
  chip: string | null;
  auto: string | null;
  bypass: string | null;
}

/**
 * First match wins: port down, a Work conversation still loading, the first message on its way (moves up only),
 * (Auto only) an open card, the agent blocker, the folder.
 */
export function permissionModeBlocked(input: {
  connected: boolean;
  loading: boolean;
  starting: boolean;
  cardOpen: boolean;
  /** The reason of the agent or engine state that stops Work. */
  blocker: string | undefined;
  /** Why the folder cannot be used. */
  folder: string | undefined;
}): PermissionBlock {
  if (!input.connected || input.loading) {
    const why = input.connected ? CHAT.startingTip : CHAT.blockers.reconnecting;
    return { chip: why, auto: why, bypass: why };
  }
  const rest = input.starting ? CHAT.startingTip : (input.blocker ?? input.folder ?? null);
  return { chip: null, auto: input.cardOpen && !input.starting ? CHAT.permission.cardOpen : rest, bypass: rest };
}

/** Why the folder of a conversation or of the New screen cannot be used. */
export function folderReason(
  isNew: boolean,
  folder: string | null,
  problem: CwdProblem | undefined,
): string | undefined {
  if (!isNew) return problem === undefined ? undefined : CHAT.cwdProblems[problem];
  if (folder === null) return CHAT.folderNeeded;
  return folderTooLong(folder) ? CHAT.folderTooLong : undefined;
}

/** Providers whose key the Work agent gets: the selected search provider, and Firecrawl when its Work skills are ready. */
export function workKeyProviders(
  host: ChatHostState['providers'] | null,
  list: ProvidersResponse | null,
): ProviderId[] {
  const out: ProviderId[] = [];
  const search = list !== null ? list.search : (host?.search ?? null);
  if (search !== null) out.push(search);
  if (list?.providers.find((provider) => provider.id === 'firecrawl')?.work.ready === true) out.push('firecrawl');
  return out;
}

/** The key sentence of a confirm, or null with no key in the Work environment. */
export function keySentence(keys: readonly ProviderId[], target: Raised): string | null {
  const words = CHAT.permission[target];
  if (keys.length === 0) return null;
  if (keys.length === 1) return fill(words.key, { provider: providerName(keys[0]!) });
  return fill(words.keys, { first: providerName(keys[0]!), second: providerName(keys[1]!) });
}

/** The keys in the Work environment, read again when the confirm text opens. */
export function useWorkKeys(open: boolean, host: ChatHostState['providers'] | null): ProviderId[] {
  const [list, setList] = useState<ProvidersResponse | null>(null);
  useEffect(() => {
    if (!open) return;
    let active = true;
    getJson<ProvidersResponse>('/providers').then(
      (body) => {
        if (active) setList(body);
      },
      () => {
        if (active) setList(null);
      },
    );
    return () => {
      active = false;
    };
  }, [open]);
  return workKeyProviders(host, list);
}

/** The note of a mode in the menu and the tooltip: Auto names the temp folder it covers. */
export function modeNote(mode: PermissionMode, folder: string | null, temp: string | null): string {
  const p = CHAT.permission;
  const words = folder === null ? p.theFolder : folderName(folder);
  if (mode === 'auto' && temp !== null) return fill(p.notes.autoTemp, { folder: words, temp });
  return fill(p.notes[mode], { folder: words });
}

/** The accessible name of a mode: tooltip, chip name and status sentence. */
export function permissionModeName(mode: PermissionMode, folder: string | null, temp: string | null = null): string {
  const p = CHAT.permission;
  return fill(p.nameOf, { mode: p.modes[mode], note: modeNote(mode, folder, temp) });
}

/** What the mode is set on: an open Work conversation, or the New Work screen's draft and its folder. */
export type PermissionSubject = { kind: 'conversation'; id: string } | { kind: 'draft'; id: string; cwd: string };

function modePath(subject: PermissionSubject): string {
  return subject.kind === 'draft'
    ? `/chat/drafts/${encodeURIComponent(subject.id)}/permission-mode`
    : `${conversationPath(subject.id)}/permission-mode`;
}

/** The New screen's draft ends with a raised mode shown: its pick goes back to Ask. */
export function clearDraftMode(id: string): void {
  mutate('POST', modePath({ kind: 'draft', id, cwd: '' }), { mode: 'ask', up: false }).catch(() => undefined);
}

/** Whether the user has tool hooks for `cwd`, read once per folder; null until it answers or with no folder. */
export function useWorkHooks(cwd: string | null): boolean | null {
  const [read, setRead] = useState<{ cwd: string; hooks: boolean } | null>(null);
  useEffect(() => {
    if (cwd === null) return;
    let active = true;
    getJson<{ userHooks: boolean }>(`/chat/work/hooks?cwd=${encodeURIComponent(cwd)}`).then(
      (body) => {
        if (active) setRead({ cwd, hooks: body.userHooks });
      },
      () => {
        if (active) setRead({ cwd, hooks: true });
      },
    );
    return () => {
      active = false;
    };
  }, [cwd]);
  return read !== null && read.cwd === cwd ? read.hooks : null;
}

/**
 * The mode call. `up` is true only from a confirm's action, which sends the temp folder its text named (and on the New
 * screen its folder): the server never moves the mode up for a request sent with `up: false`, so a pick made on a view
 * older than the server's mode cannot raise it (409 `mode-changed`). A move up is one at a time per subject (a second
 * one while it is in flight is dropped); a move down is sent at once, also while an up is in flight. A failure is a
 * toast, but 409 `mode-changed` is not: the stream shows the mode. `pending` is the up in flight of this subject only.
 * `onUp` runs as a move up is sent; `onMode` gets the mode the Control Panel answered.
 */
export function usePermissionMode(
  subject: PermissionSubject | null,
  on: { onUp?: () => void; onMode?: (mode: PermissionMode) => void } = {},
): {
  pending: boolean;
  set: (target: PermissionMode, up: boolean, temp?: string | null) => Promise<boolean>;
} {
  const [raising, setRaising] = useState<ReadonlySet<string>>(() => new Set());
  const busy = useRef(new Set<string>());
  const told = useRef(on);
  useLayoutEffect(() => {
    told.current = on;
  });
  const id = subject?.id ?? null;
  const cwd = subject?.kind === 'draft' ? subject.cwd : null;
  const set = useCallback(
    async (target: PermissionMode, up: boolean, temp: string | null = null): Promise<boolean> => {
      if (id === null || (up && busy.current.has(id))) return false;
      const flight = (on: boolean): void => {
        if (on) busy.current.add(id);
        else busy.current.delete(id);
        setRaising(new Set(busy.current));
      };
      if (up) {
        flight(true);
        told.current.onUp?.();
      }
      try {
        const named = up && target === 'auto' && temp !== null ? { temp } : {};
        const folder = up && cwd !== null ? { cwd } : {};
        const path = modePath(cwd === null ? { kind: 'conversation', id } : { kind: 'draft', id, cwd });
        const reply = await mutate<{ mode: PermissionMode }>('POST', path, { mode: target, up, ...folder, ...named });
        told.current.onMode?.(reply.mode);
        return true;
      } catch (err) {
        if (!(err instanceof DashboardApiError && err.reason === 'mode-changed')) toast.error(CHAT.permission.failed);
        return false;
      } finally {
        if (up) flight(false);
      }
    },
    [id, cwd],
  );
  return { pending: id !== null && raising.has(id), set };
}

/**
 * The confirm of a move up: its text, Back, then the action. The action counts a press only when it is a trusted
 * event of the user and the text it confirms has been on screen for {@link CARD_ARM_MS}; any change of that text
 * starts the clock again. Mount it with `key={target}` so each target has a clock of its own.
 */
function ConfirmView(props: {
  target: Raised;
  facts: Omit<PermissionFacts, 'keys'>;
  providers: ChatHostState['providers'] | null;
  pending: boolean;
  heading: boolean;
  textRef: RefObject<HTMLDivElement | null>;
  onBack: () => void;
  onConfirm: () => void;
}): ReactElement {
  const { target, facts } = props;
  const p = CHAT.permission;
  const folder = facts.folder === null ? p.theFolder : folderName(facts.folder);
  const temp = target === 'auto' ? facts.temp : null;
  const place = placeWords(facts.folder, temp);
  const keys = useWorkKeys(true, props.providers);
  const key = keySentence(keys, target);
  const read = useWorkHooks(facts.draft ? facts.folder : null);
  const userHooks = facts.draft ? read : facts.userHooks;
  const hooks = userHooks === true ? p[target].hooks : null;
  const known = userHooks !== null;
  const open = target === 'bypass' ? facts.openAllowOnce : 0;
  const shownAt = useRef<number | null>(null);
  useLayoutEffect(() => {
    shownAt.current = known ? performance.now() : null;
  }, [folder, temp, key, hooks, open, known]);
  const press = (event: MouseEvent<HTMLButtonElement>): void => {
    if (event.nativeEvent.isTrusted !== true) return;
    if (shownAt.current === null || event.timeStamp - shownAt.current < CARD_ARM_MS) return;
    props.onConfirm();
  };
  const ends = fill(p.ends, { mode: p.modes[target] });
  const lines: ReactElement[] = [];
  const line = (id: string, text: string | null): void => {
    if (text !== null)
      lines.push(
        <p key={id} className={styles.line}>
          {text}
        </p>,
      );
  };
  if (target === 'auto') {
    const a = p.auto;
    line('1', fill(a.line1, { place }));
    line('temp', temp === null ? null : fill(a.temp, { temp }));
    line('2', fill(a.line2, { place }));
    line('3', fill(a.line3, { folder }));
    lines.push(
      <p key="4" className={styles.line}>
        {`${a.line4Start} `}
        <code className={styles.code}>{a.line4Code}</code>
        {` ${a.line4End}`}
        {key !== null ? ` ${key}` : null}
      </p>,
    );
    line('hooks', hooks);
    line('ends', ends);
  } else {
    const b = p.bypass;
    line('1', fill(b.line1, { folder }));
    line('2', b.line2);
    line('3', key !== null ? `${b.line3} ${key}` : b.line3);
    line('hooks', hooks);
    line('4', b.line4);
    line('ends', ends);
    line('open', open === 0 ? null : open === 1 ? b.cardOpen : fill(b.cardsOpen, { n: open }));
  }
  line('draft', facts.draft ? p.draftFoot : null);
  return (
    <div className={styles.confirm} data-permission-confirm={target}>
      <div ref={props.textRef} tabIndex={-1} className={styles.text} data-permission-text="">
        {props.heading ? <p className={styles.title}>{p[target].title}</p> : null}
        {lines}
      </div>
      <div className={styles.actions}>
        <Button type="button" variant="ghost" size="sm" onClick={props.onBack}>
          {p.back}
        </Button>
        <Button
          type="button"
          variant={target === 'bypass' ? 'danger' : 'primary'}
          size="sm"
          loading={props.pending}
          onClick={press}
        >
          {p[target].confirm}
        </Button>
      </div>
    </div>
  );
}

export interface PermissionModeChipProps {
  place: 'composer' | 'header';
  /** The mode the stream says. */
  mode: PermissionMode;
  block: PermissionBlock;
  facts: Omit<PermissionFacts, 'keys'>;
  providers: ChatHostState['providers'] | null;
  /** A move up is in flight. */
  pending: boolean;
  onSet: (target: PermissionMode, up: boolean, temp?: string | null) => Promise<boolean>;
}

/**
 * The permission mode chip and its menu. A lower mode is one click; a higher mode opens its confirm. A confirm closed
 * while its move up is in flight calls that move back.
 */
export function PermissionModeChip(props: PermissionModeChipProps): ReactElement {
  const { mode, block } = props;
  const p = CHAT.permission;
  const [open, setOpen] = useState(false);
  const [confirm, setConfirm] = useState<{ target: Raised; from: PermissionMode } | null>(null);
  const contentRef = useRef<HTMLDivElement | null>(null);
  const textRef = useRef<HTMLDivElement | null>(null);
  const backTo = useRef<Raised | null>(null);
  const chipBlocked = block.chip !== null;
  const isOpen = open && !chipBlocked;
  const shown = confirm !== null && confirm.from === mode && block[confirm.target] === null ? confirm : null;

  // A hidden popover must not open again by itself when the block ends.
  useEffect(() => {
    if (chipBlocked) {
      setOpen(false);
      setConfirm(null);
    }
  }, [chipBlocked]);
  // A confirm whose facts went stale goes back to the menu.
  useEffect(() => {
    if (confirm !== null && shown === null) setConfirm(null);
  }, [confirm, shown]);
  const focusView = (): void => {
    if (shown !== null) {
      textRef.current?.focus({ preventScroll: true });
      return;
    }
    const radios = contentRef.current?.querySelectorAll<HTMLElement>('[role="radio"]');
    const at = backTo.current !== null ? modeRank(backTo.current) : modeRank(mode);
    backTo.current = null;
    radios?.[at]?.focus({ preventScroll: true });
  };
  const focusRef = useRef(focusView);
  useLayoutEffect(() => {
    focusRef.current = focusView;
  });
  const view = shown === null ? null : shown.target;
  useEffect(() => {
    if (isOpen) focusRef.current();
  }, [isOpen, view]);

  const options: OptionItem[] = MODES.map((target) => {
    const up = modeRank(target) > modeRank(mode);
    const why = up && target !== 'ask' ? block[target] : null;
    return {
      value: target,
      title: p.modes[target],
      note: modeNote(target, props.facts.folder, props.facts.temp),
      trailing: target === 'bypass' ? <ShieldOff size={14} strokeWidth={1.75} aria-hidden="true" /> : undefined,
      tone: target === 'bypass' ? 'danger' : undefined,
      disabled: why !== null,
      disabledReason: why ?? undefined,
    };
  });

  const pick = (value: string): void => {
    const target = MODES.find((m) => m === value);
    if (target === undefined || target === mode) return;
    if (modeRank(target) < modeRank(mode)) {
      setOpen(false);
      void props.onSet(target, false);
      return;
    }
    if (target === 'ask' || block[target] !== null) return;
    setConfirm({ target, from: mode });
  };

  const name = permissionModeName(mode, props.facts.folder, props.facts.temp);
  const header = props.place === 'header';
  return (
    <Popover
      open={isOpen}
      onOpenChange={(next) => {
        if (!next) {
          if (shown !== null && props.pending) void props.onSet(shown.from, false);
          setOpen(false);
          setConfirm(null);
        }
      }}
    >
      <Tooltip content={block.chip ?? name}>
        <PopoverTrigger asChild>
          <ToolChip
            icon={ICONS[mode]}
            label={p.modes[mode]}
            tone={mode === 'auto' ? 'active' : mode === 'bypass' ? 'danger' : undefined}
            keepLabel={mode === 'bypass'}
            aria-label={name}
            aria-disabled={chipBlocked ? true : undefined}
            className={header ? styles.header : undefined}
            data-permission-chip={props.place}
            data-permission-mode={mode}
            onClick={(event) => {
              event.preventDefault();
              if (chipBlocked) return;
              setConfirm(null);
              setOpen((current) => !current);
            }}
          />
        </PopoverTrigger>
      </Tooltip>
      <PopoverContent
        ref={contentRef}
        side={header ? 'bottom' : 'top'}
        align={header ? 'end' : 'start'}
        aria-label={shown === null ? p.name : p[shown.target].title}
        className={styles.popover}
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          focusView();
        }}
      >
        {shown === null ? (
          <div className={styles.menu}>
            <OptionList label={p.name} value={mode} onValueChange={pick} options={options} wrap />
            <p className={styles.foot}>{props.facts.draft ? p.draftFoot : p.menuFoot}</p>
          </div>
        ) : (
          <ConfirmView
            key={shown.target}
            target={shown.target}
            facts={props.facts}
            providers={props.providers}
            pending={props.pending}
            heading
            textRef={textRef}
            onBack={() => {
              if (props.pending) void props.onSet(shown.from, false);
              backTo.current = shown.target;
              setConfirm(null);
            }}
            onConfirm={() => {
              void props.onSet(shown.target, true, props.facts.temp).then((done) => {
                if (done) {
                  setOpen(false);
                  setConfirm(null);
                }
              });
            }}
          />
        )}
      </PopoverContent>
    </Popover>
  );
}

export interface PermissionModeDialogProps {
  /** The mode the palette command asked for; null = closed. */
  target: Raised | null;
  /** The mode the stream says. */
  mode: PermissionMode;
  onClose: () => void;
  facts: Omit<PermissionFacts, 'keys'>;
  providers: ChatHostState['providers'] | null;
  pending: boolean;
  onSet: (target: PermissionMode, up: boolean, temp?: string | null) => Promise<boolean>;
}

/** The confirm of a move up as a dialog, for the palette command; closed while its move is in flight, it calls it back. */
export function PermissionModeDialog(props: PermissionModeDialogProps): ReactElement {
  const textRef = useRef<HTMLDivElement | null>(null);
  const [last, setLast] = useState<Raised>(props.target ?? 'auto');
  if (props.target !== null && props.target !== last) setLast(props.target);
  const target = props.target ?? last;
  const dismiss = (): void => {
    if (props.pending) void props.onSet(props.mode, false);
    props.onClose();
  };
  return (
    <Dialog
      open={props.target !== null}
      onOpenChange={(next) => {
        if (!next) dismiss();
      }}
    >
      <DialogContent
        title={CHAT.permission[target].title}
        aria-describedby={undefined}
        className={styles.dialog}
        onOpenAutoFocus={(event) => {
          event.preventDefault();
          textRef.current?.focus({ preventScroll: true });
        }}
      >
        <ConfirmView
          key={target}
          target={target}
          facts={props.facts}
          providers={props.providers}
          pending={props.pending}
          heading={false}
          textRef={textRef}
          onBack={dismiss}
          onConfirm={() => {
            void props.onSet(target, true, props.facts.temp).then((done) => {
              if (done) props.onClose();
            });
          }}
        />
      </DialogContent>
    </Dialog>
  );
}

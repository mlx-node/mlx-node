import {
  Copy,
  FolderOpen,
  Globe,
  MessageCircle,
  Pencil,
  ScanSearch,
  Shield,
  ShieldCheck,
  ShieldOff,
  Square,
  SquareTerminal,
  Trash2,
} from 'lucide-react';
import {
  createElement,
  type ReactElement,
  type ReactNode,
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';

import { ChatFrame, JumpButton, useChatDocked } from '../components/arc-x/chat-frame/chat-frame';
import { Composer, type ComposerNotice, type ComposerPhase } from '../components/arc-x/composer/composer';
import { ConfirmDialog } from '../components/arc-x/confirm-dialog/confirm-dialog';
import type { ConversationItem } from '../components/arc-x/conversation-list/conversation-list';
import { InteractionPresence } from '../components/arc-x/interaction-card/interaction-card';
import { ModelPicker } from '../components/arc-x/model-picker/model-picker';
import { Placeholder } from '../components/arc-x/placeholder/placeholder';
import { SwapCell } from '../components/arc-x/swap-cell/swap-cell';
import { Button } from '../components/arc/button/button';
import { ChatThread, ConversationNotices, type TurnAction } from '../components/chat-thread';
import { UnloadButton } from '../components/unload-button';
import { DashboardApiError, mutate } from '../lib/api';
import { isSlashText } from '../lib/chat-input';
import type { ConversationState } from '../lib/chat-store';
import { CHAT, fill } from '../lib/chat-strings';
import { answerMarkdown, chatTitle, conversationMarkdown, splitTurns } from '../lib/chat-turns';
import type {
  ChatHostState,
  ChatMode,
  ConversationInfo,
  ConversationRow,
  Interaction,
  PermissionMode,
  TurnState,
} from '../lib/chat-types';
import { getTransportUp, subscribeConnection } from '../lib/connection';
import { formatBytes } from '../lib/format';
import { folderName, modeRank, settingsToast, tempFor } from '../lib/permission-text';
import { pickFolder } from '../lib/pick-folder';
import { useStickToBottom } from '../lib/stick-to-bottom';
import { toast } from '../lib/toast';
import type { TranscriptEntry } from '../lib/types';
import { useJson } from '../lib/use-api';
import { type AppCommand, useCommands } from '../shell/commands';
import { useCrumbs } from '../shell/crumbs';
import { type Hotkey, useHotkeys, usePageSearch } from '../shell/hotkeys';
import { useScrollRoot } from '../shell/scroll-root';
import { type BlockAction, copyDetails, infoNotice, modeBlocker, sendFailure, type StripNotice } from './chat/blockers';
import { modelBases, startModel, WebChip } from './chat/composer-tools';
import { conversationPath, deleteContent, RenameDialog, type RenameTarget } from './chat/conversation-actions';
import {
  chooseFolder,
  chosenFolder,
  folderTooLong,
  forgetPlace,
  isChatMode,
  lastPlaceOf,
  NEW_PLACE,
  pickedVariants,
  readDraft,
  recentFolders,
  rememberPlace,
  storedMode,
  storedModel,
  storeMode,
  storeModel,
  storeVariant,
  writeDraft,
} from './chat/drafts';
import { ListPane } from './chat/list-pane';
import { NewScreen } from './chat/new-screen';
import { ModeSwitch, NewButton, ThreadBar } from './chat/thread-bar';
import { useChatHost } from './chat/use-chat-host';
import { forgetConversation, useConversation } from './chat/use-conversation';
import { useUnload } from './chat/use-unload';
import { NewFolderChip, OpenFolderChip } from './chat/work/folder-chip';
import { asksSentence, type InteractionAnswer, PermissionCard, PlanCard } from './chat/work/permission';
import {
  clearDraftMode,
  folderReason,
  PermissionModeChip,
  PermissionModeDialog,
  permissionModeBlocked,
  permissionModeName,
  permissionModeShown,
  type PermissionSubject,
  usePermissionMode,
} from './chat/work/permission-mode';

interface ListResponse {
  conversations: ConversationRow[];
}

interface CreateReply {
  conversation: ConversationInfo;
  turnId: string;
}

/** A message on its way: shown as the user's bubble until the stream carries it. */
interface Sending {
  /** The conversation it goes to; null while the create call is in flight. */
  conversationId: string | null;
  mode: ChatMode;
  text: string;
  /** Entries of the conversation when it was sent: the stream has it once there are more. */
  base: number;
  lastEnd: ConversationState['lastEnd'];
  ts: number;
  /** A Work create: the mode the chip showed when Send was pressed, and the draft and folder the create named. */
  permission: { shown: PermissionMode; draft: string; cwd: string } | null;
}

const NO_RAISED: ChatHostState['permissionRaised'] = [];
const STARTING: TurnState = { turnId: '', phase: 'starting', startedAt: 0, wait: null };
let newVisits = 0;
const glyph = (icon: typeof Copy): ReactNode => createElement(icon, { size: 16, strokeWidth: 1.75 });

function chatUrl(id: string): string {
  return `/chat?c=${encodeURIComponent(id)}`;
}

function newUrl(mode: ChatMode): string {
  return `/chat?mode=${mode}`;
}

function writeClipboard(text: string): Promise<void> {
  return navigator.clipboard?.writeText(text) ?? Promise.reject(new Error('no clipboard'));
}

function dialogOpen(): boolean {
  return document.querySelector('[role="dialog"], [role="alertdialog"], [data-radix-popper-content-wrapper]') !== null;
}

function withShown(
  rows: readonly ConversationRow[] | undefined,
  shown: ConversationInfo | null,
  mode: ChatMode,
): readonly ConversationRow[] | undefined {
  if (rows === undefined || shown === null || shown.mode !== mode || rows.some((row) => row.id === shown.id))
    return rows;
  return [{ ...shown, turns: 0 }, ...rows];
}

export default function Chat(): ReactElement {
  const [params] = useSearchParams();
  const navigate = useNavigate();
  const rawId = params.get('c');
  const conversationId = rawId !== null && rawId !== '' ? rawId : null;
  const modeParam = params.get('mode');
  const connected = useSyncExternalStore(subscribeConnection, getTransportUp, getTransportUp);
  const hostSnapshot = useChatHost();
  const host = hostSnapshot.state;
  const view = useConversation(conversationId);
  const conv = view.state;
  const gone = conversationId !== null ? view.gone : null;

  const chatList = useJson<ListResponse>('/chat/conversations?mode=chat');
  const workList = useJson<ListResponse>('/chat/conversations?mode=work');
  const reloadChat = chatList.reload;
  const reloadWork = workList.reload;
  const reloadLists = useCallback(() => {
    reloadChat();
    reloadWork();
  }, [reloadChat, reloadWork]);
  const listsVersion = useRef(hostSnapshot.conversationsVersion);
  useEffect(() => {
    if (listsVersion.current === hostSnapshot.conversationsVersion) return;
    listsVersion.current = hostSnapshot.conversationsVersion;
    reloadLists();
  }, [hostSnapshot.conversationsVersion, reloadLists]);

  // The open conversation is listed while its first answer waits: the stored list hides a session with no message yet.
  const shownInfo = gone === null ? (conv?.conversation ?? null) : null;
  const chatRows = useMemo(
    () => withShown(chatList.data?.conversations, shownInfo, 'chat'),
    [chatList.data, shownInfo],
  );
  const workRows = useMemo(
    () => withShown(workList.data?.conversations, shownInfo, 'work'),
    [workList.data, shownInfo],
  );
  const rows = useMemo(() => [...(chatRows ?? []), ...(workRows ?? [])], [chatRows, workRows]);
  const rowOf = (id: string | null): ConversationRow | undefined =>
    id === null ? undefined : rows.find((row) => row.id === id);
  const recent = useMemo(() => recentFolders(workList.data?.conversations ?? []), [workList.data]);
  const [folderChoice, setFolderChoice] = useState<string | null>(() => chosenFolder());
  // The last Work folder is preselected; the user's own choice in this window wins.
  const newFolder = folderChoice ?? recent[0] ?? null;

  const [sending, setSending] = useState<Sending | null>(null);
  const isNew = conversationId === null || gone !== null;
  const place = conversationId ?? NEW_PLACE;
  const newMode: ChatMode = isChatMode(modeParam) ? modeParam : storedMode();
  const mode: ChatMode = isNew
    ? sending?.conversationId === null
      ? sending.mode
      : newMode
    : (conv?.conversation.mode ?? rowOf(conversationId)?.mode ?? sending?.mode ?? storedMode());

  useEffect(() => {
    if (conversationId === null && isChatMode(modeParam)) storeMode(modeParam);
  }, [conversationId, modeParam]);

  const openId = conv?.conversation.id ?? null;
  const openMode = conv?.conversation.mode ?? null;
  const openFolder = isNew || mode !== 'work' ? null : (conv?.conversation.cwd ?? rowOf(conversationId)?.cwd ?? null);
  const cwdProblem = isNew || mode !== 'work' ? undefined : conv?.conversation.cwdProblem;
  useEffect(() => {
    if (openId === null || openMode === null) return;
    rememberPlace(openMode, openId);
    storeMode(openMode);
  }, [openId, openMode]);

  useEffect(() => {
    if (gone !== null && conversationId !== null) forgetPlace(conversationId);
  }, [gone, conversationId]);

  // The draft of the place on screen; a place keeps its own while the page shows another.
  const [draft, setDraftState] = useState(() => readDraft(place));
  const [shownPlace, setShownPlace] = useState(place);
  const [createdId, setCreatedId] = useState<string | null>(null);
  const [threadKey, setThreadKey] = useState(place);
  const [stripError, setStripError] = useState<StripNotice | null>(null);
  const [failedText, setFailedText] = useState<{ place: string; text: string } | null>(null);
  if (shownPlace !== place) {
    setShownPlace(place);
    setDraftState(readDraft(place));
    setStripError(null);
    setFailedText(null);
    // The created conversation keeps the New screen's thread cell: the bubble stays where it was drawn.
    if (shownPlace !== NEW_PLACE || createdId !== place) {
      newVisits += 1;
      setThreadKey(place === NEW_PLACE ? `${NEW_PLACE}:${newVisits}` : place);
      if (createdId !== null) setCreatedId(null);
    }
  }
  // The New Work screen's draft: a mode picked there is kept by the Control Panel for this id and folder only.
  const [modeDraft, setModeDraft] = useState(() => ({
    id: crypto.randomUUID(),
    folder: newFolder,
    visit: threadKey,
    mode,
  }));
  if (modeDraft.folder !== newFolder || modeDraft.visit !== threadKey || modeDraft.mode !== mode)
    setModeDraft({ id: crypto.randomUUID(), folder: newFolder, visit: threadKey, mode });
  const draftShownId = useRef<string | null>(null);
  const draftSent = useRef<string | null>(null);
  useLayoutEffect(() => {
    draftShownId.current = host?.permissionDraft?.draft ?? null;
  });
  useEffect(() => {
    const id = modeDraft.id;
    return () => {
      if (draftShownId.current === id && draftSent.current !== id) clearDraftMode(id);
    };
  }, [modeDraft.id]);
  const [lowered, setLowered] = useState<{ ts: number; mode: PermissionMode } | null>(null);
  const [born, setBorn] = useState<{ conversation: string; draft: string } | null>(null);
  const placeRef = useRef(place);
  useLayoutEffect(() => {
    placeRef.current = place;
  });
  const setDraft = useCallback(
    (text: string) => {
      setDraftState(text);
      writeDraft(place, text);
    },
    [place],
  );

  const [status, setStatus] = useState({ n: 0, text: '' });
  const announce = useCallback((text: string) => setStatus((current) => ({ n: current.n + 1, text })), []);

  const [dismissedEnd, setDismissedEnd] = useState<ConversationState['lastEnd']>(null);

  // The bubble leaves once the stream carries the message, or the turn ended without it. A turn that failed before
  // it ran never carried the message: its text comes back for Retry.
  useEffect(() => {
    if (sending === null || sending.conversationId === null || sending.conversationId !== conversationId) return;
    if (gone !== null) {
      setSending(null);
      return;
    }
    if (conv === null) return;
    if (conv.entries.length > sending.base || conv.lastEnd !== sending.lastEnd) {
      setSending(null);
      const end = conv.lastEnd;
      if (conv.entries.length <= sending.base && end?.failure !== undefined && end.outcome === undefined) {
        setFailedText({ place: sending.conversationId, text: sending.text });
        if (readDraft(sending.conversationId) === '') {
          writeDraft(sending.conversationId, sending.text);
          setDraftState(sending.text);
        }
      }
    }
  }, [sending, conv, conversationId, gone]);

  const sendingHere =
    sending !== null && (sending.conversationId === null ? isNew : sending.conversationId === conversationId);
  const turn = conv?.turn ?? null;
  const running = turn !== null;
  const phase: ComposerPhase = turn !== null ? turn.phase : sendingHere ? 'creating' : 'idle';
  const busy = phase !== 'idle';

  // Models.
  const unload = useUnload(host);
  const unloadControl = unload.control;
  const resident = useMemo(() => new Set(unload.resident.map((entry) => entry.modelId)), [unload.resident]);
  const [newModel, setNewModel] = useState<string | null>(null);
  const [modelWorking, setModelWorking] = useState(false);
  const [picked, setPicked] = useState(pickedVariants);
  const pickerRef = useRef<HTMLSpanElement>(null);
  const startsOn = startModel(host, storedModel(mode));
  const chosenNew =
    newModel !== null && host?.models.some((model) => model.id === newModel) === true ? newModel : startsOn;
  const modelValue = isNew ? chosenNew : (conv?.conversation.modelId ?? null);
  const bases = useMemo(
    () => (host === null ? [] : modelBases(host, resident, mode, modelValue, picked)),
    [host, resident, mode, modelValue, picked],
  );
  const residentId = host?.models.find((model) => resident.has(model.id))?.id ?? null;
  const modelSize = useCallback(
    (id: string): number | null =>
      host?.models.find((model) => model.id === id)?.sizeBytes ??
      host?.resident.find((entry) => entry.modelId === id)?.sizeBytes ??
      null,
    [host],
  );
  const otherRunning = host?.active.some((entry) => entry.busy && entry.conversationId !== conversationId) === true;
  const residentSize = residentId === null ? null : modelSize(residentId);
  const memory =
    residentId === null
      ? undefined
      : {
          text:
            residentSize === null
              ? fill(CHAT.engine.residentNoSize, { model: residentId })
              : fill(CHAT.engine.resident, { model: residentId, size: formatBytes(residentSize) }),
          action: <UnloadButton control={unload.control(residentId)} />,
        };
  const threadModel = conv?.conversation.modelId ?? modelValue;
  const threadUnload = useMemo(
    () => (threadModel === null ? undefined : unloadControl(threadModel)),
    [threadModel, unloadControl],
  );

  const changeModel = async (id: string): Promise<void> => {
    storeModel(mode, id);
    const base = host?.models.find((model) => model.id === id)?.base;
    if (base !== undefined) {
      storeVariant(base, id);
      setPicked((current) => ({ ...current, [base.toLowerCase()]: id }));
    }
    if (isNew) {
      setNewModel(id);
      return;
    }
    if (conversationId === null) return;
    setModelWorking(true);
    try {
      await mutate<ConversationInfo>('PATCH', conversationPath(conversationId), { modelId: id });
    } catch (err) {
      const reason = err instanceof DashboardApiError ? err.reason : undefined;
      if (reason === 'conversation-busy') toast.error(CHAT.sendErrors['conversation-busy'][0]);
      else toast.error(CHAT.modelUnavailable);
    } finally {
      setModelWorking(false);
    }
  };

  // Why Send is off.
  const blocker = modeBlocker(connected, host, mode);
  let blocked = blocker?.reason;
  if (blocked === undefined && isNew && mode === 'work' && newFolder === null) blocked = CHAT.folderNeeded;
  if (blocked === undefined && isNew && mode === 'work' && newFolder !== null && folderTooLong(newFolder))
    blocked = CHAT.folderTooLong;
  if (blocked === undefined && cwdProblem !== undefined) blocked = CHAT.cwdProblems[cwdProblem];
  if (blocked === undefined && isSlashText(draft)) blocked = CHAT.slashText;
  const panelShown = isNew && blocker?.panel !== undefined;

  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  const focusComposer = useCallback(() => {
    requestAnimationFrame(() => composerRef.current?.focus({ preventScroll: true }));
  }, []);

  const root = useScrollRoot();
  const stick = useStickToBottom(root, threadKey);

  const send = async (text: string): Promise<void> => {
    if (text.trim() === '' || isSlashText(text) || phase !== 'idle') return;
    if (modeBlocker(connected, host, mode) !== null) return;
    if (isNew && mode === 'work' && (newFolder === null || folderTooLong(newFolder))) return;
    if (!isNew && cwdProblem !== undefined) return;
    setStripError(null);
    setFailedText(null);
    setDismissedEnd(conv?.lastEnd ?? null);
    const fromNew = isNew;
    const target = fromNew ? null : conversationId;
    const origin = place;
    const base = fromNew ? 0 : (conv?.entries.length ?? 0);
    const next: Sending = {
      conversationId: target,
      mode,
      text,
      base,
      lastEnd: fromNew ? null : (conv?.lastEnd ?? null),
      ts: Date.now(),
      permission:
        fromNew && mode === 'work' && newFolder !== null
          ? { shown: permissionMode, draft: modeDraft.id, cwd: newFolder }
          : null,
    };
    setSending(next);
    // A Retry of a failed message leaves text typed since then in the composer.
    if (readDraft(origin) === text) setDraft('');
    stick.pin();
    announce(CHAT.status.sent);
    // The reply belongs to the place it was sent from; the user may be elsewhere by the time it arrives.
    const stillShown = (): boolean => placeRef.current === origin;
    const drop = (): void => setSending((current) => (current === next ? null : current));
    try {
      if (fromNew) {
        const body: { mode: ChatMode; text: string; modelId?: string; cwd?: string; draft?: string } = { mode, text };
        if (chosenNew !== null) body.modelId = chosenNew;
        if (next.permission !== null) {
          body.cwd = next.permission.cwd;
          body.draft = next.permission.draft;
          draftSent.current = next.permission.draft;
        }
        let reply: CreateReply;
        try {
          reply = await mutate<CreateReply>('POST', '/chat/conversations', body);
        } catch (err) {
          if (draftSent.current === body.draft) draftSent.current = null;
          throw err;
        }
        const id = reply.conversation.id;
        const answered = reply.conversation.permissionMode;
        if (next.permission !== null) {
          if (modeRank(answered) < modeRank(startShown.current)) toast.error(CHAT.permission.failed);
          lowerStart(next.ts, answered);
          setBorn({ conversation: id, draft: next.permission.draft });
        }
        rememberPlace(mode, id);
        reloadLists();
        if (stillShown()) {
          setCreatedId(id);
          setSending((current) => (current === next ? { ...next, conversationId: id, lastEnd: null } : current));
          void navigate(chatUrl(id));
        } else {
          drop();
        }
      } else if (target !== null) {
        await mutate('POST', `${conversationPath(target)}/prompt`, { text });
      }
    } catch (err) {
      drop();
      // Text typed while the send was pending stays; Retry still sends the failed text.
      const restore = readDraft(origin) === '';
      if (restore) writeDraft(origin, text);
      const reason = err instanceof DashboardApiError ? err.reason : undefined;
      const code = err instanceof DashboardApiError ? err.code : undefined;
      const outcome = sendFailure(reason, code, mode, err instanceof Error ? err.message : undefined);
      if ('toast' in outcome) toast.error(outcome.toast);
      else if (stillShown()) {
        if (restore) setDraftState(text);
        setStripError(outcome.strip);
        setFailedText({ place: origin, text });
      } else {
        toast.error(outcome.strip.text);
      }
    }
  };

  const picking = useRef(false);
  const pickAndChoose = useCallback(async () => {
    if (picking.current) return;
    picking.current = true;
    try {
      const path = await pickFolder();
      if (path !== null) {
        chooseFolder(path);
        setFolderChoice(path);
        setStripError(null);
      }
    } finally {
      picking.current = false;
    }
    focusComposer();
  }, [focusComposer]);
  const chooseRecent = useCallback(
    (path: string) => {
      chooseFolder(path);
      setFolderChoice(path);
      setStripError(null);
      focusComposer();
    },
    [focusComposer],
  );

  const stop = useCallback(() => {
    if (conversationId === null || turn === null || turn.phase === 'cancelling') return;
    mutate('POST', `${conversationPath(conversationId)}/cancel`).catch(() => {
      toast.error(CHAT.sendErrors.other[0]);
    });
  }, [conversationId, turn]);

  const runAction = (action: BlockAction): void => {
    setStripError(null);
    switch (action.kind) {
      case 'install':
        mutate('POST', '/chat/agent', { action: 'install' }).catch(() => undefined);
        return;
      case 'restart':
        mutate('POST', '/chat/agent', { action: 'restart', mode }).catch(() => undefined);
        return;
      case 'open-engine':
        void navigate('/engine');
        return;
      case 'open-models':
        void navigate('/models');
        return;
      case 'open-providers':
        void navigate('/providers');
        return;
      case 'copy-details':
        if (host !== null)
          writeClipboard(copyDetails(host, mode)).then(
            () => toast.success(CHAT.detailsCopied),
            () => undefined,
          );
        return;
      case 'retry':
        if (turnStrip !== null && conv !== null) setDismissedEnd(conv.lastEnd);
        if (retryText !== null) void send(retryText);
        return;
      case 'pick-model':
        pickerRef.current?.querySelector('button')?.click();
        return;
      case 'choose-folder':
        void pickAndChoose();
        return;
    }
  };

  const toNotice = (strip: StripNotice): ComposerNotice => ({
    tone: strip.tone,
    text: strip.text,
    working: strip.working,
    action:
      strip.action === undefined ? undefined : { label: strip.action.label, onSelect: () => runAction(strip.action!) },
    secondary:
      strip.secondary === undefined
        ? undefined
        : { label: strip.secondary.label, onSelect: () => runAction(strip.secondary!) },
  });

  const retryText = failedText !== null && failedText.place === place ? failedText.text : null;
  const endFailure = conv?.lastEnd?.failure;
  const turnStrip =
    conv !== null && endFailure !== undefined && conv.lastEnd?.outcome === undefined && dismissedEnd !== conv.lastEnd
      ? (() => {
          const outcome = sendFailure(endFailure.code, undefined, mode, undefined, conv.conversation.modelId);
          if (!('strip' in outcome)) return null;
          // Retry needs the text of the message that never ran; without it the strip has no button.
          const strip = outcome.strip;
          return strip.action?.kind === 'retry' && retryText === null ? { ...strip, action: undefined } : strip;
        })()
      : null;
  const modeStrip = panelShown ? undefined : blocker?.notice;
  const folderStrip: StripNotice | null =
    cwdProblem !== undefined ? { tone: 'danger', text: CHAT.cwdProblems[cwdProblem] } : null;
  const strip: StripNotice | null = modeStrip ?? stripError ?? folderStrip ?? turnStrip ?? infoNotice(host, mode);

  const onDraftChange = (text: string): void => {
    setDraft(text);
    if (stripError !== null) setStripError(null);
    if (turnStrip !== null && conv !== null) setDismissedEnd(conv.lastEnd);
  };

  // Navigation.
  const switchMode = useCallback(
    (next: ChatMode) => {
      if (next === mode) return;
      storeMode(next);
      announce(next === 'work' ? CHAT.status.toWork : CHAT.status.toChat);
      if (isNew) {
        void navigate(newUrl(next), { replace: true });
        return;
      }
      const placeOf = lastPlaceOf(next);
      void navigate(placeOf !== null ? chatUrl(placeOf) : newUrl(next));
    },
    [mode, isNew, navigate, announce],
  );

  const pageRef = useRef<HTMLDivElement | null>(null);
  const docked = useChatDocked(pageRef);
  const [listOpen, setListOpen] = useState(false);
  const [searchWanted, setSearchWanted] = useState(false);
  const searchRef = useRef<HTMLInputElement | null>(null);
  usePageSearch(searchRef);
  useEffect(() => {
    if (!searchWanted) return;
    const timer = setTimeout(() => {
      searchRef.current?.focus();
      setSearchWanted(false);
    }, 0);
    return () => clearTimeout(timer);
  }, [searchWanted, listOpen]);
  const [query, setQuery] = useState('');

  const pickedFromDrawer = useRef(false);
  const openConversation = useCallback(
    (id: string) => {
      const fromDrawer = !docked && listOpen;
      pickedFromDrawer.current = fromDrawer;
      setListOpen(false);
      if (id !== conversationId) void navigate(chatUrl(id));
      else if (!fromDrawer) focusComposer();
    },
    [conversationId, navigate, focusComposer, docked, listOpen],
  );
  const listCloseFocus = useCallback((opener: Element | null): HTMLElement | null => {
    if (pickedFromDrawer.current) {
      pickedFromDrawer.current = false;
      return composerRef.current;
    }
    if (opener instanceof HTMLElement && opener.isConnected && opener !== document.body) return opener;
    return composerRef.current;
  }, []);

  const onDeleted = useCallback(
    (id: string) => {
      forgetConversation(id);
      forgetPlace(id);
      reloadLists();
      if (id === conversationId) void navigate(newUrl(mode), { replace: true });
    },
    [conversationId, mode, navigate, reloadLists],
  );

  const [renameTarget, setRenameTarget] = useState<RenameTarget | null>(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const busyOf = (id: string): boolean =>
    host?.active.some((entry) => entry.conversationId === id && (entry.busy || entry.waiting)) === true ||
    (id === conversationId && busy);
  const focusBefore = useRef<Element | null>(null);
  const noteFocus = (): void => {
    focusBefore.current = document.activeElement;
  };
  const returnFocus = (): HTMLElement | null => {
    const before = focusBefore.current;
    if (before instanceof HTMLElement && before.isConnected && before !== document.body) return before;
    return composerRef.current;
  };
  const renameItem = (item: Pick<ConversationItem, 'id' | 'title'>): void => {
    noteFocus();
    setRenameTarget({ id: item.id, title: item.title, busy: busyOf(item.id) });
  };

  const title = isNew
    ? gone !== null
      ? gone === 'deleted'
        ? CHAT.goneDeleted
        : CHAT.goneUnknown
      : mode === 'work'
        ? CHAT.newWork
        : CHAT.newChat
    : chatTitle(conv?.conversation.title ?? rowOf(conversationId)?.title ?? '', mode);
  useCrumbs([{ label: CHAT.nav, to: '/chat' }, { label: title }]);
  useEffect(() => {
    document.title = `${title} — mlx-node`;
  }, [title]);

  const openedAnnounced = useRef<string | null>(null);
  useEffect(() => {
    if (openId === null || conv === null || openedAnnounced.current === openId) return;
    openedAnnounced.current = openId;
    if (createdId !== openId)
      announce(fill(CHAT.status.opened, { title: chatTitle(conv.conversation.title, conv.conversation.mode) }));
  }, [openId, conv, announce, createdId]);

  const landing = isNew ? `${NEW_PLACE}:${mode}` : place;
  // Focus: the composer whenever the page lands on a place (after the shell's own focus move), or the panel's action.
  // The mode switch keeps focus while it is used; a pick from the drawer focuses once the drawer has closed.
  useEffect(() => {
    if (pickedFromDrawer.current && !docked) return;
    pickedFromDrawer.current = false;
    const timer = setTimeout(() => {
      if (document.activeElement?.closest(`[aria-label="${CHAT.mode}"]`) != null) return;
      const action =
        document.querySelector<HTMLElement>('[data-interaction-card] button:not(:disabled)') ??
        document.querySelector<HTMLElement>('[data-hero-action]');
      (action ?? composerRef.current)?.focus({ preventScroll: true });
    }, 0);
    return () => clearTimeout(timer);
  }, [landing]);

  const copyLastAnswer = useCallback(() => {
    if (conv === null) return;
    const turns = splitTurns(conv.entries, conv.turns);
    const lastFinished =
      conv.turn !== null && turns[turns.length - 1]?.end === null ? turns.length - 2 : turns.length - 1;
    for (let i = lastFinished; i >= 0; i -= 1) {
      const text = answerMarkdown(turns[i]!.entries);
      if (text !== '') {
        writeClipboard(text).then(
          () => toast.success(CHAT.copiedAnswer),
          () => undefined,
        );
        return;
      }
    }
  }, [conv]);

  const hotkeys: readonly Hotkey[] = [
    {
      key: 'm',
      meta: true,
      shift: true,
      allowInInput: true,
      run: () => {
        switchMode(mode === 'chat' ? 'work' : 'chat');
        focusComposer();
      },
    },
    {
      key: '/',
      run: () => {
        if (docked) return;
        setListOpen(true);
        setSearchWanted(true);
      },
    },
    { key: 'h', meta: true, shift: true, allowInInput: true, run: () => (docked ? undefined : setListOpen((o) => !o)) },
    { key: 'c', meta: true, shift: true, allowInInput: true, run: copyLastAnswer },
    {
      key: 'F2',
      allowInInput: true,
      run: (event) => {
        const target = event.target instanceof Element ? event.target : null;
        if (target?.closest(`nav[aria-label="${CHAT.conversations}"]`) != null || dialogOpen()) return;
        if (isNew || conversationId === null || busy) return;
        renameItem({ id: conversationId, title });
      },
    },
  ];
  useHotkeys(hotkeys);

  // Palette commands of this page.
  const commands = useMemo<AppCommand[]>(() => {
    const group = CHAT.commands.group;
    const list: AppCommand[] = [
      {
        id: 'chat:switch',
        label: mode === 'chat' ? CHAT.commands.switchToWork : CHAT.commands.switchToChat,
        group,
        shortcut: '⌘ ⇧ M',
        icon: glyph(mode === 'chat' ? SquareTerminal : MessageCircle),
        run: () => switchMode(mode === 'chat' ? 'work' : 'chat'),
      },
    ];
    if (running) list.push({ id: 'chat:stop', label: CHAT.commands.stop, group, icon: glyph(Square), run: stop });
    if (!isNew && conversationId !== null) {
      const id = conversationId;
      if (!busy) {
        list.push({
          id: 'chat:rename',
          label: CHAT.commands.rename,
          group,
          icon: glyph(Pencil),
          run: () => {
            focusBefore.current = document.activeElement;
            setRenameTarget({ id, title, busy: false });
          },
        });
        list.push({
          id: 'chat:delete',
          label: CHAT.commands.remove,
          group,
          icon: glyph(Trash2),
          run: () => {
            focusBefore.current = document.activeElement;
            setDeleteOpen(true);
          },
        });
      }
      list.push({
        id: 'chat:inspect',
        label: CHAT.commands.inspect,
        group,
        icon: glyph(ScanSearch),
        run: () => void navigate(`/sessions/${encodeURIComponent(id)}?tab=activity`),
      });
      if (conv !== null && conv.entries.length > 0) {
        const entries = conv.entries;
        list.push({
          id: 'chat:copy-all',
          label: CHAT.commands.copyAll,
          group,
          icon: glyph(Copy),
          run: () => {
            writeClipboard(conversationMarkdown(entries)).then(
              () => void toast.success(CHAT.copiedAll),
              () => undefined,
            );
          },
        });
      }
    }
    if (mode === 'work' && isNew)
      list.push({
        id: 'chat:choose-folder',
        label: CHAT.commands.chooseFolder,
        group,
        icon: glyph(FolderOpen),
        run: () => void pickAndChoose(),
      });
    list.push({
      id: 'chat:web',
      label: CHAT.commands.webSettings,
      group,
      icon: glyph(Globe),
      run: () => void navigate('/providers'),
    });
    return list;
  }, [mode, running, stop, isNew, conversationId, busy, title, conv, navigate, switchMode, pickAndChoose]);
  useCommands(commands);

  // Requests of the agent (Work): one card at a time, in arrival order.
  const [closedCards, setClosedCards] = useState<ReadonlySet<string>>(() => new Set());
  const openCards: Interaction[] =
    mode === 'work' && conv !== null ? conv.interactions.filter((item) => !closedCards.has(item.id)) : [];
  const card = openCards[0];
  const cardId = card?.id ?? null;
  const cardRef = useRef(card);
  useLayoutEffect(() => {
    cardRef.current = card;
  });
  const shownCard = useRef<string | null>(null);
  useEffect(() => {
    const before = shownCard.current;
    shownCard.current = cardId;
    const focused = document.activeElement?.closest('[data-interaction-card]') != null;
    if (cardId !== null && cardId !== before && cardRef.current !== undefined && !focused)
      announce(asksSentence(cardRef.current));
    if (before !== null && cardId === null) {
      const active = document.activeElement;
      if (active === null || active === document.body || active.closest('[data-interaction-card]') !== null)
        focusComposer();
    }
  }, [cardId, announce, focusComposer]);
  const answerCard = useCallback(
    async (iid: string, answer: InteractionAnswer): Promise<void> => {
      if (conversationId === null) return;
      try {
        await mutate('POST', `${conversationPath(conversationId)}/interactions/${encodeURIComponent(iid)}`, answer);
      } catch (err) {
        const code = err instanceof DashboardApiError ? err.code : undefined;
        if (code === 'E_NOT_FOUND') {
          setClosedCards((current) => new Set(current).add(iid));
          toast.error(CHAT.permClosed);
          return;
        }
        toast.error(code === 'E_BAD_REQUEST' && err instanceof Error ? err.message : CHAT.sendErrors.other[0]);
        throw err;
      }
    },
    [conversationId],
  );
  // The permission mode of the open Work conversation or of the New Work screen's draft: the stream decides what is
  // shown, never this page. While the first message is on its way the chip keeps the mode it showed at Send.
  const workInfo = !isNew && conv !== null && conv.conversation.mode === 'work' ? conv.conversation : null;
  const sent = sendingHere ? (sending?.permission ?? null) : null;
  const startedWith = sent?.shown ?? null;
  const onDraft = sent !== null || (isNew && mode === 'work');
  const subject: PermissionSubject | null =
    sent !== null
      ? { kind: 'draft', id: sent.draft, cwd: sent.cwd }
      : onDraft
        ? { kind: 'draft', id: modeDraft.id, cwd: newFolder ?? '' }
        : workInfo !== null
          ? { kind: 'conversation', id: workInfo.id }
          : null;
  const subjectId = subject?.id ?? null;
  // The chip keeps its menu when its first message lands: the conversation takes the key of the draft that made it.
  const chipKey = born !== null && subjectId === born.conversation ? born.draft : (subjectId ?? '');
  const draftMode = host?.permissionDraft?.draft === modeDraft.id ? host.permissionDraft.mode : 'ask';
  const startMode =
    startedWith === null || sending === null
      ? null
      : lowered !== null && lowered.ts === sending.ts && modeRank(lowered.mode) < modeRank(startedWith)
        ? lowered.mode
        : startedWith;
  const permissionMode: PermissionMode =
    startMode ?? (onDraft ? draftMode : permissionModeShown(host?.permissionRaised ?? null, workInfo));
  const startShown = useRef<PermissionMode>('ask');
  useLayoutEffect(() => {
    startShown.current = startMode ?? 'ask';
  });
  const lowerStart = (ts: number, answered: PermissionMode): void =>
    setLowered((current) =>
      current !== null && current.ts === ts && modeRank(current.mode) <= modeRank(answered)
        ? current
        : { ts, mode: answered },
    );
  const permissionSwitch = usePermissionMode(subject, {
    onUp: () => {
      if (subject?.kind !== 'draft' || newFolder === null) return;
      chooseFolder(newFolder);
      setFolderChoice(newFolder);
    },
    onMode: (answered) => {
      if (startMode !== null && sending !== null) lowerStart(sending.ts, answered);
    },
  });
  const permissionBlock = permissionModeBlocked({
    connected,
    loading: subject === null,
    starting: startMode !== null,
    cardOpen: openCards.length > 0,
    blocker: modeBlocker(connected, host, 'work')?.reason,
    folder: folderReason(onDraft, newFolder, cwdProblem),
  });
  const openAllowOnce = openCards.filter(
    (item) =>
      item.kind === 'permission' &&
      item.options.some((option) => option.optionId === 'allow-once' && option.kind === 'allow_once'),
  ).length;
  const permissionFacts = {
    folder: openFolder ?? newFolder,
    userHooks: onDraft ? null : workInfo?.userHooks === true,
    draft: onDraft,
    openAllowOnce,
    temp: tempFor(host, openFolder ?? newFolder),
  };
  const [permissionAsked, setPermissionAsked] = useState<{ id: string; target: 'auto' | 'bypass' } | null>(null);
  // Only in its own conversation: a dialog keyed by the next one would mount open and never finish its exit.
  const permissionDialog = permissionAsked !== null && permissionAsked.id === subjectId ? permissionAsked.target : null;
  const permissionSeen = useRef<{ id: string | null; mode: PermissionMode; marks: number }>({
    id: null,
    mode: 'ask',
    marks: 0,
  });
  const permissionMarks = workInfo === null ? undefined : conv?.permissionMarks;
  const permissionFolder = permissionFacts.folder;
  const permissionTemp = permissionFacts.temp;
  useEffect(() => {
    const seen = permissionSeen.current;
    const count = permissionMarks?.length ?? 0;
    if (subjectId !== null && seen.id === subjectId) {
      if (seen.mode !== permissionMode) announce(permissionModeName(permissionMode, permissionFolder, permissionTemp));
      const settings = (permissionMarks?.slice(seen.marks) ?? []).filter((m) => m.kind === 'settings');
      if (settings.length > 0) {
        const changed: string[] = [];
        for (const mark of settings) for (const path of mark.changed) if (!changed.includes(path)) changed.push(path);
        toast.warning(CHAT.permission.settingsLabel, { description: settingsToast(changed) });
      }
    }
    permissionSeen.current = { id: subjectId, mode: permissionMode, marks: count };
  }, [subjectId, permissionMode, permissionMarks, permissionFolder, permissionTemp, announce]);
  const setPermissionMode = permissionSwitch.set;
  // A confirm whose facts went stale closes and does not open again by itself.
  const dialogBlocked = permissionDialog !== null && permissionBlock[permissionDialog] !== null;
  useEffect(() => {
    if (dialogBlocked) setPermissionAsked(null);
  }, [dialogBlocked]);
  useEffect(() => {
    setPermissionAsked(null);
  }, [subjectId, permissionMode]);
  const permissionCommands = useMemo<AppCommand[]>(() => {
    if (subjectId === null) return [];
    const group = CHAT.commands.group;
    const p = CHAT.permission;
    const label = (target: PermissionMode): string => fill(p.command, { mode: p.modes[target] });
    const list: AppCommand[] = [];
    const free = (target: 'auto' | 'bypass'): boolean => permissionBlock[target] === null;
    if (permissionBlock.chip !== null) return list;
    if (permissionMode !== 'ask')
      list.push({
        id: 'chat:permission-ask',
        label: label('ask'),
        group,
        icon: glyph(Shield),
        run: () => void setPermissionMode('ask', false),
      });
    if (permissionMode === 'bypass')
      list.push({
        id: 'chat:permission-auto',
        label: label('auto'),
        group,
        icon: glyph(ShieldCheck),
        run: () => void setPermissionMode('auto', false),
      });
    if (permissionMode === 'ask' && free('auto'))
      list.push({
        id: 'chat:permission-auto',
        label: `${label('auto')}…`,
        group,
        icon: glyph(ShieldCheck),
        run: () => setPermissionAsked({ id: subjectId, target: 'auto' }),
      });
    if (permissionMode !== 'bypass' && free('bypass'))
      list.push({
        id: 'chat:permission-bypass',
        label: `${label('bypass')}…`,
        group,
        icon: glyph(ShieldOff),
        run: () => setPermissionAsked({ id: subjectId, target: 'bypass' }),
      });
    return list;
  }, [
    subjectId,
    permissionMode,
    permissionBlock.chip,
    permissionBlock.auto,
    permissionBlock.bypass,
    setPermissionMode,
  ]);
  useCommands(permissionCommands);

  const position = { index: 1, count: openCards.length };
  const cardNode =
    mode === 'work' ? (
      <InteractionPresence>
        {card === undefined ? null : card.kind === 'permission' ? (
          <PermissionCard
            key={card.id}
            interaction={card}
            permissionMode={permissionMode}
            folder={openFolder}
            temp={permissionTemp}
            position={position}
            composerRef={composerRef}
            onAnswer={(answer) => answerCard(card.id, answer)}
          />
        ) : (
          <PlanCard
            key={card.id}
            interaction={card}
            folder={openFolder}
            position={position}
            composerRef={composerRef}
            onAnswer={(answer) => answerCard(card.id, answer)}
          />
        )}
      </InteractionPresence>
    ) : null;

  // The other workspace waits for an answer: said once per conversation that starts waiting.
  const otherMode: ChatMode = mode === 'chat' ? 'work' : 'chat';
  const waitingOther = (host?.active ?? [])
    .filter((entry) => entry.mode === otherMode && entry.waiting)
    .map((entry) => entry.conversationId)
    .sort()
    .join('\n');
  const waitingSeen = useRef<string>('');
  useEffect(() => {
    const before = new Set(waitingSeen.current === '' ? [] : waitingSeen.current.split('\n'));
    waitingSeen.current = waitingOther;
    if (waitingOther === '') return;
    if (waitingOther.split('\n').some((id) => !before.has(id)))
      announce(fill(CHAT.status.otherWaits, { mode: otherMode === 'work' ? CHAT.modeWork : CHAT.modeChat }));
  }, [waitingOther, otherMode, announce]);

  // The thread.
  let entries: readonly TranscriptEntry[] = conv?.entries ?? [];
  let threadTurn: TurnState | null = turn;
  let lastEnd: ConversationState['lastEnd'] = conv?.lastEnd ?? null;
  if (sendingHere && sending !== null) {
    if (entries.length <= sending.base) {
      entries = [...entries, { kind: 'user', ts: sending.ts, text: sending.text, promptIndex: null }];
    }
    threadTurn = turn ?? STARTING;
    if (turn === null) lastEnd = null;
  }
  const engineReason =
    host?.inference === 'stopped' ? CHAT.blockers['inference-stopped'][1] : (blocker?.reason ?? undefined);

  const sendRef = useRef(send);
  useLayoutEffect(() => {
    sendRef.current = send;
  });
  const onTurnAction = useCallback((action: TurnAction) => {
    if (action.kind === 'update-agent') {
      mutate('POST', '/chat/agent', { action: 'install' }).catch(() => undefined);
      return;
    }
    void sendRef.current(action.text);
  }, []);

  let body: ReactNode;
  if (gone !== null && !sendingHere) {
    body = (
      <div className="grid h-full content-center justify-items-center gap-3 py-6 text-center">
        <p className="m-0 text-body text-foreground">{gone === 'deleted' ? CHAT.goneDeleted : CHAT.goneUnknown}</p>
        <Button
          type="button"
          variant="secondary"
          size="sm"
          onClick={() => void navigate(newUrl(mode), { replace: true })}
        >
          {mode === 'work' ? CHAT.newWork : CHAT.newChat}
        </Button>
      </div>
    );
  } else if (isNew && !sendingHere) {
    body = (
      <NewScreen
        mode={mode}
        model={chosenNew}
        providers={host?.providers ?? null}
        blocker={blocker}
        folder={newFolder}
        onChooseFolder={() => void pickAndChoose()}
        onStarter={(text) => {
          setDraft(text);
          focusComposer();
        }}
        onAction={runAction}
      />
    );
  } else if (conv === null && !sendingHere) {
    body = (
      <div className="grid gap-3" aria-busy="true">
        <span className="flex justify-end">
          <Placeholder width="40%" height={36} radius="panel" />
        </span>
        <Placeholder width="90%" />
        <Placeholder width="70%" />
      </div>
    );
  } else {
    body = (
      <ChatThread
        conversationId={conversationId}
        mode={mode}
        entries={entries}
        turns={conv?.turns ?? []}
        turn={threadTurn}
        lastEnd={lastEnd}
        creating={sendingHere && turn === null}
        host={
          host === null
            ? null
            : { inference: host.inference, loading: host.loading, resident: unload.resident, answers: host.answers }
        }
        modelSize={modelSize}
        unload={threadUnload}
        asOfMs={view.asOfMs}
        model={threadModel}
        transcriptError={conv?.transcriptError ?? null}
        inspectHref={
          conversationId === null ? undefined : `/sessions/${encodeURIComponent(conversationId)}?tab=activity`
        }
        onTurnAction={onTurnAction}
        actionDisabledReason={engineReason}
        onStatus={announce}
        contentRef={stick.contentRef}
        folder={openFolder}
        tempRoot={permissionTemp}
        card={cardNode}
        marks={permissionMarks}
      />
    );
  }

  const removeContent =
    !isNew && conversationId !== null
      ? deleteContent({ id: conversationId, title, mode, folder: conv?.conversation.cwd ?? null }, onDeleted)
      : null;

  const placeholder = !isNew
    ? permissionMode === 'bypass'
      ? CHAT.permission.bypass.placeholder
      : CHAT.placeholderReply
    : mode === 'work'
      ? newFolder === null
        ? CHAT.placeholderWorkNoFolder
        : fill(permissionMode === 'bypass' ? CHAT.permission.bypass.placeholderNew : CHAT.placeholderWork, {
            folder: folderName(newFolder),
          })
      : CHAT.placeholderChat;

  const tools = (
    <>
      <span ref={pickerRef} className="inline-flex min-w-0" data-give="3">
        <ModelPicker
          value={modelValue}
          bases={bases}
          fallback={startsOn}
          onValueChange={(id) => void changeModel(id)}
          disabled={!isNew && busy}
          disabledReason={!isNew && busy ? CHAT.modelBusy : undefined}
          working={modelWorking}
          crowded={mode === 'work'}
          footer={
            residentId === null
              ? CHAT.modelFooterIdle
              : (host?.models.length ?? 0) >= 2
                ? fill(CHAT.modelFooter, { resident: residentId })
                : undefined
          }
          swapWarning={residentId !== null && otherRunning ? fill(CHAT.modelSwap, { resident: residentId }) : undefined}
          memory={memory}
        />
      </span>
      {mode !== 'work' ? null : isNew ? (
        <NewFolderChip folder={newFolder} recent={recent} onChoose={chooseRecent} onPick={() => void pickAndChoose()} />
      ) : openFolder !== null ? (
        <OpenFolderChip folder={openFolder} />
      ) : null}
      {host !== null ? <WebChip providers={host.providers} mode={mode} /> : null}
      {mode === 'work' ? (
        <PermissionModeChip
          key={chipKey}
          place="composer"
          mode={permissionMode}
          block={permissionBlock}
          facts={permissionFacts}
          providers={host?.providers ?? null}
          pending={permissionSwitch.pending}
          onSet={setPermissionMode}
        />
      ) : null}
    </>
  );

  const newButton = <NewButton mode={mode} onNew={() => void navigate(newUrl(mode))} />;

  return (
    <div ref={pageRef} className="min-w-0">
      <ChatFrame
        barStart={
          <>
            <ModeSwitch mode={mode} active={host?.active ?? []} onChange={switchMode} />
            {docked ? newButton : null}
          </>
        }
        barEnd={
          <ThreadBar
            title={title}
            open={
              removeContent !== null && conversationId !== null
                ? { id: conversationId, running: busy, remove: removeContent }
                : null
            }
            onRename={() => {
              if (conversationId !== null) renameItem({ id: conversationId, title });
            }}
            trailing={docked ? null : newButton}
            folder={openFolder !== null ? <OpenFolderChip folder={openFolder} /> : null}
            permission={
              subject?.kind === 'conversation' && permissionMode !== 'ask' ? (
                <PermissionModeChip
                  key={subjectId}
                  place="header"
                  mode={permissionMode}
                  block={permissionBlock}
                  facts={permissionFacts}
                  providers={host?.providers ?? null}
                  pending={permissionSwitch.pending}
                  onSet={setPermissionMode}
                />
              ) : null
            }
          />
        }
        list={
          <ListPane
            mode={mode}
            chat={{
              rows: chatRows,
              error: chatList.error !== undefined && chatList.data === undefined,
            }}
            work={{
              rows: workRows,
              error: workList.error !== undefined && workList.data === undefined,
            }}
            active={host?.active ?? []}
            raised={host?.permissionRaised ?? NO_RAISED}
            currentId={conversationId}
            query={query}
            onQueryChange={setQuery}
            searchRef={searchRef}
            onOpen={openConversation}
            onRename={renameItem}
            onDeleted={onDeleted}
            onRetry={reloadLists}
          />
        }
        listLabel={CHAT.conversations}
        listOpen={listOpen}
        onListOpenChange={setListOpen}
        listCloseFocus={listCloseFocus}
        jump={
          stick.showJump ? (
            <JumpButton
              onJump={(button) => {
                const focused = document.activeElement === button;
                stick.jump();
                if (focused) focusComposer();
              }}
              fresh={stick.fresh}
            />
          ) : null
        }
        dock={
          <div className="grid gap-2">
            {conv !== null ? <ConversationNotices notices={conv.notices} /> : null}
            <Composer
              value={draft}
              onValueChange={onDraftChange}
              placeholder={placeholder}
              phase={phase}
              onSend={() => void send(draft)}
              onStop={stop}
              blocked={blocked}
              notice={strip === null ? undefined : toNotice(strip)}
              tools={tools}
              hint={running ? CHAT.hintStop : CHAT.hintSend}
              hintOnFocus={running}
              textareaRef={composerRef}
            />
          </div>
        }
      >
        <SwapCell active={threadKey} items={{ [threadKey]: body }} className="h-full" />
      </ChatFrame>
      <div role="status" aria-live="polite" className="sr-only">
        <span key={status.n}>{status.text}</span>
      </div>
      <RenameDialog
        target={renameTarget}
        onOpenChange={(open) => {
          if (!open) setRenameTarget(null);
        }}
        onRenamed={() => reloadLists()}
        returnFocus={returnFocus}
      />
      {subjectId !== null ? (
        <PermissionModeDialog
          key={subjectId}
          target={permissionDialog}
          mode={permissionMode}
          onClose={() => setPermissionAsked(null)}
          facts={permissionFacts}
          providers={host?.providers ?? null}
          pending={permissionSwitch.pending}
          onSet={setPermissionMode}
        />
      ) : null}
      {removeContent !== null ? (
        <ConfirmDialog {...removeContent} open={deleteOpen} onOpenChange={setDeleteOpen} returnFocus={returnFocus} />
      ) : null}
    </div>
  );
}

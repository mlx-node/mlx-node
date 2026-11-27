import {
  BellRing,
  Copy,
  FoldVertical,
  FolderOpen,
  Gauge,
  GitFork,
  Globe,
  MessageCircle,
  Pencil,
  ScanSearch,
  ScrollText,
  Shield,
  ShieldCheck,
  ShieldOff,
  Square,
  SquareTerminal,
  Trash2,
  Undo2,
  Workflow,
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

import {
  ATTACH_MAX,
  type AttachmentInput,
  attachmentView,
  estimateTokens,
  heldImages,
  imageCount,
  WINDOW_SHARE,
} from '../../../src/chat/attachment-rule.js';
import { stepLevel } from '../../../src/chat/thinking-levels.js';
import type { PendingMessage } from '../components/agent-view';
import { ActivityRail } from '../components/arc-x/activity-rail/activity-rail';
import type { AgentHandlers, AgentTarget } from '../components/arc-x/agent-controls/agent-controls';
import { type AttachmentChipItem, AttachmentStrip } from '../components/arc-x/attachment-strip/attachment-strip';
import { BranchMenu } from '../components/arc-x/branch-menu/branch-menu';
import { ChatFrame, JumpButton, useChatLayout } from '../components/arc-x/chat-frame/chat-frame';
import { ComposerTray } from '../components/arc-x/composer-tray/composer-tray';
import {
  Composer,
  type ComposerNotice,
  type ComposerPhase,
  type ContextUse,
} from '../components/arc-x/composer/composer';
import { contextShare } from '../components/arc-x/composer/context-ring';
import { ConfirmDialog } from '../components/arc-x/confirm-dialog/confirm-dialog';
import { ContextPanel } from '../components/arc-x/context-panel/context-panel';
import type { ConversationItem } from '../components/arc-x/conversation-list/conversation-list';
import { GoalStrip } from '../components/arc-x/goal-strip/goal-strip';
import { InteractionPresence } from '../components/arc-x/interaction-card/interaction-card';
import { ModelPicker } from '../components/arc-x/model-picker/model-picker';
import { useStaticMotion } from '../components/arc-x/motion/use-static-motion';
import { Placeholder } from '../components/arc-x/placeholder/placeholder';
import { QueueList } from '../components/arc-x/queue-list/queue-list';
import { RunStrip } from '../components/arc-x/run-strip/run-strip';
import { SwapCell } from '../components/arc-x/swap-cell/swap-cell';
import { Button } from '../components/arc/button/button';
import {
  type AgentMarksOpen,
  ChatThread,
  ConversationNotices,
  noticeKey,
  type TurnAction,
} from '../components/chat-thread';
import { ForkOrigin } from '../components/history-mark';
import { planModeStatus } from '../components/plan-mode-line';
import { UnloadButton } from '../components/unload-button';
import { useNow } from '../components/wait-clock';
import {
  type Activity,
  activityOf,
  agentPaletteOf,
  askRun,
  hasActivity,
  type ScheduleRow,
  type TaskRow,
  transcriptOpens,
} from '../lib/activity';
import type { MessageEnd, MessageTarget } from '../lib/agent-message';
import { agentMarkCounts, neighbourAgent } from '../lib/agent-view';
import { type AgentAsk, type AgentRow as AgentRowData, agentRowOf, runAgents, subagentRows } from '../lib/agents';
import { DashboardApiError, getJson, mutate } from '../lib/api';
import {
  type DraftAttachment,
  drawCaption,
  drawCount,
  draftTokens,
  drawingNow,
  type PdfStageSink,
  type PdfWatch,
} from '../lib/chat-attachments';
import { isSlashText, slashRest, type AppCommand as AppCommandName } from '../lib/chat-input';
import { NO_QUEUE, type ConversationState } from '../lib/chat-store';
import { CHAT, fill } from '../lib/chat-strings';
import { answerMarkdown, chatTitle, conversationMarkdown, splitTurns } from '../lib/chat-turns';
import type {
  ChatHostState,
  ChatMode,
  ConversationInfo,
  ConversationRow,
  GoalAction,
  Interaction,
  InteractionAnswerResult,
  NeverRulesResponse,
  PermissionMode,
  ThinkingLevel,
  TurnState,
} from '../lib/chat-types';
import { QUEUE_MAX } from '../lib/chat-types';
import { getTransportUp, subscribeConnection } from '../lib/connection';
import { formatBytes, formatNumber } from '../lib/format';
import { goalOf, goalStatusLine, goalStatusOf, openGoalAt } from '../lib/goal';
import { useHomeDir } from '../lib/home-dir';
import { scratchShown } from '../lib/paths';
import { folderName, modeRank, settingsToast, tempFor } from '../lib/permission-text';
import { pickFolder } from '../lib/pick-folder';
import { runStripOf } from '../lib/run-strip';
import { runsAt, shownRunNames, stopText } from '../lib/run-view';
import { sessionUsage } from '../lib/session-usage';
import { useStickToBottom } from '../lib/stick-to-bottom';
import { toast } from '../lib/toast';
import type { TranscriptEntry } from '../lib/types';
import { useJson } from '../lib/use-api';
import { oneLine } from '../lib/web-tools';
import { type AppCommand, useCommands } from '../shell/commands';
import { useCrumbs } from '../shell/crumbs';
import { type Hotkey, useHotkeys, usePageSearch } from '../shell/hotkeys';
import { useScrollRoot } from '../shell/scroll-root';
import { messageAgent, stopAgent } from './chat/agent-actions';
import { AgentPane } from './chat/agent-pane';
import {
  attachPlan,
  attachProblem,
  fileWays,
  heldPictureChars,
  messageInputs,
  resendFiles,
  sentStages,
  type BlockAction,
  copyDetails,
  engineLostNotice,
  infoNotice,
  modeBlocker,
  sendFailure,
  type StripNotice,
} from './chat/blockers';
import { ComposerMenu, modelBases, PlanChip, planChip, startModel } from './chat/composer-tools';
import { conversationPath, deleteContent, RenameDialog, type RenameTarget } from './chat/conversation-actions';
import {
  attachDrafts,
  chooseFolder,
  chosenFolder,
  drawingCount,
  firstModeDraft,
  type FolderPick,
  folderTooLong,
  forgetPlace,
  isChatMode,
  lastPlaceOf,
  NEW_PLACE,
  newPlace,
  nextModeDraft,
  pickedVariants,
  placeBlock,
  placeEpoch,
  readAttachments,
  readDraft,
  readFailed,
  readPending,
  pickPath,
  readPlan,
  recentFolders,
  rememberPlace,
  storedMode,
  storedModel,
  storedThinking,
  storeMode,
  storeModel,
  storeThinking,
  stopPending,
  storeVariant,
  subscribeDrafts,
  takePlanNotice,
  writeAttachments,
  writeDraft,
  writeFailed,
  writePending,
  writePlan,
} from './chat/drafts';
import {
  compactEndWords,
  compactOff,
  forkOff,
  historyActions,
  recapEndToast,
  recapOff,
  rewindGoalOff,
  rewindOff,
  type HistoryFacts,
} from './chat/history-actions';
import { ListPane } from './chat/list-pane';
import { NewScreen } from './chat/new-screen';
import { heldSentence, toastSteerRefusal, useQueueControls } from './chat/queue-actions';
import { slashDraft, slashHint, slashListClosed, slashRows } from './chat/slash';
import { ActivityToggle, ModeSwitch, NewButton, ThreadBar, ThreadMenu } from './chat/thread-bar';
import { useChatHost } from './chat/use-chat-host';
import { useContextPanel } from './chat/use-context-panel';
import { forgetConversation, useConversation } from './chat/use-conversation';
import { lastRewindPoint, useRewind } from './chat/use-rewind';
import { useUnload } from './chat/use-unload';
import { keepMcpDrafts, McpCard } from './chat/work/elicit';
import { NewFolderChip, OpenFolderChip } from './chat/work/folder-chip';
import type { NeverRulesControl } from './chat/work/never';
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
import { keepQuestionDrafts, QuestionCard } from './chat/work/question';
import { useWorkGit } from './chat/work/use-work-git';

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
  files: readonly DraftAttachment[];
  /** The files as they are sent, each PDF by its way. */
  inputs: readonly AttachmentInput[];
  /** Entries of the conversation when it was sent: the stream has it once there are more. */
  base: number;
  lastEnd: ConversationState['lastEnd'];
  ts: number;
  /** A Work create: the mode the chip showed when Send was pressed, and the draft and folder the create named. */
  /** Work from New: the mode shown, the draft, and its place (null: no folder). */
  permission: { shown: PermissionMode; draft: string; cwd: string | null } | null;
}

/** The last message sent: Send again sends it with its files. */
interface Resend {
  place: string;
  text: string;
  files: readonly DraftAttachment[];
}

/** A failed message comes back only to an empty composer: no text, no files, and no file still being read. */
function restorable(place: string, reading: ReadonlyMap<string, number>): boolean {
  return readDraft(place) === '' && readAttachments(place).length === 0 && !reading.has(place);
}

/**
 * The user acted on what a failed send's strip asked (a folder, a model, an update): the strip goes. A message the
 * composer holds needs no record; one it does not keeps a Retry strip.
 */
function settleFailed(place: string): void {
  const record = readFailed(place);
  if (record === null || record.strip === null || record.strip.action?.kind === 'retry') return;
  const [text, label] = CHAT.sendErrors.other;
  writeFailed(
    place,
    record.restored ? null : { ...record, strip: { tone: 'danger', text, action: { label, kind: 'retry' } } },
  );
}

/** The host's stages of the PDF tool: the pages of a PDF, one call each, in order. */
const PDF_STAGE: PdfStageSink = {
  page: async (stage, page, pages, data) =>
    (await mutate<{ stage: string }>('POST', '/chat/pdf-stage', { stage, page, pages, data })).stage,
  drop: (stage) => {
    mutate('DELETE', `/chat/pdf-stage/${encodeURIComponent(stage)}`).catch(() => undefined);
  },
};

/** The stages {@link dropSentStages} dropped in this window. */
const DROPPED_STAGES = new Set<string>();

/** A sent PDF whose pages went in the message no longer needs the stage that also holds them. */
function dropSentStages(files: readonly DraftAttachment[], inputs: readonly AttachmentInput[]): void {
  for (const stage of sentStages(files, inputs)) {
    DROPPED_STAGES.add(stage);
    PDF_STAGE.drop(stage);
  }
}

const NO_RAISED: ChatHostState['permissionRaised'] = [];
const NO_RUN_NAMES: ReadonlyMap<string, string> = new Map();
const STARTING: TurnState = { turnId: '', phase: 'starting', startedAt: 0, wait: null };

const NO_ACTIVITY: Activity = { needs: [], needsRuns: [], plan: null, running: [], scheduled: [] };
/** The kinds of card that block a permission-mode change, as the host's `cardOpen()` counts them. */
const MODE_BLOCKING_CARDS: ReadonlySet<Interaction['kind']> = new Set(['permission', 'plan']);

/** `auto`: the wide rail opens by itself while the conversation has activity. `off`: only the toggle opens it. */
type RailPref = 'auto' | 'off';
const RAIL_PREF_KEY = 'mlx.chat.activityRail';

function readRailPref(): RailPref {
  try {
    return localStorage.getItem(RAIL_PREF_KEY) === 'off' ? 'off' : 'auto';
  } catch {
    return 'auto';
  }
}

function writeRailPref(pref: RailPref): RailPref {
  try {
    localStorage.setItem(RAIL_PREF_KEY, pref);
  } catch {
    // A per-viewer convenience: the choice holds for this page only.
  }
  return pref;
}
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
  const [params, setParams] = useSearchParams();
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
  const [folderChoice, setFolderChoice] = useState<FolderPick | null>(() => chosenFolder());
  // The place of the newest Work conversation is preselected; the user's own choice in this window wins.
  const workFailed = workList.error !== undefined;
  const newPick = useMemo(
    () => newPlace(folderChoice, { rows: workList.data?.conversations, failed: workFailed }),
    [folderChoice, workList.data, workFailed],
  );
  /** The New Work folder; null: no folder, or no place yet. */
  const newFolder = pickPath(newPick);
  const unplaced = newPick.kind === 'loading' || newPick.kind === 'unset' ? newPick.kind : undefined;
  const home = useHomeDir();
  const scratchRoot = scratchShown(host?.scratchRoot, home);

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
  const failed = useSyncExternalStore(subscribeDrafts, () => readFailed(place));
  const held = useSyncExternalStore(subscribeDrafts, () => readAttachments(place));
  const drawing = useSyncExternalStore(subscribeDrafts, () => readPending(place));
  // A read outlives the Chat mount that counted it, so a pending PDF in the store holds Send on its own.
  const drawingHere = useSyncExternalStore(subscribeDrafts, () => drawingCount(place));
  /** Why the last files picked, pasted or dropped on a place were not added. */
  const [attachNote, setAttachNote] = useState<{ place: string; text: string } | null>(null);
  /** Files still being read, per place. */
  const [reading, setReading] = useState<ReadonlyMap<string, number>>(() => new Map());
  /** The same counts for a send or a turn that ends between renders. */
  const readingNow = useRef(new Map<string, number>());
  /** The last message sent: Send again sends its files with it while this window holds them. */
  const [lastSent, setLastSent] = useState<Resend | null>(null);
  if (shownPlace !== place) {
    setShownPlace(place);
    setDraftState(readDraft(place));
    setStripError(null);
    // The created conversation keeps the New screen's thread cell: the bubble stays where it was drawn.
    if (shownPlace !== NEW_PLACE || createdId !== place) {
      newVisits += 1;
      setThreadKey(place === NEW_PLACE ? `${NEW_PLACE}:${newVisits}` : place);
      if (createdId !== null) setCreatedId(null);
    }
  }
  // The New Work screen's draft: a mode picked there is kept by the Control Panel for this id and folder only.
  const [modeDraft, setModeDraft] = useState(() =>
    firstModeDraft({ place: newPick, visit: threadKey, mode }, crypto.randomUUID()),
  );
  const modeDraftNow = nextModeDraft(modeDraft, { place: newPick, visit: threadKey, mode }, () => crypto.randomUUID());
  if (modeDraftNow !== modeDraft) setModeDraft(modeDraftNow);
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
  // The copy a fork of this page made: its opening says it is a fork.
  const forkOpened = useRef<string | null>(null);

  const [dismissedEnd, setDismissedEnd] = useState<ConversationState['lastEnd']>(null);
  /** The conversation a fork press of this window runs for, until its route answered. */
  const [forking, setForking] = useState<string | null>(null);

  // The bubble leaves once the stream carries the message, or the turn ended without it. A turn that failed before
  // it ran never carried the message: its text and files come back for Retry.
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
        const id = sending.conversationId;
        const restored = restorable(id, readingNow.current);
        if (restored) {
          writeDraft(id, sending.text);
          writeAttachments(id, sending.files);
          setDraftState(sending.text);
        }
        writeFailed(id, { text: sending.text, files: sending.files, strip: null, restored });
      }
    }
  }, [sending, conv, conversationId, gone]);

  const sendingHere =
    sending !== null && (sending.conversationId === null ? isNew : sending.conversationId === conversationId);
  // The strip under the composer on New, until the first send: the folder, and in Work its branch.
  const trayShown = isNew && !sendingHere;
  const hostActive = host?.active;
  const workingWork = useMemo(
    () =>
      (hostActive ?? [])
        .filter((a) => a.mode === 'work' && (a.busy || a.background || a.tasks > 0 || a.op !== null))
        .map((a) => a.conversationId)
        .sort()
        .join('\n'),
    [hostActive],
  );
  const workGit = useWorkGit(trayShown && mode === 'work' ? newFolder : null, workingWork);
  const [branchOpen, setBranchOpen] = useState(false);
  const [branchAlert, setBranchAlert] = useState<{ lead: string; text: string } | null>(null);
  const [branchFor, setBranchFor] = useState(newFolder);
  if (branchFor !== newFolder) {
    setBranchFor(newFolder);
    setBranchOpen(false);
    setBranchAlert(null);
  }
  const turn = conv?.turn ?? null;
  const compaction = conv?.compaction ?? null;
  // A compaction with no turn is a manual run (Compact now): Stop and Esc end it as they end an answer.
  const compacting = turn === null && compaction !== null;
  const running = turn !== null || compacting;
  // The history op or recap of this conversation that runs now: this window's fork press until the host lists it, the
  // host's op (any window's), or the recap the stream carries.
  const opHere: HistoryFacts['op'] = isNew
    ? null
    : compacting
      ? 'compact'
      : forking !== null && forking === conversationId
        ? 'fork'
        : (host?.active.find((entry) => entry.conversationId === conversationId)?.op ??
          (conv?.recap != null ? 'recap' : null));
  const phase: ComposerPhase =
    turn !== null
      ? turn.phase
      : compacting
        ? compaction.phase === 'stopping'
          ? 'cancelling'
          : 'running'
        : sendingHere
          ? 'creating'
          : opHere !== null
            ? 'op'
            : 'idle';
  const busy = phase !== 'idle';
  // Work: the goal the agent works on now; between two of its rounds no answer runs, and the queue waits for it.
  const goalRunning =
    !isNew &&
    mode === 'work' &&
    conv !== null &&
    conv.conversation.id === conversationId &&
    goalOf(conv.entries, { hot: conv.conversation.hot === true, waiting: false })?.running === true;
  // Work: an answer of this conversation runs past its start, or its goal runs between rounds, so Send now and ⌘Enter
  // can put text into it.
  const steerShown =
    mode === 'work' &&
    ((turn !== null && (turn.phase === 'starting' || turn.phase === 'running')) || (turn === null && goalRunning));
  const steerOpen = steerShown && (turn === null || turn.phase === 'running');

  // Work: the "Plan first" value of this place. It follows the agent's mode lines, a press holds until the next one,
  // and it stays live while an answer runs: it only sets what the next message carries.
  const storedPlan = useSyncExternalStore(subscribeDrafts, () => readPlan(place));
  const loadedHere = !isNew && conv !== null && conv.conversation.id === conversationId;
  // Until the stream brings a Work conversation its plan mode is unknown: a message would carry `plan: false`, and that
  // ends plan mode.
  const modeUnknown = !isNew && !loadedHere && (mode === 'work' || rowOf(conversationId) === undefined);
  const planEntries = loadedHere ? conv.entries : null;
  const plan = useMemo(() => planChip(planEntries ?? [], storedPlan), [planEntries, storedPlan]);
  const togglePlan = useCallback(() => writePlan(place, { next: !plan.next, seen: plan.seen }), [place, plan]);
  // The queued messages of this conversation: the host holds them, the stream brings each change.
  const queued = loadedHere ? conv.queue : NO_QUEUE;
  // A saved edit whose row had left the queue: its text goes to that conversation's box when the box is empty.
  const keepEdit = useCallback((id: string, text: string): boolean => {
    if (!restorable(id, readingNow.current)) return false;
    writeDraft(id, text);
    if (placeRef.current === id) setDraftState(text);
    return true;
  }, []);
  const queueControls = useQueueControls(isNew ? null : conversationId, queued, announce, keepEdit);
  const heldBefore = useRef({ id: conversationId, why: queued.held?.why ?? null });
  useEffect(() => {
    const before = heldBefore.current;
    heldBefore.current = { id: conversationId, why: queued.held?.why ?? null };
    if (before.id === conversationId && before.why === null && queued.held?.why === 'stopped') {
      announce(CHAT.queue.status.held);
    }
  }, [conversationId, queued.held, announce]);
  const noPlanMode = loadedHere && conv.notices.some((notice) => notice.code === 'no-plan-mode');
  useEffect(() => {
    if (!loadedHere || conversationId === null) return;
    if (takePlanNotice(conversationId, noPlanMode)) writePlan(conversationId, { next: false, seen: plan.seen });
  }, [loadedHere, conversationId, noPlanMode, plan.seen]);
  const modeSaid = useRef({ id: conversationId, loaded: loadedHere, key: plan.seen });
  useEffect(() => {
    const said = modeSaid.current;
    modeSaid.current = { id: conversationId, loaded: loadedHere, key: plan.seen };
    if (!said.loaded || said.id !== conversationId || plan.seen === said.key || plan.mode === null) return;
    announce(planModeStatus(plan.mode));
  }, [conversationId, loadedHere, plan.seen, plan.mode, announce]);

  // Models.
  const unload = useUnload(host);
  const unloadControl = unload.control;
  const resident = useMemo(() => new Set(unload.resident.map((entry) => entry.modelId)), [unload.resident]);
  const [newModel, setNewModel] = useState<string | null>(null);
  const [newLevel, setNewLevel] = useState<{ mode: ChatMode; level: ThinkingLevel } | null>(null);
  const [modelWorking, setModelWorking] = useState(false);
  const [picked, setPicked] = useState(pickedVariants);
  const pickerRef = useRef<HTMLSpanElement>(null);
  const startsOn = startModel(host, storedModel(mode));
  const chosenNew =
    newModel !== null && host?.models.some((model) => model.id === newModel) === true ? newModel : startsOn;
  const modelValue = isNew ? chosenNew : (conv?.conversation.modelId ?? null);
  const modelRow = host?.models.find((model) => model.id === modelValue);
  // Thinking: the picker's levels column shows only when an agent that sends levels listed the model (null: none).
  const modelLevels = modelRow?.thinking;
  const levelPicked = isNew
    ? newLevel?.mode === mode
      ? newLevel.level
      : storedThinking(mode)
    : conv?.conversation.thinking;
  const levelShown = modelLevels == null ? undefined : stepLevel(levelPicked, modelLevels);
  const seesImages = mode === 'chat' && modelRow?.images === true;
  // PDF pages are images too, but they go to an image model in Work as well.
  const seesPages = modelRow?.images === true;
  const windowTokens = modelRow?.contextTokens ?? null;
  const family = modelRow?.family ?? null;
  const pdfTool = host?.pdfTool[mode] ?? false;
  const bases = useMemo(
    () => (host === null ? [] : modelBases(host, resident, mode, modelValue, picked, { level: levelPicked, busy })),
    [host, resident, mode, modelValue, picked, levelPicked, busy],
  );
  const residentId = host?.models.find((model) => resident.has(model.id))?.id ?? null;
  const modelSize = useCallback(
    (id: string): number | null =>
      host?.models.find((model) => model.id === id)?.sizeBytes ??
      host?.resident.find((entry) => entry.modelId === id)?.sizeBytes ??
      null,
    [host],
  );
  const otherRunning =
    host?.active.some((entry) => (entry.busy || entry.background) && entry.conversationId !== conversationId) === true;
  // A live workflow run or subagent holds the conversation's model and folder: no model change, no delete until it
  // ends. A command or monitor holds only the agent process: no delete.
  const liveHere = isNew ? undefined : host?.active.find((entry) => entry.conversationId === conversationId);
  const runsLive = conv?.runs.some((run) => run.live) === true;
  const runLive = !isNew && (runsLive || liveHere?.background === true);
  const helperLive = runLive && !runsLive && liveHere?.helpers === true;
  const deleteOff = busy
    ? CHAT.deleteBusy
    : runLive
      ? helperLive
        ? CHAT.deleteBusyHelper
        : CHAT.deleteBusyRun
      : (liveHere?.tasks ?? 0) > 0
        ? CHAT.deleteBusyTask
        : null;
  const engineLostStrip = engineLostNotice(host, mode, isNew ? null : conversationId);
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
  const contextUsed = conv?.contextUsed ?? null;
  const contextWindow =
    conv === null
      ? null
      : (host?.models.find((model) => model.id === conv.conversation.modelId)?.contextTokens ?? null);
  const context: ContextUse | undefined =
    contextUsed === null || contextWindow === null ? undefined : { used: contextUsed, window: contextWindow };
  const threadUnload = useMemo(
    () => (threadModel === null ? undefined : unloadControl(threadModel)),
    [threadModel, unloadControl],
  );

  const levelName = (level: ThinkingLevel): string => CHAT.thinkingLevel.levels[level];
  /** The New screen keeps the level per mode; a conversation PATCHes it (taken while an answer runs too). */
  const saveLevel = async (level: ThinkingLevel): Promise<void> => {
    if (isNew) {
      storeThinking(mode, level);
      setNewLevel({ mode, level });
      return;
    }
    if (conversationId === null) return;
    try {
      await mutate<ConversationInfo>('PATCH', conversationPath(conversationId), { thinking: level });
    } catch {
      toast.error(CHAT.thinkingLevel.failed);
    }
  };
  const changeLevel = async (level: ThinkingLevel): Promise<void> => {
    announce(fill(CHAT.thinkingLevel.status, { level: levelName(level) }));
    await saveLevel(level);
  };

  /**
   * A model, with the level picked beside it. The host steps the conversation's level on a model change, so the picked
   * level goes after the model, and only when it is not where the step lands.
   */
  const changeModel = async (id: string, level?: ThinkingLevel): Promise<void> => {
    const nextLevels = host?.models.find((model) => model.id === id)?.thinking;
    const stepped = nextLevels == null ? undefined : stepLevel(levelPicked, nextLevels);
    const lands = level ?? stepped;
    if (lands !== undefined && levelShown !== undefined && lands !== levelShown) {
      announce(fill(CHAT.thinkingLevel.status, { level: levelName(lands) }));
    }
    const setLevel = level !== undefined && level !== stepped ? level : undefined;
    storeModel(mode, id);
    const base = host?.models.find((model) => model.id === id)?.base;
    if (base !== undefined) {
      storeVariant(base, id);
      setPicked((current) => ({ ...current, [base.toLowerCase()]: id }));
    }
    if (isNew) {
      setNewModel(id);
      if (setLevel !== undefined) await saveLevel(setLevel);
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
      setModelWorking(false);
      return;
    }
    try {
      if (setLevel !== undefined) await saveLevel(setLevel);
    } finally {
      setModelWorking(false);
    }
  };

  /** The picker's pick: a level on the conversation's model, or a model with its level. */
  const pickModel = (id: string, value: string | undefined): void => {
    const level = host?.models.find((model) => model.id === id)?.thinking?.levels.find((option) => option === value);
    if (id !== modelValue) void changeModel(id, level);
    else if (level !== undefined) void changeLevel(level);
  };

  // Why Send is off.
  const canInstallAgent = host?.agent.canInstall !== false;
  const blocker = modeBlocker(connected, host, mode);
  const engineReason =
    host?.inference === 'stopped' ? CHAT.blockers['inference-stopped'][1] : (blocker?.reason ?? undefined);
  // Compact now: the menu row, the palette and `/compact` call the same action; its end comes on the stream.
  const compactReason =
    isNew || conversationId === null
      ? null
      : compactOff({
          blocked: engineReason ?? (cwdProblem === undefined ? null : CHAT.cwdProblems[cwdProblem]),
          busy,
          runLive,
          helperLive,
          engineLost: engineLostStrip !== null,
          finished: conv?.turns.length ?? 0,
          op: opHere,
        });
  // Rewind to here: the message menu, `/rewind` and the palette; off while the conversation works, as on the host.
  const rewindEntries = isNew ? null : (conv?.entries ?? null);
  const rewindEntriesRef = useRef(rewindEntries);
  useLayoutEffect(() => {
    rewindEntriesRef.current = rewindEntries;
  });
  const rewindLast = useMemo(() => (rewindEntries === null ? null : lastRewindPoint(rewindEntries)), [rewindEntries]);
  // The row of the goal that is still open: a rewind to a message before it would leave the goal with no strip.
  const goalAt = useMemo(() => (rewindEntries === null ? null : openGoalAt(rewindEntries)), [rewindEntries]);
  const rewindAt = useCallback((at: number) => rewindGoalOff(at, goalAt), [goalAt]);
  const rewindReason =
    isNew || conversationId === null
      ? null
      : rewindOff({
          blocked: engineReason ?? (cwdProblem === undefined ? null : CHAT.cwdProblems[cwdProblem]),
          busy,
          runLive,
          helperLive,
          engineLost: engineLostStrip !== null,
          finished: conv?.turns.length ?? 0,
          op: opHere,
        });
  // Fork: the ⋯ row, a message's "Fork with this answer", `/fork` and the palette; off as a rewind is, as on the host.
  const forkReason =
    isNew || conversationId === null
      ? null
      : forkOff({
          blocked: engineReason ?? (cwdProblem === undefined ? null : CHAT.cwdProblems[cwdProblem]),
          busy,
          runLive,
          helperLive,
          engineLost: engineLostStrip !== null,
          finished: conv?.turns.length ?? 0,
          op: opHere,
        });
  // Recap: the ⋯ row, `/recap` and the palette; a workflow run or a helper does not turn it off (it only reads).
  const recapWait = isNew ? null : (conv?.recap ?? null);
  const recapReason =
    isNew || conversationId === null
      ? null
      : recapOff({
          blocked: engineReason ?? (cwdProblem === undefined ? null : CHAT.cwdProblems[cwdProblem]),
          busy,
          runLive,
          helperLive,
          engineLost: engineLostStrip !== null,
          finished: conv?.turns.length ?? 0,
          op: opHere,
        });
  // The `/` list and the typed command: app commands run here, agent commands (Work) are sent as typed.
  const commandList = isNew ? null : (conv?.commands ?? null);
  const slashList = useMemo(() => slashRows(mode, commandList), [mode, commandList]);
  const slash = slashDraft(draft, mode, commandList);
  const appOff =
    slash.kind !== 'app' || slash.op === 'plan'
      ? null
      : slash.op === 'context'
        ? isNew || conversationId === null
          ? CHAT.history.contextCold
          : null
        : slash.op === 'rewind'
          ? isNew || conversationId === null || rewindLast === null
            ? CHAT.history.rewindEmpty
            : (rewindReason ?? rewindAt(rewindLast))
          : slash.op === 'fork'
            ? isNew || conversationId === null
              ? CHAT.history.forkEmpty
              : forkReason
            : slash.op === 'recap'
              ? isNew || conversationId === null
                ? CHAT.history.recapEmpty
                : recapReason
              : isNew || conversationId === null
                ? CHAT.history.compactEmpty
                : compactReason;
  let blocked = blocker?.reason;
  if (blocked === undefined && isNew && mode === 'work') blocked = placeBlock(newPick);
  if (blocked === undefined && isNew && mode === 'work' && newFolder !== null && folderTooLong(newFolder))
    blocked = CHAT.folderTooLong;
  if (blocked === undefined && isNew && mode === 'work' && workGit.switching !== null) blocked = CHAT.branch.switching;
  if (blocked === undefined && cwdProblem !== undefined) blocked = CHAT.cwdProblems[cwdProblem];
  if (blocked === undefined && modeUnknown) blocked = CHAT.threadLoading;
  if (blocked === undefined && slash.kind === 'unknown') blocked = CHAT.slash.unknown;
  if (blocked === undefined && appOff !== null) blocked = appOff;
  const readingHere = (reading.get(place) ?? 0) + drawingHere;
  const threadEntries = isNew ? undefined : conv?.entries;
  const heldPictures = useMemo(
    () => (threadEntries === undefined ? 0 : heldPictureChars(threadEntries)),
    [threadEntries],
  );
  const heldPages = useMemo(() => (threadEntries === undefined ? 0 : heldImages(threadEntries)), [threadEntries]);
  const attachFacts = {
    mode,
    seesImages,
    seesPages,
    windowTokens,
    family,
    contextUsed,
    heldPictures,
    heldImages: heldPages,
    pdfTool,
  };
  if (blocked === undefined)
    blocked = attachProblem({ ...attachFacts, text: draft, files: held, reading: readingHere }) ?? undefined;
  const queueFull = (phase === 'starting' || phase === 'running' || phase === 'op') && queued.count >= QUEUE_MAX;
  // ⌘Enter puts text alone into the running answer: no place in the list, so a full list leaves it on.
  const steerDraft = steerOpen && held.length === 0 && !isSlashText(draft);
  const slashMissed = blocked === CHAT.slash.unknown && slashListClosed(draft, slashList.rows);
  useEffect(() => {
    if (slashMissed) announce(CHAT.slash.unknown);
  }, [slashMissed, announce]);
  const panelShown = isNew && blocker?.panel !== undefined;

  const composerRef = useRef<HTMLTextAreaElement | null>(null);
  const focusComposer = useCallback(() => {
    requestAnimationFrame(() => composerRef.current?.focus({ preventScroll: true }));
  }, []);
  /** A message back in the box of its own conversation; that conversation opens when the page shows another. */
  const putBackTo = useCallback(
    (id: string, text: string): void => {
      writeDraft(id, text);
      if (placeRef.current === id) setDraftState(text);
      else void navigate(chatUrl(id));
    },
    [navigate],
  );
  const rewind = useRewind({
    conversationId: isNew ? null : conversationId,
    mode,
    hot: conv?.conversation.hot === true,
    folder: openFolder === null ? null : folderName(openFolder),
    entries: rewindEntries ?? [],
    end: isNew ? null : (conv?.rewindEnd ?? null),
    off: rewindReason,
    commands: host?.modes[mode].tasks ?? 0,
    readDraft,
    putBack: putBackTo,
    keepText: writeDraft,
    openConversation: (id) => void navigate(chatUrl(id)),
    titleOf: (id) => {
      const row = rowOf(id);
      return chatTitle(row?.title ?? '', row?.mode ?? mode);
    },
    composer: () => composerRef.current,
    focusComposer,
    announce,
  });
  const rewindOnMessage = rewind.onRewind;
  // A hook blocked a message: it left the thread, and its text and files come back to an empty box, else a toast holds
  // them. Only for an end this page saw come, never one the conversation opened with, unless this page's first message
  // made the conversation (a fast hook ends that answer before the page holds it).
  const blockSeen = useRef<{ id: string | null; end: ConversationState['lastEnd'] }>({ id: null, end: null });
  const shownEnd = loadedHere ? conv.lastEnd : null;
  const sendingTo = sending?.conversationId ?? null;
  useEffect(() => {
    const seen = blockSeen.current;
    const id = loadedHere ? conversationId : null;
    blockSeen.current = { id, end: shownEnd };
    if (id === null || (seen.id !== id && sendingTo !== id) || shownEnd === null || shownEnd === seen.end) return;
    const text = shownEnd.blocked;
    if (text === undefined) return;
    // The send still in flight, else the last one this window sent there, holds the message's files.
    const files = resendFiles(
      sending !== null && sending.conversationId === id && sending.text === text
        ? sending.files
        : lastSent !== null && lastSent.place === id && lastSent.text === text
          ? lastSent.files
          : [],
      DROPPED_STAGES,
    );
    const putBack = (): void => {
      putBackTo(id, text);
      if (files.length === 0) return;
      const kept = readAttachments(id);
      const added = files.filter((file) => !kept.some((item) => item.id === file.id));
      writeAttachments(id, [...kept, ...added].slice(0, ATTACH_MAX));
    };
    if (readDraft(id).trim() === '') {
      putBack();
      focusComposer();
      announce(CHAT.hook.promptBack);
      return;
    }
    toast.info(CHAT.hook.promptKept, {
      duration: Number.POSITIVE_INFINITY,
      action: {
        label: CHAT.history.putBack,
        onClick: (toastId) => {
          putBack();
          focusComposer();
          toast.dismiss(toastId);
        },
      },
    });
  }, [loadedHere, conversationId, shownEnd, sendingTo, sending, lastSent, putBackTo, focusComposer, announce]);
  /**
   * Fork the conversation shown: whole (`entry` null) or up to and with the answer of the message at `entry`, named by
   * its time too: the host refuses it when another entry is there now.
   */
  const forkNow = useCallback(
    (entry: number | null) => {
      if (isNew || conversationId === null) return;
      const message = entry === null ? undefined : rewindEntriesRef.current?.[entry];
      if (entry !== null && message === undefined) return;
      const from = conversationId;
      setForking(from);
      void historyActions
        .fork(from, entry === null || message === undefined ? null : { entry, ts: message.ts }, {
          folder: openFolder,
          here: () => placeRef.current === from,
          open: (copy) => {
            forkOpened.current = copy;
            void navigate(chatUrl(copy));
          },
          back: () => void navigate(chatUrl(from)),
        })
        .finally(() => setForking((current) => (current === from ? null : current)));
    },
    [isNew, conversationId, openFolder, navigate],
  );
  const threadRewind = useMemo(
    () => (isNew ? undefined : { off: rewindReason, offAt: rewindAt, onRewind: rewindOnMessage, onFork: forkNow }),
    [isNew, rewindReason, rewindAt, rewindOnMessage, forkNow],
  );
  // The "+" menu's Commands row: a `/` before the text, the caret after it, so the list opens.
  const openCommands = useCallback(() => {
    const text = readDraft(placeRef.current);
    setDraft(text.trim() === '' ? '/' : `/ ${text}`);
    requestAnimationFrame(() => {
      const box = composerRef.current;
      if (box === null) return;
      box.setSelectionRange(1, 1);
      box.focus({ preventScroll: true });
    });
  }, [setDraft]);

  const root = useScrollRoot();
  const stick = useStickToBottom(root, threadKey);
  const threadJump = useRef<((at: number) => void) | null>(null);

  /** A typed app command runs its op as the menu row does; `/context` also while an answer runs. */
  const runAppCommand = (text: string, op: AppCommandName): void => {
    const takeCommand = (): void => {
      if (readDraft(place) === text) setDraft(slashRest(text));
    };
    if (op === 'plan') {
      takeCommand();
      togglePlan();
      focusComposer();
      return;
    }
    if (conversationId === null || isNew) return;
    if (op === 'context') {
      takeCommand();
      historyActions.showContext(conversationId);
      return;
    }
    if (op === 'rewind') {
      if (rewindReason !== null || rewindLast === null || rewindAt(rewindLast) !== null) return;
      takeCommand();
      historyActions.rewindLast(conversationId);
      return;
    }
    if (op === 'fork') {
      if (forkReason !== null) return;
      takeCommand();
      forkNow(null);
      return;
    }
    if (op === 'recap') {
      if (recapReason !== null) return;
      takeCommand();
      void historyActions.recap(conversationId);
      focusComposer();
      return;
    }
    if (op !== 'compact' || compactReason !== null) return;
    takeCommand();
    void historyActions.compact(conversationId);
    focusComposer();
  };

  /** Files still being read count only for a send from the composer: Retry and Send again send finished files. */
  const send = async (
    text: string,
    files: readonly DraftAttachment[] = [],
    from: 'composer' | 'resend' = 'resend',
  ): Promise<void> => {
    if (text.trim() === '' || phase !== 'idle') return;
    const typed = slashDraft(text, mode, commandList);
    if (typed.kind === 'unknown') return;
    if (typed.kind === 'app') {
      runAppCommand(text, typed.op);
      return;
    }
    if (modeBlocker(connected, host, mode) !== null) return;
    if (isNew && mode === 'work' && placeBlock(newPick) !== undefined) return;
    if (isNew && mode === 'work' && newFolder !== null && folderTooLong(newFolder)) return;
    if (isNew && mode === 'work' && workGit.switching !== null) return;
    if (!isNew && cwdProblem !== undefined) return;
    if (modeUnknown) return;
    const problem = attachProblem({ ...attachFacts, text, files, reading: from === 'composer' ? readingHere : 0 });
    if (problem !== null) {
      setStripError({ tone: 'warning', text: problem });
      return;
    }
    setStripError(null);
    setAttachNote(null);
    setDismissedEnd(conv?.lastEnd ?? null);
    const fromNew = isNew;
    const target = fromNew ? null : conversationId;
    const origin = place;
    const epoch = placeEpoch(origin);
    // Retry takes the failed message; another send leaves it until this one goes through, unless the composer has it.
    const prior = readFailed(origin);
    if (prior !== null && (prior.restored || (prior.text === text && prior.files === files))) {
      writeFailed(origin, null, epoch);
    }
    const sent = (): void => {
      if (prior !== null && readFailed(origin) === prior) writeFailed(origin, null, epoch);
    };
    const base = fromNew ? 0 : (conv?.entries.length ?? 0);
    const inputs = messageInputs(files, attachPlan({ ...attachFacts, text, files }));
    const next: Sending = {
      conversationId: target,
      mode,
      text,
      files,
      inputs,
      base,
      lastEnd: fromNew ? null : (conv?.lastEnd ?? null),
      ts: Date.now(),
      permission: fromNew && mode === 'work' ? { shown: permissionMode, draft: modeDraft.id, cwd: newFolder } : null,
    };
    setSending(next);
    // A Retry of a failed message leaves text typed and files attached since then in the composer.
    if (readDraft(origin) === text) setDraft('');
    const stored = readAttachments(origin);
    if (files.length > 0 && stored.length === files.length && stored.every((item, i) => item.id === files[i]!.id)) {
      writeAttachments(origin, []);
    }
    const attachments: AttachmentInput[] | undefined = files.length > 0 ? inputs : undefined;
    // Work: `false` makes the agent leave plan mode, so a message always carries what the chip shows.
    const planSent = mode === 'work' ? plan.next : undefined;
    stick.pin();
    announce(CHAT.status.sent);
    // The reply belongs to the place it was sent from; the user may be elsewhere by the time it arrives.
    const stillShown = (): boolean => placeRef.current === origin;
    const drop = (): void => setSending((current) => (current === next ? null : current));
    try {
      if (fromNew) {
        const body: {
          mode: ChatMode;
          text: string;
          modelId?: string;
          cwd?: string;
          noFolder?: true;
          draft?: string;
          attachments?: AttachmentInput[];
          plan?: boolean;
          thinking?: ThinkingLevel;
        } = { mode, text };
        if (chosenNew !== null) body.modelId = chosenNew;
        if (levelShown !== undefined && levelPicked !== undefined) body.thinking = levelShown;
        if (attachments !== undefined) body.attachments = attachments;
        if (planSent !== undefined) body.plan = planSent;
        if (next.permission !== null) {
          if (next.permission.cwd === null) body.noFolder = true;
          else body.cwd = next.permission.cwd;
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
        sent();
        dropSentStages(files, inputs);
        writePlan(origin, null);
        setLastSent({ place: id, text, files });
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
        await mutate('POST', `${conversationPath(target)}/prompt`, {
          text,
          ...(attachments === undefined ? {} : { attachments }),
          ...(planSent === undefined ? {} : { plan: planSent }),
        });
        sent();
        dropSentStages(files, inputs);
        setLastSent({ place: target, text, files });
      }
    } catch (err) {
      drop();
      // Text typed or files attached while the send was pending stay; the place keeps the failed message for Retry.
      const restored = restorable(origin, readingNow.current);
      if (restored) {
        writeDraft(origin, text);
        writeAttachments(origin, files, epoch);
        if (stillShown()) setDraftState(text);
      }
      const reason = err instanceof DashboardApiError ? err.reason : undefined;
      const code = err instanceof DashboardApiError ? err.code : undefined;
      const outcome = sendFailure(
        reason,
        code,
        mode,
        err instanceof Error ? err.message : undefined,
        undefined,
        canInstallAgent,
        scratchRoot,
      );
      if ('toast' in outcome && restored) {
        toast.error(outcome.toast);
        return;
      }
      const failure: StripNotice =
        'toast' in outcome
          ? { tone: 'danger', text: outcome.toast, action: { label: CHAT.tryAgain, kind: 'retry' } }
          : outcome.strip;
      writeFailed(origin, { text, files, strip: failure, restored }, epoch);
      if (!stillShown()) toast.error(failure.text);
    }
  };

  /**
   * Enter while an answer starts or runs: the message joins the host's queue and the box empties. It runs when the
   * answer ends, with the files and the Plan first value it has now. A refusal puts the text back into an empty box;
   * with a newer draft there, the place keeps the refused message for Try again, which queues it while the answer
   * runs. `now` (Work's ⌘Enter): text alone, while the answer runs, goes into it; else it queues as Enter does.
   */
  const queueMessage = async (
    text: string,
    files: readonly DraftAttachment[],
    now = false,
    from: 'composer' | 'resend' = 'composer',
  ): Promise<void> => {
    if (isNew || conversationId === null || text.trim() === '') return;
    // App commands are never queued: one whose op is on while the agent is busy (`/context`) runs now.
    const typed = slashDraft(text, mode, commandList);
    if (typed.kind === 'app') {
      runAppCommand(text, typed.op);
      return;
    }
    if (typed.kind === 'unknown') return;
    if (phase !== 'starting' && phase !== 'running' && phase !== 'op') return;
    if (modeBlocker(connected, host, mode) !== null || cwdProblem !== undefined || modeUnknown) return;
    const whenBusy = now && steerOpen && files.length === 0 && !isSlashText(text) ? 'now' : 'queue';
    if (whenBusy === 'queue' && queued.count >= QUEUE_MAX) return;
    const reading = from === 'composer' ? readingHere : 0;
    const problem = attachProblem({ ...attachFacts, text, files, reading });
    if (problem !== null) {
      setStripError({ tone: 'warning', text: problem });
      return;
    }
    setStripError(null);
    setAttachNote(null);
    const target = conversationId;
    const origin = place;
    const epoch = placeEpoch(origin);
    const prior = readFailed(origin);
    if (prior !== null && (prior.restored || (prior.text === text && prior.files === files))) {
      writeFailed(origin, null, epoch);
    }
    if (readDraft(origin) === text) setDraft('');
    const stored = readAttachments(origin);
    if (files.length > 0 && stored.length === files.length && stored.every((item, i) => item.id === files[i]!.id)) {
      writeAttachments(origin, []);
    }
    const inputs = messageInputs(files, attachPlan({ ...attachFacts, text, files }));
    const attachments: AttachmentInput[] | undefined = files.length > 0 ? inputs : undefined;
    const planSent = mode === 'work' ? plan.next : undefined;
    try {
      const reply = await mutate<
        { turnId: string } | { queued: string; steer?: 'refused' | 'late' } | { steered: true }
      >('POST', `${conversationPath(target)}/prompt`, {
        text,
        ...(attachments === undefined ? {} : { attachments }),
        ...(planSent === undefined ? {} : { plan: planSent }),
        whenBusy,
      });
      dropSentStages(files, inputs);
      if ('steered' in reply) announce(CHAT.queue.status.sent);
      else if ('queued' in reply) {
        announce(fill(CHAT.queue.status.queued, { n: queued.items.length + 1 }));
        if (reply.steer !== undefined) toastSteerRefusal(reply.steer === 'late' ? 'steer-late' : 'steer-busy');
      } else {
        // The answer ended first: the message ran as a new answer.
        announce(CHAT.status.sent);
        setLastSent({ place: target, text, files });
      }
    } catch (err) {
      const restored = restorable(origin, readingNow.current);
      if (restored) {
        writeDraft(origin, text);
        writeAttachments(origin, files, epoch);
        if (placeRef.current === origin) setDraftState(text);
      }
      const reason = err instanceof DashboardApiError ? err.reason : undefined;
      const code = err instanceof DashboardApiError ? err.code : undefined;
      const outcome = sendFailure(
        reason,
        code,
        mode,
        err instanceof Error ? err.message : undefined,
        undefined,
        canInstallAgent,
      );
      if (!restored) {
        const retry = { label: CHAT.tryAgain, kind: 'retry' } as const;
        const failure: StripNotice =
          'toast' in outcome
            ? { tone: 'danger', text: outcome.toast, action: retry }
            : { ...outcome.strip, action: outcome.strip.action ?? retry };
        writeFailed(origin, { text, files, strip: failure, restored: false }, epoch);
        if (placeRef.current !== origin) toast.error(failure.text);
        return;
      }
      if ('toast' in outcome) toast.error(outcome.toast);
      else if (placeRef.current === origin) setStripError({ ...outcome.strip, action: undefined });
      else toast.error(outcome.strip.text);
    }
  };

  const alive = useRef(true);
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  // A read keeps the place it started in: the user may be elsewhere, or Chat may be closed, by the time it ends.
  const addFiles = (files: readonly File[]): void => {
    if (files.length === 0) return;
    const origin = place;
    const epoch = placeEpoch(origin);
    const count = (step: number): void => {
      const now = readingNow.current;
      const n = (now.get(origin) ?? 0) + step;
      if (n === 0) now.delete(origin);
      else now.set(origin, n);
      setReading(new Map(now));
    };
    setAttachNote(null);
    count(1);
    // A PDF's chip shows from the start of its read; it leaves once its draft is in the store, or at once when its
    // read ends with nothing added.
    const started: string[] = [];
    const watch: PdfWatch = {
      start: (id, name) => {
        const stop = new AbortController();
        const now = Date.now();
        started.push(id);
        writePending(
          origin,
          id,
          { id, name, done: 0, total: null, bytes: 0, startedAt: now, at: now, thumbnail: null, stop },
          epoch,
        );
        return stop.signal;
      },
      progress: (id, progress) => {
        const item = readPending(origin).find((entry) => entry.id === id);
        if (item === undefined) return;
        const thumbnail = progress.thumbnail ?? item.thumbnail;
        writePending(
          origin,
          id,
          { ...item, done: progress.done, total: progress.total, bytes: progress.bytes, at: Date.now(), thumbnail },
          epoch,
        );
      },
      end: (id) => writePending(origin, id, null, epoch),
    };
    attachDrafts(
      files,
      { mode, seesImages, seesPages, watch, ...(pdfTool ? { stage: PDF_STAGE } : {}) },
      { read: () => readAttachments(origin), write: (items) => writeAttachments(origin, items, epoch) },
    )
      .then(
        (result) => {
          for (const item of result.added) {
            if (item.kind !== 'pdf') continue;
            const pages = item.sizes.length;
            announce(fill(pages === 1 ? CHAT.attach.pdfReadyOne : CHAT.attach.pdfReady, { name: item.name, n: pages }));
          }
          const first = result.refused[0];
          if (first === undefined) return;
          const text =
            result.refused.length > 1
              ? fill(CHAT.attach.more, { text: first, n: files.length - result.added.length - 1 })
              : first;
          if (alive.current && placeRef.current === origin) setAttachNote({ place: origin, text });
          else toast.error(text);
        },
        () => undefined,
      )
      .finally(() => {
        for (const id of started) writePending(origin, id, null, epoch);
        count(-1);
      });
  };
  const removeAttachment = (id: string): void => {
    if (!stopPending(place, id)) {
      const removed = readAttachments(place).find((item) => item.id === id);
      writeAttachments(
        place,
        readAttachments(place).filter((item) => item.id !== id),
      );
      if (removed?.kind === 'pdf' && removed.stage !== null) PDF_STAGE.drop(removed.stage);
    }
    setAttachNote(null);
  };
  const pickerInput = useRef<HTMLInputElement | null>(null);
  const bindPicker = useCallback(
    (node: HTMLInputElement | null) => {
      pickerInput.current = node;
      if (node === null) return;
      node.addEventListener('cancel', focusComposer);
      return () => {
        node.removeEventListener('cancel', focusComposer);
        pickerInput.current = null;
      };
    },
    [focusComposer],
  );
  const openPicker = useCallback(() => pickerInput.current?.click(), []);

  const picking = useRef(false);
  /** The native dialog; true when it named a folder, which is then the New Work folder. */
  const pickAndChoose = useCallback(async (): Promise<boolean> => {
    if (picking.current) return false;
    picking.current = true;
    let picked = false;
    try {
      const path = await pickFolder('work');
      if (path !== null) {
        picked = true;
        const pick: FolderPick = { kind: 'folder', path };
        chooseFolder(pick);
        setFolderChoice(pick);
        setStripError(null);
        settleFailed(placeRef.current);
      }
    } finally {
      picking.current = false;
    }
    focusComposer();
    return picked;
  }, [focusComposer]);
  const choosePlace = useCallback(
    (pick: FolderPick) => {
      chooseFolder(pick);
      setFolderChoice(pick);
      setStripError(null);
      settleFailed(placeRef.current);
      focusComposer();
    },
    [focusComposer],
  );

  const stop = useCallback(() => {
    if (conversationId === null || !running || phase === 'cancelling') return;
    mutate('POST', `${conversationPath(conversationId)}/cancel`).catch(() => {
      toast.error(CHAT.sendErrors.other[0]);
    });
  }, [conversationId, running, phase]);

  const compactFull =
    context === undefined ? null : fill(CHAT.history.compactFull, { n: Math.round(contextShare(context) * 100) });
  const compactNow = useCallback(() => {
    if (conversationId === null) return;
    void historyActions.compact(conversationId);
    focusComposer();
  }, [conversationId, focusComposer]);
  const compactionEnd = conv?.compactionEnd ?? null;
  // The Context panel: the ring opens it, `/context` and the palette too (historyActions.showContext).
  const contextPanel = useContextPanel(isNew ? null : conversationId, conv?.turns.length ?? 0, compactionEnd);
  const turnMetrics = conv?.turns;
  const usage = useMemo(() => sessionUsage(turnMetrics ?? []), [turnMetrics]);
  const threadEntriesNow = conv?.entries;
  const endSeen = useRef({ id: conversationId, end: compactionEnd });
  useEffect(() => {
    const seen = endSeen.current;
    endSeen.current = { id: conversationId, end: compactionEnd };
    if (seen.id !== conversationId || compactionEnd === null || compactionEnd === seen.end) return;
    const words = compactEndWords(compactionEnd.outcome, threadEntriesNow ?? []);
    if (words.toast === 'info') toast.info(words.text);
    else if (words.toast === 'error') toast.error(words.text);
    announce(words.text);
  }, [conversationId, compactionEnd, threadEntriesNow, announce]);
  const recapNow = useCallback(() => {
    if (conversationId === null) return;
    void historyActions.recap(conversationId);
    focusComposer();
  }, [conversationId, focusComposer]);
  // A recap that ended with no row says so; `shown` is read by the thread when its row arrives.
  const recapEnd = conv?.recapEnd ?? null;
  const recapSeen = useRef({ id: conversationId, end: recapEnd });
  useEffect(() => {
    const seen = recapSeen.current;
    recapSeen.current = { id: conversationId, end: recapEnd };
    if (seen.id !== conversationId || recapEnd === null || recapEnd === seen.end) return;
    const said = recapEndToast(recapEnd.outcome);
    if (said === null) return;
    if (said.tone === 'info') toast.info(said.text);
    else toast.error(said.text);
    announce(said.text);
  }, [conversationId, recapEnd, announce]);
  const stopRun = useCallback(
    async (runId: string): Promise<void> => {
      if (conversationId === null) return;
      try {
        await mutate('POST', `${conversationPath(conversationId)}/runs/${encodeURIComponent(runId)}/stop`);
      } catch (err) {
        // A run that is not live is reported by the confirmation alone.
        if (!(err instanceof DashboardApiError && err.code === 'E_NOT_FOUND')) {
          toast.error(CHAT.run.stopError, err instanceof Error ? { description: err.message } : undefined);
        }
        throw err;
      }
    },
    [conversationId],
  );

  // The confirm already asked; a failed press is a toast whose Try again sends the Stop once more.
  const stopTask = useCallback(
    (row: TaskRow): void => {
      if (conversationId === null) return;
      const send = (): void => {
        mutate('POST', `${conversationPath(conversationId)}/tasks/${encodeURIComponent(row.id)}/stop`).then(
          () => announce(CHAT.status.taskStopped),
          (err: unknown) => {
            if (err instanceof DashboardApiError && err.code === 'E_NOT_FOUND') {
              toast.info(CHAT.activity.stopGone);
              return;
            }
            toast.error(fill(CHAT.activity.stopError, { title: row.title }), {
              action: { label: CHAT.activity.tryAgain, onClick: send },
            });
          },
        );
      };
      send();
    },
    [conversationId, announce],
  );

  // Work: what was typed in each subagent's message field, by child session id. A ref: a keystroke draws the field alone.
  const messageDrafts = useRef(new Map<string, string>()).current;
  const pendingMessages = useRef(new Map<string, PendingMessage>()).current;

  // A workflow agent's Stop takes the same route: the host sends it to the newest attempt of the row.
  const stopOneAgent = useCallback(
    (row: AgentTarget): void => {
      if (conversationId !== null) stopAgent(conversationId, row, messageDrafts, announce);
    },
    [conversationId, announce, messageDrafts],
  );

  // A message to a running subagent from its view's field.
  const messageSubagent = useCallback(
    (row: MessageTarget, text: string): Promise<MessageEnd> =>
      conversationId === null
        ? Promise.resolve({ kind: 'close', clear: false })
        : messageAgent(conversationId, row, text, {
            drafts: messageDrafts,
            announce,
            open: (path) => void navigate(path),
          }),
    [conversationId, announce, navigate, messageDrafts],
  );

  // Work: Read it now of a PDF the PDF tool did not read. The agent starts the run itself, as a turn of the host's.
  const readPdfNow = useCallback(
    async (pdf: string): Promise<void> => {
      if (conversationId === null) return;
      await mutate('POST', `${conversationPath(conversationId)}/pdf-read`, { pdf });
    },
    [conversationId],
  );

  // Work: the goal strip's buttons. The strip keeps its last true state; a press that failed is a toast. After Clear the
  // composer takes focus; after Pause the focus stays on the same button, which then reads Resume.
  const [goalPress, setGoalPress] = useState<{ id: string; action: GoalAction } | null>(null);
  const goalAction = useCallback(
    (action: GoalAction): void => {
      if (conversationId === null) return;
      const id = conversationId;
      setGoalPress({ id, action });
      mutate('POST', `${conversationPath(id)}/goal`, { action })
        .then(
          () => {
            if (action === 'clear') focusComposer();
          },
          () => toast.error(CHAT.goal.failed[action]),
        )
        .finally(() => setGoalPress((current) => (current?.id === id ? null : current)));
    },
    [conversationId, focusComposer],
  );
  // Each change of the goal's state is said once (polite); its clock never is.
  const goalEntries = loadedHere ? conv.entries : null;
  const goalSaid = useRef<{ id: string | null; status: ReturnType<typeof goalStatusOf> }>({ id: null, status: null });
  useEffect(() => {
    const before = goalSaid.current;
    goalSaid.current = { id: conversationId, status: goalEntries === null ? null : goalStatusOf(goalEntries) };
    if (goalEntries === null || before.id !== conversationId) return;
    const line = goalStatusLine(before.status, goalEntries);
    if (line !== null) announce(line);
  }, [conversationId, goalEntries, announce]);

  // The confirm already asked; a failed press is a toast whose Try again sends the Delete once more.
  const deleteSchedule = useCallback(
    (row: ScheduleRow): void => {
      if (conversationId === null) return;
      const send = (): void => {
        mutate('POST', `${conversationPath(conversationId)}/schedules/${encodeURIComponent(row.id)}/delete`).then(
          () => announce(CHAT.status.scheduleDeleted),
          (err: unknown) => {
            if (err instanceof DashboardApiError && err.code === 'E_NOT_FOUND') {
              toast.info(CHAT.activity.deleteGone);
              return;
            }
            const slow = err instanceof DashboardApiError && err.reason === 'load-slow';
            toast.error(slow ? CHAT.history.loadSlow : CHAT.activity.deleteError, {
              ...(err instanceof DashboardApiError && err.code === 'E_CONFLICT' ? { description: err.message } : {}),
              action: { label: CHAT.activity.tryAgain, onClick: send },
            });
          },
        );
      };
      send();
    },
    [conversationId, announce],
  );

  const runAction = (action: BlockAction, fromFailed = false): void => {
    setStripError(null);
    if (fromFailed) settleFailed(place);
    switch (action.kind) {
      case 'install':
        if (!canInstallAgent) return;
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
        if (failed === null) return;
        if (phase === 'idle' || isNew) void send(failed.text, failed.files);
        else void queueMessage(failed.text, failed.files, false, 'resend');
        return;
      case 'pick-model':
        pickerRef.current?.querySelector('button')?.click();
        return;
      case 'choose-folder':
        void pickAndChoose();
        return;
      case 'go-to-ask': {
        const own = ownAsks.find((item) => item.id === cardId) ?? ownAsks[0];
        if (own !== undefined) goToAsk(own.id);
        return;
      }
    }
  };

  const toNotice = (strip: StripNotice): ComposerNotice => ({
    tone: strip.tone,
    text: strip.text,
    title: strip.title,
    working: strip.working,
    action:
      strip.action === undefined
        ? undefined
        : { label: strip.action.label, onSelect: () => runAction(strip.action!, strip === failed?.strip) },
    secondary:
      strip.secondary === undefined
        ? undefined
        : { label: strip.secondary.label, onSelect: () => runAction(strip.secondary!) },
  });

  const endFailure = conv?.lastEnd?.failure;
  const turnStrip =
    conv !== null && endFailure !== undefined && conv.lastEnd?.outcome === undefined && dismissedEnd !== conv.lastEnd
      ? (() => {
          const outcome = sendFailure(
            endFailure.code,
            undefined,
            mode,
            undefined,
            conv.conversation.modelId,
            canInstallAgent,
          );
          if (!('strip' in outcome)) return null;
          // Retry needs the text of the message that never ran; without it the strip has no button.
          const strip = outcome.strip;
          return strip.action?.kind === 'retry' && failed === null ? { ...strip, action: undefined } : strip;
        })()
      : null;
  const modeStrip = panelShown ? undefined : blocker?.notice;
  const folderStrip: StripNotice | null =
    cwdProblem !== undefined ? { tone: 'danger', text: CHAT.cwdProblems[cwdProblem] } : null;
  const convEntries = conv?.entries;
  const sentPictures = useMemo(
    () =>
      convEntries?.some(
        (entry) => entry.kind === 'user' && entry.attachments !== undefined && imageCount(entry.attachments) > 0,
      ) === true,
    [convEntries],
  );
  const unseenStrip: StripNotice | null =
    conv !== null &&
    sentPictures &&
    host !== null &&
    host.models.find((model) => model.id === conv.conversation.modelId)?.images !== true
      ? { tone: 'warning', text: CHAT.attach.imagesUnseen }
      : null;

  const onDraftChange = (text: string): void => {
    setDraft(text);
    if (stripError !== null) setStripError(null);
    if (failed?.restored === true) writeFailed(place, null);
    if (attachNote !== null) setAttachNote(null);
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

  // A pick in the branch menu. A refusal stays in the menu; one the app made before git ran (uncommitted changes, a
  // running conversation) shows as the menu's note once the read after it lands.
  const switchTo = workGit.switchTo;
  const pickBranch = useCallback(
    async (branch: string) => {
      setBranchAlert(null);
      const result = await switchTo(branch);
      if (result === null) return;
      switch (result.kind) {
        case 'switched':
          setBranchOpen(false);
          focusComposer();
          announce(fill(CHAT.branch.switched, { branch }));
          return;
        case 'refused':
          setBranchAlert({ lead: CHAT.branch.refused, text: result.text });
          return;
        case 'timeout':
          setBranchAlert({ lead: CHAT.branch.timeout, text: '' });
          return;
        case 'error':
          if (result.reason === 'dirty' || result.reason === 'status-unknown' || result.reason === 'folder-busy')
            return;
          setBranchAlert({ lead: result.message === '' ? CHAT.sendErrors.other[0] : result.message, text: '' });
      }
    },
    [switchTo, focusComposer, announce],
  );

  const pageRef = useRef<HTMLDivElement | null>(null);
  const { docked, railDocked, viewPlace } = useChatLayout(pageRef);
  const staticMotion = useStaticMotion();
  const [railPref, setRailPref] = useState<RailPref>(readRailPref);
  const [railPeek, setRailPeek] = useState(false);
  const [railDrawer, setRailDrawer] = useState(false);
  const [railFor, setRailFor] = useState(conversationId);
  if (railFor !== conversationId) {
    setRailFor(conversationId);
    setRailPeek(false);
    setRailDrawer(false);
  }
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
  // A group's "+": New Work in that folder or with no folder, the composer focused; from the drawer, as a row pick.
  const newIn = useCallback(
    (pick: FolderPick) => {
      const fromDrawer = !docked && listOpen;
      pickedFromDrawer.current = fromDrawer;
      setListOpen(false);
      chooseFolder(pick);
      setFolderChoice(pick);
      settleFailed(NEW_PLACE);
      if (isNew && mode === 'work') setStripError(null);
      else void navigate(newUrl('work'));
      if (!fromDrawer) focusComposer();
    },
    [docked, listOpen, isNew, mode, navigate, focusComposer],
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
  // The palette's Delete asks about the conversation it was opened in, and closes once the page shows another.
  const [deleteTarget, setDeleteTarget] = useState<Parameters<typeof deleteContent>[0] | null>(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  if (deleteOpen && (isNew || deleteTarget?.id !== conversationId)) setDeleteOpen(false);
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
    document.title = `${title} — Portato`;
  }, [title]);

  const openedAnnounced = useRef<string | null>(null);
  useEffect(() => {
    if (openId === null || conv === null || openedAnnounced.current === openId) return;
    openedAnnounced.current = openId;
    if (forkOpened.current === openId) {
      forkOpened.current = null;
      const origin = conv.conversation.forkOf?.title ?? conv.conversation.title;
      announce(fill(CHAT.history.forkedStatus, { title: chatTitle(origin, conv.conversation.mode) }));
    } else if (createdId !== openId)
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
            setDeleteTarget({
              id,
              title,
              mode,
              folder: conv?.conversation.cwd ?? null,
              noFolder: conv?.conversation.noFolder === true,
            });
            setDeleteOpen(true);
          },
        });
      }
      if (compactReason === null) {
        list.push({
          id: 'chat:compact',
          label: CHAT.commands.compact,
          group,
          icon: glyph(FoldVertical),
          run: compactNow,
        });
      }
      if (forkReason === null) {
        list.push({
          id: 'chat:fork',
          label: CHAT.commands.fork,
          group,
          icon: glyph(GitFork),
          run: () => forkNow(null),
        });
      }
      if (recapReason === null) {
        list.push({
          id: 'chat:recap',
          label: CHAT.commands.recap,
          group,
          icon: glyph(ScrollText),
          run: recapNow,
        });
      }
      if (rewindReason === null && rewindLast !== null && rewindAt(rewindLast) === null) {
        list.push({
          id: 'chat:rewind-last',
          label: CHAT.commands.rewindLast,
          group,
          icon: glyph(Undo2),
          run: () => historyActions.rewindLast(id),
        });
      }
      list.push({
        id: 'chat:context',
        label: CHAT.commands.context,
        group,
        icon: glyph(Gauge),
        run: () => historyActions.showContext(id),
      });
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
  }, [
    mode,
    running,
    stop,
    isNew,
    conversationId,
    busy,
    title,
    conv,
    navigate,
    switchMode,
    pickAndChoose,
    compactReason,
    compactNow,
    rewindReason,
    rewindLast,
    forkReason,
    forkNow,
    recapReason,
    recapNow,
  ]);
  useCommands(commands);

  // Requests of the agent (Work), oldest first. One card shows: the one the person brought to the front (a rail row,
  // "Go to request", the palette), else the oldest.
  const [closedCards, setClosedCards] = useState<ReadonlySet<string>>(() => new Set());
  /** Conversation notices closed with ✕ in this window, by conversation. */
  const [closedNotices, setClosedNotices] = useState<{ id: string | null; keys: ReadonlySet<string> }>(() => ({
    id: null,
    keys: new Set(),
  }));
  const openCards: Interaction[] =
    mode === 'work' && conv !== null ? conv.interactions.filter((item) => !closedCards.has(item.id)) : [];
  const [front, setFront] = useState<{ conversation: string; id: string } | null>(null);
  const frontId = front !== null && front.conversation === conversationId ? front.id : null;
  const card = openCards.find((item) => item.id === frontId) ?? openCards[0];
  const cardId = card?.id ?? null;
  const cardRef = useRef(card);
  useLayoutEffect(() => {
    cardRef.current = card;
  });
  /** The person picked the shown card: focus moves to it, so its arrival is not said. */
  const askPicked = useRef(false);
  const shownCard = useRef<{ id: string | null; count: number }>({ id: null, count: 0 });
  const openCount = openCards.length;
  useEffect(() => {
    const before = shownCard.current;
    shownCard.current = { id: cardId, count: openCount };
    const focused = document.activeElement?.closest('[data-interaction-card]') != null;
    if (openCount > before.count && openCount > 1) announce(fill(CHAT.status.asksMore, { n: openCount }));
    else if (cardId !== null && cardId !== before.id && cardRef.current !== undefined && !focused && !askPicked.current)
      announce(asksSentence(cardRef.current));
    askPicked.current = false;
    if (before.id !== null && cardId === null) {
      const active = document.activeElement;
      if (active === null || active === document.body || active.closest('[data-interaction-card]') !== null)
        focusComposer();
    }
  }, [cardId, openCount, announce, focusComposer]);
  const withdrawn = conv?.withdrawn ?? 0;
  const withdrawnSeen = useRef<{ id: string | null; n: number }>({ id: null, n: 0 });
  useEffect(() => {
    const before = withdrawnSeen.current;
    withdrawnSeen.current = { id: conversationId, n: withdrawn };
    if (before.id === conversationId && withdrawn > before.n) announce(CHAT.status.askWithdrawn);
  }, [conversationId, withdrawn, announce]);
  const unanswered = conv?.unanswered ?? 0;
  const unansweredSeen = useRef<{ id: string | null; n: number }>({ id: null, n: 0 });
  useEffect(() => {
    const before = unansweredSeen.current;
    unansweredSeen.current = { id: conversationId, n: unanswered };
    if (before.id === conversationId && unanswered > before.n) announce(CHAT.status.askClosed);
  }, [conversationId, unanswered, announce]);
  const mcpUnanswered = conv?.mcpUnanswered ?? 0;
  const mcpUnansweredSeen = useRef<{ id: string | null; n: number }>({ id: null, n: 0 });
  useEffect(() => {
    const before = mcpUnansweredSeen.current;
    mcpUnansweredSeen.current = { id: conversationId, n: mcpUnanswered };
    if (before.id === conversationId && mcpUnanswered > before.n) announce(CHAT.status.mcpClosed);
  }, [conversationId, mcpUnanswered, announce]);
  const mcpOpen =
    conv === null
      ? null
      : openCards
          .filter((item) => item.kind === 'mcp-input')
          .map((item) => item.id)
          .join('\n');
  useEffect(() => {
    if (conversationId === null || mcpOpen === null) return;
    keepMcpDrafts(conversationId, mcpOpen === '' ? [] : mcpOpen.split('\n'));
  }, [conversationId, mcpOpen]);
  const questionOpen =
    conv === null
      ? null
      : openCards
          .filter((item) => item.kind === 'question')
          .map((item) => item.id)
          .join('\n');
  useEffect(() => {
    if (conversationId === null || questionOpen === null) return;
    keepQuestionDrafts(conversationId, questionOpen === '' ? [] : questionOpen.split('\n'));
  }, [conversationId, questionOpen]);
  /** The shown card, scrolled into view: its first enabled button, else the card. */
  const frontTarget = (): HTMLElement | null => {
    const id = cardRef.current?.id;
    const box =
      id === undefined
        ? undefined
        : [...document.querySelectorAll<HTMLElement>('[data-ask-id]')].find((node) => node.dataset.askId === id);
    if (box === undefined) return null;
    box.scrollIntoView({ block: 'nearest', behavior: staticMotion ? 'auto' : 'smooth' });
    return box.querySelector<HTMLElement>('button:not(:disabled)') ?? box;
  };
  /** A rail row, "Go to request" or the palette: that card goes to the front, into view, and takes focus. */
  const [askGo, setAskGo] = useState(0);
  const askAfterDrawer = useRef(false);
  const goToAsk = (id: string): void => {
    if (conversationId === null) return;
    askPicked.current = id !== cardId;
    setFront({ conversation: conversationId, id });
    if (!railDocked && railDrawer) {
      askAfterDrawer.current = true;
      setRailDrawer(false);
      return;
    }
    setAskGo((n) => n + 1);
  };
  const goToAskRef = useRef(goToAsk);
  useLayoutEffect(() => {
    goToAskRef.current = goToAsk;
  });
  useEffect(() => {
    if (askGo !== 0) frontTarget()?.focus({ preventScroll: true });
  }, [askGo]);
  const nextAskId = (openCards.find((item) => item.id !== cardId) ?? card)?.id ?? null;
  const askCommands = useMemo<AppCommand[]>(
    () =>
      nextAskId === null
        ? []
        : [
            {
              id: 'chat:next-ask',
              label: CHAT.commands.nextAsk,
              group: CHAT.commands.group,
              icon: glyph(BellRing),
              run: () => goToAskRef.current(nextAskId),
            },
          ],
    [nextAskId],
  );
  useCommands(askCommands);
  // Work: the Agent view. `?agent=` names the agent (any of its attempts), so a reload keeps it open; it opens from a
  // row or its Message, closes with Escape, ✕ or ‹ Activity, and focus then goes back to the row it opened from.
  const agentParam = mode === 'work' ? params.get('agent') : null;
  const viewEntries = agentParam !== null && loadedHere ? conv.entries : null;
  const viewRunsKept = useRef<ReadonlyMap<string, readonly AgentRowData[]> | undefined>(undefined);
  const viewRuns = useMemo(() => {
    if (viewEntries === null) return null;
    const next = runAgents(viewEntries, viewRunsKept.current);
    viewRunsKept.current = next;
    return next;
  }, [viewEntries]);
  const viewTasksKept = useRef<ReadonlyMap<string, AgentRowData> | undefined>(undefined);
  const viewTasks = useMemo(() => {
    if (viewEntries === null) return null;
    const next = subagentRows(viewEntries, viewTasksKept.current);
    viewTasksKept.current = next;
    return next;
  }, [viewEntries]);
  const viewPlaces = useMemo(() => (viewEntries === null ? null : runsAt(viewEntries)), [viewEntries]);
  const viewRow = useMemo((): AgentRowData | null => {
    if (agentParam === null || viewRuns === null || viewTasks === null) return null;
    return agentRowOf([...viewTasks.values(), ...[...viewRuns.values()].flat()], agentParam);
  }, [agentParam, viewRuns, viewTasks]);
  /** What opened the view: its row's id, and the element focus goes back to while it is still in the page. */
  const viewOpener = useRef<{ id: string; element: HTMLElement | null; rail: boolean } | null>(null);
  /** Where focus goes once the view closed, read by the drawer's close or, in the column, after the next paint. */
  const viewReturn = useRef<{ id: string; element: HTMLElement | null; rail: boolean } | null>(null);
  const [viewFocus, setViewFocus] = useState({ field: 0, title: 0 });
  /** The last open asked for the message field (M, ✉): a drawer then opens with focus in it. */
  const viewWantsField = useRef(false);
  const [marksOpen, setMarksOpen] = useState<AgentMarksOpen | null>(null);
  const setAgentParam = useCallback(
    (id: string | null): void => {
      setParams(
        (current) => {
          const next = new URLSearchParams(current);
          if (id === null) next.delete('agent');
          else next.set('agent', id);
          return next;
        },
        { replace: true },
      );
    },
    [setParams],
  );
  const openAgent = useCallback(
    (target: AgentTarget, opener: HTMLElement | null, field: boolean): void => {
      viewOpener.current = {
        id: target.id,
        element: opener,
        rail: opener?.closest('[data-activity-rail]') != null,
      };
      viewWantsField.current = field;
      setViewFocus((now) => (field ? { ...now, field: now.field + 1 } : { ...now, title: now.title + 1 }));
      setRailDrawer(false);
      setAgentParam(target.id);
    },
    [setAgentParam],
  );
  // Work: what an agent's buttons and its row do on every surface. One object while the conversation stays the same.
  const agentHandlers = useMemo<AgentHandlers>(
    () => ({
      onStop: stopOneAgent,
      onReview: (id) => goToAskRef.current(id),
      onOpen: openAgent,
    }),
    [stopOneAgent, openAgent],
  );
  const agentMessaging = useMemo(
    () => ({ drafts: messageDrafts, pending: pendingMessages, onSend: messageSubagent }),
    [messageDrafts, pendingMessages, messageSubagent],
  );
  /** Removes a never-allow rule of a Work conversation's folder; false when it was not removed (said in a toast). */
  const removeRule = useCallback(
    async (conversation: string, ruleId: string): Promise<boolean> => {
      try {
        await mutate('DELETE', `${conversationPath(conversation)}/never/${encodeURIComponent(ruleId)}`);
      } catch (err) {
        if (!(err instanceof DashboardApiError && err.code === 'E_NOT_FOUND')) {
          toast.error(CHAT.never.removeFailed);
          return false;
        }
      }
      announce(CHAT.status.neverRemoved);
      return true;
    },
    [announce],
  );
  /** The saved toast's Undo: the rule of the conversation it was saved in, whichever the page shows now. */
  const undoRule = useCallback(
    (toastId: string, conversation: string, ruleId: string, words: string): void => {
      toast.update(toastId, { type: 'loading', title: fill(CHAT.never.toast, { words }), action: undefined });
      void removeRule(conversation, ruleId).then((ok) => {
        if (ok) toast.update(toastId, { type: 'success', title: CHAT.never.undone, action: undefined });
        else toast.dismiss(toastId);
      });
    },
    [removeRule],
  );
  const removeThreadRule = useCallback(
    async (ruleId: string): Promise<void> => {
      if (conversationId === null || !(await removeRule(conversationId, ruleId))) throw new Error('not removed');
    },
    [conversationId, removeRule],
  );
  /** Calls the user denied with "Never allow" on their card: their rule mark is not said again. */
  const ownNever = useRef(new Set<string>());
  const answerCard = useCallback(
    async (iid: string, answer: InteractionAnswer): Promise<InteractionAnswerResult> => {
      if (conversationId === null) return {};
      try {
        const result = await mutate<InteractionAnswerResult>(
          'POST',
          `${conversationPath(conversationId)}/interactions/${encodeURIComponent(iid)}`,
          answer,
        );
        if ('note' in answer && answer.note !== undefined) {
          announce(CHAT.status.deniedNote);
          composerRef.current?.focus({ preventScroll: true });
        }
        if ('never' in answer) {
          const never = result?.never;
          if (never?.saved === true) {
            const words = oneLine(never.words);
            toast.success(fill(CHAT.never.toast, { words }), {
              action: {
                label: CHAT.never.undo,
                onClick: (toastId) => undoRule(toastId, conversationId, never.ruleId, words),
              },
            });
          } else {
            toast.error(CHAT.never.saveFailed);
          }
          composerRef.current?.focus({ preventScroll: true });
        }
        return result ?? {};
      } catch (err) {
        const code = err instanceof DashboardApiError ? err.code : undefined;
        if (code === 'E_NOT_FOUND') {
          setClosedCards((current) => new Set(current).add(iid));
          toast.error(CHAT.permClosed);
          return {};
        }
        if ('answers' in answer || 'reply' in answer || 'mcp' in answer) toast.error(CHAT.ask.sendFailed);
        else toast.error(code === 'E_BAD_REQUEST' && err instanceof Error ? err.message : CHAT.sendErrors.other[0]);
        throw err;
      }
    },
    [conversationId, announce, undoRule],
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
        ? { kind: 'draft', id: modeDraft.id, cwd: newFolder }
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
      if (subject?.kind !== 'draft' || (newPick.kind !== 'folder' && newPick.kind !== 'none')) return;
      chooseFolder(newPick);
      setFolderChoice(newPick);
    },
    onMode: (answered) => {
      if (startMode !== null && sending !== null) lowerStart(sending.ts, answered);
    },
  });
  const permissionBlock = permissionModeBlocked({
    connected,
    loading: subject === null,
    starting: startMode !== null,
    cardOpen: openCards.some((item) => MODE_BLOCKING_CARDS.has(item.kind)),
    blocker: modeBlocker(connected, host, 'work')?.reason,
    folder:
      (isNew && mode === 'work' ? placeBlock(newPick) : undefined) ?? folderReason(onDraft, newFolder, cwdProblem),
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
  const neverId = sent === null && workInfo !== null && workInfo.cwd !== null ? workInfo.id : null;
  const neverFolder = workInfo?.cwd ?? null;
  const neverCount = workInfo?.neverRules ?? 0;
  const loadRules = useCallback(async () => {
    if (neverId === null) return [];
    return (await getJson<NeverRulesResponse>(`${conversationPath(neverId)}/never`)).rules;
  }, [neverId]);
  const neverControl = useMemo<NeverRulesControl | undefined>(
    () =>
      neverId === null || neverFolder === null
        ? undefined
        : {
            count: neverCount,
            folder: neverFolder,
            load: loadRules,
            remove: (ruleId) => removeRule(neverId, ruleId),
          },
    [neverId, neverFolder, neverCount, loadRules, removeRule],
  );
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
      for (const mark of permissionMarks?.slice(seen.marks) ?? []) {
        if (mark.kind !== 'denied' || mark.by !== 'rule' || ownNever.current.has(mark.toolCallId)) continue;
        if (typeof mark.words === 'string') announce(fill(CHAT.status.neverDenied, { words: oneLine(mark.words) }));
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

  const position = { index: card === undefined ? 1 : openCards.indexOf(card) + 1, count: openCards.length };
  // An agent asks: its row and its run's block say so. A run agent's card does not count as the typed turn waiting.
  const interactions = conv?.interactions;
  const agentAsks = useMemo(() => {
    const list: AgentAsk[] = [];
    if (mode !== 'work' || interactions === undefined) return list;
    for (const item of interactions) {
      if (closedCards.has(item.id) || item.kind !== 'permission' || item.from === undefined) continue;
      const from = item.from;
      list.push({
        id: item.id,
        child: from.child,
        run: from.run,
        label: from.label,
        title: item.title,
        access: item.access,
        openedAt: item.openedAt,
      });
    }
    return list;
  }, [mode, interactions, closedCards]);
  // The asks of the conversation's own answer or wake: the composer follows only these.
  const ownAsks = openCards.filter((item) => askRun(item) === null);
  const turnAsks = ownAsks.length > 0;
  // The rows of these calls open by themselves: an edit's change shows before Allow.
  const askCallKey = ownAsks.map((item) => item.toolCallId).join('\n');
  const askCalls = useMemo(() => new Set(askCallKey === '' ? [] : askCallKey.split('\n')), [askCallKey]);
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
            onAnswer={(answer) => {
              if ('never' in answer) ownNever.current.add(card.toolCallId);
              return answerCard(card.id, answer);
            }}
          />
        ) : card.kind === 'mcp-input' ? (
          <McpCard
            key={card.id}
            interaction={card}
            folder={openFolder}
            position={position}
            composerRef={composerRef}
            onAnswer={(answer) => answerCard(card.id, answer)}
          />
        ) : card.kind === 'question' ? (
          <QuestionCard
            key={card.id}
            interaction={card}
            folder={openFolder}
            position={position}
            composerRef={composerRef}
            onStatus={announce}
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

  const liveRun = (conv?.runs ?? []).some((run) => run.live);
  const stripNames = useMemo(
    () => (liveRun && convEntries !== undefined ? shownRunNames(runsAt(convEntries)) : NO_RUN_NAMES),
    [liveRun, convEntries],
  );

  // The thread.
  let entries: readonly TranscriptEntry[] = conv?.entries ?? [];
  let threadTurn: TurnState | null = turn;
  let lastEnd: ConversationState['lastEnd'] = conv?.lastEnd ?? null;
  if (sendingHere && sending !== null) {
    if (entries.length <= sending.base) {
      const sent: TranscriptEntry = { kind: 'user', ts: sending.ts, text: sending.text, promptIndex: null };
      if (sending.inputs.length > 0) sent.attachments = sending.inputs.map(attachmentView);
      entries = [...entries, sent];
    }
    threadTurn = turn ?? STARTING;
    if (turn === null) lastEnd = null;
  }
  const activity = isNew
    ? NO_ACTIVITY
    : activityOf({
        entries,
        interactions: openCards,
        frontId: cardId,
        runs: conv?.runs ?? [],
        live: conv?.live,
        turnRunning: phase !== 'idle' && phase !== 'op',
        agentLive: conv?.conversation.hot === true,
      });
  const activeNow = hasActivity(activity);
  const stripLines = mode === 'work' && !railDocked && !isNew && conv !== null ? runStripOf(activity, stripNames) : [];

  // An ask of the conversation's own answer waits while the thread is scrolled away from its card.
  const ownRows = activity.needs.filter((row) => ownAsks.some((item) => item.id === row.id));
  const askStrip: StripNotice | null =
    ownRows.length > 0 && stick.showJump
      ? {
          tone: 'neutral',
          text: ownRows.length === 1 ? CHAT.needs.composerOne : fill(CHAT.needs.composerMany, { n: ownRows.length }),
          title:
            ownRows.length === 1
              ? fill(CHAT.needs.composerOneTip, { title: ownRows[0]!.title })
              : fill(CHAT.status.asksMore, { n: ownRows.length }),
          action: { label: CHAT.needs.composerGo, kind: 'go-to-ask' },
        }
      : null;
  const strip: StripNotice | null =
    modeStrip ??
    stripError ??
    failed?.strip ??
    folderStrip ??
    askStrip ??
    turnStrip ??
    engineLostStrip ??
    unseenStrip ??
    infoNotice(host, mode);
  const railOpen = railDocked ? (railPref === 'auto' && activeNow) || railPeek : railDrawer;
  const viewOpen = viewRow !== null && conversationId !== null;
  const viewInColumn = viewOpen && viewPlace === 'column';
  /** The view closed: focus goes back to the row it opened from (its rail row after ‹ Activity), else the composer. */
  const closeAgent = (toRail = false): void => {
    const opener = viewOpener.current;
    const id = viewRow?.childId ?? agentParam;
    viewReturn.current =
      id === null
        ? null
        : { id, element: toRail ? null : (opener?.element ?? null), rail: toRail || opener?.rail === true };
    viewOpener.current = null;
    setAgentParam(null);
  };
  const toggleRail = (): void => {
    if (viewInColumn) {
      closeAgent(true);
      setRailPeek(true);
      return;
    }
    if (!railDocked) {
      setRailDrawer((open) => !open);
      return;
    }
    if (railOpen) {
      setRailPeek(false);
      if (activeNow) setRailPref(writeRailPref('off'));
    } else if (activeNow) setRailPref(writeRailPref('auto'));
    else setRailPeek(true);
  };
  const jumpToEntry = (anchor: number): void => {
    threadJump.current?.(anchor);
    if (!railDocked) setRailDrawer(false);
  };
  const showRail = !isNew && conv !== null && (mode === 'work' || activeNow);

  /** The element `viewReturn` names: the opener while it is in the page, else the row of that agent, rail or thread. */
  const viewReturnTarget = (): HTMLElement | null => {
    const want = viewReturn.current;
    viewReturn.current = null;
    if (want === null) return null;
    if (want.element?.isConnected === true) return want.element;
    const rows = [...document.querySelectorAll<HTMLElement>('[data-agent-id]')].filter(
      (node) => node.dataset.agentId === want.id,
    );
    const rail = rows.find((node) => node.closest('[data-activity-rail]') !== null);
    const thread = rows.find((node) => node.closest('[data-activity-rail]') === null);
    const pick = want.rail ? (rail ?? thread) : (thread ?? rail);
    return pick?.querySelector<HTMLElement>(':scope > [data-jump], [data-agent-focus]') ?? composerRef.current ?? null;
  };
  // In the column the rail comes back in the view's place: focus moves once it is drawn.
  const viewWasColumn = useRef(false);
  const viewShownNow = viewInColumn;
  useEffect(() => {
    const was = viewWasColumn.current;
    viewWasColumn.current = viewShownNow;
    if (!was || viewShownNow || viewReturn.current === null) return;
    const frame = requestAnimationFrame(() => viewReturnTarget()?.focus({ preventScroll: true }));
    return () => cancelAnimationFrame(frame);
  });
  // A row that is no longer in the loaded thread (a rewind took it) closes its view.
  useEffect(() => {
    if (agentParam !== null && viewEntries !== null && viewRow === null) setAgentParam(null);
  }, [agentParam, viewEntries, viewRow, setAgentParam]);
  const backToRail = (): void => {
    closeAgent(true);
    if (!railDocked) setRailDrawer(true);
    else setRailPeek(true);
  };
  /** Where the row of an agent is drawn in the thread: a run's agent in the run's block. */
  const rowAnchor = (row: AgentRowData): number | null => {
    if (row.run !== null) return viewPlaces?.runs.get(row.run)?.place.at ?? null;
    return row.at[0] ?? null;
  };
  const stepAgent = (step: -1 | 1): void => {
    if (viewRow === null || viewTasks === null || viewRuns === null) return;
    const list = viewRow.run === null ? [...viewTasks.values()] : (viewRuns.get(viewRow.run) ?? []);
    const next = neighbourAgent(list, viewRow.childId, step);
    if (next === null) return;
    viewOpener.current = { id: next.childId, element: null, rail: viewOpener.current?.rail === true };
    setAgentParam(next.childId);
  };
  let agentView: ReactNode = null;
  if (viewOpen) {
    const row = viewRow;
    const at = rowAnchor(row);
    const progress = conv?.live.helpers.find((helper) => helper.childId === row.childId) ?? null;
    const showAt = (then?: () => void): (() => void) | null =>
      at === null
        ? null
        : () => {
            // The jump moves focus to the row, so the drawer's close leaves it there.
            if (!viewInColumn) {
              closeAgent();
              viewReturn.current = null;
            }
            then?.();
            jumpToEntry(at);
          };
    agentView = (
      <AgentPane
        key={row.attempts[0]}
        conversationId={conversationId}
        row={row}
        place={viewPlace}
        runName={row.run === null ? null : (viewPlaces?.names.get(row.run) ?? null)}
        agentLive={conv?.conversation.hot === true}
        live={conv?.live}
        asks={agentAsks}
        needs={activity.needs.length + activity.needsRuns.length}
        marks={agentMarkCounts(permissionMarks, row.attempts)}
        opens={transcriptOpens(row.entry?.exists, progress, row.entry?.endTs !== undefined ? row.entry : undefined)}
        handlers={agentHandlers}
        messaging={agentMessaging}
        fieldTick={viewFocus.field}
        titleTick={viewFocus.title}
        onBack={backToRail}
        onClose={() => closeAgent()}
        onShowInThread={showAt()}
        onShowMarks={showAt(() => setMarksOpen((now) => ({ id: row.childId, tick: (now?.tick ?? 0) + 1 })))}
        onStep={stepAgent}
      />
    );
  }

  // The palette's Agents group: every running agent and run of the open conversation, and Stop for each run.
  const paletteItems = mode === 'work' ? agentPaletteOf(activity) : [];
  const paletteKey = JSON.stringify(paletteItems);
  const paletteNow = useRef({ items: paletteItems, jump: jumpToEntry, open: openAgent });
  paletteNow.current = { items: paletteItems, jump: jumpToEntry, open: openAgent };
  const [paletteStop, setPaletteStop] = useState<{ runId: string; name: string; running: number } | null>(null);
  const agentCommandList = useMemo<AppCommand[]>(() => {
    const group = CHAT.commands.agents;
    const list: AppCommand[] = [];
    for (const item of paletteNow.current.items) {
      if (item.kind === 'agent') {
        list.push({
          id: `agent:${item.id}`,
          label: item.label,
          description: item.run === null ? CHAT.subagent.label : fill(CHAT.commands.agentOf, { run: item.run }),
          group,
          icon: glyph(item.run === null ? GitFork : Workflow),
          run: () => paletteNow.current.open({ id: item.id, title: item.label }, null, false),
        });
        continue;
      }
      list.push({
        id: `run:${item.id}`,
        label: fill(CHAT.commands.workflow, { name: item.name }),
        description: item.detail,
        group,
        icon: glyph(Workflow),
        run: () => paletteNow.current.jump(item.anchor),
      });
      if (item.stoppable)
        list.push({
          id: `run-stop:${item.id}`,
          label: fill(CHAT.commands.stopWorkflow, { name: item.name }),
          group,
          icon: glyph(Square),
          run: () => setPaletteStop({ runId: item.id, name: item.name, running: item.running }),
        });
    }
    return list;
  }, [paletteKey]);
  useCommands(agentCommandList);

  // Send again carries the files while this window still holds the message it was sent with.
  const keptFor =
    lastSent !== null && lastSent.place === conversationId && lastSent.files.length > 0 ? lastSent.text : null;
  const resend = (action: Extract<TurnAction, { text: string }>): void => {
    const files =
      action.kind === 'send-again' && keptFor !== null && keptFor === action.text && lastSent !== null
        ? resendFiles(lastSent.files, DROPPED_STAGES)
        : [];
    void send(action.text, files);
  };
  const resendRef = useRef(resend);
  useLayoutEffect(() => {
    resendRef.current = resend;
  });
  const canInstallRef = useRef(canInstallAgent);
  useLayoutEffect(() => {
    canInstallRef.current = canInstallAgent;
  });
  const onTurnAction = useCallback((action: TurnAction) => {
    if (action.kind === 'update-agent') {
      if (!canInstallRef.current) return;
      mutate('POST', '/chat/agent', { action: 'install' }).catch(() => undefined);
      return;
    }
    if (action.kind === 'pick-model') {
      pickerRef.current?.querySelector('button')?.click();
      return;
    }
    resendRef.current(action);
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
        noFolder={newPick.kind === 'none' ? { root: scratchRoot } : null}
        onStarter={(text) => {
          setDraft(text);
          focusComposer();
        }}
        onAction={runAction}
      />
    );
  } else if (rewind.atStart && !sendingHere) {
    body = (
      <p className="m-0 py-6 text-center text-body text-muted" data-rewound-start="">
        {CHAT.history.empty}
      </p>
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
            : {
                inference: host.inference,
                loading: host.loading,
                resident: unload.resident,
                answers: host.answers,
                requests: host.requests,
                appEnds: host.appEnds,
              }
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
        canInstallAgent={canInstallAgent}
        keptFor={keptFor}
        actionDisabledReason={engineReason}
        onStatus={announce}
        contentRef={stick.contentRef}
        folder={openFolder}
        tempRoot={permissionTemp}
        card={cardNode}
        asking={turnAsks}
        askCalls={askCalls}
        marks={permissionMarks}
        compaction={conv?.compaction ?? null}
        hookRun={conv?.hookRun ?? null}
        recap={recapWait}
        retry={conv?.retry ?? null}
        runs={conv?.runs}
        agentAsks={agentAsks}
        jumpRef={threadJump}
        onStopRun={stopRun}
        agentHandlers={mode === 'work' ? agentHandlers : undefined}
        openAgent={viewRow?.childId ?? null}
        agentMarksOpen={marksOpen}
        live={conv?.live}
        onReadPdf={mode === 'work' && seesPages && pdfTool ? readPdfNow : undefined}
        forked={conv?.conversation.forkOf !== undefined}
        agentLive={conv?.conversation.hot === true}
        rewind={threadRewind}
        onRemoveRule={mode === 'work' ? removeThreadRule : undefined}
        top={
          conv?.conversation.forkOf === undefined ? undefined : (
            <ForkOrigin
              title={
                conv.conversation.forkOf.title === null
                  ? null
                  : chatTitle(conv.conversation.forkOf.title, conv.conversation.mode)
              }
              to={chatUrl(conv.conversation.forkOf.id)}
            />
          )
        }
      />
    );
  }

  const removeContent =
    !isNew && conversationId !== null
      ? deleteContent(
          {
            id: conversationId,
            title,
            mode,
            folder: conv?.conversation.cwd ?? null,
            noFolder: conv?.conversation.noFolder === true,
          },
          onDeleted,
        )
      : null;

  const placeholder =
    !isNew && turnAsks
      ? CHAT.needs.composerPlaceholder
      : held.length > 0
        ? CHAT.attach.placeholder
        : !isNew
          ? permissionMode === 'bypass'
            ? CHAT.permission.bypass.placeholder
            : CHAT.placeholderReply
          : mode === 'work'
            ? newFolder === null
              ? permissionMode === 'bypass'
                ? CHAT.permission.bypass.placeholderNewNone
                : CHAT.placeholderWorkNone
              : fill(permissionMode === 'bypass' ? CHAT.permission.bypass.placeholderNew : CHAT.placeholderWork, {
                  folder: folderName(newFolder),
                })
            : CHAT.placeholderChat;

  // A PDF whose draft is already in the store keeps one chip while its pending entry leaves.
  const pendingChips = drawing.filter((entry) => !held.some((item) => item.id === entry.id));
  const full = held.length + pendingChips.length >= ATTACH_MAX;
  const attachRow = {
    open: openPicker,
    full,
    note: full
      ? CHAT.attach.noteFull
      : mode === 'work'
        ? seesPages
          ? CHAT.attach.noteWorkPdf
          : CHAT.attach.noteWork
        : seesImages
          ? undefined
          : CHAT.attach.noteNoImages,
  };
  const pictureProblem = seesImages ? undefined : mode === 'work' ? CHAT.attach.chipWork : CHAT.attach.chipModel;
  // Which way each held PDF goes now: a model change or a fuller conversation can move one, with no second draw.
  const heldPlan = held.length > 0 ? attachPlan({ ...attachFacts, text: draft, files: held }) : null;
  const heldWays = fileWays(held, heldPlan);
  const chips = held.map((item, index): AttachmentChipItem => {
    if (item.kind === 'file') return { id: item.id, kind: 'file', name: item.name, bytes: item.bytes };
    if (item.kind === 'pdf') {
      return {
        id: item.id,
        kind: 'pdf',
        name: item.name,
        bytes: item.bytes,
        pages: item.sizes.length,
        thumbnail: item.thumbnail,
        way: heldWays[index],
        problem: seesPages ? undefined : CHAT.attach.chipModel,
      };
    }
    return {
      id: item.id,
      kind: 'image',
      name: item.name,
      mediaType: item.mediaType,
      bytes: item.bytes,
      thumbnail: item.dataUrl,
      problem: pictureProblem,
    };
  });
  for (const entry of pendingChips) {
    const ready = entry.total !== null && entry.done === entry.total;
    chips.push({
      id: entry.id,
      kind: 'pdf',
      name: entry.name,
      bytes: entry.bytes,
      ...(ready ? { pages: entry.done, problem: seesPages ? undefined : CHAT.attach.chipModel } : { drawing: true }),
      ...(entry.thumbnail === null ? {} : { thumbnail: entry.thumbnail }),
    });
  }
  const drawn = drawingNow(pendingChips);
  const drawNow = useNow(drawn !== undefined);
  const estimate =
    held.length > 0
      ? estimateTokens(
          draft,
          held.map((item, index) => draftTokens(item, heldWays[index])),
          family,
        )
      : 0;
  const nearFull = windowTokens !== null && (contextUsed ?? 0) + estimate > windowTokens * WINDOW_SHARE;
  const readTokens = heldWays.includes('tool') ? (heldPlan?.readTokens ?? 0) : null;
  const attachStrip = (
    <AttachmentStrip
      items={chips}
      onRemove={removeAttachment}
      note={attachNote !== null && attachNote.place === place ? attachNote.text : undefined}
      caption={
        drawn !== undefined
          ? { text: drawCaption(drawn, drawNow), tone: 'muted' }
          : held.length === 0
            ? undefined
            : {
                text:
                  readTokens !== null
                    ? fill(CHAT.attach.pdfToolTokens, { n: formatNumber(estimate), m: formatNumber(readTokens) })
                    : fill(nearFull ? CHAT.attach.tokensNearFull : CHAT.attach.tokens, { n: formatNumber(estimate) }),
                tone: nearFull ? 'warning' : 'muted',
              }
      }
      progress={
        drawn !== undefined
          ? {
              label: fill(CHAT.attach.pdfDrawBar, { name: drawn.name }),
              text: drawCount(drawn.done, drawn.total),
              done: drawn.done,
              total: drawn.total,
            }
          : undefined
      }
    />
  );

  const levelNotSet = !isNew && conv?.notices.some((notice) => notice.code === 'thinking-not-set') === true;
  // A model with levels keeps the chip open while an answer runs, for its levels; only the model rows are off.
  const levelled = (modelLevels?.levels.length ?? 0) > 0;
  const modelOff = !isNew && (busy || runLive);
  const modelOffReason =
    !isNew && busy ? CHAT.modelBusy : runLive ? (helperLive ? CHAT.modelBusyHelper : CHAT.modelBusyRun) : undefined;

  const tools = (
    <>
      <ComposerMenu providers={host?.providers ?? null} mode={mode} attach={attachRow} onCommands={openCommands} />
      <span ref={pickerRef} className="inline-flex min-w-0" data-give="3">
        <ModelPicker
          value={modelValue}
          bases={bases}
          fallback={startsOn}
          onValueChange={pickModel}
          disabled={modelOff && !levelled}
          disabledReason={modelOffReason}
          listDisabled={modelOff && levelled}
          listDisabledReason={modelOffReason}
          levelWarning={levelNotSet}
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
          never={neverControl}
        />
      ) : null}
      {mode === 'work' ? <PlanChip view={plan} busy={busy} onToggle={togglePlan} /> : null}
    </>
  );

  const repoGit = workGit.git?.kind === 'repo' && workGit.problem === null ? workGit.git : null;
  let branchChip: ReactElement | null = null;
  if (repoGit !== null) {
    const repoHead = repoGit.head;
    const where = folderName(repoGit.top);
    const changes = repoGit.changes;
    branchChip = (
      <BranchMenu
        part="branch"
        label={'branch' in repoHead ? repoHead.branch : 'detached' in repoHead ? repoHead.detached : repoHead.unborn}
        detached={'detached' in repoHead}
        name={
          'branch' in repoHead
            ? fill(CHAT.branch.chip, { branch: repoHead.branch })
            : 'detached' in repoHead
              ? fill(CHAT.branch.detached, { hash: repoHead.detached })
              : fill(CHAT.branch.unborn, { branch: repoHead.unborn })
        }
        off={
          changes === null
            ? fill(CHAT.branch.dirtyUnknown, { folder: where })
            : changes === 1
              ? fill(CHAT.branch.dirtyOne, { folder: where })
              : changes > 1
                ? fill(CHAT.branch.dirtyMany, { folder: where, n: formatNumber(changes) })
                : null
        }
        open={branchOpen}
        onOpenChange={(open) => {
          setBranchOpen(open);
          setBranchAlert(null);
          if (open) workGit.refresh();
        }}
        top={repoGit.top}
        branches={repoGit.branches}
        total={repoGit.total}
        busy={repoGit.busy > 0 ? CHAT.branch.busy : null}
        pending={workGit.switching}
        alert={branchAlert}
        onPick={(branch) => void pickBranch(branch)}
      />
    );
  }
  const trayContent = !trayShown ? null : mode === 'work' ? (
    <>
      <NewFolderChip
        folder={newFolder}
        recent={recent}
        onChoose={(path) => choosePlace({ kind: 'folder', path })}
        onPick={() => void pickAndChoose()}
        problem={workGit.problem}
        none={{ selected: newPick.kind === 'none', onChoose: () => choosePlace({ kind: 'none' }), root: scratchRoot }}
        {...(unplaced !== undefined ? { unplaced } : {})}
      />
      {branchChip}
    </>
  ) : (
    <NewFolderChip
      folder={null}
      recent={recent}
      label={CHAT.tray.workIn}
      onChoose={(path) => {
        choosePlace({ kind: 'folder', path });
        switchMode('work');
      }}
      onPick={() =>
        void pickAndChoose().then((picked) => {
          if (picked) switchMode('work');
        })
      }
    />
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
                ? { id: conversationId, running: busy, deleteOff, remove: removeContent }
                : null
            }
            onRename={() => {
              if (conversationId !== null) renameItem({ id: conversationId, title });
            }}
            trailing={docked ? null : newButton}
            menu={
              <ThreadMenu
                fork={{ off: forkReason, onFork: () => forkNow(null) }}
                compact={{ off: compactReason, full: compactFull, onCompact: compactNow }}
                recap={{ off: recapReason, onRecap: recapNow }}
              />
            }
            activity={
              showRail ? (
                <ActivityToggle
                  open={railOpen}
                  running={activity.running.length}
                  needs={activity.needs.length}
                  onToggle={toggleRail}
                />
              ) : null
            }
            folder={
              openFolder !== null ? (
                <OpenFolderChip
                  folder={openFolder}
                  noFolder={(conv?.conversation.noFolder ?? rowOf(conversationId)?.noFolder) === true}
                />
              ) : null
            }
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
                  never={neverControl}
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
            onNewIn={newIn}
            scratchRoot={host?.scratchRoot ?? null}
          />
        }
        listLabel={CHAT.conversations}
        listOpen={listOpen}
        onListOpenChange={setListOpen}
        listCloseFocus={listCloseFocus}
        rail={
          showRail
            ? {
                content: (
                  <ActivityRail
                    activity={activity}
                    onJump={jumpToEntry}
                    onAsk={goToAsk}
                    onStopTask={mode === 'work' ? stopTask : undefined}
                    agents={mode === 'work' ? agentHandlers : undefined}
                    onStopRun={mode === 'work' ? stopRun : undefined}
                    onDeleteSchedule={mode === 'work' ? deleteSchedule : undefined}
                    deleteOff={engineLostStrip !== null ? CHAT.history.engineLost : null}
                    docked={railDocked}
                  />
                ),
                label: CHAT.activity.label,
                docked: railDocked,
                open: railOpen,
                onOpenChange: setRailDrawer,
                ...(agentView !== null && viewRow !== null
                  ? {
                      view: {
                        content: agentView,
                        title: viewRow.label,
                        place: viewPlace,
                        onClose: () => closeAgent(),
                        openFocus: () =>
                          viewWantsField.current
                            ? document.querySelector<HTMLElement>('[data-agent-view] [data-agent-field] textarea')
                            : null,
                        closeFocus: viewReturnTarget,
                      },
                    }
                  : {}),
                closeFocus: () => {
                  if (!askAfterDrawer.current) return null;
                  askAfterDrawer.current = false;
                  return frontTarget();
                },
              }
            : undefined
        }
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
            {conv !== null ? (
              <ConversationNotices
                notices={conv.notices.filter(
                  (notice) => !(closedNotices.id === conversationId && closedNotices.keys.has(noticeKey(notice))),
                )}
                onClose={(notice) =>
                  setClosedNotices((current) => ({
                    id: conversationId,
                    keys: new Set(current.id === conversationId ? current.keys : []).add(noticeKey(notice)),
                  }))
                }
              />
            ) : null}
            {stripLines.length > 0 ? (
              <RunStrip
                lines={stripLines}
                onJump={jumpToEntry}
                onOpen={() => setRailDrawer(true)}
                onReview={goToAsk}
                onStop={stopRun}
              />
            ) : null}
            {mode === 'work' && activity.goal != null ? (
              <GoalStrip
                goal={activity.goal}
                pending={goalPress?.id === conversationId ? goalPress.action : null}
                busy={busy}
                onAction={goalAction}
              />
            ) : null}
            <ComposerTray tray={trayContent} label={CHAT.tray.label}>
              <Composer
                value={draft}
                onValueChange={onDraftChange}
                placeholder={placeholder}
                phase={phase}
                onSend={() => void send(draft, held, 'composer')}
                onQueue={isNew ? undefined : () => void queueMessage(draft, held)}
                onSendNow={
                  isNew || mode !== 'work' || (queueFull && !steerDraft)
                    ? undefined
                    : () => void queueMessage(draft, held, true)
                }
                onStop={stop}
                blocked={blocked}
                queueOff={queueFull ? CHAT.queue.full : undefined}
                hint={slashHint(slash)}
                commands={slashList}
                notice={strip === null ? undefined : toNotice(strip)}
                queue={
                  queued.items.length > 0 ? (
                    <QueueList
                      items={queued.items}
                      held={heldSentence(queued.held, host, conversationId, mode)}
                      waits={goalRunning ? CHAT.queue.held.goal : null}
                      editing={queueControls.editing}
                      onEditingChange={queueControls.setEditing}
                      onSave={queueControls.save}
                      onRemove={queueControls.remove}
                      onClear={queueControls.clear}
                      onSendNext={queueControls.sendNext}
                      onFocusOut={focusComposer}
                      sendNow={
                        steerShown
                          ? {
                              off: (item) =>
                                !steerOpen
                                  ? CHAT.queue.sendNowOffStarting
                                  : item.files.length > 0
                                    ? CHAT.queue.sendNowOffFiles
                                    : isSlashText(item.text)
                                      ? CHAT.queue.sendNowOffSlash
                                      : null,
                              busy: queueControls.sending,
                              onSend: queueControls.sendNow,
                            }
                          : undefined
                      }
                    />
                  ) : undefined
                }
                onEditLast={
                  queued.items.length > 0 ? () => queueControls.setEditing(queued.items.at(-1)!.id) : undefined
                }
                onCancelEdit={queueControls.editing !== null ? () => queueControls.setEditing(null) : undefined}
                tools={tools}
                context={context}
                contextPanel={
                  isNew || conversationId === null
                    ? undefined
                    : {
                        open: contextPanel.open,
                        onOpenChange: contextPanel.setOpen,
                        content: (
                          <ContextPanel
                            used={contextUsed ?? contextPanel.used}
                            window={contextWindow}
                            parts={contextPanel.parts}
                            usage={usage}
                            compact={{
                              off: compactReason,
                              onCompact: () => {
                                contextPanel.setOpen(false);
                                compactNow();
                              },
                            }}
                            inspectTo={`/sessions/${encodeURIComponent(conversationId)}?tab=activity`}
                            onInspect={() => contextPanel.setOpen(false)}
                          />
                        ),
                      }
                }
                textareaRef={composerRef}
                attachments={attachStrip}
                onFiles={addFiles}
              />
            </ComposerTray>
            <input
              ref={bindPicker}
              type="file"
              multiple
              className="sr-only"
              tabIndex={-1}
              aria-label={CHAT.attach.row}
              onChange={(event) => {
                const input = event.currentTarget;
                addFiles(Array.from(input.files ?? []));
                input.value = '';
                focusComposer();
              }}
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
      {rewind.dialog}
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
      {deleteTarget !== null ? (
        <ConfirmDialog
          key={deleteTarget.id}
          {...deleteContent(deleteTarget, onDeleted)}
          open={deleteOpen}
          onOpenChange={setDeleteOpen}
          returnFocus={returnFocus}
        />
      ) : null}
      {paletteStop !== null ? (
        <ConfirmDialog
          title={CHAT.run.stopTitle}
          description={stopText(paletteStop.running)}
          confirmLabel={CHAT.run.buttons.stopWorkflow}
          cancelLabel={CHAT.run.buttons.keepRunning}
          workingLabel={CHAT.run.status.stopping}
          successLabel={CHAT.run.status.stopping}
          errorTitle={CHAT.run.stopError}
          focusCancel
          open
          onOpenChange={(open) => {
            if (!open) setPaletteStop(null);
          }}
          onConfirm={() => stopRun(paletteStop.runId)}
        />
      ) : null}
    </div>
  );
}

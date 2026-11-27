import {
  BellRing,
  CircleHelp,
  Cpu,
  FilePen,
  FileText,
  Folder,
  RotateCw,
  Search,
  SquareTerminal,
  Workflow,
} from 'lucide-react';
import {
  createContext,
  memo,
  use,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactElement,
  type ReactNode,
} from 'react';
import { Link } from 'react-router-dom';

import {
  eventsWords,
  progressOf,
  taskLiveOf,
  transcriptOpens,
  transcriptPath,
  wakeReportOf,
  wakeReportsOf,
  type WakeTask,
} from '../lib/activity';
import {
  type AgentAsk,
  agentButtons,
  agentLiveOf,
  anyButton,
  type AgentRow as AgentRowData,
  agentState,
  runAgents,
  stoppedByYou,
  subagentRows,
} from '../lib/agents';
import type { AppEndContext } from '../lib/app-ends';
import type { ConversationState } from '../lib/chat-store';
import { CHAT, fill } from '../lib/chat-strings';
import {
  answerMarkdown,
  compactionDetail,
  continuingTurn,
  engineNote,
  engineWait,
  fullLimit,
  hookHeadline,
  hookWaitLine,
  hookWhen,
  isSideEntry,
  loadFor,
  loadVariant,
  messagePages,
  modelHeadline,
  openRetry,
  pendingRow,
  pictureHeadline,
  retryLine,
  retryReading,
  mountedFor,
  splitTurns,
  thoughtDuration,
  turnOutcome,
  turnPrompt,
  turnStartedAt,
  turnStats,
  turnStatsTitle,
  type EngineRequest,
  type OutcomeLine,
  type ReadCounts,
  type RetryNotice,
  type ThreadTurn,
} from '../lib/chat-turns';
import type {
  PermissionMark,
  ChatHostState,
  ChatMode,
  CompactionState,
  ConversationNotice,
  HookRunState,
  LiveActivity,
  ModelSwitch,
  RecapState,
  RetryLive,
  RunView,
  TaskLive,
  TurnOutcome,
  TurnState,
} from '../lib/chat-types';
import { formatBytes, formatClock, formatCount } from '../lib/format';
import { type RowWords, rowWords } from '../lib/helper-words';
import { truncateMiddle } from '../lib/labels';
import { hostStartWords, pdfNames, pdfRunName, pdfUnread } from '../lib/pdf-run';
import {
  type AllowedMark,
  allowedWord,
  type DeniedMark,
  deniedWord,
  neverNoteWords,
  placeMarks,
  subagentAllowedText,
  subagentDeniedText,
} from '../lib/permission-text';
import { countWords, runCounts, type RunsAt, runsAt, startedPdfs } from '../lib/run-view';
import { wakeTitle } from '../lib/sessions';
import type { TranscriptEntry, TurnMetric } from '../lib/types';
import type { WebResult } from '../lib/web-result-text';
import {
  conversationHosts,
  errorNeedsSettings,
  httpUrl,
  matchCitations,
  oneLine,
  providerName,
  refusedCall,
  resultCountText,
  splitAnswerSources,
  turnSources,
  webCallOf,
  webErrorText,
  type Refusal,
  type WebCall,
} from '../lib/web-tools';
import { visibleText, type WorkCall, workCallOf, type WorkflowForm, workflowFormOf } from '../lib/work-tools';
import { useScrollRoot } from '../shell/scroll-root';
import { TaskEvents } from './arc-x/activity-rail/task-events';
import { AgentControls, type AgentHandlers } from './arc-x/agent-controls/agent-controls';
import { AgentList } from './arc-x/agent-list/agent-list';
import { AgentRow } from './arc-x/agent-row/agent-row';
import { AttachmentStrip, sentChips } from './arc-x/attachment-strip/attachment-strip';
import { ButtonLink } from './arc-x/button-link/button-link';
import { ChatMarkdown, type LinkRule } from './arc-x/chat-markdown/chat-markdown';
import { CodePanel } from './arc-x/code-panel/code-panel';
import { diffStateOf, EditDiffs, editCounts } from './arc-x/diff-view/diff-view';
import { Disclosure } from './arc-x/disclosure/disclosure';
import { EventRow, RuleLabel } from './arc-x/event-row/event-row';
import { GoalRow } from './arc-x/goal-strip/goal-strip';
import { KeyValue } from './arc-x/key-value/key-value';
import { LoadCard } from './arc-x/load-card/load-card';
import { MessageActions } from './arc-x/message-actions/message-actions';
import { Message } from './arc-x/message/message';
import { useStaticMotion } from './arc-x/motion/use-static-motion';
import { NoticeLine, NoticeMore } from './arc-x/notice-line/notice-line';
import { PictureScope, PictureStrip, picturesDetail } from './arc-x/picture-strip/picture-strip';
import { Placeholder } from './arc-x/placeholder/placeholder';
import { PlanList } from './arc-x/plan-list/plan-list';
import { ReasoningBlock } from './arc-x/reasoning-block/reasoning-block';
import { SourceList, type SourceItem } from './arc-x/source-list/source-list';
import { Meter } from './arc-x/split-bar/split-bar';
import { ToolCall, ToolCallField, ToolCallGroup, type ToolStatus } from './arc-x/tool-call/tool-call';
import { TurnFooter } from './arc-x/turn-footer/turn-footer';
import { WaitLine } from './arc-x/wait-line/wait-line';
import { Alert } from './arc/alert/alert';
import { Button } from './arc/button/button';
import { CopyButton } from './arc/copy-button/copy-button';
import { ExpandableText } from './arc/expandable-text/expandable-text';
import { AllowedCard, HelperMarkLists, MarkLine } from './chat-permission-marks';
import { MessageText, SENT_TEXT_STYLE } from './command-text';
import { CompactionMark } from './compaction-mark';
import { RecapRow } from './history-mark';
import { NoteRow } from './note-row';
import { NoticeRow } from './notice-row';
import { PdfUnreadRow } from './pdf-unread-row';
import { PlanModeLine } from './plan-mode-line';
import { RuleRow } from './rule-row';
import { type HelperMarks, RunBlock } from './run-block';
import { RunEndRow } from './run-end-row';
import { freshOffset, toolStatus } from './transcript';
import { UnloadButton, type UnloadControl } from './unload-button';
import { useNow, WaitClock } from './wait-clock';

/** What a turn's action asks the page to do. `text` is the message to send. */
export type TurnAction =
  | { kind: 'send-again'; text: string }
  | { kind: 'continue'; text: string }
  | { kind: 'update-agent' }
  | { kind: 'pick-model' };

export interface ChatThreadProps {
  /** The open conversation: outcomes this thread saw end belong to it alone. */
  conversationId?: string | null;
  mode: ChatMode;
  entries: readonly TranscriptEntry[];
  turns: readonly TurnMetric[];
  /** The running turn, or null. */
  turn: TurnState | null;
  /** How the last turn of this stream ended (`ConversationState.lastEnd`). */
  lastEnd: ConversationState['lastEnd'];
  /** The create call is in flight: the pending row says "Starting…". */
  creating?: boolean;
  /**
   * For the pending row's engine words and the engine note under the last end line; `appEnds` for the line of a turn
   * that a Control Panel restart stopped.
   */
  host?:
    | (Pick<ChatHostState, 'inference' | 'loading' | 'resident' | 'answers'> &
        Partial<Pick<ChatHostState, 'appEnds' | 'requests'>>)
    | null;
  /** Size on disk of a model, null when unknown: the load card prints it. */
  modelSize?: (modelId: string) => number | null;
  /** The Unload action of the engine note, for the conversation's model. None: the note has no button. */
  unload?: UnloadControl;
  /** Arrival time of the last stream event: the thinking clock and the quiet rule follow it. */
  asOfMs?: number;
  /** The conversation's model: the stats tooltip, the load card and the engine note. */
  model?: string | null;
  transcriptError?: string | null;
  /** Where "Inspect" goes from the transcript error. */
  inspectHref?: string;
  onTurnAction?: (action: TurnAction) => void;
  /** The message text whose files the page still holds for this conversation: Send again sends them with it. */
  keptFor?: string | null;
  /** Why the turn actions are off now (the engine is stopped). */
  actionDisabledReason?: string;
  /** False hides the "Update the agent" turn action when in-app installation is unavailable. */
  canInstallAgent?: boolean;
  /** One status sentence for the page's polite live region. */
  onStatus?: (sentence: string) => void;
  /** The message column: `useStickToBottom` follows its size. */
  contentRef?: (node: HTMLElement | null) => void;
  /** Work: the conversation's folder; paths inside it read relative. */
  folder?: string | null;
  /** Work: the temp folder Auto covers, as people type it; named on the calls and marks that used it. */
  tempRoot?: string | null;
  /** The open request of the agent, drawn after the newest entry (the tool row it waits on). */
  card?: ReactNode;
  /** A request of the agent is open: the turn waits for the person, so no wait line shows. */
  asking?: boolean;
  /** Work: the tool call ids whose request is open now; their rows open by themselves (pass one Set while it holds the same ids). */
  askCalls?: ReadonlySet<string>;
  /** Work: the permission marks of the conversation, oldest first. */
  marks?: readonly PermissionMark[];
  /** The agent compacts the conversation now: its line takes the place of the wait line. */
  compaction?: CompactionState | null;
  /** The agent waits on a batch of hooks: its line shows at the thread end once the batch is 1 s old. */
  hookRun?: HookRunState | null;
  /** A recap the user asked for is being written: its line shows at the thread end while no answer runs. */
  recap?: RecapState | null;
  /** The running answer's last retry episode (`ConversationState.retry`): an open one is the retry line at the end. */
  retry?: RetryLive | null;
  /** Work: the host's runs; a run block whose run is not live there reads "Ended early". */
  runs?: readonly RunView[];
  /** Work: the open requests of the conversation's agents (run agents and subagents), oldest first. */
  agentAsks?: readonly AgentAsk[];
  /** Work: Stop run; rejects with the reason to show. */
  onStopRun?: (runId: string) => Promise<void>;
  /** Work: what an agent's buttons and its row do (Stop, Review, its Agent view); absent: rows link to Sessions. */
  agentHandlers?: AgentHandlers;
  /** Work: the agent whose view is open (any attempt's id): its row reads in accent ink. */
  openAgent?: string | null;
  /** Work: the view asked this row to open its permission lists. */
  agentMarksOpen?: AgentMarksOpen | null;
  /** Work: the host's live facts (agent progress). */
  live?: LiveActivity;
  /** Work, while the conversation's model sees images: Read it now of a PDF sent to the PDF tool. */
  onReadPdf?: (id: string) => Promise<void>;
  /**
   * Set to the thread's jump: it mounts the turn of entry `at` first (older turns wait behind a button), scrolls to it
   * and flashes it; a run's block opens and its head takes focus.
   */
  jumpRef?: { current: ((at: number) => void) | null };
  /**
   * The conversation is loaded in a running agent process. False: a helper with no end died with the process and reads
   * "not finished". Default true.
   */
  agentLive?: boolean;
  /** Rewind to here on each sent message; none: the message has Copy only. */
  rewind?: MessageRewind;
  /** The first line of the thread, above every message (a fork's origin). */
  top?: ReactNode;
  /** Work: removes a never-allow rule (the "Your rule" row); rejects when it was not removed. */
  onRemoveRule?: (ruleId: string, words: string) => Promise<void>;
}

/**
 * The `⋯` of a sent message: Rewind to here, and Fork with this answer when `onFork` is given (both have the same walls,
 * so one `off`). Keep one object while its fields stay the same.
 */
export interface MessageRewind {
  /** Why it is off now (the row stays focusable and reads it); null when it may run. */
  off: string | null;
  /** Why Rewind to the message at entry `at` alone is off (a goal began after it), or null. */
  offAt?: (at: number) => string | null;
  /** The message at entry `at` (its index in the conversation's entries); `from`: the `⋯` button. */
  onRewind: (at: number, from: HTMLElement | null) => void;
  /** Fork with the answer of the message at entry `at`. */
  onFork?: (at: number) => void;
}

interface ThreadRuns {
  runs: readonly RunView[];
  onStop: ((runId: string) => Promise<void>) | undefined;
  live: LiveActivity | undefined;
  agentLive: boolean;
  /** The polite region. */
  announce: ((sentence: string) => void) | undefined;
  /** The engine's requests now: a PDF reader's row reads its own. */
  requests: readonly EngineRequest[];
}

const NO_RUNS: ThreadRuns = {
  runs: [],
  onStop: undefined,
  live: undefined,
  agentLive: true,
  announce: undefined,
  requests: [],
};

/**
 * The PDF tool in the thread: the names of the PDFs messages sent to it, the ids its runs read, whether an answer runs
 * now, and Read it now (Work, while the model sees images).
 */
interface ThreadPdf {
  names: ReadonlyMap<string, string>;
  started: ReadonlySet<string>;
  running: boolean;
  session: string | null;
  onRead: ((id: string) => Promise<void>) | undefined;
}

const PdfContext = createContext<ThreadPdf>({
  names: new Map(),
  started: new Set(),
  running: false,
  session: null,
  onRead: undefined,
});

/**
 * The live facts every agent row reads (progress, stopping, messages, stopped by you), its asks and its buttons; the
 * agent whose view is open, and the row whose permission lists the view asked to open (`tick` moves each time).
 */
interface ThreadAgents {
  live: LiveActivity | undefined;
  asks: readonly AgentAsk[];
  handlers: AgentHandlers | undefined;
  openId: string | null;
  marksOpen: AgentMarksOpen | null;
}

/** The view's permission counts asked the row of `id` (any attempt) to open its lists. */
export interface AgentMarksOpen {
  id: string;
  tick: number;
}

const NO_AGENTS: ThreadAgents = { live: undefined, asks: [], handlers: undefined, openId: null, marksOpen: null };

/** Work: the never-allow rules the marks name and still hold, by their words, and how to remove one. */
interface ThreadRules {
  ids: ReadonlyMap<string, string>;
  onRemove: ((ruleId: string, words: string) => Promise<void>) | undefined;
}

const RulesContext = createContext<ThreadRules>({ ids: new Map(), onRemove: undefined });

/** The rule's words to its id, from the rule marks whose rule is still kept; the newest mark wins. */
function ruleIds(marks: readonly PermissionMark[] | undefined): ReadonlyMap<string, string> {
  const ids = new Map<string, string>();
  for (const mark of marks ?? []) {
    if (mark.kind === 'denied' && mark.ruleId !== undefined && typeof mark.words === 'string') {
      ids.set(mark.words, mark.ruleId);
    }
  }
  return ids;
}

/** The "Your rule" row of a call a never-allow rule denied; Remove rule while the rule is kept. */
function ThreadRuleRow({ words, ts }: { words: string; ts?: number }): ReactElement {
  const { ids, onRemove } = useContext(RulesContext);
  const [removing, setRemoving] = useState(false);
  const id = ids.get(words);
  const remove =
    id === undefined || onRemove === undefined
      ? undefined
      : () => {
          if (removing) return;
          setRemoving(true);
          onRemove(id, words).then(
            () => setRemoving(false),
            () => setRemoving(false),
          );
        };
  return <RuleRow words={words} ts={ts} onRemove={remove} removing={removing} />;
}

/** The host's wake reports by wake `ts` (live only); read by the wake rows alone. */
const WakeReportsContext = createContext<ReadonlyMap<number, readonly WakeTask[]>>(new Map());
/**
 * Read by each run block, agent row and tool row whose task may still run, so a finished turn does not re-render when
 * a run's host state or an agent's progress changes.
 */
const RunsContext = createContext<ThreadRuns>(NO_RUNS);
/** Read by every agent row and run block: an agent's live state reaches a finished turn only through it. */
const AgentsContext = createContext<ThreadAgents>(NO_AGENTS);
/** The agents of each run, by run id ({@link runAgents}); read by the run blocks alone. */
const RunHelpersContext = createContext<ReadonlyMap<string, readonly AgentRowData[]>>(new Map());
/** The Task subagents' rows, by child id ({@link subagentRows}); read by their rows alone. */
const SubagentRowsContext = createContext<ReadonlyMap<string, AgentRowData>>(new Map());
const NO_RUNS_AT: RunsAt = { runs: new Map(), blocks: new Map(), dataOnly: new Map(), names: new Map() };
/** Where each run's block stands and what it reads ({@link runsAt}). */
const RunsAtContext = createContext<RunsAt>(NO_RUNS_AT);

interface ThreadJump {
  to: ((at: number) => void) | undefined;
  /** The last jump: a block drawn at `at` opens and focuses its head when `tick` moves. */
  target: { at: number; tick: number } | null;
}

const JumpContext = createContext<ThreadJump>({ to: undefined, target: null });

/**
 * Marks the card of entry `at` (its index in the conversation's entries) for the activity rail's jump. It draws no box
 * of its own.
 */
function EntryAnchor({ at, children }: { at: number; children: ReactNode }): ReactElement {
  return (
    <div data-entry-at={at} className="contents">
      {children}
    </div>
  );
}

/** The permission marks of a row's attempts, from its turn's marks. */
function marksOfRow(marks: TurnMarks | undefined): ((row: AgentRowData) => HelperMarks) | undefined {
  return marks === undefined
    ? undefined
    : (row) => ({
        allowed: row.at.flatMap((at) => marks.subagent.get(at) ?? []),
        denied: row.at.flatMap((at) => marks.deniedSubagent.get(at) ?? []),
      });
}

/** A run's block in the thread: everything that changes while it runs comes through context. */
function ThreadRunBlock(props: {
  runId: string;
  marks: TurnMarks | undefined;
  /** The words of the mark a mode left on its launch call. */
  launchMark: string | null;
  temp: string | null;
}): ReactElement | null {
  const { runId } = props;
  const { runs, onStop, agentLive, announce, requests } = useContext(RunsContext);
  const { live, asks, handlers, openId, marksOpen } = useContext(AgentsContext);
  const facts = useContext(RunsAtContext).runs.get(runId);
  const agents = useContext(RunHelpersContext).get(runId) ?? [];
  const jump = useContext(JumpContext);
  if (facts === undefined) return null;
  const view = runs.find((run) => run.runId === runId);
  const name = facts.entry?.name ?? null;
  const target = jump.target;
  const focusTick = target !== null && (target.at === facts.place.at || target.at === facts.entryAt) ? target.tick : 0;
  return (
    <RunBlock
      runId={runId}
      entry={facts.entry}
      agents={agents}
      live
      view={view}
      sessionLive={agentLive}
      by={view !== undefined ? view.by : facts.by}
      wakes={facts.wakes}
      resultAt={facts.resultAt}
      onJump={jump.to}
      launch={facts.launch}
      launchMark={props.launchMark}
      asks={name === null ? [] : asks.filter((ask) => ask.run === name)}
      progress={live?.helpers}
      onStop={onStop === undefined ? undefined : () => onStop(runId)}
      agentControls={handlers === undefined ? undefined : { live, handlers, openId, marksOpen }}
      marksOf={marksOfRow(props.marks)}
      temp={props.temp}
      onStatus={announce}
      focusTick={focusTick}
      pdf={facts.pdf}
      requests={requests}
    />
  );
}

/** The end row of a run's wake, with its link back to the run's block. */
function ThreadRunEndRow({ entry }: { entry: Entry<'wake'> }): ReactElement | null {
  const runsAtNow = useContext(RunsAtContext);
  const jump = useContext(JumpContext);
  const runId = entry.workflow?.runId;
  const facts = runId === undefined ? undefined : runsAtNow.runs.get(runId);
  const pdf = facts?.pdf ?? null;
  return (
    <RunEndRow
      entry={entry}
      runAt={facts?.place.at ?? null}
      onJump={jump.to}
      {...(pdf === null ? {} : { name: pdfRunName(pdf) })}
    />
  );
}

/** The main agent's `workflow` call that is no launch: a check, a launch that did not start, a pause, stop or resume. */
function WorkflowToolRow({ entry, form }: { entry: Entry<'tool'>; form: Exclude<WorkflowForm, { form: 'block' }> }) {
  const runsAtNow = useContext(RunsAtContext);
  const jump = useContext(JumpContext);
  const icon = <Workflow size={14} strokeWidth={1.75} />;
  if (form.kind === 'control') {
    const place = form.runId === null ? undefined : runsAtNow.runs.get(form.runId)?.place;
    return (
      <EventRow
        icon={icon}
        label={form.title}
        text=""
        action={
          place !== undefined && jump.to !== undefined ? (
            <Button type="button" size="sm" variant="ghost" onClick={() => jump.to?.(place.at)}>
              {CHAT.run.links.showRun}
            </Button>
          ) : undefined
        }
        ts={entry.ts}
      />
    );
  }
  return (
    <ToolCall
      name={form.title}
      nameStyle="word"
      icon={icon}
      status={form.kind === 'checked' ? 'done' : 'failed'}
      detail={form.kind === 'checked' ? undefined : (form.error ?? undefined)}
      durationMs={durationOf(entry)}
      ts={entry.ts}
      input={entry.input}
      output={entry.output}
      truncated={entry.truncated}
      defaultOpen={false}
    />
  );
}

type Entry<K extends TranscriptEntry['kind']> = Extract<TranscriptEntry, { kind: K }>;

/** Turns mounted at once; older ones wait behind a button. */
export const MOUNTED_TURNS = 40;
/** How long a jump's target keeps its flash. */
const ENTRY_FLASH_MS = 1_200;

/** The first box a jump can scroll to inside an anchor (anchors draw no box of their own), else above it. */
function entryBox(target: Element | null): Element | null {
  let box: Element | null = target;
  while (box !== null && box.getClientRects().length === 0) box = box.firstElementChild;
  if (box !== null) return box;
  box = target?.parentElement ?? null;
  while (box !== null && box.getClientRects().length === 0) box = box.parentElement;
  return box;
}
/** The newest turns that always render; older ones may skip painting while off screen. */
const ALWAYS_PAINTED = 2;
const QUIET_MS = 3_000;
const OFFSCREEN = { contentVisibility: 'auto', containIntrinsicSize: 'auto 160px' } as const;

/**
 * A run's agents are drawn in its block, and a `workflow` entry whose block stands at its launch call is data only. A
 * run that only spawn lines name draws its block at the first of them.
 */
function isDrawn(entry: TranscriptEntry, at: number, mode: ChatMode, places: RunsAt): boolean {
  if (entry.kind === 'subagent') return entry.run === undefined || places.blocks.get(at) === entry.run;
  if (entry.kind === 'workflow') return !places.dataOnly.has(at);
  return !(mode === 'chat' && entry.kind === 'tool' && entry.name === 'search_tool');
}

function durationOf(entry: { ts: number; endTs?: number }): number | null {
  return entry.endTs !== undefined && entry.ts > 0 ? entry.endTs - entry.ts : null;
}

function webTitle(call: WebCall): string {
  return call.kind === 'search' ? `"${oneLine(call.query)}"` : call.host;
}

function WebBody({ result, raw }: { result: WebResult | null; raw: string | undefined }): ReactNode {
  if (result === null)
    return raw === undefined ? undefined : <CodePanel code={raw} label={CHAT.output} maxLines={8} wrap />;
  if (result.kind === 'error') {
    return (
      <div className="grid gap-2">
        <p className="text-body text-secondary">{webErrorText(result.code)}</p>
        {errorNeedsSettings(result.code) ? (
          <span>
            <ButtonLink to="/providers" size="sm" variant="ghost">
              {CHAT.webSettings}
            </ButtonLink>
          </span>
        ) : null}
      </div>
    );
  }
  if (result.kind === 'search') {
    const sources: SourceItem[] = [];
    for (const item of result.results) {
      const url = httpUrl(item.url);
      if (url === null) continue;
      const host = url.hostname.startsWith('www.') ? url.hostname.slice(4) : url.hostname;
      sources.push({ url: item.url, host, title: item.title, date: item.date });
    }
    return (
      <div className="grid gap-1">
        <p className="text-caption text-secondary">
          {fill(CHAT.resultsBy, { provider: providerName(result.provider), n: result.results.length })}
        </p>
        <SourceList sources={sources} layout="list" />
      </div>
    );
  }
  const url = httpUrl(result.url);
  return (
    <div className="grid gap-2">
      <KeyValue
        layout="rows"
        items={[
          {
            key: 'address',
            label: CHAT.address,
            value:
              url === null ? (
                result.url
              ) : (
                <a className="link" href={url.href} title={result.url} target="_blank" rel="noreferrer">
                  {result.url}
                </a>
              ),
          },
          { key: 'title', label: CHAT.pageTitle, value: result.title },
          { key: 'provider', label: CHAT.provider, value: providerName(result.provider) },
        ]}
      />
      <CodePanel code={result.text} label={CHAT.pageTitle} maxLines={8} wrap />
    </div>
  );
}

/** A web call as "Searched" / "Read", for both call shapes. */
function WebRow(props: {
  entry: Entry<'tool'>;
  call: WebCall;
  result: WebResult | null;
  live: boolean;
  /** Why the app refused this call, when it did. */
  refusal: Refusal | null;
}): ReactElement {
  const { entry, call, result } = props;
  const mapped = toolStatus(entry.status);
  const search = call.kind === 'search';
  const words = search
    ? { running: CHAT.searching, done: CHAT.searched, failed: CHAT.searchFailed }
    : { running: CHAT.reading, done: CHAT.read, failed: CHAT.readFailed };
  const icon = search ? <Search size={14} strokeWidth={1.75} /> : <FileText size={14} strokeWidth={1.75} />;
  let status: ToolStatus = mapped.status;
  let name: string = words.done;
  let detail: string | undefined;
  let body: ReactNode = undefined;
  if (mapped.status === 'running' || mapped.status === 'pending') {
    name = words.running;
  } else if (entry.blocked !== undefined) {
    status = 'failed';
    name = CHAT.hook.blockedWord;
    detail = CHAT.hook.blockedBy;
    body = <BlockedText text={entry.blocked} />;
  } else if (result !== null && result.kind === 'error') {
    status = 'failed';
    name = words.failed;
    detail = webErrorText(result.code);
    body = <WebBody result={result} raw={entry.output} />;
  } else if (mapped.status === 'failed' || mapped.status === 'other') {
    status = 'failed';
    name = words.failed;
    detail = props.refusal === null ? CHAT.webErrors['provider-error'] : CHAT.refusedCall[props.refusal];
  } else if (result === null) {
    body = <WebBody result={null} raw={entry.output} />;
  } else if (result.kind === 'search') {
    detail = resultCountText(result.results.length);
    body = <WebBody result={result} raw={entry.output} />;
  } else if (result.kind === 'page') {
    detail = result.cut ? CHAT.cut : undefined;
    body = <WebBody result={result} raw={entry.output} />;
  }
  return (
    <ToolCall
      name={name}
      nameStyle="word"
      icon={icon}
      title={webTitle(call)}
      status={status}
      detail={detail}
      durationMs={status === 'running' || status === 'pending' ? null : durationOf(entry)}
      ts={entry.ts}
      body={body}
      defaultOpen={false}
      live={props.live && status === 'running'}
    />
  );
}

const WORK_ICONS: Record<WorkCall['kind'], ReactNode> = {
  run: <SquareTerminal size={14} strokeWidth={1.75} />,
  edit: <FilePen size={14} strokeWidth={1.75} />,
  read: <FileText size={14} strokeWidth={1.75} />,
  list: <Folder size={14} strokeWidth={1.75} />,
  grep: <Search size={14} strokeWidth={1.75} />,
};

/** Characters of a stop reason the agent named, on a command row. */
const SIGNAL_CHARS = 40;

/** How a command ended, when it did not end with exit code 0: a timeout, a stop the agent named, or the exit code. */
function exitLine(entry: Entry<'tool'>): string | undefined {
  if (entry.timedOut === true) return CHAT.exitTimedOut;
  if (entry.signal !== undefined && entry.signal !== '') {
    return fill(CHAT.exitSignal, { signal: truncateMiddle(oneLine(entry.signal), SIGNAL_CHARS) });
  }
  if (entry.exitCode !== undefined) return fill(CHAT.exitCode, { code: entry.exitCode });
  return undefined;
}

/**
 * Where the background command or monitor this call started is now, with a running monitor's event count; undefined
 * for a call that started none. A task with no end whose agent process is gone ended with it.
 */
function taskWord(entry: Entry<'tool'>, agentLive: boolean, live: TaskLive | null): string | undefined {
  const task = entry.task;
  if (task === undefined) return undefined;
  if (task.killed === true) return CHAT.task.stopped;
  if (task.endTs !== undefined || !agentLive) return CHAT.task.ended;
  return live !== null && live.events > 0
    ? `${CHAT.task.inBackground} · ${eventsWords(live.events)}`
    : CHAT.task.inBackground;
}

interface TaskState {
  agentLive: boolean;
  /** The live facts of the task while it runs in the running agent; null otherwise. */
  live: TaskLive | null;
}

const NO_TASK: TaskState = { agentLive: true, live: null };

/**
 * The agent's state and the live facts of the task this call started. Only a call whose task may still run reads the
 * thread's live state; any other row that read it would re-render on each live update.
 */
function useTaskLive(entry: Entry<'tool'>): TaskState {
  const task = entry.task;
  if (task === undefined || task.endTs !== undefined || task.killed === true) return NO_TASK;
  const { live, agentLive } = use(RunsContext);
  return { agentLive, live: agentLive ? taskLiveOf(live, task.id) : null };
}

/** A tool row, with a running monitor's events under it. The row keeps its place, so it is not mounted again. */
function withEvents(row: ReactElement, task: TaskLive | null): ReactElement {
  return (
    <div className="grid min-w-0 gap-1">
      {row}
      {task !== null && task.events > 0 && task.recent.length > 0 ? <TaskEvents task={task} /> : null}
    </div>
  );
}

/** A row's detail: what a mode allowed or that the call was denied with a note, then the row's own fact. */
function allowedDetail(
  exit: string | undefined,
  mark: AllowedMark | DeniedMark | undefined,
  temp: string | null,
): string | undefined {
  if (mark === undefined) return exit;
  const word = mark.kind === 'denied' ? deniedWord(mark) : allowedWord(mark, temp);
  return exit === undefined ? word : `${word} · ${exit}`;
}

interface TaskRowView {
  row: AgentRowData;
  words: RowWords;
  to: string | undefined;
  /** Work: the row opens its Agent view. */
  onOpen: ((opener: HTMLElement) => void) | undefined;
  open: boolean;
  /** Its permission lists open: the tick of the view's ask. */
  marksTick: number;
  kind: string;
  /** It runs in the agent process now. */
  running: boolean;
  /** Its buttons, with what they do; null on a thread with no handlers. */
  controls: ReactNode;
}

/** What a Task subagent's row reads now: the same row as a workflow agent's, with its kind on the second line. */
function taskRowView(
  entry: Entry<'subagent'>,
  row: AgentRowData,
  runs: ThreadRuns,
  agents: ThreadAgents,
  now: number,
): TaskRowView {
  const ended = entry.endTs !== undefined;
  const running = !ended && runs.agentLive;
  const progress = running ? progressOf(agents.live, entry.childId) : null;
  const held = agentLiveOf(agents.live, row);
  const ask = agents.asks.find((item) => item.run === null && row.attempts.includes(item.child)) ?? null;
  const state = agentState(row, { live: runs.agentLive, now });
  const words = rowWords(state, {
    progress,
    now,
    ask: ask === null ? null : { title: ask.title, since: ask.openedAt },
    stopping: held?.stopping === true,
    stoppedByYou: stoppedByYou(agents.live, row),
  });
  const opens = transcriptOpens(entry.exists, progress, ended ? entry : undefined);
  const handlers = agents.handlers;
  const buttons = agentButtons({ workflow: false, state: state.state, held, askId: ask?.id ?? null });
  const target = { id: row.childId, title: row.label };
  return {
    row,
    words,
    to: opens ? transcriptPath(entry.childId) : undefined,
    onOpen: handlers === undefined ? undefined : (opener) => handlers.onOpen(target, opener, false),
    open: agents.openId !== null && row.attempts.includes(agents.openId),
    marksTick: agents.marksOpen !== null && row.attempts.includes(agents.marksOpen.id) ? agents.marksOpen.tick : 0,
    kind: entry.type !== '' ? `${CHAT.subagent.label} · ${entry.type}` : CHAT.subagent.label,
    running,
    controls:
      handlers === undefined || !anyButton(buttons) ? null : (
        <AgentControls
          id={row.childId}
          label={row.label}
          workflow={false}
          buttons={buttons}
          onStop={() => handlers.onStop(target)}
          onReview={handlers.onReview}
          onMessage={(opener) => handlers.onOpen(target, opener, true)}
        />
      ),
  };
}

/**
 * A subagent's row; with calls a mode allowed for it or the user denied with a note, their counts on its second line
 * and their lines in closed disclosures under it.
 */
function TaskAgentRow(props: {
  view: TaskRowView;
  marks: readonly AllowedMark[] | undefined;
  denied: readonly DeniedMark[] | undefined;
  temp: string | null;
  pulse: boolean;
  ts?: number;
}): ReactElement {
  const { view } = props;
  const allowed = props.marks !== undefined && props.marks.length > 0 ? props.marks : null;
  const denied = props.denied !== undefined && props.denied.length > 0 ? props.denied : null;
  const more: string[] = [];
  if (allowed !== null) more.push(subagentAllowedText(allowed));
  if (denied !== null) more.push(subagentDeniedText(denied.length));
  return (
    <AgentRow
      id={view.row.childId}
      label={view.row.label}
      words={view.words}
      kind={view.kind}
      {...(view.onOpen !== undefined ? { onOpen: view.onOpen, open: view.open } : { to: view.to })}
      openLabel={fill(CHAT.agent.open, { label: view.row.label })}
      pulse={props.pulse && view.running}
      more={more}
      ts={props.ts}
      controls={view.controls}
    >
      {allowed !== null || denied !== null ? (
        <HelperMarkLists allowed={allowed} denied={denied} temp={props.temp} openTick={view.marksTick} />
      ) : null}
    </AgentRow>
  );
}

function ThreadSubagentRow(props: {
  entry: Entry<'subagent'>;
  marks: readonly AllowedMark[] | undefined;
  denied: readonly DeniedMark[] | undefined;
  temp: string | null;
}): ReactElement | null {
  const runs = useContext(RunsContext);
  const agents = useContext(AgentsContext);
  const row = useContext(SubagentRowsContext).get(props.entry.childId);
  const now = useNow(props.entry.endTs === undefined && runs.agentLive);
  if (row === undefined) return null;
  return (
    <TaskAgentRow
      view={taskRowView(props.entry, row, runs, agents, now)}
      marks={props.marks}
      denied={props.denied}
      temp={props.temp}
      pulse
      ts={props.entry.ts}
    />
  );
}

/** Task subagents this many in a row of one turn share one list. */
const SUBAGENT_LIST_FROM = 4;

/** Four or more subagents in a row: one list under `Subagents (8) · 3 running`. */
function ThreadSubagentList(props: {
  items: ReadonlyArray<{ entry: Entry<'subagent'>; at: number }>;
  marks: TurnMarks | undefined;
  temp: string | null;
}): ReactElement {
  const runs = useContext(RunsContext);
  const agents = useContext(AgentsContext);
  const rows = useContext(SubagentRowsContext);
  const now = useNow(runs.agentLive && props.items.some(({ entry }) => entry.endTs === undefined));
  const views: Array<{ view: TaskRowView; at: number }> = [];
  for (const { entry, at } of props.items) {
    const row = rows.get(entry.childId);
    if (row !== undefined) views.push({ view: taskRowView(entry, row, runs, agents, now), at });
  }
  const running = views.filter(({ view }) => view.running).length;
  const title = fill(CHAT.run.groups.subagents, { n: formatCount(views.length) });
  return (
    <AgentList
      label={title}
      groups={[
        {
          key: 'subagents',
          title,
          mode: 'open',
          note: running > 0 ? countWords({ ...runCounts([]), running }).join(' · ') : undefined,
          rows: views.map(({ view, at }) => ({
            id: view.row.childId,
            node: (
              <EntryAnchor at={at}>
                <TaskAgentRow
                  view={view}
                  marks={props.marks?.subagent.get(at)}
                  denied={props.marks?.deniedSubagent.get(at)}
                  temp={props.temp}
                  pulse={false}
                />
              </EntryAnchor>
            ),
          })),
        },
      ]}
    />
  );
}

/** An edit row's counts: "+4 −2", "New file · +40", or "Writes the whole file · +40" for a write before its result. */
function editDetail(entry: Entry<'tool'>): string | undefined {
  const edits = entry.edits;
  if (edits === undefined || edits.length === 0) return undefined;
  const { added, removed } = editCounts(edits);
  const plus = fill(CHAT.diff.addedOnly, { added: formatCount(added) });
  if (edits.every((edit) => edit.create === true)) return `${CHAT.diff.newFile} · ${plus}`;
  if (entry.name === 'write' && entry.status !== 'completed') return `${CHAT.diff.wholeFile} · ${plus}`;
  return fill(CHAT.diff.counts, { added: formatCount(added), removed: formatCount(removed) });
}

/** A read row's pictures: "1 picture · 412 KB", "3 of 12 pages · 1.4 MB". */
function readDetail(entry: Entry<'tool'>): string | undefined {
  const pictures = entry.pictures;
  return pictures === undefined || pictures.length === 0 ? undefined : picturesDetail(pictures, entry.pdfPages);
}

function WorkRow(props: {
  entry: Entry<'tool'>;
  call: WorkCall;
  live: boolean;
  /** The permission mode allowed this call: the row says so and shows the text its card would have shown. */
  allowed?: AllowedMark;
  /** The user denied this call with a note: the row says so. */
  denied?: DeniedMark;
  temp: string | null;
  folder: string | null;
  /** The call's request waits for the person: the row opens, so the change shows before Allow. */
  asking: boolean;
}): ReactElement {
  const { entry, call } = props;
  const { agentLive, live: taskLive } = useTaskLive(entry);
  const mapped = toolStatus(entry.status);
  const unfinished = !props.live && (mapped.status === 'running' || mapped.status === 'pending');
  const status: ToolStatus = unfinished ? 'other' : mapped.status;
  const [running, done, failed] = call.words;
  const blocked = entry.blocked;
  const name =
    blocked !== undefined
      ? CHAT.hook.blockedWord
      : status === 'running' || status === 'pending'
        ? running
        : status === 'failed'
          ? failed
          : done;
  const output = blocked === undefined && entry.output !== undefined && entry.output !== '' ? entry.output : null;
  const parts: ReactNode[] = [];
  const field = (key: string, label: string, code: string, maxLines?: number): ReactNode => (
    <ToolCallField key={key} label={label}>
      <CodePanel code={code} label={label} maxLines={maxLines} wrap />
    </ToolCallField>
  );
  const card = props.allowed?.card ?? null;
  if (card !== null) parts.push(<AllowedCard key="card" card={card} />);
  else if (call.command !== undefined) parts.push(field('command', CHAT.command, call.command));
  const edits = call.kind === 'edit' && entry.edits !== undefined && entry.edits.length > 0 ? entry.edits : null;
  if (edits !== null) {
    parts.push(<EditDiffs key="diff" edits={edits} state={diffStateOf(entry.status)} folder={props.folder} />);
    if (output !== null) parts.push(field('output', CHAT.output, output, 12));
  } else {
    // An edit shows the change as the agent wrote it; before it answers, the change its input asks for.
    const change = call.kind === 'edit' ? (output ?? call.change) : undefined;
    if (change !== undefined) parts.push(field('change', CHAT.change, change, 12));
    if (output !== null && call.kind !== 'edit') parts.push(field('output', CHAT.output, output, 12));
  }
  if (entry.pictures !== undefined && entry.pictures.length > 0) {
    parts.push(<PictureStrip key="pictures" call={entry.id} pictures={entry.pictures} path={call.title} />);
  }
  if (blocked !== undefined) parts.push(<BlockedText key="blocked" text={blocked} />);
  const ended = status === 'done' || status === 'failed';
  if (call.kind === 'run' && ended && output === null && blocked === undefined) {
    parts.push(
      <p key="output" className="m-0 text-body text-secondary">
        {CHAT.noOutput}
      </p>,
    );
  }
  return withEvents(
    <ToolCall
      name={name}
      nameStyle="word"
      icon={WORK_ICONS[call.kind]}
      title={call.title}
      detail={
        blocked !== undefined
          ? CHAT.hook.blockedBy
          : allowedDetail(
              call.kind === 'run'
                ? (taskWord(entry, agentLive, taskLive) ?? exitLine(entry))
                : call.kind === 'edit'
                  ? editDetail(entry)
                  : readDetail(entry),
              props.allowed ?? props.denied,
              props.temp,
            )
      }
      status={status}
      statusText={unfinished ? 'not finished' : undefined}
      durationMs={status === 'running' || status === 'pending' ? null : durationOf(entry)}
      ts={entry.ts}
      body={parts.length > 0 ? <div className="grid min-w-0 gap-2">{parts}</div> : undefined}
      truncated={entry.truncated}
      defaultOpen={false}
      openNow={props.asking}
      live={props.live && status === 'running'}
    />,
    taskLive,
  );
}

/** The agent's words on a call a hook denied: the hook's name and its reason, as plain text. */
function BlockedText({ text }: { text: string }): ReactElement {
  return (
    <p className="m-0 whitespace-pre-wrap text-body text-secondary [overflow-wrap:anywhere]" data-blocked="">
      {visibleText(text)}
    </p>
  );
}

/** The agent's request to leave plan mode, answered on the plan card. */
const PLAN_TOOL = 'exit_plan_mode';

function GenericRow({
  entry,
  live,
  refusal,
  allowed,
  denied,
  temp,
}: {
  entry: Entry<'tool'>;
  live: boolean;
  refusal: Refusal | null;
  /** The permission mode allowed this call: the row says so and shows the text its card would have shown. */
  allowed?: AllowedMark;
  /** The user denied this call with a note: the row says so. */
  denied?: DeniedMark;
  temp: string | null;
}): ReactElement {
  const { agentLive, live: taskLive } = useTaskLive(entry);
  const mapped = toolStatus(entry.status);
  const unfinished = !live && (mapped.status === 'running' || mapped.status === 'pending');
  const status: ToolStatus = unfinished ? 'other' : mapped.status;
  const output =
    refusal !== null
      ? CHAT.refusedCall[refusal]
      : entry.blocked !== undefined
        ? visibleText(entry.blocked)
        : entry.output;
  const card = allowed?.card ?? null;
  const body =
    card === null ? undefined : (
      <div className="grid min-w-0 gap-2">
        <AllowedCard card={card} />
        {output !== undefined && output !== '' ? (
          <ToolCallField label={CHAT.output}>
            <CodePanel code={output} label={CHAT.output} maxLines={12} wrap />
          </ToolCallField>
        ) : null}
      </div>
    );
  return withEvents(
    <ToolCall
      name={entry.name === PLAN_TOOL ? CHAT.planReview : entry.name}
      nameStyle={entry.name === PLAN_TOOL ? 'word' : undefined}
      title={entry.name === PLAN_TOOL && entry.title === PLAN_TOOL ? undefined : oneLine(entry.title)}
      status={status}
      statusText={unfinished ? 'not finished' : mapped.statusText}
      durationMs={durationOf(entry)}
      ts={entry.ts}
      detail={
        entry.blocked !== undefined
          ? CHAT.hook.blockedRow
          : allowedDetail(taskWord(entry, agentLive, taskLive), allowed ?? denied, temp)
      }
      input={entry.input}
      output={output}
      body={body}
      truncated={entry.truncated}
      live={live && status === 'running'}
    />,
    taskLive,
  );
}

/** A turn the agent started by itself: what woke it, with the monitors the host saw report before it. */
function WakeRow({ entry }: { entry: Entry<'wake'> }): ReactElement {
  const reports = useContext(WakeReportsContext);
  return (
    <EventRow
      icon={<BellRing size={14} strokeWidth={1.75} />}
      label="No prompt typed"
      text={wakeTitle(entry, wakeReportOf(entry, reports))}
      ts={entry.ts}
    />
  );
}

interface TurnViewProps {
  turn: ThreadTurn;
  mode: ChatMode;
  links: LinkRule;
  /** This turn runs now. */
  running: boolean;
  /** The running turn's newest thought or text still streams; false while the wait line stands for the turn. */
  tailLive: boolean;
  isLast: boolean;
  outcome: OutcomeLine | null;
  /** The live outcome of this turn when it is the last one and ended: its refused call says why. */
  endOutcome: TurnOutcome | undefined;
  /** This page saw the turn end by a Stop: a message sent into it may not have been read. */
  stopped: boolean;
  asOfMs: number | undefined;
  model: string | null;
  folder: string | null;
  tempRoot: string | null;
  offscreen: boolean;
  /** Entries from this index on arrived live: a user message among them rises in. */
  liveFrom: number;
  shownText: Map<number, string>;
  actionDisabledReason: string | undefined;
  canInstallAgent: boolean;
  onTurnAction: ((action: TurnAction) => void) | undefined;
  keptFor: string | null;
  unload: UnloadControl | undefined;
  turnMarks: TurnMarks | undefined;
  /** Work: the calls of this turn whose request is open now. */
  askCalls: ReadonlySet<string> | undefined;
  rewind: MessageRewind | undefined;
}

/**
 * The permission marks of one turn: lines before an entry (by its index in the conversation), allowed tool rows, and
 * the allowed calls of a subagent (by the index of its entry).
 */
interface TurnMarks {
  before: Map<number, PermissionMark[]>;
  allowed: Map<string, AllowedMark>;
  subagent: Map<number, AllowedMark[]>;
  /** Tool rows denied with a note, by tool call id; a subagent's calls denied with a note, by its entry's index. */
  denied: Map<string, DeniedMark>;
  deniedSubagent: Map<number, DeniedMark[]>;
  /** Changes when anything drawn from these marks changes. */
  key: string;
}

function markKey(mark: PermissionMark): string {
  if (mark.kind === 'denied') {
    const words = typeof mark.words === 'string' ? mark.words : '';
    return `d${mark.at}:${mark.toolCallId}:${mark.card === null ? 0 : 1}:${mark.by}:${words}`;
  }
  if (mark.kind === 'mcp') return `m${mark.at}:${mark.mode}:${mark.outcome}`;
  return mark.kind === 'allowed'
    ? `a${mark.at}:${mark.mode}:${mark.toolCallId}:${mark.card === null ? 0 : 1}:${mark.temp === true ? 1 : 0}`
    : `${mark.kind}${mark.at}`;
}

function sameEntries(a: readonly TranscriptEntry[], b: readonly TranscriptEntry[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) if (a[i] !== b[i]) return false;
  return true;
}

/** A finished turn re-renders only when its own entries, end or outcome change. */
function sameTurn(prev: TurnViewProps, next: TurnViewProps): boolean {
  if (next.running || prev.running) return false;
  return (
    prev.turn.start === next.turn.start &&
    sameEntries(prev.turn.entries, next.turn.entries) &&
    prev.turn.end === next.turn.end &&
    prev.turn.metric === next.turn.metric &&
    prev.isLast === next.isLast &&
    prev.outcome?.text === next.outcome?.text &&
    prev.outcome?.detail === next.outcome?.detail &&
    prev.outcome?.action === next.outcome?.action &&
    prev.outcome?.note?.text === next.outcome?.note?.text &&
    prev.outcome?.note?.action === next.outcome?.note?.action &&
    prev.unload === next.unload &&
    prev.endOutcome === next.endOutcome &&
    prev.stopped === next.stopped &&
    prev.links === next.links &&
    prev.mode === next.mode &&
    prev.folder === next.folder &&
    prev.tempRoot === next.tempRoot &&
    prev.offscreen === next.offscreen &&
    prev.liveFrom === next.liveFrom &&
    prev.actionDisabledReason === next.actionDisabledReason &&
    prev.canInstallAgent === next.canInstallAgent &&
    prev.onTurnAction === next.onTurnAction &&
    prev.keptFor === next.keptFor &&
    prev.turnMarks?.key === next.turnMarks?.key &&
    prev.askCalls === next.askCalls &&
    prev.rewind === next.rewind
  );
}

/** Under a message sent into a running answer: when it went, or that a Stop may have dropped it unread. */
function steerCaption(ts: number, stopped: boolean): string {
  if (stopped) return CHAT.queue.steerUnread;
  return ts > 0 ? `${CHAT.queue.steered} · ${formatClock(ts)}` : CHAT.queue.steered;
}

/** The turn's prompt was sent with files or pictures. */
function turnHasFiles(turn: ThreadTurn): boolean {
  const user = turn.entries.find((entry) => entry.kind === 'user');
  return user?.kind === 'user' && user.attachments !== undefined && user.attachments.length > 0;
}

const TurnView = memo(function TurnView(props: TurnViewProps): ReactElement | null {
  const { turn, mode, running } = props;
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const [focusUrl, setFocusUrl] = useState<string | null>(null);
  const rendered: Array<[number, string]> = [];
  useEffect(() => {
    for (const [index, text] of rendered) props.shownText.set(index, text);
  });

  const finished = !running;
  const sources = useMemo(
    () => (finished && mode === 'chat' ? turnSources(turn.entries) : []),
    [finished, mode, turn.entries],
  );
  const lastAnswer = useMemo(() => {
    for (let i = turn.entries.length - 1; i >= 0; i -= 1) if (turn.entries[i]!.kind === 'assistant') return i;
    return -1;
  }, [turn.entries]);
  const answerSplit = useMemo(() => {
    if (sources.length === 0 || lastAnswer === -1) return null;
    const entry = turn.entries[lastAnswer] as Entry<'assistant'>;
    const split = splitAnswerSources(entry.text);
    const cited = matchCitations(split.listed, sources);
    return cited.size > 0 ? { body: split.body, cited } : null;
  }, [sources, lastAnswer, turn.entries]);
  const citations = useMemo(() => {
    if (answerSplit === null) return undefined;
    const map = new Map<number, { title: string; host: string }>();
    for (const [n, source] of answerSplit.cited) map.set(n, { title: source.title, host: source.host });
    return map;
  }, [answerSplit]);
  const sourceItems = useMemo<SourceItem[]>(() => {
    const byUrl = new Map<string, number>();
    if (answerSplit !== null)
      for (const [n, source] of answerSplit.cited) if (!byUrl.has(source.url)) byUrl.set(source.url, n);
    return sources.map((source) => {
      const n = byUrl.get(source.url);
      return { url: source.url, host: source.host, title: source.title, date: source.date, cited: n !== undefined, n };
    });
  }, [sources, answerSplit]);
  const onCite = useMemo(
    () =>
      answerSplit === null
        ? undefined
        : (n: number): void => {
            const source = answerSplit.cited.get(n);
            setSourcesOpen(true);
            setFocusUrl(source === undefined ? null : source.url);
          },
    [answerSplit],
  );

  const refused = useMemo(() => refusedCall(turn.entries, props.endOutcome), [props.endOutcome, turn.entries]);
  const places = useContext(RunsAtContext);
  const pdfThread = useContext(PdfContext);
  const all = turn.entries.filter((entry, k) => isDrawn(entry, turn.start + k, mode, places));
  let cut = all.length;
  if (finished) while (cut > 1 && all[cut - 1]!.kind === 'model') cut -= 1;
  const drawn = all.slice(0, cut);
  const after = all.slice(cut) as Array<Entry<'model'>>;
  const tail = turn.entries[turn.entries.length - 1];
  const items: ReactNode[] = [];
  const before = props.turnMarks?.before;
  const linesDrawn = new Set<number>();
  const pushLines = (index: number): void => {
    const marks = before?.get(index);
    if (marks === undefined || linesDrawn.has(index)) return;
    linesDrawn.add(index);
    marks.forEach((mark, n) => items.push(<MarkLine key={`mark-${index}-${n}`} mark={mark} temp={props.tempRoot} />));
  };
  /** A run's block drawn at entry `at`: its `workflow` entry's lines attach to the block's foot. */
  const pushBlock = (at: number, runId: string, launch: Entry<'tool'> | null): void => {
    const facts = places.runs.get(runId);
    const mark = launch === null ? undefined : props.turnMarks?.allowed.get(launch.id);
    const block = (
      <ThreadRunBlock
        runId={runId}
        marks={props.turnMarks}
        launchMark={mark === undefined ? null : allowedWord(mark, props.tempRoot)}
        temp={props.tempRoot}
      />
    );
    const entryAt = facts?.entryAt ?? null;
    items.push(
      <EntryAnchor key={at} at={at}>
        {entryAt !== null && entryAt !== at ? <EntryAnchor at={entryAt}>{block}</EntryAnchor> : block}
      </EntryAnchor>,
    );
    if (entryAt !== null && entryAt !== at && entryAt >= turn.start && entryAt < turn.start + turn.entries.length)
      pushLines(entryAt);
  };
  /** The block a launch call stands for: a refused launch keeps its refused row. */
  const launchBlock = (tool: TranscriptEntry, at: number): string | undefined =>
    tool.kind === 'tool' && tool !== refused?.entry ? places.blocks.get(at) : undefined;
  let i = 0;
  while (i < drawn.length) {
    const entry = drawn[i]!;
    const at = turn.start + turn.entries.indexOf(entry);
    pushLines(at);
    const launchRun = mode === 'work' ? launchBlock(entry, at) : undefined;
    if (launchRun !== undefined && entry.kind === 'tool') {
      i += 1;
      pushBlock(at, launchRun, entry);
      continue;
    }
    if (entry.kind === 'tool' && entry.ask !== undefined) {
      i += 1;
      items.push(
        <EntryAnchor key={at} at={at}>
          <QuestionRow entry={entry} ask={entry.ask} live={running} waiting={props.askCalls?.has(entry.id) === true} />
        </EntryAnchor>,
      );
      continue;
    }
    if (entry.kind === 'tool') {
      const run: Array<Entry<'tool'>> = [];
      while (
        i < drawn.length &&
        drawn[i]!.kind === 'tool' &&
        (drawn[i] as Entry<'tool'>).ask === undefined &&
        (run.length === 0 || before?.has(turn.start + turn.entries.indexOf(drawn[i]!)) !== true) &&
        (mode !== 'work' || launchBlock(drawn[i]!, turn.start + turn.entries.indexOf(drawn[i]!)) === undefined)
      ) {
        run.push(drawn[i] as Entry<'tool'>);
        i += 1;
      }
      const rows = run.map((tool) => {
        const key = turn.start + turn.entries.indexOf(tool);
        const web = mode === 'chat' ? webCallOf(tool) : null;
        const work = mode === 'work' ? workCallOf(tool, props.folder) : null;
        const form = mode === 'work' && tool !== refused?.entry ? workflowFormOf(tool, places.names) : null;
        if (form !== null && form.form === 'row')
          return (
            <EntryAnchor key={key} at={key}>
              <WorkflowToolRow entry={tool} form={form} />
            </EntryAnchor>
          );
        if (work !== null)
          return (
            <EntryAnchor key={key} at={key}>
              <WorkRow
                entry={tool}
                call={work}
                live={running}
                allowed={props.turnMarks?.allowed.get(tool.id)}
                denied={props.turnMarks?.denied.get(tool.id)}
                temp={props.tempRoot}
                folder={props.folder}
                asking={props.askCalls?.has(tool.id) === true}
              />
            </EntryAnchor>
          );
        return (
          <EntryAnchor key={key} at={key}>
            {web !== null ? (
              <WebRow
                entry={tool}
                call={web.call}
                result={web.result}
                live={running}
                refusal={tool === refused?.entry ? refused.refusal : null}
              />
            ) : (
              <GenericRow
                entry={tool}
                live={running}
                refusal={tool === refused?.entry ? refused.refusal : null}
                allowed={props.turnMarks?.allowed.get(tool.id)}
                denied={props.turnMarks?.denied.get(tool.id)}
                temp={props.tempRoot}
              />
            )}
          </EntryAnchor>
        );
      });
      if (run.length === 1) items.push(rows[0]);
      else {
        const failed = run.filter((tool) => tool.status === 'failed').length;
        const busy = running ? run.filter((tool) => tool.status === 'in_progress').length : 0;
        items.push(
          <ToolCallGroup key={`tools-${at}`} count={run.length} failed={failed} running={busy} live={running}>
            {rows}
          </ToolCallGroup>,
        );
      }
      continue;
    }
    i += 1;
    switch (entry.kind) {
      case 'user':
        if (entry.note === true) {
          const rule = neverNoteWords(entry.text);
          items.push(
            rule === null ? (
              <NoteRow key={at} text={entry.text} ts={entry.ts} />
            ) : (
              <ThreadRuleRow key={at} words={rule} ts={entry.ts} />
            ),
          );
          break;
        }
        if (entry.continued === true) {
          items.push(
            <RuleLabel key={at}>
              <RotateCw size={12} strokeWidth={1.75} aria-hidden="true" />
              {CHAT.continued}
            </RuleLabel>,
          );
          break;
        }
        const started = hostStartWords(entry.text, pdfThread.names);
        if (started !== null) {
          items.push(
            <EntryAnchor key={at} at={at}>
              <EventRow icon={<FileText size={14} strokeWidth={1.75} />} label={started} text="" ts={entry.ts} />
            </EntryAnchor>,
          );
          break;
        }
        items.push(
          <Message
            key={at}
            role="user"
            variant="chat"
            ts={entry.ts}
            enter={at >= props.liveFrom}
            caption={entry.interjection === true ? steerCaption(entry.ts, props.stopped) : undefined}
            footer={
              props.rewind !== undefined && entry.interjection !== true ? (
                <MessageActions
                  text={entry.text}
                  copyLabel={CHAT.copyMessage}
                  menuLabel={CHAT.history.messageMenu}
                  rewind={{
                    label: CHAT.history.rewind,
                    note: CHAT.history.rewindNote,
                    off: props.rewind.off ?? props.rewind.offAt?.(at) ?? null,
                    onSelect: (from) => props.rewind?.onRewind(at, from),
                  }}
                  fork={
                    props.rewind.onFork === undefined
                      ? undefined
                      : {
                          label: CHAT.history.forkHere,
                          note: CHAT.history.forkHereNote,
                          off: props.rewind.off,
                          onSelect: () => props.rewind?.onFork?.(at),
                        }
                  }
                />
              ) : (
                <CopyButton value={entry.text} label={CHAT.copyMessage} iconOnly variant="plain" />
              )
            }
          >
            {entry.attachments !== undefined && entry.attachments.length > 0 ? (
              <div className="grid gap-2">
                {entry.text !== '' ? (
                  <ExpandableText lines={12} style={SENT_TEXT_STYLE}>
                    {entry.text}
                  </ExpandableText>
                ) : null}
                <AttachmentStrip readOnly items={sentChips(entry.attachments)} />
              </div>
            ) : (
              <MessageText text={entry.text} />
            )}
          </Message>,
        );
        break;
      case 'assistant': {
        const streaming = running && props.tailLive && entry === tail;
        if (streaming) rendered.push([at, entry.text]);
        const index = turn.entries.indexOf(entry);
        const text = answerSplit !== null && index === lastAnswer ? answerSplit.body : entry.text;
        items.push(
          <Message key={at} role="assistant" variant="chat" ts={entry.ts} streaming={streaming} status={CHAT.writing}>
            <ChatMarkdown
              text={text}
              streaming={streaming}
              links={props.links}
              citations={index === lastAnswer ? citations : undefined}
              onCite={index === lastAnswer ? onCite : undefined}
            />
          </Message>,
        );
        break;
      }
      case 'thinking': {
        const live = running && props.tailLive && entry === tail;
        if (live) rendered.push([at, entry.text]);
        items.push(
          <ReasoningBlock
            key={at}
            text={entry.text}
            label={live ? CHAT.thinking : CHAT.thought}
            durationMs={live ? null : thoughtDuration(entry)}
            live={live}
            startTs={live ? entry.ts : undefined}
            asOfMs={live ? props.asOfMs : undefined}
            freshFrom={live ? freshOffset(props.shownText.get(at), entry.text) : undefined}
            links={props.links}
          />,
        );
        break;
      }
      case 'wake':
        items.push(
          entry.cause === 'workflow' && entry.workflow !== undefined ? (
            <EntryAnchor key={at} at={at}>
              <ThreadRunEndRow entry={entry} />
            </EntryAnchor>
          ) : (
            <WakeRow key={at} entry={entry} />
          ),
        );
        break;
      case 'plan':
        items.push(
          <EntryAnchor key={at} at={at}>
            <PlanList items={entry.entries} ts={entry.ts} />
          </EntryAnchor>,
        );
        break;
      case 'subagent': {
        if (entry.run !== undefined) {
          pushBlock(at, entry.run, null);
          break;
        }
        const group: Array<{ entry: Entry<'subagent'>; at: number }> = [{ entry, at }];
        while (i < drawn.length) {
          const next = drawn[i]!;
          const nextAt = turn.start + turn.entries.indexOf(next);
          if (next.kind !== 'subagent' || next.run !== undefined || before?.has(nextAt) === true) break;
          group.push({ entry: next, at: nextAt });
          i += 1;
        }
        if (group.length >= SUBAGENT_LIST_FROM) {
          items.push(<ThreadSubagentList key={at} items={group} marks={props.turnMarks} temp={props.tempRoot} />);
          break;
        }
        for (const item of group)
          items.push(
            <EntryAnchor key={item.at} at={item.at}>
              <ThreadSubagentRow
                entry={item.entry}
                marks={props.turnMarks?.subagent.get(item.at)}
                denied={props.turnMarks?.deniedSubagent.get(item.at)}
                temp={props.tempRoot}
              />
            </EntryAnchor>,
          );
        break;
      }
      case 'model':
        items.push(
          <RuleLabel key={at}>
            <Cpu size={12} strokeWidth={1.75} aria-hidden="true" />
            {`Model switched to ${entry.modelId}`}
          </RuleLabel>,
        );
        break;
      case 'plan-mode':
        items.push(<PlanModeLine key={at} mode={entry.mode} />);
        break;
      case 'compaction':
        items.push(<CompactionMark key={at} entry={entry} />);
        break;
      case 'recap':
        items.push(<RecapRow key={at} entry={entry} />);
        break;
      case 'goal':
        items.push(
          <EntryAnchor key={at} at={at}>
            <GoalRow entry={entry} />
          </EntryAnchor>,
        );
        break;
      case 'workflow':
        pushBlock(at, entry.runId, null);
        break;
      case 'notice':
        // While its turn runs, an episode with no end is the retry line at the thread end, not a row.
        if (!(running && entry.code === 'retry' && entry.endTs === undefined))
          items.push(<NoticeRow key={at} entry={entry} running={running} />);
        break;
      case 'turn-end':
        break;
      default: {
        const kind = (entry as { kind: string }).kind;
        items.push(<EventRow key={at} label={kind} text="" ts={(entry as { ts?: number }).ts} />);
      }
    }
  }

  if (before !== undefined)
    for (let k = 0; k < turn.entries.length; k += 1) if (!linesDrawn.has(turn.start + k)) pushLines(turn.start + k);

  if (sourceItems.length > 0) {
    items.push(
      <SourceList
        key="sources"
        sources={sourceItems}
        label={answerSplit !== null ? CHAT.sources : CHAT.sourcesFound}
        open={sourcesOpen}
        onOpenChange={(open) => {
          setSourcesOpen(open);
          if (!open) setFocusUrl(null);
        }}
        focusUrl={focusUrl}
      />,
    );
  }

  if (finished && mode === 'work')
    for (const pdf of pdfUnread(turn.entries, 0, pdfThread.running, pdfThread.started, pdfThread.session))
      items.push(<PdfUnreadRow key={`pdf-${pdf.id}`} pdf={pdf} onRead={pdfThread.onRead} />);

  if (finished && (turn.end !== null || props.outcome !== null || turnPrompt(turn) !== null)) {
    const answer = answerMarkdown(turn.entries);
    const outcome = props.outcome;
    const prompt = turnPrompt(turn);
    let action: { label: string; onSelect: () => void; disabled?: boolean; disabledReason?: string } | undefined;
    const onTurnAction = props.onTurnAction;
    if (props.isLast && outcome !== null && outcome.action !== null && onTurnAction !== undefined) {
      const disabled =
        props.actionDisabledReason !== undefined &&
        outcome.action !== 'update-agent' &&
        outcome.action !== 'pick-model';
      const base = { disabled, disabledReason: props.actionDisabledReason };
      if (outcome.action === 'send-again' && prompt !== null) {
        const filesLost = turnHasFiles(turn) && props.keptFor !== prompt;
        action = {
          ...base,
          label: filesLost ? CHAT.sendAgainNoFiles : CHAT.sendAgain,
          onSelect: () => onTurnAction({ kind: 'send-again', text: prompt }),
        };
      } else if (outcome.action === 'continue')
        action = {
          ...base,
          label: CHAT.continueAnswer,
          onSelect: () => onTurnAction({ kind: 'continue', text: CHAT.continuePrompt }),
        };
      else if (outcome.action === 'update-agent' && props.canInstallAgent)
        action = { label: CHAT.updateAgent, onSelect: () => onTurnAction({ kind: 'update-agent' }) };
      else if (outcome.action === 'pick-model')
        action = { label: CHAT.pickModel, onSelect: () => onTurnAction({ kind: 'pick-model' }) };
    }
    items.push(
      <TurnFooter
        key="footer"
        stats={turnStats(turn.metric, turn.end)}
        statsTitle={turnStatsTitle(turn.metric, props.model)}
        outcome={
          outcome === null
            ? undefined
            : { tone: outcome.tone, text: outcome.text, detail: outcome.detail, glyph: outcome.glyph }
        }
        action={action}
        note={
          outcome?.note === undefined
            ? undefined
            : {
                text: outcome.note.text,
                action:
                  outcome.note.action === 'open-engine' ? (
                    <Link className="link" to="/engine">
                      {CHAT.engine.open}
                    </Link>
                  ) : outcome.note.action === 'unload' && props.unload !== undefined ? (
                    <UnloadButton control={props.unload} />
                  ) : undefined,
              }
        }
        copy={answer !== '' ? { value: answer, label: CHAT.copyAnswer } : undefined}
        quiet={!props.isLast}
      />,
    );
  }
  for (const entry of after) {
    items.push(
      <RuleLabel key={turn.start + turn.entries.indexOf(entry)}>
        <Cpu size={12} strokeWidth={1.75} aria-hidden="true" />
        {`Model switched to ${entry.modelId}`}
      </RuleLabel>,
    );
  }

  // A group of run helpers' spawns alone (a run's later step after its turn ended) draws nothing here.
  if (items.length === 0) return null;
  return (
    <div data-turn={turn.start} className="grid min-w-0 gap-3" style={props.offscreen ? OFFSCREEN : undefined}>
      {items}
    </div>
  );
}, sameTurn);

/** The prompt read of a wait line: done of total, or a busy track with no fill while nothing is counted yet. */
function ReadMeter({ progress }: { progress: ReadCounts }): ReactElement {
  const done = progress.done;
  if (done === undefined) return <Placeholder width="12rem" height={4} radius="pill" className="min-w-0" />;
  return (
    <Meter
      label={CHAT.pendingReadMeter}
      value={done}
      max={progress.total}
      formatValue={() => fill(CHAT.pendingReadCount, { done: formatCount(done), total: formatCount(progress.total) })}
      size="dense"
      warnAt={2}
      dangerAt={2}
      busy
      className="w-48 min-w-0"
    />
  );
}

function PendingRow(props: {
  creating: boolean;
  continuing: boolean;
  turn: TurnState | null;
  host: ChatThreadProps['host'];
  model: string | null;
  conversationId: string | null;
  /** No stream event for 3 s while the turn runs. */
  quiet: boolean;
  /** The turn has shown work. */
  work: boolean;
  /** Start of this wait (epoch ms): the clock counts from it; null shows no clock. */
  since: number | null;
  /** The PDF pages the turn's message carries. */
  pages: number;
  modelSize: ChatThreadProps['modelSize'];
  onStatus: ChatThreadProps['onStatus'];
}): ReactElement {
  const since = props.since;
  const phase = props.turn?.phase ?? null;
  const loading = loadFor(props.host?.loading ?? [], props.model);
  const answers = props.host?.answers ?? [];
  const row = pendingRow({
    creating: props.creating,
    phase,
    wait: props.turn?.wait ?? null,
    model: props.model,
    loading,
    inference: props.host?.inference ?? null,
    quiet: props.quiet,
    full: fullLimit(answers, props.model, props.turn?.startedAt ?? null),
    continuing: props.continuing,
    work: props.work,
    pages: props.pages,
    engine: engineWait({
      requests: props.host?.requests ?? [],
      session: props.conversationId,
      model: props.model,
      since,
      hidden: answers.find((entry) => entry.modelId === props.model)?.queued ?? 0,
    }),
  });
  const loadModel = row.kind === 'load' ? row.modelId : null;
  const variant = loadVariant(loadModel ?? '', loading, props.host?.resident ?? []);
  const loadStatus =
    loadModel === null
      ? null
      : variant === 'in-memory'
        ? CHAT.load.inMemory
        : fill(CHAT.load.status, { model: loadModel });
  const onStatus = props.onStatus;
  useEffect(() => {
    if (loadStatus !== null) onStatus?.(loadStatus);
  }, [loadStatus, onStatus]);
  if (row.kind === 'label') {
    const progress = row.progress;
    return (
      <WaitLine
        orb={row.orb}
        text={row.text}
        enter
        trail={
          <>
            {progress === undefined ? null : <ReadMeter progress={progress} />}
            {since === null || props.creating ? null : <WaitClock since={since} />}
          </>
        }
      />
    );
  }
  const loadSince =
    variant === 'loading' && loading !== null ? loading.startedAt : (props.turn?.startedAt ?? Date.now());
  return (
    <Message role="assistant" variant="chat" enter>
      <LoadCard
        modelId={row.modelId}
        variant={variant}
        other={loading?.modelId ?? null}
        since={loadSince}
        sizeBytes={props.modelSize?.(row.modelId) ?? null}
        progress={variant === 'loading' ? loading : null}
      />
    </Message>
  );
}

/** The thread end while the agent compacts: one wait line with the fold orb, and a note the first time. */
function CompactionRow(props: {
  compaction: CompactionState;
  requests: ChatHostState['requests'];
  conversationId: string | null;
  /** No earlier compaction in this conversation. */
  first: boolean;
  onStatus: ChatThreadProps['onStatus'];
}): ReactElement {
  const detail = compactionDetail({
    requests: props.requests,
    session: props.conversationId,
    since: props.compaction.startedAt,
  });
  const progress = detail?.progress;
  const onStatus = props.onStatus;
  useEffect(() => {
    onStatus?.(CHAT.compacting);
  }, [onStatus]);
  return (
    <div className="grid min-w-0 gap-1" data-compaction-line="">
      <WaitLine
        orb="fold"
        text={CHAT.compacting}
        enter
        trail={
          <>
            {detail === null ? null : <span className="min-w-0 truncate text-caption text-muted">{detail.text}</span>}
            {progress === undefined ? null : <ReadMeter progress={progress} />}
            <WaitClock since={props.compaction.startedAt} />
          </>
        }
      />
      {props.first ? (
        <p className="text-caption text-muted" style={{ paddingInlineStart: 'calc(20px + var(--space-2))' }}>
          {CHAT.compactingHelp}
        </p>
      ) : null}
    </div>
  );
}

/** The thread end while a recap is written: "Writing a recap…" and its clock; said once. */
function RecapWaitRow(props: { since: number; onStatus: ChatThreadProps['onStatus'] }): ReactElement {
  const onStatus = props.onStatus;
  useEffect(() => {
    onStatus?.(CHAT.history.recapSaid);
  }, [onStatus]);
  return (
    <div className="grid min-w-0" data-recap-line="">
      <WaitLine orb="think" text={CHAT.history.recapWriting} enter trail={<WaitClock since={props.since} />} />
    </div>
  );
}

type Ask = NonNullable<Entry<'tool'>['ask']>;

/** A question row's texts, cut: the question as the card takes it, an answer as the answer route takes own text. */
const ASK_QUESTION_CHARS = 1_000;
const ASK_ANSWER_CHARS = 4_000;
const ASK_SENTENCE_CHARS = 16_384;

/** Model text cut to `max` characters with "…", never inside a surrogate pair, then with hidden characters shown. */
function askText(text: string, max: number): string {
  if (text.length <= max) return visibleText(text);
  let end = max;
  const code = text.charCodeAt(end - 1);
  if (code >= 0xd800 && code <= 0xdbff) end -= 1;
  return `${visibleText(text.slice(0, end))}…`;
}

/** Where a question call ended, after "Asked 2 questions"; null when the row says nothing more. */
function questionState(entry: Entry<'tool'>, ask: Ask, live: boolean, waiting: boolean): string | null {
  if (waiting) return CHAT.ask.rowWaiting;
  if (entry.status === 'failed') return CHAT.ask.rowFailed;
  const result = ask.result;
  if (result === null) return live ? null : CHAT.ask.rowUnfinished;
  switch (result.kind) {
    case 'answered':
      return CHAT.ask.rowAnswered;
    case 'none':
      return CHAT.ask.rowNone;
    case 'discuss':
      return CHAT.ask.rowDiscuss;
    case 'plan-now':
      return CHAT.ask.rowPlanNow;
    case 'text':
      return null;
  }
}

/** One answer as the row shows it: the picks, then the user's own text; an answer read only as written stays so. */
function answerText(answer: Extract<Ask['result'], { kind: 'answered' }>['answers'][number]): string {
  if (answer === null) return CHAT.ask.noAnswer;
  if ('raw' in answer) return askText(answer.raw, ASK_ANSWER_CHARS);
  const picks = askText(answer.picks.join(', '), ASK_ANSWER_CHARS);
  if (answer.other === null) return picks;
  const other = askText(answer.other, ASK_ANSWER_CHARS);
  return picks === '' ? other : `${picks} · ${other}`;
}

/**
 * The agent's question call: what it asked and what came back. Drawn from the stored call alone, so it reads the same
 * live and after a restart; never folded into a run of tool rows. Every text is plain.
 */
function QuestionRow({
  entry,
  ask,
  live,
  waiting,
}: {
  entry: Entry<'tool'>;
  ask: Ask;
  live: boolean;
  /** Its card is open now. */
  waiting: boolean;
}): ReactElement {
  const n = ask.questions.length;
  const head = n === 1 ? CHAT.ask.row : fill(CHAT.ask.rowMany, { n });
  const state = questionState(entry, ask, live, waiting);
  const result = ask.result;
  return (
    <div className="min-w-0" data-question-row="" data-question-state={result?.kind ?? (waiting ? 'waiting' : 'open')}>
      <Disclosure
        title={
          <span className="flex min-w-0 items-center gap-2">
            <CircleHelp size={14} strokeWidth={1.75} aria-hidden="true" className="flex-none text-question" />
            <span className="min-w-0 truncate">
              <span className="text-question" data-question-head="">
                {head}
              </span>
              {state === null ? null : <span className="text-secondary">{` · ${state}`}</span>}
            </span>
          </span>
        }
      >
        <div className="grid min-w-0 gap-2 pb-2 ps-8 text-body">
          {result?.kind === 'answered' ? (
            <dl className="m-0 grid min-w-0 gap-2">
              {ask.questions.map((question, index) => (
                <div key={index} className="grid min-w-0 gap-0.5">
                  <dt className="whitespace-pre-wrap text-secondary [overflow-wrap:anywhere]">
                    {askText(question.text, ASK_QUESTION_CHARS)}
                  </dt>
                  <dd className="m-0 whitespace-pre-wrap [overflow-wrap:anywhere]">
                    {answerText(result.answers[index] ?? null)}
                  </dd>
                </div>
              ))}
            </dl>
          ) : (
            <ul className="m-0 grid min-w-0 list-none gap-1 p-0">
              {ask.questions.map((question, index) => (
                <li key={index} className="whitespace-pre-wrap text-secondary [overflow-wrap:anywhere]">
                  {askText(question.text, ASK_QUESTION_CHARS)}
                </li>
              ))}
            </ul>
          )}
          {result?.kind === 'text' ? (
            <p className="m-0 whitespace-pre-wrap [overflow-wrap:anywhere]" data-question-text="">
              {askText(result.text, ASK_SENTENCE_CHARS)}
            </p>
          ) : null}
        </div>
      </Disclosure>
    </div>
  );
}

/**
 * The thread end while the agent sends a model request again: what happened and the try, the engine's read of the new
 * request when it lists one, and the clock since the first try of this episode.
 */
function RetryRow(props: {
  entry: RetryNotice;
  requests: ChatHostState['requests'];
  conversationId: string | null;
}): ReactElement {
  const progress = retryReading({
    requests: props.requests,
    session: props.conversationId,
    since: props.entry.ts,
  });
  return (
    <div className="grid min-w-0" data-retry-line={props.entry.cause}>
      <WaitLine
        orb="hold"
        text={retryLine(props.entry)}
        enter
        trail={
          <>
            {progress === null ? null : <ReadMeter progress={progress} />}
            <WaitClock since={props.entry.ts} />
          </>
        }
      />
    </div>
  );
}

/** A hook batch shows at the thread end from this age: most hooks end well before it. */
const HOOK_WAIT_MS = 1_000;

/** True once `since` (epoch ms) is `ms` old; false while it is null. */
function useOlderThan(since: number | null, ms: number): boolean {
  const [, setTick] = useState(0);
  const old = since !== null && Date.now() - since >= ms;
  useEffect(() => {
    if (since === null || old) return;
    const timer = setTimeout(() => setTick((n) => n + 1), since + ms - Date.now());
    return () => clearTimeout(timer);
  }, [since, ms, old]);
  return old;
}

/** The thread end while the agent waits on a slow batch of hooks: how many and when, and the clock since it started. */
function HookRow(props: { hookRun: HookRunState; onStatus: ChatThreadProps['onStatus'] }): ReactElement {
  const { event, tool } = props.hookRun;
  const onStatus = props.onStatus;
  useEffect(() => {
    onStatus?.(fill(CHAT.hook.status, { when: hookWhen(event, tool) }));
  }, [onStatus, event, tool]);
  return (
    <div className="grid min-w-0" data-hook-line={props.hookRun.event}>
      <WaitLine
        orb="hold"
        text={hookWaitLine(props.hookRun)}
        enter
        trail={<WaitClock since={props.hookRun.startedAt} />}
      />
    </div>
  );
}

/** The marks of each turn, by the turn's first entry index. */
function marksByTurn(
  placed: ReturnType<typeof placeMarks>,
  turns: readonly ThreadTurn[],
  entries: readonly TranscriptEntry[],
  places: RunsAt,
): Map<number, TurnMarks> {
  const out = new Map<number, TurnMarks>();
  const turnOf = (index: number): ThreadTurn | undefined => {
    let lo = 0;
    let hi = turns.length - 1;
    let found: ThreadTurn | undefined;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (turns[mid]!.start <= index) {
        found = turns[mid];
        lo = mid + 1;
      } else hi = mid - 1;
    }
    return found;
  };
  const slot = (turn: ThreadTurn): TurnMarks => {
    let marks = out.get(turn.start);
    if (marks === undefined) {
      marks = {
        before: new Map(),
        allowed: new Map(),
        subagent: new Map(),
        denied: new Map(),
        deniedSubagent: new Map(),
        key: '',
      };
      out.set(turn.start, marks);
    }
    return marks;
  };
  for (const [index, list] of placed.before) {
    const turn = turnOf(index);
    if (turn === undefined) continue;
    const marks = slot(turn);
    marks.before.set(index, list);
    marks.key += `|${index}:${list.map(markKey).join(',')}`;
  }
  if (placed.allowed.size > 0) {
    entries.forEach((entry, index) => {
      if (entry.kind !== 'tool') return;
      const mark = placed.allowed.get(entry.id);
      const turn = mark === undefined ? undefined : turnOf(index);
      if (mark === undefined || turn === undefined) return;
      const marks = slot(turn);
      marks.allowed.set(entry.id, mark);
      marks.key += `|${markKey(mark)}`;
    });
  }
  // A run's agent is drawn in the run's block: its calls go to the block's turn.
  const drawnAt = (index: number): number => {
    const entry = entries[index];
    return entry?.kind === 'subagent' && entry.run !== undefined
      ? (places.runs.get(entry.run)?.place.at ?? index)
      : index;
  };
  if (placed.bySubagent.size > 0) {
    // A child with more than one entry lists its calls under the last one.
    const last = new Map<string, number>();
    entries.forEach((entry, index) => {
      if (entry.kind === 'subagent' && placed.bySubagent.has(entry.childId)) last.set(entry.childId, index);
    });
    for (const [child, index] of last) {
      const turn = turnOf(drawnAt(index));
      const list = placed.bySubagent.get(child);
      if (turn === undefined || list === undefined) continue;
      const marks = slot(turn);
      marks.subagent.set(index, list);
      marks.key += `|s${index}:${list.map(markKey).join(',')}`;
    }
  }
  if (placed.denied.size > 0) {
    entries.forEach((entry, index) => {
      if (entry.kind !== 'tool') return;
      const mark = placed.denied.get(entry.id);
      const turn = mark === undefined ? undefined : turnOf(index);
      if (mark === undefined || turn === undefined) return;
      const marks = slot(turn);
      marks.denied.set(entry.id, mark);
      marks.key += `|${markKey(mark)}`;
    });
  }
  if (placed.deniedBySubagent.size > 0) {
    const last = new Map<string, number>();
    entries.forEach((entry, index) => {
      if (entry.kind === 'subagent' && placed.deniedBySubagent.has(entry.childId)) last.set(entry.childId, index);
    });
    for (const [child, index] of last) {
      const turn = turnOf(drawnAt(index));
      const list = placed.deniedBySubagent.get(child);
      if (turn === undefined || list === undefined) continue;
      const marks = slot(turn);
      marks.deniedSubagent.set(index, list);
      marks.key += `|x${index}:${list.map(markKey).join(',')}`;
    }
  }
  return out;
}

/** The work of a turn has started: anything but its prompt, model switches and side entries (notices, notes). */
function hasWork(turn: ThreadTurn | undefined): boolean {
  return (
    turn !== undefined &&
    turn.entries.some(
      (entry) => !(entry.kind === 'user' || entry.kind === 'wake' || entry.kind === 'model' || isSideEntry(entry)),
    )
  );
}

/** Something of a running turn moves on screen: a drawn tool row runs, or text or a thought got an event in 3 s. */
function moving(turn: ThreadTurn, mode: ChatMode, quiet: boolean): boolean {
  for (const entry of turn.entries) {
    if (entry.kind !== 'tool' || (mode === 'chat' && entry.name === 'search_tool')) continue;
    const status = toolStatus(entry.status).status;
    if (status === 'running' || status === 'pending') return true;
  }
  const tail = turn.entries[turn.entries.length - 1];
  return !quiet && (tail?.kind === 'assistant' || tail?.kind === 'thinking');
}

/** The last group that holds a turn (a prompt, a wake or a stored end); the last group when none does. */
function lastTurnIndex(list: readonly ThreadTurn[]): number {
  for (let i = list.length - 1; i >= 0; i -= 1) {
    const turn = list[i]!;
    if (turn.end !== null || turnStartedAt(turn) !== null) return i;
  }
  return list.length - 1;
}

/** The newest time of a turn's steps (a step's end, else its start); null when none carries one. */
function lastStepAt(turn: ThreadTurn): number | null {
  let at = 0;
  for (const entry of turn.entries) {
    if (isSideEntry(entry)) continue;
    const end = 'endTs' in entry && entry.endTs !== undefined ? entry.endTs : entry.ts;
    if (end > at) at = end;
  }
  return at > 0 ? at : null;
}

/** Status sentence of a finished turn; `model`: the agent could not use the conversation's model. */
function endSentence(outcome: OutcomeLine | null, model: ModelSwitch | undefined): string {
  if (outcome === null) return CHAT.status.finished;
  if (model === undefined) return outcome.text;
  const head = modelHeadline(model);
  return `${outcome.text} ${head.endsWith('.') ? head : `${head}.`}`;
}

/**
 * A live conversation as a flat thread: each turn's prompt, thinking, tool rows, answer, sources and footer, in the
 * order they arrived. Only the running turn re-renders while text streams.
 */
export function ChatThread(props: ChatThreadProps): ReactElement {
  const { mode, entries, turns } = props;
  const running = props.turn !== null;
  const list = useMemo(() => splitTurns(entries, turns), [entries, turns]);
  // The rule keeps its identity while the hosts stay the same, so finished turns do not re-render on each event.
  const hostsKey = useMemo(
    () => (mode === 'chat' ? [...conversationHosts(entries)].sort().join('\n') : null),
    [entries, mode],
  );
  const links = useMemo<LinkRule>(
    () =>
      hostsKey === null
        ? { mode: 'work' }
        : { mode: 'chat', hosts: new Set(hostsKey === '' ? [] : hostsKey.split('\n')) },
    [hostsKey],
  );
  const [liveFrom] = useState(() => entries.length - (props.creating === true ? 1 : 0));
  const shownText = useRef(new Map<number, string>()).current;
  const [mounted, setMounted] = useState(MOUNTED_TURNS);
  const root = useScrollRoot();
  const before = useRef<number | null>(null);

  useLayoutEffect(() => {
    const height = before.current;
    before.current = null;
    if (height === null || root === null) return;
    root.scrollTop += root.scrollHeight - height;
  }, [mounted, root]);

  const placesKept = useRef<RunsAt | undefined>(undefined);
  const places = useMemo(() => {
    const next = runsAt(entries, placesKept.current);
    placesKept.current = next;
    return next;
  }, [entries]);
  const marks = props.marks;
  const placed = useMemo(
    () => (marks === undefined || marks.length === 0 ? null : placeMarks(entries, marks)),
    [entries, marks],
  );
  const turnMarks = useMemo(
    () => (placed === null ? null : marksByTurn(placed, list, entries, places)),
    [placed, list, entries, places],
  );

  // A jump mounts the turn it lands in first, then scrolls to it once that turn is drawn.
  const staticMotion = useStaticMotion();
  const listNow = useRef(list);
  useLayoutEffect(() => {
    listNow.current = list;
  });
  const [jumpTarget, setJumpTarget] = useState<{ at: number; tick: number } | null>(null);
  const jumpTo = useMemo(
    () =>
      (at: number): void => {
        setMounted((n) => mountedFor(listNow.current, at, n));
        setJumpTarget((previous) => ({ at, tick: (previous?.tick ?? 0) + 1 }));
      },
    [],
  );
  useLayoutEffect(() => {
    if (jumpTarget === null) return;
    const box = entryBox(document.querySelector(`[data-entry-at="${String(jumpTarget.at)}"]`));
    if (box === null) return;
    box.scrollIntoView({ block: 'center', behavior: staticMotion ? 'auto' : 'smooth' });
    box.setAttribute('data-entry-flash', '');
    const timer = setTimeout(() => box.removeAttribute('data-entry-flash'), ENTRY_FLASH_MS);
    return () => clearTimeout(timer);
  }, [jumpTarget, staticMotion]);
  const jumpRef = props.jumpRef;
  useLayoutEffect(() => {
    if (jumpRef === undefined) return;
    jumpRef.current = jumpTo;
    return () => {
      if (jumpRef.current === jumpTo) jumpRef.current = null;
    };
  }, [jumpRef, jumpTo]);
  const threadJump = useMemo<ThreadJump>(() => ({ to: jumpTo, target: jumpTarget }), [jumpTo, jumpTarget]);

  const last = list[list.length - 1];
  const lastIndex = list.length - 1;
  const hidden = Math.max(0, list.length - mounted);
  const visible = list.slice(hidden);
  // A turn runs before its prompt entry arrives: until then the last turn is the previous, ended one.
  const liveLast = running && last !== undefined && last.end === null;
  // The live end is the last turn's: a group after it of only a recap or a compaction holds no turn of its own.
  const endIndex = lastTurnIndex(list);
  const endTurn = list[endIndex];
  const endLive = liveLast && endIndex === lastIndex;
  const appEnds = props.host?.appEnds;
  const conversationId = props.conversationId ?? null;
  const app = (next: ThreadTurn | undefined): AppEndContext | undefined =>
    appEnds === undefined || appEnds.length === 0 || conversationId === null
      ? undefined
      : { ends: appEnds, conversationId, next: next === undefined ? null : { startedAt: turnStartedAt(next) } };
  const endLine =
    endTurn === undefined
      ? null
      : turnOutcome(endTurn, { last: true, running: endLive, live: props.lastEnd, mode, app: app(undefined) });
  const note =
    endLine === null || endLive || props.lastEnd == null
      ? null
      : engineNote({
          loading: loadFor(props.host?.loading ?? [], props.model ?? null),
          unsent: props.lastEnd.unsent === true,
          model: props.model ?? null,
          resident: props.host?.resident ?? [],
          formatSize: formatBytes,
        });
  const lastOutcome = endLine === null || note === null ? endLine : { ...endLine, note };
  // The disk keeps only "cancelled": a turn this page saw end keeps the outcome it was told (a deny stays a deny).
  const endsSeen = useRef({
    owner: props.conversationId ?? null,
    ends: new Map<number, NonNullable<ConversationState['lastEnd']>>(),
  }).current;
  if (endsSeen.owner !== (props.conversationId ?? null)) {
    endsSeen.owner = props.conversationId ?? null;
    endsSeen.ends.clear();
  }
  const seenEnds = endsSeen.ends;
  if (endTurn !== undefined && !endLive && props.lastEnd != null && props.lastEnd.outcome !== undefined)
    seenEnds.set(endTurn.start, props.lastEnd);
  const answered = liveLast && last.entries.some((entry) => entry.kind === 'assistant');

  const announce = useRef<{ started: number | null; end: ConversationState['lastEnd'] }>({
    started: answered ? last.start : null,
    end: props.lastEnd,
  });
  const onStatus = props.onStatus;
  useEffect(() => {
    const seen = announce.current;
    if (answered) {
      if (seen.started !== last.start) {
        seen.started = last.start;
        onStatus?.(CHAT.status.started);
      }
    }
    if (props.lastEnd !== seen.end) {
      seen.end = props.lastEnd;
      const end = props.lastEnd;
      if (end !== null && !running)
        onStatus?.(endSentence(lastOutcome, end.outcome === undefined ? undefined : end.model));
    }
  });
  // A picture notice or a recap is said once when it arrives while the thread shows; never one the thread opened with.
  const picturesSeen = useRef<{ owner: string | null; length: number } | null>(null);
  const owner = props.conversationId ?? null;
  useEffect(() => {
    const seen = picturesSeen.current;
    if (seen === null || seen.owner !== owner || entries.length < seen.length) {
      picturesSeen.current = { owner, length: entries.length };
      return;
    }
    for (let i = seen.length; i < entries.length; i += 1) {
      const entry = entries[i]!;
      if (entry.kind === 'notice' && entry.code === 'picture') onStatus?.(pictureHeadline(entry));
      else if (entry.kind === 'notice' && (entry.code === 'hook' || entry.code === 'hook-note'))
        onStatus?.(hookHeadline(entry));
      else if (entry.kind === 'recap') onStatus?.(fill(CHAT.history.recapStatus, { text: entry.text }));
    }
    seen.length = entries.length;
  }, [entries, owner, onStatus]);

  const timing = running && props.turn?.phase === 'running';
  const [quiet, setQuiet] = useState(false);
  useEffect(() => {
    setQuiet(false);
    if (!timing) return;
    const timer = setTimeout(() => setQuiet(true), QUIET_MS);
    return () => clearTimeout(timer);
  }, [timing, props.asOfMs]);
  const working = liveLast && hasWork(last);
  const compaction = props.compaction ?? null;
  // The thread end, first match wins: an open request, the compaction line, the retry line, the pending rule.
  const retry = liveLast && compaction === null && props.asking !== true ? openRetry(props.retry) : null;
  const hookOld = useOlderThan(props.hookRun?.startedAt ?? null, HOOK_WAIT_MS);
  const hook =
    props.hookRun != null && hookOld && compaction === null && retry === null && props.asking !== true
      ? props.hookRun
      : null;
  const pending =
    compaction === null &&
    retry === null &&
    hook === null &&
    (props.creating === true || running) &&
    !(working && (props.asking === true || moving(last, mode, quiet)));
  // A recap can only start with no answer running; one that starts drops it.
  const recapWait =
    props.recap != null && !running && props.creating !== true && compaction === null && retry === null
      ? props.recap
      : null;
  const retrySaid = useRef<{ ts: number | null; open: boolean }>({ ts: null, open: false });
  useEffect(() => {
    const said = retrySaid.current;
    if (retry !== null) {
      if (said.ts !== retry.ts) onStatus?.(fill(CHAT.retry.status, { cause: CHAT.retry.cause[retry.cause] }));
      said.ts = retry.ts;
      said.open = true;
      return;
    }
    if (!said.open) return;
    said.open = false;
    const ended = props.retry ?? null;
    if (running && ended !== null && ended.ts === said.ts && ended.outcome === 'recovered')
      onStatus?.(CHAT.retry.statusBack);
  });
  const waitSince = (working ? lastStepAt(last) : null) ?? props.turn?.startedAt ?? null;
  const runs = props.runs;
  const agentAsks = props.agentAsks;
  const onStopRun = props.onStopRun;
  const live = props.live;
  const agentLive = props.agentLive ?? true;
  const requests = props.host?.requests ?? NO_RUNS.requests;
  const threadRuns = useMemo<ThreadRuns>(
    () => ({
      runs: runs ?? NO_RUNS.runs,
      onStop: onStopRun,
      live,
      agentLive,
      announce: onStatus,
      requests,
    }),
    [runs, onStopRun, live, agentLive, onStatus, requests],
  );
  const namesKept = useRef<ReadonlyMap<string, string>>(new Map());
  const pdfNamesKept = useMemo(() => {
    const next = pdfNames(entries);
    const kept = namesKept.current;
    if (kept.size === next.size && [...next].every(([id, name]) => kept.get(id) === name)) return kept;
    namesKept.current = next;
    return next;
  }, [entries]);
  const pdfStarted = useMemo(() => startedPdfs(places), [places]);
  const pdfRunning = liveLast || props.creating === true;
  const onReadPdf = props.onReadPdf;
  const pdfSession = props.conversationId ?? null;
  const threadPdf = useMemo<ThreadPdf>(
    () => ({ names: pdfNamesKept, started: pdfStarted, running: pdfRunning, session: pdfSession, onRead: onReadPdf }),
    [pdfNamesKept, pdfStarted, pdfRunning, pdfSession, onReadPdf],
  );
  const agentHandlers = props.agentHandlers;
  const openAgent = props.openAgent ?? null;
  const marksOpen = props.agentMarksOpen ?? null;
  const threadAgents = useMemo<ThreadAgents>(
    () => ({ live, asks: agentAsks ?? NO_AGENTS.asks, handlers: agentHandlers, openId: openAgent, marksOpen }),
    [live, agentAsks, agentHandlers, openAgent, marksOpen],
  );
  const wakeReports = useMemo(() => wakeReportsOf(live, entries), [live, entries]);
  const helpersKept = useRef<ReadonlyMap<string, readonly AgentRowData[]> | undefined>(undefined);
  const helpersByRun = useMemo(() => {
    const next = runAgents(entries, helpersKept.current);
    helpersKept.current = next;
    return next;
  }, [entries]);
  const subagentsKept = useRef<ReadonlyMap<string, AgentRowData> | undefined>(undefined);
  const subagentsById = useMemo(() => {
    const next = subagentRows(entries, subagentsKept.current);
    subagentsKept.current = next;
    return next;
  }, [entries]);
  const onRemoveRule = props.onRemoveRule;
  const threadRules = useMemo<ThreadRules>(
    () => ({ ids: ruleIds(marks), onRemove: onRemoveRule }),
    [marks, onRemoveRule],
  );

  const thread = (
    <RunsContext.Provider value={threadRuns}>
      <AgentsContext.Provider value={threadAgents}>
        <RulesContext.Provider value={threadRules}>
          <WakeReportsContext.Provider value={wakeReports}>
            <div ref={props.contentRef} className="grid min-w-0 gap-6">
              {props.top}
              {props.transcriptError !== undefined && props.transcriptError !== null ? (
                <Alert tone="warning" title={CHAT.transcriptError}>
                  {props.inspectHref !== undefined ? (
                    <ButtonLink to={props.inspectHref} size="sm" variant="ghost">
                      {CHAT.inspect}
                    </ButtonLink>
                  ) : null}
                </Alert>
              ) : null}
              {hidden > 0 ? (
                <span className="flex justify-center">
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      before.current = root?.scrollHeight ?? null;
                      setMounted((n) => n + MOUNTED_TURNS);
                    }}
                  >
                    {fill(CHAT.earlier, { n: Math.min(MOUNTED_TURNS, hidden) })}
                  </Button>
                </span>
              ) : null}
              {visible.map((turn, index) => {
                const absolute = hidden + index;
                const isLast = absolute === lastIndex;
                const isEnd = absolute === endIndex;
                const seen = seenEnds.get(turn.start);
                const outcome = isEnd
                  ? lastOutcome
                  : turnOutcome(turn, {
                      last: seen !== undefined,
                      running: false,
                      live: seen ?? null,
                      mode,
                      app: app(list[absolute + 1]),
                    });
                return (
                  <TurnView
                    key={turn.start}
                    turn={turn}
                    mode={mode}
                    links={links}
                    running={isLast && liveLast}
                    tailLive={!((pending || compaction !== null || retry !== null || hook !== null) && working)}
                    isLast={isEnd}
                    outcome={outcome}
                    endOutcome={isEnd && !endLive ? props.lastEnd?.outcome : undefined}
                    stopped={seen?.outcome === 'cancelled'}
                    asOfMs={props.asOfMs}
                    model={props.model ?? null}
                    folder={props.folder ?? null}
                    tempRoot={props.tempRoot ?? null}
                    offscreen={absolute < list.length - ALWAYS_PAINTED}
                    liveFrom={liveFrom}
                    shownText={shownText}
                    actionDisabledReason={props.actionDisabledReason}
                    canInstallAgent={props.canInstallAgent !== false}
                    onTurnAction={props.onTurnAction}
                    keptFor={isEnd ? (props.keptFor ?? null) : null}
                    unload={isEnd ? props.unload : undefined}
                    turnMarks={turnMarks?.get(turn.start)}
                    askCalls={isLast ? props.askCalls : undefined}
                    rewind={props.rewind}
                  />
                );
              })}
              {placed !== null && placed.end.length > 0 ? (
                <div className="grid min-w-0 gap-3" data-thread-end-marks="">
                  {placed.end.map((mark, n) => (
                    <MarkLine key={`mark-end-${n}`} mark={mark} temp={props.tempRoot ?? null} />
                  ))}
                </div>
              ) : null}
              {props.card}
              {compaction !== null ? (
                <CompactionRow
                  compaction={compaction}
                  requests={props.host?.requests ?? []}
                  conversationId={conversationId}
                  first={!entries.some((entry) => entry.kind === 'compaction')}
                  onStatus={props.onStatus}
                />
              ) : null}
              {retry !== null ? (
                <RetryRow entry={retry} requests={props.host?.requests ?? []} conversationId={conversationId} />
              ) : null}
              {hook !== null ? <HookRow key={hook.startedAt} hookRun={hook} onStatus={props.onStatus} /> : null}
              {recapWait !== null ? <RecapWaitRow since={recapWait.startedAt} onStatus={props.onStatus} /> : null}
              {pending ? (
                <PendingRow
                  creating={props.creating === true}
                  continuing={running && continuingTurn(last, props.turn?.continues)}
                  turn={props.turn}
                  host={props.host}
                  model={props.model ?? null}
                  conversationId={conversationId}
                  quiet={quiet}
                  work={working}
                  since={waitSince}
                  pages={messagePages(last)}
                  modelSize={props.modelSize}
                  onStatus={props.onStatus}
                />
              ) : null}
            </div>
          </WakeReportsContext.Provider>
        </RulesContext.Provider>
      </AgentsContext.Provider>
    </RunsContext.Provider>
  );
  return (
    <PictureScope session={conversationId}>
      <RunsAtContext.Provider value={places}>
        <JumpContext.Provider value={threadJump}>
          <SubagentRowsContext.Provider value={subagentsById}>
            <RunHelpersContext.Provider value={helpersByRun}>
              <PdfContext.Provider value={threadPdf}>{thread}</PdfContext.Provider>
            </RunHelpersContext.Provider>
          </SubagentRowsContext.Provider>
        </JumpContext.Provider>
      </RunsAtContext.Provider>
    </PictureScope>
  );
}

export function noticeKey(notice: ConversationNotice): string {
  return `${notice.code}:${notice.provider}`;
}

/** The notices to show, one per code and provider: danger first, then newest. */
export function noticeOrder(notices: readonly ConversationNotice[]): ConversationNotice[] {
  const seen = new Set<string>();
  const unique: Array<{ notice: ConversationNotice; at: number }> = [];
  notices.forEach((notice, at) => {
    const key = noticeKey(notice);
    if (seen.has(key)) return;
    seen.add(key);
    unique.push({ notice, at });
  });
  const rank = (notice: ConversationNotice): number => (notice.tone === 'danger' ? 0 : 1);
  unique.sort((a, b) => rank(a.notice) - rank(b.notice) || b.at - a.at);
  return unique.map((item) => item.notice);
}

function noticeText(notice: ConversationNotice): string {
  if (notice.code === 'key-printed') return fill(CHAT.keyPrinted, { provider: providerName(notice.provider) });
  if (notice.code === 'thinking-not-set') return CHAT.thinkingLevel.notSet;
  return notice.code === 'question-too-long' ? CHAT.ask.tooLong : CHAT.planMode.notice;
}

/** Codes that keep the alert: a step to take, and the notice stays for the conversation. */
const ALERT_CODES: ReadonlySet<ConversationNotice['code']> = new Set(['key-printed']);

/**
 * Conversation notices, pinned above the composer: one shows (danger first, then newest), the others behind "+n". A
 * printed key keeps its alert and stays; every other code is one line the person can close.
 */
export function ConversationNotices({
  notices,
  onClose,
}: {
  notices: readonly ConversationNotice[];
  onClose?: (notice: ConversationNotice) => void;
}): ReactElement | null {
  const ordered = noticeOrder(notices);
  const shown = ordered[0];
  if (shown === undefined) return null;
  const rest = ordered.slice(1);
  const more =
    rest.length === 0 ? undefined : (
      <NoticeMore
        text={fill(CHAT.notices.more, { n: rest.length })}
        label={fill(CHAT.notices.moreName, { n: rest.length })}
        notices={rest.map((notice) => ({ key: noticeKey(notice), tone: notice.tone, text: noticeText(notice) }))}
      />
    );
  if (ALERT_CODES.has(shown.code)) {
    return (
      <Alert tone={shown.tone} title={CHAT.keyPrintedTitle}>
        <span className="grid gap-2">
          <span>{noticeText(shown)}</span>
          <span className="flex items-center gap-1">
            <ButtonLink to="/providers" size="sm" variant="ghost">
              {CHAT.webSettings}
            </ButtonLink>
            {more}
          </span>
        </span>
      </Alert>
    );
  }
  return (
    <NoticeLine
      tone={shown.tone}
      text={noticeText(shown)}
      more={more}
      onClose={onClose === undefined ? undefined : { label: CHAT.notices.close, run: () => onClose(shown) }}
    />
  );
}

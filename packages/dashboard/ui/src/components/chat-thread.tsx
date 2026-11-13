import { BellRing, Cpu, FilePen, FileText, Folder, GitBranch, RotateCw, Search, SquareTerminal } from 'lucide-react';
import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactElement, type ReactNode } from 'react';
import { Link } from 'react-router-dom';

import type { ConversationState } from '../lib/chat-store';
import { CHAT, fill } from '../lib/chat-strings';
import {
  answerMarkdown,
  continuingTurn,
  engineNote,
  fullLimit,
  loadFor,
  loadVariant,
  pendingRow,
  splitTurns,
  thoughtDuration,
  turnOutcome,
  turnPrompt,
  turnStats,
  turnStatsTitle,
  type OutcomeLine,
  type ThreadTurn,
} from '../lib/chat-turns';
import type {
  PermissionMark,
  ChatHostState,
  ChatMode,
  ConversationNotice,
  TurnOutcome,
  TurnState,
} from '../lib/chat-types';
import { formatBytes, formatCount, formatCountNoun, formatDuration } from '../lib/format';
import { truncateMiddle } from '../lib/labels';
import { type AllowedMark, allowedWord, placeMarks } from '../lib/permission-text';
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
import { type WorkCall, workCallOf } from '../lib/work-tools';
import { useScrollRoot } from '../shell/scroll-root';
import { ButtonLink } from './arc-x/button-link/button-link';
import { ChatMarkdown, type LinkRule } from './arc-x/chat-markdown/chat-markdown';
import { CodePanel } from './arc-x/code-panel/code-panel';
import { EventRow, RuleLabel } from './arc-x/event-row/event-row';
import { KeyValue } from './arc-x/key-value/key-value';
import { LoadCard } from './arc-x/load-card/load-card';
import { Message } from './arc-x/message/message';
import { PlanList } from './arc-x/plan-list/plan-list';
import { ReasoningBlock } from './arc-x/reasoning-block/reasoning-block';
import { SourceList, type SourceItem } from './arc-x/source-list/source-list';
import { ToolCall, ToolCallField, ToolCallGroup, type ToolStatus } from './arc-x/tool-call/tool-call';
import { TurnFooter } from './arc-x/turn-footer/turn-footer';
import { WaitLine } from './arc-x/wait-line/wait-line';
import { Alert } from './arc/alert/alert';
import { Button } from './arc/button/button';
import { CopyButton } from './arc/copy-button/copy-button';
import { ExpandableText } from './arc/expandable-text/expandable-text';
import { AllowedCard, AllowedLine, ModeLine } from './chat-permission-marks';
import { freshOffset, toolStatus } from './transcript';
import { UnloadButton, type UnloadControl } from './unload-button';

/** What a turn's action asks the page to do. `text` is the message to send. */
export type TurnAction =
  | { kind: 'send-again'; text: string }
  | { kind: 'continue'; text: string }
  | { kind: 'update-agent' };

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
  /** For the pending row's engine words and the engine note under the last end line. */
  host?: Pick<ChatHostState, 'inference' | 'loading' | 'resident' | 'answers'> | null;
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
  /** Why the turn actions are off now (the engine is stopped). */
  actionDisabledReason?: string;
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
  /** Work: the permission marks of the conversation, oldest first. */
  marks?: readonly PermissionMark[];
}

type Entry<K extends TranscriptEntry['kind']> = Extract<TranscriptEntry, { kind: K }>;

/** Turns mounted at once; older ones wait behind a button. */
export const MOUNTED_TURNS = 40;
/** The newest turns that always render; older ones may skip painting while off screen. */
const ALWAYS_PAINTED = 2;
const QUIET_MS = 3_000;
const OFFSCREEN = { contentVisibility: 'auto', containIntrinsicSize: 'auto 160px' } as const;

function isDrawn(entry: TranscriptEntry, mode: ChatMode): boolean {
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
  } else if (result !== null && result.kind === 'error') {
    status = 'failed';
    name = words.failed;
    detail = webErrorText(result.code);
    body = <WebBody result={result} raw={entry.output} />;
  } else if (mapped.status === 'failed' || mapped.status === 'other') {
    status = 'failed';
    name = words.failed;
    detail =
      props.refusal === 'address'
        ? CHAT.webErrors.blocked
        : props.refusal === null
          ? CHAT.webErrors['provider-error']
          : CHAT.refusedCall[props.refusal];
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

function allowedDetail(
  exit: string | undefined,
  allowed: AllowedMark | undefined,
  temp: string | null,
): string | undefined {
  if (allowed === undefined) return exit;
  const word = allowedWord(allowed, temp);
  return exit === undefined ? word : `${word} · ${exit}`;
}

/** A Work tool row: a command with its output, an edit with its change, a read, a listing, a file search. */
function WorkRow(props: {
  entry: Entry<'tool'>;
  call: WorkCall;
  live: boolean;
  /** The permission mode allowed this call: the row says so and shows the text its card would have shown. */
  allowed?: AllowedMark;
  temp: string | null;
}): ReactElement {
  const { entry, call } = props;
  const mapped = toolStatus(entry.status);
  const unfinished = !props.live && (mapped.status === 'running' || mapped.status === 'pending');
  const status: ToolStatus = unfinished ? 'other' : mapped.status;
  const [running, done, failed] = call.words;
  const name = status === 'running' || status === 'pending' ? running : status === 'failed' ? failed : done;
  const output = entry.output !== undefined && entry.output !== '' ? entry.output : null;
  const parts: ReactNode[] = [];
  const field = (key: string, label: string, code: string, maxLines?: number): ReactNode => (
    <ToolCallField key={key} label={label}>
      <CodePanel code={code} label={label} maxLines={maxLines} wrap />
    </ToolCallField>
  );
  const card = props.allowed?.card ?? null;
  if (card !== null) parts.push(<AllowedCard key="card" card={card} />);
  else if (call.command !== undefined) parts.push(field('command', CHAT.command, call.command));
  // An edit shows the change as the agent wrote it; before it answers, the change its input asks for.
  const change = call.kind === 'edit' ? (output ?? call.change) : undefined;
  if (change !== undefined) parts.push(field('change', CHAT.change, change, 12));
  if (output !== null && call.kind !== 'edit') parts.push(field('output', CHAT.output, output, 12));
  const ended = status === 'done' || status === 'failed';
  if (call.kind === 'run' && ended && output === null) {
    parts.push(
      <p key="output" className="m-0 text-body text-secondary">
        {CHAT.noOutput}
      </p>,
    );
  }
  return (
    <ToolCall
      name={name}
      nameStyle="word"
      icon={WORK_ICONS[call.kind]}
      title={call.title}
      detail={allowedDetail(call.kind === 'run' ? exitLine(entry) : undefined, props.allowed, props.temp)}
      status={status}
      statusText={unfinished ? 'not finished' : undefined}
      durationMs={status === 'running' || status === 'pending' ? null : durationOf(entry)}
      ts={entry.ts}
      body={parts.length > 0 ? <div className="grid min-w-0 gap-2">{parts}</div> : undefined}
      truncated={entry.truncated}
      defaultOpen={false}
      live={props.live && status === 'running'}
    />
  );
}

/** The agent's request to leave plan mode, answered on the plan card. */
const PLAN_TOOL = 'exit_plan_mode';

function GenericRow({
  entry,
  live,
  refusal,
  allowed,
  temp,
}: {
  entry: Entry<'tool'>;
  live: boolean;
  refusal: Refusal | null;
  /** The permission mode allowed this call: the row says so and shows the text its card would have shown. */
  allowed?: AllowedMark;
  temp: string | null;
}): ReactElement {
  const mapped = toolStatus(entry.status);
  const unfinished = !live && (mapped.status === 'running' || mapped.status === 'pending');
  const status: ToolStatus = unfinished ? 'other' : mapped.status;
  const output =
    refusal === null ? entry.output : refusal === 'address' ? CHAT.webErrors.blocked : CHAT.refusedCall[refusal];
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
  return (
    <ToolCall
      name={entry.name === PLAN_TOOL ? CHAT.planReview : entry.name}
      nameStyle={entry.name === PLAN_TOOL ? 'word' : undefined}
      title={entry.name === PLAN_TOOL && entry.title === PLAN_TOOL ? undefined : oneLine(entry.title)}
      status={status}
      statusText={unfinished ? 'not finished' : mapped.statusText}
      durationMs={durationOf(entry)}
      ts={entry.ts}
      detail={allowedDetail(undefined, allowed, temp)}
      input={entry.input}
      output={output}
      body={body}
      truncated={entry.truncated}
      live={live && status === 'running'}
    />
  );
}

interface TurnViewProps {
  turn: ThreadTurn;
  mode: ChatMode;
  links: LinkRule;
  /** This turn runs now. */
  running: boolean;
  isLast: boolean;
  outcome: OutcomeLine | null;
  /** The live outcome of this turn when it is the last one and ended: its refused call says why. */
  endOutcome: TurnOutcome | undefined;
  asOfMs: number | undefined;
  model: string | null;
  folder: string | null;
  tempRoot: string | null;
  offscreen: boolean;
  /** Entries from this index on arrived live: a user message among them rises in. */
  liveFrom: number;
  shownText: Map<number, string>;
  actionDisabledReason: string | undefined;
  onTurnAction: ((action: TurnAction) => void) | undefined;
  unload: UnloadControl | undefined;
  turnMarks: TurnMarks | undefined;
}

/** The permission marks of one turn: lines before an entry (by its index in the conversation) and allowed tool rows. */
interface TurnMarks {
  before: Map<number, PermissionMark[]>;
  allowed: Map<string, AllowedMark>;
  /** Changes when anything drawn from these marks changes. */
  key: string;
}

function markKey(mark: PermissionMark): string {
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
    prev.outcome?.note?.text === next.outcome?.note?.text &&
    prev.outcome?.note?.action === next.outcome?.note?.action &&
    prev.unload === next.unload &&
    prev.endOutcome === next.endOutcome &&
    prev.links === next.links &&
    prev.mode === next.mode &&
    prev.folder === next.folder &&
    prev.tempRoot === next.tempRoot &&
    prev.offscreen === next.offscreen &&
    prev.liveFrom === next.liveFrom &&
    prev.actionDisabledReason === next.actionDisabledReason &&
    prev.onTurnAction === next.onTurnAction &&
    prev.turnMarks?.key === next.turnMarks?.key
  );
}

const TurnView = memo(function TurnView(props: TurnViewProps): ReactElement {
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
  const all = turn.entries.filter((entry) => isDrawn(entry, mode));
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
    marks.forEach((mark, n) =>
      items.push(
        mark.kind === 'allowed' ? (
          <AllowedLine key={`mark-${index}-${n}`} mark={mark} temp={props.tempRoot} />
        ) : (
          <ModeLine key={`mark-${index}-${n}`} mark={mark} temp={props.tempRoot} />
        ),
      ),
    );
  };
  let i = 0;
  while (i < drawn.length) {
    const entry = drawn[i]!;
    const at = turn.start + turn.entries.indexOf(entry);
    pushLines(at);
    if (entry.kind === 'tool') {
      const run: Array<Entry<'tool'>> = [];
      while (
        i < drawn.length &&
        drawn[i]!.kind === 'tool' &&
        (run.length === 0 || before?.has(turn.start + turn.entries.indexOf(drawn[i]!)) !== true)
      ) {
        run.push(drawn[i] as Entry<'tool'>);
        i += 1;
      }
      const rows = run.map((tool) => {
        const key = turn.start + turn.entries.indexOf(tool);
        const web = mode === 'chat' ? webCallOf(tool) : null;
        const work = mode === 'work' ? workCallOf(tool, props.folder) : null;
        if (work !== null)
          return (
            <WorkRow
              key={key}
              entry={tool}
              call={work}
              live={running}
              allowed={props.turnMarks?.allowed.get(tool.id)}
              temp={props.tempRoot}
            />
          );
        return web !== null ? (
          <WebRow
            key={key}
            entry={tool}
            call={web.call}
            result={web.result}
            live={running}
            refusal={tool === refused?.entry ? refused.refusal : null}
          />
        ) : (
          <GenericRow
            key={key}
            entry={tool}
            live={running}
            refusal={tool === refused?.entry ? refused.refusal : null}
            allowed={props.turnMarks?.allowed.get(tool.id)}
            temp={props.tempRoot}
          />
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
        if (entry.continued === true) {
          items.push(
            <RuleLabel key={at}>
              <RotateCw size={12} strokeWidth={1.75} aria-hidden="true" />
              {CHAT.continued}
            </RuleLabel>,
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
            footer={<CopyButton value={entry.text} label={CHAT.copyMessage} iconOnly variant="plain" />}
          >
            <ExpandableText lines={12}>{entry.text}</ExpandableText>
          </Message>,
        );
        break;
      case 'assistant': {
        const streaming = running && entry === tail;
        const fresh = streaming ? freshOffset(props.shownText.get(at), entry.text) : undefined;
        if (streaming) rendered.push([at, entry.text]);
        const index = turn.entries.indexOf(entry);
        const text = answerSplit !== null && index === lastAnswer ? answerSplit.body : entry.text;
        items.push(
          <Message key={at} role="assistant" variant="chat" ts={entry.ts} streaming={streaming} status={CHAT.writing}>
            <ChatMarkdown
              text={text}
              streaming={streaming}
              freshFrom={fresh}
              links={props.links}
              citations={index === lastAnswer ? citations : undefined}
              onCite={index === lastAnswer ? onCite : undefined}
            />
          </Message>,
        );
        break;
      }
      case 'thinking': {
        const live = running && entry === tail;
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
          />,
        );
        break;
      }
      case 'wake':
        items.push(
          <EventRow
            key={at}
            icon={<BellRing size={14} strokeWidth={1.75} />}
            label="No prompt typed"
            text={wakeTitle(entry)}
            ts={entry.ts}
          />,
        );
        break;
      case 'plan':
        items.push(<PlanList key={at} items={entry.entries} ts={entry.ts} />);
        break;
      case 'subagent':
        items.push(
          <EventRow
            key={at}
            icon={<GitBranch size={14} strokeWidth={1.75} />}
            label={`Subagent${entry.type !== '' ? ` · ${entry.type}` : ''}`}
            text={entry.description !== '' ? entry.description : entry.childId}
            to={entry.exists === true ? `/sessions/${encodeURIComponent(entry.childId)}` : undefined}
            facts={[
              entry.status,
              entry.durationMs !== undefined ? formatDuration(entry.durationMs) : null,
              entry.tokens !== undefined ? formatCountNoun(entry.tokens, 'token', 'tokens', formatCount) : null,
            ]}
            tone={entry.error !== undefined ? 'danger' : 'neutral'}
            error={entry.error}
            ts={entry.ts}
          />,
        );
        break;
      case 'model':
        items.push(
          <RuleLabel key={at}>
            <Cpu size={12} strokeWidth={1.75} aria-hidden="true" />
            {`Model switched to ${entry.modelId}`}
          </RuleLabel>,
        );
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

  if (finished && (turn.end !== null || props.outcome !== null || turnPrompt(turn) !== null)) {
    const answer = answerMarkdown(turn.entries);
    const outcome = props.outcome;
    const prompt = turnPrompt(turn);
    let action: { label: string; onSelect: () => void; disabled?: boolean; disabledReason?: string } | undefined;
    const onTurnAction = props.onTurnAction;
    if (props.isLast && outcome !== null && outcome.action !== null && onTurnAction !== undefined) {
      const disabled = props.actionDisabledReason !== undefined && outcome.action !== 'update-agent';
      const base = { disabled, disabledReason: props.actionDisabledReason };
      if (outcome.action === 'send-again' && prompt !== null)
        action = { ...base, label: CHAT.sendAgain, onSelect: () => onTurnAction({ kind: 'send-again', text: prompt }) };
      else if (outcome.action === 'continue')
        action = {
          ...base,
          label: CHAT.continueAnswer,
          onSelect: () => onTurnAction({ kind: 'continue', text: CHAT.continuePrompt }),
        };
      else if (outcome.action === 'update-agent')
        action = { label: CHAT.updateAgent, onSelect: () => onTurnAction({ kind: 'update-agent' }) };
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

  return (
    <div data-turn={turn.start} className="grid min-w-0 gap-3" style={props.offscreen ? OFFSCREEN : undefined}>
      {items}
    </div>
  );
}, sameTurn);

function PendingRow(props: {
  creating: boolean;
  continuing: boolean;
  turn: TurnState | null;
  host: ChatThreadProps['host'];
  asOfMs: number | undefined;
  model: string | null;
  modelSize: ChatThreadProps['modelSize'];
  onStatus: ChatThreadProps['onStatus'];
}): ReactElement {
  const [quiet, setQuiet] = useState(false);
  const phase = props.turn?.phase ?? null;
  useEffect(() => {
    setQuiet(false);
    if (phase !== 'running') return;
    const timer = setTimeout(() => setQuiet(true), QUIET_MS);
    return () => clearTimeout(timer);
  }, [phase, props.asOfMs]);
  const loading = loadFor(props.host?.loading ?? [], props.model);
  const row = pendingRow({
    creating: props.creating,
    phase,
    wait: props.turn?.wait ?? null,
    model: props.model,
    loading,
    inference: props.host?.inference ?? null,
    quiet,
    full: fullLimit(props.host?.answers ?? [], props.model, props.turn?.startedAt ?? null),
    continuing: props.continuing,
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
  if (row.kind === 'label') return <WaitLine orb={row.orb} text={row.text} enter />;
  const since = variant === 'loading' && loading !== null ? loading.startedAt : (props.turn?.startedAt ?? Date.now());
  return (
    <Message role="assistant" variant="chat" enter>
      <LoadCard
        modelId={row.modelId}
        variant={variant}
        other={loading?.modelId ?? null}
        since={since}
        sizeBytes={props.modelSize?.(row.modelId) ?? null}
        progress={variant === 'loading' ? loading : null}
      />
    </Message>
  );
}

/** The marks of each turn, by the turn's first entry index. */
function marksByTurn(
  placed: ReturnType<typeof placeMarks>,
  turns: readonly ThreadTurn[],
  entries: readonly TranscriptEntry[],
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
      marks = { before: new Map(), allowed: new Map(), key: '' };
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
  return out;
}

/** The work of a turn has started: anything but its prompt and model switches. */
function hasWork(turn: ThreadTurn | undefined): boolean {
  return (
    turn !== undefined &&
    turn.entries.some((entry) => !(entry.kind === 'user' || entry.kind === 'wake' || entry.kind === 'model'))
  );
}

/** Status sentence of a finished turn. */
function endSentence(outcome: OutcomeLine | null): string {
  return outcome === null ? CHAT.status.finished : outcome.text;
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

  const marks = props.marks;
  const placed = useMemo(
    () => (marks === undefined || marks.length === 0 ? null : placeMarks(entries, marks)),
    [entries, marks],
  );
  const turnMarks = useMemo(
    () => (placed === null ? null : marksByTurn(placed, list, entries)),
    [placed, list, entries],
  );

  const last = list[list.length - 1];
  const lastIndex = list.length - 1;
  const hidden = Math.max(0, list.length - mounted);
  const visible = list.slice(hidden);
  // A turn runs before its prompt entry arrives: until then the last turn is the previous, ended one.
  const liveLast = running && last !== undefined && last.end === null;
  const endLine =
    last === undefined ? null : turnOutcome(last, { last: true, running: liveLast, live: props.lastEnd, mode });
  const note =
    endLine === null || liveLast || props.lastEnd == null
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
  if (last !== undefined && !liveLast && props.lastEnd != null && props.lastEnd.outcome !== undefined)
    seenEnds.set(last.start, props.lastEnd);
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
      if (props.lastEnd !== null && !running) onStatus?.(endSentence(lastOutcome));
    }
  });

  const pending = (props.creating === true || running) && !(liveLast && hasWork(last));

  return (
    <div ref={props.contentRef} className="grid min-w-0 gap-6">
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
        const seen = seenEnds.get(turn.start);
        const outcome = isLast
          ? lastOutcome
          : turnOutcome(turn, { last: seen !== undefined, running: false, live: seen ?? null, mode });
        return (
          <TurnView
            key={turn.start}
            turn={turn}
            mode={mode}
            links={links}
            running={isLast && liveLast}
            isLast={isLast}
            outcome={outcome}
            endOutcome={isLast && !liveLast ? props.lastEnd?.outcome : undefined}
            asOfMs={props.asOfMs}
            model={props.model ?? null}
            folder={props.folder ?? null}
            tempRoot={props.tempRoot ?? null}
            offscreen={absolute < list.length - ALWAYS_PAINTED}
            liveFrom={liveFrom}
            shownText={shownText}
            actionDisabledReason={props.actionDisabledReason}
            onTurnAction={props.onTurnAction}
            unload={isLast ? props.unload : undefined}
            turnMarks={turnMarks?.get(turn.start)}
          />
        );
      })}
      {placed !== null && placed.end.length > 0 ? (
        <div className="grid min-w-0 gap-3" data-thread-end-marks="">
          {placed.end.map((mark, n) =>
            mark.kind === 'allowed' ? (
              <AllowedLine key={`mark-end-${n}`} mark={mark} temp={props.tempRoot ?? null} />
            ) : (
              <ModeLine key={`mark-end-${n}`} mark={mark} temp={props.tempRoot ?? null} />
            ),
          )}
        </div>
      ) : null}
      {props.card}
      {pending ? (
        <PendingRow
          creating={props.creating === true}
          continuing={running && continuingTurn(last, props.turn?.continues)}
          turn={props.turn}
          host={props.host}
          asOfMs={props.asOfMs}
          model={props.model ?? null}
          modelSize={props.modelSize}
          onStatus={props.onStatus}
        />
      ) : null}
    </div>
  );
}

/** Conversation notices (a key printed in Work), for the page to pin above the composer. */
export function ConversationNotices({ notices }: { notices: readonly ConversationNotice[] }): ReactElement | null {
  if (notices.length === 0) return null;
  const seen = new Set<string>();
  const shown = notices.filter((notice) => {
    const key = `${notice.code}:${notice.provider}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  return (
    <div className="grid gap-2">
      {shown.map((notice) => (
        <Alert key={`${notice.code}:${notice.provider}`} tone="danger" title={CHAT.keyPrintedTitle}>
          <span className="grid gap-2">
            <span>{fill(CHAT.keyPrinted, { provider: providerName(notice.provider) })}</span>
            <span>
              <ButtonLink to="/providers" size="sm" variant="ghost">
                {CHAT.webSettings}
              </ButtonLink>
            </span>
          </span>
        </Alert>
      ))}
    </div>
  );
}

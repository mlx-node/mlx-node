import { BellRing, Cpu, FileText, RotateCw } from 'lucide-react';
import { useEffect, useMemo, useRef, useState, type MouseEvent, type ReactElement, type ReactNode } from 'react';

import { type AgentRow as AgentRowData, agentState, runAgents, subagentRows } from '../lib/agents';
import { CHAT, fill } from '../lib/chat-strings';
import { thoughtDuration } from '../lib/chat-turns';
import { formatClock, formatDuration } from '../lib/format';
import { rowWords } from '../lib/helper-words';
import type { LinkRule } from '../lib/link-rule';
import { hostStartWords, pdfNames, pdfRunName, pdfUnread } from '../lib/pdf-run';
import { neverNoteWords } from '../lib/permission-text';
import { type RunsAt, runsAt, startedPdfs } from '../lib/run-view';
import { groupTurns, turnTitle, wakeTitle, type TurnGroup } from '../lib/sessions';
import type { TranscriptEntry, TurnMetric } from '../lib/types';
import { visibleText, workCallOf } from '../lib/work-tools';
import { AgentRow } from './arc-x/agent-row/agent-row';
import { AttachmentStrip, sentChips } from './arc-x/attachment-strip/attachment-strip';
import { ChatMarkdown } from './arc-x/chat-markdown/chat-markdown';
import { CodePanel } from './arc-x/code-panel/code-panel';
import { CompactField } from './arc-x/compact-field/compact-field';
import { diffStateOf, EditDiffs } from './arc-x/diff-view/diff-view';
import { Disclosure } from './arc-x/disclosure/disclosure';
import { EventRow, RuleLabel } from './arc-x/event-row/event-row';
import { GoalRow } from './arc-x/goal-strip/goal-strip';
import { LiveLabel } from './arc-x/live-label/live-label';
import { Message } from './arc-x/message/message';
import { ModelBadge } from './arc-x/model-badge/model-badge';
import { Panel } from './arc-x/panel/panel';
import { PictureScope, PictureStrip, picturesDetail } from './arc-x/picture-strip/picture-strip';
import { PlanList } from './arc-x/plan-list/plan-list';
import { ReasoningBlock } from './arc-x/reasoning-block/reasoning-block';
import { ToolCall, ToolCallField, ToolCallGroup, type ToolStatus } from './arc-x/tool-call/tool-call';
import { Alert } from './arc/alert/alert';
import { Button } from './arc/button/button';
import { EmptyState } from './arc/empty-state/empty-state';
import { SearchField } from './arc/search-field/search-field';
import SegmentedControl, { type Segment } from './arc/segmented-control/segmented-control';
import { Tooltip } from './arc/tooltip/tooltip';
import { CompactionMark } from './compaction-mark';
import { RecapRow } from './history-mark';
import { NoteRow } from './note-row';
import { NoticeRow } from './notice-row';
import { PdfUnreadRow } from './pdf-unread-row';
import { PlanModeLine } from './plan-mode-line';
import { RuleRow } from './rule-row';
import { RunBlock } from './run-block';
import { RunEndRow } from './run-end-row';

export interface TranscriptProps {
  /** The session shown: its tool rows read their pictures from it. */
  sessionId?: string | null;
  entries: TranscriptEntry[];
  turns: TurnMetric[];
  /** 1-based turn to open and scroll to. */
  focusTurn?: number | null;
  onInspectTurn: (turn: number) => void;
  /** Session model before any switch; labels assistant messages. */
  model: string | null;
  /** The session runs (its body says so), fresh or not. Without it an unclosed last turn reads as cut off. */
  sessionLive: boolean;
  /** True while the session is live AND this body is fresh. Default false. */
  live?: boolean;
  /** Client time when this body arrived; the "as of" of every running time shown. */
  asOfMs?: number;
  /** The reader's error for this transcript; shown above it. */
  transcriptError?: string;
}

type Entry<K extends TranscriptEntry['kind']> = Extract<TranscriptEntry, { kind: K }>;
type Show = 'all' | 'messages';

const SHOW_OPTIONS: Segment[] = [
  { value: 'all', label: 'All' },
  { value: 'messages', label: 'Messages' },
];
const WORK_LINKS: LinkRule = { mode: 'work' };
const FIND_DEBOUNCE_MS = 200;
const FIND_OPEN = 5;
const EXPAND_ALL_MAX = 60;
const EXPAND_HINT = 'Too many turns to open at once. Use find, or open turns one by one.';

function isShow(value: string): value is Show {
  return value === 'all' || value === 'messages';
}

export function toolStatus(raw: string): { status: ToolStatus; statusText?: string } {
  if (raw === 'completed') return { status: 'done' };
  if (raw === 'failed') return { status: 'failed' };
  if (raw === 'in_progress') return { status: 'running' };
  if (raw === 'pending') return { status: 'pending' };
  return { status: 'other', statusText: raw };
}

/** An edit call's body, as the Work row draws it: the diff of each file, then the agent's words (an error) if any. */
function EditBody({ tool }: { tool: Entry<'tool'> }): ReactElement {
  return (
    <div className="grid min-w-0 gap-2">
      <EditDiffs edits={tool.edits ?? []} state={diffStateOf(tool.status)} folder={null} />
      {tool.output !== undefined && tool.output !== '' ? (
        <ToolCallField label={CHAT.output}>
          <CodePanel code={tool.output} label={CHAT.output} maxLines={12} wrap />
        </ToolCallField>
      ) : null}
    </div>
  );
}

/** A picture read's body, as the Work row draws it: the tiles, then the agent's words if any. */
function PicturesBody({ tool }: { tool: Entry<'tool'> }): ReactElement {
  return (
    <div className="grid min-w-0 gap-2">
      <PictureStrip call={tool.id} pictures={tool.pictures ?? []} path={workCallOf(tool, null)?.title ?? tool.title} />
      {tool.output !== undefined && tool.output !== '' ? (
        <ToolCallField label={CHAT.output}>
          <CodePanel code={tool.output} label={CHAT.output} maxLines={12} wrap />
        </ToolCallField>
      ) : null}
    </div>
  );
}

function entryMatches(entry: TranscriptEntry, needle: string): boolean {
  const has = (value: string | undefined): boolean => value !== undefined && value.toLowerCase().includes(needle);
  switch (entry.kind) {
    case 'user':
      return has(entry.text) || (entry.attachments?.some((item) => item.kind !== 'image' && has(item.name)) ?? false);
    case 'assistant':
    case 'thinking':
      return has(entry.text);
    case 'wake':
      return has(entry.text) || has(entry.workflow?.name);
    case 'tool':
      return has(entry.name) || has(entry.title) || has(entry.input) || has(entry.output) || has(entry.blocked);
    case 'subagent':
      return has(entry.description);
    case 'plan':
      return entry.entries.some((item) => has(item.content));
    case 'workflow':
      return has(entry.name) || has(entry.report) || entry.agents.some((agent) => has(agent.label));
    case 'notice':
      if (entry.code === 'picture') return entry.notes.some(has);
      if (entry.code === 'hook') return has(entry.hook) || has(entry.detail);
      return entry.code === 'hook-note' ? has(entry.text) : has(entry.detail);
    case 'recap':
      return has(entry.text);
    case 'goal':
      return has(entry.objective);
    case 'model':
    case 'plan-mode':
    case 'compaction':
    case 'turn-end':
      return false;
  }
}

function isMessage(entry: TranscriptEntry): boolean {
  return (
    entry.kind !== 'thinking' &&
    entry.kind !== 'tool' &&
    entry.kind !== 'plan' &&
    entry.kind !== 'subagent' &&
    entry.kind !== 'notice'
  );
}

/** Where a live entry's text grew since the last render: the offset the added part starts at. */
export function freshOffset(previous: string | undefined, text: string): number | undefined {
  if (previous === undefined || !text.startsWith(previous)) return undefined;
  return previous.length;
}

interface GroupView {
  group: TurnGroup;
  /** Index of the group in the full list; the key of its open state. */
  index: number;
}

function GroupTitle({ group, trailing, live }: { group: TurnGroup; trailing: boolean; live: boolean }): ReactElement {
  if (group.number === 0) return <span>Earlier messages</span>;
  const summary = turnTitle(group.entries);
  return (
    <span className="flex min-w-0 items-baseline gap-2">
      <span className="inline-block min-w-[7ch] shrink-0 tabular-nums">
        {`Turn ${group.number}`}
        {trailing ? (
          <>
            {' · '}
            <LiveLabel active={live}>in progress</LiveLabel>
          </>
        ) : null}
      </span>
      <span className="min-w-0 truncate font-normal text-secondary">{summary}</span>
    </span>
  );
}

function GroupMeta({ group }: { group: TurnGroup }): ReactElement {
  const tools = group.entries.filter((entry): entry is Entry<'tool'> => entry.kind === 'tool');
  const failed = tools.filter((tool) => tool.status === 'failed').length;
  const elapsed = group.metric?.elapsedMs ?? group.end?.elapsedMs ?? null;
  const parts: string[] = [];
  if (group.startTs !== null) parts.push(formatClock(group.startTs));
  if (elapsed !== null) parts.push(formatDuration(elapsed));
  if (tools.length > 0) parts.push(`${tools.length} ${tools.length === 1 ? 'tool' : 'tools'}`);
  return (
    <span className="tabular-nums">
      {parts.join(' · ')}
      {failed > 0 ? (
        <>
          {' · '}
          <span className="text-danger">
            {`${failed} ✕`}
            <span className="sr-only"> failed</span>
          </span>
        </>
      ) : null}
    </span>
  );
}

interface BodyProps {
  group: TurnGroup;
  trailing: boolean;
  live: boolean;
  asOfMs: number | undefined;
  show: Show;
  model: string | null;
  modelAt: ReadonlyMap<TranscriptEntry, string | null>;
  indexOf: ReadonlyMap<TranscriptEntry, number>;
  shownText: Map<number, string>;
  onInspectTurn: (turn: number) => void;
  helpers: ReadonlyMap<string, readonly AgentRowData[]>;
  subagents: ReadonlyMap<string, AgentRowData>;
  places: RunsAt;
  /** The name of each PDF the session's messages sent to the PDF tool, by its id. */
  pdfNames: ReadonlyMap<string, string>;
  /** The session runs in an agent process now (its body says so), fresh or not. */
  sessionLive: boolean;
  sessionId: string | null;
  onJump: (at: number) => void;
}

/** An entry a jump lands on: it draws no box of its own. */
function Anchor({ at, children }: { at: number; children: ReactNode }): ReactElement {
  return (
    <div data-entry-at={at} className="contents">
      {children}
    </div>
  );
}

/**
 * A run's agents are drawn in its block, and a `workflow` entry whose block stands at its launch call is data only. A
 * run that only spawn lines name draws its block at the first of them.
 */
function drawnHere(entry: TranscriptEntry, at: number, places: RunsAt): boolean {
  if (entry.kind === 'subagent') return entry.run === undefined || places.blocks.get(at) === entry.run;
  if (entry.kind === 'workflow') return !places.dataOnly.has(at);
  return true;
}

function GroupBody(props: BodyProps): ReactElement {
  const { group, trailing, live, asOfMs, show } = props;
  const open = trailing && group.end === null;
  const places = props.places;
  const visible = group.entries.filter(
    (entry) => (show !== 'messages' || isMessage(entry)) && drawnHere(entry, props.indexOf.get(entry) ?? -1, places),
  );
  const block = (at: number, runId: string): ReactElement | null => {
    const facts = places.runs.get(runId);
    if (facts === undefined) return null;
    return (
      <Anchor key={at} at={at}>
        <RunBlock
          runId={runId}
          entry={facts.entry}
          agents={props.helpers.get(runId) ?? []}
          live={false}
          sessionLive={props.sessionLive}
          by={facts.by}
          wakes={facts.wakes}
          resultAt={facts.resultAt}
          onJump={props.onJump}
          launch={facts.launch}
          workHref={props.sessionId === null ? undefined : `/chat?c=${encodeURIComponent(props.sessionId)}`}
          pdf={facts.pdf}
        />
      </Anchor>
    );
  };
  // The reader stamps endTs on every text chunk, so only the open group's final entry is still being written.
  const tail = group.entries[group.entries.length - 1];
  const rendered: Array<[number, string]> = [];

  useEffect(() => {
    for (const [index, text] of rendered) props.shownText.set(index, text);
  });

  const items: ReactNode[] = [];
  let i = 0;
  while (i < visible.length) {
    const entry = visible[i]!;
    const at = props.indexOf.get(entry) ?? -1;
    const launchRun = entry.kind === 'tool' ? places.blocks.get(at) : undefined;
    if (launchRun !== undefined) {
      i += 1;
      items.push(block(at, launchRun));
      continue;
    }
    if (entry.kind === 'tool') {
      const run: Array<Entry<'tool'>> = [];
      while (
        i < visible.length &&
        visible[i]!.kind === 'tool' &&
        places.blocks.get(props.indexOf.get(visible[i]!) ?? -1) === undefined
      ) {
        run.push(visible[i] as Entry<'tool'>);
        i += 1;
      }
      const calls = run.map((tool) => {
        const mapped = toolStatus(tool.status);
        const unfinished = !open && (mapped.status === 'running' || mapped.status === 'pending');
        const status: ToolStatus = unfinished ? 'other' : mapped.status;
        const statusText = unfinished ? 'not finished' : mapped.statusText;
        return (
          <ToolCall
            key={props.indexOf.get(tool) ?? tool.id}
            name={tool.name}
            title={tool.title}
            status={status}
            statusText={statusText}
            durationMs={tool.endTs !== undefined && tool.ts > 0 ? tool.endTs - tool.ts : null}
            ts={tool.ts}
            detail={
              tool.blocked !== undefined
                ? CHAT.hook.blockedRow
                : tool.pictures !== undefined && tool.pictures.length > 0
                  ? picturesDetail(tool.pictures, tool.pdfPages)
                  : undefined
            }
            input={tool.input}
            output={tool.blocked === undefined ? tool.output : visibleText(tool.blocked)}
            body={
              tool.edits !== undefined && tool.edits.length > 0 ? (
                <EditBody tool={tool} />
              ) : tool.pictures !== undefined && tool.pictures.length > 0 ? (
                <PicturesBody tool={tool} />
              ) : undefined
            }
            truncated={tool.truncated}
            live={open && status === 'running' ? live : false}
          />
        );
      });
      if (run.length === 1) {
        items.push(calls[0]);
      } else {
        const failed = run.filter((tool) => tool.status === 'failed').length;
        const running = open ? run.filter((tool) => tool.status === 'in_progress').length : 0;
        items.push(
          <ToolCallGroup key={`tools-${at}`} count={run.length} failed={failed} running={running} live={open && live}>
            {calls}
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
              <RuleRow key={at} words={rule} ts={entry.ts} />
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
        const started = hostStartWords(entry.text, props.pdfNames);
        if (started !== null) {
          items.push(
            <EventRow
              key={at}
              icon={<FileText size={14} strokeWidth={1.75} />}
              label={started}
              text=""
              ts={entry.ts}
            />,
          );
          break;
        }
        items.push(
          <Message
            key={at}
            role="user"
            label={entry.interjection === true ? 'You · interjection' : 'You'}
            ts={entry.ts}
          >
            {entry.attachments !== undefined && entry.attachments.length > 0 ? (
              <div className="grid gap-2">
                {entry.text !== '' ? <ChatMarkdown text={entry.text} links={WORK_LINKS} /> : null}
                <AttachmentStrip readOnly items={sentChips(entry.attachments)} />
              </div>
            ) : (
              <ChatMarkdown text={entry.text} links={WORK_LINKS} />
            )}
          </Message>,
        );
        break;
      case 'wake': {
        const runId = entry.cause === 'workflow' ? entry.workflow?.runId : undefined;
        const pdf = runId === undefined ? null : (places.runs.get(runId)?.pdf ?? null);
        items.push(
          runId !== undefined ? (
            <Anchor key={at} at={at}>
              <RunEndRow
                entry={entry}
                runAt={places.runs.get(runId)?.place.at ?? null}
                onJump={props.onJump}
                {...(pdf === null ? {} : { name: pdfRunName(pdf) })}
              />
            </Anchor>
          ) : (
            <EventRow
              key={at}
              icon={<BellRing size={14} strokeWidth={1.75} />}
              label="No prompt typed"
              text={wakeTitle(entry)}
              ts={entry.ts}
            />
          ),
        );
        if (show === 'all' && entry.text.trim() !== '') {
          items.push(
            <ReasoningBlock
              key={`${String(at)}-hidden`}
              label={entry.truncated === true ? 'Hidden prompt, first 16 KB' : 'Hidden prompt'}
              text={entry.text}
            />,
          );
        }
        break;
      }
      case 'assistant': {
        const current = props.modelAt.get(entry) ?? group.metric?.model ?? props.model;
        const streaming = open && entry === tail;
        if (streaming) rendered.push([at, entry.text]);
        items.push(
          <Message
            key={at}
            role="assistant"
            label={current !== null ? <ModelBadge name={current} size="sm" showLogo /> : 'Assistant'}
            ts={entry.ts}
            streaming={streaming && live}
          >
            <ChatMarkdown text={entry.text} streaming={streaming && live} links={WORK_LINKS} />
          </Message>,
        );
        break;
      }
      case 'thinking': {
        const running = open && entry === tail;
        if (running) rendered.push([at, entry.text]);
        items.push(
          <ReasoningBlock
            key={at}
            text={entry.text}
            durationMs={running ? null : thoughtDuration(entry)}
            live={running && live}
            startTs={running ? entry.ts : undefined}
            asOfMs={running ? asOfMs : undefined}
            freshFrom={running && live ? freshOffset(props.shownText.get(at), entry.text) : undefined}
            links={WORK_LINKS}
          />,
        );
        break;
      }
      case 'plan':
        items.push(<PlanList key={at} items={entry.entries} ts={entry.ts} />);
        break;
      case 'subagent': {
        if (entry.run !== undefined) {
          items.push(block(at, entry.run));
          break;
        }
        const row = props.subagents.get(entry.childId);
        if (row === undefined) break;
        items.push(
          <AgentRow
            key={at}
            id={row.childId}
            label={row.label}
            words={rowWords(agentState(row, { live: false, now: asOfMs ?? 0, open: props.sessionLive }), {
              progress: null,
              now: asOfMs ?? 0,
            })}
            kind={entry.type !== '' ? `${CHAT.subagent.label} · ${entry.type}` : CHAT.subagent.label}
            to={entry.exists === true ? `/sessions/${encodeURIComponent(entry.childId)}` : undefined}
            openLabel={fill(CHAT.agent.open, { label: row.label })}
            ts={entry.ts}
          />,
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
      case 'workflow':
        items.push(block(at, entry.runId));
        break;
      case 'notice':
        items.push(<NoticeRow key={at} entry={entry} running={open} />);
        break;
      case 'recap':
        items.push(<RecapRow key={at} entry={entry} />);
        break;
      case 'goal':
        items.push(<GoalRow key={at} entry={entry} />);
        break;
      case 'turn-end':
        break;
    }
  }

  for (const pdf of pdfUnread(group.entries, 0, open, startedPdfs(places)))
    items.push(<PdfUnreadRow key={`pdf-${pdf.id}`} pdf={pdf} />);

  const end = group.end;
  if (end !== null && group.number > 0) {
    const elapsed = end.elapsedMs !== undefined ? ` · ${formatDuration(end.elapsedMs)}` : '';
    const inspect = (event: MouseEvent<HTMLAnchorElement>): void => {
      event.preventDefault();
      props.onInspectTurn(group.number);
    };
    items.push(
      <RuleLabel key="turn-end" tone={end.error !== undefined ? 'danger' : 'neutral'} detail={end.error}>
        {`Turn ended · ${end.stopReason}${elapsed}`}
        {' · '}
        <a className="link" href={`?tab=activity&turn=${group.number}`} onClick={inspect}>
          Inspect
        </a>
      </RuleLabel>,
    );
  }

  return <div className="grid min-w-0 gap-3 px-3 pt-1 pb-4">{items}</div>;
}

/** The assistant entries' model: the session model until a model row switches it. */
function modelsAt(entries: readonly TranscriptEntry[]): ReadonlyMap<TranscriptEntry, string | null> {
  const map = new Map<TranscriptEntry, string | null>();
  let current: string | null = null;
  for (const entry of entries) {
    if (entry.kind === 'model') current = entry.modelId;
    else if (entry.kind === 'assistant') map.set(entry, current);
  }
  return map;
}

export interface TranscriptStreamProps {
  /** The session the lines are from: its tool rows read their pictures from it. */
  sessionId: string | null;
  entries: readonly TranscriptEntry[];
  /** Its lines still come: the last text streams and a running call reads as running. */
  live: boolean;
  model: string | null;
  asOfMs?: number;
}

/**
 * One session's lines as one stream with the Transcript's rows, in no turn groups: the Agent view's body. After the
 * stream ends, a call with no end reads not finished.
 */
export function TranscriptStream(props: TranscriptStreamProps): ReactElement {
  const entries = props.entries;
  const group = useMemo<TurnGroup>(
    () => ({ number: 0, entries: [...entries], end: null, metric: null, startTs: null, endTs: null }),
    [entries],
  );
  const indexOf = useMemo(() => new Map(entries.map((entry, index) => [entry, index] as const)), [entries]);
  const helpers = useMemo(() => runAgents(entries), [entries]);
  const subagents = useMemo(() => subagentRows(entries), [entries]);
  const places = useMemo(() => runsAt(entries), [entries]);
  const names = useMemo(() => pdfNames(entries), [entries]);
  const modelAt = useMemo(() => modelsAt(entries), [entries]);
  const shownText = useRef(new Map<number, string>()).current;
  return (
    <PictureScope session={props.sessionId}>
      <GroupBody
        group={group}
        trailing={props.live}
        live={props.live}
        asOfMs={props.asOfMs}
        show="all"
        model={props.model}
        modelAt={modelAt}
        indexOf={indexOf}
        shownText={shownText}
        onInspectTurn={() => undefined}
        helpers={helpers}
        subagents={subagents}
        places={places}
        pdfNames={names}
        sessionLive={props.live}
        sessionId={props.sessionId}
        onJump={() => undefined}
      />
    </PictureScope>
  );
}

/**
 * A session transcript grouped by turn, oldest first. Closed groups mount nothing until opened. In a live session whose
 * last body is fresh, the open trailing group shows the running tool, the open thinking block and the arriving answer.
 */
export function Transcript(props: TranscriptProps): ReactElement {
  const live = props.live === true;
  const groups = useMemo(() => groupTurns(props.entries, props.turns), [props.entries, props.turns]);
  const indexOf = useMemo(() => new Map(props.entries.map((entry, index) => [entry, index] as const)), [props.entries]);
  const helpers = useMemo(() => runAgents(props.entries), [props.entries]);
  const subagents = useMemo(() => subagentRows(props.entries), [props.entries]);
  const places = useMemo(() => runsAt(props.entries), [props.entries]);
  const names = useMemo(() => pdfNames(props.entries), [props.entries]);
  const modelAt = useMemo(() => modelsAt(props.entries), [props.entries]);
  const shownText = useRef(new Map<number, string>()).current;

  const [query, setQuery] = useState('');
  const [needle, setNeedle] = useState('');
  const [show, setShow] = useState<Show>('all');
  const [openSet, setOpenSet] = useState<ReadonlySet<number>>(() => {
    if (groups.length <= 3) return new Set(groups.map((_group, index) => index));
    const initial = new Set<number>([groups.length - 1]);
    const focused = groups.findIndex((group) => group.number === props.focusTurn);
    if (focused !== -1) initial.add(focused);
    return initial;
  });
  const [lastSeen, setLastSeen] = useState(groups.length - 1);
  const lastIndex = groups.length - 1;
  if (lastIndex !== lastSeen) {
    setLastSeen(lastIndex);
    if (lastIndex > lastSeen && (lastSeen < 0 || openSet.has(lastSeen)))
      setOpenSet((previous) => new Set([...previous, lastIndex]));
  }
  const [scrollTo, setScrollTo] = useState<number | null>(() => {
    const focused = groups.findIndex((group) => group.number === props.focusTurn);
    return focused === -1 ? null : focused;
  });
  const nodes = useRef(new Map<number, HTMLDivElement>()).current;
  // A link to a run's block or end row opens its turn first, then scrolls to it once it is drawn and moves focus there.
  const [jumpAt, setJumpAt] = useState<number | null>(null);
  const jump = (at: number): void => {
    const index = groups.findIndex((group) => group.entries.some((entry) => indexOf.get(entry) === at));
    if (index !== -1) setOpenSet((previous) => new Set([...previous, index]));
    setJumpAt(at);
  };
  useEffect(() => {
    if (jumpAt === null) return;
    const target = document.querySelector(`[data-entry-at="${String(jumpAt)}"]`)?.firstElementChild;
    target?.scrollIntoView?.({ block: 'center' });
    target?.querySelector<HTMLElement>('h3 button, button, a[href]')?.focus({ preventScroll: true });
    setJumpAt(null);
  }, [jumpAt]);

  useEffect(() => {
    const handle = setTimeout(() => setNeedle(query.trim().toLowerCase()), FIND_DEBOUNCE_MS);
    return () => clearTimeout(handle);
  }, [query]);

  const matched = useMemo<GroupView[]>(() => {
    const all = groups.map((group, index) => ({ group, index }));
    if (needle === '') return all;
    return all.filter(({ group }) => group.entries.some((entry) => entryMatches(entry, needle)));
  }, [groups, needle]);

  const [openedFor, setOpenedFor] = useState('');
  if (needle !== openedFor) {
    setOpenedFor(needle);
    if (needle !== '') setOpenSet(new Set(matched.slice(0, FIND_OPEN).map((view) => view.index)));
  }

  const [focusedFor, setFocusedFor] = useState(props.focusTurn ?? null);
  if ((props.focusTurn ?? null) !== focusedFor) {
    setFocusedFor(props.focusTurn ?? null);
    const index = groups.findIndex((group) => group.number === props.focusTurn);
    if (index !== -1) {
      setOpenSet((previous) => new Set([...previous, index]));
      setScrollTo(index);
    }
  }

  useEffect(() => {
    if (scrollTo === null) return;
    nodes.get(scrollTo)?.scrollIntoView?.({ block: 'start' });
    setScrollTo(null);
  }, [scrollTo, nodes]);

  if (props.transcriptError !== undefined && props.entries.length === 0) {
    return (
      <Alert tone="warning" title="Transcript could not be read">
        {`Transcript could not be read: ${props.transcriptError}`}
      </Alert>
    );
  }
  if (props.entries.length === 0) {
    return <EmptyState title="No messages in this session." description="Messages appear after the first prompt." />;
  }

  const allOpen = matched.length > 0 && matched.every((view) => openSet.has(view.index));
  const tooMany = matched.length > EXPAND_ALL_MAX;
  const toggleAll = (): void => {
    setOpenSet(allOpen ? new Set() : new Set(matched.map((view) => view.index)));
  };
  const expandButton = (
    <Button type="button" variant="ghost" size="sm" disabled={tooMany} onClick={toggleAll}>
      {allOpen ? 'Collapse all' : 'Expand all'}
    </Button>
  );

  const page = (
    <div className="grid min-w-0 gap-3">
      {props.transcriptError !== undefined ? (
        <Alert tone="warning" title="Transcript could not be read">
          {`Transcript could not be read: ${props.transcriptError}`}
        </Alert>
      ) : null}
      <div className="flex flex-wrap items-center gap-2">
        <CompactField className="w-60">
          <SearchField
            label="Find in transcript"
            placeholder="Find in transcript…"
            value={query}
            onValueChange={setQuery}
          />
        </CompactField>
        <SegmentedControl
          label="Show"
          options={SHOW_OPTIONS}
          value={show}
          onValueChange={(value) => {
            if (isShow(value)) setShow(value);
          }}
        />
        <span className="text-caption text-secondary tabular-nums" aria-live="polite">
          {needle !== '' ? `${matched.length} of ${groups.length} turns match` : ''}
        </span>
        <span className="ml-auto flex items-center gap-1">
          {tooMany ? (
            <Tooltip content={EXPAND_HINT}>
              <span className="inline-flex">{expandButton}</span>
            </Tooltip>
          ) : (
            expandButton
          )}
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => {
              setOpenSet((previous) => new Set([...previous, lastIndex]));
              setScrollTo(lastIndex);
            }}
          >
            Latest
          </Button>
        </span>
      </div>
      <Panel padding="none" ariaLabel="Transcript">
        {matched.map(({ group, index }) => {
          const trailing = props.sessionLive && index === lastIndex && group.end === null && group.number > 0;
          return (
            <div
              key={index}
              ref={(node) => {
                if (node === null) nodes.delete(index);
                else nodes.set(index, node);
              }}
              data-turn={group.number}
            >
              <Disclosure
                title={<GroupTitle group={group} trailing={trailing} live={live} />}
                meta={<GroupMeta group={group} />}
                open={openSet.has(index)}
                onOpenChange={(next) => {
                  setOpenSet((previous) => {
                    const copy = new Set(previous);
                    if (next) copy.add(index);
                    else copy.delete(index);
                    return copy;
                  });
                }}
              >
                <GroupBody
                  group={group}
                  trailing={trailing}
                  live={live}
                  asOfMs={props.asOfMs}
                  show={show}
                  model={props.model}
                  modelAt={modelAt}
                  indexOf={indexOf}
                  shownText={shownText}
                  onInspectTurn={props.onInspectTurn}
                  helpers={helpers}
                  subagents={subagents}
                  places={places}
                  pdfNames={names}
                  sessionLive={props.sessionLive}
                  sessionId={props.sessionId ?? null}
                  onJump={jump}
                />
              </Disclosure>
            </div>
          );
        })}
      </Panel>
    </div>
  );
  return <PictureScope session={props.sessionId ?? null}>{page}</PictureScope>;
}

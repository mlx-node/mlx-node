import { BellRing, Cpu, GitBranch, RotateCw } from 'lucide-react';
import { useEffect, useMemo, useRef, useState, type MouseEvent, type ReactElement, type ReactNode } from 'react';

import { CHAT } from '../lib/chat-strings';
import { thoughtDuration } from '../lib/chat-turns';
import { formatClock, formatCount, formatCountNoun, formatDuration } from '../lib/format';
import { groupTurns, turnTitle, wakeTitle, type TurnGroup } from '../lib/sessions';
import type { TranscriptEntry, TurnMetric } from '../lib/types';
import { CompactField } from './arc-x/compact-field/compact-field';
import { Disclosure } from './arc-x/disclosure/disclosure';
import { EventRow, RuleLabel } from './arc-x/event-row/event-row';
import { LiveLabel } from './arc-x/live-label/live-label';
import { Markdown } from './arc-x/markdown/markdown';
import { Message } from './arc-x/message/message';
import { ModelBadge } from './arc-x/model-badge/model-badge';
import { Panel } from './arc-x/panel/panel';
import { PlanList } from './arc-x/plan-list/plan-list';
import { ReasoningBlock } from './arc-x/reasoning-block/reasoning-block';
import { ToolCall, ToolCallGroup, type ToolStatus } from './arc-x/tool-call/tool-call';
import { Alert } from './arc/alert/alert';
import { Button } from './arc/button/button';
import { EmptyState } from './arc/empty-state/empty-state';
import { SearchField } from './arc/search-field/search-field';
import SegmentedControl, { type Segment } from './arc/segmented-control/segmented-control';
import { Tooltip } from './arc/tooltip/tooltip';

export interface TranscriptProps {
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

function entryMatches(entry: TranscriptEntry, needle: string): boolean {
  const has = (value: string | undefined): boolean => value !== undefined && value.toLowerCase().includes(needle);
  switch (entry.kind) {
    case 'user':
    case 'assistant':
    case 'thinking':
      return has(entry.text);
    case 'wake':
      return has(entry.text) || has(entry.workflow?.name);
    case 'tool':
      return has(entry.name) || has(entry.title) || has(entry.input) || has(entry.output);
    case 'subagent':
      return has(entry.description);
    case 'plan':
      return entry.entries.some((item) => has(item.content));
    case 'model':
    case 'turn-end':
      return false;
  }
}

function isMessage(entry: TranscriptEntry): boolean {
  return entry.kind !== 'thinking' && entry.kind !== 'tool' && entry.kind !== 'plan' && entry.kind !== 'subagent';
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
}

function GroupBody(props: BodyProps): ReactElement {
  const { group, trailing, live, asOfMs, show } = props;
  const open = trailing && group.end === null;
  const visible = show === 'messages' ? group.entries.filter(isMessage) : group.entries;
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
    if (entry.kind === 'tool') {
      const run: Array<Entry<'tool'>> = [];
      while (i < visible.length && visible[i]!.kind === 'tool') {
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
            input={tool.input}
            output={tool.output}
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
            label={entry.interjection === true ? 'You · interjection' : 'You'}
            ts={entry.ts}
          >
            <Markdown text={entry.text} />
          </Message>,
        );
        break;
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
      case 'assistant': {
        const current = props.modelAt.get(entry) ?? group.metric?.model ?? props.model;
        const streaming = open && entry === tail;
        const fresh = streaming && live ? freshOffset(props.shownText.get(at), entry.text) : undefined;
        if (streaming) rendered.push([at, entry.text]);
        items.push(
          <Message
            key={at}
            role="assistant"
            label={current !== null ? <ModelBadge name={current} size="sm" showLogo /> : 'Assistant'}
            ts={entry.ts}
            streaming={streaming && live}
          >
            <Markdown text={entry.text} freshFrom={fresh} />
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
          />,
        );
        break;
      }
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
    }
  }

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

/**
 * A session transcript grouped by turn, oldest first. Closed groups mount nothing until opened. In a live session whose
 * last body is fresh, the open trailing group shows the running tool, the open thinking block and the arriving answer.
 */
export function Transcript(props: TranscriptProps): ReactElement {
  const live = props.live === true;
  const groups = useMemo(() => groupTurns(props.entries, props.turns), [props.entries, props.turns]);
  const indexOf = useMemo(() => new Map(props.entries.map((entry, index) => [entry, index] as const)), [props.entries]);
  const modelAt = useMemo(() => {
    const map = new Map<TranscriptEntry, string | null>();
    let current: string | null = null;
    for (const entry of props.entries) {
      if (entry.kind === 'model') current = entry.modelId;
      else if (entry.kind === 'assistant') map.set(entry, current);
    }
    return map;
  }, [props.entries]);
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

  return (
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
                />
              </Disclosure>
            </div>
          );
        })}
      </Panel>
    </div>
  );
}

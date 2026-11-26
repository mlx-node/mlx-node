import { Workflow } from 'lucide-react';
import { motion } from 'motion/react';
import { useEffect, useId, useRef, useState, type KeyboardEvent, type ReactElement, type ReactNode } from 'react';
import { Link } from 'react-router-dom';

import { runStoppable, transcriptOpens, transcriptPath } from '../lib/activity';
import { agentKeyOf } from '../lib/agent-keys';
import {
  type AgentAsk,
  agentButtons,
  agentLiveOf,
  anyButton,
  type AgentNow,
  agentsNow,
  type AgentRow as AgentRowData,
  stoppedByYou,
} from '../lib/agents';
import { CHAT, fill } from '../lib/chat-strings';
import { formatElapsed } from '../lib/chat-turns';
import type { HelperProgress, LiveActivity, RunBy, RunView } from '../lib/chat-types';
import { formatClock, formatCount, formatDuration } from '../lib/format';
import { rowWords } from '../lib/helper-words';
import { type AllowedMark, type DeniedMark, subagentAllowedText, subagentDeniedText } from '../lib/permission-text';
import {
  agentGroups,
  BUDGET_METER_FROM,
  budgetWords,
  chipAgents,
  countWords,
  logCount,
  logSteps,
  nowLine,
  reportCut,
  reportPlace,
  runAnnouncements,
  runCounts,
  type RunEntry,
  runElapsed,
  runSnapshot,
  type RunSnapshot,
  runState,
  type RunState,
  stepChips,
  type StepChip,
  stepOf,
  stepWords,
  stopText,
  summaryWords,
} from '../lib/run-view';
import type { TranscriptEntry } from '../lib/types';
import { launchDetails } from '../lib/work-tools';
import {
  AgentControls,
  type AgentHandlers,
  CONTROL_KEYS,
  pressControl,
  useFocusFallback,
} from './arc-x/agent-controls/agent-controls';
import { AgentList, type AgentListGroup } from './arc-x/agent-list/agent-list';
import { AgentRow } from './arc-x/agent-row/agent-row';
import { ChatMarkdown } from './arc-x/chat-markdown/chat-markdown';
import { CodePanel } from './arc-x/code-panel/code-panel';
import { ConfirmPopover } from './arc-x/confirm-popover/confirm-popover';
import { Disclosure } from './arc-x/disclosure/disclosure';
import { KeyValue, type KeyValueItem } from './arc-x/key-value/key-value';
import { LiveDot } from './arc-x/live-dot/live-dot';
import { useStaticMotion } from './arc-x/motion/use-static-motion';
import { PhaseRail } from './arc-x/phase-rail/phase-rail';
import { Meter } from './arc-x/split-bar/split-bar';
import { AnimatedCounter } from './arc/animated-counter/animated-counter';
import { Button } from './arc/button/button';
import { ExpandableText } from './arc/expandable-text/expandable-text';
import { motionTokens } from './arc/motion-tokens';
import { HelperMarkLists } from './chat-permission-marks';
import styles from './run-block.module.css';
import { useNow } from './wait-clock';

type ToolEntry = Extract<TranscriptEntry, { kind: 'tool' }>;

export interface HelperMarks {
  allowed: readonly AllowedMark[];
  denied: readonly DeniedMark[];
}

/**
 * Work: what an agent's buttons and its row do, the host's live facts they read, the agent whose view is open and the
 * row whose permission lists the view asked to open.
 */
export interface BlockControls {
  live: LiveActivity | undefined;
  handlers: AgentHandlers;
  openId: string | null;
  marksOpen: { id: string; tick: number } | null;
}

export interface RunBlockProps {
  runId: string;
  /** The run's newest update; null for a run only spawn lines name (no update of it is here). */
  entry: RunEntry | null;
  /** Its agents ({@link runAgents}): no other row is drawn for them. */
  agents: readonly AgentRowData[];
  /** The thread of the open conversation, where the host's runs are known. False on the Sessions page. */
  live: boolean;
  /** The host's view of this run, live or ended (the thread only). */
  view?: RunView;
  /** The agent process holds the conversation (the thread) or the stored session (the Sessions page). */
  sessionLive?: boolean;
  /** Who stopped or paused it ({@link runBy}). */
  by: RunBy | null;
  /** The run's wake rows: where its report lives. */
  wakes: readonly TranscriptEntry[];
  /** The newest wake row of the run, by its entry index: `The result ↓`; null when none is loaded. */
  resultAt: number | null;
  onJump?: (at: number) => void;
  /** The main agent's launch call, and the words of the mark a mode left on it. */
  launch?: ToolEntry | null;
  launchMark?: string | null;
  /** Open requests of its agents. */
  asks?: readonly AgentAsk[];
  /** The live ticks of the conversation's agents. */
  progress?: readonly HelperProgress[];
  /** Stop run: offered while the host holds it live or waiting on the person, and no stop was sent. */
  onStop?: () => Promise<void>;
  /** Work: the agents' buttons and rows, and what they read. */
  agentControls?: BlockControls;
  /** The calls a mode allowed for an agent and those the user denied with a note (the thread only). */
  marksOf?: (row: AgentRowData) => HelperMarks;
  temp?: string | null;
  /** The polite region. */
  onStatus?: (sentence: string) => void;
  /** Sessions page: where its conversation opens in Work. */
  workHref?: string;
  /** Changes when a jump lands on the block: it opens and its head takes focus. */
  focusTick?: number;
}

const INSTANT = { duration: 0 } as const;

/** `Done ({n})` with its count ticking. */
function countTitle(template: string, n: number): ReactNode {
  const at = template.indexOf('{n}');
  if (at === -1) return template;
  return (
    <>
      {template.slice(0, at)}
      <AnimatedCounter value={n} />
      {template.slice(at + 3)}
    </>
  );
}

function chipCount(chip: StepChip): string | null {
  if (chip.total === 0) return null;
  return chip.done === chip.total
    ? formatCount(chip.done)
    : fill(CHAT.run.chipCount, { done: formatCount(chip.done), total: formatCount(chip.total) });
}

/** An end that closes the list: the run's agents no longer run. */
function endedOf(state: RunState | null): boolean {
  return state !== null && !state.working && state.kind !== 'running';
}

function BlockRow(props: {
  agent: AgentNow;
  live: boolean;
  progress: HelperProgress | null;
  now: number;
  ask: AgentAsk | null;
  marks: HelperMarks | null;
  temp: string | null;
  controls?: BlockControls;
}): ReactElement {
  const { agent, progress, controls } = props;
  const row = agent.row;
  const entry = row.entry;
  const opens = props.live
    ? transcriptOpens(entry?.exists, progress, entry?.endTs !== undefined ? entry : undefined)
    : entry?.exists === true;
  const allowed = props.marks !== null && props.marks.allowed.length > 0 ? props.marks.allowed : null;
  const denied = props.marks !== null && props.marks.denied.length > 0 ? props.marks.denied : null;
  const more: string[] = [];
  if (allowed !== null) more.push(subagentAllowedText(allowed));
  if (denied !== null) more.push(subagentDeniedText(denied.length));
  const held = controls === undefined ? null : agentLiveOf(controls.live, row);
  const words = rowWords(agent, {
    progress,
    now: props.now,
    ask: props.ask === null ? null : { title: props.ask.title, since: props.ask.openedAt },
    stopping: held?.stopping === true,
    stoppedByYou: controls !== undefined && stoppedByYou(controls.live, row),
  });
  const buttons = agentButtons({ workflow: true, state: agent.state, held, askId: props.ask?.id ?? null });
  const target = { id: row.childId, title: row.label };
  const marksOpen = controls?.marksOpen ?? null;
  const openId = controls?.openId ?? null;
  return (
    <AgentRow
      id={row.childId}
      label={row.label}
      words={words}
      {...(controls !== undefined
        ? {
            onOpen: (opener: HTMLElement) => controls.handlers.onOpen(target, opener, false),
            open: openId !== null && row.attempts.includes(openId),
          }
        : { to: opens ? transcriptPath(row.childId) : undefined })}
      openLabel={fill(CHAT.agent.open, { label: row.label })}
      more={more}
      controls={
        controls === undefined || !anyButton(buttons) ? undefined : (
          <AgentControls
            id={row.childId}
            label={row.label}
            workflow
            buttons={buttons}
            onStop={() => controls.handlers.onStop(target)}
            onReview={controls.handlers.onReview}
          />
        )
      }
    >
      {allowed !== null || denied !== null ? (
        <HelperMarkLists
          allowed={allowed}
          denied={denied}
          temp={props.temp}
          openTick={marksOpen !== null && row.attempts.includes(marksOpen.id) ? marksOpen.tick : 0}
        />
      ) : null}
    </AgentRow>
  );
}

/** The run's Stop in its head: once it goes (the run stops), its head takes focus. */
function HeadStop(props: { head: { current: HTMLButtonElement | null }; children: ReactNode }): ReactElement {
  const ref = useRef<HTMLSpanElement>(null);
  const head = props.head;
  useFocusFallback(ref, () => () => head.current);
  return (
    <span ref={ref} className={styles.stop} data-agent-stop="">
      {props.children}
    </span>
  );
}

function LaunchFold({ launch, mark, runId }: { launch: ToolEntry; mark: string | null; runId: string }): ReactElement {
  const l = CHAT.run.launch;
  const details = launchDetails(launch);
  const items: KeyValueItem[] = [];
  if (details !== null) {
    items.push({
      key: 'source',
      label: l.source,
      value: details.value === null ? l.sources[details.source] : `${l.sources[details.source]} · ${details.value}`,
      mono: details.source === 'script_path',
    });
    if (details.budget !== null) items.push({ key: 'budget', label: l.budget, value: formatCount(details.budget) });
  }
  if (mark !== null) items.push({ key: 'permission', label: l.permission, value: mark });
  if (launch.ts > 0) items.push({ key: 'started', label: l.started, value: formatClock(launch.ts, true) });
  items.push({ key: 'run', label: l.runId, value: runId, mono: true, copy: runId });
  return (
    <div className="grid min-w-0 gap-2">
      <KeyValue layout="rows" items={items} />
      {details?.args !== null && details?.args !== undefined ? (
        <CodePanel code={details.args} label={l.args} maxLines={8} wrap />
      ) : null}
    </div>
  );
}

/**
 * One workflow run where its launch call stands: its head (name, state, clock, Stop), its goal, where it is and what it
 * used, its steps as chips that filter the list, its now line, its agents by need inside a fixed height, then its
 * log, its launch and, when no end row holds it, its report. The thread and the Sessions page draw the same block.
 */
export function RunBlock(props: RunBlockProps): ReactElement {
  const { entry, view, live } = props;
  const staticMotion = useStaticMotion();
  const headId = useId();
  const bodyId = useId();
  const root = useRef<HTMLElement>(null);
  const head = useRef<HTMLButtonElement>(null);
  const runLive =
    entry !== null && entry.status === 'active' && (live ? view?.live === true : props.sessionLive === true);
  const draining = live && entry !== null && view?.stopping === true;
  const open = entry === null ? props.sessionLive === true : runLive || draining;
  const ticks = live && (view !== undefined || entry === null) ? (props.progress ?? []) : [];
  const now = useNow(live && open);
  const agents = agentsNow(props.agents, { live: live && open, open, now });
  const asks = props.asks ?? [];
  const askOf = new Map<string, AgentAsk>();
  for (const ask of asks) if (!askOf.has(ask.child)) askOf.set(ask.child, ask);
  const firstAsk = asks[0] ?? null;
  const state =
    entry === null
      ? null
      : runState(entry, agents, {
          live,
          view,
          asking: firstAsk === null ? null : { label: firstAsk.label, access: firstAsk.access },
          by: props.by,
          sessionLive: props.sessionLive,
        });
  const ended = endedOf(state);

  const [bodyOpen, setBodyOpen] = useState(true);
  const [selected, setSelected] = useState<string | null>(null);
  const [logOpen, setLogOpen] = useState(false);
  const [doneOpen, setDoneOpen] = useState(false);
  const [touched, setTouched] = useState(false);

  const focusTick = props.focusTick;
  useEffect(() => {
    if (focusTick === undefined || focusTick === 0) return;
    setBodyOpen(true);
    head.current?.focus({ preventScroll: true });
  }, [focusTick]);

  const said = useRef<RunSnapshot | null>(null);
  const onStatus = props.onStatus;
  useEffect(() => {
    if (entry === null || state === null || !live) return;
    const snap = runSnapshot(entry, agents, state);
    for (const sentence of runAnnouncements(entry, agents, said.current, snap)) onStatus?.(sentence);
    said.current = snap;
  });

  const chips = entry === null ? [] : stepChips(entry, agents, state!);
  const chip = selected === null ? undefined : chips.find((item) => item.title === selected);
  const shown = chip === undefined ? agents : chipAgents(agents, chip);
  const tickOf = (agent: AgentNow): HelperProgress | null =>
    agent.state === 'running' ? (ticks.find((tick) => tick.childId === agent.row.childId) ?? null) : null;
  const node = (agent: AgentNow): { id: string; node: ReactNode } => ({
    id: agent.row.childId,
    node: (
      <BlockRow
        agent={agent}
        live={live}
        progress={tickOf(agent)}
        now={now}
        ask={askOf.get(agent.row.childId) ?? null}
        marks={props.marksOf?.(agent.row) ?? null}
        temp={props.temp ?? null}
        controls={live ? props.agentControls : undefined}
      />
    ),
  });
  const asking = new Set(asks.map((ask) => ask.child));
  const groups = agentGroups(shown, asking);
  const g = CHAT.run.groups;
  const list: AgentListGroup[] = [];
  if (groups.attention.length > 0)
    list.push({
      key: 'attention',
      title: fill(g.attention, { n: formatCount(groups.attention.length) }),
      mode: 'open',
      rows: groups.attention.map(node),
    });
  if (ended && !touched) {
    const rest = [...groups.running, ...groups.done];
    if (rest.length > 0)
      list.push({
        key: 'agents',
        title: fill(g.agents, { n: formatCount(rest.length) }),
        mode: 'fold',
        rows: rest.map(node),
        open: doneOpen,
        onOpenChange: (next) => {
          setDoneOpen(next);
        },
      });
  } else {
    if (groups.running.length > 0)
      list.push({
        key: 'running',
        title: fill(g.running, { n: formatCount(groups.running.length) }),
        mode: 'open',
        rows: groups.running.map(node),
      });
    if (groups.done.length > 0)
      list.push({
        key: 'done',
        title: countTitle(g.done, groups.done.length),
        mode: 'fold',
        rows: groups.done.map(node),
        open: doneOpen,
        onOpenChange: (next) => {
          setDoneOpen(next);
          if (!ended) setTouched(true);
        },
      });
  }

  const counts = runCounts(agents);
  const failedAt = agents.find((agent) => agent.state === 'failed')?.row.childId ?? null;
  const scrollToFailed = (): void => {
    if (failedAt === null) return;
    setSelected(null);
    root.current?.querySelector(`[data-agent-id="${CSS.escape(failedAt)}"]`)?.scrollIntoView({ block: 'nearest' });
  };

  const step = entry === null ? null : stepOf(entry);
  const steps = entry === null ? [] : logSteps(entry);
  const logs = logCount(steps);
  const nowText = entry !== null && state !== null && state.working ? nowLine(entry) : null;
  const place =
    entry === null ? { home: null, writingUp: false } : reportPlace(entry, props.wakes, view?.wakeDue === true);
  const budget = entry === null ? null : budgetWords(entry);
  const lines = state === null ? [] : state.line === state.word ? [...state.notes] : [state.line, ...state.notes];
  if (place.writingUp) lines.push(CHAT.run.writingUp);
  const workHref = !live && state?.kind === 'running' ? props.workHref : undefined;
  // A status update reaches the page before the host's runs event: no Stop until both say the same status.
  const canStop =
    live &&
    view !== undefined &&
    runStoppable(view) &&
    (entry === null || entry.status === view.status) &&
    props.onStop !== undefined;
  const review = state?.button === 'review' && firstAsk !== null ? props.agentControls?.handlers.onReview : undefined;
  const onHeadKey = (event: KeyboardEvent<HTMLElement>): void => {
    if (event.defaultPrevented || !(event.target instanceof HTMLElement)) return;
    if (!event.currentTarget.contains(event.target)) return;
    if (agentKeyOf(event, 'run-head', { pause: false }) !== 'stop') return;
    if (pressControl(event.currentTarget, 'stop')) event.preventDefault();
  };
  const elapsed =
    entry === null
      ? null
      : runLive && live
        ? formatElapsed(runElapsed(entry, true, now))
        : entry.elapsedMs > 0
          ? formatDuration(entry.elapsedMs)
          : null;
  const name = entry?.name ?? CHAT.run.bare;
  const countParts = countWords({ ...counts, failed: 0 });

  return (
    <section
      ref={root}
      className={styles.root}
      aria-labelledby={headId}
      aria-busy={state?.working === true ? true : undefined}
      data-run={props.runId}
      data-run-status={entry?.status ?? 'bare'}
      data-run-state={state?.kind ?? 'bare'}
    >
      <header className={styles.head} onKeyDown={onHeadKey}>
        <h3 className={styles.name} id={headId}>
          <button
            ref={head}
            type="button"
            className={styles.nameButton}
            aria-expanded={bodyOpen}
            aria-controls={bodyId}
            onClick={() => setBodyOpen((value) => !value)}
          >
            <Workflow size={14} strokeWidth={1.75} aria-hidden="true" className={styles.icon} />
            {entry !== null ? <span className="sr-only">{CHAT.run.namePrefix}</span> : null}
            <span className={styles.nameText} title={name}>
              {name}
            </span>
          </button>
        </h3>
        <div className={styles.state}>
          {state !== null ? (
            <LiveDot
              tone={state.tone}
              pulse={state.pulse && live}
              label={state.word}
              showLabel
              shimmer={state.pulse && live}
            />
          ) : null}
          {elapsed !== null ? <span className={styles.clock}>{elapsed}</span> : null}
          {canStop ? (
            <HeadStop head={head}>
              <ConfirmPopover
                trigger={{
                  kind: 'button',
                  label: CHAT.run.buttons.stop,
                  size: 'sm',
                  variant: 'secondary',
                  keys: CONTROL_KEYS.stop,
                }}
                title={CHAT.run.stopTitle}
                description={stopText(counts.running)}
                confirmLabel={CHAT.run.buttons.stopWorkflow}
                cancelLabel={CHAT.run.buttons.keepRunning}
                workingLabel={CHAT.run.status.stopping}
                successLabel={CHAT.run.status.stopping}
                errorTitle={CHAT.run.stopError}
                onConfirm={props.onStop!}
              />
            </HeadStop>
          ) : null}
        </div>
      </header>
      {lines.length > 0 || workHref !== undefined ? (
        <div className={styles.status} data-tone={state?.tone ?? 'neutral'}>
          {lines.map((line, n) => (
            <p key={n} className={styles.statusLine}>
              {line}
              {n === 0 && review !== undefined && firstAsk !== null ? (
                <Button
                  type="button"
                  size="sm"
                  variant="secondary"
                  className={styles.review}
                  aria-label={fill(CHAT.agent.review, { label: firstAsk.label })}
                  onClick={() => review(firstAsk.id)}
                >
                  {CHAT.run.buttons.review}
                </Button>
              ) : null}
            </p>
          ))}
          {workHref !== undefined ? (
            <Link className="link" to={workHref}>
              {CHAT.run.openInWork}
            </Link>
          ) : null}
        </div>
      ) : null}
      <motion.div
        id={bodyId}
        className={styles.bodyClip}
        hidden={!bodyOpen && staticMotion}
        initial={false}
        animate={{ height: bodyOpen ? 'auto' : 0, opacity: bodyOpen ? 1 : 0 }}
        transition={staticMotion ? INSTANT : motionTokens.spring.smooth}
      >
        <div className={styles.body}>
          {entry?.objective !== undefined && entry.objective !== '' ? (
            <ExpandableText lines={2} moreLabel={CHAT.run.showMore} lessLabel={CHAT.run.showLess}>
              <p className={styles.goal}>{entry.objective}</p>
            </ExpandableText>
          ) : null}
          {entry !== null ? (
            <div className={styles.glance}>
              <span className={styles.glanceWords}>
                {ended && state?.kind !== 'not-finished'
                  ? summaryWords(entry, agents).join(' · ')
                  : [stepWords(step), ...countParts].filter((part) => part !== null).join(' · ')}
                {!ended && counts.failed > 0 ? (
                  <>
                    {[stepWords(step), ...countParts].some((part) => part !== null) ? ' · ' : ''}
                    <button type="button" className={styles.failedLink} onClick={scrollToFailed}>
                      {fill(CHAT.run.counts.failed, { n: formatCount(counts.failed) })}
                    </button>
                  </>
                ) : null}
              </span>
              {budget !== null ? (
                <span className={styles.budget}>
                  <span>{budget.text}</span>
                  {budget.meter && entry.agentBudget !== null ? (
                    <Meter
                      label={budget.text}
                      value={entry.agentsUsed}
                      max={entry.agentBudget}
                      formatValue={() => budget.text}
                      size="dense"
                      warnAt={BUDGET_METER_FROM}
                      className={styles.meter}
                    />
                  ) : null}
                </span>
              ) : null}
            </div>
          ) : (
            <p className={styles.note}>{CHAT.run.noLaunch}</p>
          )}
          {chips.length > 0 ? (
            <PhaseRail
              label={fill(CHAT.run.stepsLabel, { name })}
              chips={chips}
              selected={chip === undefined ? null : chip.title}
              onSelect={setSelected}
              stepWords={state?.working === true ? stepWords(step) : null}
              countOf={chipCount}
            />
          ) : null}
          {nowText !== null && logs > 0 ? (
            <button type="button" className={styles.now} onClick={() => setLogOpen(true)}>
              <span className={styles.nowText}>{nowText.text}</span>
              {nowText.ts !== null ? <span className={styles.clock}>{formatClock(nowText.ts)}</span> : null}
            </button>
          ) : nowText !== null ? (
            <p className={styles.now}>
              <span className={styles.nowText}>{nowText.text}</span>
              {nowText.ts !== null ? <span className={styles.clock}>{formatClock(nowText.ts)}</span> : null}
            </p>
          ) : null}
          {list.length > 0 ? (
            <AgentList
              label={fill(CHAT.run.listLabel, { name })}
              groups={list}
              onScrolled={() => {
                if (!ended) setTouched(true);
              }}
            />
          ) : null}
          {logs > 0 || props.launch != null || place.home === 'block' || (ended && props.resultAt !== null) ? (
            <div className={styles.folds}>
              {logs > 0 ? (
                <Disclosure
                  title={fill(CHAT.run.links.log, { n: formatCount(logs) })}
                  open={logOpen}
                  onOpenChange={setLogOpen}
                >
                  <div className="grid min-w-0 gap-2">
                    {steps.map((item, n) => (
                      <div key={n} className="grid min-w-0 gap-1">
                        {item.title !== null ? (
                          <p className={styles.logHead}>
                            {item.detail === null ? item.title : `${item.title} · ${item.detail}`}
                          </p>
                        ) : null}
                        {item.lines.map((line, k) => (
                          <p key={k} className={styles.logLine}>
                            <span className={styles.clock}>{formatClock(line.ts, true)}</span>
                            <span>{line.text}</span>
                          </p>
                        ))}
                      </div>
                    ))}
                  </div>
                </Disclosure>
              ) : null}
              {props.launch != null ? (
                <Disclosure title={CHAT.run.links.launch}>
                  <LaunchFold launch={props.launch} mark={props.launchMark ?? null} runId={props.runId} />
                </Disclosure>
              ) : null}
              {place.home === 'block' && entry?.report !== undefined ? (
                <Disclosure title={CHAT.run.links.report} defaultOpen>
                  <div className="grid min-w-0 gap-2">
                    <ChatMarkdown text={entry.report} links={{ mode: 'work' }} />
                    {reportCut(entry.report) ? <p className={styles.note}>{CHAT.run.reportCut.note}</p> : null}
                  </div>
                </Disclosure>
              ) : null}
              {ended && props.resultAt !== null && props.onJump !== undefined ? (
                <button type="button" className={styles.jump} onClick={() => props.onJump?.(props.resultAt!)}>
                  {CHAT.run.links.result}
                </button>
              ) : null}
            </div>
          ) : null}
        </div>
      </motion.div>
    </section>
  );
}

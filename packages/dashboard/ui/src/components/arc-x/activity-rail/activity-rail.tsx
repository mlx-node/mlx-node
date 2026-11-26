import {
  Check,
  CircleHelp,
  Clock,
  GitBranch,
  Radar,
  Square,
  SquareTerminal,
  Target,
  Trash2,
  Workflow,
  X,
} from 'lucide-react';
import {
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  type KeyboardEvent,
  type ReactElement,
  type ReactNode,
} from 'react';

import {
  type Activity,
  type ActivityRow,
  type AgentRailRow,
  askOfAgent,
  type AskRow,
  type EntryAnchor,
  eventsWords,
  runRailDetail,
  type RunRailRow,
  type ScheduleRow,
  type TaskRow,
} from '../../../lib/activity';
import { agentKeyOf } from '../../../lib/agent-keys';
import { agentButtons, anyButton } from '../../../lib/agents';
import { CHAT, fill } from '../../../lib/chat-strings';
import type { HelperProgress } from '../../../lib/chat-types';
import { formatClock } from '../../../lib/format';
import { goalMeta, goalStateWords, type GoalView } from '../../../lib/goal';
import { stopText } from '../../../lib/run-view';
import { visibleText } from '../../../lib/work-tools';
import { useNow, WaitClock } from '../../wait-clock';
import { AgentControls, type AgentHandlers, CONTROL_KEYS, pressControl } from '../agent-controls/agent-controls';
import { useAgentRoving } from '../agent-list/agent-list';
import { ConfirmPopover } from '../confirm-popover/confirm-popover';
import { IconButton } from '../icon-button/icon-button';
import { LiveDot } from '../live-dot/live-dot';
import { PlanList } from '../plan-list/plan-list';
import styles from './activity-rail.module.css';
import { AskRowView } from './ask-row';
import { ProgressLine } from './progress-line';

export interface ActivityRailProps {
  activity: Activity;
  /** Scrolls the thread to the card of that entry. */
  onJump: (anchor: EntryAnchor) => void;
  /** Brings the card of that open request to the front, scrolls to it and focuses it. */
  onAsk: (id: string) => void;
  /** Work: the confirmed Stop of a command or monitor. Absent: its rows offer no Stop. */
  onStopTask?: (row: TaskRow) => void;
  /**
   * Work: what an agent's buttons do (a subagent's and a run's agents'), and its row's click, which opens its Agent
   * view. Absent: agent rows offer no buttons and jump to the thread.
   */
  agents?: AgentHandlers;
  /** Work: Stop run; rejects with the reason to show. Absent: run rows offer no Stop. */
  onStopRun?: (runId: string) => Promise<void>;
  /** Work: the confirmed Delete of a schedule. Absent: its rows offer no Delete. */
  onDeleteSchedule?: (row: ScheduleRow) => void;
  /** Why no schedule can be deleted now; its Delete stays, off, with this as its tooltip and description. */
  deleteOff?: string | null;
  /** The rail sits beside the thread: row buttons show on hover and focus. In the drawer they always show. */
  docked?: boolean;
}

const ICONS: Record<ActivityRow['kind'], ReactNode> = {
  subagent: <GitBranch size={14} strokeWidth={1.75} aria-hidden="true" />,
  workflow: <Workflow size={14} strokeWidth={1.75} aria-hidden="true" />,
  command: <SquareTerminal size={14} strokeWidth={1.75} aria-hidden="true" />,
  monitor: <Radar size={14} strokeWidth={1.75} aria-hidden="true" />,
};

/** The second line of a running row: its kind and what the kind adds. */
function detailOf(row: ActivityRow): string {
  const kind: string = CHAT.activity.kinds[row.kind];
  switch (row.kind) {
    case 'subagent': {
      if (row.stopping) return `${kind} · ${CHAT.activity.stopping}`;
      const parts = [kind];
      if (row.type !== '') parts.push(row.type);
      if (row.progress !== null && row.progress.turns > 0)
        parts.push(fill(CHAT.activity.progressTurn, { n: row.progress.turns }));
      return parts.join(' · ');
    }
    case 'workflow':
      return runRailDetail(row);
    case 'command':
    case 'monitor':
      if (row.stopping) return `${kind} · ${CHAT.activity.stopping}`;
      return row.events > 0 ? `${kind} · ${eventsWords(row.events)}` : kind;
  }
}

/** The first of `targets` still in the page takes focus. */
function focusFirst(targets: ReadonlyArray<HTMLElement | null>): void {
  for (const target of targets) {
    if (target?.isConnected === true) {
      target.focus({ preventScroll: true });
      return;
    }
  }
}

/**
 * The row holds focus, or a popover one of its buttons opened does: the popover is drawn outside the row, and its
 * button names it by `aria-controls`.
 */
function holdsFocus(item: HTMLElement): boolean {
  const active = document.activeElement;
  if (active === null) return false;
  if (item.contains(active)) return true;
  const dialog = active.closest('[role="dialog"]');
  if (dialog === null || dialog.id === '') return false;
  for (const button of item.querySelectorAll('[aria-controls]')) {
    if (button.getAttribute('aria-controls') === dialog.id) return true;
  }
  return false;
}

/** A row's own buttons beside its jump button. When they go while they hold focus, focus goes to `fallback`. */
function RowActions(props: {
  children: ReactNode;
  fallback: () => HTMLElement | null;
  spanRef?: { current: HTMLSpanElement | null };
}): ReactElement {
  const ref = useRef<HTMLSpanElement | null>(null);
  const spanRef = props.spanRef;
  const fallback = useRef(props.fallback);
  fallback.current = props.fallback;
  // A layout cleanup: focus is still inside before React removes the node.
  useLayoutEffect(() => {
    const span = ref.current;
    return () => {
      if (span === null || !span.contains(document.activeElement)) return;
      queueMicrotask(() => focusFirst([fallback.current()]));
    };
  }, []);
  return (
    <span
      ref={(node) => {
        ref.current = node;
        if (spanRef !== undefined) spanRef.current = node;
      }}
      className={styles.actions}
    >
      {props.children}
    </span>
  );
}

function Row(props: {
  icon: ReactNode;
  title: string;
  /** None: the title and its live line only (an agent under its run). */
  detail?: string;
  meta: ReactNode;
  anchor: EntryAnchor;
  onJump: (anchor: EntryAnchor) => void;
  /** A helper's live line under the detail, read once as the button's description. */
  progress?: { value: HelperProgress | null; now: number };
  /** The detail holds a count the jump button's name lacks (a monitor's events): it is the button's description. */
  describeDetail?: boolean;
  /** A monitor's newest output line under the detail; not read (the detail has its count). */
  lastEvent?: string | null;
  /** The row's own button (Stop, Delete), a sibling of the jump button: a button inside a button is not valid. */
  action?: ReactNode;
  /** An agent's or a run's row: its buttons always show, X and M open them. */
  pinned?: boolean;
  /** An agent's row: the child session of its newest attempt, and its click, which opens its Agent view. */
  agent?: { id: string; onOpen: (opener: HTMLElement) => void };
  /** An agent's row: its jump button is a row of its list's one tab stop (↑ ↓ Home End). */
  roving?: boolean;
  /** Under the row, as wide as it: a run's agents. */
  children?: ReactNode;
}): ReactElement {
  const progressId = useId();
  const detailId = useId();
  const agent = props.agent;
  const itemRef = useRef<HTMLLIElement | null>(null);
  const jumpRef = useRef<HTMLButtonElement | null>(null);
  const actionsRef = useRef<HTMLSpanElement | null>(null);
  // Only from the row's own jump button or buttons: a key on a row under it is that row's.
  const onKeyDown = (event: KeyboardEvent<HTMLLIElement>): void => {
    const target = event.target;
    if (event.defaultPrevented || props.pinned !== true || !(target instanceof Node)) return;
    if (jumpRef.current?.contains(target) !== true && actionsRef.current?.contains(target) !== true) return;
    const action = agentKeyOf(event, 'row', { pause: false });
    if (action !== 'stop' && action !== 'message') return;
    const actions = actionsRef.current;
    if (actions !== null && pressControl(actions, action)) event.preventDefault();
  };
  // The row leaves while it or its popover holds focus: the next row's jump button takes it, else the section heading,
  // else the rail.
  useLayoutEffect(() => {
    const item = itemRef.current;
    return () => {
      if (item === null || !holdsFocus(item)) return;
      const next = item.nextElementSibling?.querySelector<HTMLElement>('[data-jump]') ?? null;
      const heading = item.closest('section')?.querySelector<HTMLElement>('[data-heading]') ?? null;
      const rail = item.closest<HTMLElement>('[data-activity-rail]');
      queueMicrotask(() => focusFirst([next, heading, rail]));
    };
  }, []);
  // The action went while its confirm held focus (the confirm is drawn outside the row): the jump button takes it.
  const hasAction = props.action !== undefined;
  const hadAction = useRef(hasAction);
  useEffect(() => {
    const had = hadAction.current;
    hadAction.current = hasAction;
    if (!had || hasAction) return;
    const timer = setTimeout(() => {
      const active = document.activeElement;
      if (active === null || active === document.body) focusFirst([jumpRef.current]);
    }, 0);
    return () => clearTimeout(timer);
  }, [hasAction]);
  return (
    <li
      ref={itemRef}
      className={styles.item}
      data-actions={hasAction ? '' : undefined}
      data-pinned={props.pinned === true ? '' : undefined}
      data-agent-id={agent?.id}
      onKeyDown={props.pinned === true ? onKeyDown : undefined}
    >
      <button
        ref={jumpRef}
        type="button"
        className={styles.row}
        data-jump=""
        data-agent-focus={props.roving === true ? '' : undefined}
        aria-label={
          agent !== undefined
            ? fill(CHAT.agent.open, { label: props.title })
            : fill(CHAT.activity.jump, { title: props.title })
        }
        aria-describedby={
          props.progress !== undefined ? progressId : props.describeDetail === true ? detailId : undefined
        }
        onClick={(event) => (agent !== undefined ? agent.onOpen(event.currentTarget) : props.onJump(props.anchor))}
      >
        <span className={styles.icon}>{props.icon}</span>
        <span className={styles.text}>
          <span className={styles.title} title={props.title}>
            {props.title}
          </span>
          {props.detail !== undefined ? (
            <span id={detailId} className={styles.detail}>
              {props.detail}
            </span>
          ) : null}
          {props.progress === undefined ? null : (
            <ProgressLine id={progressId} progress={props.progress.value} now={props.progress.now} />
          )}
          {props.lastEvent === undefined || props.lastEvent === null ? null : (
            <span className={styles.event} aria-hidden="true" data-last-event="" title={visibleText(props.lastEvent)}>
              {visibleText(props.lastEvent)}
            </span>
          )}
        </span>
        {props.meta !== null ? <span className={styles.meta}>{props.meta}</span> : null}
      </button>
      {props.action === undefined ? null : (
        <RowActions fallback={() => jumpRef.current} spanRef={actionsRef}>
          {props.action}
        </RowActions>
      )}
      {props.children}
    </li>
  );
}

/** A command's or monitor's Stop: a confirm at the button; the press goes to the page and returns at once. */
function StopTask(props: { row: TaskRow; onStop: (row: TaskRow) => void }): ReactElement {
  const { row } = props;
  return (
    <ConfirmPopover
      trigger={{
        kind: 'icon',
        label: fill(CHAT.activity.stop, { title: row.title }),
        icon: <Square size={12} strokeWidth={2} aria-hidden="true" />,
      }}
      align="end"
      title={row.kind === 'command' ? CHAT.activity.stopTitleCommand : CHAT.activity.stopTitleMonitor}
      description={CHAT.activity.stopText}
      confirmLabel={CHAT.activity.stopConfirm}
      cancelLabel={CHAT.activity.stopKeep}
      workingLabel={CHAT.activity.stopping}
      successLabel={CHAT.activity.stopping}
      onConfirm={async () => props.onStop(row)}
    />
  );
}

/** An agent's buttons on the rail: the same set as on its thread row, 32 px in the drawer. Null with none. */
function agentActions(
  row: AgentRailRow,
  workflow: boolean,
  needs: readonly AskRow[],
  handlers: AgentHandlers | undefined,
  large: boolean,
): ReactNode {
  if (handlers === undefined) return null;
  const buttons = agentButtons({
    workflow,
    state: 'running',
    held: row.live ? { stopping: row.stopping, messages: row.messages ?? 'no-address' } : null,
    askId: askOfAgent(needs, row),
  });
  if (!anyButton(buttons)) return null;
  const target = { id: row.id, title: row.title };
  return (
    <AgentControls
      id={row.id}
      label={row.title}
      workflow={workflow}
      buttons={buttons}
      onStop={() => handlers.onStop(target)}
      onReview={handlers.onReview}
      onMessage={(opener) => handlers.onOpen(target, opener, true)}
      large={large}
    />
  );
}

/** An agent row's click: its Agent view, when the page has one. */
function agentOpen(
  row: AgentRailRow,
  handlers: AgentHandlers | undefined,
): { agent: { id: string; onOpen: (opener: HTMLElement) => void } } | Record<string, never> {
  if (handlers === undefined) return {};
  const target = { id: row.id, title: row.title };
  return { agent: { id: row.id, onOpen: (opener) => handlers.onOpen(target, opener, false) } };
}

/** A run's Stop on its row: the same confirm as on its block. */
function StopRun(props: { row: RunRailRow; onStop: (runId: string) => Promise<void>; large: boolean }): ReactElement {
  const { row } = props;
  return (
    <span className={styles.runStop} data-agent-stop="" data-large={props.large ? '' : undefined}>
      <ConfirmPopover
        trigger={{
          kind: 'icon',
          label: fill(CHAT.commands.stopWorkflow, { name: row.title }),
          icon: <Square size={12} strokeWidth={2} aria-hidden="true" />,
          tooltip: fill(CHAT.agent.keyTip, {
            name: fill(CHAT.commands.stopWorkflow, { name: row.title }),
            key: CONTROL_KEYS.stop,
          }),
          keys: CONTROL_KEYS.stop,
        }}
        align="end"
        title={CHAT.run.stopTitle}
        description={stopText(row.agentsRunning)}
        confirmLabel={CHAT.run.buttons.stopWorkflow}
        cancelLabel={CHAT.run.buttons.keepRunning}
        workingLabel={CHAT.run.status.stopping}
        successLabel={CHAT.run.status.stopping}
        errorTitle={CHAT.run.stopError}
        onConfirm={() => props.onStop(row.id)}
      />
    </span>
  );
}

/** The agents under a run's row: the ones that ask or run (at most four), then `+n more` and its ended ones. */
function RunAgents(props: {
  row: RunRailRow;
  needs: readonly AskRow[];
  now: number;
  onJump: (anchor: EntryAnchor) => void;
  handlers: AgentHandlers | undefined;
  large: boolean;
}): ReactElement | null {
  const { row } = props;
  const roving = useAgentRoving<HTMLUListElement>();
  const ended = row.failed > 0 || row.done > 0;
  if (row.agents.length === 0 && row.moreAgents === 0 && !ended) return null;
  return (
    <ul
      ref={roving.ref}
      className={styles.nested}
      data-run-agents=""
      data-agent-list=""
      onFocus={roving.onFocus}
      onKeyDown={roving.onKeyDown}
    >
      {row.agents.map((agent) => {
        const asking = askOfAgent(props.needs, agent) !== null;
        const action = agentActions(agent, true, props.needs, props.handlers, props.large);
        return (
          <Row
            key={agent.id}
            icon={
              asking ? (
                <CircleHelp size={14} strokeWidth={1.75} aria-hidden="true" className={styles.asking} />
              ) : (
                <LiveDot tone="success" pulse={!agent.stopping} size={6} label="" />
              )
            }
            title={agent.title}
            {...(asking
              ? { detail: CHAT.agent.state.asking }
              : agent.stopping
                ? { detail: CHAT.activity.stopping }
                : {})}
            meta={<WaitClock since={agent.startTs} />}
            anchor={row.anchor}
            onJump={props.onJump}
            progress={{ value: agent.progress, now: props.now }}
            pinned
            roving
            {...agentOpen(agent, props.handlers)}
            {...(action !== null ? { action } : {})}
          />
        );
      })}
      {row.moreAgents > 0 ? (
        <li className={styles.item}>
          <button
            type="button"
            className={styles.foot}
            data-jump=""
            aria-label={fill(CHAT.activity.moreAgentsName, { n: row.moreAgents, title: row.title })}
            onClick={() => props.onJump(row.anchor)}
          >
            {fill(CHAT.activity.moreAgents, { n: row.moreAgents })}
          </button>
        </li>
      ) : null}
      {ended ? (
        <li className={styles.item}>
          <button
            type="button"
            className={styles.foot}
            data-jump=""
            aria-label={fill(CHAT.activity.jump, { title: row.title })}
            onClick={() => props.onJump(row.anchor)}
          >
            {row.failed > 0 ? (
              <span className={styles.footPart} data-tone="danger">
                <X size={12} strokeWidth={2} aria-hidden="true" />
                {fill(CHAT.run.counts.failed, { n: row.failed })}
              </span>
            ) : null}
            {row.done > 0 ? (
              <span className={styles.footPart} data-tone="success">
                <Check size={12} strokeWidth={2} aria-hidden="true" />
                {fill(CHAT.run.counts.done, { n: row.done })}
              </span>
            ) : null}
          </button>
        </li>
      ) : null}
    </ul>
  );
}

/** A run's row with its agents under it, in Running or in Needs you. */
function RunRow(props: {
  row: RunRailRow;
  needs: readonly AskRow[];
  now: number;
  onJump: (anchor: EntryAnchor) => void;
  handlers: AgentHandlers | undefined;
  onStopRun: ((runId: string) => Promise<void>) | undefined;
  large: boolean;
}): ReactElement {
  const { row } = props;
  return (
    <Row
      icon={ICONS.workflow}
      title={row.title}
      detail={detailOf(row)}
      meta={<WaitClock since={row.startTs} />}
      anchor={row.anchor}
      onJump={props.onJump}
      pinned
      {...(row.stoppable && props.onStopRun !== undefined
        ? { action: <StopRun row={row} onStop={props.onStopRun} large={props.large} /> }
        : {})}
    >
      <RunAgents
        row={row}
        needs={props.needs}
        now={props.now}
        onJump={props.onJump}
        handlers={props.handlers}
        large={props.large}
      />
    </Row>
  );
}

/** A schedule's Delete: a confirm at the button (with the overdue line for a paused one whose time passed). */
function DeleteSchedule(props: {
  row: ScheduleRow;
  off: string | null;
  onDelete: (row: ScheduleRow) => void;
}): ReactElement {
  const { row, off } = props;
  const reasonId = useId();
  const label = fill(CHAT.activity.deleteSchedule, { title: row.title });
  const icon = <Trash2 size={12} strokeWidth={2} aria-hidden="true" />;
  if (off !== null) {
    return (
      <>
        <IconButton
          label={label}
          icon={icon}
          tooltip={off}
          aria-disabled={true}
          aria-describedby={reasonId}
          onClick={(event) => event.preventDefault()}
        />
        <span id={reasonId} hidden>
          {off}
        </span>
      </>
    );
  }
  const overdue = row.state === 'paused' && row.nextTs !== null && row.nextTs <= Date.now();
  return (
    <ConfirmPopover
      trigger={{ kind: 'icon', label, icon }}
      align="end"
      title={CHAT.activity.deleteTitle}
      description={CHAT.activity.deleteText}
      {...(overdue ? { notice: CHAT.activity.deleteOverdue } : {})}
      confirmLabel={CHAT.activity.deleteConfirm}
      cancelLabel={CHAT.activity.deleteKeep}
      workingLabel={CHAT.activity.removing}
      successLabel={CHAT.activity.removing}
      onConfirm={async () => props.onDelete(row)}
    />
  );
}

/** What a schedule row says at its end: its state, else its next time. */
function scheduleMeta(row: ScheduleRow): ReactNode {
  switch (row.state) {
    case 'opening':
      return CHAT.activity.opening;
    case 'removing':
      return CHAT.activity.removing;
    case 'paused':
      return <span title={CHAT.activity.pausedTip}>{CHAT.activity.paused}</span>;
    case 'firing':
      return CHAT.activity.runningNow;
    case 'waiting':
      if (row.nextTs === null) return null;
      return row.nextTs <= Date.now() ? CHAT.activity.due : fill(CHAT.activity.next, { time: formatClock(row.nextTs) });
  }
}

function scheduleDetail(row: ScheduleRow): string {
  return row.repeat !== null && row.repeat !== ''
    ? `${CHAT.activity.kinds.schedule} · ${row.repeat}`
    : CHAT.activity.kinds.schedule;
}

/** The goal's section: its row jumps to where it began; the objective, what it does now, its clock while it runs. */
function GoalSection({ goal, onJump }: { goal: GoalView; onJump: (anchor: EntryAnchor) => void }): ReactElement {
  const id = useId();
  const detailId = useId();
  const now = useNow(goal.running);
  const { clock, tokens } = goalMeta(goal, now);
  const objective = visibleText(goal.entry.objective);
  return (
    <section className={styles.section} aria-labelledby={id} data-goal="">
      <h2 className={styles.heading} data-heading="" tabIndex={-1}>
        <span id={id} className={styles.headingText}>
          {CHAT.activity.goal}
        </span>
      </h2>
      <ul className={styles.rows}>
        <li className={styles.item}>
          <button
            type="button"
            className={styles.row}
            data-jump=""
            aria-label={CHAT.activity.goalJump}
            aria-describedby={detailId}
            onClick={() => onJump(goal.anchor)}
          >
            <span className={styles.icon}>
              <Target size={14} strokeWidth={1.75} aria-hidden="true" />
            </span>
            <span className={styles.text}>
              <span className={styles.title} title={objective}>
                {objective}
              </span>
              <span id={detailId} className={styles.detail}>
                {[goalStateWords(goal), tokens].filter((part) => part !== null).join(' · ')}
              </span>
            </span>
            {clock !== null ? (
              <span className={styles.meta} aria-hidden="true">
                {clock}
              </span>
            ) : null}
          </button>
        </li>
      </ul>
    </section>
  );
}

/**
 * What the conversation has going on now: the requests that wait for the person, its goal, its plan, the work that
 * still runs and its schedules.
 */
export function ActivityRail({
  activity,
  onJump,
  onAsk,
  onStopTask,
  agents,
  onStopRun,
  onDeleteSchedule,
  deleteOff = null,
  docked = true,
}: ActivityRailProps): ReactElement {
  const needsId = useId();
  const planId = useId();
  const runningId = useId();
  const scheduledId = useId();
  const { needs, needsRuns, plan, running, scheduled } = activity;
  const goal = activity.goal ?? null;
  // One clock for every agent's quiet time.
  const now = useNow(
    running.some((row) => row.kind === 'subagent' || (row.kind === 'workflow' && row.agents.length > 0)),
  );
  const done = plan === null ? 0 : plan.items.filter((item) => item.status === 'completed').length;
  const waiting = needs.length + needsRuns.length;
  const empty = waiting === 0 && goal === null && plan === null && running.length === 0 && scheduled.length === 0;
  const large = !docked;
  // Its subagent rows are one tab stop; a run's agents are their own list under its row.
  const runningRoving = useAgentRoving<HTMLUListElement>();
  return (
    <div className={styles.root} data-activity-rail="" data-actions-shown={docked ? undefined : ''} tabIndex={-1}>
      {waiting > 0 ? (
        <section className={styles.section} aria-labelledby={needsId} data-needs="">
          <h2 className={styles.heading} data-heading="" tabIndex={-1}>
            <span id={needsId} className={styles.headingText}>
              {CHAT.needs.title}
            </span>
            <span className={styles.count}>
              <LiveDot
                tone="warning"
                pulse
                size={6}
                label={waiting === 1 ? CHAT.needs.countOne : fill(CHAT.needs.count, { n: waiting })}
              />
              {waiting}
            </span>
          </h2>
          <ul className={styles.rows}>
            {needs.map((row) => (
              <AskRowView key={row.id} row={row} onAsk={onAsk} />
            ))}
            {needsRuns.map((row) => (
              <RunRow
                key={`workflow:${row.id}`}
                row={row}
                needs={needs}
                now={now}
                onJump={onJump}
                handlers={agents}
                onStopRun={onStopRun}
                large={large}
              />
            ))}
          </ul>
        </section>
      ) : null}
      {goal !== null ? <GoalSection goal={goal} onJump={onJump} /> : null}
      {plan !== null ? (
        <section className={styles.section} aria-labelledby={planId}>
          <button
            type="button"
            className={styles.heading}
            aria-label={fill(CHAT.activity.jump, { title: CHAT.activity.plan })}
            onClick={() => onJump(plan.anchor)}
          >
            <span id={planId} className={styles.headingText}>
              {CHAT.activity.plan}
            </span>
            <span className={styles.count}>{fill(CHAT.activity.planCount, { done, total: plan.items.length })}</span>
          </button>
          <PlanList items={plan.items} bare className={styles.plan} />
        </section>
      ) : null}
      {running.length > 0 ? (
        <section className={styles.section} aria-labelledby={runningId}>
          <h2 className={styles.heading} data-heading="" tabIndex={-1}>
            <span id={runningId} className={styles.headingText}>
              {CHAT.activity.nowRunning}
            </span>
            <span className={styles.count}>
              <LiveDot tone="success" pulse size={6} label={fill(CHAT.activity.running, { n: running.length })} />
              {running.length}
            </span>
          </h2>
          <ul
            ref={runningRoving.ref}
            className={styles.rows}
            data-agent-list=""
            onFocus={runningRoving.onFocus}
            onKeyDown={runningRoving.onKeyDown}
          >
            {running.map((row) => {
              if (row.kind === 'workflow')
                return (
                  <RunRow
                    key={`${row.kind}:${row.id}`}
                    row={row}
                    needs={needs}
                    now={now}
                    onJump={onJump}
                    handlers={agents}
                    onStopRun={onStopRun}
                    large={large}
                  />
                );
              const helper = row.kind === 'subagent' ? agentActions(row, false, needs, agents, large) : null;
              return (
                <Row
                  key={`${row.kind}:${row.id}`}
                  icon={ICONS[row.kind]}
                  title={row.title}
                  detail={detailOf(row)}
                  meta={<WaitClock since={row.startTs} />}
                  anchor={row.anchor}
                  onJump={onJump}
                  {...(row.kind === 'subagent'
                    ? { progress: { value: row.progress, now }, pinned: true, roving: true, ...agentOpen(row, agents) }
                    : {})}
                  {...((row.kind === 'command' || row.kind === 'monitor') && row.events > 0
                    ? { describeDetail: true, lastEvent: row.lastEvent }
                    : {})}
                  {...((row.kind === 'command' || row.kind === 'monitor') &&
                  row.live &&
                  !row.stopping &&
                  onStopTask !== undefined
                    ? { action: <StopTask row={row} onStop={onStopTask} /> }
                    : {})}
                  {...(helper !== null ? { action: helper } : {})}
                />
              );
            })}
          </ul>
        </section>
      ) : null}
      {scheduled.length > 0 ? (
        <section className={styles.section} aria-labelledby={scheduledId}>
          <h2 className={styles.heading} data-heading="" tabIndex={-1}>
            <span id={scheduledId} className={styles.headingText}>
              {CHAT.activity.scheduled}
            </span>
            <span className={styles.count}>{scheduled.length}</span>
          </h2>
          <ul className={styles.rows}>
            {scheduled.map((row) => (
              <Row
                key={row.id}
                icon={<Clock size={14} strokeWidth={1.75} aria-hidden="true" />}
                title={row.title}
                detail={scheduleDetail(row)}
                meta={scheduleMeta(row)}
                anchor={row.anchor}
                onJump={onJump}
                {...(onDeleteSchedule !== undefined && row.state !== 'opening' && row.state !== 'removing'
                  ? { action: <DeleteSchedule row={row} off={deleteOff} onDelete={onDeleteSchedule} /> }
                  : {})}
              />
            ))}
          </ul>
        </section>
      ) : null}
      {empty ? <p className={styles.empty}>{CHAT.activity.empty}</p> : null}
    </div>
  );
}

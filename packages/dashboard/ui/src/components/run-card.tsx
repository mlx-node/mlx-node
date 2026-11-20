import { GitBranch } from 'lucide-react';
import type { ReactElement } from 'react';

import { CHAT, fill } from '../lib/chat-strings';
import type { RunView } from '../lib/chat-types';
import { formatCount, formatCountNoun, formatDuration } from '../lib/format';
import type { TranscriptEntry } from '../lib/types';
import { ChatMarkdown } from './arc-x/chat-markdown/chat-markdown';
import { ConfirmPopover } from './arc-x/confirm-popover/confirm-popover';
import { Disclosure } from './arc-x/disclosure/disclosure';
import { EventRow } from './arc-x/event-row/event-row';
import { LiveDot, type LiveDotProps } from './arc-x/live-dot/live-dot';
import { Panel } from './arc-x/panel/panel';
import { Meter } from './arc-x/split-bar/split-bar';
import { Steps } from './arc-x/steps/steps';

export type RunEntry = Extract<TranscriptEntry, { kind: 'workflow' }>;

export interface RunCardProps {
  entry: RunEntry;
  /** The thread of the open conversation: the host's list of live runs is known. False on the Sessions page. */
  live: boolean;
  /** The host still holds this run (live only). */
  view?: RunView;
  /** A helper of this run asks for permission now: its label. */
  asking?: string | null;
  /** Stop run; offered only while live, `active`, held by the host and not stopping. */
  onStop?: () => Promise<void>;
}

export interface RunStatus {
  text: string;
  tone: LiveDotProps['tone'];
  pulse: boolean;
  /** The run still works: Stop run may show and the helpers list starts open. */
  working: boolean;
}

const PAUSED = new Set(['user_paused', 'back_off_paused', 'no_progress_paused', 'infra_paused']);

/** The status word of a run: the agent's status, then what the host says of it while the thread is live. */
export function runStatus(
  entry: Pick<RunEntry, 'status'>,
  live: boolean,
  view: RunView | undefined,
  asking: string | null,
): RunStatus {
  const words = CHAT.run.status;
  switch (entry.status) {
    case 'active':
      if (!live) return { text: words.active, tone: 'success', pulse: false, working: false };
      if (view === undefined) return { text: words.interrupted, tone: 'neutral', pulse: false, working: false };
      if (view.stopping) return { text: CHAT.run.stopping, tone: 'neutral', pulse: true, working: true };
      if (asking !== null) return { text: CHAT.run.waitingYou, tone: 'warning', pulse: true, working: true };
      return { text: words.active, tone: 'success', pulse: true, working: true };
    case 'blocked':
      return { text: words.blocked, tone: 'warning', pulse: false, working: false };
    case 'budget_limited':
      return { text: words.budget_limited, tone: 'warning', pulse: false, working: false };
    case 'complete':
      return { text: words.complete, tone: 'success', pulse: false, working: false };
    case 'failed':
      return { text: words.failed, tone: 'danger', pulse: false, working: false };
    case 'cancelled':
      return { text: words.cancelled, tone: 'neutral', pulse: false, working: false };
    case 'interrupted':
      return { text: words.interrupted, tone: 'neutral', pulse: false, working: false };
    default:
      if (PAUSED.has(entry.status)) return { text: words.paused, tone: 'warning', pulse: false, working: false };
      return { text: entry.status, tone: 'neutral', pulse: false, working: false };
  }
}

/** The index `Steps` marks current: the active phase, else the first not done, else past the end. */
export function currentStep(phases: RunEntry['phases']): number {
  const active = phases.findIndex((phase) => phase.state === 'active');
  if (active !== -1) return active;
  const open = phases.findIndex((phase) => phase.state !== 'done');
  return open === -1 ? phases.length : open;
}

/** "Step 2 of 4: Research" while the run has a current phase and has not ended; null otherwise. */
export function phaseLine(entry: Pick<RunEntry, 'phases' | 'currentPhase'>, ended: boolean): string | null {
  if (ended || entry.currentPhase === null || entry.phases.length === 0) return null;
  const at = entry.phases.findIndex((phase) => phase.title === entry.currentPhase);
  const i = at === -1 ? currentStep(entry.phases) : at;
  if (i >= entry.phases.length) return null;
  return fill(CHAT.run.phase, { i: i + 1, n: entry.phases.length, phase: entry.currentPhase });
}

function helperState(state: string): string {
  const words: Record<string, string> = CHAT.run.helperState;
  return words[state] ?? state;
}

const ENDED = new Set(['complete', 'failed', 'cancelled', 'interrupted', 'budget_limited']);

/** One workflow run: status, steps, helpers used of the budget, one row per helper, the report, and Stop run. */
export function RunCard(props: RunCardProps): ReactElement {
  const { entry, live } = props;
  const asking = props.asking ?? null;
  const status = runStatus(entry, live, props.view, asking);
  const endedEarly = live && entry.status === 'active' && props.view === undefined;
  const ended = ENDED.has(entry.status) || endedEarly;
  const step = phaseLine(entry, ended);
  const onStop = props.onStop;
  const canStop = live && entry.status === 'active' && props.view !== undefined && !props.view.stopping;
  const used =
    entry.agentBudget === null
      ? fill(CHAT.run.helpersUsedNoBudget, { used: formatCount(entry.agentsUsed) })
      : fill(CHAT.run.helpersUsed, { used: formatCount(entry.agentsUsed), budget: formatCount(entry.agentBudget) });
  return (
    <Panel
      title={fill(CHAT.run.title, { name: entry.name })}
      headingLevel={3}
      busy={status.working}
      actions={
        <>
          <LiveDot tone={status.tone} pulse={status.pulse} label={status.text} showLabel shimmer={status.pulse} />
          {canStop && onStop !== undefined ? (
            <ConfirmPopover
              trigger={{ kind: 'button', label: CHAT.run.stop, size: 'sm', variant: 'secondary' }}
              title={CHAT.run.stopTitle}
              description={CHAT.run.stopText}
              confirmLabel={CHAT.run.stop}
              cancelLabel={CHAT.run.stopKeep}
              workingLabel={CHAT.run.stopping}
              successLabel={CHAT.run.stopping}
              errorTitle={CHAT.run.stopError}
              onConfirm={onStop}
            />
          ) : null}
        </>
      }
    >
      <div className="grid min-w-0 gap-3" data-run={entry.runId} data-run-status={entry.status}>
        {entry.phases.length > 0 ? (
          <Steps
            label={CHAT.run.steps}
            steps={entry.phases.map((phase) => phase.title)}
            current={currentStep(entry.phases)}
          />
        ) : null}
        {step !== null ? <p className="m-0 text-body text-secondary">{step}</p> : null}
        {asking !== null && !ended ? (
          <p className="m-0 text-body text-secondary">{fill(CHAT.run.asks, { label: asking })}</p>
        ) : null}
        {entry.pauseMessage !== undefined && entry.pauseMessage !== '' ? (
          <p className="m-0 text-body text-secondary">
            {fill(entry.status === 'failed' ? CHAT.run.error : CHAT.run.paused, { message: entry.pauseMessage })}
          </p>
        ) : null}
        <div className="flex items-center gap-3">
          <span className="min-w-0 truncate text-caption text-secondary">{used}</span>
          {entry.agentBudget !== null ? (
            <Meter
              label={used}
              value={entry.agentsUsed}
              max={entry.agentBudget}
              formatValue={() => used}
              size="dense"
              className="w-48 min-w-0"
            />
          ) : null}
        </div>
        {entry.agents.length > 0 ? (
          <Disclosure title={fill(CHAT.run.helpers, { n: entry.agents.length })} defaultOpen={status.working}>
            <div className="grid min-w-0 gap-2">
              {entry.agents.map((agent) => (
                <EventRow
                  key={agent.agentId}
                  icon={<GitBranch size={14} strokeWidth={1.75} />}
                  label={agent.label}
                  text={helperState(agent.state)}
                  facts={[
                    agent.phase,
                    formatCountNoun(agent.tokens, 'token', 'tokens', formatCount),
                    agent.durationMs > 0 ? formatDuration(agent.durationMs) : null,
                  ]}
                  tone={agent.state === 'failed' ? 'danger' : 'neutral'}
                />
              ))}
            </div>
          </Disclosure>
        ) : null}
        {entry.report !== undefined && entry.report !== '' ? (
          <Disclosure title={CHAT.run.report} defaultOpen={entry.status === 'complete'}>
            <div className="grid min-w-0 gap-2">
              <ChatMarkdown text={entry.report} links={{ mode: 'work' }} />
              {entry.truncated === true ? <p className="m-0 text-caption text-muted">{CHAT.run.reportCut}</p> : null}
            </div>
          </Disclosure>
        ) : null}
      </div>
    </Panel>
  );
}

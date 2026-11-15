/**
 * Gallery sections for the transcript parts, the setup parts and the download card, on fixture data. Live states move
 * on timers from their first frame: a running tool ends, thinking and an answer grow then settle, a download moves,
 * stalls and finishes.
 */

import { GitBranch } from 'lucide-react';
import { type ReactElement, useEffect, useMemo, useState } from 'react';

import { AgentMark } from '../../../ui/src/components/arc-x/agent-mark/agent-mark';
import { ChoiceCard } from '../../../ui/src/components/arc-x/choice-card/choice-card';
import { EventRow, RuleLabel } from '../../../ui/src/components/arc-x/event-row/event-row';
import { Markdown } from '../../../ui/src/components/arc-x/markdown/markdown';
import { Message } from '../../../ui/src/components/arc-x/message/message';
import { ModelBadge } from '../../../ui/src/components/arc-x/model-badge/model-badge';
import { PlanList } from '../../../ui/src/components/arc-x/plan-list/plan-list';
import { ReasoningBlock } from '../../../ui/src/components/arc-x/reasoning-block/reasoning-block';
import { Steps } from '../../../ui/src/components/arc-x/steps/steps';
import { ToolCall, ToolCallGroup } from '../../../ui/src/components/arc-x/tool-call/tool-call';
import { Badge } from '../../../ui/src/components/arc/badge/badge';
import { ModeLine } from '../../../ui/src/components/chat-permission-marks';
import {
  DownloadProgressView,
  INITIAL_PROGRESS,
  reduceProgress,
  type ProgressEvent,
  type ProgressState,
} from '../../../ui/src/components/download-progress';
import { Transcript } from '../../../ui/src/components/transcript';
import type { LineMark } from '../../../ui/src/lib/permission-text';
import type { TranscriptEntry, TurnMetric } from '../../../ui/src/lib/types';

/** 0 for the first `everyMs`, then 1, 2 … up to `max`. */
function useStep(everyMs: number, max: number): number {
  const [step, setStep] = useState(0);
  useEffect(() => {
    if (step >= max) return undefined;
    const timer = setTimeout(() => setStep((value) => value + 1), everyMs);
    return () => clearTimeout(timer);
  }, [step, everyMs, max]);
  return step;
}

const NOW = Date.now();
const MODEL = 'qwen3.6-27b';

const READ_OUTPUT = `pub struct Planner {
    queue: VecDeque<Request>,
    budget: TokenBudget,
}

impl Planner {
    pub fn plan(&mut self) -> Plan {
        let mut plan = Plan::default();
        while let Some(request) = self.queue.pop_front() {
            if !self.budget.admit(&request) {
                self.queue.push_front(request);
                break;
            }
            plan.push(request);
        }
        plan
    }
}
// 25 more lines`;

const THOUGHT = [
  'The planner keeps every request in one queue.',
  '\nSplitting it means the budget check moves into its own module,',
  ' and the tests that poke at the queue need a new seam.',
  '\nRun the suite first so a failure is not blamed on the split.',
];

const ANSWER = [
  'The planner is now split into **three** modules:',
  '\n\n- `queue.rs` holds the pending requests',
  '\n- `budget.rs` admits a request when it fits',
  '\n- `plan.rs` builds the batch\n\nAll 214 tests pass.',
];

export function ToolCallSection(): ReactElement {
  const step = useStep(2600, 2);
  const ended = step >= 1;
  return (
    <div className="gallery-stage">
      <ToolCallGroup count={3} failed={1} running={ended ? 0 : 1} live>
        <ToolCall
          name="read_file"
          title="planner.rs"
          status="done"
          durationMs={120}
          ts={NOW - 60_000}
          input={'{ "path": "crates/mlx-core/src/planner.rs" }'}
          output={READ_OUTPUT}
        />
        <ToolCall
          name="run_terminal_command"
          title="cargo test -p mlx-core"
          status={ended ? 'done' : 'running'}
          durationMs={ended ? 41_200 : null}
          ts={NOW - 50_000}
          input="cargo test -p mlx-core"
          live
        />
        <ToolCall
          name="search"
          title="could not reach index"
          status="failed"
          ts={NOW - 40_000}
          input={'{ "query": "TokenBudget" }'}
          output="error: index at ~/.cache/search is locked by another process"
          truncated
        />
      </ToolCallGroup>
      <ToolCall
        name="list_dir"
        title="crates/mlx-core/src"
        status={step >= 2 ? 'done' : step >= 1 ? 'running' : 'pending'}
        live
        durationMs={step >= 2 ? 8 : null}
      />
      <ToolCall name="write_file" title="planner.rs" status="running" input="…" />
      <ToolCall name="apply_patch" title="budget.rs" status="other" statusText="not finished" />
    </div>
  );
}

export function ReasoningSection(): ReactElement {
  const step = useStep(1400, THOUGHT.length + 1);
  const live = step < THOUGHT.length;
  const text = THOUGHT.slice(0, Math.min(step + 1, THOUGHT.length)).join('');
  const previous = THOUGHT.slice(0, Math.min(step, THOUGHT.length)).join('');
  const startTs = NOW;
  return (
    <div className="gallery-stage">
      <ReasoningBlock
        text={text}
        live={live}
        startTs={startTs}
        asOfMs={startTs + 4000 + step * 1400}
        freshFrom={previous.length}
        durationMs={live ? null : 9600}
      />
      <ReasoningBlock text={THOUGHT.join('')} durationMs={3200} />
      <ReasoningBlock text="A stale body: the label is solid and has no time." />
    </div>
  );
}

export function MessageSection(): ReactElement {
  const step = useStep(1400, ANSWER.length);
  const streaming = step < ANSWER.length - 1 || step === 0;
  const text = ANSWER.slice(0, step + 1).join('');
  const previous = ANSWER.slice(0, step).join('');
  return (
    <div className="gallery-stage">
      <Message role="user" label="You" ts={NOW - 120_000}>
        <Markdown text="Split the planner into three modules and move the tests." />
      </Message>
      <Message
        role="assistant"
        label={<ModelBadge name={MODEL} size="sm" showLogo />}
        ts={NOW - 100_000}
        streaming={streaming}
      >
        <Markdown text={text} freshFrom={streaming ? previous.length : undefined} />
      </Message>
      <Message role="user" label="You · interjection" ts={NOW - 20_000}>
        <Markdown text="Keep the old file as a re-export for now." />
      </Message>
    </div>
  );
}

export function PlanEventSection(): ReactElement {
  return (
    <div className="gallery-stage">
      <PlanList
        ts={NOW - 300_000}
        items={[
          { content: 'Read the planner', status: 'completed' },
          { content: 'Split the queue and the budget into their own modules', status: 'in_progress' },
          { content: 'Move the tests', status: 'pending' },
        ]}
      />
      <EventRow
        icon={<GitBranch />}
        label="Subagent · explore"
        text="Explore the cache layout and report what holds the blocks"
        to="/sessions/child-1"
        facts={['completed', '42.1 s', '18.2K tokens']}
        ts={NOW - 200_000}
      />
      <EventRow
        icon={<GitBranch />}
        label="Subagent · build"
        text="child-2"
        facts={['failed', null, '']}
        tone="danger"
        error={'cargo build failed: linker `cc` not found\n  = note: install the Xcode command line tools'}
        ts={NOW - 100_000}
      />
      <RuleLabel>Model switched to gemma-4-e4b</RuleLabel>
      <RuleLabel>
        {'Turn ended · end_turn · 1m 12s · '}
        <a className="link" href="#inspect">
          Inspect
        </a>
      </RuleLabel>
      <RuleLabel tone="danger" detail="The model stopped: context window exceeded (131,072 tokens).">
        Turn ended · error · 4.00 s
      </RuleLabel>
      {THREAD_MARKS.map((mark) => (
        <ModeLine key={`${mark.kind}-${mark.at}`} mark={mark} temp="/tmp" />
      ))}
    </div>
  );
}

/** Every case of a thread mark: each mode, the second lines, the two app-made switches, the settings change. */
const THREAD_MARKS: LineMark[] = [
  { kind: 'mode', at: NOW - 90_000, mode: 'ask', from: 'auto', by: 'user' },
  { kind: 'mode', at: NOW - 89_000, mode: 'auto', from: 'ask', by: 'user' },
  { kind: 'mode', at: NOW - 88_500, mode: 'auto', from: 'ask', by: 'user', temp: true },
  { kind: 'mode', at: NOW - 88_000, mode: 'auto', from: 'ask', by: 'user', editsUnasked: true, userHooks: true },
  { kind: 'mode', at: NOW - 87_000, mode: 'auto', from: 'bypass', by: 'user' },
  { kind: 'mode', at: NOW - 86_000, mode: 'bypass', from: 'auto', by: 'user' },
  { kind: 'mode', at: NOW - 85_000, mode: 'bypass', from: 'ask', by: 'user', userHooks: true },
  { kind: 'mode', at: NOW - 84_000, mode: 'ask', from: 'bypass', by: 'restart' },
  { kind: 'mode', at: NOW - 83_000, mode: 'ask', from: 'bypass', by: 'record-failed' },
  { kind: 'settings', at: NOW - 82_000, changed: ['~/.claude.json'] },
  { kind: 'settings', at: NOW - 81_000, changed: ['~/.mlx-agent/config.toml', '~/.claude/settings.json'] },
  {
    kind: 'settings',
    at: NOW - 80_000,
    changed: ['~/.mlx-agent/config.toml', '~/code/app/.claude/settings.json', '~/code/app/.mcp.json', '~/.claude.json'],
  },
  {
    kind: 'settings',
    at: NOW - 79_000,
    changed: [
      '~/.claude/settings.json',
      '/private/var/folders/n9/kr5mprh51zn5m0vwyt2gr6_w0000gn/T/work-3f839611-f8c7-48e3-8900/projects/app/.mcp.json',
    ],
  },
  {
    kind: 'settings',
    at: NOW - 78_000,
    changed: ['~/workspace/github/mlx-node/.worktrees/app-redesign/.claude/settings.local.json'],
  },
];

const CHOICES = [
  { value: 'mlx-node/Qwen3.6-35B-A3B-mxfp4', title: 'Qwen3.6 35B A3B', size: 19.6, badge: 'Recommended' },
  { value: 'mlx-node/gemma-4-26b-a4b-mxfp4', title: 'Gemma 4 26B A4B', size: 14.2, badge: null },
  { value: 'mlx-node/AgentWorld-8B', title: 'AgentWorld 8B', size: 5.1, badge: null },
];

export function ChoiceSection(): ReactElement {
  const [value, setValue] = useState(CHOICES[0]!.value);
  return (
    <div className="gallery-grid">
      {CHOICES.map((choice, index) => (
        <ChoiceCard
          key={choice.value}
          name="gallery-model"
          value={choice.value}
          checked={value === choice.value}
          disabled={index === 2}
          onChange={setValue}
          ariaLabel={choice.title}
          describedBy={`gallery-${index}`}
          descriptionId={`gallery-${index}`}
          media={<ModelBadge name={choice.title} showLogo />}
          badge={
            choice.badge !== null ? (
              <Badge tone="info" size="sm">
                {choice.badge}
              </Badge>
            ) : undefined
          }
          title={choice.title}
          description="Our recommended starting point for working with tools, with fast answers on a laptop."
          meta={`~${choice.size} GB · ${index === 2 ? 'Folder needs cleanup' : 'One-time download'}`}
          warning={index === 0 ? 'Larger than the free space on this disk (12.0 GB).' : undefined}
        />
      ))}
      <ChoiceCard
        name="gallery-model"
        value=""
        checked={false}
        onChange={() => undefined}
        ariaLabel=""
        title=""
        description=""
        loading
      />
    </div>
  );
}

const STEP_NAMES = ['Choose a model', 'Download', 'Ready'];

export function StepsSection(): ReactElement {
  const step = useStep(1600, 2);
  return (
    <div className="gallery-stage">
      <Steps label="Setup" steps={STEP_NAMES} current={step} />
      <div style={{ maxWidth: 400 }}>
        <Steps label="Setup, narrow" steps={STEP_NAMES} current={step} />
      </div>
    </div>
  );
}

export function AgentMarkSection(): ReactElement {
  return (
    <div className="gallery-row">
      {(['claude', 'codex', 'grok'] as const).map((id) => (
        <span key={id} className="gallery-row">
          <AgentMark id={id} />
          <AgentMark id={id} size={20} />
        </span>
      ))}
    </div>
  );
}

const MIB = 1024 * 1024;
const TOTAL = 22_300 * MIB;
const FILES = ['config.json', 'model-00001-of-00005.safetensors', 'model-00002-of-00005.safetensors'];

function progress(index: number, bytes: number): ProgressEvent {
  return {
    type: 'progress',
    id: 'gallery',
    file: FILES[Math.min(index, FILES.length - 1)]!,
    receivedBytes: bytes,
    jobReceivedBytes: bytes,
    totalBytes: bytes,
    fileIndex: Math.min(index + 1, 4),
    fileCount: 5,
  };
}

const START: ProgressEvent = { type: 'start', id: 'gallery', repo: 'org/model', totalBytes: TOTAL, fileCount: 5 };

/** A state after `events`, each one second after the last on the rate clock. */
function stateOf(events: readonly ProgressEvent[]): ProgressState {
  return events.reduce<ProgressState>(
    (state, event, index) => reduceProgress(state, event, index * 1000),
    INITIAL_PROGRESS,
  );
}

export function DownloadSection(): ReactElement {
  const tick = useStep(900, 9);
  const moving = useMemo(() => {
    const events: ProgressEvent[] = [START];
    for (let i = 0; i <= tick; i += 1)
      events.push(progress(Math.floor(i / 3), 6_000 * MIB + i * 38 * 1024 * MIB * 0.05));
    return stateOf(events);
  }, [tick]);
  const stalled = useMemo(
    () => reduceProgress(stateOf([START, progress(1, 8_100 * MIB), progress(1, 8_140 * MIB)]), { type: 'stalled' }),
    [],
  );
  const done = useMemo(
    () => stateOf([START, progress(2, 20_000 * MIB), { type: 'done', id: 'gallery', outputDir: '/m' }]),
    [],
  );
  const failed = useMemo(
    () =>
      stateOf([
        START,
        progress(1, 9_000 * MIB),
        { type: 'error', id: 'gallery', message: 'No space left on device (28 GB needed)' },
      ]),
    [],
  );
  const cancelled = useMemo(
    () => stateOf([START, progress(1, 12_400 * MIB), { type: 'cancelled', id: 'gallery' }]),
    [],
  );
  const finishing = tick >= 9;
  return (
    <div className="gallery-grid">
      <div className="gallery-cell">
        <span>connecting</span>
        <DownloadProgressView state={INITIAL_PROGRESS} />
      </div>
      <div className="gallery-cell">
        <span>running (moves, then completes)</span>
        <DownloadProgressView
          state={finishing ? reduceProgress(moving, { type: 'done', id: 'gallery', outputDir: '/m' }) : moving}
        />
      </div>
      <div className="gallery-cell">
        <span>stalled (no event for 5 s)</span>
        <DownloadProgressView state={stalled} />
      </div>
      <div className="gallery-cell">
        <span>done</span>
        <DownloadProgressView state={done} />
      </div>
      <div className="gallery-cell">
        <span>error</span>
        <DownloadProgressView state={failed} />
      </div>
      <div className="gallery-cell">
        <span>cancelled</span>
        <DownloadProgressView state={cancelled} />
      </div>
    </div>
  );
}

function metric(ts: number, elapsedMs: number): TurnMetric {
  return {
    promptId: `p${ts}`,
    ts,
    inputTokens: 12_000,
    outputTokens: 900,
    cachedTokens: 10_000,
    reasoningTokens: 200,
    elapsedMs,
    ttftMs: 900,
    prefillMs: 3_100,
    prefillTokens: 2_000,
    prefillCalls: 1,
    modelCalls: 3,
    decodeTps: 48,
    model: MODEL,
  };
}

function closedTurn(n: number, prompt: string, tools: number, failed: boolean): TranscriptEntry[] {
  const ts = NOW - (10 - n) * 600_000;
  const entries: TranscriptEntry[] = [
    { kind: 'user', ts, text: prompt, promptIndex: n },
    { kind: 'thinking', ts: ts + 900, text: 'Look at the tests before changing anything.', endTs: ts + 4100 },
  ];
  for (let i = 0; i < tools; i += 1) {
    entries.push({
      kind: 'tool',
      ts: ts + 5000 + i * 2000,
      id: `t${n}-${i}`,
      name: i === 0 ? 'read_file' : 'run_terminal_command',
      title: i === 0 ? 'planner.rs' : 'cargo test -p mlx-core',
      status: failed && i === tools - 1 ? 'failed' : 'completed',
      output: failed && i === tools - 1 ? 'test planner::split ... FAILED' : 'ok',
      endTs: ts + 6000 + i * 2000,
    });
  }
  entries.push(
    {
      kind: 'assistant',
      ts: ts + 20_000,
      text: `Done with step ${n}. **Next:** the budget module.`,
      endTs: ts + 26_000,
    },
    { kind: 'turn-end', ts: ts + 30_000, stopReason: 'end_turn', elapsedMs: 30_000 },
  );
  return entries;
}

export function TranscriptSection(): ReactElement {
  const step = useStep(1500, 4);
  const live = step < 4;
  const base = useMemo(
    () => [
      ...closedTurn(1, 'Read the planner and tell me what it does', 1, false),
      ...closedTurn(2, 'Run the suite', 2, true),
      ...closedTurn(3, 'Split the planner into three modules', 3, false),
    ],
    [],
  );
  const turns = useMemo(
    () => [metric(NOW - 5_400_000, 30_000), metric(NOW - 4_800_000, 30_000), metric(NOW - 4_200_000, 30_000)],
    [],
  );
  const openTs = NOW - 20_000;
  const entries = useMemo<TranscriptEntry[]>(
    () => [
      ...base,
      { kind: 'user', ts: openTs, text: 'Summarise what changed', promptIndex: 4 },
      {
        kind: 'thinking',
        ts: openTs + 500,
        text: THOUGHT.slice(0, Math.min(step + 1, THOUGHT.length)).join(''),
        ...(step >= 2 ? { endTs: openTs + 5000 } : {}),
      },
      {
        kind: 'tool',
        ts: openTs + 5200,
        id: 'live-a',
        name: 'read_file',
        title: 'planner.rs',
        status: 'completed',
        endTs: openTs + 5400,
        output: READ_OUTPUT,
      },
      {
        kind: 'tool',
        ts: openTs + 5600,
        id: 'live-b',
        name: 'run_terminal_command',
        title: 'cargo test',
        status: step >= 3 ? 'completed' : 'in_progress',
        ...(step >= 3 ? { endTs: openTs + 9000 } : {}),
      },
      ...(step >= 2 ? [{ kind: 'assistant' as const, ts: openTs + 9500, text: ANSWER.slice(0, step).join('') }] : []),
    ],
    [base, step, openTs],
  );
  return (
    <Transcript
      entries={entries}
      turns={turns}
      model={MODEL}
      sessionLive={live}
      live={live}
      asOfMs={openTs + 7000 + step * 1500}
      onInspectTurn={() => undefined}
    />
  );
}

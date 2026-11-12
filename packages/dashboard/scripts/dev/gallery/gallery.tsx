/**
 * The dev harness's `/__gallery` page: every own component in its loading -> ready and idle -> confirming -> working ->
 * success states, with a Replay button per section. Dev only: nothing under `ui/` imports it.
 */

import { Trash2 } from 'lucide-react';
import { MotionConfig } from 'motion/react';
import { type ReactElement, type ReactNode, StrictMode, useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { MemoryRouter } from 'react-router-dom';

import '../../../ui/src/index.css';
import { CodePanel } from '../../../ui/src/components/arc-x/code-panel/code-panel';
import { ConfirmPopover } from '../../../ui/src/components/arc-x/confirm-popover/confirm-popover';
import { DataTable, type DataColumn } from '../../../ui/src/components/arc-x/data-table/data-table';
import { Disclosure } from '../../../ui/src/components/arc-x/disclosure/disclosure';
import { KeyValue } from '../../../ui/src/components/arc-x/key-value/key-value';
import { LiveDot } from '../../../ui/src/components/arc-x/live-dot/live-dot';
import { LiveLabel } from '../../../ui/src/components/arc-x/live-label/live-label';
import { Markdown } from '../../../ui/src/components/arc-x/markdown/markdown';
import { ModelBadge } from '../../../ui/src/components/arc-x/model-badge/model-badge';
import { MorphLoader } from '../../../ui/src/components/arc-x/morph-loader/morph-loader';
import type { LoaderStatus } from '../../../ui/src/components/arc-x/morph-loader/shapes';
import { NumberValue } from '../../../ui/src/components/arc-x/number-value/number-value';
import { Placeholder } from '../../../ui/src/components/arc-x/placeholder/placeholder';
import { SwapText } from '../../../ui/src/components/arc-x/swap-text/swap-text';
import { Button } from '../../../ui/src/components/arc/button/button';
import { ArcProvider } from '../../../ui/src/components/arc/lib/arc-provider';
import { ToastStack, ToastStackProvider } from '../../../ui/src/components/arc/toast-stack/toast-stack';
import { ToastBridge } from '../../../ui/src/lib/toast';
import { ArcLink } from '../../../ui/src/shell/arc-link';
import { type ThemePreference, initTheme, useTheme } from '../../../ui/src/theme/theme';
import {
  AgentTimelineSection,
  BarListSection,
  CellGridSection,
  ChartCardSection,
  DrilldownSection,
  HistogramSection,
  LiveStreamSection,
  LiveTurnSection,
  QuantileSection,
  ScatterSection,
  SplitSection,
  StackedSection,
  StatSection,
  SweepSection,
  TickerSection,
  TimelineSection,
  TrendSection,
} from './data-sections';
import {
  AgentMarkSection,
  ChoiceSection,
  DownloadSection,
  MessageSection,
  PlanEventSection,
  ReasoningSection,
  StepsSection,
  ToolCallSection,
  TranscriptSection,
} from './transcript-sections';

/** `steps[i]` from `at[i]` ms after mount; the first entry shows at once. `at` must be a module constant. */
function useTimeline<T>(steps: readonly T[], at: readonly number[]): T {
  const [index, setIndex] = useState(0);
  useEffect(() => {
    const timers = at.slice(1).map((ms, i) => setTimeout(() => setIndex(i + 1), ms));
    return () => timers.forEach(clearTimeout);
  }, [at]);
  return steps[Math.min(index, steps.length - 1)]!;
}

function Section({ title, note, children }: { title: string; note: string; children: ReactNode }): ReactElement {
  const [replay, setReplay] = useState(0);
  return (
    <section className="gallery-section" aria-label={title} data-gallery={title}>
      <div className="gallery-head">
        <h2>{title}</h2>
        <Button variant="secondary" size="sm" onClick={() => setReplay((value) => value + 1)}>
          Replay
        </Button>
      </div>
      <p className="gallery-note">{note}</p>
      <div key={replay} className="gallery-body">
        {children}
      </div>
    </section>
  );
}

function Cell({ label, children }: { label: string; children: ReactNode }): ReactElement {
  return (
    <div className="gallery-cell">
      <span>{label}</span>
      {children}
    </div>
  );
}

const COUNT_AT = [0, 700, 2000, 3200, 4400];
const COUNT_STEPS = [null, '1,284', '1,302', '12.9K', '12.9K'];
const DURATION_STEPS = ['—', '1m 05s', '59.8 s', '61.2 s', '2m 05s'];
const SHARE_STEPS = [null, '73%', '74%', '>99%', '100%'];
const READY_STEPS = [true, false];
const PLACEHOLDER_AT = [0, 1400];
const TABLE_AT = [0, 1200];
const TEXT_AT = [0, 1000];
const LOADER_AT = [0, 600, 2200, 3400, 5000];
const LABEL_STEPS = ['Starting', 'Loading model', 'Running', 'Running'];
const LABEL_AT = [0, 1200, 2600, 4000];

function Numbers(): ReactElement {
  const count = useTimeline(COUNT_STEPS, COUNT_AT);
  const duration = useTimeline(DURATION_STEPS, COUNT_AT);
  const share = useTimeline(SHARE_STEPS, COUNT_AT);
  return (
    <div className="gallery-row">
      <Cell label="count up, then roll">
        <span className="gallery-value">
          <NumberValue text={count} reserve="6ch" />
        </span>
      </Cell>
      <Cell label="compound text -> number swaps">
        <span className="gallery-value">
          <NumberValue text={duration} reserve="7ch" />
        </span>
      </Cell>
      <Cell label="no count-up">
        <span className="gallery-value">
          <NumberValue text={share} countUp={false} reserve="5ch" />
        </span>
      </Cell>
    </div>
  );
}

function Placeholders(): ReactElement {
  const loading = useTimeline(READY_STEPS, PLACEHOLDER_AT);
  return (
    <div className="gallery-row">
      <Cell label="text bar -> content">
        <Placeholder loading={loading} width="18ch">
          <span>Qwen3.6-35B-A3B-4bit</span>
        </Placeholder>
      </Cell>
      <Cell label="block -> content">
        <Placeholder loading={loading} width={220} height={56} radius="control">
          <span className="gallery-value">42.7 tok/s</span>
        </Placeholder>
      </Cell>
      <Cell label="inline">
        <span>
          Loaded in{' '}
          <Placeholder loading={loading} inline width="5ch">
            <span>2.4 s</span>
          </Placeholder>
        </span>
      </Cell>
    </div>
  );
}

const LOADER_STEPS: readonly LoaderStatus[] = ['pending', 'loading', 'success', 'loading', 'error'];

function Loaders(): ReactElement {
  const status = useTimeline(LOADER_STEPS, LOADER_AT);
  return (
    <div className="gallery-row">
      {([14, 16, 20] as const).map((size) => (
        <Cell key={size} label={`${size} px: ${status}`}>
          <MorphLoader status={status} size={size} label={status} />
        </Cell>
      ))}
    </div>
  );
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function Confirms(): ReactElement {
  return (
    <div className="gallery-row">
      <ConfirmPopover
        trigger={{ kind: 'button', label: 'Delete model', icon: <Trash2 size={14} aria-hidden />, variant: 'danger' }}
        title="Delete Qwen3.6-35B-A3B-4bit?"
        description="The folder and its 19.4 GB of weights are removed from disk."
        confirmLabel="Delete"
        workingLabel="Deleting…"
        successLabel="Deleted"
        tone="danger"
        align="start"
        onConfirm={() => wait(1400)}
      />
      <ConfirmPopover
        trigger={{ kind: 'button', label: 'Clear cache', variant: 'secondary' }}
        title="Clear the cold cache?"
        description="Fails on purpose, to show the error face."
        confirmLabel="Clear"
        workingLabel="Clearing…"
        successLabel="Cleared"
        align="start"
        onConfirm={async () => {
          await wait(1000);
          throw new Error('The cache folder is busy');
        }}
      />
      <ConfirmPopover
        trigger={{ kind: 'icon', label: 'Remove', icon: <Trash2 size={14} aria-hidden /> }}
        title="Remove this row?"
        description="Icon trigger."
        confirmLabel="Remove"
        workingLabel="Removing…"
        successLabel="Removed"
        onConfirm={() => wait(800)}
      />
    </div>
  );
}

interface TableRow {
  id: string;
  model: string;
  turns: number;
  tps: number | null;
}

const TABLE_ROWS: readonly TableRow[] = Array.from({ length: 24 }, (_, i) => ({
  id: `row-${i}`,
  model: ['Qwen3.6-35B-A3B', 'Gemma-4-31B-IT', 'LFM2.5-8B', 'Nemotron-3.5'][i % 4]!,
  turns: 40 + ((i * 37) % 300),
  tps: i % 7 === 3 ? null : 20 + ((i * 13) % 60),
}));

const TABLE_COLUMNS: readonly DataColumn<TableRow>[] = [
  {
    id: 'model',
    header: 'Model',
    cell: (row) => <ModelBadge name={row.model} size="sm" />,
    sortValue: (row) => row.model,
  },
  { id: 'turns', header: 'Turns', cell: (row) => String(row.turns), sortValue: (row) => row.turns, align: 'end' },
  {
    id: 'tps',
    header: 'Decode tok/s',
    cell: (row) => (row.tps === null ? '—' : row.tps.toFixed(1)),
    sortValue: (row) => row.tps,
    align: 'end',
  },
];

function Table(): ReactElement {
  const loading = useTimeline(READY_STEPS, TABLE_AT);
  return (
    <DataTable
      caption="Turns by model"
      columns={TABLE_COLUMNS}
      rows={TABLE_ROWS}
      rowKey={(row) => row.id}
      loading={loading}
      defaultSort={{ id: 'turns', direction: 'desc' }}
    />
  );
}

const MARKDOWN = `## Plan

1. Read **the trace** for the slow turn.
2. Compare \`prefill_ms\` with the cold-tier restore.

> Restores beat prefill by 3.3-21.4x on this machine.

\`\`\`ts
const ratio = restoreMs / prefillMs;
\`\`\`
`;

function Texts(): ReactElement {
  const loading = useTimeline(READY_STEPS, TEXT_AT);
  const label = useTimeline(LABEL_STEPS, LABEL_AT);
  return (
    <div className="gallery-row" style={{ alignItems: 'flex-start' }}>
      <div style={{ display: 'grid', gap: 12, flex: '1 1 320px' }}>
        <KeyValue
          loading={loading}
          ariaLabel="Process"
          items={[
            { key: 'pid', label: 'PID', value: '48211', mono: true, copy: '48211' },
            { key: 'up', label: 'Uptime', value: '6h 12m' },
            { key: 'mem', label: 'Memory', value: '21.6 GB', numeric: true, hint: 'of 128 GB' },
            { key: 'none', label: 'Queue limit', value: null },
          ]}
        />
        <div className="gallery-row">
          <LiveDot tone="success" pulse label="Running" showLabel />
          <LiveDot tone="warning" label="Loading" showLabel shimmer />
          <LiveDot tone="danger" label="Failed" showLabel />
          <LiveLabel active={label !== 'Running'}>{label}</LiveLabel>
          <SwapText text={label} />
          <SwapText text={label} granularity="char" />
        </div>
        <div className="gallery-row">
          <ModelBadge name="Qwen3.6-35B-A3B" quant="4bit" />
          <ModelBadge name="gemma-4-31b-it" quant="Q4_K_M" size="sm" />
          <ModelBadge name="LFM2.5-8B-A1B" showLogo={false} />
        </div>
      </div>
      <div style={{ display: 'grid', gap: 12, flex: '1 1 320px' }}>
        <Disclosure title="Request" meta="1.2 KB" defaultOpen>
          <CodePanel label="JSON" code={JSON.stringify({ model: 'qwen3.6', max_tokens: 512, stream: true }, null, 2)} />
        </Disclosure>
        <Disclosure title="Response" meta="collapsed">
          <Markdown text={MARKDOWN} />
        </Disclosure>
      </div>
    </div>
  );
}

const THEMES: readonly ThemePreference[] = ['system', 'light', 'dark'];

function Gallery(): ReactElement {
  const theme = useTheme();
  return (
    <main className="gallery" data-scroll-root="">
      <div className="gallery-head">
        <h1>Own components</h1>
        <div className="gallery-row">
          {THEMES.map((value) => (
            <Button
              key={value}
              size="sm"
              variant={theme.preference === value ? 'primary' : 'ghost'}
              onClick={() => theme.setPreference(value)}
            >
              {value}
            </Button>
          ))}
        </div>
      </div>
      <Section title="NumberValue" note="null -> count up -> roll; compound text -> number swaps; no count-up.">
        <Numbers />
      </Section>
      <Section title="Placeholder" note="Skeleton -> content morph after 1.4 s.">
        <Placeholders />
      </Section>
      <Section title="MorphLoader" note="pending -> loading -> success -> loading -> error.">
        <Loaders />
      </Section>
      <Section title="ConfirmPopover" note="Click: idle -> confirming -> working -> success pill (second one fails).">
        <Confirms />
      </Section>
      <Section title="DataTable" note="Loading rows in the loaded shape, then the first 12 rows reveal.">
        <Table />
      </Section>
      <Section
        title="Text and detail"
        note="KeyValue, LiveDot, LiveLabel, SwapText, ModelBadge, Disclosure, CodePanel, Markdown."
      >
        <Texts />
      </Section>
      <Section
        title="TrendChart"
        note="Line trace on entry; data reshapes after 2.6 s; gap stays a gap; n=3 hollow marker; band + ghost."
      >
        <TrendSection />
      </Section>
      <Section
        title="StackedBarChart"
        note="Bars grow from the baseline; reshape after 2.6 s; click a column to select it."
      >
        <StackedSection />
      </Section>
      <Section title="Histogram" note="Bars grow, markers reveal; bins and markers move after 2.6 s.">
        <HistogramSection />
      </Section>
      <Section title="BarList" note="Rows reveal and fill; values re-sort on travel after 2.6 s; n=3 hatched.">
        <BarListSection />
      </Section>
      <Section title="ChartCard" note="loading -> ready -> held (dims) -> ready; error with Try again; empty.">
        <ChartCardSection />
      </Section>
      <Section
        title="StatTile"
        note="Count up once, roll on change, underline glides between selected tiles, loading and error tiles."
      >
        <StatSection />
      </Section>
      <Section
        title="SplitBar and Meter"
        note="Bar draws, legend dims the others on hover; meters fill and change tone."
      >
        <SplitSection />
      </Section>
      <Section title="SpanTimeline" note="Spans draw row by row; arrows move between spans.">
        <TimelineSection />
      </Section>
      <Section
        title="AgentTimeline · finished"
        note="Opens settled. Play replays recorded times. Click a span for its path."
      >
        <AgentTimelineSection variant="finished" />
      </Section>
      <Section title="AgentTimeline · long turn" note="12 minutes: 60× and 120×.">
        <AgentTimelineSection variant="long" />
      </Section>
      <Section title="AgentTimeline · live" note="Follows the clock for 10 s, then stale.">
        <AgentTimelineSection variant="live" />
      </Section>
      <Section title="AgentTimeline · one lane" note="A turn with no subagent.">
        <AgentTimelineSection variant="one-lane" />
      </Section>
      <Section title="AgentTimeline · honest cases" note="No step record, open ends, folded lane, not finished.">
        <AgentTimelineSection variant="honest" />
      </Section>
      <Section
        title="AgentTimeline · woken turn"
        note="A background workflow woke the agent: idle line, turn start rule, the status counts the turn only."
      >
        <AgentTimelineSection variant="woken" />
      </Section>
      <Section title="AgentTimeline · narrow" note="Container 360 px.">
        <AgentTimelineSection variant="narrow" />
      </Section>
      <Section title="AgentTimeline · states" note="Loading, error, no times.">
        <AgentTimelineSection variant="states" />
      </Section>
      <Section title="ScatterChart" note="Points wave in; same keys move; new question (step 3) replays the entrance.">
        <ScatterSection />
      </Section>
      <Section title="QuantileBar" note="Band opens from the median; no band below 20 samples; ring below 5.">
        <QuantileSection />
      </Section>
      <Section
        title="LiveStream"
        note="One sample per second; the line slides left; a 30 s hole is a gap; pill pauses."
      >
        <LiveStreamSection />
      </Section>
      <Section title="CellGrid" note="Cells take their kind in a wave; changed cells move on each step.">
        <CellGridSection />
      </Section>
      <Section
        title="LiveTurnRow"
        note="Prefill meter with shine, then decode count rolls; the queued row leaves and returns."
      >
        <LiveTurnSection />
      </Section>
      <Section
        title="KpiDrilldown"
        note="Click a tile with the expand mark: it travels into the panel; Escape or the close button travels back."
      >
        <DrilldownSection />
      </Section>
      <Section
        title="MetricTicker"
        note="Loading -> values roll every 2 s with chip flash; narrow strip drifts; stale strip stands still."
      >
        <TickerSection />
      </Section>
      <Section
        title="DraftSweep"
        note="Cells fill left to right once; rates move after 2.6 s; positions past 8 fold into +n."
      >
        <SweepSection />
      </Section>
      <Section
        title="ToolCall"
        note="Group: the running call shimmers and turns, then ends in a check at 2.6 s; failed call open; bare rows: pending -> running -> done, a recorded running row, not finished."
      >
        <ToolCallSection />
      </Section>
      <Section
        title="ReasoningBlock"
        note="Live: shimmering Thinking, time as of the body, feathered tail growing every 1.4 s; then it settles. Finished and stale blocks below."
      >
        <ReasoningSection />
      </Section>
      <Section
        title="Message"
        note="A streaming answer: Writing shimmers and new blocks fade in per step, then it settles."
      >
        <MessageSection />
      </Section>
      <Section
        title="PlanList, EventRow, RuleLabel"
        note="Plan states, a linked subagent row, a failed one with its error, rule labels."
      >
        <PlanEventSection />
      </Section>
      <Section title="ChoiceCard" note="Radio cards: selected, plain, blocked; a loading card. Click to choose.">
        <ChoiceSection />
      </Section>
      <Section
        title="Steps"
        note="The current step advances every 1.6 s; the narrow copy shows only the current label."
      >
        <StepsSection />
      </Section>
      <Section title="AgentMark" note="claude, codex, grok at 28 and 20 px, in the text color.">
        <AgentMarkSection />
      </Section>
      <Section
        title="DownloadProgress"
        note="connecting; running (bytes move, shine on the fill, rate and time left, then complete); stalled; done; error; cancelled."
      >
        <DownloadSection />
      </Section>
      <Section
        title="Transcript"
        note="Three closed turns and a live trailing turn: thinking grows then ends, the running tool ends, the answer arrives; live look ends at 6 s."
      >
        <TranscriptSection />
      </Section>
    </main>
  );
}

initTheme();
const rootElement = document.getElementById('root');
if (rootElement === null) throw new Error('#root element not found');
createRoot(rootElement).render(
  <StrictMode>
    <MotionConfig reducedMotion="user">
      <ToastStackProvider>
        <ToastBridge />
        <MemoryRouter>
          <ArcProvider link={ArcLink}>
            <Gallery />
          </ArcProvider>
        </MemoryRouter>
        <ToastStack position="bottom-right" label="Notifications" />
      </ToastStackProvider>
    </MotionConfig>
  </StrictMode>,
);

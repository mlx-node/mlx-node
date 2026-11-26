/**
 * The main agent's `workflow` calls as the thread draws them: a launch that came back with a run id is the run's
 * block, every other call is a row of its own words. The lines are a real Work session's (`fixtures/workflow-run`):
 * a call the agent could not parse, a check that failed, a check that passed, then the launch.
 */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

import { parseUpdates } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { CHAT, fill } from '../ui/src/lib/chat-strings.js';
import { launchDetails, runPlaces, workflowFormOf } from '../ui/src/lib/work-tools.js';

type ToolEntry = Extract<TranscriptEntry, { kind: 'tool' }>;

const RUN = 'wf_01a2168534c27fd1a477edb6fad7b0a4';
const LINES = readFileSync(fileURLToPath(new URL('./fixtures/workflow-run/parent.jsonl', import.meta.url)), 'utf8');
const ENTRIES = parseUpdates(LINES).transcript;
const CALLS = ENTRIES.filter((entry): entry is ToolEntry => entry.kind === 'tool' && entry.name === 'workflow');
const NAMES = new Map([[RUN, 'research-advisory']]);
const W = CHAT.run.toolRows;

const call = (over: Partial<ToolEntry>): ToolEntry => ({
  kind: 'tool',
  ts: 1,
  id: 'call-1',
  name: 'workflow',
  title: 'Workflow: launch script',
  status: 'completed',
  ...over,
});

describe('the workflow tool forms', () => {
  // Mutation: draw a block for any `workflow` call → a refused launch, a check and a pause each draw a run block.
  it("R's calls: an unreadable one stays a plain row, a failed check, a passed check, then the run's block", () => {
    expect(CALLS.map((entry) => workflowFormOf(entry, NAMES))).toEqual([
      null,
      {
        form: 'row',
        kind: 'check-failed',
        title: W.checkFailed,
        error:
          'Tool `workflow` failed: dry-run: failed: Function not found: write_scratch_file (&str | ImmutableString | String, ()) (line 247, position 12)',
      },
      { form: 'row', kind: 'checked', title: fill(W.checked, { name: 'research-advisory' }) },
      { form: 'block', runId: RUN },
    ]);
  });

  it('a refused or failed launch is a row, never a block; a launch still running is the plain row', () => {
    const refused = call({
      status: 'failed',
      output: 'Too many active workflows (4).\n',
      run: { id: null, kind: 'launch' },
    });
    expect(workflowFormOf(refused, NAMES)).toEqual({
      form: 'row',
      kind: 'not-started',
      title: W.notStarted,
      error: 'Too many active workflows (4).',
    });
    expect(workflowFormOf(call({ run: { id: null, kind: 'launch' } }), NAMES)).toMatchObject({ kind: 'not-started' });
    expect(workflowFormOf(call({ status: 'in_progress', run: { id: null, kind: 'launch' } }), NAMES)).toBeNull();
    expect(workflowFormOf(call({ name: 'read_file', run: { id: RUN, kind: 'launch' } }), NAMES)).toBeNull();
  });

  it("the main agent's pause, stop and resume name the run and link to it; a resume draws no second block", () => {
    const control = (kind: 'pause' | 'stop' | 'resume', id: string | null, input?: object) =>
      workflowFormOf(
        call({ run: { id, kind }, ...(input === undefined ? {} : { input: JSON.stringify(input) }) }),
        NAMES,
      );
    expect(control('pause', RUN)).toEqual({
      form: 'row',
      kind: 'control',
      verb: 'pause',
      title: fill(W.paused, { name: 'research-advisory' }),
      runId: RUN,
    });
    expect(control('stop', RUN)).toMatchObject({ title: fill(W.stopped, { name: 'research-advisory' }) });
    expect(control('resume', RUN)).toMatchObject({
      form: 'row',
      title: fill(W.resumed, { name: 'research-advisory' }),
    });
    // A name the transcript knows no run of is shown as the agent wrote it, with no link.
    expect(control('stop', null, { source: { type: 'stop', run_id: 'deep-research' } })).toMatchObject({
      title: fill(W.stopped, { name: 'deep-research' }),
      runId: null,
    });
    expect(control('resume', null, { source: { type: 'resume', resume_from_run_id: 'wf_x' } })).toMatchObject({
      title: fill(W.resumed, { name: 'wf_x' }),
    });
    expect(workflowFormOf(call({ status: 'failed', run: { id: RUN, kind: 'stop' } }), NAMES)).toBeNull();
  });

  it('the block goes at the launch; else at the run entry; else at the first agent that names the run', () => {
    const launch = ENTRIES.indexOf(CALLS[3]!);
    expect(runPlaces(ENTRIES).get(RUN)).toEqual({ at: launch, kind: 'launch' });
    const cut = ENTRIES.filter((entry) => entry !== CALLS[3]);
    expect(runPlaces(cut).get(RUN)).toEqual({ at: cut.findIndex((e) => e.kind === 'workflow'), kind: 'entry' });
    const bare = cut.filter((entry) => entry.kind !== 'workflow');
    expect(runPlaces(bare).get(RUN)).toEqual({ at: bare.findIndex((e) => e.kind === 'subagent'), kind: 'bare' });
    const refused = ENTRIES.map((entry) => (entry === CALLS[3] ? { ...CALLS[3]!, status: 'failed' } : entry));
    expect(runPlaces(refused).get(RUN)?.kind).toBe('entry');
  });
});

describe('the launch details', () => {
  // Mutation: read `source.path` for a script file → R's launch shows no script path.
  it("R's launch: a script file, its path, its args and the default limit; a saved one with a limit", () => {
    const launch = CALLS.find((entry) => entry.run?.kind === 'launch')!;
    expect(launchDetails(launch)).toEqual({
      source: 'script_path',
      value: '/Users/dev/napi-rs/.grok/workflows/research-advisory.rhai',
      args: '{\n  "ghsa": "GHSA-0000-0000-0000"\n}',
      budget: null,
    });
    const saved = call({
      input: JSON.stringify({
        source: { type: 'name', name: 'deep-research' },
        args: { topic: 'x' },
        agent_budget: 32,
      }),
    });
    expect(launchDetails(saved)).toEqual({
      source: 'name',
      value: 'deep-research',
      args: '{\n  "topic": "x"\n}',
      budget: 32,
    });
    const inline = call({ input: JSON.stringify({ source: { type: 'script', script: 'phase("a")' } }) });
    expect(launchDetails(inline)).toMatchObject({ source: 'script', value: null });
    expect(launchDetails(call({ input: JSON.stringify({ source: { type: 'url' } }) }))).toBeNull();
    expect(launchDetails(call({}))).toBeNull();
  });
});

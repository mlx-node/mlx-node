/** A workflow run's `state.json` as the agent writes it (manifest version 4): what the reader takes of it. */

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

import { describe, expect, it } from 'vite-plus/test';

import { mergeRunState, parseRunState, runStatePath } from '../src/agent-sessions/run-state.js';
import { parseUpdates, splitCompleteLines } from '../src/agent-sessions/transcript.js';
import type { TranscriptEntry } from '../src/agent-sessions/types.js';

type Json = Record<string, unknown>;
type Run = Extract<TranscriptEntry, { kind: 'workflow' }>;

/** The real run's `state.json` at its end (revision 26), its report replaced. */
const STATE = readFileSync(fileURLToPath(new URL('./fixtures/workflow-run/state.json', import.meta.url)), 'utf8');
const PARENT = readFileSync(fileURLToPath(new URL('./fixtures/workflow-run/parent.jsonl', import.meta.url)), 'utf8');
const RUN = 'wf_01a2168534c27fd1a477edb6fad7b0a4';

const doc = (): Json => JSON.parse(STATE) as Json;

describe('parseRunState', () => {
  // Mutations: drop the version guard → a version 5 file parses; read top-level fields → nothing comes back.
  it('reads every field under `state` of a version 4 file, and nothing of another version', () => {
    const state = parseRunState(STATE)!;
    expect(state.runId).toBe(RUN);
    expect(state.revision).toBe(26);
    expect(state.update).toMatchObject({
      run_id: RUN,
      revision: 26,
      name: 'research-advisory',
      status: 'complete',
      current_phase: 'Synthesize',
      agent_budget: 128,
      agents_used: 7,
      elapsed_ms: 2_323_877,
      phases: [
        { title: 'Recon', state: 'done' },
        { title: 'Verify', state: 'done' },
        { title: 'Synthesize', state: 'done' },
      ],
      last_event: 'workflow_completed',
    });
    expect((state.update.agents as Json[]).map((a) => a.label)).toEqual([
      'track:fix-reality',
      'track:claims-check',
      'track:holes',
      'track:process',
      'verify:fix-vectors',
      'verify:claims',
      'synthesize',
    ]);
    expect([...state.details]).toEqual([
      ['Recon', 'four independent read-only tracks'],
      ['Verify', 'adversarial checks of the key claims'],
      ['Synthesize', 'final maintainer-ready report'],
    ]);
    expect(state.history.filter((line) => line.event === 'log')).toEqual([
      {
        event: 'log',
        detail: 'Recon: 4 read-only tracks over GHSA-0000-0000-0000',
        ts: Date.parse('2026-11-26T06:22:38.302Z'),
      },
      {
        event: 'log',
        detail: 'Verify: adversarial checks of the two key claims',
        ts: Date.parse('2026-11-26T06:36:43.801Z'),
      },
      { event: 'log', detail: 'Synthesize: final maintainer-ready report', ts: Date.parse('2026-11-26T06:55:55.702Z') },
    ]);
    expect(state.history[0]).toEqual({
      event: 'workflow_started',
      detail: null,
      ts: Date.parse('2026-11-26T06:22:38.292Z'),
    });

    expect(parseRunState(JSON.stringify({ ...doc(), version: 5 }))).toBeNull();
    expect(parseRunState(JSON.stringify({ ...doc(), version: '4' }))).toBeNull();
    const flat = { version: 4, ...(doc().state as Json) };
    expect(parseRunState(JSON.stringify(flat))).toBeNull();
    expect(parseRunState('{"version":4,')).toBeNull();
  });

  it('marks the steps as the agent does: before the current one done, the current one active until the run is complete', () => {
    const at = (status: string, current: string | undefined) => {
      const state = doc().state as Json;
      const shaped = {
        ...doc(),
        state: {
          ...state,
          status,
          ...(current === undefined ? { current_phase: undefined } : { current_phase: current }),
        },
      };
      return (parseRunState(JSON.stringify(shaped))!.update.phases as Json[]).map((p) => p.state);
    };
    expect(at('active', 'Verify')).toEqual(['done', 'active', 'pending']);
    expect(at('cancelled', 'Verify')).toEqual(['done', 'active', 'pending']);
    expect(at('complete', 'Verify')).toEqual(['done', 'done', 'pending']);
    expect(at('active', 'Unlisted')).toEqual(['pending', 'pending', 'pending']);
    expect(at('active', undefined)).toEqual(['pending', 'pending', 'pending']);
  });
});

describe('mergeRunState', () => {
  const stored = (): Run => {
    const lines = splitCompleteLines(PARENT);
    const upTo = lines.findIndex((l) => l.includes('"revision":14'));
    return parseUpdates(`${lines.slice(0, upTo + 1).join('\n')}\n`).transcript.find(
      (e): e is Run => e.kind === 'workflow',
    )!;
  };

  it('a newer file replaces the fields and adds the details, the history and every step start; the entry stays', () => {
    const entry = stored();
    const before = structuredClone(entry);
    const merged = mergeRunState(entry, parseRunState(STATE)!, 1_795_676_500_000);
    expect(entry).toEqual(before);
    expect(merged).toMatchObject({
      revision: 26,
      status: 'complete',
      updatedTs: 1_795_676_500_000,
      elapsedMs: 2_323_877,
    });
    expect(merged.agents).toHaveLength(7);
    expect(merged.phases.map((p) => p.detail)).toEqual([
      'four independent read-only tracks',
      'adversarial checks of the key claims',
      'final maintainer-ready report',
    ]);
    expect(merged.phasesEntered.map((p) => p.title)).toEqual(['Recon', 'Verify', 'Synthesize']);
    expect(merged.history).toHaveLength(8);
  });

  it('an older file changes no field of the update, and still gives the details and the history', () => {
    const entry = stored();
    const old = doc();
    const state = old.state as Json;
    const stale = parseRunState(
      JSON.stringify({
        ...old,
        state: { ...state, revision: 9, status: 'active', agents: (state.agents as Json[]).slice(0, 2) },
      }),
    )!;
    const merged = mergeRunState(entry, stale, 1);
    expect(merged).toMatchObject({ revision: 14, status: 'active', updatedTs: entry.updatedTs, elapsedMs: 845_501 });
    expect(merged.agents.map((a) => a.label)).toEqual(entry.agents.map((a) => a.label));
    expect(merged.phases[1]).toEqual({
      title: 'Verify',
      state: 'active',
      detail: 'adversarial checks of the key claims',
    });
    expect(merged.history?.length).toBe(8);
  });
});

describe('runStatePath', () => {
  it('takes a run id that is one folder name only', () => {
    expect(runStatePath('/s/x', RUN)).toBe(`/s/x/workflows/${RUN}/state.json`);
    for (const bad of ['', '.', '..', '../x', 'a/b', 'a\\b', 'a\u0000b']) expect(runStatePath('/s/x', bad)).toBeNull();
  });
});

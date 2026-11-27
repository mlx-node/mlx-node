/**
 * Which prompt of the agent a message is: a stored message by its own `promptIndex`, a message sent
 * in this app run by order and text after the last stored one; a message the agent never ran, a steer, a note, a
 * continue divider and a wake are refused. And what a rewind to it removes, and the files of its plan.
 */

import { describe, expect, it } from 'vite-plus/test';

import type { TranscriptEntry } from '../src/agent-sessions/types.js';
import { visibleText } from '../src/chat/agent/visible-text.js';
import { answerEnd, planFiles, rewindCounts, targetOf } from '../src/chat/rewind-target.js';
import { REWIND_FILES_MAX } from '../src/chat/types.js';

const typed = (text: string, promptIndex: number | null = null): TranscriptEntry => ({
  kind: 'user',
  ts: 1,
  text,
  promptIndex,
});
const said = (text: string): TranscriptEntry => ({ kind: 'assistant', ts: 2, text });
const end = (): TranscriptEntry => ({ kind: 'turn-end', ts: 3, stopReason: 'end_turn' });

describe('targetOf', () => {
  const disk = [typed('one', 0), said('a'), end(), typed('two', 1), said('b'), end(), typed('three', 2), end()];

  it('a stored message is its own prompt index, when the disk holds that prompt with that text', () => {
    expect(targetOf(disk, disk, 3)).toEqual({ target: 1 });
    expect(targetOf(disk, disk, 0)).toEqual({ target: 0 });
    const other = [...disk.slice(0, 3), typed('changed', 1)];
    expect(targetOf(disk, other, 3)).toEqual({ refused: 'not-sent' });
  });

  it('live messages pair in order with the stored prompts after the last message that has an index', () => {
    const view = [typed('one', 0), said('a'), end(), typed('two'), said('b'), end(), typed('three'), end()];
    expect(targetOf(view, disk, 3)).toEqual({ target: 1 });
    expect(targetOf(view, disk, 6)).toEqual({ target: 2 });
  });

  it('a conversation started in this app run pairs from the first stored prompt', () => {
    const view = [typed('one'), said('a'), end(), typed('two'), said('b'), end()];
    expect(targetOf(view, disk, 0)).toEqual({ target: 0 });
    expect(targetOf(view, disk, 3)).toEqual({ target: 1 });
  });

  it('a message the agent never ran is no rewind point and takes no stored prompt from the next one', () => {
    const view = [typed('one', 0), said('a'), end(), typed('refused'), end(), typed('two'), said('b'), end()];
    expect(targetOf(view, disk, 3)).toEqual({ refused: 'not-sent' });
    expect(targetOf(view, disk, 5)).toEqual({ target: 1 });
  });

  it('a repeated text: the first sent one pairs with the first stored, the unsent second is refused', () => {
    const stored = [typed('again', 0), end()];
    const view = [typed('again'), end(), typed('again'), end()];
    expect(targetOf(view, stored, 0, undefined, new Set([2]))).toEqual({ target: 0 });
    expect(targetOf(view, stored, 2, undefined, new Set([2]))).toEqual({ refused: 'not-sent' });
  });

  // Mutation: drop the unsent rows from the pairing → the failed send takes the retry's prompt and the retry is refused.
  it('a send that never reached the agent, then the same text again: the retry is its prompt, the failed one none', () => {
    const stored = [typed('one', 0), said('a'), end(), typed('again', 1), said('b'), end()];
    const view = [typed('one', 0), said('a'), end(), typed('again'), typed('again'), said('b'), end()];
    const unsent = new Set([3]);
    expect(targetOf(view, stored, 3, undefined, unsent)).toEqual({ refused: 'not-sent' });
    expect(targetOf(view, stored, 4, undefined, unsent)).toEqual({ target: 1 });
  });

  // Mutation: pair only from the start (no pairing from the end) → the middle message takes the third's prompt, and a
  // rewind to it keeps it in the agent while the thread cuts it.
  it('the same text three times with one never stored and no host mark: no message of them is a rewind point', () => {
    const stored = [typed('one', 0), end(), typed('go on', 1), end(), typed('go on', 2), end()];
    const view = [typed('one', 0), end(), typed('go on'), typed('go on'), end(), typed('go on'), end()];
    for (const at of [2, 3, 5]) expect(targetOf(view, stored, at)).toEqual({ refused: 'not-sent' });
    // The host marked the failed one: the other two are their own prompts.
    const unsent = new Set([2]);
    expect(targetOf(view, stored, 2, undefined, unsent)).toEqual({ refused: 'not-sent' });
    expect(targetOf(view, stored, 3, undefined, unsent)).toEqual({ target: 1 });
    expect(targetOf(view, stored, 5, undefined, unsent)).toEqual({ target: 2 });
  });

  it('the automatic continue takes its stored prompt, so the message after it pairs with its own', () => {
    const stored = [
      typed('one', 0),
      end(),
      { ...typed('continue', 1), continued: true as const },
      end(),
      typed('two', 2),
    ];
    const view = [typed('one'), end(), { ...typed('continue'), continued: true as const }, end(), typed('two')];
    expect(targetOf(view, stored, 4)).toEqual({ target: 2 });
    expect(targetOf(view, stored, 2)).toEqual({ refused: 'not-a-message' });
  });

  // Mutation: drop the `message` wake step of the pairing walk → the message after it pairs with the steer's stored
  // prompt and is refused.
  it('a Send now the agent ran as its own turn takes its stored typed prompt, so the next message pairs with its own', () => {
    const stored = [
      typed('one', 0),
      said('a'),
      end(),
      typed('fix the test too', 1),
      said('ok'),
      end(),
      typed('two', 2),
      said('b'),
      end(),
    ];
    const view: TranscriptEntry[] = [
      typed('one', 0),
      said('a'),
      end(),
      { kind: 'user', ts: 1, text: 'fix the test too', promptIndex: null, interjection: true },
      { kind: 'wake', ts: 1, cause: 'message', text: 'fix the test too' },
      said('ok'),
      end(),
      typed('two'),
      said('b'),
      end(),
    ];
    expect(targetOf(view, stored, 7)).toEqual({ target: 2 });
    expect(targetOf(view, stored, 3)).toEqual({ refused: 'not-a-message' });
    expect(targetOf(view, stored, 4)).toEqual({ refused: 'not-a-message' });
    // A conversation started in this app run: the wake still takes its prompt.
    const fresh = view.map((e) => (e.kind === 'user' && e.promptIndex !== null ? { ...e, promptIndex: null } : e));
    expect(targetOf(fresh, stored, 0)).toEqual({ target: 0 });
    expect(targetOf(fresh, stored, 7)).toEqual({ target: 2 });
  });

  it('steers, notes, wakes and answers are refused', () => {
    const view: TranscriptEntry[] = [
      typed('one', 0),
      { kind: 'user', ts: 1, text: 'steer', promptIndex: null, interjection: true },
      { kind: 'user', ts: 1, text: 'note', promptIndex: null, note: true },
      { kind: 'wake', ts: 1, cause: 'task', text: 'woken' },
      said('a'),
    ];
    for (const at of [1, 2, 3, 4, 9]) expect(targetOf(view, disk, at)).toEqual({ refused: 'not-a-message' });
  });

  it('compares the disk text in the view form (the view is redacted)', () => {
    const stored = [typed('key sk-SECRET', 0)];
    const view = [typed('key [redacted]')];
    expect(targetOf(view, stored, 0)).toEqual({ refused: 'not-sent' });
    expect(targetOf(view, stored, 0, (text) => text.replace('sk-SECRET', '[redacted]'))).toEqual({ target: 0 });
  });
});

describe('rewindCounts', () => {
  it('counts the removed messages and steps, the live commands and the schedules the removed part made', () => {
    const view: TranscriptEntry[] = [
      typed('one', 0),
      {
        kind: 'tool',
        ts: 2,
        id: 'c0',
        name: 'run',
        title: 'run',
        status: 'completed',
        task: { id: 'early', kind: 'command', label: 'x' },
      },
      end(),
      { kind: 'user', ts: 1, text: 'two', promptIndex: 1, attachments: [{ kind: 'file', name: 'a.txt', bytes: 3 }] },
      { kind: 'model', ts: 2, modelId: 'm' },
      {
        kind: 'tool',
        ts: 2,
        id: 'c1',
        name: 'run',
        title: 'run',
        status: 'completed',
        task: { id: 'live', kind: 'command', label: 'dev' },
      },
      {
        kind: 'tool',
        ts: 2,
        id: 'c2',
        name: 'run',
        title: 'run',
        status: 'completed',
        task: { id: 'ended', kind: 'monitor', label: 'm', endTs: 5 },
      },
      {
        kind: 'tool',
        ts: 2,
        id: 'c3',
        name: 'sched',
        title: 'sched',
        status: 'completed',
        schedule: { id: 's1', prompt: 'p', repeat: null, nextTs: null },
      },
      {
        kind: 'tool',
        ts: 2,
        id: 'c4',
        name: 'sched',
        title: 'sched',
        status: 'completed',
        schedule: { id: 's2', prompt: 'p', repeat: null, nextTs: null, endTs: 9 },
      },
      { kind: 'subagent', ts: 2, childId: 'h', type: 'general', description: 'd' },
      said('b'),
      end(),
      typed('three', 2),
    ];
    expect(rewindCounts(view, 3, new Set(['early', 'live', 'ended']))).toEqual({
      messages: 2,
      steps: 7,
      tasks: ['live'],
      schedules: ['s1'],
      runs: [],
      helpers: true,
      attachments: true,
    });
    expect(rewindCounts(view, 12, new Set(['live']))).toEqual({
      messages: 1,
      steps: 0,
      tasks: [],
      schedules: [],
      runs: [],
      helpers: false,
      attachments: false,
    });
  });

  const launch = (id: string, runId: string | null, status = 'completed'): TranscriptEntry => ({
    kind: 'tool',
    ts: 2,
    id,
    name: 'workflow',
    title: 'Run workflow deep-research',
    status,
    run: { id: runId, kind: 'launch' },
  });
  const workflow = (runId: string): TranscriptEntry => ({
    kind: 'workflow',
    ts: 2,
    updatedTs: 2,
    runId,
    revision: 1,
    name: 'deep-research',
    status: 'user_paused',
    elapsedMs: 0,
    phases: [],
    currentPhase: null,
    agentBudget: null,
    agentsUsed: 0,
    agents: [],
    phasesEntered: [],
  });

  // Mutation: count every stoppable run → a run started before the target is stopped by the rewind.
  // Mutation: place a run at its run entry only → a run launched before the target whose entry came after it stops.
  it('lists the stoppable runs launched in the removed part: at the launch call, else at the run entry (T39)', () => {
    const view: TranscriptEntry[] = [
      typed('one', 0),
      launch('w0', 'wf_early'),
      end(),
      typed('two', 1),
      launch('w1', 'wf_late'),
      launch('w2', 'wf_done'),
      launch('w3', null),
      launch('w4', 'wf_refused', 'failed'),
      end(),
      workflow('wf_early'),
      workflow('wf_cut'),
      typed('three', 2),
      workflow('wf_late'),
    ];
    const stoppable = new Set(['wf_early', 'wf_late', 'wf_cut', 'wf_refused']);
    expect(rewindCounts(view, 3, new Set(), stoppable).runs).toEqual(['wf_late', 'wf_cut']);
    expect(rewindCounts(view, 11, new Set(), stoppable).runs).toEqual([]);
    expect(rewindCounts(view, 0, new Set(), stoppable).runs).toEqual(['wf_early', 'wf_late', 'wf_cut']);
    expect(rewindCounts(view, 3, new Set()).runs).toEqual([]);
  });
});

describe('answerEnd (the last prompt a fork with this answer keeps)', () => {
  const continued = (promptIndex: number): TranscriptEntry => ({
    kind: 'user',
    ts: 1,
    text: 'continue',
    promptIndex,
    continued: true,
  });
  const steer: TranscriptEntry = { kind: 'user', ts: 1, text: 'also', promptIndex: null, interjection: true };

  // Mutation: return `index` alone → the continued part of the answer is cut from the copy.
  it('takes the automatic continues of the answer, and stops at the next typed message', () => {
    const disk = [
      typed('one', 0),
      said('a'),
      continued(1),
      said('a2'),
      steer,
      continued(2),
      end(),
      typed('two', 3),
      said('b'),
      continued(4),
      end(),
    ];
    expect(answerEnd(disk, 0)).toBe(2);
    expect(answerEnd(disk, 3)).toBe(4);
  });

  it('an answer with no continue ends at its own prompt; an index the disk lacks stays as it is', () => {
    const disk = [typed('one', 0), said('a'), end(), typed('two', 1), said('b'), end()];
    expect(answerEnd(disk, 0)).toBe(0);
    expect(answerEnd(disk, 1)).toBe(1);
    expect(answerEnd(disk, 5)).toBe(5);
  });
});

describe('the files of a rewind plan', () => {
  // Mutation: sort by path alone → a conflict past the 200th file falls out of the list.
  it('changed ones first, then by path; each path shown as the page may draw it; at most 200 and a count of the rest', () => {
    const out = planFiles(
      [
        { path: 'src/b.ts', state: 'clean' },
        { path: 'src/a.ts', state: 'clean' },
        { path: 'z.md', state: 'deleted' },
        { path: 'docs/\u202Eevil.md', state: 'modified' },
      ],
      visibleText,
    );
    expect(out).toEqual({
      files: [
        { path: 'docs/\\u{202E}evil.md', state: 'modified' },
        { path: 'z.md', state: 'deleted' },
        { path: 'src/a.ts', state: 'clean' },
        { path: 'src/b.ts', state: 'clean' },
      ],
      filesMore: 0,
    });
    const many = Array.from({ length: REWIND_FILES_MAX + 5 }, (_, i) => ({
      path: `f${String(i).padStart(3, '0')}.ts`,
      state: 'clean' as const,
    }));
    const capped = planFiles([...many, { path: 'zz.ts', state: 'created' }], (path) => path);
    expect(capped.files).toHaveLength(REWIND_FILES_MAX);
    expect(capped.files[0]).toEqual({ path: 'zz.ts', state: 'created' });
    expect(capped.filesMore).toBe(6);
    expect(planFiles([], (path) => path)).toEqual({ files: [], filesMore: 0 });
  });
});

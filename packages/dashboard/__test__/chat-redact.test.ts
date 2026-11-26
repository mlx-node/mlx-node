import { describe, expect, it } from 'vite-plus/test';

import { MAX_ROW_CHARS } from '../src/agent-sessions/line-diff.js';
import { parseUpdates } from '../src/agent-sessions/transcript.js';
import { ActivityTracker, TASK_EVENT_CHARS } from '../src/chat/agent/activity-tracker.js';
import { Conversation, type HostView } from '../src/chat/conversation.js';
import {
  needleList,
  REDACTED,
  redactCutText,
  redactDeep,
  redactEntry,
  redactText,
  StreamRedactor,
} from '../src/chat/redact.js';
import { promptTitle } from '../src/chat/titles.js';
import type { ConversationEvent, ConversationInfo, LiveActivity } from '../src/chat/types.js';

const KEY = 'marker-KEY-12345678';
const OLD = 'retired-KEY-87654321';
const TOKEN = 'tok-INFERENCE-abcdefghijklmnop';

describe('redactText', () => {
  const needles = needleList([KEY, OLD, TOKEN, '']);

  it('replaces every occurrence of every needle, current and retired', () => {
    expect(redactText(`a ${KEY} b ${OLD} c ${KEY}${TOKEN}`, needles)).toBe(
      `a ${REDACTED} b ${REDACTED} c ${REDACTED}${REDACTED}`,
    );
  });

  it('leaves text without a needle unchanged and ignores empty needles', () => {
    expect(needleList(['', 'x', 'x'])).toEqual(['x']);
    expect(redactText('nothing here', needles)).toBe('nothing here');
    expect(redactText('', needles)).toBe('');
  });

  it('prefers the longest needle at a position', () => {
    expect(redactText('abcdef', needleList(['abc', 'abcdef']))).toBe(REDACTED);
  });

  it('redacts every string of a value, keys included', () => {
    const value = { [KEY]: [`x${KEY}`, 1, null, { deep: TOKEN }], n: 2 };
    expect(redactDeep(value, needles)).toEqual({ [REDACTED]: [`x${REDACTED}`, 1, null, { deep: REDACTED }], n: 2 });
  });
});

describe('StreamRedactor', () => {
  it('catches a secret split over two chunks by holding back the end that could start it', () => {
    const r = new StreamRedactor(needleList([KEY]));
    const out = [r.push('The key is marker-'), r.push('KEY-12345678 done'), r.flush()].join('');
    expect(out).toBe(`The key is ${REDACTED} done`);
    expect(out).not.toContain('marker-');
  });

  it('sends nothing that could still become a secret, and the rest at once', () => {
    const r = new StreamRedactor(needleList([KEY]));
    const first = r.push('hello marker-KEY-1234567');
    expect(first).toBe('hello ');
    expect(r.push('8!')).toBe(`${REDACTED}!`);
    expect(r.flush()).toBe('');
  });

  // Mutation: hold back a fixed `longest needle − 1` characters → red (the 43-character token holds the whole thought).
  it('sends prose that cannot start a needle at once, with a 43-character token among the needles', () => {
    const token = 'Zq3'.repeat(14) + 'x';
    const r = new StreamRedactor(needleList([token, KEY]));
    expect(r.push('Now write ')).toBe('Now write ');
    expect(r.push('the workflow file.')).toBe('the workflow file.');
    expect(r.push(' Zq3Zq')).toBe(' ');
    expect(r.flush()).toBe('Zq3Zq');
  });

  // Mutation: hold the shortest end that starts a needle, or test only the longest needle → red (part of a key leaks).
  it('at every split point gives the redacted whole, needles that repeat their own start included', () => {
    const needles = needleList(['aaab', 'abab', 'abcab', 'ab-long-needle-xyz']);
    const texts = ['xaaaab', 'aababab', 'abcabcab-', 'zabababz', 'ab-long-needle-xyab', 'aab-long-needle-xyzab'];
    for (const text of texts) {
      for (let cut = 0; cut <= text.length; cut += 1) {
        for (let cut2 = cut; cut2 <= text.length; cut2 += 1) {
          const r = new StreamRedactor(needles);
          const out = r.push(text.slice(0, cut)) + r.push(text.slice(cut, cut2)) + r.push(text.slice(cut2)) + r.flush();
          expect(out, `${text} @${cut},${cut2}`).toBe(redactText(text, needles));
        }
      }
    }
  });

  it('chunk by chunk gives the same text as redacting the whole', () => {
    const text = `x${KEY}yy${KEY}${KEY}z${OLD}end`;
    const needles = needleList([KEY, OLD]);
    for (const size of [1, 2, 3, 7, 19, 100]) {
      const r = new StreamRedactor(needles);
      let out = '';
      for (let i = 0; i < text.length; i += size) out += r.push(text.slice(i, i + size));
      out += r.flush();
      expect(out).toBe(redactText(text, needles));
    }
  });

  it('never splits a surrogate pair at the hold-back border', () => {
    const r = new StreamRedactor(needleList(['abc']));
    const out = r.push('x\uD83D');
    expect(out).toBe('x');
    expect(out + r.push('\uDE00y') + r.flush()).toBe('x😀y');
  });

  it('passes text straight through with no needles', () => {
    const r = new StreamRedactor([]);
    expect(r.push('abc')).toBe('abc');
    expect(r.flush()).toBe('');
  });
});

describe('redactCutText', () => {
  const key = 'fc-0123456789abcdef';
  it('drops the start of a key cut in two at the end', () => {
    expect(redactCutText(`output ${key.slice(0, 9)}`, [key])).toBe('output ');
    expect(redactCutText(`a ${key} b ${key.slice(0, 4)}`, [key])).toBe(`a ${REDACTED} b `);
  });

  it('keeps an end that cannot start a key', () => {
    expect(redactCutText('output done', [key])).toBe('output done');
    expect(redactCutText(`ends with ${key}`, [key])).toBe(`ends with ${REDACTED}`);
  });
});

describe('a prompt title cut inside a key', () => {
  const key = 'sk-ant-api03-A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0';
  it('shows no part of the key under a title key, at any depth', () => {
    const title = promptTitle(`use this key ${key} to call the API`);
    expect(title).toContain(key.slice(0, 20));
    const out = redactDeep({ title, rows: [{ title }], parent: { id: 'a', title } }, [key]);
    for (const shown of [out.title, out.rows[0]!.title, out.parent.title]) {
      expect(shown).toBe('use this key…');
    }
    expect(redactDeep({ title: `named ${key}` }, [key]).title).toBe(`named ${REDACTED}`);
    expect(redactDeep({ text: `cut s…` }, [key]).text).toBe('cut s…');
  });
});

describe('a provider key inside an edit diff', () => {
  const key = 'sk-ant-api03-A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0';
  const entryOf = (oldText: string, newText: string) => {
    const update = {
      sessionUpdate: 'tool_call_update',
      toolCallId: 'call_1',
      status: 'completed',
      content: [{ type: 'diff', path: '/w/.env', oldText, newText, _meta: { details: [] } }],
    };
    const line = JSON.stringify({ method: 'session/update', params: { sessionId: 's', update } });
    return parseUpdates(`${line}\n`).transcript[0]!;
  };

  it('is redacted in a hunk body', () => {
    const out = redactEntry(entryOf('API_KEY=\n', `API_KEY=${key}\n`), [key]);
    const body = out.kind === 'tool' ? out.edits![0]!.hunks[0]!.body : '';
    expect(body).toBe(`-API_KEY=\n+API_KEY=${REDACTED}`);
    expect(JSON.stringify(out)).not.toContain(key.slice(0, 12));
  });

  it('leaves no start of a key in a row cut at its length', () => {
    const pad = 'x'.repeat(MAX_ROW_CHARS - 10);
    const out = redactEntry(entryOf('', `${pad}${key}\n`), [key]);
    const row = out.kind === 'tool' ? out.edits![0]!.hunks[0]!.body : '';
    expect(row).toBe(`+${pad}…`);
  });
});

describe("a key the agent's cut splits at the end of a run report", () => {
  const key = 'sk-ant-api03-A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0';
  const line = (update: Record<string, unknown>, xai = true) =>
    JSON.stringify({ method: xai ? '_x.ai/session/update' : 'session/update', params: { sessionId: 's', update } });
  const run = (revision: number, status: string, extra: Record<string, unknown> = {}) =>
    line({ sessionUpdate: 'workflow_updated', run_id: 'wf_1', revision, name: 'probe', status, ...extra });
  // The agent cuts a report in bytes and ends it with `…`; the app's own 20 KiB cap never comes into it.
  const report = `# Findings\n\nThe token is ${key.slice(0, 14)}…`;
  const entries = parseUpdates(
    [
      run(1, 'active'),
      run(2, 'complete', { result_summary: report }),
      line(
        {
          sessionUpdate: 'user_message_chunk',
          content: { type: 'text', text: 'A background workflow stopped.' },
          _meta: { hideFromScrollback: true },
        },
        false,
      ),
      line({ sessionUpdate: 'turn_completed', prompt_id: 'workflow-completed-wf_1-2', stop_reason: 'end_turn' }),
      '',
    ].join('\n'),
  ).transcript;

  // Mutation: redact a report's cut end only when the app cut it (`truncated`) → the start of the key shows.
  it('drops the start of the key in the run entry and in the wake row, with no app cut', () => {
    const entry = entries.find((e) => e.kind === 'workflow');
    const wake = entries.find((e) => e.kind === 'wake');
    expect(entry?.kind === 'workflow' && entry.truncated).toBeFalsy();
    expect(wake?.kind === 'wake' && wake.workflow?.snapshot?.report).toBe(report);
    const shownRun = redactEntry(entry!, [key]);
    const shownWake = redactEntry(wake!, [key]);
    expect(shownRun.kind === 'workflow' && shownRun.report).toBe('# Findings\n\nThe token is …');
    expect(shownWake.kind === 'wake' && shownWake.workflow?.snapshot?.report).toBe('# Findings\n\nThe token is …');
    expect(JSON.stringify([shownRun, shownWake])).not.toContain(key.slice(0, 8));
    const other = redactEntry(entry!, ['zz-other-KEY-1234']);
    expect(other.kind === 'workflow' && other.report).toBe(report);
  });
});

describe('a key in a hook result', () => {
  const key = 'sk-ant-api03-A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0';
  const cut = `${'x'.repeat(40)} ${key.slice(0, 14)}…`;
  const hook = (detail: string) =>
    ({
      kind: 'notice',
      ts: 1,
      code: 'hook',
      event: 'stop',
      hook: 'global/lint',
      detail,
      times: 1,
      lastTs: 1,
    }) as const;
  const tool = (blocked: string) =>
    ({ kind: 'tool', ts: 1, id: 'c1', name: 'x', title: 'x', status: 'failed', blocked }) as const;
  const goal = (pauseMessage: string) =>
    ({
      kind: 'goal',
      ts: 1,
      goalId: 'g1',
      objective: 'fix',
      status: 'blocked',
      phase: 'idle',
      tokensUsed: 0,
      elapsedMs: 0,
      pauseMessage,
      asOf: 1,
    }) as const;

  // Mutation: drop the hook branch of `redactEntry` → the cut end keeps the start of the key.
  it('is redacted whole in the error, the note and the blocked words; a cut end that could start it is dropped', () => {
    for (const [whole, short] of [
      [hook(`token ${key} bad`), hook(cut)],
      [
        { kind: 'notice', ts: 1, code: 'hook-note', text: `token ${key} bad` } as const,
        { kind: 'notice', ts: 1, code: 'hook-note', text: cut } as const,
      ],
      [tool(`token ${key} bad`), tool(cut)],
      [goal(`token ${key} bad`), goal(cut)],
    ] as const) {
      expect(JSON.stringify(redactEntry(whole, [key]))).toContain(`token ${REDACTED} bad`);
      const out = JSON.stringify(redactEntry(short, [key]));
      expect(out).not.toContain(key.slice(0, 6));
      expect(out).toContain(`${'x'.repeat(40)} …`);
    }
  });
});

describe('a key in a monitor’s output (T2)', () => {
  const key = 'sk-ant-api03-A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8S9t0';
  const info: ConversationInfo = {
    id: 'c-mon',
    mode: 'work',
    title: 'Monitor',
    cwd: '/w',
    modelId: 'm',
    createdAt: 1,
    updatedAt: 1,
    hot: true,
    permissionMode: 'ask',
  };
  /** The host tracker's own facts after these output lines of one monitor, and a wake after them. */
  const live = (texts: string[]): LiveActivity => {
    const t = new ActivityTracker();
    t.apply({
      xai: true,
      update: {
        sessionUpdate: 'task_backgrounded',
        tool_call_id: 'call-mon-1',
        task_id: 'mon-1',
        command: 'tail -f build.log',
        cwd: '/w',
        output_file: '/tmp/mon-1.log',
        monitor_description: 'Watch the build log',
      },
    });
    texts.forEach((text, i) => t.event('mon-1', text, i + 1, `Watch ${key}`));
    t.wakeStarted(9);
    return t.live();
  };
  const hostView = (activity: LiveActivity): HostView => ({
    turn: null,
    interactions: [],
    compaction: null,
    hookRun: null,
    recap: null,
    runs: [],
    live: activity,
    commands: null,
  });
  // The host's cut lands 19 characters into the key.
  const pad = 'x'.repeat(TASK_EVENT_CHARS - 20);
  // A line the monitor itself ends with `…`, uncut, whose last character starts the key.
  const ellipsis = '[build] Watching for file changes…';

  // Mutation: send `live` as it is in `activityChanged` → the activity event carries the key.
  // Mutation: read a cut from a closing `…` → the monitor's own "changes…" loses its "s".
  it('is redacted in the activity event and in the sync; a cut end that could start it is dropped, an uncut end kept', () => {
    const conversation = new Conversation(info, { needles: () => [key], flushMs: 1 });
    conversation.seed([], [], null);
    const events: ConversationEvent[] = [];
    const activity = live([`export KEY=${key}`, `${pad} ${key}`, ellipsis]);
    conversation.attach((event) => events.push(event), hostView(activity));
    conversation.activityChanged(activity);
    const sync = events[0] as Extract<ConversationEvent, { type: 'sync' }>;
    const sent = events[1] as Extract<ConversationEvent, { type: 'activity' }>;
    expect(sent.type).toBe('activity');
    for (const out of [sync.live, sent.live]) {
      expect(JSON.stringify(out)).not.toContain(key.slice(0, 6));
      expect(out.tasks[0]!.recent.map((e) => e.text)).toEqual([`export KEY=${REDACTED}`, `${pad} …`, ellipsis]);
      expect(out.reports).toEqual([{ wakeTs: 9, monitors: [{ id: 'mon-1', description: `Watch ${REDACTED}` }] }]);
    }
    // The host's own copy is not changed by the redaction.
    expect(activity.tasks[0]!.recent[0]!.text).toBe(`export KEY=${key}`);
    expect(activity.reports[0]!.monitors[0]!.description).toBe(`Watch ${key}`);
  });

  // Mutation: copy an agent's `now` as it is → the title of its open call carries the key to the page.
  it("an agent's open call it names now is redacted on the way out", () => {
    const t = new ActivityTracker();
    const child = 'agent-child-1';
    t.apply({
      xai: true,
      update: {
        sessionUpdate: 'subagent_spawned',
        subagent_id: child,
        child_session_id: child,
        subagent_type: 'general-purpose',
        description: 'track:process',
        workflow_run_id: 'wf_1',
      },
    });
    t.progress(
      {
        childId: child,
        turns: 1,
        toolCalls: 0,
        tokens: 0,
        contextPct: 0,
        errors: 0,
        durationMs: 0,
        contextWindow: null,
      },
      1,
    );
    t.childLine(child, { sessionUpdate: 'tool_call', toolCallId: 't1', title: `curl -H "token: ${key}" api` });
    const activity = t.live();
    const conversation = new Conversation(info, { needles: () => [key], flushMs: 1 });
    conversation.seed([], [], null);
    const events: ConversationEvent[] = [];
    conversation.attach((event) => events.push(event), hostView(activity));
    conversation.activityChanged(activity);
    const sent = events[1] as Extract<ConversationEvent, { type: 'activity' }>;
    for (const out of [(events[0] as Extract<ConversationEvent, { type: 'sync' }>).live, sent.live]) {
      expect(out.helpers[0]!.now).toEqual({ kind: 'tool', title: `curl -H "token: ${REDACTED}" api` });
    }
    expect(activity.helpers[0]!.now).toEqual({ kind: 'tool', title: `curl -H "token: ${key}" api` });
  });
});

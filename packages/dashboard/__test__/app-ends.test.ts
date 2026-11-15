/**
 * The line of a turn a Control Panel restart stopped, rebuilt from the stored thread and MAIN's record. The numbers
 * are those of the 2026-11-13 22:13 (+08) incident.
 */

import { describe, expect, it } from 'vite-plus/test';

import { appEndFor } from '../ui/src/lib/app-ends.js';
import { CHAT } from '../ui/src/lib/chat-strings.js';
import { splitTurns, turnOutcome, turnStartedAt } from '../ui/src/lib/chat-turns.js';
import type { AppEnd } from '../ui/src/lib/chat-types.js';
import type { TranscriptEntry } from '../ui/src/lib/types.js';

const CONV = '01a1d519-7b3c-7d4e-8f50-61728394a5b6';
const OTHER = '019a1f2e-7b3c-7d4e-8f50-000000000001';

/** 2026-11-13 hh:mm:ss.ms at +08:00, epoch ms. */
const at = (h: number, m: number, s: number, ms = 0): number => Date.UTC(2026, 10, 13, h - 8, m, s, ms);

const NO_PING: AppEnd = {
  trigger: 'no-ping',
  decidedAt: at(22, 13, 11, 870),
  exitAt: at(22, 13, 13, 900),
  silentMs: 120_000,
  code: 0,
  convs: [CONV],
  cancelled: [CONV],
};

const NO_PING_LINE =
  'The app restarted its Control Panel because it did not answer for 120 s. The turn was stopped by the app, not by you.';

function incident(): TranscriptEntry[] {
  return [
    { kind: 'user', ts: at(21, 29, 57), text: 'build it', promptIndex: 0 },
    { kind: 'assistant', ts: at(22, 12, 1), text: 'Building.' },
    { kind: 'turn-end', ts: at(22, 13, 11), stopReason: 'cancelled' },
    { kind: 'wake', ts: at(22, 13, 11), cause: 'task', text: 'task finished' },
  ];
}

function nextOf(turns: ReturnType<typeof splitTurns>, index: number): { startedAt: number | null } | null {
  const next = turns[index + 1];
  return next === undefined ? null : { startedAt: turnStartedAt(next) };
}

function outcomes(entries: TranscriptEntry[], conversationId: string, ends: AppEnd[]): Array<string | null> {
  const turns = splitTurns(entries, []);
  return turns.map(
    (turn, index) =>
      turnOutcome(turn, {
        last: false,
        running: false,
        live: null,
        mode: 'work',
        app: { ends, conversationId, next: nextOf(turns, index) },
      })?.text ?? null,
  );
}

describe('the thread after a Control Panel restart', () => {
  it('gives the stopped turn and the woken turn the no-ping line', () => {
    expect(outcomes(incident(), CONV, [NO_PING])).toEqual([NO_PING_LINE, NO_PING_LINE]);
    const turns = splitTurns(incident(), []);
    expect(
      turnOutcome(turns[0]!, {
        last: false,
        running: false,
        live: null,
        mode: 'work',
        app: { ends: [NO_PING], conversationId: CONV, next: nextOf(turns, 0) },
      }),
    ).toMatchObject({ tone: 'warning', action: 'continue' });
  });

  it('without the record the same thread keeps its old words', () => {
    expect(outcomes(incident(), CONV, [])).toEqual([CHAT.outcomes['stopped-stored'], CHAT.outcomes.unfinished]);
  });

  it('a Stop the user pressed 20 s earlier keeps "stopped before the end"', () => {
    const entries: TranscriptEntry[] = [
      { kind: 'user', ts: at(22, 12, 30), text: 'go', promptIndex: 0 },
      { kind: 'turn-end', ts: at(22, 12, 51), stopReason: 'cancelled' },
    ];
    expect(outcomes(entries, CONV, [NO_PING])).toEqual([CHAT.outcomes['stopped-stored']]);
  });

  it('an old unfinished turn of another conversation keeps "did not finish"', () => {
    const entries: TranscriptEntry[] = [{ kind: 'user', ts: at(21, 0, 0), text: 'old', promptIndex: 0 }];
    expect(outcomes(entries, OTHER, [NO_PING])).toEqual([CHAT.outcomes.unfinished]);
  });

  it('with no cancelled list a stored cancelled end is never blamed on the app', () => {
    const lost: AppEnd = { ...NO_PING, cancelled: null };
    expect(outcomes(incident(), CONV, [lost])).toEqual([CHAT.outcomes['stopped-stored'], NO_PING_LINE]);
  });

  it('a turn started after the restart is not given the line', () => {
    const entries: TranscriptEntry[] = [{ kind: 'user', ts: at(22, 20, 0), text: 'later', promptIndex: 0 }];
    expect(outcomes(entries, CONV, [NO_PING])).toEqual([CHAT.outcomes.unfinished]);
  });

  it('the stopped turn keeps its line after Continue starts the next one', () => {
    const killed: AppEnd = { ...NO_PING, cancelled: null };
    const entries: TranscriptEntry[] = [
      { kind: 'user', ts: at(22, 10, 0), text: 'build it', promptIndex: 0 },
      { kind: 'assistant', ts: at(22, 11, 0), text: 'Building.' },
      { kind: 'user', ts: at(22, 14, 0), text: CHAT.continuePrompt, promptIndex: 1 },
    ];
    expect(outcomes(entries, CONV, [killed])).toEqual([NO_PING_LINE, CHAT.outcomes.unfinished]);
  });

  it('an unfinished turn followed by one that started before the exit is not the stopped turn', () => {
    const entries: TranscriptEntry[] = [
      { kind: 'user', ts: at(22, 10, 0), text: 'first', promptIndex: 0 },
      { kind: 'user', ts: at(22, 12, 0), text: 'second', promptIndex: 1 },
    ];
    expect(outcomes(entries, CONV, [NO_PING])).toEqual([CHAT.outcomes.unfinished, NO_PING_LINE]);
  });

  it('a live outcome the page saw still wins', () => {
    const turns = splitTurns(incident(), []);
    expect(
      turnOutcome(turns[1]!, {
        last: true,
        running: false,
        live: { outcome: 'interrupted' },
        mode: 'work',
        app: { ends: [NO_PING], conversationId: CONV, next: null },
      })?.text,
    ).toBe(CHAT.outcomes.interrupted);
  });

  it('names each trigger', () => {
    const turns = splitTurns(incident(), []);
    const line = (end: AppEnd): string | undefined =>
      turnOutcome(turns[0]!, {
        last: false,
        running: false,
        live: null,
        mode: 'work',
        app: { ends: [end], conversationId: CONV, next: nextOf(turns, 0) },
      })?.text;
    expect(line({ ...NO_PING, trigger: 'crash', code: 9 })).toBe(
      'The Control Panel of the app stopped by itself (exit code 9) and was restarted. The turn was stopped by the app, not by you.',
    );
    expect(line({ ...NO_PING, trigger: 'signal' })).toBe(
      'Something outside the app stopped its Control Panel. The turn was stopped with it, not by you.',
    );
    expect(line({ ...NO_PING, trigger: 'user' })).toBe('The app was closed during this turn.');
  });

  it('picks the earliest end that fits', () => {
    const later: AppEnd = { ...NO_PING, decidedAt: at(23, 0, 0), exitAt: at(23, 0, 2), silentMs: 30_000 };
    expect(
      appEndFor({ end: null, startedAt: at(22, 13, 11) }, { ends: [later, NO_PING], conversationId: CONV, next: null }),
    ).toBe(NO_PING);
  });
});

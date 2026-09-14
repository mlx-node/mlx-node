import type { FileEntry } from '@earendil-works/pi-coding-agent';
import { describe, expect, it } from 'vite-plus/test';

import { deriveDelegation } from '../src/ingest/delegation.js';
import { activeBranchEntries } from '../src/ingest/sessions.js';
import { delegationFixture } from './helpers/delegation-fixture.js';

function measure(records: object[], complete = true, id = 'delegate') {
  const entries = records as FileEntry[];
  return deriveDelegation(entries, activeBranchEntries(entries).at(-1)?.id, id, complete);
}

describe('delegate evidence compression', () => {
  it('counts successful evidence minus final text, excluding model usage, thinking and prompts', () => {
    expect(measure(delegationFixture())).toEqual({
      status: 'complete',
      tokenizer: 'o200k_base',
      evidenceTokens: 3,
      handoffTokens: 1,
      savedTokens: 2,
      savingsRatio: 2 / 3,
    });
  });

  it('retains negative savings and zero savings', () => {
    expect(measure(delegationFixture('delegate', 'OK', 'alpha beta gamma'))).toMatchObject({
      savedTokens: -2,
      savingsRatio: -2,
    });
    expect(measure(delegationFixture('delegate', 'OK', 'OK'))).toMatchObject({ savedTokens: 0, savingsRatio: 0 });
  });

  it('does not infer delegation from user text or inherit a fork parent classification', () => {
    const ordinary = delegationFixture();
    ordinary.splice(1, 1);
    expect(measure(ordinary)).toBeNull();
    expect(measure(delegationFixture(), true, 'fork-child')).toBeNull();
  });

  it('deduplicates evidence and excludes tool errors', () => {
    const entries = delegationFixture();
    const evidence = entries[4]!;
    entries.splice(
      5,
      0,
      { ...evidence, id: 'duplicate', parentId: 'e' },
      {
        ...evidence,
        id: 'error',
        parentId: 'duplicate',
        message: { ...evidence.message!, isError: true, content: [{ type: 'text', text: 'error '.repeat(100) }] },
      },
    );
    entries.at(-1)!.parentId = 'error';
    expect(measure(entries)).toMatchObject({ evidenceTokens: 3, handoffTokens: 1, savedTokens: 2 });
  });

  it.each(['error', 'aborted', 'length', 'toolUse'])('withholds a %s handoff', (stopReason) => {
    const entries = delegationFixture();
    entries.at(-1)!.message!.stopReason = stopReason;
    expect(measure(entries)).toEqual({ status: 'incomplete', reason: 'no-final-handoff' });
  });

  it('withholds an unfinished write, an empty handoff, and a run with no successful evidence', () => {
    expect(measure(delegationFixture(), false)).toEqual({ status: 'incomplete', reason: 'partial-record' });
    expect(measure(delegationFixture('delegate', 'OK', ''))).toMatchObject({ status: 'incomplete' });
    expect(measure(delegationFixture('delegate', '', 'done'))).toEqual({
      status: 'unavailable',
      reason: 'no-evidence',
    });
  });

  it('does not count a regular resume after the delegate handoff', () => {
    const entries = delegationFixture();
    const user = entries[2]!;
    const result = entries.at(-1)!;
    entries.push(
      { ...user, id: 'ordinary-u', parentId: 'f' },
      {
        ...result,
        id: 'ordinary-a',
        parentId: 'ordinary-u',
        message: { ...result.message!, content: [{ type: 'text', text: 'ordinary '.repeat(200) }] },
      },
    );
    expect(measure(entries)).toMatchObject({ evidenceTokens: 3, handoffTokens: 1 });
  });

  it('sums explicitly marked follow-ups once each and includes their output overhead', () => {
    const entries = delegationFixture();
    const follow = delegationFixture('delegate', '', 'done')
      .slice(1)
      .map((e) => ({
        ...e,
        id: `next-${e.id}`,
        parentId: e.parentId ? `next-${e.parentId}` : 'f',
      }));
    expect(measure([...entries, ...follow])).toMatchObject({ evidenceTokens: 3, handoffTokens: 2, savedTokens: 1 });
  });

  it('retains pre-compaction evidence and ignores abandoned branches and detached rename metadata', () => {
    const entries: object[] = delegationFixture();
    entries.push({
      type: 'compaction',
      id: 'c',
      parentId: 'f',
      timestamp: '',
      firstKeptEntryId: 'f',
      summary: 'compacted',
      tokensBefore: 50,
    });
    expect(measure(entries)).toMatchObject({ evidenceTokens: 3, handoffTokens: 1 });
    const clean = delegationFixture();
    clean.splice(4, 0, {
      ...clean[4]!,
      id: 'abandoned',
      message: { ...clean[4]!.message!, content: [{ type: 'text', text: 'discard '.repeat(1000) }] },
    });
    const renamed = [...clean, { type: 'session_info', id: 'rename', parentId: null, timestamp: '', name: 'New name' }];
    expect(measure(renamed)).toMatchObject({ evidenceTokens: 3, handoffTokens: 1 });
  });

  it('never reports savings after a deterministic blocked handoff', () => {
    const entries: object[] = delegationFixture();
    entries.splice(5, 0, {
      type: 'custom',
      id: 'blocked',
      parentId: 'e',
      timestamp: '',
      customType: 'mlx-delegate-handoff',
      data: { reason: 'permission denied' },
    });
    (entries.at(-1) as { parentId: string }).parentId = 'blocked';
    expect(measure(entries)).toMatchObject({ status: 'incomplete' });
  });

  it('handles literal special-token text and withholds non-text comparisons', () => {
    expect(measure(delegationFixture('delegate', '<|endoftext|>', 'done'))).toMatchObject({ status: 'complete' });
    const entries = delegationFixture();
    entries[4]!.message!.content = [{ type: 'image', data: 'abc', mimeType: 'image/png' }] as never;
    expect(measure(entries)).toEqual({ status: 'unavailable', reason: 'unsupported-content' });
  });
});

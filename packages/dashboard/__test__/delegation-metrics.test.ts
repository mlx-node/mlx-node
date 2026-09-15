import type { FileEntry } from '@earendil-works/pi-coding-agent';
import { describe, expect, it, vi } from 'vite-plus/test';

import { deriveDelegation } from '../src/ingest/delegation.js';
import { activeBranchEntries } from '../src/ingest/sessions.js';
import * as tokenizer from '../src/ingest/tokenizer.js';
import { delegationFixture } from './helpers/delegation-fixture.js';

function measure(records: object[], complete = true, id = 'delegate') {
  const entries = records as FileEntry[];
  return deriveDelegation(entries, activeBranchEntries(entries).at(-1)?.id, id, complete);
}

describe('delegate evidence compression', () => {
  it('keeps delegate identity with an unavailable estimate when native tokenization fails', async () => {
    const count = vi.spyOn(tokenizer, 'countDelegateTokens').mockRejectedValueOnce(new Error('Addon unavailable'));
    try {
      expect(await measure(delegationFixture())).toEqual({ status: 'unavailable', reason: 'tokenizer-unavailable' });
    } finally {
      count.mockRestore();
    }
  });
  it('counts successful evidence minus final text, excluding model usage, thinking and prompts', async () => {
    expect(await measure(delegationFixture())).toEqual({
      status: 'complete',
      tokenizer: 'o200k_base',
      evidenceTokens: 3,
      handoffTokens: 1,
      savedTokens: 2,
      savingsRatio: 2 / 3,
    });
  });

  it('retains negative savings and zero savings', async () => {
    expect(await measure(delegationFixture('delegate', 'OK', 'alpha beta gamma'))).toMatchObject({
      savedTokens: -2,
      savingsRatio: -2,
    });
    expect(await measure(delegationFixture('delegate', 'OK', 'OK'))).toMatchObject({ savedTokens: 0, savingsRatio: 0 });
  });

  it('does not infer delegation from user text or inherit a fork parent classification', async () => {
    const ordinary = delegationFixture();
    ordinary.splice(1, 1);
    expect(await measure(ordinary)).toBeNull();
    expect(await measure(delegationFixture(), true, 'fork-child')).toBeNull();
  });

  it('deduplicates evidence and excludes tool errors', async () => {
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
    expect(await measure(entries)).toMatchObject({ evidenceTokens: 3, handoffTokens: 1, savedTokens: 2 });
  });

  it.each(['error', 'aborted', 'length', 'toolUse'])('withholds a %s handoff', async (stopReason) => {
    const entries = delegationFixture();
    entries.at(-1)!.message!.stopReason = stopReason;
    expect(await measure(entries)).toEqual({ status: 'incomplete', reason: 'no-final-handoff' });
  });

  it('withholds an unfinished write, an empty handoff, and a run with no successful evidence', async () => {
    expect(await measure(delegationFixture(), false)).toEqual({ status: 'incomplete', reason: 'partial-record' });
    expect(await measure(delegationFixture('delegate', 'OK', ''))).toMatchObject({ status: 'incomplete' });
    expect(await measure(delegationFixture('delegate', '', 'done'))).toEqual({
      status: 'unavailable',
      reason: 'no-evidence',
    });
  });

  it('does not count a regular resume after the delegate handoff', async () => {
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
    expect(await measure(entries)).toMatchObject({ evidenceTokens: 3, handoffTokens: 1 });
  });

  it('sums explicitly marked follow-ups once each and includes their output overhead', async () => {
    const entries = delegationFixture();
    const follow = delegationFixture('delegate', '', 'done')
      .slice(1)
      .map((e) => ({
        ...e,
        id: `next-${e.id}`,
        parentId: e.parentId ? `next-${e.parentId}` : 'f',
      }));
    expect(await measure([...entries, ...follow])).toMatchObject({
      evidenceTokens: 3,
      handoffTokens: 2,
      savedTokens: 1,
    });
  });

  it.each([false, true])('deduplicates evidence across resumed invocations (compacted: %s)', async (compacted) => {
    const entries: object[] = delegationFixture();
    if (compacted)
      entries.push({
        type: 'compaction',
        id: 'c',
        parentId: 'f',
        timestamp: '',
        firstKeptEntryId: 'f',
        summary: 'compacted',
        tokensBefore: 50,
      });
    const follow = delegationFixture()
      .slice(1)
      .map((e) => ({
        ...e,
        id: `next-${e.id}`,
        parentId: e.parentId ? `next-${e.parentId}` : compacted ? 'c' : 'f',
      }));
    expect(await measure([...entries, ...follow])).toMatchObject({
      evidenceTokens: 3,
      handoffTokens: 2,
      savedTokens: 1,
      savingsRatio: 1 / 3,
    });
    // A changed result is new evidence even when the command is the same.
    follow[3]!.message!.content = [{ type: 'text', text: 'OK' }];
    expect(await measure([...entries, ...follow])).toMatchObject({
      evidenceTokens: 4,
      handoffTokens: 2,
      savedTokens: 2,
    });
  });

  it('does not inherit a legacy blocked delegate identity when a session is forked', async () => {
    const entries = delegationFixture();
    entries.splice(1, 1);
    entries[1]!.parentId = null;
    entries.push({
      type: 'custom',
      id: 'blocked',
      parentId: 'f',
      timestamp: '',
      customType: 'mlx-delegate-handoff',
      data: { reason: 'permission denied', sessionFile: '/sessions/delegate.jsonl' },
    });
    expect(await measure(entries)).toEqual({ status: 'unavailable', reason: 'legacy' });
    const fork = [{ ...entries[0]!, id: 'fork-child', parentSession: '/sessions/delegate.jsonl' }, ...entries.slice(1)];
    expect(await measure(fork, true, 'fork-child')).toBeNull();
    const ownRun = delegationFixture('fork-child')
      .slice(1)
      .map((e) => ({
        ...e,
        id: `own-${e.id}`,
        parentId: e.parentId ? `own-${e.parentId}` : 'blocked',
      }));
    expect(await measure([...fork, ...ownRun], true, 'fork-child')).toMatchObject({
      status: 'complete',
      evidenceTokens: 3,
      handoffTokens: 1,
      savedTokens: 2,
    });
  });

  it('retains pre-compaction evidence and ignores abandoned branches and detached rename metadata', async () => {
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
    expect(await measure(entries)).toMatchObject({ evidenceTokens: 3, handoffTokens: 1 });
    const clean = delegationFixture();
    clean.splice(4, 0, {
      ...clean[4]!,
      id: 'abandoned',
      message: { ...clean[4]!.message!, content: [{ type: 'text', text: 'discard '.repeat(1000) }] },
    });
    const renamed = [...clean, { type: 'session_info', id: 'rename', parentId: null, timestamp: '', name: 'New name' }];
    expect(await measure(renamed)).toMatchObject({ evidenceTokens: 3, handoffTokens: 1 });
  });

  it('never reports savings after a deterministic blocked handoff', async () => {
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
    expect(await measure(entries)).toMatchObject({ status: 'incomplete' });
  });

  it('handles literal special-token text and withholds non-text comparisons', async () => {
    expect(await measure(delegationFixture('delegate', '<|endoftext|>', 'done'))).toMatchObject({ status: 'complete' });
    const entries = delegationFixture();
    entries[4]!.message!.content = [{ type: 'image', data: 'abc', mimeType: 'image/png' }] as never;
    expect(await measure(entries)).toEqual({ status: 'unavailable', reason: 'unsupported-content' });
  });
});

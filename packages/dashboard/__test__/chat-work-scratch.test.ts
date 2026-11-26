/**
 * Work without a folder: the scratch folders under `~/Portato` (made new per conversation, removed only when empty)
 * and the route fields that ask for one (`noFolder`, never beside `cwd`).
 */

import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  realpathSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vite-plus/test';

import { ApiError } from '../src/api/errors.js';
import { createConversationInput, draftPermissionModeInput, workHooksQuery } from '../src/chat/route-input.js';
import {
  isScratch,
  makeScratch,
  realScratchRoot,
  removeIfEmpty,
  scratchRoot,
  stampName,
} from '../src/chat/work-scratch.js';

const AT = new Date(2026, 10, 26, 20, 57, 31).getTime();
const STAMP = '2026-11-26-2057';

let base = '';
afterEach(() => {
  if (base !== '') rmSync(base, { recursive: true, force: true });
  base = '';
});

function home(): string {
  base = mkdtempSync(join(tmpdir(), 'mlx-scratch-'));
  const dir = join(base, 'home');
  mkdirSync(dir);
  return dir;
}

function refusal(run: () => unknown): { code: string; reason?: string } {
  try {
    run();
  } catch (error) {
    if (error instanceof ApiError) return { code: error.code, ...(error.reason ? { reason: error.reason } : {}) };
    throw error;
  }
  throw new Error('expected a refusal');
}

describe('the scratch folders', () => {
  // Mutation: UTC fields in place of local ones → another hour on any machine off UTC.
  it('names a folder by the local minute', () => {
    expect(stampName(AT)).toBe(STAMP);
    expect(stampName(new Date(2027, 0, 2, 3, 4).getTime())).toBe('2027-01-02-0304');
    expect(scratchRoot('/Users/me')).toBe(join('/Users/me', 'Portato'));
  });

  // Mutations: the child made with `recursive` (a taken name is reused); the root or the child not 0700.
  it('makes the root and a new 0700 folder; a second one in the minute gets -2; both stay', async () => {
    const root = scratchRoot(home());
    const first = await makeScratch(root, AT);
    const second = await makeScratch(root, AT);
    const real = realpathSync(root);
    expect([first, second]).toEqual([join(real, STAMP), join(real, `${STAMP}-2`)]);
    for (const path of [root, first!, second!]) expect([path, statSync(path).mode & 0o777]).toEqual([path, 0o700]);
  });

  // Mutation: no end to the names → a tenth folder in one minute.
  it('gives up after -9 and leaves an existing root as it is', async () => {
    const root = scratchRoot(home());
    mkdirSync(root, { mode: 0o755 });
    for (const name of [STAMP, ...[2, 3, 4, 5, 6, 7, 8, 9].map((n) => `${STAMP}-${n}`)]) mkdirSync(join(root, name));
    expect(await makeScratch(root, AT)).toBeNull();
    expect(existsSync(join(root, `${STAMP}-10`))).toBe(false);
    expect(statSync(root).mode & 0o777).toBe(0o755);
  });

  // Mutation: a failed root `mkdir` thrown as it is → a raw fs error reaches the route.
  it('a file where the root goes: no folder', async () => {
    const root = scratchRoot(home());
    writeFileSync(root, 'x');
    expect(await makeScratch(root, AT)).toBeNull();
  });

  // Mutation: the length rule left out → a folder the agent cannot keep conversations for.
  it('a path whose encoded form is over 255 bytes is not made', async () => {
    const deep = join(home(), 'd'.repeat(120), 'e'.repeat(120));
    mkdirSync(deep, { recursive: true });
    const root = scratchRoot(deep);
    expect(await makeScratch(root, AT)).toBeNull();
    expect(existsSync(join(root, STAMP))).toBe(false);
  });

  // Mutation: a text prefix test → a deeper folder or the root itself reads as a No folder folder.
  it('marks a child of the real root only; a root behind a link counts by its real path', async () => {
    const h = home();
    const elsewhere = join(base, 'disk', 'Portato');
    mkdirSync(elsewhere, { recursive: true });
    symlinkSync(elsewhere, scratchRoot(h));
    const made = await makeScratch(scratchRoot(h), AT);
    const real = realScratchRoot(scratchRoot(h));
    expect(real).toBe(realpathSync(elsewhere));
    expect(made).toBe(join(real!, STAMP));
    expect(isScratch(made!, real)).toBe(true);
    expect(isScratch(real!, real)).toBe(false);
    expect(isScratch(join(made!, 'sub'), real)).toBe(false);
    expect(isScratch(`${real!}-other/${STAMP}`, real)).toBe(false);
    expect(isScratch(join(scratchRoot(h), STAMP), real)).toBe(false);
    expect(isScratch(made!, null)).toBe(false);
    expect(realScratchRoot(join(h, 'missing'))).toBeNull();
  });

  // Mutation: a recursive remove → the user's file goes with the folder.
  it('removes an empty folder only', async () => {
    const root = scratchRoot(home());
    const empty = (await makeScratch(root, AT))!;
    const kept = (await makeScratch(root, AT))!;
    writeFileSync(join(kept, 'notes.md'), 'mine');
    expect(await removeIfEmpty(empty)).toBe(true);
    expect(await removeIfEmpty(kept)).toBe(false);
    expect(existsSync(empty)).toBe(false);
    expect(existsSync(join(kept, 'notes.md'))).toBe(true);
    expect(await removeIfEmpty(join(root, 'missing'))).toBe(false);
  });
});

describe('the route fields of Work without a folder', () => {
  // Mutations: `noFolder` beside `cwd` accepted; `false` read as absent; `noFolder` in Chat accepted.
  it('a create takes `noFolder: true` in Work only, and never with `cwd`', () => {
    expect(createConversationInput({ mode: 'work', text: 'hi', noFolder: true })).toEqual({
      mode: 'work',
      text: 'hi',
      noFolder: true,
    });
    for (const body of [
      { mode: 'work', text: 'hi', noFolder: true, cwd: '/Users/me/code' },
      { mode: 'work', text: 'hi', noFolder: false },
      { mode: 'work', text: 'hi', noFolder: 'true' },
      { mode: 'chat', text: 'hi', noFolder: true },
    ]) {
      expect([body, refusal(() => createConversationInput(body)).code]).toEqual([body, 'E_BAD_REQUEST']);
    }
    expect(createConversationInput({ mode: 'work', text: 'hi' })).toEqual({ mode: 'work', text: 'hi' });
  });

  // Mutations: an up with no place accepted; an up with both; a down that names no folder refused.
  it('a draft move up names a folder or no folder, never both; a move down names neither', () => {
    expect(draftPermissionModeInput({ mode: 'auto', up: true, noFolder: true, temp: '/tmp' })).toEqual({
      mode: 'auto',
      up: true,
      cwd: null,
      temp: '/tmp',
    });
    expect(draftPermissionModeInput({ mode: 'bypass', up: true, cwd: '/Users/me/code' })).toMatchObject({
      cwd: '/Users/me/code',
    });
    expect(draftPermissionModeInput({ mode: 'ask', up: false })).toEqual({
      mode: 'ask',
      up: false,
      cwd: null,
      temp: null,
    });
    for (const body of [
      { mode: 'bypass', up: true },
      { mode: 'bypass', up: true, cwd: '/Users/me/code', noFolder: true },
      { mode: 'bypass', up: true, noFolder: false },
      { mode: 'ask', up: false, noFolder: true },
    ]) {
      expect([body, refusal(() => draftPermissionModeInput(body)).code]).toEqual([body, 'E_BAD_REQUEST']);
    }
  });

  // Mutation: `noFolder` read as any value → `?noFolder=0` asks for no folder.
  it('the hooks query names a folder or `noFolder=1`', () => {
    expect(workHooksQuery(new URLSearchParams('noFolder=1'))).toBeNull();
    expect(workHooksQuery(new URLSearchParams('cwd=%2FUsers%2Fme%2Fcode'))).toBe('/Users/me/code');
    expect(refusal(() => workHooksQuery(new URLSearchParams('noFolder=0'))).code).toBe('E_BAD_REQUEST');
    expect(refusal(() => workHooksQuery(new URLSearchParams('noFolder=1&cwd=%2Fx'))).code).toBe('E_BAD_REQUEST');
    expect(refusal(() => workHooksQuery(new URLSearchParams(''))).reason).toBe('cwd-invalid');
  });
});

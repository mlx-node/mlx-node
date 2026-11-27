/**
 * The page files of the PDF tool: a stage takes its pages in order, Send moves it whole to the page folder of its id
 * with a `meta.json` naming the session (and a second Send finds the same folder), the delete of a conversation and
 * the day sweep remove what is no one's, and a page belongs to a session only by its real path.
 */

import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  rmSync,
  statSync,
  symlinkSync,
  utimesSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, beforeEach, describe, expect, it } from 'vite-plus/test';

import { isStageId, pdfToolId } from '../src/chat/attachment-rule.js';
import {
  ownsPage,
  pageFile,
  PdfStore,
  readPdfMeta,
  removePdf,
  removeSessionPdfs,
  STAGE_MAX_AGE_MS,
} from '../src/chat/pdf-store.js';

const page = (n: number): Buffer => Buffer.from([0xff, 0xd8, 0xff, 0xe0, n]);

let root: string;
let stages: string;
let folder: string;

beforeEach(() => {
  root = realpathSync(mkdtempSync(join(tmpdir(), 'mlx-pdf-store-')));
  stages = join(root, 'pdf-stage');
  folder = join(root, 'work-pdf');
});

afterEach(() => {
  rmSync(root, { recursive: true, force: true });
});

/** A stage of `pages` pages, every page sent in order. */
async function staged(store: PdfStore, pages: number): Promise<string> {
  const stage = await store.stagePage(null, 1, pages, page(1));
  if (stage === null) throw new Error('refused');
  for (let n = 2; n <= pages; n++) expect(await store.stagePage(stage, n, pages, page(n))).toBe(stage);
  return stage;
}

describe('a stage', () => {
  // Mutation: a page out of order taken → red.
  it('takes pages in order only, as files of their own, folders 0700 and files 0600', async () => {
    const store = new PdfStore(stages);
    const stage = await store.stagePage(null, 1, 3, page(1));
    expect(stage !== null && isStageId(stage)).toBe(true);
    const id = stage!;
    expect(await store.stagePage(id, 3, 3, page(3))).toBeNull();
    expect(await store.stagePage(id, 1, 3, page(1))).toBeNull();
    expect(await store.stagePage(id, 2, 4, page(2))).toBeNull();
    expect(store.complete(id, 3)).toBe(false);
    expect(await store.stagePage(id, 2, 3, page(2))).toBe(id);
    expect(await store.stagePage(id, 3, 3, page(3))).toBe(id);
    expect(await store.stagePage(id, 4, 3, page(4))).toBeNull();
    expect(store.complete(id, 3)).toBe(true);
    expect(store.complete(id, 4)).toBe(false);
    expect(await store.stagePage(null, 2, 3, page(2))).toBeNull();
    expect(await store.stagePage('a'.repeat(26), 1, 3, page(1))).toBeNull();
    const dir = join(stages, id);
    expect(readdirSync(dir).sort()).toEqual(['page-001.jpg', 'page-002.jpg', 'page-003.jpg']);
    expect(readFileSync(join(dir, 'page-002.jpg'))).toEqual(page(2));
    expect(statSync(stages).mode & 0o777).toBe(0o700);
    expect(statSync(dir).mode & 0o777).toBe(0o700);
    expect(statSync(join(dir, 'page-001.jpg')).mode & 0o777).toBe(0o600);
  });

  it('pages sent together are written one at a time, in the order sent', async () => {
    const store = new PdfStore(stages);
    const stage = (await store.stagePage(null, 1, 4, page(1)))!;
    const answers = await Promise.all([2, 3, 4].map((n) => store.stagePage(stage, n, 4, page(n))));
    expect(answers).toEqual([stage, stage, stage]);
    expect(store.complete(stage, 4)).toBe(true);
  });

  it('a dropped stage is gone with its pages; a stage this run does not hold is nothing to drop', async () => {
    const store = new PdfStore(stages);
    const stage = await staged(store, 2);
    await store.drop(stage);
    expect(existsSync(join(stages, stage))).toBe(false);
    expect(await store.stagePage(stage, 3, 2, page(3))).toBeNull();
    await store.drop('b'.repeat(26));
  });

  it('page files have three digits, as the reader script names them', () => {
    expect([pageFile(1), pageFile(42), pageFile(320)]).toEqual(['page-001.jpg', 'page-042.jpg', 'page-320.jpg']);
  });
});

describe('place', () => {
  // Mutation: a second place of the same stage refused (place not repeatable) → red.
  it('moves a complete stage to the page folder of its id; a second Send of it finds the same folder', async () => {
    const store = new PdfStore(stages);
    const stage = await staged(store, 3);
    const id = await store.place(stage, 3, folder, 'session-a', 'report.pdf');
    expect(id).toBe(pdfToolId(stage, 3));
    expect(id).toBe(`${stage.slice(0, 8)}-3`);
    expect(existsSync(join(stages, stage))).toBe(false);
    expect(readdirSync(join(folder, id!)).sort()).toEqual([
      'meta.json',
      'page-001.jpg',
      'page-002.jpg',
      'page-003.jpg',
    ]);
    expect(JSON.parse(readFileSync(join(folder, id!, 'meta.json'), 'utf8'))).toEqual({
      v: 1,
      session: 'session-a',
      name: 'report.pdf',
      pages: 3,
      bytes: 15,
      stage,
    });
    expect(statSync(folder).mode & 0o777).toBe(0o700);
    expect(statSync(join(folder, id!, 'meta.json')).mode & 0o777).toBe(0o600);
    expect(await store.place(stage, 3, folder, 'session-a', 'report.pdf')).toBe(id);
    expect(await new PdfStore(stages).place(stage, 3, folder, 'session-a', 'report.pdf')).toBe(id);
    expect(await readPdfMeta(folder, id!)).toMatchObject({ session: 'session-a', pages: 3 });
  });

  it('refuses a stage that is not complete, another count, and a stage placed for another session', async () => {
    const store = new PdfStore(stages);
    const stage = (await store.stagePage(null, 1, 3, page(1)))!;
    expect(await store.place(stage, 3, folder, 'session-a', 'a.pdf')).toBeNull();
    await store.stagePage(stage, 2, 3, page(2));
    await store.stagePage(stage, 3, 3, page(3));
    expect(await store.place(stage, 2, folder, 'session-a', 'a.pdf')).toBeNull();
    const id = await store.place(stage, 3, folder, 'session-a', 'a.pdf');
    expect(id).not.toBeNull();
    expect(await store.place(stage, 3, folder, 'session-b', 'a.pdf')).toBeNull();
    expect(await store.place('c'.repeat(26), 3, folder, 'session-a', 'a.pdf')).toBeNull();
  });

  it('a folder of that id whose meta.json names another stage is never taken over', async () => {
    const store = new PdfStore(stages);
    const stage = await staged(store, 1);
    const id = pdfToolId(stage, 1);
    mkdirSync(join(folder, id), { recursive: true });
    writeFileSync(
      join(folder, id, 'meta.json'),
      JSON.stringify({ v: 1, session: 'session-a', name: 'x', pages: 1, bytes: 1, stage: 'other' }),
    );
    expect(await store.place(stage, 1, folder, 'session-a', 'a.pdf')).toBeNull();
    expect(existsSync(join(stages, stage))).toBe(true);
  });
});

describe('what goes', () => {
  it("a conversation's delete removes its page folders and keeps the others", async () => {
    const store = new PdfStore(stages);
    const mine = await store.place(await staged(store, 1), 1, folder, 'session-a', 'a.pdf');
    const also = await store.place(await staged(store, 2), 2, folder, 'session-a', 'b.pdf');
    const theirs = await store.place(await staged(store, 1), 1, folder, 'session-b', 'c.pdf');
    mkdirSync(join(folder, 'notes'));
    await removeSessionPdfs(folder, 'session-a');
    expect(readdirSync(folder).sort()).toEqual([theirs!, 'notes'].sort());
    expect([mine, also].every((id) => id !== null && !existsSync(join(folder, id)))).toBe(true);
    await removeSessionPdfs(join(root, 'none'), 'session-a');
  });

  // Mutation: one folder removed whatever session its meta.json names → red.
  it("a queued row's removal takes its page folder only when the folder is that conversation's", async () => {
    const store = new PdfStore(stages);
    const id = (await store.place(await staged(store, 2), 2, folder, 'session-a', 'a.pdf'))!;
    await removePdf(folder, id, 'session-b');
    expect(existsSync(join(folder, id))).toBe(true);
    await removePdf(folder, id, 'session-a');
    expect(existsSync(join(folder, id))).toBe(false);
    await removePdf(folder, id, 'session-a');
  });

  // Mutation: the sweep removes a stage this run holds → red.
  it('the sweep at host start removes stages older than a day and keeps newer ones and the ones it holds', async () => {
    const now = Date.now();
    const old = new Date(now - STAGE_MAX_AGE_MS - 60_000);
    const store = new PdfStore(stages, () => now);
    const held = await staged(store, 1);
    const left = join(stages, 'd'.repeat(26));
    const fresh = join(stages, 'e'.repeat(26));
    for (const dir of [left, fresh]) {
      mkdirSync(dir, { recursive: true });
      writeFileSync(join(dir, 'page-001.jpg'), page(1));
    }
    utimesSync(left, old, old);
    utimesSync(join(stages, held), old, old);
    await store.sweep();
    expect(readdirSync(stages).sort()).toEqual([held, 'e'.repeat(26)].sort());
    await new PdfStore(join(root, 'none')).sweep();
  });
});

describe('ownsPage', () => {
  // Mutation: the real-path check dropped (the path as written) → red.
  it('a page belongs to the session by its real path only: links are followed, `..` is resolved', async () => {
    const store = new PdfStore(stages);
    const mine = (await store.place(await staged(store, 2), 2, folder, 'session-a', 'a.pdf'))!;
    const theirs = (await store.place(await staged(store, 2), 2, folder, 'session-b', 'b.pdf'))!;
    const outside = join(root, 'outside');
    mkdirSync(outside);
    writeFileSync(join(outside, 'secret.jpg'), page(9));
    expect(await ownsPage(folder, join(folder, mine, 'page-001.jpg'), 'session-a')).toBe(true);
    expect(await ownsPage(folder, join(folder, mine, 'page-002.jpg'), 'session-a')).toBe(true);
    expect(await ownsPage(folder, join(folder, mine, 'page-001.jpg'), 'session-b')).toBe(false);
    expect(await ownsPage(folder, join(folder, theirs, 'page-001.jpg'), 'session-a')).toBe(false);
    expect(await ownsPage(folder, join(folder, mine, '..', theirs, 'page-001.jpg'), 'session-a')).toBe(false);
    expect(await ownsPage(folder, join(folder, theirs, '..', mine, 'page-001.jpg'), 'session-a')).toBe(true);
    expect(await ownsPage(folder, join(folder, mine, 'page-003.jpg'), 'session-a')).toBe(false);
    writeFileSync(join(folder, mine, 'page-999.jpg'), page(9));
    writeFileSync(join(folder, mine, 'page-000.jpg'), page(9));
    writeFileSync(join(folder, mine, 'page-01.jpg'), page(9));
    for (const name of ['page-999.jpg', 'page-000.jpg', 'page-01.jpg', 'meta.json']) {
      expect([name, await ownsPage(folder, join(folder, mine, name), 'session-a')]).toEqual([name, false]);
    }
    expect(await ownsPage(folder, join(outside, 'secret.jpg'), 'session-a')).toBe(false);
    rmSync(join(folder, mine, 'page-002.jpg'));
    symlinkSync(join(outside, 'secret.jpg'), join(folder, mine, 'page-002.jpg'));
    expect(await ownsPage(folder, join(folder, mine, 'page-002.jpg'), 'session-a')).toBe(false);
    symlinkSync(join(folder, mine, 'page-001.jpg'), join(outside, 'link.jpg'));
    expect(await ownsPage(folder, join(outside, 'link.jpg'), 'session-a')).toBe(true);
    const linked = join(root, 'linked');
    symlinkSync(folder, linked);
    expect(await ownsPage(linked, join(linked, mine, 'page-001.jpg'), 'session-a')).toBe(true);
    expect(await ownsPage(folder, join(folder, mine, 'gone.jpg'), 'session-a')).toBe(false);
  });
});

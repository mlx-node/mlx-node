/**
 * The page files of the PDFs the PDF tool reads. While the page draws a long PDF it sends each page here, in order,
 * into a stage (`pdf-stage/<stage>/page-001.jpg …`). On Send the stage moves whole to the mode's page folder as
 * `<id>/`, with a `meta.json` that names the session; the id is the stage's first letters and the page count, so a
 * second Send of the same stage finds the same folder. Folders 0700, files 0600.
 */

import { randomBytes } from 'node:crypto';
import { chmod, lstat, mkdir, readdir, readFile, realpath, rename, rm, writeFile } from 'node:fs/promises';
import { join, sep } from 'node:path';

import { PDF_STAGE_CHARS, pdfIdPages, pdfToolId } from './attachment-rule.js';

/** A stage older than this at host start is removed: its page is gone. */
export const STAGE_MAX_AGE_MS = 24 * 60 * 60 * 1000;
const META_FILE = 'meta.json';
const META_MAX_BYTES = 4_096;
const BASE32 = 'abcdefghijklmnopqrstuvwxyz234567';

export interface PdfMeta {
  v: 1;
  /** The session whose message carried the PDF. */
  session: string;
  name: string;
  pages: number;
  bytes: number;
  stage: string;
}

interface Stage {
  pages: number;
  /** The page the stage takes next. */
  next: number;
  bytes: number;
  chain: Promise<unknown>;
}

/** `page-007.jpg`: page numbers have three digits, as the reader's script writes them. */
export function pageFile(page: number): string {
  return `page-${String(page).padStart(3, '0')}.jpg`;
}

function newStageId(): string {
  const bytes = randomBytes(PDF_STAGE_CHARS);
  let id = '';
  for (const byte of bytes) id += BASE32.charAt(byte % BASE32.length);
  return id;
}

function missing(error: unknown): boolean {
  const code = (error as NodeJS.ErrnoException).code;
  return code === 'ENOENT' || code === 'ENOTDIR';
}

async function ensureFolder(path: string): Promise<void> {
  await mkdir(path, { recursive: true, mode: 0o700 });
  await chmod(path, 0o700);
}

function readMeta(text: string, id: string): PdfMeta | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return null;
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return null;
  const m = parsed as Record<string, unknown>;
  if (m.v !== 1 || typeof m.session !== 'string' || m.session === '' || typeof m.name !== 'string') return null;
  if (m.pages !== pdfIdPages(id) || typeof m.bytes !== 'number' || typeof m.stage !== 'string') return null;
  return { v: 1, session: m.session, name: m.name, pages: m.pages as number, bytes: m.bytes, stage: m.stage };
}

/** The `meta.json` of the page folder `id` under `folder`; null when `id` is not an id or the file is not one. */
export async function readPdfMeta(folder: string, id: string): Promise<PdfMeta | null> {
  if (pdfIdPages(id) === null) return null;
  const path = join(folder, id, META_FILE);
  try {
    const info = await lstat(path);
    if (!info.isFile() || info.size > META_MAX_BYTES) return null;
    return readMeta(await readFile(path, 'utf8'), id);
  } catch {
    return null;
  }
}

/** Removes every page folder under `folder` whose `meta.json` names `session`. */
export async function removeSessionPdfs(folder: string, session: string): Promise<void> {
  let names: string[];
  try {
    names = await readdir(folder);
  } catch (error) {
    if (missing(error)) return;
    throw error;
  }
  for (const name of names) {
    if ((await readPdfMeta(folder, name))?.session === session) await rm(join(folder, name), { recursive: true });
  }
}

/**
 * Whether `path` is, on disk with its links followed, a page file `page-NNN.jpg` of a page folder under `folder` whose
 * `meta.json` names `session`, with NNN one of its pages.
 */
export async function ownsPage(folder: string, path: string, session: string): Promise<boolean> {
  let real: string;
  let top: string;
  try {
    real = await realpath(path);
    top = await realpath(folder);
  } catch {
    return false;
  }
  if (!real.startsWith(top + sep)) return false;
  const parts = real.slice(top.length + 1).split(sep);
  if (parts.length !== 2) return false;
  const [id, file] = parts as [string, string];
  const pages = pdfIdPages(id);
  if (pages === null || !file.startsWith('page-') || !file.endsWith('.jpg')) return false;
  const digits = file.slice('page-'.length, file.length - '.jpg'.length);
  let page = 0;
  for (let i = 0; i < digits.length; i++) {
    const code = digits.charCodeAt(i);
    if (code < 0x30 || code > 0x39) return false;
    page = page * 10 + (code - 0x30);
  }
  if (digits.length !== 3 || page < 1 || page > pages) return false;
  if ((await readPdfMeta(top, id))?.session !== session) return false;
  try {
    return (await lstat(real)).isFile();
  } catch {
    return false;
  }
}

/** The stages this host run holds; a stage of an earlier run is never taken again. */
export class PdfStore {
  private readonly root: string;
  private readonly now: () => number;
  private readonly stages = new Map<string, Stage>();

  /** `root`: the stage folder (`pdf-stage/`). */
  constructor(root: string, now: () => number = Date.now) {
    this.root = root;
    this.now = now;
  }

  private serial<T>(stage: Stage, run: () => Promise<T>): Promise<T> {
    const next = stage.chain.then(run, run);
    stage.chain = next.catch(() => {});
    return next;
  }

  /**
   * Writes page `page` of `pages`: page 1 with no stage opens a new one, any other must be the stage's next page of
   * the same count. The stage id; null when the page is refused.
   */
  async stagePage(stageId: string | null, page: number, pages: number, data: Buffer): Promise<string | null> {
    if (stageId === null) {
      if (page !== 1) return null;
      const id = newStageId();
      const stage: Stage = { pages, next: 1, bytes: 0, chain: Promise.resolve() };
      this.stages.set(id, stage);
      const written = await this.serial(stage, async () => {
        await ensureFolder(this.root);
        await mkdir(join(this.root, id), { mode: 0o700 });
        return this.write(id, stage, page, data);
      }).catch((error: unknown) => {
        this.stages.delete(id);
        throw error;
      });
      return written ? id : null;
    }
    const stage = this.stages.get(stageId);
    if (stage === undefined || stage.pages !== pages) return null;
    return (await this.serial(stage, () => this.write(stageId, stage, page, data))) ? stageId : null;
  }

  private async write(id: string, stage: Stage, page: number, data: Buffer): Promise<boolean> {
    if (this.stages.get(id) !== stage || page !== stage.next || page > stage.pages) return false;
    await writeFile(join(this.root, id, pageFile(page)), data, { mode: 0o600, flag: 'wx' });
    stage.next += 1;
    stage.bytes += data.length;
    return true;
  }

  /** The stage is held and has all `pages` of its pages. */
  complete(stageId: string, pages: number): boolean {
    const stage = this.stages.get(stageId);
    return stage !== undefined && stage.pages === pages && stage.next === pages + 1;
  }

  /** Removes a stage that was not placed; nothing for one this run does not hold. */
  async drop(stageId: string): Promise<void> {
    const stage = this.stages.get(stageId);
    if (stage === undefined) return;
    await this.serial(stage, async () => {
      if (this.stages.get(stageId) !== stage) return;
      this.stages.delete(stageId);
      await rm(join(this.root, stageId), { recursive: true, force: true });
    });
  }

  /**
   * Moves a complete stage to `folder` as the page folder of its id, for `session`; a stage already placed for that
   * session gives the same id. Null when the stage is not complete or its folder is another session's.
   */
  async place(stageId: string, pages: number, folder: string, session: string, name: string): Promise<string | null> {
    const id = pdfToolId(stageId, pages);
    const placed = async (): Promise<string | null | undefined> => {
      const meta = await readPdfMeta(folder, id);
      if (meta !== null) return meta.session === session && meta.stage === stageId ? id : null;
      return undefined;
    };
    const stage = this.stages.get(stageId);
    if (stage === undefined) return (await placed()) ?? null;
    return this.serial(stage, async () => {
      const before = await placed();
      if (before !== undefined) return before;
      if (this.stages.get(stageId) !== stage || !this.complete(stageId, pages)) return null;
      const from = join(this.root, stageId);
      const meta: PdfMeta = { v: 1, session, name, pages, bytes: stage.bytes, stage: stageId };
      await writeFile(join(from, META_FILE), `${JSON.stringify(meta)}\n`, { mode: 0o600, flag: 'w' });
      await ensureFolder(folder);
      try {
        await rename(from, join(folder, id));
      } catch {
        return (await placed()) ?? null;
      }
      this.stages.delete(stageId);
      return id;
    });
  }

  /** Removes every stage folder older than {@link STAGE_MAX_AGE_MS}. */
  async sweep(): Promise<void> {
    let names: string[];
    try {
      names = await readdir(this.root);
    } catch (error) {
      if (missing(error)) return;
      throw error;
    }
    const before = this.now() - STAGE_MAX_AGE_MS;
    for (const name of names) {
      if (this.stages.has(name)) continue;
      const path = join(this.root, name);
      try {
        if ((await lstat(path)).mtimeMs < before) await rm(path, { recursive: true, force: true });
      } catch (error) {
        if (!missing(error)) throw error;
      }
    }
  }
}

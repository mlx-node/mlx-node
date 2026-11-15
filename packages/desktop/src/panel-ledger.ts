/**
 * MAIN's record of why each CONTROL PANEL life ended: `<mlx-node home>/ledger/main-<startMs>-<pid>.1.jsonl`.
 *
 * One writer per file, one synchronous append per row, no fsync. A row holds numbers, booleans, list members, UUIDs,
 * row ids and route patterns only: {@link encodeRow} is the one function that makes a line, and it drops any field
 * that is not in its kind's table or fails its check. The next panel reads the ends back with {@link readAppEnds}.
 *
 * Node builtins only: MAIN and the panel both import it.
 */

import { closeSync, mkdirSync, openSync, writeSync, constants as fsConstants } from 'node:fs';
import { open, readdir, stat, unlink } from 'node:fs/promises';
import { join } from 'node:path';

import { CONTROL_PANEL_BROKER_KILL_GRACE_MS } from './control-panel/shutdown-timings.js';

export const LEDGER_DIR_ENV = 'MLX_PANEL_LEDGER_DIR';
const RETENTION_MS = 30 * 24 * 60 * 60 * 1000;
const READ_MAX_ENDS = 32;
const READ_MAX_BYTES = 1024 * 1024;

export type StopCause =
  | 'main.panel_no_pong'
  | 'main.kill_escalation'
  | 'main.port_refused'
  | 'panel.worker_down'
  | 'panel.exit_unasked'
  | 'os.signal_term'
  | 'user.app_quit';

const CAUSES: readonly StopCause[] = [
  'main.panel_no_pong',
  'main.kill_escalation',
  'main.port_refused',
  'panel.worker_down',
  'panel.exit_unasked',
  'os.signal_term',
  'user.app_quit',
];

export type LedgerRow =
  | { k: 'proc'; role: 'main'; host: 'desktop'; pid: number; ppid: number; ver: string; seq: 1 }
  | { k: 'child.spawn'; role: 'panel'; gen: number; pid?: number }
  | { k: 'slow'; op: 'loop'; target: 'panel'; state: 'open' | 'done'; ageMs: number }
  | {
      k: 'slow';
      op: 'rpc';
      state: 'open';
      route?: string;
      thread?: 'main' | 'worker';
      ageMs: number;
      wq: number;
      wbusyMs: number;
      wroute?: string;
    }
  | {
      k: 'intent';
      what: 'worker.restart';
      cause: 'panel.worker_down';
      reason: 'error' | 'exit' | 'no-ack' | 'gave-up';
      route?: string;
      ageMs?: number;
      attempt: number;
      open: number;
      target?: string;
    }
  | {
      k: 'intent';
      what: 'panel.stop';
      cause: 'main.panel_no_pong';
      stage: 'start' | 'idle' | 'turn';
      ageMs: number;
      open: number;
      convs: string[];
      route?: string;
      thread?: 'main' | 'worker';
      target?: string;
    }
  | {
      k: 'intent';
      what: 'panel.stop';
      cause: 'main.port_refused' | 'user.app_quit';
      open: number;
      convs: string[];
      target?: string;
    }
  | { k: 'intent'; what: 'panel.kill'; cause: 'main.kill_escalation'; by?: string; target?: string }
  | {
      k: 'child.exit';
      role: 'panel';
      child?: string;
      pid?: number;
      code?: number;
      asked: boolean;
      cause: StopCause;
      by?: string;
      sinceIntentMs?: number;
      uptimeMs: number;
      crashes: number;
      open: number;
      convs: string[];
      cancelled?: string[];
    }
  | { k: 'note'; what: 'refused' | 'write_failed'; count: number; from?: number; to?: number };

type Kind = LedgerRow['k'];

type FieldType =
  | 'int'
  | 'num'
  | 'bool'
  | 'uuids'
  | 'route'
  | 'rowid'
  | 'ver'
  | { readonly enum: readonly (string | number)[] };

const THREADS = ['main', 'worker'] as const;

const FIELDS: Record<Kind, Record<string, FieldType>> = {
  proc: { role: { enum: ['main'] }, host: { enum: ['desktop'] }, pid: 'int', ppid: 'int', ver: 'ver', seq: 'int' },
  'child.spawn': { role: { enum: ['panel'] }, gen: 'int', pid: 'int' },
  slow: {
    op: { enum: ['loop', 'rpc'] },
    target: { enum: ['panel'] },
    state: { enum: ['open', 'done'] },
    ageMs: 'num',
    route: 'route',
    thread: { enum: THREADS },
    wq: 'num',
    wbusyMs: 'num',
    wroute: 'route',
  },
  intent: {
    what: { enum: ['panel.stop', 'panel.kill', 'worker.restart'] },
    cause: { enum: CAUSES },
    stage: { enum: ['start', 'idle', 'turn'] },
    reason: { enum: ['error', 'exit', 'no-ack', 'gave-up'] },
    route: 'route',
    thread: { enum: THREADS },
    ageMs: 'num',
    attempt: 'int',
    open: 'int',
    convs: 'uuids',
    by: 'rowid',
    target: 'rowid',
  },
  'child.exit': {
    role: { enum: ['panel'] },
    child: 'rowid',
    pid: 'int',
    code: 'int',
    asked: 'bool',
    cause: { enum: CAUSES },
    by: 'rowid',
    sinceIntentMs: 'num',
    uptimeMs: 'num',
    crashes: 'int',
    open: 'int',
    convs: 'uuids',
    cancelled: 'uuids',
  },
  note: { what: { enum: ['refused', 'write_failed'] }, count: 'int', from: 'num', to: 'num' },
};

const KINDS = Object.keys(FIELDS) as Kind[];

function isHex(c: string): boolean {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

function isDigits(s: string): boolean {
  if (s.length === 0 || s.length > 20) return false;
  for (const c of s) if (c < '0' || c > '9') return false;
  return true;
}

/** 8-4-4-4-12 hex. A prompt, a path, a URL or an error text cannot pass. */
export function isUuid(value: unknown): value is string {
  if (typeof value !== 'string' || value.length !== 36) return false;
  for (let i = 0; i < 36; i++) {
    const c = value[i]!;
    if (i === 8 || i === 13 || i === 18 || i === 23) {
      if (c !== '-') return false;
    } else if (!isHex(c)) return false;
  }
  return true;
}

const VERBS = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'];

/**
 * A route pattern from the route table: `GET /api/sessions/:id`. A verb from the list, then 1–8 segments of `a-z`
 * and `-`, each with an optional leading `:`. A digit cannot pass, so neither can a UUID or a model name.
 */
export function isRoutePattern(value: unknown): value is string {
  if (typeof value !== 'string' || value.length > 128) return false;
  const space = value.indexOf(' ');
  if (space < 0 || !VERBS.includes(value.slice(0, space))) return false;
  const path = value.slice(space + 1);
  if (!path.startsWith('/')) return false;
  const segments = path.slice(1).split('/');
  if (segments.length < 1 || segments.length > 8) return false;
  for (const segment of segments) {
    const body = segment.startsWith(':') ? segment.slice(1) : segment;
    if (body.length === 0) return false;
    for (const c of body) if ((c < 'a' || c > 'z') && c !== '-') return false;
  }
  return true;
}

/** `<pid>.<startMs>`. */
export function isProcId(value: unknown): value is string {
  if (typeof value !== 'string') return false;
  const dot = value.indexOf('.');
  return dot > 0 && isDigits(value.slice(0, dot)) && isDigits(value.slice(dot + 1));
}

/** `<ProcId>#<n>`. */
export function isRowId(value: unknown): value is string {
  if (typeof value !== 'string') return false;
  const hash = value.indexOf('#');
  return hash > 0 && isProcId(value.slice(0, hash)) && isDigits(value.slice(hash + 1));
}

function isVer(value: unknown): value is string {
  if (typeof value !== 'string' || value.length < 1 || value.length > 32) return false;
  for (const c of value)
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || c === '.' || c === '-')) return false;
  return true;
}

function check(type: FieldType, value: unknown): unknown {
  if (typeof type === 'object') return type.enum.includes(value as string | number) ? value : undefined;
  switch (type) {
    case 'int':
      return typeof value === 'number' && Number.isSafeInteger(value) ? value : undefined;
    case 'num':
      return typeof value === 'number' && Number.isFinite(value) ? Math.round(value) : undefined;
    case 'bool':
      return typeof value === 'boolean' ? value : undefined;
    case 'uuids': {
      if (!Array.isArray(value) || value.length > 8) return undefined;
      return value.every(isUuid) ? [...value] : undefined;
    }
    case 'route':
      return isRoutePattern(value) ? value : undefined;
    case 'rowid':
      return isRowId(value) ? value : undefined;
    case 'ver':
      return isVer(value) ? value : undefined;
  }
}

export interface Envelope {
  t: number;
  m: number;
  p: string;
  n: number;
}

/** The one line maker. Unknown keys and values that fail their check are dropped and counted; null = no such kind. */
export function encodeRow(envelope: Envelope, row: Record<string, unknown>): { line: string; dropped: number } | null {
  const kind = KINDS.find((k) => k === row.k);
  if (kind === undefined) return null;
  const table = FIELDS[kind];
  const out: Record<string, unknown> = {
    v: 1,
    t: Math.round(envelope.t),
    m: Math.round(envelope.m),
    p: envelope.p,
    n: envelope.n,
    k: kind,
  };
  let dropped = 0;
  for (const [key, value] of Object.entries(row)) {
    if (key === 'k' || value === undefined) continue;
    const type = table[key];
    const checked = type === undefined ? undefined : check(type, value);
    if (checked === undefined) dropped += 1;
    else out[key] = checked;
  }
  return { line: `${JSON.stringify(out)}\n`, dropped };
}

/** A parsed row: the envelope plus the fields of its kind that pass their checks. Null for anything else. */
export type ParsedRow = Envelope & { k: Kind } & Record<string, unknown>;

export function parseRow(line: string): ParsedRow | null {
  let value: unknown;
  try {
    value = JSON.parse(line);
  } catch {
    return null;
  }
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return null;
  const raw = value as Record<string, unknown>;
  const kind = KINDS.find((k) => k === raw.k);
  if (raw.v !== 1 || kind === undefined || !isProcId(raw.p)) return null;
  if (typeof raw.t !== 'number' || typeof raw.m !== 'number' || typeof raw.n !== 'number') return null;
  const row: ParsedRow = { t: raw.t, m: raw.m, p: raw.p, n: raw.n, k: kind };
  for (const [key, type] of Object.entries(FIELDS[kind])) {
    const checked = raw[key] === undefined ? undefined : check(type, raw[key]);
    if (checked !== undefined) row[key] = checked;
  }
  return row;
}

export interface LedgerIo {
  mkdir(dir: string): void;
  /** Open for append; the file is created 0600 and a symlink is refused. */
  open(path: string): number;
  write(fd: number, line: string): void;
  close(fd: number): void;
}

const nodeIo: LedgerIo = {
  mkdir: (dir) => {
    mkdirSync(dir, { recursive: true, mode: 0o700 });
  },
  open: (path) =>
    openSync(path, fsConstants.O_WRONLY | fsConstants.O_APPEND | fsConstants.O_CREAT | fsConstants.O_NOFOLLOW, 0o600),
  write: (fd, line) => {
    writeSync(fd, line);
  },
  close: (fd) => {
    closeSync(fd);
  },
};

export interface PanelLedgerOptions {
  dir: string;
  pid: number;
  ppid: number;
  ver: string;
  /** This process's start, epoch ms. */
  startMs: number;
  now: () => number;
  /** Monotonic ms since this process started. */
  mono: () => number;
  io?: LedgerIo;
  /** One line to stderr, the first time a write fails. */
  log?: (line: string) => void;
}

export interface PanelLedger {
  /** Write one row. Returns its row id whether or not the write landed; a failed write never throws. */
  record(row: LedgerRow): string;
  /** Writes that failed so far. */
  failures(): number;
  close(): void;
}

export function ledgerFileName(startMs: number, pid: number): string {
  return `main-${String(Math.round(startMs))}-${String(pid)}.1.jsonl`;
}

export function createPanelLedger(options: PanelLedgerOptions): PanelLedger {
  const io = options.io ?? nodeIo;
  const p = `${String(options.pid)}.${String(Math.round(options.startMs))}`;
  let n = 0;
  let fd: number | null = null;
  let failed = 0;
  let failedFrom: number | null = null;
  let failedTo = 0;
  let logged = false;

  const fail = (error: unknown): void => {
    failed += 1;
    const at = options.now();
    failedFrom ??= at;
    failedTo = at;
    if (logged) return;
    logged = true;
    const code = (error as { code?: unknown } | null)?.code;
    options.log?.(`[mlx] panel ledger write failed (${typeof code === 'string' ? code : 'error'})`);
  };

  try {
    io.mkdir(options.dir);
    fd = io.open(join(options.dir, ledgerFileName(options.startMs, options.pid)));
  } catch (error) {
    fd = null;
    fail(error);
  }

  const put = (row: Record<string, unknown>): string => {
    n += 1;
    const id = `${p}#${String(n)}`;
    const encoded = encodeRow({ t: options.now(), m: options.mono(), p, n }, row);
    if (encoded === null || fd === null) {
      if (fd === null) failed += 1;
      return id;
    }
    try {
      io.write(fd, encoded.line);
    } catch (error) {
      fail(error);
      return id;
    }
    if (encoded.dropped > 0) put({ k: 'note', what: 'refused', count: encoded.dropped });
    return id;
  };

  const record = (row: LedgerRow): string => {
    if (failedFrom !== null && fd !== null) {
      const from = failedFrom;
      failedFrom = null;
      put({ k: 'note', what: 'write_failed', count: failed, from, to: failedTo });
    }
    return put(row as unknown as Record<string, unknown>);
  };

  record({ k: 'proc', role: 'main', host: 'desktop', pid: options.pid, ppid: options.ppid, ver: options.ver, seq: 1 });

  return {
    record,
    failures: () => failed,
    close(): void {
      if (fd === null) return;
      try {
        io.close(fd);
      } catch {
        // Closing at quit: nothing is left to record it.
      }
      fd = null;
    },
  };
}

/** Delete ledger files whose last write is over 30 days old, except `keep`. Never throws. */
export async function sweepLedger(dir: string, now: number, keep: string): Promise<number> {
  let names: string[];
  try {
    names = await readdir(dir);
  } catch {
    return 0;
  }
  let removed = 0;
  for (const name of names) {
    if (name === keep || !name.startsWith('main-') || !name.endsWith('.jsonl')) continue;
    try {
      const info = await stat(join(dir, name));
      if (now - info.mtimeMs <= RETENTION_MS) continue;
      await unlink(join(dir, name));
      removed += 1;
    } catch {
      // Gone already, or not ours to delete.
    }
  }
  return removed;
}

/** How a panel life ended, as the next panel shows it. Kept equal to `AppEnd` in `@mlx-node/dashboard`. */
export interface AppEnd {
  trigger: 'no-ping' | 'crash' | 'signal' | 'user';
  /** When the stop was decided (the intent row); for an exit nobody asked for, the exit less the kill grace. */
  decidedAt: number;
  exitAt: number;
  /** For `no-ping`: how long the panel had been silent. */
  silentMs: number | null;
  code: number | null;
  /** The turns MAIN's last pong listed. */
  convs: string[];
  /** The turns the dying panel said it cancelled; null when that note never arrived. */
  cancelled: string[] | null;
}

const TRIGGERS: Partial<Record<StopCause, AppEnd['trigger']>> = {
  'main.panel_no_pong': 'no-ping',
  'panel.exit_unasked': 'crash',
  'os.signal_term': 'signal',
  'user.app_quit': 'user',
};

/** `main-<startMs>-<pid>.1.jsonl` → startMs, or null for any other name. */
function startOf(name: string): number | null {
  if (!name.startsWith('main-') || !name.endsWith('.jsonl')) return null;
  const rest = name.slice('main-'.length);
  const dash = rest.indexOf('-');
  if (dash <= 0 || !isDigits(rest.slice(0, dash))) return null;
  return Number(rest.slice(0, dash));
}

export interface ReadIo {
  readdir(dir: string): Promise<string[]>;
  /** The last `maxBytes` of a file; a line cut at the start does not parse and is skipped like a torn one. */
  readTail(path: string, maxBytes: number): Promise<string>;
}

const nodeReadIo: ReadIo = {
  readdir: (dir) => readdir(dir),
  readTail: async (path, maxBytes) => {
    const handle = await open(path, 'r');
    try {
      const { size } = await handle.stat();
      const length = Math.min(size, Math.max(0, maxBytes));
      const buffer = Buffer.alloc(length);
      const { bytesRead } = await handle.read(buffer, 0, length, size - length);
      return buffer.toString('utf8', 0, bytesRead);
    } finally {
      await handle.close();
    }
  },
};

/** The ends of one file's panel lives, oldest first. A line that does not parse (a torn last line) is skipped. */
function endsOf(text: string): AppEnd[] {
  const intents = new Map<string, ParsedRow>();
  const ends: AppEnd[] = [];
  for (const line of text.split('\n')) {
    if (line === '') continue;
    const row = parseRow(line);
    if (row === null) continue;
    if (row.k === 'intent') {
      intents.set(`${row.p}#${String(row.n)}`, row);
      continue;
    }
    if (row.k !== 'child.exit' || row.role !== 'panel') continue;
    const trigger = TRIGGERS[row.cause as StopCause];
    if (trigger === undefined) continue;
    const intent = typeof row.by === 'string' ? intents.get(row.by) : undefined;
    const exitAt = row.t;
    ends.push({
      trigger,
      decidedAt: intent?.t ?? exitAt - CONTROL_PANEL_BROKER_KILL_GRACE_MS,
      exitAt,
      silentMs: trigger === 'no-ping' && typeof intent?.ageMs === 'number' ? intent.ageMs : null,
      code: typeof row.code === 'number' ? row.code : null,
      convs: Array.isArray(row.convs) ? (row.convs as string[]) : [],
      cancelled: Array.isArray(row.cancelled) ? (row.cancelled as string[]) : null,
    });
  }
  return ends;
}

/** The panel ends in `dir`, newest first; at most 32, from the last 1 MiB of the newest files. Never throws. */
export async function readAppEnds(dir: string | undefined, io: ReadIo = nodeReadIo): Promise<AppEnd[]> {
  if (dir === undefined || dir === '') return [];
  let names: string[];
  try {
    names = await io.readdir(dir);
  } catch {
    return [];
  }
  const files = names
    .map((name) => ({ name, start: startOf(name) }))
    .filter((f): f is { name: string; start: number } => f.start !== null)
    .sort((a, b) => b.start - a.start);
  const out: AppEnd[] = [];
  let bytes = 0;
  for (const file of files) {
    if (out.length >= READ_MAX_ENDS || bytes >= READ_MAX_BYTES) break;
    let text: string;
    try {
      text = await io.readTail(join(dir, file.name), READ_MAX_BYTES - bytes);
    } catch {
      continue;
    }
    bytes += text.length;
    const ends = endsOf(text);
    for (let i = ends.length - 1; i >= 0 && out.length < READ_MAX_ENDS; i--) out.push(ends[i]!);
  }
  return out;
}

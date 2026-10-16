/// <reference types="node" />

import { spawn } from 'node:child_process';
import { appendFileSync, closeSync, existsSync, openSync, statSync } from 'node:fs';
import { mkdir, readdir, rename } from 'node:fs/promises';
import { basename, join } from 'node:path';

import { ROOT, sh, utcStamp, writeJson } from '../common.js';

const MEMORY_SHARE = 0.45;
const MIN_AVAILABLE_PERCENT = 20;
const LOG_LIMIT_BYTES = 32 * 1024 * 1024;
const MONITOR_MS = 5000;
const NOTABLE = /(?:^|\/)(?:cargo|rustc|clang|llama-server|oxnode)$/;

export interface JobRecord {
  label: string;
  command: string;
  args: string[];
  code?: number | null;
  signal?: string | null;
  error?: string;
  reason: string | null;
  maxRssBytes: number;
  elapsedMs: number;
  finished: string;
  hostBefore?: unknown;
}

export const jobPassed = (record: JobRecord) => record.code === 0 && record.reason === null;

// A rerun of a failed job must not append to the failed attempt's resource or
// tuning logs, so its files move to attempts/ first.
async function archiveAttempt(dir: string, label: string, extraFiles: string[]): Promise<void> {
  const paths = [
    ...(await readdir(dir)).filter((name) => name.startsWith(`${label}.`)).map((name) => join(dir, name)),
    ...extraFiles.filter((path) => existsSync(path)),
  ];
  if (!paths.length) return;
  const target = join(dir, 'attempts', `${label}-${utcStamp()}`);
  await mkdir(target, { recursive: true });
  for (const path of paths) await rename(path, join(target, basename(path)));
}

// Runs one job in its own process group and stops only that group on excessive
// RSS, low free memory, time, or log size. Writes <label>.log,
// <label>.resources.jsonl (every 5 s) and <label>.job.json.
export async function guarded(options: {
  dir: string;
  label: string;
  command: string;
  args: string[];
  env: NodeJS.ProcessEnv;
  maxMinutes: number;
  extraFiles?: string[];
  hostBefore?: unknown;
}): Promise<JobRecord> {
  const { dir, label, command, args, env, maxMinutes } = options;
  await archiveAttempt(dir, label, options.extraFiles ?? []);
  const total = Number(sh('sysctl', ['-n', 'hw.memsize']));
  const log = join(dir, `${label}.log`);
  const resources = join(dir, `${label}.resources.jsonl`);
  const harness = new Set([process.pid, process.ppid]);
  const fd = openSync(log, 'w');
  const started = Date.now();
  const child = spawn(command, args, { cwd: ROOT, detached: true, stdio: ['ignore', fd, fd], env });
  closeSync(fd);
  let maxRss = 0;
  let reason: string | null = null;
  const stop = (message: string) => {
    if (reason) return;
    reason = message;
    if (child.pid === undefined) return;
    const group = -child.pid;
    try {
      process.kill(group, 'SIGTERM');
    } catch {}
    setTimeout(() => {
      try {
        process.kill(group, 'SIGKILL');
      } catch {}
    }, 1500).unref();
  };
  // The job runs in its own process group, so a terminal Ctrl-C reaches only this process.
  const interrupt = () => stop('interrupted');
  process.on('SIGINT', interrupt);
  process.on('SIGTERM', interrupt);
  const monitor = setInterval(() => {
    try {
      const tree = sh('ps', ['-axo', 'pid=,ppid=,rss='])
        .split('\n')
        .map((line) => line.trim().split(/\s+/).map(Number));
      const owned = new Set([child.pid]);
      let grew = true;
      while (grew) {
        grew = false;
        for (const [pid, ppid] of tree) {
          if (owned.has(ppid) && !owned.has(pid)) {
            owned.add(pid);
            grew = true;
          }
        }
      }
      const rss = tree.filter(([pid]) => owned.has(pid)).reduce((sum, row) => sum + row[2] * 1024, 0);
      maxRss = Math.max(maxRss, rss);
      const otherActivity = sh('ps', ['-axo', 'pid=,pcpu=,comm='])
        .split('\n')
        .map((line) => /^(\d+)\s+([\d.]+)\s+(.+)$/.exec(line.trim()))
        .filter((match): match is RegExpExecArray => match !== null)
        .map((match) => ({ pid: Number(match[1]), cpu: Number(match[2]), command: match[3] }))
        .filter((row) => !owned.has(row.pid) && !harness.has(row.pid) && (row.cpu > 15 || NOTABLE.test(row.command)));
      const available = Number(/free percentage: (\d+)%/.exec(sh('memory_pressure', ['-Q']))?.[1]);
      if (!Number.isFinite(available)) throw new Error('memory_pressure reports no free percentage');
      appendFileSync(
        resources,
        `${JSON.stringify({ at: new Date().toISOString(), rssBytes: rss, availablePercent: available, otherActivity })}\n`,
      );
      if (rss > total * MEMORY_SHARE || available < MIN_AVAILABLE_PERCENT) stop('memory guard');
      if (Date.now() - started > maxMinutes * 60_000) stop(`${maxMinutes}-minute job limit`);
      if (statSync(log).size > LOG_LIMIT_BYTES) stop('32 MiB log limit');
    } catch (error) {
      stop(`monitor failed: ${(error as Error).message}`);
    }
  }, MONITOR_MS);
  const result = await new Promise<{ code?: number | null; signal?: string | null; error?: string }>((done) => {
    child.once('error', (error) => done({ error: error.message }));
    child.once('exit', (code, signal) => done({ code, signal }));
  });
  clearInterval(monitor);
  process.off('SIGINT', interrupt);
  process.off('SIGTERM', interrupt);
  const record: JobRecord = {
    label,
    command,
    args,
    ...result,
    reason,
    maxRssBytes: maxRss,
    elapsedMs: Date.now() - started,
    finished: new Date().toISOString(),
    hostBefore: options.hostBefore,
  };
  await writeJson(join(dir, `${label}.job.json`), record);
  console.log(JSON.stringify({ ...record, hostBefore: undefined }));
  return record;
}
